from abc import ABCMeta
from random import Random

import modules.util.multi_gpu_util as multi
from modules.model.LongCatImageModel import LongCatImageModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDebugMixin import ModelSetupDebugMixin
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupFlowMatchingMixin import ModelSetupFlowMatchingMixin
from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.modelSetup.mixin.ModelSetupText2ImageMixin import ModelSetupText2ImageMixin
from modules.util.checkpointing_util import (
    enable_checkpointing_for_flux_transformer,
    enable_checkpointing_for_qwen25vl_encoder_layers,
)
from modules.util.compile_util import disable_inductor_mixed_order_reduction
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dtype_util import create_autocast_context, disable_fp16_autocast_context
from modules.util.quantization_util import quantize_layers
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor


class BaseLongCatImageSetup(
    BaseModelSetup,
    ModelSetupDiffusionLossMixin,
    ModelSetupDebugMixin,
    ModelSetupNoiseMixin,
    ModelSetupFlowMatchingMixin,
    ModelSetupText2ImageMixin,
    metaclass=ABCMeta,
):
    LAYER_PRESETS = {
        "attn-mlp": ["attn", "ff", "proj_mlp"],
        "attn-only": ["attn"],
        "blocks": ["transformer_blocks", "single_transformer_blocks"],
        "full": [],
    }

    def setup_optimizations(self, model: LongCatImageModel, config: TrainConfig):
        # LongCat intentionally uses the same double/single stream block structure as Flux.
        if config.compile:
            disable_inductor_mixed_order_reduction()
        model.transformer_offload_conductor = enable_checkpointing_for_flux_transformer(
            model.transformer, config, config.transformer,
        )
        model.text_encoder_offload_conductor = enable_checkpointing_for_qwen25vl_encoder_layers(
            model.text_encoder, config, config.text_encoder,
        )

        model.autocast_context, model.train_dtype = create_autocast_context(
            self.train_device, config.train_dtype, config.enable_autocast_cache,
        )
        model.text_encoder_autocast_context, model.text_encoder_train_dtype = disable_fp16_autocast_context(
            self.train_device,
            config.train_dtype,
            config.fallback_train_dtype,
            config.enable_autocast_cache,
        )

        quantize_layers(model.text_encoder, self.train_device, model.text_encoder_train_dtype, config)
        quantize_layers(model.vae, self.train_device, model.train_dtype, config)
        quantize_layers(model.transformer, self.train_device, model.train_dtype, config)
        self._set_attention_backend(model.transformer, config.attention_mechanism, mask=False)

    def predict(
            self,
            model: LongCatImageModel,
            batch: dict,
            config: TrainConfig,
            train_progress: TrainProgress,
            *,
            deterministic: bool = False,
    ) -> dict:
        with model.autocast_context:
            batch_seed = 0 if deterministic else train_progress.global_step * multi.world_size() + multi.rank()
            generator = torch.Generator(device=config.train_device)
            generator.manual_seed(batch_seed)
            rand = Random(batch_seed)

            text_encoder_output = model.encode_text(
                train_device=self.train_device,
                batch_size=batch["latent_image"].shape[0],
                rand=rand,
                text_encoder_sequence_length=config.text_encoder_sequence_length,
                text_encoder_output=batch.get("text_encoder_hidden_state"),
                text_encoder_dropout_probability=(
                    config.text_encoder.dropout_probability if not deterministic else None
                ),
            )

            scaled_latent_image = model.scale_latents(batch["latent_image"])
            latent_noise = self._create_noise(scaled_latent_image, config, generator)

            shift = model.calculate_timestep_shift(
                scaled_latent_image.shape[-2], scaled_latent_image.shape[-1],
            )
            timestep = self._get_timestep_discrete(
                model.noise_scheduler.config["num_train_timesteps"],
                deterministic,
                generator,
                scaled_latent_image.shape[0],
                config,
                shift=shift if config.dynamic_timestep_shifting else config.timestep_shift,
            )
            scaled_noisy_latent_image, sigma = self._add_noise_discrete(
                scaled_latent_image,
                latent_noise,
                timestep,
                model.noise_scheduler.timesteps,
            )

            text_sequence_length = text_encoder_output.shape[1]
            image_ids = model.prepare_image_ids(
                1,
                text_sequence_length,
                scaled_noisy_latent_image.shape[-2],
                scaled_noisy_latent_image.shape[-1],
                self.train_device,
            )
            packed_target = model.pack_latents(scaled_noisy_latent_image)
            target_sequence_length = packed_target.shape[1]
            latent_input = packed_target

            if "latent_conditioning_image" in batch:
                scaled_source = model.scale_latents(batch["latent_conditioning_image"])
                packed_source = model.pack_latents(scaled_source)
                latent_input = torch.cat([packed_target, packed_source], dim=1)
                source_ids = model.prepare_image_ids(
                    2,
                    text_sequence_length,
                    scaled_source.shape[-2],
                    scaled_source.shape[-1],
                    self.train_device,
                )
                image_ids = torch.cat([image_ids, source_ids], dim=0)

            text_ids = model.prepare_text_ids(text_encoder_output)
            packed_predicted_flow = model.transformer(
                hidden_states=latent_input.to(dtype=model.train_dtype.torch_dtype()),
                timestep=(timestep / 1000).to(dtype=model.train_dtype.torch_dtype()),
                guidance=None,
                encoder_hidden_states=text_encoder_output.to(dtype=model.train_dtype.torch_dtype()),
                txt_ids=text_ids,
                img_ids=image_ids,
                return_dict=False,
            )[0][:, :target_sequence_length]

            predicted_flow = model.unpack_latents(
                packed_predicted_flow,
                scaled_noisy_latent_image.shape[-2],
                scaled_noisy_latent_image.shape[-1],
            )
            flow = latent_noise - scaled_latent_image
            model_output_data = {
                "loss_type": "target",
                "timestep": timestep,
                "predicted": predicted_flow,
                "target": flow,
            }

            if config.debug_mode:
                with torch.no_grad():
                    predicted_scaled_latent_image = scaled_noisy_latent_image - predicted_flow * sigma
                    self._save_latent("1-noise", latent_noise, config, train_progress)
                    self._save_latent("2-noisy_image", scaled_noisy_latent_image, config, train_progress)
                    self._save_latent("3-predicted_flow", predicted_flow, config, train_progress)
                    self._save_latent("4-flow", flow, config, train_progress)
                    self._save_latent("5-predicted_image", predicted_scaled_latent_image, config, train_progress)
                    self._save_latent("6-image", scaled_latent_image, config, train_progress)

        return model_output_data

    def calculate_loss(
            self,
            model: LongCatImageModel,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor:
        return self._flow_matching_losses(
            batch=batch,
            data=data,
            config=config,
            train_device=self.train_device,
            sigmas=model.noise_scheduler.sigmas,
        ).mean()

    def prepare_text_caching(self, model: LongCatImageModel, config: TrainConfig):
        model.to(self.temp_device)
        model.text_encoder_to(self.train_device)
        model.eval()
        torch_gc()
