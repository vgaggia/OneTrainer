import copy
import math
from collections.abc import Callable

from modules.model.LongCatImageModel import LongCatImageModel
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.util import factory
from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.AudioFormat import AudioFormat
from modules.util.enum.FileType import FileType
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.NoiseScheduler import NoiseScheduler
from modules.util.enum.VideoFormat import VideoFormat
from modules.util.image_util import load_image
from modules.util.torch_util import torch_gc

import torch

import numpy as np
from tqdm import tqdm


def combine_longcat_cfg(
        positive: torch.Tensor,
        negative: torch.Tensor,
        cfg_scale: float,
        renormalize: bool,
) -> torch.Tensor:
    guided = negative + cfg_scale * (positive - negative)
    if renormalize:
        conditional_norm = torch.norm(positive, dim=-1, keepdim=True)
        guided_norm = torch.norm(guided, dim=-1, keepdim=True)
        guided = guided * (conditional_norm / (guided_norm + 1e-8)).clamp(min=0.0, max=1.0)
    return guided


@factory.register(BaseModelSampler, ModelType.LONGCAT_IMAGE_EDIT)
class LongCatImageSampler(BaseModelSampler):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            model: LongCatImageModel,
            model_type: ModelType,
    ):
        super().__init__(train_device, temp_device)
        self.model = model
        self.model_type = model_type

    @torch.no_grad()
    def __sample_base(
            self,
            prompt: str,
            negative_prompt: str,
            height: int,
            width: int,
            seed: int,
            random_seed: bool,
            diffusion_steps: int,
            cfg_scale: float,
            noise_scheduler: NoiseScheduler,
            sample_editing: bool = False,
            base_image_path: str = "",
            text_encoder_sequence_length: int | None = None,
            on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ) -> ModelSamplerOutput:
        with self.model.autocast_context:
            generator = torch.Generator(device=self.train_device)
            if random_seed:
                generator.seed()
            else:
                generator.manual_seed(seed)

            scheduler = copy.deepcopy(self.model.noise_scheduler)
            source_pil = None
            source_image = None
            prompt_image = None
            pipeline = self.model.create_pipeline(edit=sample_editing)
            image_processor = pipeline.image_processor

            if sample_editing:
                if not base_image_path:
                    raise ValueError("LongCat edit sampling requires a base image path.")
                source_pil = load_image(base_image_path, convert_mode="RGB")
                source_pil = image_processor.resize(source_pil, height, width)
                prompt_image = image_processor.resize(source_pil, height // 2, width // 2)
                source_image = image_processor.preprocess(source_pil, height, width)

            # Prompt encoding uses the T2I system prompt for generation-only samples and the
            # image-aware edit system prompt when a source is supplied.
            self.model.text_encoder_to(self.train_device)
            prompt_embedding = self.model.encode_text(
                train_device=self.train_device,
                text=prompt,
                conditioning_image=prompt_image,
                text_encoder_sequence_length=text_encoder_sequence_length,
            )
            negative_embedding = None
            if cfg_scale > 1.0:
                negative_embedding = self.model.encode_text(
                    train_device=self.train_device,
                    text=negative_prompt or "",
                    conditioning_image=prompt_image,
                    text_encoder_sequence_length=text_encoder_sequence_length,
                )
            self.model.text_encoder_to(self.temp_device)
            torch_gc()

            latent_height = height // 8
            latent_width = width // 8
            target_latents = torch.randn(
                (1, 16, latent_height, latent_width),
                generator=generator,
                device=self.train_device,
                dtype=self.model.train_dtype.torch_dtype(),
            )
            packed_target = self.model.pack_latents(target_latents)
            target_sequence_length = packed_target.shape[1]

            text_sequence_length = prompt_embedding.shape[1]
            image_ids = self.model.prepare_image_ids(
                1, text_sequence_length, latent_height, latent_width, self.train_device,
            )
            source_latents = None
            if source_image is not None:
                self.model.vae_to(self.train_device)
                source_image = source_image.to(
                    device=self.train_device,
                    dtype=self.model.train_dtype.torch_dtype(),
                )
                source_latents = self.model.vae.encode(source_image).latent_dist.mode()
                source_latents = self.model.pack_latents(self.model.scale_latents(source_latents))
                source_ids = self.model.prepare_image_ids(
                    2, text_sequence_length, latent_height, latent_width, self.train_device,
                )
                image_ids = torch.cat([image_ids, source_ids], dim=0)
                self.model.vae_to(self.temp_device)
                torch_gc()

            text_ids = self.model.prepare_text_ids(prompt_embedding)
            negative_text_ids = (
                self.model.prepare_text_ids(negative_embedding) if negative_embedding is not None else None
            )

            shift = self.model.calculate_timestep_shift(latent_height, latent_width)
            sigmas = np.linspace(1.0, 1.0 / diffusion_steps, diffusion_steps)
            scheduler.set_timesteps(
                diffusion_steps,
                device=self.train_device,
                sigmas=sigmas,
                mu=math.log(shift),
            )

            self.model.transformer_to(self.train_device)
            for index, timestep in enumerate(tqdm(scheduler.timesteps, desc="sampling")):
                latent_input = packed_target
                if source_latents is not None:
                    latent_input = torch.cat([packed_target, source_latents], dim=1)
                expanded_timestep = timestep.expand(latent_input.shape[0]).to(latent_input.dtype)

                noise_pred_positive = self.model.transformer(
                    hidden_states=latent_input,
                    timestep=expanded_timestep / 1000,
                    guidance=None,
                    encoder_hidden_states=prompt_embedding.to(dtype=latent_input.dtype),
                    txt_ids=text_ids,
                    img_ids=image_ids,
                    return_dict=False,
                )[0][:, :target_sequence_length]

                if negative_embedding is not None:
                    noise_pred_negative = self.model.transformer(
                        hidden_states=latent_input,
                        timestep=expanded_timestep / 1000,
                        guidance=None,
                        encoder_hidden_states=negative_embedding.to(dtype=latent_input.dtype),
                        txt_ids=negative_text_ids,
                        img_ids=image_ids,
                        return_dict=False,
                    )[0][:, :target_sequence_length]
                    noise_pred = combine_longcat_cfg(
                        noise_pred_positive,
                        noise_pred_negative,
                        cfg_scale,
                        # The native T2I pipeline enables CFG renormalization by default;
                        # the native edit pipeline intentionally uses ordinary CFG.
                        renormalize=not sample_editing,
                    )
                else:
                    noise_pred = noise_pred_positive

                packed_target = scheduler.step(
                    noise_pred, timestep, packed_target, return_dict=False,
                )[0]
                on_update_progress(index + 1, len(scheduler.timesteps))

            self.model.transformer_to(self.temp_device)
            torch_gc()

            latents = self.model.unpack_latents(packed_target, latent_height, latent_width)
            latents = self.model.unscale_latents(latents)
            self.model.vae_to(self.train_device)
            image = self.model.vae.decode(
                latents.to(dtype=self.model.vae.dtype), return_dict=False,
            )[0]
            image = image_processor.postprocess(image, output_type="pil")
            self.model.vae_to(self.temp_device)
            torch_gc()

            return ModelSamplerOutput(file_type=FileType.IMAGE, data=image[0])

    def sample(
            self,
            sample_config: SampleConfig,
            destination: str,
            image_format: ImageFormat | None = None,
            video_format: VideoFormat | None = None,
            audio_format: AudioFormat | None = None,
            on_sample: Callable[[ModelSamplerOutput], None] = lambda _: None,
            on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ):
        sampler_output = self.__sample_base(
            prompt=sample_config.prompt,
            negative_prompt=sample_config.negative_prompt,
            height=self.quantize_resolution(sample_config.height, 16),
            width=self.quantize_resolution(sample_config.width, 16),
            seed=sample_config.seed,
            random_seed=sample_config.random_seed,
            diffusion_steps=sample_config.diffusion_steps,
            cfg_scale=sample_config.cfg_scale,
            noise_scheduler=sample_config.noise_scheduler,
            sample_editing=sample_config.sample_inpainting,
            base_image_path=sample_config.base_image_path,
            text_encoder_sequence_length=sample_config.text_encoder_1_sequence_length,
            on_update_progress=on_update_progress,
        )
        self.save_sampler_output(
            sampler_output, destination, image_format, video_format, audio_format,
        )
        on_sample(sampler_output)
