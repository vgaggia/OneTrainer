import math
from contextlib import nullcontext
from random import Random

from modules.model.BaseModel import BaseModel
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.convert_util import chunk_swap
from modules.util.enum.ModelType import ModelType
from modules.util.LayerOffloadConductor import LayerOffloadConductor

import torch
from torch import Tensor

from diffusers import (
    AutoencoderKL,
    DiffusionPipeline,
    FlowMatchEulerDiscreteScheduler,
    LongCatImageEditPipeline,
    LongCatImagePipeline,
    LongCatImageTransformer2DModel,
)
from diffusers.pipelines.longcat_image.pipeline_longcat_image_edit import prepare_pos_ids
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor


class LongCatImageModel(BaseModel):
    tokenizer: Qwen2Tokenizer | None
    text_processor: Qwen2VLProcessor | None
    noise_scheduler: FlowMatchEulerDiscreteScheduler | None
    text_encoder: Qwen2_5_VLForConditionalGeneration | None
    vae: AutoencoderKL | None
    transformer: LongCatImageTransformer2DModel | None

    text_encoder_autocast_context: torch.autocast | nullcontext
    text_encoder_offload_conductor: LayerOffloadConductor | None
    transformer_offload_conductor: LayerOffloadConductor | None

    transformer_lora: LoRAModuleWrapper | None
    lora_state_dict: dict | None

    def __init__(self, model_type: ModelType):
        super().__init__(model_type=model_type)
        self.tokenizer = None
        self.text_processor = None
        self.noise_scheduler = None
        self.text_encoder = None
        self.vae = None
        self.transformer = None

        self.text_encoder_autocast_context = nullcontext()
        self.text_encoder_offload_conductor = None
        self.transformer_offload_conductor = None

        self.transformer_lora = None
        self.lora_state_dict = None

    def adapters(self) -> list[LoRAModuleWrapper]:
        return [adapter for adapter in [self.transformer_lora] if adapter is not None]

    def fusion_groups(self) -> list | None:
        # LongCat's native/Comfy checkpoint fuses QKV in double blocks and QKV+MLP in single blocks.
        # Diffusers exposes all of these as separate Linear modules while training.
        return [
            ("transformer_blocks.{i}", ["attn.to_q", "attn.to_k", "attn.to_v"], "attn.qkv", "img_attn.qkv"),
            ("transformer_blocks.{i}", ["attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"], "attn.added_qkv", "txt_attn.qkv"),
            ("single_transformer_blocks.{i}", ["attn.to_q", "attn.to_k", "attn.to_v", "proj_mlp"], "attn.to_qkv_mlp_proj", "linear1"),
        ]

    def diffusers_to_original(self) -> list | None:
        # This is the native LongCat namespace used by the official checkpoint and by
        # Comfy-Org/LongCat-Image's longcat_image_edit_bf16.safetensors repack.
        return [
            ("context_embedder", "txt_in"),
            ("x_embedder", "img_in"),
            ("time_embed.timestep_embedder", "time_in", [
                ("linear_1", "in_layer"),
                ("linear_2", "out_layer"),
            ]),
            ("proj_out", "final_layer.linear"),
            *chunk_swap("norm_out.linear", "final_layer.adaLN_modulation.1"),
            ("transformer_blocks.{i}", "double_blocks.{i}", [
                ("attn.qkv", "img_attn.qkv"),
                ("attn.added_qkv", "txt_attn.qkv"),
                ("attn.norm_k.weight", "img_attn.norm.key_norm.weight"),
                ("attn.norm_q.weight", "img_attn.norm.query_norm.weight"),
                ("attn.to_out.0", "img_attn.proj"),
                ("ff.net.0.proj", "img_mlp.0"),
                ("ff.net.2", "img_mlp.2"),
                ("norm1.linear", "img_mod.lin"),
                ("attn.norm_added_k.weight", "txt_attn.norm.key_norm.weight"),
                ("attn.norm_added_q.weight", "txt_attn.norm.query_norm.weight"),
                ("attn.to_add_out", "txt_attn.proj"),
                ("ff_context.net.0.proj", "txt_mlp.0"),
                ("ff_context.net.2", "txt_mlp.2"),
                ("norm1_context.linear", "txt_mod.lin"),
            ]),
            ("single_transformer_blocks.{i}", "single_blocks.{i}", [
                ("attn.to_qkv_mlp_proj", "linear1"),
                ("attn.norm_k.weight", "norm.key_norm.weight"),
                ("attn.norm_q.weight", "norm.query_norm.weight"),
                ("proj_out", "linear2"),
                ("norm.linear", "modulation.lin"),
            ]),
        ]

    def vae_to(self, device: torch.device):
        self.vae.to(device=device)

    def text_encoder_to(self, device: torch.device):
        if self.text_encoder_offload_conductor is not None:
            self.text_encoder_offload_conductor.to(device)
        else:
            self.text_encoder.to(device=device)

    def transformer_to(self, device: torch.device):
        if self.transformer_offload_conductor is not None:
            self.transformer_offload_conductor.to(device)
        else:
            self.transformer.to(device=device)
        if self.transformer_lora is not None:
            self.transformer_lora.to(device)

    def to(self, device: torch.device):
        self.vae_to(device)
        self.text_encoder_to(device)
        self.transformer_to(device)

    def eval(self):
        self.vae.eval()
        self.text_encoder.eval()
        self.transformer.eval()

    def create_pipeline(self, edit: bool = True) -> DiffusionPipeline:
        pipeline_class = LongCatImageEditPipeline if edit else LongCatImagePipeline
        return pipeline_class(
            transformer=self.transformer,
            scheduler=self.noise_scheduler,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            text_processor=self.text_processor,
        )

    def encode_text(
            self,
            train_device: torch.device,
            batch_size: int = 1,
            rand: Random | None = None,
            text: str | list[str] | None = None,
            conditioning_image: Tensor | None = None,
            text_encoder_sequence_length: int | None = None,
            text_encoder_dropout_probability: float | None = None,
            text_encoder_output: Tensor | None = None,
    ) -> Tensor:
        if text_encoder_dropout_probability is not None and text_encoder_dropout_probability > 0.0:
            raise NotImplementedError("Text-encoder dropout is not implemented for LongCat Image.")

        if text_encoder_output is None:
            if text is None:
                raise ValueError("A prompt is required when LongCat text embeddings are not cached.")
            prompts = [text] if isinstance(text, str) else text

            # The edit system prompt deliberately includes the source image through Qwen2.5-VL.
            # Without a source, use LongCat's caption/T2I system prompt.  This mirrors the model's
            # joint edit + T2I training and is what enables ordinary captioned-image fine-tuning.
            pipeline = self.create_pipeline(edit=conditioning_image is not None)
            # During caching only the text encoder is on the train device. DiffusionPipeline.device
            # otherwise resolves from the VAE first and would move token/pixel inputs to the VAE's
            # (usually CPU) device. Prompt encoding does not use either denoising component.
            pipeline.vae = None
            pipeline.transformer = None
            if text_encoder_sequence_length is not None:
                pipeline.tokenizer_max_length = text_encoder_sequence_length

            image = None
            if conditioning_image is not None:
                image = conditioning_image
                if isinstance(image, Tensor) and image.dim() == 3:
                    image = image.unsqueeze(0)

            with self.text_encoder_autocast_context:
                if image is None:
                    text_encoder_output, _ = pipeline.encode_prompt(prompt=prompts)
                else:
                    # Upstream edit prompt encoding currently supports one prompt/source pair at a time.
                    # MGDS calls this path per item; cached batches are assembled afterwards.
                    if len(prompts) != 1:
                        raise ValueError("LongCat edit prompt encoding expects one source image per prompt.")
                    text_encoder_output, _ = pipeline.encode_prompt(prompt=prompts, image=image)

        return text_encoder_output

    @staticmethod
    def pack_latents(latents: Tensor) -> Tensor:
        batch_size, channels, height, width = latents.shape
        latents = latents.view(batch_size, channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        return latents.reshape(batch_size, (height // 2) * (width // 2), channels * 4)

    @staticmethod
    def unpack_latents(latents: Tensor, height: int, width: int) -> Tensor:
        batch_size, _, channels = latents.shape
        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        return latents.reshape(batch_size, channels // 4, height, width)

    @staticmethod
    def prepare_text_ids(text_encoder_output: Tensor) -> Tensor:
        return prepare_pos_ids(
            modality_id=0,
            type="text",
            start=(0, 0),
            num_token=text_encoder_output.shape[1],
        ).to(text_encoder_output.device)

    @staticmethod
    def prepare_image_ids(
            modality_id: int,
            text_sequence_length: int,
            latent_height: int,
            latent_width: int,
            device: torch.device,
    ) -> Tensor:
        return prepare_pos_ids(
            modality_id=modality_id,
            type="image",
            start=(text_sequence_length, text_sequence_length),
            height=latent_height // 2,
            width=latent_width // 2,
        ).to(device)

    def scale_latents(self, latents: Tensor) -> Tensor:
        return (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

    def unscale_latents(self, latents: Tensor) -> Tensor:
        return latents / self.vae.config.scaling_factor + self.vae.config.shift_factor

    def calculate_timestep_shift(self, latent_height: int, latent_width: int) -> float:
        image_seq_len = (latent_height // 2) * (latent_width // 2)
        base_seq_len = self.noise_scheduler.config.base_image_seq_len
        max_seq_len = self.noise_scheduler.config.max_image_seq_len
        base_shift = self.noise_scheduler.config.base_shift
        max_shift = self.noise_scheduler.config.max_shift
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        return math.exp(image_seq_len * m + b)
