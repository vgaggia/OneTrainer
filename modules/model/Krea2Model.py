import math
from contextlib import nullcontext
from random import Random

from modules.model.BaseModel import BaseModel
from modules.model.krea2.mmdit import SingleStreamDiT
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType
from modules.util.LayerOffloadConductor import LayerOffloadConductor

import torch
from torch import Tensor

from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

# Krea 2 conditions on a stack of 12 Qwen3-VL hidden-state layers (the MMDiT's
# TextFusionTransformer collapses the layer axis back to one). The layers are
# concatenated along the feature axis (cat dim=-1) -> (L, 12*2560); predict /
# the sampler reshape that back to (L, 12, 2560) before the MMDiT call.
KREA2_SELECT_LAYERS: tuple[int, ...] = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
KREA2_TXT_LAYERS = len(KREA2_SELECT_LAYERS)  # 12
KREA2_TXT_DIM = 2560
KREA2_TXT_FLAT_DIM = KREA2_TXT_LAYERS * KREA2_TXT_DIM

# Fixed instruction template. Identical to Qwen-Image's template (same Qwen
# tokenizer family, same system prefix), so the first CROP_START tokens (the
# system prefix) are sliced off the conditioning, exactly as for Qwen.
KREA2_PROMPT_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and "
    "background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
KREA2_PROMPT_TEMPLATE_CROP_START = 34
PROMPT_MAX_LENGTH = 512

# OneTrainer trains adapters against a native `transformer.*` module prefix, but
# the Krea/ai-toolkit/ComfyUI ecosystem expects the denoiser namespace to be
# `diffusion_model.*`. Keep both conversions explicit so external LoRAs can be
# loaded for resume/testing and saved LoRAs actually apply on Turbo in ComfyUI.
KREA2_LORA_ONETRAINER_PREFIX = "transformer."
KREA2_LORA_EXTERNAL_PREFIX = "diffusion_model."


def _replace_lora_prefix_once(key: str, prefix_from: str, prefix_to: str) -> str:
    return prefix_to + key[len(prefix_from):] if key.startswith(prefix_from) else key


def convert_krea2_lora_state_dict_to_external(state_dict: dict) -> dict:
    """Convert OneTrainer-native Krea 2 LoRA keys to the external ecosystem prefix."""
    return {
        _replace_lora_prefix_once(key, KREA2_LORA_ONETRAINER_PREFIX, KREA2_LORA_EXTERNAL_PREFIX): value
        for key, value in state_dict.items()
    }


def convert_krea2_lora_state_dict_to_onetrainer(state_dict: dict) -> dict:
    """Convert external Krea 2 LoRA keys to OneTrainer's native module prefix."""
    return {
        _replace_lora_prefix_once(key, KREA2_LORA_EXTERNAL_PREFIX, KREA2_LORA_ONETRAINER_PREFIX): value
        for key, value in state_dict.items()
    }


class Krea2Model(BaseModel):
    # base model data
    tokenizer: AutoTokenizer | None
    noise_scheduler: FlowMatchEulerDiscreteScheduler | None
    text_encoder: Qwen3VLForConditionalGeneration | None
    vae: AutoencoderKLQwenImage | None
    transformer: SingleStreamDiT | None

    # autocast context
    text_encoder_autocast_context: torch.autocast | nullcontext

    text_encoder_train_dtype: DataType

    text_encoder_offload_conductor: LayerOffloadConductor | None
    transformer_offload_conductor: LayerOffloadConductor | None

    # persistent lora training data
    text_encoder_lora: LoRAModuleWrapper | None
    transformer_lora: LoRAModuleWrapper | None
    lora_state_dict: dict | None

    def __init__(
            self,
            model_type: ModelType,
    ):
        super().__init__(
            model_type=model_type,
        )

        self.tokenizer = None
        self.noise_scheduler = None
        self.text_encoder = None
        self.vae = None
        self.transformer = None

        self.text_encoder_autocast_context = nullcontext()

        self.text_encoder_train_dtype = DataType.FLOAT_32

        self.text_encoder_offload_conductor = None
        self.transformer_offload_conductor = None

        self.text_encoder_lora = None
        self.transformer_lora = None
        self.lora_state_dict = None

    def adapters(self) -> list[LoRAModuleWrapper]:
        return [a for a in [
            self.text_encoder_lora,
            self.transformer_lora,
        ] if a is not None]

    def vae_to(self, device: torch.device):
        self.vae.to(device=device)

    def text_encoder_to(self, device: torch.device):
        if self.text_encoder is not None:
            if self.text_encoder_offload_conductor is not None and \
                    self.text_encoder_offload_conductor.layer_offload_activated():
                self.text_encoder_offload_conductor.to(device)
            else:
                self.text_encoder.to(device=device)

        if self.text_encoder_lora is not None:
            self.text_encoder_lora.to(device)

    def transformer_to(self, device: torch.device):
        if self.transformer_offload_conductor is not None and \
                self.transformer_offload_conductor.layer_offload_activated():
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
        if self.text_encoder is not None:
            self.text_encoder.eval()
        self.transformer.eval()

    @staticmethod
    def _validate_text_features(text_encoder_output: Tensor):
        if text_encoder_output.shape[-1] != KREA2_TXT_FLAT_DIM:
            raise ValueError(
                "Krea 2 expects cached/encoded text features with "
                f"{KREA2_TXT_LAYERS}x{KREA2_TXT_DIM}={KREA2_TXT_FLAT_DIM} channels, "
                f"but got {text_encoder_output.shape[-1]}. Check the Qwen3-VL model, "
                "selected hidden-state layers, tokenizer template and cache contents."
            )

    def encode_text(
            self,
            train_device: torch.device,
            batch_size: int = 1,
            rand: Random | None = None,
            text: str | list[str] = None,
            tokens: Tensor = None,
            tokens_mask: Tensor = None,
            text_encoder_dropout_probability: float | None = None,
            text_encoder_output: Tensor = None,
    ) -> tuple[Tensor, Tensor]:
        # Cached `text_encoder_output` is already the crop-and-flattened
        # (B, L, 12*2560) hidden-state stack (see EncodeQwenText with the 12
        # Krea2 layer indices in the data loader).
        if tokens is None and text is not None:
            if isinstance(text, str):
                text = [text]

            text = [KREA2_PROMPT_TEMPLATE.format(t) for t in text]
            tokenizer_output = self.tokenizer(
                text,
                max_length=PROMPT_MAX_LENGTH + KREA2_PROMPT_TEMPLATE_CROP_START,
                padding='max_length',
                truncation=True,
                return_tensors="pt",
            )
            tokens = tokenizer_output.input_ids.to(self.text_encoder.device)
            tokens_mask = tokenizer_output.attention_mask.to(self.text_encoder.device)

        if text_encoder_output is None and self.text_encoder is not None:
            with self.text_encoder_autocast_context:
                encoder_output = self.text_encoder(
                    tokens,
                    attention_mask=tokens_mask.float(),
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )
                # concatenate the selected layers along the feature axis ->
                # (B, L, 12*2560), then drop the system prefix tokens.
                hidden_state = torch.cat(
                    [encoder_output.hidden_states[i] for i in KREA2_SELECT_LAYERS], dim=-1,
                )
                tokens_mask = tokens_mask[:, KREA2_PROMPT_TEMPLATE_CROP_START:]
                text_encoder_output = hidden_state[:, KREA2_PROMPT_TEMPLATE_CROP_START:, :] \
                    * tokens_mask.unsqueeze(-1)

        if text_encoder_output is None:
            raise ValueError("Krea 2 text encoding requires tokens/text or a cached text_encoder_output.")
        self._validate_text_features(text_encoder_output)

        if text_encoder_dropout_probability is not None and text_encoder_dropout_probability > 0.0:
            raise NotImplementedError #https://github.com/Nerogar/OneTrainer/issues/957

        # prune tokens masked in all batch samples (efficiency), then pad to a
        # multiple of 16 to avoid uneven-sequence-length attention/compile issues.
        seq_lengths = tokens_mask.sum(dim=1)
        max_seq_length = seq_lengths.max().item()
        if max_seq_length % 16 > 0 and (seq_lengths != max_seq_length).any():
            max_seq_length += (16 - max_seq_length % 16)

        text_encoder_output = text_encoder_output[:, :max_seq_length, :]
        bool_attention_mask = tokens_mask[:, :max_seq_length].bool()

        return (text_encoder_output, bool_attention_mask)

    @staticmethod
    def pack_latents(latents: Tensor) -> Tensor:
        batch_size, channels, frames, height, width = latents.shape
        assert frames == 1

        latents = latents.view(batch_size, channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), channels * 4)

        return latents

    @staticmethod
    def unpack_latents(latents, height: int, width: int) -> Tensor:
        batch_size, _, channels = latents.shape

        height = height // 2
        width = width // 2

        latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), 1, height * 2, width * 2)

        return latents

    @staticmethod
    def build_pos_mask(
            batch_size: int,
            text_mask: Tensor,
            latent_height: int,
            latent_width: int,
            patch: int,
            device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        """Build the combined [text | image] 3-axis position ids and key-padding mask.

        Image tokens get position (0, row, col); text tokens (0, 0, 0). Returns
        (pos (B, L_txt+L_img, 3) float, mask (B, L_txt+L_img) bool). The MMDiT pads
        the combined sequence to a multiple of 256 internally.
        """
        h_, w_ = latent_height // patch, latent_width // patch
        img_ids = torch.zeros(h_, w_, 3, device=device)
        img_ids[..., 1] = torch.arange(h_, device=device)[:, None]
        img_ids[..., 2] = torch.arange(w_, device=device)[None, :]
        img_pos = img_ids.reshape(1, h_ * w_, 3).expand(batch_size, -1, -1)
        img_mask = torch.ones(batch_size, h_ * w_, dtype=torch.bool, device=device)

        txt_len = text_mask.shape[1]
        txt_pos = torch.zeros(batch_size, txt_len, 3, device=device)
        pos = torch.cat([txt_pos, img_pos], dim=1).float()
        mask = torch.cat([text_mask.to(device).bool(), img_mask], dim=1)
        return pos, mask

    def scale_latents(self, latents: Tensor) -> Tensor:
        latents_mean = torch.tensor(self.vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(1, self.vae.config.z_dim, 1, 1, 1)
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(1, self.vae.config.z_dim, 1, 1, 1)
        return (latents - latents_mean) * latents_std

    def unscale_latents(self, latents: Tensor) -> Tensor:
        latents_mean = torch.tensor(self.vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(1, self.vae.config.z_dim, 1, 1, 1)
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(1, self.vae.config.z_dim, 1, 1, 1)
        return latents / latents_std + latents_mean

    def calculate_timestep_shift(self, latent_width: int, latent_height: int):
        # Krea 2's resolution-aware mu shift is exactly the diffusers exponential
        # dynamic shift; the scheduler is configured with Krea's endpoints
        # (base_seq 256 -> 0.5, max_seq 6400 -> 1.15) so this matches the reference.
        base_seq_len = self.noise_scheduler.config.base_image_seq_len
        max_seq_len = self.noise_scheduler.config.max_image_seq_len
        base_shift = self.noise_scheduler.config.base_shift
        max_shift = self.noise_scheduler.config.max_shift
        patch_size = 2

        image_seq_len = (latent_width // patch_size) * (latent_height // patch_size)
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return math.exp(mu)
