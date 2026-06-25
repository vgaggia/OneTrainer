import os
import traceback

from modules.model.BaseModel import BaseModel
from modules.model.krea2.mmdit import SingleMMDiTConfig, SingleStreamDiT, single_mmdit_large_wide
from modules.model.Krea2Model import Krea2Model
from modules.modelLoader.GenericFineTuneModelLoader import make_fine_tune_model_loader
from modules.modelLoader.GenericLoRAModelLoader import make_lora_model_loader
from modules.modelLoader.mixin.HFModelLoaderMixin import HFModelLoaderMixin
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.convert.lora.convert_lora_util import LoraConversionKeySet
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.quantization_util import is_quantized_parameter, replace_linear_with_quantized_layers

import torch

from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

import huggingface_hub
from huggingface_hub.errors import EntryNotFoundError
from safetensors.torch import load_file

# Krea 2's text encoder and VAE are fixed components; only the transformer
# checkpoint (Raw / Turbo) varies. Both default to the public HF repos and can be
# overridden (VAE via vae_model, text encoder via text_encoder_4).
KREA2_DEFAULT_TEXT_ENCODER = "Qwen/Qwen3-VL-4B-Instruct"
KREA2_DEFAULT_VAE = "Qwen/Qwen-Image"

# Matches Krea 2's mu schedule: mu interpolated in image-token count between
# (256-seq -> 0.5) and (6400-seq -> 1.15), applied as an exponential time shift.
KREA2_SCHEDULER_CONFIG = {
    "base_image_seq_len": 256,
    "max_image_seq_len": 6400,
    "base_shift": 0.5,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "use_dynamic_shifting": True,
    "time_shift_type": "exponential",
}


def _resolve_transformer_path(name_or_path: str) -> str:
    """Resolve a Krea 2 transformer checkpoint to a local .safetensors path.

    Accepts a .safetensors file, a directory containing exactly one, or an HF
    repo id (filename derived from the repo name, e.g. krea/Krea-2-Raw ->
    raw.safetensors).
    """
    if name_or_path.endswith(".safetensors") and os.path.isfile(name_or_path):
        return name_or_path

    if os.path.isdir(name_or_path):
        candidates = [f for f in os.listdir(name_or_path) if f.endswith(".safetensors")]
        if len(candidates) == 1:
            return os.path.join(name_or_path, candidates[0])
        raise FileNotFoundError(
            f"Could not pick a Krea 2 transformer checkpoint in {name_or_path}: found {candidates}."
        )

    fname = name_or_path.split("/")[-1].split("-")[-1].lower() + ".safetensors"
    try:
        return huggingface_hub.hf_hub_download(repo_id=name_or_path, filename=fname)
    except EntryNotFoundError as e:
        raise FileNotFoundError(
            f"Could not find {fname!r} in hub repo {name_or_path!r}."
        ) from e


class Krea2ModelLoader(
    HFModelLoaderMixin,
):
    def __init__(self):
        super().__init__()

    def __load_transformer(
            self,
            transformer_path: str,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
    ) -> SingleStreamDiT:
        dtype = weight_dtypes.transformer
        if dtype.is_gguf():
            raise NotImplementedError(
                "GGUF loading of the Krea 2 transformer is not supported; use a "
                ".safetensors checkpoint (bfloat16 or NF4 / INT8 / FP8 quantization)."
            )

        config: SingleMMDiTConfig = single_mmdit_large_wide
        # Build on meta, then materialize straight from the checkpoint.
        with torch.device("meta"):
            transformer = SingleStreamDiT(config)

        state_dict = load_file(_resolve_transformer_path(transformer_path))

        if dtype.is_quantized():
            # Swap the targeted nn.Linear layers for quantized ones (scoped by the
            # quantization layer filter), assign the float checkpoint weights, and
            # let quantize_layers() in setup_optimizations finalize them on GPU.
            replace_linear_with_quantized_layers(transformer, dtype, [], quantization, copy_parameters=False)
            self.__assign_state_dict(transformer, state_dict, dtype, weight_dtypes.train_dtype)
        else:
            float_dtype = dtype.torch_dtype() or torch.bfloat16
            state_dict = {
                k: (v.to(float_dtype) if v.is_floating_point() else v) for k, v in state_dict.items()
            }
            transformer.load_state_dict(state_dict, strict=True, assign=True)

        del state_dict
        return transformer

    @staticmethod
    def __assign_state_dict(module, state_dict, dtype, train_dtype):
        # Quantization-aware per-parameter assignment mirroring HFModelLoaderMixin:
        # quantized params keep their loaded dtype (quantize() casts them later),
        # everything else is converted to the train dtype.
        for key, value in state_dict.items():
            target = module
            splits = key.split(".")
            for split in splits[:-1]:
                target = getattr(target, split)
            tensor_name = splits[-1]

            is_buffer = tensor_name in target._buffers
            if not is_buffer and tensor_name not in target._parameters:
                continue
            old_value = target._buffers[tensor_name] if is_buffer else target._parameters[tensor_name]

            if torch.is_floating_point(old_value):
                old_type = type(old_value)
                if not is_quantized_parameter(target, tensor_name):
                    value = value.to(dtype=train_dtype.torch_dtype() if dtype.is_quantized() else dtype.torch_dtype())
                new_value = old_type(value)
                if is_buffer:
                    target._buffers[tensor_name] = new_value
                else:
                    target._parameters[tensor_name] = new_value

    def __load(
            self,
            model: Krea2Model,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            text_encoder_name: str,
            quantization: QuantizationConfig,
    ):
        # Text encoder (Qwen3-VL-4B): only used for text, so drop the vision tower.
        te_path = text_encoder_name or KREA2_DEFAULT_TEXT_ENCODER
        tokenizer = AutoTokenizer.from_pretrained(te_path)
        text_encoder = self._load_transformers_sub_module(
            Qwen3VLForConditionalGeneration,
            weight_dtypes.text_encoder,
            weight_dtypes.fallback_train_dtype,
            te_path,
        )
        # Qwen3-VL ties lm_head to the input embeddings (tie_word_embeddings=True), so
        # the checkpoint has no lm_head weight and the manual sub-module loader leaves
        # it on meta. Reconstruct it from the embeddings (cf. ZImageModelLoader) or
        # quantization / .to(device) fails with "Cannot copy out of meta tensor".
        output_embeddings = text_encoder.get_output_embeddings()
        if output_embeddings is not None and output_embeddings.weight.is_meta:
            input_embeddings = text_encoder.get_input_embeddings()
            output_embeddings.weight = type(output_embeddings.weight)(input_embeddings.weight)

        if getattr(text_encoder.model, "visual", None) is not None:
            text_encoder.model.visual = None

        # VAE: the Qwen-Image VAE (f8, 16 channels).
        vae = self._load_diffusers_sub_module(
            AutoencoderKLQwenImage,
            weight_dtypes.vae,
            weight_dtypes.train_dtype,
            vae_model_name or KREA2_DEFAULT_VAE,
            None if vae_model_name else "vae",
        )

        transformer = self.__load_transformer(
            transformer_model_name or base_model_name, weight_dtypes, quantization,
        )

        model.model_type = model_type
        model.tokenizer = tokenizer
        model.noise_scheduler = FlowMatchEulerDiscreteScheduler(**KREA2_SCHEDULER_CONFIG)
        model.text_encoder = text_encoder
        model.vae = vae
        model.transformer = transformer

    def load(
            self,
            model: Krea2Model,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
    ):
        stacktraces = []

        try:
            self.__load(
                model, model_type, weight_dtypes, model_names.base_model,
                model_names.transformer_model, model_names.vae_model, model_names.text_encoder_4,
                quantization,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load model: " + model_names.base_model)


class Krea2LoRALoader(
    LoRALoaderMixin
):
    def __init__(self):
        super().__init__()

    def _get_convert_key_sets(self, model: BaseModel) -> list[LoraConversionKeySet] | None:
        return None

    def load(
            self,
            model: Krea2Model,
            model_names: ModelNames,
    ):
        return self._load(model, model_names)


Krea2LoRAModelLoader = make_lora_model_loader(
    model_spec_map={
        ModelType.KREA_2: "resources/sd_model_spec/krea2-lora.json",
    },
    model_class=Krea2Model,
    model_loader_class=Krea2ModelLoader,
    lora_loader_class=Krea2LoRALoader,
    embedding_loader_class=None,
)

Krea2FineTuneModelLoader = make_fine_tune_model_loader(
    model_spec_map={
        ModelType.KREA_2: "resources/sd_model_spec/krea2.json",
    },
    model_class=Krea2Model,
    model_loader_class=Krea2ModelLoader,
    embedding_loader_class=None,
)
