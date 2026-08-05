import os
import traceback

from modules.model.LongCatImageModel import LongCatImageModel
from modules.modelLoader.mixin.HFModelLoaderMixin import HFModelLoaderMixin
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.convert_util import convert, reverse_conversion, split_fused_state_dict
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes

import torch

from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, LongCatImageTransformer2DModel
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor

from safetensors.torch import load_file


class LongCatImageModelLoader(HFModelLoaderMixin):
    def __init__(self):
        super().__init__()

    def __load_internal(
            self,
            model: LongCatImageModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
    ):
        if not os.path.isfile(os.path.join(base_model_name, "meta.json")):
            raise Exception("not an internal model")
        # Internal fine-tune backups contain the trained transformer; never replace it with
        # the original external override when resuming.
        self.__load_diffusers(
            model, model_type, weight_dtypes, base_model_name, "", vae_model_name, quantization,
        )

    def __load_diffusers(
            self,
            model: LongCatImageModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
    ):
        # Request Transformers' Mistral-regex compatibility handling. Current Qwen
        # tokenizers are unchanged by it, while older backends avoid a false warning.
        tokenizer = Qwen2Tokenizer.from_pretrained(
            base_model_name, subfolder="tokenizer", fix_mistral_regex=True,
        )
        text_processor = Qwen2VLProcessor.from_pretrained(
            base_model_name, subfolder="text_processor", fix_mistral_regex=True,
        )
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(base_model_name, subfolder="scheduler")
        text_encoder = self._load_transformers_sub_module(
            Qwen2_5_VLForConditionalGeneration,
            weight_dtypes.text_encoder,
            weight_dtypes.fallback_train_dtype,
            base_model_name,
            "text_encoder",
        )

        if vae_model_name:
            vae = self._load_diffusers_sub_module(
                AutoencoderKL, weight_dtypes.vae, weight_dtypes.train_dtype, vae_model_name,
            )
        else:
            vae = self._load_diffusers_sub_module(
                AutoencoderKL, weight_dtypes.vae, weight_dtypes.train_dtype, base_model_name, "vae",
            )

        if transformer_model_name:
            if weight_dtypes.transformer.is_gguf():
                raise NotImplementedError("GGUF loading of the LongCat Image transformer is not supported yet.")

            transformer_config = LongCatImageTransformer2DModel.load_config(
                base_model_name, subfolder="transformer",
            )
            with torch.device("meta"):
                transformer = LongCatImageTransformer2DModel.from_config(transformer_config)

            # ComfyUI's longcat_image_edit_bf16 file is the official native transformer namespace.
            # Rename it to canonical keys, then split native fused QKV/QKV+MLP tensors to match Diffusers.
            state_dict = convert(
                load_file(transformer_model_name),
                reverse_conversion(model.diffusers_to_original()),
            )
            state_dict = split_fused_state_dict(
                state_dict, transformer.state_dict(), model.fusion_groups(),
            )
            float_dtype = weight_dtypes.transformer.torch_dtype() or torch.bfloat16
            state_dict = {
                key: value.to(float_dtype) if value.is_floating_point() else value
                for key, value in state_dict.items()
            }
            transformer.load_state_dict(state_dict, strict=True, assign=True)
            del state_dict
            transformer = self._convert_diffusers_sub_module_to_dtype(
                transformer, weight_dtypes.transformer, weight_dtypes.train_dtype, quantization,
            )
        else:
            transformer = self._load_diffusers_sub_module(
                LongCatImageTransformer2DModel,
                weight_dtypes.transformer,
                weight_dtypes.train_dtype,
                base_model_name,
                "transformer",
                quantization,
            )

        model.model_type = model_type
        model.tokenizer = tokenizer
        model.text_processor = text_processor
        model.noise_scheduler = noise_scheduler
        model.text_encoder = text_encoder
        model.vae = vae
        model.transformer = transformer

    def __load_safetensors(self, *args, **kwargs):
        raise NotImplementedError(
            "A LongCat single file only contains the transformer. Select the official Diffusers base model "
            "and provide longcat_image_edit_bf16.safetensors as the transformer override."
        )

    def load(
            self,
            model: LongCatImageModel,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
    ):
        stacktraces = []

        try:
            self.__load_internal(
                model, model_type, weight_dtypes, model_names.base_model,
                model_names.transformer_model, model_names.vae_model, quantization,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_diffusers(
                model, model_type, weight_dtypes, model_names.base_model,
                model_names.transformer_model, model_names.vae_model, quantization,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_safetensors(
                model, model_type, weight_dtypes, model_names.base_model,
                model_names.transformer_model, model_names.vae_model, quantization,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load model: " + model_names.base_model)
