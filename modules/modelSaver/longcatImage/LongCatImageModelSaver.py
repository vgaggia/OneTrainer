import os.path
from pathlib import Path

from modules.model.LongCatImageModel import LongCatImageModel
from modules.modelSaver.mixin.DtypeModelSaverMixin import DtypeModelSaverMixin
from modules.module.quantized.mixin.QuantizedLinearMixin import QuantizedLinearMixin
from modules.util.convert_util import convert
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.quantization_util import get_unquantized_weight

import torch

from safetensors.torch import save_file


class LongCatImageModelSaver(DtypeModelSaverMixin):
    def __init__(self):
        super().__init__()

    def __save_diffusers(self, model: LongCatImageModel, destination: str, dtype: torch.dtype | None):
        pipeline = model.create_pipeline(edit=True)
        pipeline.to("cpu")
        save_pipeline = self._copy_pipeline_to_dtype(pipeline, dtype, pipeline.tokenizer, pipeline.text_processor)

        os.makedirs(Path(destination).absolute(), exist_ok=True)
        save_pipeline.save_pretrained(destination)
        if dtype is not None:
            del save_pipeline

    def __save_original(self, model: LongCatImageModel, destination: str, dtype: torch.dtype | None):
        # ORIGINAL_TRANSFORMER is deliberately the raw native namespace loaded by ComfyUI's
        # diffusion-model loader (including fused QKV and QKV+MLP projections).
        dequant_dtype = dtype if dtype is not None else torch.float32
        state_dict = model.transformer.state_dict()
        for name, module in model.transformer.named_modules():
            if isinstance(module, QuantizedLinearMixin) and (name + ".weight") in state_dict:
                state_dict[name + ".weight"] = get_unquantized_weight(
                    module, dequant_dtype, torch.device("cpu")
                ).to("cpu")

        state_dict = convert(state_dict, model.checkpoint_diffusers_to_original())

        # Remove training-only quantizer scales, while preserving genuine RMSNorm weights.
        for key in [
            key for key in state_dict
            if key.endswith(".scale") and (key.removesuffix(".scale") + ".weight") in state_dict
        ]:
            del state_dict[key]

        save_state_dict = self._convert_state_dict_dtype(state_dict, dtype)
        self._convert_state_dict_to_contiguous(save_state_dict)
        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)
        save_file(save_state_dict, destination, self._create_safetensors_header(model, save_state_dict))

    def save(
            self,
            model: LongCatImageModel,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        match output_model_format:
            case ModelFormat.DIFFUSERS:
                self.__save_diffusers(model, output_model_destination, dtype)
            case ModelFormat.ORIGINAL_TRANSFORMER:
                self.__save_original(model, output_model_destination, dtype)
            case ModelFormat.INTERNAL:
                self.__save_diffusers(model, output_model_destination, None)
            case _:
                raise NotImplementedError(f"Unsupported output format: {output_model_format}")
