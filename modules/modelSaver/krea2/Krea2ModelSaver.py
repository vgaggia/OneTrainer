import os.path
from pathlib import Path

from modules.model.Krea2Model import Krea2Model
from modules.modelSaver.mixin.DtypeModelSaverMixin import DtypeModelSaverMixin
from modules.module.quantized.mixin.QuantizedLinearMixin import QuantizedLinearMixin
from modules.util.convert_util import convert
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.quantization_util import get_unquantized_weight

import torch

from safetensors.torch import save_file


class Krea2ModelSaver(
    DtypeModelSaverMixin,
):
    def __init__(self):
        super().__init__()

    def __save_diffusers(
            self,
            model: Krea2Model,
            destination: str,
            dtype: torch.dtype | None,
    ):
        # Copy the model to cpu by first moving the original model to cpu. This preserves some VRAM.
        pipeline = model.create_pipeline()
        pipeline.to("cpu")
        save_pipeline = self._copy_pipeline_to_dtype(pipeline, dtype, pipeline.tokenizer)

        os.makedirs(Path(destination).absolute(), exist_ok=True)
        save_pipeline.save_pretrained(destination)

        if dtype is not None:
            del save_pipeline

    def __save_original(
            self,
            model: Krea2Model,
            destination: str,
            dtype: torch.dtype | None,
    ):
        # ORIGINAL_TRANSFORMER: krea/Krea-2-Raw's native "raw.safetensors" namespace.
        # Materialize any quantized (INT_W8A8 / FP8 / ...) linears back to real dense weights so a
        # training-time weight_dtype does not leak into the saved checkpoint. The quantized modules
        # keep low-bit codes in .weight plus a per-out-channel .scale buffer; dequantize folds them.
        # Weight *values* are swapped before name conversion (key set unchanged, so convert behaves
        # exactly as before); the now-orphan quant .scale sidecars are dropped after conversion. A
        # .scale is a quant sidecar iff it has a sibling .weight -- genuine norm .scale params don't.
        dequant_dtype = dtype if dtype is not None else torch.float32
        state_dict = model.transformer.state_dict()
        for name, module in model.transformer.named_modules():
            if isinstance(module, QuantizedLinearMixin) and (name + ".weight") in state_dict:
                state_dict[name + ".weight"] = \
                    get_unquantized_weight(module, dequant_dtype, torch.device("cpu")).to("cpu")

        state_dict = convert(state_dict, model.checkpoint_diffusers_to_original())

        for key in [k for k in state_dict
                    if k.endswith(".scale") and (k[:-len(".scale")] + ".weight") in state_dict]:
            del state_dict[key]

        save_state_dict = self._convert_state_dict_dtype(state_dict, dtype)
        self._convert_state_dict_to_contiguous(save_state_dict)

        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)

        save_file(save_state_dict, destination, self._create_safetensors_header(model, save_state_dict))

    def __save_internal(
            self,
            model: Krea2Model,
            destination: str,
    ):
        self.__save_diffusers(model, destination, None)

    def save(
            self,
            model: Krea2Model,
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
                self.__save_internal(model, output_model_destination)
            case _:
                raise NotImplementedError(f"Unsupported output format: {output_model_format}")
