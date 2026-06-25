import os.path
from pathlib import Path

from modules.model.Krea2Model import Krea2Model
from modules.modelSaver.mixin.DtypeModelSaverMixin import DtypeModelSaverMixin
from modules.util.enum.ModelFormat import ModelFormat

import torch

from safetensors.torch import save_file


class Krea2ModelSaver(
    DtypeModelSaverMixin,
):
    def __init__(self):
        super().__init__()

    def __save_safetensors(
            self,
            model: Krea2Model,
            destination: str,
            dtype: torch.dtype | None,
    ):
        state_dict = model.transformer.state_dict()
        save_state_dict = self._convert_state_dict_dtype(state_dict, dtype)
        self._convert_state_dict_to_contiguous(save_state_dict)

        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)

        save_file(save_state_dict, destination, self._create_safetensors_header(model, save_state_dict))

    def __save_internal(
            self,
            model: Krea2Model,
            destination: str,
    ):
        # No diffusers pipeline for SingleStreamDiT, so an internal backup is just
        # the transformer safetensors inside the backup dir; the loader resolves it
        # as the lone .safetensors when base_model points at the dir.
        os.makedirs(Path(destination).absolute(), exist_ok=True)
        self.__save_safetensors(model, os.path.join(destination, "transformer.safetensors"), None)

    def save(
            self,
            model: Krea2Model,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        match output_model_format:
            case ModelFormat.DIFFUSERS:
                raise NotImplementedError("Krea 2 has no diffusers pipeline; use the safetensors output format.")
            case ModelFormat.SAFETENSORS:
                self.__save_safetensors(model, output_model_destination, dtype)
            case ModelFormat.INTERNAL:
                self.__save_internal(model, output_model_destination)
