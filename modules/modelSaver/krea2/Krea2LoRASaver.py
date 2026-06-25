import os
from pathlib import Path

from modules.model.Krea2Model import Krea2Model, convert_krea2_lora_state_dict_to_external
from modules.modelSaver.mixin.LoRASaverMixin import LoRASaverMixin
from modules.util.convert.lora.convert_lora_util import LoraConversionKeySet
from modules.util.enum.ModelFormat import ModelFormat

import torch
from torch import Tensor

from safetensors.torch import save_file


class Krea2LoRASaver(
    LoRASaverMixin,
):
    def __init__(self):
        super().__init__()

    def _get_convert_key_sets(self, model: Krea2Model) -> list[LoraConversionKeySet] | None:
        # Krea 2 uses a simple explicit transformer. -> diffusion_model. export
        # below; the generic OMI/legacy converters would underscore the module path.
        return None

    def _get_state_dict(
            self,
            model: Krea2Model,
    ) -> dict[str, Tensor]:
        state_dict = {}
        if model.transformer_lora is not None:
            state_dict |= model.transformer_lora.state_dict()
        if model.lora_state_dict is not None:
            state_dict |= model.lora_state_dict
        return state_dict

    def __save_external_safetensors(
            self,
            model: Krea2Model,
            destination: str,
            dtype: torch.dtype | None,
    ):
        state_dict = self._get_state_dict(model)
        state_dict = convert_krea2_lora_state_dict_to_external(state_dict)
        save_state_dict = self._convert_state_dict_dtype(state_dict, dtype)
        self._convert_state_dict_to_contiguous(save_state_dict)

        if not save_state_dict:
            raise RuntimeError("Refusing to save an empty Krea 2 LoRA state dict.")
        if any(key.startswith("transformer.") for key in save_state_dict):
            raise RuntimeError("Krea 2 external LoRA conversion left native transformer.* keys.")

        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)
        save_file(
            save_state_dict,
            destination,
            self._create_safetensors_header(model, save_state_dict),
        )

    def save(
            self,
            model: Krea2Model,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        match output_model_format:
            case ModelFormat.SAFETENSORS | ModelFormat.LEGACY_SAFETENSORS | ModelFormat.COMFY_LORA:
                # Canonical Krea/ai-toolkit/ComfyUI namespace. Current ComfyUI
                # accepts diffusion_model., transformer. and bare names; using
                # diffusion_model. also matches ai-toolkit's published export.
                self.__save_external_safetensors(model, output_model_destination, dtype)
            case ModelFormat.INTERNAL:
                # Keep OneTrainer-native transformer.* keys inside resumable backups.
                self._save(model, output_model_format, output_model_destination, dtype)
            case ModelFormat.DIFFUSERS:
                raise NotImplementedError("Krea 2 has no diffusers pipeline directory format.")
            case _:
                raise NotImplementedError(f"Unsupported Krea 2 LoRA output format: {output_model_format}")
