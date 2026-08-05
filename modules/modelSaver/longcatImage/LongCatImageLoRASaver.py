from modules.model.LongCatImageModel import LongCatImageModel
from modules.modelSaver.mixin.LoRASaverMixin import LoRASaverMixin

from torch import Tensor


class LongCatImageLoRASaver(LoRASaverMixin):
    def __init__(self):
        super().__init__()

    def _get_state_dict(self, model: LongCatImageModel) -> dict[str, Tensor]:
        state_dict = {}
        if model.transformer_lora is not None:
            state_dict |= model.transformer_lora.state_dict()
        if model.lora_state_dict is not None:
            state_dict |= model.lora_state_dict
        return state_dict
