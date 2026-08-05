from modules.model.LongCatImageModel import LongCatImageModel

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import torch


def prepare_longcat_qwen_image(image: torch.Tensor) -> torch.Tensor:
    """Convert MGDS' [0, 1] image tensor to Qwen's expected [0, 255] input range."""
    return image.to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0).mul(255.0)


def split_longcat_edit_variation(
        variation: int,
        image_variations: int,
        text_variations: int,
) -> tuple[int, int]:
    """Map a joint edit-cache variation to aligned image and independent text variations."""
    image_variations = max(1, image_variations)
    text_variations = max(1, text_variations)
    return variation % image_variations, (variation // image_variations) % text_variations


class EncodeLongCatText(PipelineModule, RandomAccessPipelineModule):
    def __init__(
            self,
            prompt_name: str,
            hidden_state_out_name: str,
            model: LongCatImageModel,
            conditioning_image_name: str | None = None,
            image_variations_name: str | None = None,
            text_variations_name: str | None = None,
            max_token_length: int | None = None,
    ):
        super().__init__()
        self.prompt_name = prompt_name
        self.hidden_state_out_name = hidden_state_out_name
        self.model = model
        self.conditioning_image_name = conditioning_image_name
        self.image_variations_name = image_variations_name
        self.text_variations_name = text_variations_name
        self.max_token_length = max_token_length

    def length(self) -> int:
        return self._get_previous_length(self.prompt_name)

    def get_inputs(self) -> list[str]:
        inputs = [self.prompt_name]
        if self.conditioning_image_name is not None:
            inputs.append(self.conditioning_image_name)
        if self.image_variations_name is not None:
            inputs.append(self.image_variations_name)
        if self.text_variations_name is not None:
            inputs.append(self.text_variations_name)
        return inputs

    def get_outputs(self) -> list[str]:
        return [self.hidden_state_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        image_variation = variation
        text_variation = variation
        if self.image_variations_name is not None and self.text_variations_name is not None:
            image_variations = int(self._get_previous_item(0, self.image_variations_name, index))
            text_variations = int(self._get_previous_item(0, self.text_variations_name, index))
            image_variation, text_variation = split_longcat_edit_variation(
                variation, image_variations, text_variations,
            )

        prompt = self._get_previous_item(text_variation, self.prompt_name, index)
        conditioning_image = None
        if self.conditioning_image_name is not None:
            conditioning_image = self._get_previous_item(
                image_variation, self.conditioning_image_name, index,
            )
            # Qwen's image processor expects image values in [0, 255] and applies its
            # own 1 / 255 rescale. MGDS supplies [0, 1], so restore the expected input
            # range while keeping the tensor on CPU and NumPy-compatible float32.
            conditioning_image = prepare_longcat_qwen_image(conditioning_image)

        hidden_state = self.model.encode_text(
            train_device=self.model.text_encoder.device,
            text=prompt,
            conditioning_image=conditioning_image,
            text_encoder_sequence_length=self.max_token_length,
        ).squeeze(0)

        return {self.hidden_state_out_name: hidden_state}
