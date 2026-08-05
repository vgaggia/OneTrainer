from modules.model.LongCatImageModel import LongCatImageModel
from modules.modelSetup.BaseLongCatImageSetup import BaseLongCatImageSetup
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.TrainProgress import TrainProgress

import torch


@factory.register(BaseModelSetup, ModelType.LONGCAT_IMAGE_EDIT, TrainingMethod.LORA)
class LongCatImageLoRASetup(BaseLongCatImageSetup):
    def __init__(self, train_device: torch.device, temp_device: torch.device, debug_mode: bool):
        super().__init__(train_device=train_device, temp_device=temp_device, debug_mode=debug_mode)

    def create_parameters(
            self, model: LongCatImageModel, config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        parameters = NamedParameterGroupCollection()
        self._create_model_part_parameters(
            parameters, "transformer", model.transformer_lora, config.transformer,
        )
        return parameters

    def __setup_requires_grad(self, model: LongCatImageModel, config: TrainConfig):
        model.text_encoder.requires_grad_(False)
        model.transformer.requires_grad_(False)
        model.vae.requires_grad_(False)
        self._setup_model_part_requires_grad(
            "transformer", model.transformer_lora, config.transformer, model.train_progress,
        )

    def setup_model(self, model: LongCatImageModel, config: TrainConfig):
        model.transformer_lora = LoRAModuleWrapper(
            model.transformer,
            "transformer",
            config,
            config.layer_filter.split(","),
            fusion_spec=model.fusion_groups(),
            fuse=config.output_model_format.needs_qkv_fusion(),
        )
        if model.lora_state_dict:
            model.transformer_lora.load_state_dict(model.lora_state_dict)
            model.lora_state_dict = None

        model.transformer_lora.set_dropout(config.dropout_probability)
        model.transformer_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
        model.transformer_lora.hook_to_module()

        parameters = self.create_parameters(model, config)
        self.__setup_requires_grad(model, config)
        init_model_parameters(model, parameters, self.train_device)

    def setup_train_device(self, model: LongCatImageModel, config: TrainConfig):
        model.text_encoder_to(self.temp_device if config.latent_caching else self.train_device)
        model.vae_to(self.temp_device if config.latent_caching else self.train_device)
        model.transformer_to(self.train_device)
        model.text_encoder.eval()
        model.vae.eval()
        model.transformer.train(config.transformer.train)

    def after_optimizer_step(
            self, model: LongCatImageModel, config: TrainConfig, train_progress: TrainProgress,
    ):
        self.__setup_requires_grad(model, config)
