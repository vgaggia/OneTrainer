from modules.model.LongCatImageModel import LongCatImageModel
from modules.modelSetup.BaseLongCatImageSetup import BaseLongCatImageSetup
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ModuleFilter import ModuleFilter
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.TrainProgress import TrainProgress

import torch


@factory.register(BaseModelSetup, ModelType.LONGCAT_IMAGE_EDIT, TrainingMethod.FINE_TUNE)
class LongCatImageFineTuneSetup(BaseLongCatImageSetup):
    def __init__(self, train_device: torch.device, temp_device: torch.device, debug_mode: bool):
        super().__init__(train_device=train_device, temp_device=temp_device, debug_mode=debug_mode)

    def create_parameters(
            self, model: LongCatImageModel, config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        parameters = NamedParameterGroupCollection()
        self._create_model_part_parameters(
            parameters,
            "transformer",
            model.transformer,
            config.transformer,
            freeze=ModuleFilter.create(config),
            debug=config.debug_mode,
        )
        return parameters

    def __setup_requires_grad(self, model: LongCatImageModel, config: TrainConfig):
        self._setup_model_part_requires_grad(
            "transformer", model.transformer, config.transformer, model.train_progress,
        )
        model.vae.requires_grad_(False)
        model.text_encoder.requires_grad_(False)

    def setup_model(self, model: LongCatImageModel, config: TrainConfig):
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
