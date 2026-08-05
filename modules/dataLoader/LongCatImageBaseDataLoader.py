import os

from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.dataLoader.longcatImage.EncodeLongCatText import EncodeLongCatText
from modules.dataLoader.mixin.DataLoaderText2ImageMixin import DataLoaderText2ImageMixin
from modules.dataLoader.pipelineModules.MultiplyInputs import MultiplyInputs
from modules.model.LongCatImageModel import LongCatImageModel
from modules.modelSetup.BaseLongCatImageSetup import BaseLongCatImageSetup
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.TrainProgress import TrainProgress

from mgds.pipelineModules.DecodeVAE import DecodeVAE
from mgds.pipelineModules.EncodeVAE import EncodeVAE
from mgds.pipelineModules.RescaleImageChannels import RescaleImageChannels
from mgds.pipelineModules.SampleVAEDistribution import SampleVAEDistribution
from mgds.pipelineModules.SaveImage import SaveImage
from mgds.pipelineModules.SaveText import SaveText
from mgds.pipelineModules.ScaleImage import ScaleImage


@factory.register(BaseDataLoader, ModelType.LONGCAT_IMAGE_EDIT)
class LongCatImageBaseDataLoader(BaseDataLoader, DataLoaderText2ImageMixin):
    def _preparation_modules(self, config: TrainConfig, model: LongCatImageModel):
        modules = []

        if config.custom_conditioning_image:
            # Qwen2.5-VL sees a half-resolution copy; the transformer sees the full-resolution VAE latent.
            modules.append(ScaleImage(
                in_name="custom_conditioning_image",
                out_name="prompt_conditioning_image",
                factor=0.5,
            ))

        modules.extend([
            RescaleImageChannels(
                image_in_name="image",
                image_out_name="image",
                in_range_min=0,
                in_range_max=1,
                out_range_min=-1,
                out_range_max=1,
            ),
            EncodeVAE(
                in_name="image",
                out_name="latent_image_distribution",
                vae=model.vae,
                autocast_contexts=[model.autocast_context],
                dtype=model.train_dtype.torch_dtype(),
            ),
            SampleVAEDistribution(
                in_name="latent_image_distribution",
                out_name="latent_image",
                mode="mean",
            ),
        ])

        if config.custom_conditioning_image:
            modules.extend([
                RescaleImageChannels(
                    image_in_name="custom_conditioning_image",
                    image_out_name="custom_conditioning_image",
                    in_range_min=0,
                    in_range_max=1,
                    out_range_min=-1,
                    out_range_max=1,
                ),
                EncodeVAE(
                    in_name="custom_conditioning_image",
                    out_name="latent_conditioning_image_distribution",
                    vae=model.vae,
                    autocast_contexts=[model.autocast_context],
                    dtype=model.train_dtype.torch_dtype(),
                ),
                SampleVAEDistribution(
                    in_name="latent_conditioning_image_distribution",
                    out_name="latent_conditioning_image",
                    mode="mean",
                ),
            ])

        modules.append(EncodeLongCatText(
            prompt_name="prompt",
            conditioning_image_name=("prompt_conditioning_image" if config.custom_conditioning_image else None),
            image_variations_name=("concept.image_variations" if config.custom_conditioning_image else None),
            text_variations_name=("concept.text_variations" if config.custom_conditioning_image else None),
            hidden_state_out_name="text_encoder_hidden_state",
            model=model,
            max_token_length=config.text_encoder_sequence_length,
        ))
        return modules

    def _cache_modules(
            self,
            config: TrainConfig,
            model: LongCatImageModel,
            model_setup: BaseLongCatImageSetup,
    ):
        image_split_names = ["latent_image", "original_resolution", "crop_offset"]
        image_aggregate_names = ["crop_resolution", "image_path"]
        text_split_names = ["text_encoder_hidden_state"]

        modules = []
        text_variations_in_name = "concept.text_variations"
        text_variations_group_in_names = None
        if config.custom_conditioning_image:
            # Edit embeddings depend on both the aligned source crop and the independently
            # varied prompt. Cache their Cartesian product without duplicating VAE latents.
            image_split_names.append("latent_conditioning_image")
            text_variations_in_name = "longcat_edit_variations"
            text_variations_group_in_names = [
                "concept.path",
                "concept.seed",
                "concept.include_subdirectories",
                "concept.image",
                "concept.text",
            ]
            modules.append(MultiplyInputs(
                in_names=["concept.image_variations", "concept.text_variations"],
                out_name=text_variations_in_name,
            ))

        sort_names = image_aggregate_names + image_split_names + ["prompt", "concept"]

        modules.extend(self._cache_modules_from_names(
            model,
            model_setup,
            image_split_names=image_split_names,
            image_aggregate_names=image_aggregate_names,
            text_split_names=text_split_names,
            sort_names=sort_names,
            config=config,
            text_caching=True,
            text_variations_in_name=text_variations_in_name,
            text_variations_group_in_names=text_variations_group_in_names,
        ))
        return modules

    def _output_modules(
            self,
            config: TrainConfig,
            model: LongCatImageModel,
            model_setup: BaseLongCatImageSetup,
    ):
        output_names = [
            "image_path",
            "latent_image",
            "prompt",
            "text_encoder_hidden_state",
            "original_resolution",
            "crop_resolution",
            "crop_offset",
        ]
        if config.custom_conditioning_image:
            output_names.append("latent_conditioning_image")

        return self._output_modules_from_out_names(
            model,
            model_setup,
            output_names=output_names,
            config=config,
            use_conditioning_image=False,
            vae=model.vae,
            autocast_context=[model.autocast_context],
            train_dtype=model.train_dtype,
        )

    def _debug_modules(self, config: TrainConfig, model: LongCatImageModel):
        debug_dir = os.path.join(config.debug_dir, "dataloader")

        def before_save_fun():
            model.vae_to(self.train_device)

        decode_image = DecodeVAE(
            in_name="latent_image",
            out_name="decoded_image",
            vae=model.vae,
            autocast_contexts=[model.autocast_context],
            dtype=model.train_dtype.torch_dtype(),
        )
        save_image = SaveImage(
            image_in_name="decoded_image",
            original_path_in_name="image_path",
            path=debug_dir,
            in_range_min=-1,
            in_range_max=1,
            before_save_fun=before_save_fun,
        )
        save_prompt = SaveText(
            text_in_name="prompt",
            original_path_in_name="image_path",
            path=debug_dir,
            before_save_fun=before_save_fun,
        )
        modules = [decode_image, save_image, save_prompt]

        if config.custom_conditioning_image:
            decode_source = DecodeVAE(
                in_name="latent_conditioning_image",
                out_name="decoded_conditioning_image",
                vae=model.vae,
                autocast_contexts=[model.autocast_context],
                dtype=model.train_dtype.torch_dtype(),
            )
            save_source = SaveImage(
                image_in_name="decoded_conditioning_image",
                original_path_in_name="image_path",
                path=os.path.join(debug_dir, "conditioning"),
                in_range_min=-1,
                in_range_max=1,
                before_save_fun=before_save_fun,
            )
            modules.extend([decode_source, save_source])

        return modules

    def _create_dataset(
            self,
            config: TrainConfig,
            model: LongCatImageModel,
            model_setup: BaseLongCatImageSetup,
            train_progress: TrainProgress,
            is_validation: bool = False,
    ):
        return DataLoaderText2ImageMixin._create_dataset(
            self,
            config,
            model,
            model_setup,
            train_progress,
            is_validation,
            aspect_bucketing_quantization=16,
            # LongCat's optional source is an edit reference, not an inpainting fallback.
            supports_inpainting=False,
        )
