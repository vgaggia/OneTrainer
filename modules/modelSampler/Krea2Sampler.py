from collections.abc import Callable

from modules.model.Krea2Model import Krea2Model
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.util import factory
from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.AudioFormat import AudioFormat
from modules.util.enum.FileType import FileType
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.NoiseScheduler import NoiseScheduler
from modules.util.enum.VideoFormat import VideoFormat
from modules.util.torch_util import torch_gc

import torch

import numpy as np
from PIL import Image
from tqdm import tqdm


class Krea2Sampler(BaseModelSampler):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            model: Krea2Model,
            model_type: ModelType,
    ):
        super().__init__(train_device, temp_device)

        self.model = model
        self.model_type = model_type

    @torch.no_grad()
    def __sample_base(
            self,
            prompt: str,
            negative_prompt: str,
            height: int,
            width: int,
            seed: int,
            random_seed: bool,
            diffusion_steps: int,
            cfg_scale: float,
            noise_scheduler: NoiseScheduler,
            on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ) -> ModelSamplerOutput:
        model = self.model
        with model.autocast_context:
            train_dtype = model.train_dtype.torch_dtype()
            generator = torch.Generator(device=self.train_device)
            if random_seed:
                generator.seed()
            else:
                generator.manual_seed(seed)

            do_cfg = cfg_scale > 1.0
            vae_scale_factor = 8
            latent_channels = model.transformer.config.channels

            latent_h = height // vae_scale_factor
            latent_w = width // vae_scale_factor

            # prompt conditioning
            model.text_encoder_to(self.train_device)
            text_encoder_output, text_mask = model.encode_text(
                text=[prompt, negative_prompt] if do_cfg else [prompt],
                batch_size=2 if do_cfg else 1,
                train_device=self.train_device,
            )
            model.text_encoder_to(self.temp_device)
            torch_gc()

            b = text_encoder_output.shape[0]
            context = text_encoder_output.reshape(
                b, text_encoder_output.shape[1], model.transformer.config.txtlayers, -1,
            ).to(dtype=train_dtype)

            # starting noise (single sample, tiled over the conditioning batch)
            noise = torch.randn(
                size=(1, latent_channels, 1, latent_h, latent_w),
                generator=generator,
                device=self.train_device,
                dtype=torch.float32,
            )
            x = model.pack_latents(noise)  # (1, L_img, 64)

            pos, mask = model.build_pos_mask(
                b, text_mask, latent_h, latent_w,
                model.transformer.config.patch, self.train_device,
            )

            # resolution-aware schedule, tied to the training time-shift for consistency.
            shift = model.calculate_timestep_shift(latent_h, latent_w)
            grid = torch.linspace(1, 0, diffusion_steps + 1)
            ts = (shift / (shift + (1.0 / grid - 1.0))).tolist()

            model.transformer_to(self.train_device)
            for i, (t_curr, t_prev) in enumerate(zip(tqdm(ts[:-1], desc="sampling"), ts[1:], strict=False)):
                x_in = x.expand(b, -1, -1).to(dtype=train_dtype)
                t = torch.full((b,), t_curr, device=self.train_device, dtype=train_dtype)
                v = model.transformer(img=x_in, context=context, t=t, pos=pos, mask=mask)
                v = v.to(torch.float32)
                if do_cfg:
                    v_cond, v_uncond = v[0:1], v[1:2]
                    v = v_cond + cfg_scale * (v_cond - v_uncond)
                x = x + (t_prev - t_curr) * v
                on_update_progress(i + 1, len(ts) - 1)

            model.transformer_to(self.temp_device)
            torch_gc()

            # decode
            model.vae_to(self.train_device)
            scaled_latent = model.unpack_latents(x, height=latent_h, width=latent_w)
            raw_latent = model.unscale_latents(scaled_latent).to(dtype=model.vae.dtype)
            image = model.vae.decode(raw_latent).sample  # (1, 3, 1, H, W)
            image = image.squeeze(2)
            model.vae_to(self.temp_device)
            torch_gc()

            image = (image.float().clamp(-1.0, 1.0) * 0.5 + 0.5)
            image = (image * 255.0).round().to(torch.uint8)
            image = image.permute(0, 2, 3, 1).cpu().numpy()

            return ModelSamplerOutput(
                file_type=FileType.IMAGE,
                data=Image.fromarray(np.ascontiguousarray(image[0])),
            )

    def sample(
            self,
            sample_config: SampleConfig,
            destination: str,
            image_format: ImageFormat | None = None,
            video_format: VideoFormat | None = None,
            audio_format: AudioFormat | None = None,
            on_sample: Callable[[ModelSamplerOutput], None] = lambda _: None,
            on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ):
        sampler_output = self.__sample_base(
            prompt=sample_config.prompt,
            negative_prompt=sample_config.negative_prompt,
            height=self.quantize_resolution(sample_config.height, 16),
            width=self.quantize_resolution(sample_config.width, 16),
            seed=sample_config.seed,
            random_seed=sample_config.random_seed,
            diffusion_steps=sample_config.diffusion_steps,
            cfg_scale=sample_config.cfg_scale,
            noise_scheduler=sample_config.noise_scheduler,
            on_update_progress=on_update_progress,
        )

        self.save_sampler_output(
            sampler_output, destination,
            image_format, video_format, audio_format,
        )

        on_sample(sampler_output)

factory.register(BaseModelSampler, Krea2Sampler, ModelType.KREA_2)
