from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from modules.dataLoader.longcatImage.EncodeLongCatText import (
    prepare_longcat_qwen_image,
    split_longcat_edit_variation,
)
from modules.model.LongCatImageModel import LongCatImageModel
from modules.modelSampler.LongCatImageSampler import combine_longcat_cfg
from modules.modelSaver.longcatImage.LongCatImageLoRASaver import LongCatImageLoRASaver
from modules.modelSetup.LongCatImageFineTuneSetup import LongCatImageFineTuneSetup
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util import checkpointing_util, compile_util
from modules.util.config.TrainConfig import TrainConfig
from modules.util.convert_util import convert, reverse_conversion, split_fused_state_dict
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType

import torch

from diffusers import LongCatImageTransformer2DModel

from safetensors.torch import load_file


def _tiny_transformer() -> LongCatImageTransformer2DModel:
    return LongCatImageTransformer2DModel(
        in_channels=64,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=16,
        pooled_projection_dim=16,
        axes_dims_rope=[2, 2, 4],
    )


def test_longcat_native_checkpoint_round_trip():
    model = LongCatImageModel(ModelType.LONGCAT_IMAGE_EDIT)
    canonical = _tiny_transformer().state_dict()

    native = convert(canonical, model.checkpoint_diffusers_to_original())

    assert "double_blocks.0.img_attn.qkv.weight" in native
    assert "double_blocks.0.txt_attn.qkv.weight" in native
    assert "single_blocks.0.linear1.weight" in native
    assert "final_layer.adaLN_modulation.1.weight" in native

    canonical_fused = convert(native, reverse_conversion(model.diffusers_to_original()))
    restored = split_fused_state_dict(canonical_fused, canonical, model.fusion_groups())

    assert restored.keys() == canonical.keys()
    for key in canonical:
        assert torch.equal(restored[key], canonical[key]), key


def test_split_fused_state_dict_uses_target_shapes():
    fused = {"blocks.0.qkv.weight": torch.arange(24).view(6, 4)}
    target = {
        "blocks.0.q.weight": torch.empty(1, 4),
        "blocks.0.k.weight": torch.empty(2, 4),
        "blocks.0.v.weight": torch.empty(3, 4),
    }
    groups = [("blocks.{i}", ["q", "k", "v"], "qkv", "qkv")]

    split = split_fused_state_dict(fused, target, groups)

    assert [split[key].shape[0] for key in target] == [1, 2, 3]
    assert torch.equal(torch.cat([split[key] for key in target]), fused["blocks.0.qkv.weight"])


def test_longcat_qwen_image_uses_processor_input_range():
    image = torch.tensor([0.0, 0.25, 1.0], dtype=torch.bfloat16)

    prepared = prepare_longcat_qwen_image(image)

    assert prepared.device.type == "cpu"
    assert prepared.dtype == torch.float32
    assert torch.equal(prepared, torch.tensor([0.0, 63.75, 255.0]))


def test_longcat_edit_variations_cover_cartesian_product():
    variations = [split_longcat_edit_variation(i, 2, 3) for i in range(6)]

    assert variations == [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)]


def test_longcat_t2i_cfg_renormalizes_to_conditional_norm():
    positive = torch.tensor([[[3.0, 4.0]]])
    negative = torch.tensor([[[0.0, 0.0]]])

    guided = combine_longcat_cfg(positive, negative, cfg_scale=4.5, renormalize=True)
    ordinary = combine_longcat_cfg(positive, negative, cfg_scale=4.5, renormalize=False)

    assert torch.allclose(torch.norm(guided, dim=-1), torch.tensor([[5.0]]))
    assert torch.equal(ordinary, positive * 4.5)


def test_longcat_comfy_lora_export_uses_native_fused_names():
    model = LongCatImageModel(ModelType.LONGCAT_IMAGE_EDIT)
    model.transformer = _tiny_transformer()
    config = TrainConfig.default_values()
    config.lora_rank = 2
    config.lora_alpha = 1.0
    config.layer_filter = "attn,ff,proj_mlp"
    model.transformer_lora = LoRAModuleWrapper(
        model.transformer,
        "transformer",
        config,
        config.layer_filter.split(","),
        fusion_spec=model.fusion_groups(),
        fuse=True,
    )

    with TemporaryDirectory() as directory:
        destination = str(Path(directory) / "longcat-comfy-lora.safetensors")
        LongCatImageLoRASaver().save(model, ModelFormat.COMFY_LORA, destination, torch.bfloat16)
        keys = set(load_file(destination))

    assert any(key.startswith("diffusion_model.double_blocks.0.img_attn.qkv.") for key in keys)
    assert any(key.startswith("diffusion_model.single_blocks.0.linear1.") for key in keys)
    assert not any("transformer_blocks" in key for key in keys)


def test_flux_checkpoint_helper_accepts_compile_override():
    captured = {}

    def fake_enable_checkpointing(model, config, part, compile, layers):
        captured["compile"] = compile
        captured["layers"] = layers

    model = SimpleNamespace(transformer_blocks=object(), single_transformer_blocks=object())
    config = SimpleNamespace(compile=True)

    with patch.object(checkpointing_util, "enable_checkpointing", fake_enable_checkpointing):
        checkpointing_util.enable_checkpointing_for_flux_transformer(
            model,
            config,
            object(),
            compile=False,
        )

    assert captured["compile"] is False
    assert len(captured["layers"]) == 2


def test_inductor_mixed_order_reduction_workaround():
    original = torch._inductor.config.triton.mix_order_reduction
    try:
        torch._inductor.config.triton.mix_order_reduction = True

        compile_util.disable_inductor_mixed_order_reduction()

        assert torch._inductor.config.triton.mix_order_reduction is False
    finally:
        torch._inductor.config.triton.mix_order_reduction = original


def test_longcat_setup_keeps_compile_enabled_with_workaround():
    setup = LongCatImageFineTuneSetup(torch.device("cpu"), torch.device("cpu"), False)
    model = SimpleNamespace(transformer=object(), text_encoder=object(), vae=object())
    config = TrainConfig.default_values()
    config.compile = True

    with (
        patch("modules.modelSetup.BaseLongCatImageSetup.disable_inductor_mixed_order_reduction") as workaround,
        patch("modules.modelSetup.BaseLongCatImageSetup.enable_checkpointing_for_flux_transformer") as checkpoint,
        patch("modules.modelSetup.BaseLongCatImageSetup.enable_checkpointing_for_qwen25vl_encoder_layers"),
        patch(
            "modules.modelSetup.BaseLongCatImageSetup.create_autocast_context",
            return_value=(SimpleNamespace(), torch.bfloat16),
        ),
        patch(
            "modules.modelSetup.BaseLongCatImageSetup.disable_fp16_autocast_context",
            return_value=(SimpleNamespace(), torch.bfloat16),
        ),
        patch("modules.modelSetup.BaseLongCatImageSetup.quantize_layers"),
        patch.object(setup, "_set_attention_backend"),
    ):
        setup.setup_optimizations(model, config)

    workaround.assert_called_once_with()
    checkpoint.assert_called_once_with(model.transformer, config, config.transformer)
