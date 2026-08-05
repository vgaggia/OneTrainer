import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from modules.modelLoader.krea2.Krea2ModelLoader import Krea2ModelLoader
from modules.modelLoader.mixin.HFModelLoaderMixin import _parameter_from_loaded_tensor
from modules.modelLoader.mixin.InternalModelLoaderMixin import InternalModelLoaderMixin
from modules.modelSaver.mixin.InternalModelSaverMixin import InternalModelSaverMixin
from modules.module.quantized.LinearW8A8 import LinearW8A8
from modules.util.quantized_weight_training import (
    FusedQuantizedAdafactor,
    quantized_weight_training_state_dict,
)

import torch
from torch import nn


class QuantizedWeightTrainingResumeTest(unittest.TestCase):
    @staticmethod
    def _qwt_root():
        root = nn.Sequential(LinearW8A8(torch.int8, 4, 3, bias=False))
        module = root[0]
        module.weight = nn.Parameter(
            torch.arange(-6, 6, dtype=torch.int8).reshape(3, 4),
            requires_grad=False,
        )
        module.quantize()
        module.qwt_updater = FusedQuantizedAdafactor(
            module,
            lr=1e-5,
            clip_grad_norm=1.0,
            initial_step=41,
        )
        module.qwt_updater.row = torch.tensor([1.0, 2.0, 3.0])
        module.qwt_updater.col = torch.tensor([4.0, 5.0, 6.0, 7.0])
        return root

    def test_already_quantized_weight_and_scale_are_preserved(self):
        source = LinearW8A8(torch.int8, 16, 32, bias=False)
        source.weight.data.normal_(mean=0.0, std=0.2)
        source.quantize()

        saved_weight = source.weight.detach().clone()
        # Legacy internal backups stored scale in the bf16 model dtype.
        saved_scale = source.scale.detach().to(torch.bfloat16)

        restored = LinearW8A8(torch.int8, 16, 32, bias=False)
        restored.weight = nn.Parameter(saved_weight.clone(), requires_grad=False)
        restored.scale.data = saved_scale.clone()
        restored.quantize()

        self.assertTrue(torch.equal(restored.weight, saved_weight))
        self.assertEqual(restored.scale.dtype, torch.float32)
        self.assertTrue(torch.equal(restored.scale, saved_scale.float()))

    def test_integer_checkpoint_weight_becomes_a_frozen_parameter(self):
        placeholder = nn.Parameter(torch.empty(3, 4, device="meta"), requires_grad=True)
        value = torch.ones(3, 4, dtype=torch.int8)

        loaded = _parameter_from_loaded_tensor(placeholder, value)

        self.assertEqual(loaded.dtype, torch.int8)
        self.assertFalse(loaded.requires_grad)

    def test_krea_internal_resume_ignores_original_transformer_override(self):
        loader = Krea2ModelLoader()
        load_diffusers = Mock()
        loader._Krea2ModelLoader__load_diffusers = load_diffusers

        with patch("modules.modelLoader.krea2.Krea2ModelLoader.os.path.isfile", return_value=True):
            loader._Krea2ModelLoader__load_internal(
                Mock(), Mock(), Mock(), "backup", "original-transformer.safetensors", "vae", Mock()
            )

        self.assertEqual(load_diffusers.call_args.args[4], "")

    def test_new_quantization_keeps_scale_in_float32(self):
        module = LinearW8A8(torch.int8, 16, 32, bias=False)
        module.scale.data = module.scale.to(torch.bfloat16)
        module.quantize()

        self.assertEqual(module.weight.dtype, torch.int8)
        self.assertEqual(module.scale.dtype, torch.float32)

    def test_qwt_optimizer_state_round_trip(self):
        root = self._qwt_root()
        module = root[0]

        saved = quantized_weight_training_state_dict(root)
        restored = FusedQuantizedAdafactor(
            module,
            lr=1e-5,
            clip_grad_norm=1.0,
            state_dict=saved["0"],
        )

        self.assertEqual(restored.step_count, 41)
        self.assertTrue(torch.equal(restored.row, module.qwt_updater.row))
        self.assertTrue(torch.equal(restored.col, module.qwt_updater.col))

    def test_internal_backup_saves_and_extracts_qwt_state(self):
        root = self._qwt_root()
        regular_parameter = nn.Parameter(torch.tensor([1.0]))
        model = SimpleNamespace(
            optimizer=torch.optim.SGD([regular_parameter], lr=0.1),
            param_group_mapping=["transformer"],
            train_config=SimpleNamespace(optimizer=SimpleNamespace(optimizer="ADAFACTOR")),
            ema=None,
            train_progress=SimpleNamespace(epoch=0, epoch_step=41, epoch_sample=246, global_step=41),
            quantized_weight_training_components=lambda: {"transformer": root},
        )

        with tempfile.TemporaryDirectory(dir=".") as backup_dir:
            InternalModelSaverMixin()._save_internal_data(model, backup_dir)

            loaded_model = SimpleNamespace(
                optimizer_state_dict=None,
                quantized_weight_training_state_dict=None,
                ema_state_dict=None,
                train_progress=None,
            )
            InternalModelLoaderMixin()._load_internal_data(loaded_model, backup_dir)

        qwt_state = loaded_model.quantized_weight_training_state_dict["transformer"]["0"]
        self.assertEqual(qwt_state["step_count"], 41)
        self.assertTrue(torch.equal(qwt_state["row"], root[0].qwt_updater.row))
        self.assertNotIn("quantized_weight_training", loaded_model.optimizer_state_dict)


if __name__ == "__main__":
    unittest.main()
