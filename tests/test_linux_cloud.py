from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, Mock

from modules.cloud.LinuxCloud import LinuxCloud


class LinuxCloudTest(TestCase):
    def test_parse_git_clone_branch(self):
        repo, branch = LinuxCloud._LinuxCloud__parse_git_clone_install_cmd(
            "git clone --branch krea2-cloud https://github.com/vgaggia/OneTrainer.git"
        )

        self.assertEqual(repo, "https://github.com/vgaggia/OneTrainer.git")
        self.assertEqual(branch, "krea2-cloud")

    def test_parse_git_clone_short_branch_option(self):
        repo, branch = LinuxCloud._LinuxCloud__parse_git_clone_install_cmd(
            "git clone -b feature https://example.com/OneTrainer.git"
        )

        self.assertEqual(repo, "https://example.com/OneTrainer.git")
        self.assertEqual(branch, "feature")

    def test_non_clone_command_is_not_aligned(self):
        repo, branch = LinuxCloud._LinuxCloud__parse_git_clone_install_cmd("./install-custom.sh")

        self.assertIsNone(repo)
        self.assertIsNone(branch)

    def test_can_reattach_requires_a_live_matching_process(self):
        cloud = object.__new__(LinuxCloud)
        cloud.connection = Mock()
        cloud.connection.run.return_value.exited = 0
        cloud.pid_file = "/workspace/job 1.pid"
        cloud.config_file = "/workspace/job 1.json"

        self.assertTrue(cloud.can_reattach())

        command = cloud.connection.run.call_args.args[0]
        self.assertIn("kill -0", command)
        self.assertIn("/proc/$pid/cmdline", command)
        self.assertIn("grep -F -- '/workspace/job 1.json'", command)
        self.assertEqual(cloud.connection.run.call_count, 1)

    def test_can_reattach_removes_a_stale_pid_file(self):
        cloud = object.__new__(LinuxCloud)
        cloud.connection = Mock()
        cloud.connection.run.side_effect = [
            SimpleNamespace(exited=1),
            SimpleNamespace(exited=0),
        ]
        cloud.pid_file = "/workspace/job 1.pid"
        cloud.config_file = "/workspace/job 1.json"

        self.assertFalse(cloud.can_reattach())

        cleanup = cloud.connection.run.call_args_list[1]
        self.assertEqual(cleanup.args[0], "rm -f '/workspace/job 1.pid'")
        self.assertTrue(cleanup.kwargs["hide"])

    def test_detached_command_forwards_tokens_and_cuda_libraries(self):
        cloud = object.__new__(LinuxCloud)
        cloud.config = SimpleNamespace(
            cloud=SimpleNamespace(
                detach_trainer=True,
                huggingface_cache_dir="/workspace/hf cache",
                on_detached_error=None,
                on_detached_finish=None,
                onetrainer_dir="/workspace/One Trainer",
            ),
            secrets=SimpleNamespace(huggingface_token="hf_token with space"),
        )
        cloud.connection = Mock()
        cloud.can_reattach = Mock(return_value=False)
        cloud._get_action_cmd = Mock(return_value=":")
        cloud._LinuxCloud__trail_detached_trainer = Mock()
        cloud.exit_status_file = "/workspace/job.exit"
        cloud.log_file = "/workspace/job.log"
        cloud.pid_file = "/workspace/job.pid"
        cloud.environment_file = "/workspace/.job.env"
        cloud.callback_file = "/workspace/job.callback"
        cloud.command_pipe = "/workspace/job.command"
        cloud.config_file = "/workspace/job.json"
        remote_environment_file = Mock()
        remote_environment_context = MagicMock()
        remote_environment_context.__enter__.return_value = remote_environment_file
        cloud.connection.sftp.return_value.file.return_value = remote_environment_context

        cloud.run_trainer()

        command = cloud.connection.run.call_args_list[1].args[0]
        self.assertIn("site-packages/nvidia/*/lib", command)
        self.assertIn("export LD_LIBRARY_PATH=", command)
        self.assertIn("export HF_HUB_DISABLE_XET=0", command)
        self.assertIn("export HF_XET_HIGH_PERFORMANCE=1", command)
        self.assertIn("export HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY=1", command)
        self.assertIn("export HF_HUB_DOWNLOAD_TIMEOUT=60", command)
        self.assertIn("HF_HUB_DISABLE_XET=1", command)
        self.assertIn("model_type", command)
        self.assertIn("KREA_2", command)
        self.assertIn("raw.safetensors", command)
        self.assertIn("-c '", command)
        self.assertIn(" /workspace/job.json", command)
        self.assertNotIn("hf_token with space", command)
        self.assertIn(". /workspace/.job.env", command)
        self.assertIn("export HF_HOME='/workspace/hf cache'", command)
        self.assertIn("'/workspace/One Trainer'/run-cmd.sh", command)
        remote_environment_file.write.assert_called_once_with(
            "export HF_TOKEN='hf_token with space'\n"
            "export HUGGING_FACE_HUB_TOKEN='hf_token with space'\n"
        )
        cloud.connection.sftp.return_value.chmod.assert_called_once_with("/workspace/.job.env", 0o600)
        cloud._LinuxCloud__trail_detached_trainer.assert_called_once_with()
