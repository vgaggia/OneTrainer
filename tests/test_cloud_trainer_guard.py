from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock

from modules.trainer.CloudTrainer import CloudTrainer
from modules.util.enum.CloudType import CloudType

# The guard is name-mangled; bind it once so the tests read cleanly.
check = CloudTrainer._CloudTrainer__check_cache_not_cleared_on_network_volume


def make_config(
        network_volume_id="vol-1",
        clear_cache=True,
        latent_caching=True,
        cloud_type=CloudType.RUNPOD,
) -> SimpleNamespace:
    return SimpleNamespace(
        clear_cache_before_training=clear_cache,
        latent_caching=latent_caching,
        cloud=SimpleNamespace(network_volume_id=network_volume_id, type=cloud_type),
    )


class ClearCacheOnNetworkVolumeGuardTest(TestCase):
    def test_refuses_to_clear_the_cache_on_a_network_volume(self):
        with self.assertRaises(ValueError) as caught:
            check(make_config())

        # the volume id belongs in the message - that is what the user has to go look at
        self.assertIn("vol-1", str(caught.exception))

    def test_allows_clearing_a_pod_local_cache(self):
        check(make_config(network_volume_id=""))

    def test_allows_a_network_volume_when_the_cache_is_kept(self):
        check(make_config(clear_cache=False))

    def test_ignores_the_flag_when_latent_caching_is_off(self):
        # GenericTrainer only clears the cache when latent_caching is on, so this combination
        # never deletes anything and must not block the run.
        check(make_config(latent_caching=False))

    def test_ignores_runpod_volume_settings_for_vast(self):
        check(make_config(cloud_type=CloudType.VAST))


class CloudTrainerReattachTest(TestCase):
    def test_missing_run_skips_install_and_explains_to_use_start(self):
        trainer = object.__new__(CloudTrainer)
        trainer.callbacks = SimpleNamespace(on_update_status=Mock())
        trainer.cloud = SimpleNamespace(
            setup=Mock(),
            can_reattach=Mock(return_value=False),
            can_recover_completed_run=Mock(return_value=False),
        )
        trainer.reattach = True
        trainer.config = SimpleNamespace(cloud=SimpleNamespace(run_id="job1"))
        trainer.error_caught = False

        with self.assertRaisesRegex(ValueError, "normal Start button"):
            trainer.start()

        trainer.cloud.setup.assert_called_once_with(install=False)
        self.assertTrue(trainer.error_caught)
