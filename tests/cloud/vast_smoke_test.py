"""Opt-in, paid Vast.ai end-to-end smoke test.

The API key is read only by this process from VAST_AI_API_KEY and is never logged.
Every instance created by this script is destroyed in the cleanup block.

Run from the repository root with:
    python -m tests.cloud.vast_smoke_test
"""

import argparse
import os
import shlex
import tempfile
import time
import uuid
from pathlib import Path

from modules.cloud.VastCloud import VastCloud
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.CloudFileSync import CloudFileSync
from modules.util.enum.CloudType import CloudType


def _price(offer: dict) -> float:
    try:
        return float(offer.get("dph_total", float("inf")))
    except (TypeError, ValueError):
        return float("inf")


def _wait_until_destroyed(cloud: VastCloud, instance_id: str):
    for _ in range(12):
        if cloud.api.get_instance(instance_id) is None:
            print(f"cleanup confirmed for Vast.ai instance {instance_id}")
            return
        time.sleep(5)
    raise RuntimeError(f"Vast.ai instance {instance_id} still exists after destroy request")


def run_smoke_test(max_hourly_price: float, requested_gpu_type: str):
    api_key = os.environ.get("VAST_AI_API_KEY", "")
    if not api_key:
        raise RuntimeError("VAST_AI_API_KEY is not available to the smoke-test process")

    config = TrainConfig.default_values()
    config.cloud.enabled = True
    config.cloud.type = CloudType.VAST
    config.cloud.file_sync = CloudFileSync.NATIVE_SCP
    config.cloud.create = True
    config.cloud.name = f"OneTrainer Vast smoke {uuid.uuid4().hex[:8]}"
    config.cloud.tensorboard_tunnel = False
    config.cloud.vast_instance_type = "ondemand"
    config.cloud.gpu_count = 1
    config.cloud.volume_size = 10
    config.cloud.min_download = 0
    config.cloud.remote_dir = "/workspace"
    config.cloud.install_onetrainer = False
    config.cloud.update_onetrainer = False
    config.secrets.cloud.vast_api_key = api_key
    config.secrets.cloud.id = ""
    config.secrets.cloud.host = ""
    config.secrets.cloud.port = 0
    config.secrets.cloud.user = "root"
    config.secrets.cloud.key_file = ""

    cloud = VastCloud(config)
    instance_id = ""
    detached_marker = f"/tmp/onetrainer-smoke-{uuid.uuid4().hex}.txt"
    try:
        offers = cloud.api.search_offers(cloud._offer_filters(
            gpu_type=requested_gpu_type,
            instance_type=config.cloud.vast_instance_type,
            volume_size=config.cloud.volume_size,
            min_download=config.cloud.min_download,
            gpu_count=config.cloud.gpu_count,
            limit=100,
        ))
        offers = [
            offer for offer in offers
            if offer.get("gpu_name") and _price(offer) <= max_hourly_price
        ]
        if not offers:
            raise RuntimeError(
                f"No direct Vast.ai offer is available at or below ${max_hourly_price:.2f}/hr"
            )
        selected_offer = min(offers, key=_price)
        config.cloud.gpu_type = str(selected_offer["gpu_name"])
        print(
            f"smoke test selected {config.cloud.gpu_type} "
            f"at an advertised ${_price(selected_offer):.4f}/hr maximum candidate price"
        )

        # Exercise the same create -> wait -> attach key -> SSH path used by CloudTrainer.
        cloud.setup(install=False)
        instance_id = str(config.secrets.cloud.id)

        result = cloud.connection.run(
            "printf onetrainer-vast-ssh-ok",
            hide=True,
            in_stream=False,
        )
        if result.stdout != "onetrainer-vast-ssh-ok":
            raise RuntimeError("SSH command round-trip returned unexpected output")
        print("SSH command round-trip passed")

        with tempfile.TemporaryDirectory(prefix="onetrainer-vast-smoke-") as temp_dir:
            local_source = Path(temp_dir) / "remote-config.json"
            local_result = Path(temp_dir) / "downloaded-config.json"
            expected = f"native-scp-round-trip-{uuid.uuid4().hex}"
            local_source.write_text(expected, encoding="utf-8")
            remote_file = Path(cloud.config_file)

            # This is the exact wrapper/path used by CloudTrainer for its first config upload.
            cloud._upload_config_file(local_source)
            cloud.file_sync.sync_down_file(local_result, remote_file)
            if local_result.read_text(encoding="utf-8") != expected:
                raise RuntimeError("Cloud config SCP upload/download content did not match")

            fallback_source = Path(temp_dir) / "fallback.txt"
            fallback_result = Path(temp_dir) / "fallback-result.txt"
            fallback_expected = f"fabric-fallback-{uuid.uuid4().hex}"
            fallback_source.write_text(fallback_expected, encoding="utf-8")
            fallback_remote = Path(f"/tmp/{uuid.uuid4().hex}-fallback.txt")
            original_program = cloud.file_sync.base_args[0]
            cloud.file_sync.base_args[0] = "onetrainer-intentionally-missing-scp"
            try:
                cloud.file_sync.upload_file(fallback_source, fallback_remote)
                cloud.file_sync.download_file(fallback_result, fallback_remote)
            finally:
                cloud.file_sync.base_args[0] = original_program
            if fallback_result.read_text(encoding="utf-8") != fallback_expected:
                raise RuntimeError("Live Fabric SFTP fallback content did not match")

            cloud.connection.run(
                f"rm -f {shlex.quote(remote_file.as_posix())} "
                f"{shlex.quote(fallback_remote.as_posix())}",
                hide=True,
                in_stream=False,
            )
        print("exact cloud-config SCP round-trip and live SFTP fallback passed")

        worker = f"sleep 8; printf detached-ok > {shlex.quote(detached_marker)}"
        cloud.connection.run(
            f"nohup bash -c {shlex.quote(worker)} >/dev/null 2>&1 < /dev/null &",
            disown=True,
        )
        cloud.close()
        time.sleep(12)

        # Rebuild the provider object to simulate OneTrainer being closed and reopened.
        cloud = VastCloud(config)
        cloud.setup(install=False)
        result = cloud.connection.run(
            f"cat {shlex.quote(detached_marker)}",
            hide=True,
            in_stream=False,
        )
        if result.stdout != "detached-ok":
            raise RuntimeError("Detached process did not survive the SSH disconnect")
        cloud.connection.run(
            f"rm -f {shlex.quote(detached_marker)}",
            hide=True,
            in_stream=False,
        )
        print("disconnect/reconnect and detached-process survival passed")
        print(f"Vast.ai end-to-end smoke test passed on instance {instance_id}")
    finally:
        try:
            cloud.close()
        finally:
            instance_id = instance_id or str(config.secrets.cloud.id).strip()
            if instance_id:
                print(f"destroying smoke-test Vast.ai instance {instance_id}...")
                cloud.api.destroy_instance(instance_id)
                _wait_until_destroyed(cloud, instance_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-hourly-price", type=float, default=0.30)
    parser.add_argument("--gpu-type", default="")
    args = parser.parse_args()
    run_smoke_test(args.max_hourly_price, args.gpu_type)


if __name__ == "__main__":
    main()
