import time
from typing import Any

from modules.cloud.LinuxCloud import LinuxCloud
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.CloudAction import CloudAction

import requests


class VastApiError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class VastApi:
    BASE_URL = "https://console.vast.ai/api/v0"

    def __init__(self, api_key: str, session: requests.Session | None = None):
        self.session = session or requests.Session()
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _request(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        try:
            response = self.session.request(
                method,
                f"{self.BASE_URL}/{path.lstrip('/')}",
                headers=self.headers,
                timeout=30,
                **kwargs,
            )
        except requests.RequestException as exc:
            raise VastApiError("Could not reach the Vast.ai API") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise VastApiError(
                f"Vast.ai returned an invalid response (HTTP {response.status_code})",
                response.status_code,
            ) from exc

        if not response.ok or (isinstance(data, dict) and data.get("success") is False):
            message = "Vast.ai API request failed"
            if isinstance(data, dict):
                message = data.get("msg") or data.get("error") or message
            raise VastApiError(f"{message} (HTTP {response.status_code})", response.status_code)

        if not isinstance(data, dict):
            raise VastApiError("Vast.ai returned an unexpected response")
        return data

    def search_offers(self, filters: dict[str, Any]) -> list[dict[str, Any]]:
        data = self._request("POST", "bundles/", json=filters)
        offers = data.get("offers", [])
        if isinstance(offers, dict):
            return [offers]
        if not isinstance(offers, list):
            raise VastApiError("Vast.ai returned an unexpected offers response")
        return offers

    def get_instance(self, instance_id: str) -> dict[str, Any] | None:
        try:
            data = self._request("GET", f"instances/{instance_id}/")
        except VastApiError as exc:
            if exc.status_code == 404:
                return None
            raise

        instance = data.get("instances")
        if isinstance(instance, list):
            return instance[0] if instance else None
        if instance is not None and not isinstance(instance, dict):
            raise VastApiError("Vast.ai returned an unexpected instance response")
        return instance

    def create_instance(self, offer_id: int, payload: dict[str, Any]) -> str:
        data = self._request("PUT", f"asks/{offer_id}/", json=payload)
        instance_id = data.get("new_contract")
        if instance_id is None:
            raise VastApiError("Vast.ai did not return the new instance ID")
        return str(instance_id)

    def set_instance_state(self, instance_id: str, state: str):
        self._request("PUT", f"instances/{instance_id}/", json={"state": state})

    def destroy_instance(self, instance_id: str):
        self._request("DELETE", f"instances/{instance_id}/")


class VastCloud(LinuxCloud):
    DEFAULT_IMAGE = "vastai/pytorch:cuda-12.8.1-auto"
    READY_TIMEOUT_SECONDS = 30 * 60
    TERMINAL_STATUS_GRACE_SECONDS = 60
    POLL_INTERVAL_SECONDS = 10
    SSH_ATTEMPTS = 12

    def __init__(self, config: TrainConfig):
        super().__init__(config)
        self.api = VastApi(config.secrets.cloud.vast_api_key)

    @staticmethod
    def _offer_filters(
            gpu_type: str = "",
            instance_type: str = "ondemand",
            volume_size: int = 10,
            min_download: int = 0,
            limit: int = 100,
    ) -> dict[str, Any]:
        filters: dict[str, Any] = {
            "limit": limit,
            "type": instance_type or "ondemand",
            "verified": {"eq": True},
            "rentable": {"eq": True},
            "rented": {"eq": False},
            "num_gpus": {"eq": 1},
            "disk_space": {"gte": volume_size},
            "allocated_storage": volume_size,
            "order": [["dph_total", "asc"]],
        }
        if gpu_type:
            filters["gpu_name"] = {"eq": gpu_type}
        if min_download > 0:
            filters["inet_down"] = {"gte": min_download}
        return filters

    @staticmethod
    def get_gpu_types(
            api_key: str,
            instance_type: str = "ondemand",
            volume_size: int = 10,
            min_download: int = 0,
    ) -> list[str]:
        if not api_key.strip():
            raise ValueError("A Vast.ai API key is required to list GPU types")
        api = VastApi(api_key)
        offers = api.search_offers(VastCloud._offer_filters(
            instance_type=instance_type,
            volume_size=volume_size,
            min_download=min_download,
            limit=100,
        ))
        return sorted({offer["gpu_name"] for offer in offers if offer.get("gpu_name")})

    @staticmethod
    def get_gpu_availability(
            api_key: str,
            instance_type: str = "ondemand",
            volume_size: int = 10,
            min_download: int = 0,
    ) -> list[dict[str, Any]]:
        """Return matching Vast.ai GPU types, offer counts, and lowest hourly prices."""
        if not api_key.strip():
            raise ValueError("A Vast.ai API key is required to list GPU availability")

        offers = VastApi(api_key).search_offers(VastCloud._offer_filters(
            instance_type=instance_type,
            volume_size=volume_size,
            min_download=min_download,
            limit=100,
        ))
        by_gpu: dict[str, dict[str, Any]] = {}
        for offer in offers:
            gpu_name = offer.get("gpu_name")
            if not gpu_name:
                continue

            info = by_gpu.setdefault(gpu_name, {
                "id": gpu_name,
                "available": 0,
                "price": None,
            })
            info["available"] += 1
            try:
                price = float(offer["dph_total"])
            except (KeyError, TypeError, ValueError):
                continue
            if info["price"] is None or price < info["price"]:
                info["price"] = price

        return [by_gpu[gpu_name] for gpu_name in sorted(by_gpu)]

    def _connect(self):
        config = self.config.cloud
        secrets = self.config.secrets.cloud
        create_via_api = config.create and secrets.host == "" and str(secrets.port).strip() in {"", "0"}
        if (secrets.id != "" or create_via_api) and not secrets.vast_api_key.strip():
            raise ValueError("A Vast.ai API key is required to manage an instance")

        instance = None
        if secrets.id != "":
            instance = self.api.get_instance(secrets.id)
            if instance is None:
                raise ValueError(f"Vast.ai instance {secrets.id} does not exist")
        elif create_via_api:
            self._create()
            instance = self.api.get_instance(secrets.id)
            if instance is None:
                raise ValueError("Could not create Vast.ai instance")

        if instance is not None:
            status = str(instance.get("actual_status") or "").lower()
            if status in {"exited", "stopped"}:
                self._start()
            self._wait_for_host_port()

        self._connect_ssh()

    def _connect_ssh(self):
        last_error = None
        for attempt in range(self.SSH_ATTEMPTS):
            try:
                super()._connect()
                return
            except Exception as exc:  # noqa: PERF203 - retries are intentional while SSH starts
                last_error = exc
                if attempt + 1 < self.SSH_ATTEMPTS:
                    print("waiting for Vast.ai SSH service...")
                    time.sleep(5)
        assert last_error is not None
        raise last_error

    def _wait_for_host_port(self):
        secrets = self.config.secrets.cloud
        deadline = time.monotonic() + self.READY_TIMEOUT_SECONDS
        terminal_status_since = None

        while time.monotonic() < deadline:
            now = time.monotonic()
            instance = self.api.get_instance(secrets.id)
            if instance is None:
                raise ValueError(f"Vast.ai instance {secrets.id} does not exist")

            status = str(instance.get("actual_status") or "provisioning").lower()
            host = instance.get("ssh_host")
            port = instance.get("ssh_port")
            if status == "running" and host and port:
                secrets.host = str(host)
                secrets.port = str(port)
                return

            if status in {"exited", "unknown", "offline"}:
                terminal_status_since = terminal_status_since or now
                if now - terminal_status_since >= self.TERMINAL_STATUS_GRACE_SECONDS:
                    raise ValueError(
                        f"Vast.ai instance {secrets.id} entered terminal status {status}"
                    )
            else:
                terminal_status_since = None

            print(
                f"waiting for Vast.ai instance... Status: {status}. "
                "https://cloud.vast.ai/instances/"
            )
            time.sleep(self.POLL_INTERVAL_SECONDS)

        raise TimeoutError(f"Timed out waiting for Vast.ai instance {secrets.id}")

    def _create(self):
        config = self.config.cloud
        if not config.gpu_type.strip():
            raise ValueError("Select a Vast.ai GPU type before creating an instance")
        offers = self.api.search_offers(self._offer_filters(
            gpu_type=config.gpu_type,
            instance_type=config.vast_instance_type,
            volume_size=config.volume_size,
            min_download=config.min_download,
            limit=25,
        ))
        if not offers:
            raise ValueError("No matching Vast.ai offers are currently available")

        def offer_price(offer: dict[str, Any]) -> float:
            try:
                return float(offer.get("dph_total", float("inf")))
            except (TypeError, ValueError):
                return float("inf")

        offers.sort(key=offer_price)
        payload = {
            "image": self.DEFAULT_IMAGE,
            "label": config.name,
            "disk": config.volume_size,
            "runtype": "ssh",
            "cancel_unavail": True,
        }

        last_error = None
        for offer in offers:
            offer_id = offer.get("id") or offer.get("ask_contract_id")
            if offer_id is None:
                continue
            create_payload = payload.copy()
            if config.vast_instance_type == "bid":
                bid_price = offer.get("min_bid", offer.get("dph_total"))
                if bid_price is None:
                    continue
                create_payload["price"] = float(bid_price)
            try:
                self.config.secrets.cloud.id = self.api.create_instance(int(offer_id), create_payload)
                return
            except VastApiError as exc:
                last_error = exc
                if exc.status_code not in {404, 410}:
                    raise

        if last_error is not None:
            raise last_error
        raise ValueError("Vast.ai did not return a usable offer")

    def delete(self):
        self.api.destroy_instance(self.config.secrets.cloud.id)

    def stop(self):
        self.api.set_instance_state(self.config.secrets.cloud.id, "stopped")

    def _start(self):
        self.api.set_instance_state(self.config.secrets.cloud.id, "running")

    def _get_action_cmd(self, action: CloudAction):
        if action == CloudAction.STOP:
            return (
                'source /etc/environment && vastai stop instance "$CONTAINER_ID" '
                '--api-key "$CONTAINER_API_KEY"'
            )
        if action == CloudAction.DELETE:
            return (
                'source /etc/environment && vastai destroy instance "$CONTAINER_ID" '
                '--api-key "$CONTAINER_API_KEY"'
            )
        return ":"
