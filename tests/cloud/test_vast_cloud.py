import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from modules.cloud.LinuxCloud import LinuxCloud
from modules.cloud.VastCloud import VastApi, VastCloud
from modules.util.config.CloudConfig import CloudSecretsConfig
from modules.util.enum.CloudAction import CloudAction

import paramiko


class FakeResponse:
    def __init__(self, status_code: int, data: dict | None, text: str = ""):
        self.status_code = status_code
        self._data = data
        self.text = text
        self.ok = 200 <= status_code < 400

    def json(self):
        if self._data is None:
            raise ValueError("not JSON")
        return self._data


class FakeSession:
    def __init__(self, *responses: FakeResponse):
        self.responses = list(responses)
        self.requests = []

    def request(self, method, url, **kwargs):
        self.requests.append((method, url, kwargs))
        return self.responses.pop(0)


class VastApiTest(unittest.TestCase):
    def test_search_offers_uses_bearer_auth(self):
        session = FakeSession(FakeResponse(200, {"offers": [{"id": 1}]}))
        api = VastApi("test-key", session=session)

        offers = api.search_offers({"limit": 1})

        self.assertEqual([{"id": 1}], offers)
        method, url, kwargs = session.requests[0]
        self.assertEqual("POST", method)
        self.assertEqual("https://console.vast.ai/api/v0/bundles/", url)
        self.assertEqual("Bearer test-key", kwargs["headers"]["Authorization"])
        self.assertEqual({"limit": 1}, kwargs["json"])

    def test_missing_instance_returns_none(self):
        session = FakeSession(FakeResponse(404, {"success": False, "msg": "not found"}))
        api = VastApi("test-key", session=session)

        self.assertIsNone(api.get_instance("123"))

    def test_rate_limit_with_plain_text_body_is_retried(self):
        session = FakeSession(
            FakeResponse(429, None, "API requests too frequent"),
            FakeResponse(200, {"instances": {"id": 123, "actual_status": "loading"}}),
        )
        api = VastApi("test-key", session=session)

        with patch("modules.cloud.VastCloud.time.sleep") as sleep:
            instance = api.get_instance("123")

        self.assertEqual(123, instance["id"])
        self.assertEqual(2, len(session.requests))
        sleep.assert_called_once_with(1.0)

    def test_instance_ssh_key_endpoints(self):
        session = FakeSession(
            FakeResponse(200, {
                "success": True,
                "ssh_keys": '[{"id": 1, "public_key": "ssh-rsa AAAA existing"}]',
            }),
            FakeResponse(200, {"success": True}),
        )
        api = VastApi("test-key", session=session)

        keys = api.get_instance_ssh_keys("123")
        api.attach_ssh_key("123", "ssh-rsa BBBB new")

        self.assertEqual("ssh-rsa AAAA existing", keys[0]["public_key"])
        self.assertEqual("GET", session.requests[0][0])
        self.assertEqual(
            "https://console.vast.ai/api/v0/instances/123/ssh/",
            session.requests[0][1],
        )
        self.assertEqual("POST", session.requests[1][0])
        self.assertEqual({"ssh_key": "ssh-rsa BBBB new"}, session.requests[1][2]["json"])


class VastCloudTest(unittest.TestCase):
    def test_legacy_cloud_key_migrates_to_runpod_key(self):
        secrets = CloudSecretsConfig.default_values().from_dict({
            "__version": 0,
            "api_key": "old-runpod-key",
        })

        self.assertEqual("old-runpod-key", secrets.runpod_api_key)
        self.assertEqual("", secrets.vast_api_key)
        self.assertNotIn("api_key", secrets.to_dict())

    def test_create_selects_cheapest_matching_offer(self):
        class FakeApi:
            def __init__(self):
                self.created_offer_id = None
                self.filters = None
                self.payload = None

            def search_offers(self, filters):
                self.filters = filters
                return [
                    {"id": 10, "dph_total": 1.25},
                    {"id": 20, "dph_total": 0.75},
                ]

            def create_instance(self, offer_id, payload):
                self.created_offer_id = offer_id
                self.payload = payload
                return "456"

        cloud = VastCloud.__new__(VastCloud)
        cloud.api = FakeApi()
        cloud.config = SimpleNamespace(
            cloud=SimpleNamespace(
                gpu_type="RTX 4090",
                vast_instance_type="ondemand",
                volume_size=100,
                min_download=500,
                gpu_count=2,
                name="OneTrainer",
            ),
            secrets=SimpleNamespace(cloud=SimpleNamespace(id="")),
        )

        cloud._create()

        self.assertEqual(20, cloud.api.created_offer_id)
        self.assertEqual("456", cloud.config.secrets.cloud.id)
        self.assertEqual("ssh_direct", cloud.api.payload["runtype"])
        self.assertEqual({"eq": 2}, cloud.api.filters["num_gpus"])
        self.assertEqual({"gte": 1}, cloud.api.filters["direct_port_count"])

    def test_gpu_availability_aggregates_offer_count_and_lowest_price(self):
        offers = [
            {"gpu_name": "RTX 4090", "dph_total": 0.55},
            {"gpu_name": "RTX 4090", "dph_total": 0.42},
            {"gpu_name": "A100 SXM4", "dph_total": 1.25},
        ]

        with patch.object(VastApi, "search_offers", return_value=offers):
            availability = VastCloud.get_gpu_availability("test-key")

        self.assertEqual([
            {"id": "A100 SXM4", "available": 1, "price": 1.25},
            {"id": "RTX 4090", "available": 2, "price": 0.42},
        ], availability)

    def test_bid_instance_uses_offer_minimum_bid(self):
        class FakeApi:
            def __init__(self):
                self.payload = None

            def search_offers(self, _filters):
                return [{"id": 10, "dph_total": 0.5, "min_bid": 0.4}]

            def create_instance(self, _offer_id, payload):
                self.payload = payload
                return "456"

        cloud = VastCloud.__new__(VastCloud)
        cloud.api = FakeApi()
        cloud.config = SimpleNamespace(
            cloud=SimpleNamespace(
                gpu_type="RTX 4090",
                vast_instance_type="bid",
                volume_size=100,
                min_download=500,
                gpu_count=1,
                name="OneTrainer",
            ),
            secrets=SimpleNamespace(cloud=SimpleNamespace(id="")),
        )

        cloud._create()

        self.assertEqual(0.4, cloud.api.payload["price"])

    def test_create_switch_provisions_before_ssh_even_with_stale_host_port(self):
        cloud = VastCloud.__new__(VastCloud)
        secrets = SimpleNamespace(
            id="",
            host="old-runpod-host",
            port="22000",
            user="root",
            vast_api_key="test-key",
        )
        cloud.config = SimpleNamespace(
            cloud=SimpleNamespace(create=True),
            secrets=SimpleNamespace(cloud=secrets),
        )
        cloud.api = SimpleNamespace(get_instance=Mock())

        def create():
            secrets.id = "123"

        with (
            patch.object(cloud, "_create", side_effect=create) as create_mock,
            patch.object(cloud, "_wait_for_host_port") as wait_mock,
            patch.object(cloud, "_ensure_instance_ssh_key") as key_mock,
            patch.object(cloud, "_connect_ssh") as ssh_mock,
        ):
            cloud._connect()

        create_mock.assert_called_once_with()
        wait_mock.assert_called_once_with(
            initial_instance=None,
            tolerate_initial_missing=True,
        )
        key_mock.assert_called_once_with()
        ssh_mock.assert_called_once_with()
        self.assertEqual("", secrets.host)
        self.assertEqual(0, secrets.port)
        cloud.api.get_instance.assert_not_called()

    def test_create_ignores_a_stale_runpod_id(self):
        cloud = VastCloud.__new__(VastCloud)
        secrets = SimpleNamespace(
            id="old-runpod-id",
            host="",
            port=0,
            user="root",
            vast_api_key="test-key",
        )
        cloud.config = SimpleNamespace(
            cloud=SimpleNamespace(create=True),
            secrets=SimpleNamespace(cloud=secrets),
        )
        cloud.api = SimpleNamespace(get_instance=Mock(return_value={"actual_status": "running"}))

        def create():
            secrets.id = "123"

        with (
            patch.object(cloud, "_create", side_effect=create) as create_mock,
            patch.object(cloud, "_wait_for_host_port"),
            patch.object(cloud, "_ensure_instance_ssh_key"),
            patch.object(cloud, "_connect_ssh"),
        ):
            cloud._connect()

        create_mock.assert_called_once_with()
        self.assertEqual("123", secrets.id)

    def test_initial_instance_is_reused_without_an_immediate_second_get(self):
        cloud = VastCloud.__new__(VastCloud)
        secrets = SimpleNamespace(id="123", host="", port=0)
        cloud.config = SimpleNamespace(secrets=SimpleNamespace(cloud=secrets))
        cloud.api = SimpleNamespace(get_instance=Mock())

        cloud._wait_for_host_port(initial_instance={
            "actual_status": "running",
            "ssh_host": "ssh.example.test",
            "ssh_port": 12345,
        })

        cloud.api.get_instance.assert_not_called()
        self.assertEqual("ssh.example.test", secrets.host)
        self.assertEqual("12345", secrets.port)

    def test_new_instance_visibility_delay_is_retried(self):
        cloud = VastCloud.__new__(VastCloud)
        secrets = SimpleNamespace(id="123", host="", port=0)
        cloud.config = SimpleNamespace(secrets=SimpleNamespace(cloud=secrets))
        cloud.api = SimpleNamespace(get_instance=Mock(side_effect=[
            None,
            {
                "actual_status": "running",
                "ssh_host": "ssh.example.test",
                "ssh_port": 12345,
            },
        ]))

        with patch("modules.cloud.VastCloud.time.sleep") as sleep:
            cloud._wait_for_host_port(tolerate_initial_missing=True)

        self.assertEqual(2, cloud.api.get_instance.call_count)
        sleep.assert_called_once_with(cloud.POLL_INTERVAL_SECONDS)
        self.assertEqual("ssh.example.test", secrets.host)

    def test_missing_manual_host_port_fails_without_ssh_retries(self):
        cloud = VastCloud.__new__(VastCloud)
        cloud.config = SimpleNamespace(
            secrets=SimpleNamespace(cloud=SimpleNamespace(
                id="",
                host="",
                port=0,
                user="root",
            )),
        )

        with (
            patch.object(LinuxCloud, "_connect") as connect,
            self.assertRaisesRegex(ValueError, "Create cloud via API"),
        ):
            cloud._connect_ssh()

        connect.assert_not_called()

    def test_ssh_authentication_error_is_reported_without_retries(self):
        cloud = VastCloud.__new__(VastCloud)
        cloud.config = SimpleNamespace(
            secrets=SimpleNamespace(cloud=SimpleNamespace(
                id="123",
                host="ssh.example.test",
                port=22,
                user="root",
            )),
        )

        with (
            patch.object(
                LinuxCloud,
                "_connect",
                side_effect=paramiko.AuthenticationException("denied"),
            ) as connect,
            self.assertRaisesRegex(ConnectionError, "authentication failed"),
        ):
            cloud._connect_ssh()

        connect.assert_called_once_with()

    def test_configured_public_key_is_attached_to_instance(self):
        with TemporaryDirectory() as temp_dir:
            private_key_path = Path(temp_dir) / "vast-key"
            private_key_path.write_text("not-read-when-public-key-exists", encoding="utf-8")
            Path(f"{private_key_path}.pub").write_text(
                "ssh-rsa AAAATEST onetrainer-test\n",
                encoding="utf-8",
            )
            secrets = SimpleNamespace(
                id="123",
                expanded_key_file=lambda: str(private_key_path),
            )
            api = SimpleNamespace(
                get_instance_ssh_keys=Mock(return_value=[]),
                attach_ssh_key=Mock(),
            )
            cloud = VastCloud.__new__(VastCloud)
            cloud.config = SimpleNamespace(secrets=SimpleNamespace(cloud=secrets))
            cloud.api = api

            cloud._ensure_instance_ssh_key()

        api.attach_ssh_key.assert_called_once_with(
            "123",
            "ssh-rsa AAAATEST onetrainer-test",
        )

    def test_detached_actions_use_vast_instance_credentials(self):
        cloud = VastCloud.__new__(VastCloud)

        stop = cloud._get_action_cmd(CloudAction.STOP)
        delete = cloud._get_action_cmd(CloudAction.DELETE)

        self.assertIn("vastai stop instance", stop)
        self.assertIn("vastai destroy instance", delete)
        self.assertIn("$CONTAINER_ID", stop)
        self.assertIn("$CONTAINER_API_KEY", stop)


if __name__ == "__main__":
    unittest.main()
