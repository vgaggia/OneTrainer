import unittest
from types import SimpleNamespace
from unittest.mock import patch

from modules.cloud.VastCloud import VastApi, VastCloud
from modules.util.config.CloudConfig import CloudSecretsConfig
from modules.util.enum.CloudAction import CloudAction


class FakeResponse:
    def __init__(self, status_code: int, data: dict):
        self.status_code = status_code
        self._data = data
        self.ok = 200 <= status_code < 400

    def json(self):
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

            def search_offers(self, _filters):
                return [
                    {"id": 10, "dph_total": 1.25},
                    {"id": 20, "dph_total": 0.75},
                ]

            def create_instance(self, offer_id, _payload):
                self.created_offer_id = offer_id
                return "456"

        cloud = VastCloud.__new__(VastCloud)
        cloud.api = FakeApi()
        cloud.config = SimpleNamespace(
            cloud=SimpleNamespace(
                gpu_type="RTX 4090",
                vast_instance_type="ondemand",
                volume_size=100,
                min_download=500,
                name="OneTrainer",
            ),
            secrets=SimpleNamespace(cloud=SimpleNamespace(id="")),
        )

        cloud._create()

        self.assertEqual(20, cloud.api.created_offer_id)
        self.assertEqual("456", cloud.config.secrets.cloud.id)

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
                name="OneTrainer",
            ),
            secrets=SimpleNamespace(cloud=SimpleNamespace(id="")),
        )

        cloud._create()

        self.assertEqual(0.4, cloud.api.payload["price"])

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
