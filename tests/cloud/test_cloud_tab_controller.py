import unittest
from types import SimpleNamespace
from unittest.mock import patch

from modules.ui.CloudTabController import CloudTabController
from modules.util.enum.CloudType import CloudType


class CloudTabControllerTest(unittest.TestCase):
    @staticmethod
    def _config(cloud_type: CloudType):
        return SimpleNamespace(
            cloud=SimpleNamespace(
                type=cloud_type,
                gpu_count=2,
                sub_type="SECURE",
                vast_instance_type="ondemand",
                volume_size=100,
                min_download=500,
            ),
            secrets=SimpleNamespace(
                cloud=SimpleNamespace(
                    runpod_api_key="runpod-test-key",
                    vast_api_key="vast-test-key",
                ),
            ),
        )

    def test_runpod_availability_uses_shape_filters_and_runpod_key(self):
        response = {
            "data": {
                "gpuTypes": [{
                    "id": "NVIDIA GeForce RTX 4090",
                    "lowestPrice": {
                        "stockStatus": "High",
                        "uninterruptablePrice": 0.69,
                    },
                }],
            },
        }
        config = self._config(CloudType.RUNPOD)

        with patch("runpod.api.graphql.run_graphql_query", return_value=response) as query:
            availability = CloudTabController(config, None).get_gpu_availability()

        self.assertEqual([{
            "id": "NVIDIA GeForce RTX 4090",
            "stock_status": "High",
            "price": 0.69,
        }], availability)
        self.assertIn("gpuCount: 2", query.call_args.args[0])
        self.assertIn("secureCloud: true", query.call_args.args[0])

        import runpod
        self.assertEqual("runpod-test-key", runpod.api_key)

    def test_vast_availability_uses_vast_key_and_offer_filters(self):
        config = self._config(CloudType.VAST)
        expected = [{"id": "RTX 4090", "available": 3, "price": 0.42}]

        with patch(
            "modules.cloud.VastCloud.VastCloud.get_gpu_availability",
            return_value=expected,
        ) as availability:
            result = CloudTabController(config, None).get_gpu_availability()

        self.assertEqual(expected, result)
        availability.assert_called_once_with(
            api_key="vast-test-key",
            instance_type="ondemand",
            volume_size=100,
            min_download=500,
            gpu_count=2,
        )


if __name__ == "__main__":
    unittest.main()
