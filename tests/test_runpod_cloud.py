from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from modules.cloud.RunpodCloud import RunpodCloud

import requests


def make_cloud() -> RunpodCloud:
    cloud = object.__new__(RunpodCloud)
    cloud.config = SimpleNamespace(
        secrets=SimpleNamespace(
            cloud=SimpleNamespace(api_key="test-api-key"),
        ),
    )
    return cloud


class RunpodCloudTest(TestCase):
    def test_public_template_image_is_loaded_from_rest_api(self):
        response = Mock()
        response.json.return_value = {"imageName": "example/onetrainer:latest"}

        with patch("modules.cloud.RunpodCloud.requests.get", return_value=response) as get:
            image_name = make_cloud()._RunpodCloud__get_template_image_name("template-id")

        self.assertEqual(image_name, "example/onetrainer:latest")
        response.raise_for_status.assert_called_once_with()
        get.assert_called_once_with(
            "https://rest.runpod.io/v1/templates/template-id",
            headers={"Authorization": "Bearer test-api-key"},
            params={
                "includePublicTemplates": "true",
                "includeRunpodTemplates": "true",
            },
            timeout=30,
        )

    def test_template_lookup_failure_has_context(self):
        response = Mock()
        response.raise_for_status.side_effect = requests.HTTPError("404 Client Error")

        with (
            patch("modules.cloud.RunpodCloud.requests.get", return_value=response),
            self.assertRaisesRegex(ValueError, "Could not retrieve RunPod template template-id"),
        ):
            make_cloud()._RunpodCloud__get_template_image_name("template-id")

    def test_template_without_an_image_is_rejected(self):
        response = Mock()
        response.json.return_value = {"id": "template-id"}

        with (
            patch("modules.cloud.RunpodCloud.requests.get", return_value=response),
            self.assertRaisesRegex(ValueError, "RunPod template template-id has no container image"),
        ):
            make_cloud()._RunpodCloud__get_template_image_name("template-id")
