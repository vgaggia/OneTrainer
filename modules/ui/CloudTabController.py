
import traceback
import webbrowser

from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.CloudType import CloudType


class CloudTabController:
    def __init__(self, config: TrainConfig, parent):
        self.config = config
        self.parent = parent
        self.reattach = False

    def do_reattach(self):
        self.reattach = True
        try:
            self.parent.start_training()
        finally:
            self.reattach = False

    def get_gpu_types(self) -> list[str]:
        if self.config.cloud.type == CloudType.RUNPOD:
            import runpod
            runpod.api_key = self.config.secrets.cloud.runpod_api_key
            gpus = runpod.get_gpus()
            return [gpu['id'] for gpu in gpus]
        if self.config.cloud.type == CloudType.VAST:
            from modules.cloud.VastCloud import VastCloud

            return VastCloud.get_gpu_types(
                api_key=self.config.secrets.cloud.vast_api_key,
                instance_type=self.config.cloud.vast_instance_type,
                volume_size=self.config.cloud.volume_size,
                min_download=self.config.cloud.min_download,
            )
        return []

    def get_gpu_availability(self) -> list[dict]:
        """Return GPU ids with current availability and hourly pricing."""
        if self.config.cloud.type == CloudType.RUNPOD:
            import runpod
            from runpod.api.graphql import run_graphql_query

            runpod.api_key = self.config.secrets.cloud.runpod_api_key
            try:
                gpu_count = max(1, int(self.config.cloud.gpu_count or 1))
                input_parts = [f"gpuCount: {gpu_count}"]
                if self.config.cloud.sub_type == "SECURE":
                    input_parts.append("secureCloud: true")
                elif self.config.cloud.sub_type == "COMMUNITY":
                    input_parts.append("secureCloud: false")
                query = (
                    "query { gpuTypes { id lowestPrice(input: {"
                    + ", ".join(input_parts)
                    + "}) { stockStatus uninterruptablePrice } } }"
                )
                response = run_graphql_query(query)
                result = []
                for gpu_type in response.get("data", {}).get("gpuTypes", []):
                    lowest_price = gpu_type.get("lowestPrice") or {}
                    result.append({
                        "id": gpu_type["id"],
                        "stock_status": lowest_price.get("stockStatus"),
                        "price": lowest_price.get("uninterruptablePrice"),
                    })
                return result
            except Exception:
                traceback.print_exc()
                return [{"id": gpu_id} for gpu_id in self.get_gpu_types()]

        if self.config.cloud.type == CloudType.VAST:
            from modules.cloud.VastCloud import VastCloud

            try:
                return VastCloud.get_gpu_availability(
                    api_key=self.config.secrets.cloud.vast_api_key,
                    instance_type=self.config.cloud.vast_instance_type,
                    volume_size=self.config.cloud.volume_size,
                    min_download=self.config.cloud.min_download,
                )
            except Exception:
                traceback.print_exc()
                return [{"id": gpu_id} for gpu_id in self.get_gpu_types()]

        return []

    def open_create_cloud_url(self):
        if self.config.cloud.type == CloudType.RUNPOD:
            webbrowser.open("https://www.runpod.io/console/deploy?template=1a33vbssq9&type=gpu", new=0, autoraise=False)
        elif self.config.cloud.type == CloudType.VAST:
            webbrowser.open("https://cloud.vast.ai/create/", new=0, autoraise=False)
