import secrets as pysecrets
import time

from modules.cloud.LinuxCloud import LinuxCloud
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.CloudAction import CloudAction

import requests
import runpod


class RunpodCloud(LinuxCloud):
    def __init__(self, config: TrainConfig):
        super().__init__(config)

        runpod.api_key=config.secrets.cloud.runpod_api_key

    def __get_host_port(self):
        secrets=self.config.secrets.cloud
        resumed=False
        while True:
            if (pod:=runpod.get_pod(secrets.id)) is None and not resumed:
                raise ValueError(f"Runpod {secrets.id} does not exist")
            if pod and pod['desiredStatus'] == "EXITED":
                self._start()
                #In edge cases runpod returns incorrect information for resumed pods:
                #The pod id seems to disappear for a while, and on recently stopped pods the old public IP and port is still being reported
                #Therefore, on resumed pods, iterate until there is a successful connection
                resumed=True
            elif pod and (runtime:=pod['runtime']) is not None and 'ports' in runtime and runtime['ports'] is not None:
                for port in runtime['ports']:
                    # ponytail: runpod also exposes a public UDP port, and the order is not stable.
                    # Without the type check this happily hands SSH the UDP port number.
                    if port['isIpPublic'] and port.get('type') == 'tcp':
                        secrets.host=port['ip']
                        secrets.port=port['publicPort']
                        if resumed:
                            try:
                                super()._connect()
                            except Exception:
                                continue
                        return
            if secrets.id == "":
                print("waiting for public IP...")
            else:
                print(f"waiting for public IP... Status: https://www.runpod.io/console/pods?id={secrets.id}")
            time.sleep(5)


    def _connect(self):
        config=self.config.cloud
        secrets=self.config.secrets.cloud

        pod=None
        if secrets.id != "":
            pod=runpod.get_pod(secrets.id)
            if pod is None:
                raise ValueError(f"Runpod {secrets.id} does not exist")
        elif config.create:
            self._create()
            pod=runpod.get_pod(secrets.id)
            if pod is None:
                raise ValueError("Could not create cloud")

        if pod is not None:
            self.__get_host_port()
        super()._connect()

    __TEMPLATE_ID = "1a33vbssq9"

    def __get_template_image_name(self, template_id: str) -> str:
        # RunPod's create_pod requires a non-empty image_name even when a template_id is given,
        # so look up the public template's current image instead of hardcoding a version tag here.
        # `myself.podTemplates` cannot be used because it only contains templates owned by the
        # current account, while OneTrainer's template is public and owned by another account.
        try:
            response = requests.get(
                f"https://rest.runpod.io/v1/templates/{template_id}",
                headers={"Authorization": f"Bearer {self.config.secrets.cloud.runpod_api_key}"},
                params={
                    "includePublicTemplates": "true",
                    "includeRunpodTemplates": "true",
                },
                timeout=30,
            )
            response.raise_for_status()
            template = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise ValueError(f"Could not retrieve RunPod template {template_id}: {exc}") from exc

        image_name = template.get("imageName") if isinstance(template, dict) else None
        if not image_name:
            raise ValueError(f"RunPod template {template_id} has no container image")
        return image_name

    def _gpu_count(self) -> int:
        return max(1, self.config.cloud.gpu_count)

    def _allowed_cuda_versions(self):
        # ponytail: blank = no filter (current behaviour); e.g. "13.0" pins hosts whose driver supports torch cu130
        vals = [v.strip() for v in self.config.cloud.cuda_version.split(",") if v.strip()]
        return vals or None

    def _create(self):
        config=self.config.cloud
        secrets=self.config.secrets.cloud
        pod=runpod.create_pod(
            name=config.name,
            image_name=self.__get_template_image_name(self.__TEMPLATE_ID),
            template_id=self.__TEMPLATE_ID,
            gpu_type_id=config.gpu_type,
            gpu_count=self._gpu_count(),
            allowed_cuda_versions=self._allowed_cuda_versions(),
            cloud_type=config.sub_type,
            support_public_ip=True,
            # ponytail: a network volume replaces the pod volume at volume_mount_path, so volume_size is ignored then
            network_volume_id=config.network_volume_id or None,
            data_center_id=config.data_center_id or None,
            volume_in_gb=config.volume_size,
            container_disk_in_gb=20,
            volume_mount_path="/workspace",
            min_download=config.min_download,
            env={"JUPYTER_PASSWORD": pysecrets.token_urlsafe(16)},
        )
        secrets.id=pod['id']

    def delete(self):
        runpod.terminate_pod(self.config.secrets.cloud.id)

    def stop(self):
        runpod.stop_pod(self.config.secrets.cloud.id)

    def _start(self):
        runpod.resume_pod(self.config.secrets.cloud.id,gpu_count=self._gpu_count())

    def _get_action_cmd(self,action : CloudAction):
        if action == CloudAction.STOP:
            return "source /etc/rp_environment && runpodctl stop pod $RUNPOD_POD_ID"
        elif action == CloudAction.DELETE:
            return "source /etc/rp_environment && runpodctl remove pod $RUNPOD_POD_ID"
        else:
            return ":"
