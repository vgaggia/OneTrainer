from enum import Enum


class CloudType(Enum):
    RUNPOD = 'RUNPOD'
    VAST = 'VAST'
    LINUX = 'LINUX'

    def __str__(self):
        return self.value
