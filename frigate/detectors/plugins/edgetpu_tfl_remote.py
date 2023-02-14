import logging
import numpy as np

from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import BaseDetectorConfig
from typing import Literal
from pydantic import Extra, Field
import tflite_runtime.interpreter as tflite
from tflite_runtime.interpreter import load_delegate
import requests
import msgpack
import msgpack_numpy as m
m.patch()


logger = logging.getLogger(__name__)

DETECTOR_KEY = "edgetpu_remote"


class EdgeTpuRemoteDetectorConfig(BaseDetectorConfig):
    type: Literal[DETECTOR_KEY]
    device: str = Field(default=None, title="Device Type")
    ip: str = Field(default=None, title="Ip Address")
    port: str  = Field(default=None, title="Port")


class EdgeTpuTflRemote(DetectionApi):
    type_key = DETECTOR_KEY

    def __init__(self, detector_config: EdgeTpuRemoteDetectorConfig):
        self.uri = "http://"+detector_config.ip+":"+detector_config.port+"/"


    def detect_raw(self, tensor_input):
        r = requests.post(self.uri,data=msgpack.packb(tensor_input))
        detections = msgpack.unpackb(r.content)
        return detections
