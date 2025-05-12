import yaml
import glob
import os
import sys
from pathlib import Path

# make sure you import carla python api
try:
    sys.path.append(glob.glob('%s/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        "C:/CARLA_0.9.15/WindowsNoEditor" if os.name == 'nt' else str(Path.home()) + "/CARLA_0.9.15",
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])

    sys.path.append(glob.glob('../../')[0])

except IndexError:
    pass


import carla

def load_sensor_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

from carla import VehicleLightState as vls
from datasets import ply
import logging
import queue
import struct
import math
import numpy as np
import random
import threading
