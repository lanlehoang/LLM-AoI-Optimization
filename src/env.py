from gym import Env
from gym.spaces import Box, Discrete
import numpy as np
from src.utils.generators import *
from src.utils.get_config import get_system_config


class SatelliteEnv(Env):
    """
    Custom Gym environment for satellite operations.
    """
    def __init__(self):
        super().__init__()
        self.system_config = get_system_config()

    def reset(self):
        pass

    def step(self, action):
        pass