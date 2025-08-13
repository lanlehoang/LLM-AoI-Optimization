from gym import Env
from gym.spaces import Box, Discrete
import numpy as np
from src.utils.generators import *
from src.utils.get_config import get_system_config
import heapq    # Used for event priority queue
from enum import Enum


class EventType(Enum):
    """
    Enum for different types of events in the satellite environment.
    Types of events:
    - PROCESS: A package is processed by satellite i at time t
    - TRANSFER: A package is transferred from satellite i to j at time t
    """
    PROCESS = "process"
    TRANSFER = "transfer"    


class SatelliteEnv(Env):
    """
    Custom Gym environment for satellite operations.
    """
    def __init__(self):
        super().__init__()
        self.system_config = get_system_config()

        # Initialize satellites
        np.random.seed(RANDOM_SEED) # Set random seed each time the environment is created
        self.satellite_pos = generate_satallite_positions(self.system_config['n_satellites'])
        self.satellite_mu = generate_satellite_processing_rates(self.system_config['n_satellites'])
        start, end = choose_start_end(self.satellite_pos)
        self.start_satellite = start
        self.end_satellite = end

        # Time and event management
        self.time = 0
        self.event_queue = []

        # Action space
        self.action_space = None  # To be decided

    def reset(self):
        """
        Reset the environment to its initial state.
        """
        # Initialize satellites
        np.random.seed(RANDOM_SEED) # Set random seed each time the environment is created
        self.satellite_pos = generate_satallite_positions(self.system_config['n_satellites'])
        self.satellite_mu = generate_satellite_processing_rates(self.system_config['n_satellites'])
        start, end = choose_start_end(self.satellite_pos)
        self.start_satellite = start
        self.end_satellite = end

        # Time and event
        self.time = 0
        self.event_queue = []

    def step(self, action):
        pass