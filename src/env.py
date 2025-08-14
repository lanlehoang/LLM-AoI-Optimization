from gym import Env
from gym.spaces import Box, Discrete
import numpy as np
from src.utils.generators import *
from src.utils.geometry import *
from src.utils.get_config import get_system_config
import heapq    # Used for event priority queue
from src.system_classes import Event, Satellite, Package, EventType

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)


class SatelliteEnv(Env):
    """
    Custom Gym environment for satellite operations.
    """
    def __init__(self):
        super().__init__()
        self.system_config = get_system_config()
        self.n_satellites = self.system_config['n_satellites']
        self.radius = self.system_config['physics']['r_earth'] + \
            self.system_config['satellite']['height']  # Radius of the satellites' orbit

        # Initialize satellite positions
        satellite_positions = generate_satellite_positions(self.n_satellites, self.radius)
        self.satellite_positions = satellite_positions  # Keep persistent for neighbour calculations

        # Choose start and end satellites
        start, end = choose_start_end(satellite_positions)
        self.start = start
        self.end = end

        # Initialize satellite processing rates
        mu_lower = self.system_config['satellite']['processing_rate']['lower']
        mu_upper = self.system_config['satellite']['processing_rate']['upper']
        satellite_processing_rates = generate_satellite_processing_rates(
            self.n_satellites,
            lower=mu_lower,
            upper=mu_upper
        )
        # Scale processing rates for start satellite to intentionally create congestion in the network,
        # which forces the agent to learn dynamic routing strategies
        satellite_processing_rates[start] *= self.system_config['satellite']['mu']['start_scale']

        # Create satellite objects
        self.satellites = [
            Satellite(
                position=satellite_positions[i],
                processing_rate=satellite_processing_rates[i]
            ) for i in range(self.n_satellites)
        ]

        # Generate all packages at the start satellite
        mu = self.satellites[self.start].processing_rate
        simulation_time = self.system_config['simulation_time']
        packages = generate_all_packages(mu, simulation_time)

        self.packages = [Package(pkg_id, gen_time) for pkg_id, gen_time in packages]
        self.event_queue = [
            Event(pkg.package_id, pkg.generation_time, EventType.PROCESS, self.start)
            for pkg in self.packages
        ]
        self.satellites[self.start].set_queue_length(len(self.packages))  # Set initial queue length for start satellite

        # State and action spaces
        # self.action_space = None  # To be decided
        self.state = None # To be decided

    def reset(self, *, seed=None, options=None):
        """
        Empty satellite queues and reset packages.
        Args:
            seed (int, optional): Random seed.
            options (dict, optional): Additional options.
        Returns:
            Tuple[state, info]: The initial state and info dictionary.
        """
        # Empty satellite queues
        for satellite in self.satellites:
            satellite.reset_queue()

        # Generate all packages at the start satellite
        mu = self.satellites[self.start].processing_rate
        simulation_time = self.system_config['simulation_time']
        packages = generate_all_packages(mu, simulation_time)

        self.packages = [Package(pkg_id, gen_time) for pkg_id, gen_time in packages]
        self.event_queue = [
            Event(pkg.package_id, pkg.generation_time, EventType.PROCESS, self.start)
            for pkg in self.packages
        ]
        self.satellites[self.start].set_queue_length(len(self.packages))  # Set initial queue length for start satellite

        # Initialize state (customize as needed)
        self.state = None  # Replace with actual state initialization

        info = {}
        return self.state, info

    def step(self, action):
        """
        After each PROCESS event, take an action of transferring a package to another satellite.
        Note that the state right after the action IS NOT s'.
        By definition, s' should be the state at which we are ready to take the next action.
        Thus, we need to wait for the next PROCESS event of the SAME package to update the state.
        """
        return None, 0.0, False, False, {}  # Replace with actual step logic