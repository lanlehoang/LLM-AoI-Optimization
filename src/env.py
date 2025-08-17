from gym import Env
from gym.spaces import Box, Discrete
import numpy as np
from src.utils.generators import *
from src.utils.geometry import *
from src.utils.get_config import get_system_config
from env_classes import *

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)


class SatelliteEnv(Env):
    """
    Custom Gym environment for satellite operations.
    """

    def __init__(self):
        super().__init__()
        self.system_config = get_system_config()
        self.n_satellites = self.system_config["n_satellites"]
        self.radius = (
            self.system_config["physics"]["r_earth"]
            + self.system_config["satellite"]["height"]
        )  # Radius of the satellites' orbit

        # Initialize satellite positions
        satellite_positions = generate_satellite_positions(
            self.n_satellites, self.radius
        )
        self.satellite_positions = (
            satellite_positions  # Keep persistent for neighbour calculations
        )

        # Choose start and end satellites
        start, end = choose_start_end(satellite_positions)
        self.start = start
        self.end = end

        # Initialize satellite processing rates
        mu_lower = self.system_config["satellite"]["processing_rate"]["lower"]
        mu_upper = self.system_config["satellite"]["processing_rate"]["upper"]
        satellite_processing_rates = generate_satellite_processing_rates(
            self.n_satellites, lower=mu_lower, upper=mu_upper
        )
        # Scale processing rates for start satellite to intentionally create congestion in the network,
        # which forces the agent to learn dynamic routing strategies
        scale = self.system_config["satellite"]["mu"]["start_scale"]
        satellite_processing_rates[start] *= scale

        # Create satellite objects
        self.satellites = [
            Satellite(
                position=satellite_positions[i],
                processing_rate=satellite_processing_rates[i],
            )
            for i in range(self.n_satellites)
        ]

        # Generate all packages at the start satellite
        mu = self.satellites[self.start].processing_rate
        simulation_time = self.system_config["simulation_time"]
        packages = generate_all_packages(mu, simulation_time)

        self.packages = [Package(pkg_id, gen_time) for pkg_id, gen_time in packages]
        self.event_queue = EventQueue(
            [
                Event(
                    pkg.package_id,
                    pkg.generation_time,
                    EventType.PROCESS.value,
                    self.start,
                )
                for pkg in self.packages
            ]
        )
        self.satellites[self.start].set_queue_length(
            len(self.packages)
        )  # Set initial queue length for start satellite

        # Initialize the buffer for storing experiences
        self.buffer = ExperienceBuffer()

        # State and action spaces
        self.action_space = Discrete(self.system_config["n_neighbours"])
        self.state = self.get_state(self.start)

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
        simulation_time = self.system_config["simulation_time"]
        packages = generate_all_packages(mu, simulation_time)

        self.packages = [Package(pkg_id, gen_time) for pkg_id, gen_time in packages]
        self.event_queue = EventQueue(
            [
                Event(
                    pkg.package_id,
                    pkg.generation_time,
                    EventType.PROCESS.value,
                    self.start,
                )
                for pkg in self.packages
            ]
        )
        self.satellites[self.start].set_queue_length(
            len(self.packages)
        )  # Set initial queue length for start satellite

        # Initialize state (customize as needed)
        self.state = self.get_state(self.start)
        return self.state, {}

    def get_state(self, cur_satellite):
        """
        The state at each step includes:
        - The current satellite and the destination positions
        - State of each neighbour, including its position, processing rate, and queue length
        """
        # Current and destination positions
        cur_pos = self.satellite_positions[cur_satellite]
        dst_pos = self.satellite_positions[self.end]

        # Find all neighbours
        neighbour_indices = find_neighbours(
            cur_idx=cur_satellite,
            dst_pos=dst_pos,
            satellite_positions=self.satellite_positions,
        )
        neighbours = [self.satellites[i] for i in neighbour_indices]
        neighbour_states = np.array([])
        for sat in neighbours:
            neighbour_states = np.hstack(
                [neighbour_states, sat.position, sat.processing_rate, sat.queue_length]
            )

        # Concatenate everything and flatten into an 1-d array
        return np.hstack([cur_pos, dst_pos, neighbour_states])

    def handle_transfer_event(self, event: Event):
        """
        Handle the TRANSFER event in the event queue.
        If the package is sent to the destination, record the event time as the package arrival time.
        Else, generate its processing time and put a PROCESS event into the queue
        Finally, update the previous state relating to the package in the buffer
        """
        # Unpack event attributes
        package_id = event.package_id
        event_time = event.event_time
        src = event.src
        dst = event.dst

        # Calculate transmission time
        c = self.system_config["physics"]["c"]
        dist = np.linalg.norm(
            self.satellite_positions[src] - self.satellite_positions[dst]
        )
        arrival_time = event_time + dist / c

        # Handle the transfer logic
        if dst == self.end:
            self.packages[package_id].record_end_time(arrival_time)
        else:
            sat: Satellite = self.satellites[dst]
            start_time = max(
                sat.busy_time, arrival_time
            )  # Start processing at arrival, or when the satellite is free
            processing_time = generate_processing_time(mu=sat.processing_rate)
            sat.enqueue_package(processing_time=processing_time)
            new_event = Event(
                package_id=package_id,
                event_time=start_time + processing_time,
                event_type=EventType.PROCESS.value,
                src=dst,
            )
            self.event_queue.push(new_event)

        # Update the experience buffer
        new_state = self.get_state(cur_satellite=dst)
        self.buffer.update_experience(package_id=package_id, new_state=new_state)
        # --- TO BE CONTINUED ---

    def step(self, action):
        """
        After each PROCESS event, take an action of transferring a package to another satellite.
        Note that the state right after the action IS NOT s'.
        By definition, s' should be the state at which we are ready to take the next action.
        Thus, we need to wait for the next PROCESS event of the SAME package to update the state.
        """
        return None, 0.0, False, False, {}  # Replace with actual step logic
