import numpy as np
from src.utils.generators import *
from src.utils.geometry import *
from src.utils.get_config import get_system_config
from src.env.env_classes import *
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SatelliteEnv:
    """
    Custom Gym environment for satellite operations.
    """

    def __init__(self):
        super().__init__()
        self.system_config = get_system_config()
        self.n_satellites = self.system_config["satellite"]["n_satellites"]
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
        mu_lower = self.system_config["satellite"]["mu"]["lower"]
        mu_upper = self.system_config["satellite"]["mu"]["upper"]
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
        simulation_time = self.system_config["physics"]["simulation_time"]
        # Generate all packages at the start satellite
        self.initial_packages = generate_all_packages(mu, simulation_time)
        logger.info(f"Number of packages generated: {len(self.initial_packages)}")

    def reset(self):
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

        # Reset packages and event queue
        self.packages = [Package(gen_time) for _, gen_time in self.initial_packages]

        self.event_queue = EventQueue(
            [
                Event(pkg_id, generation_time, self.start)
                for pkg_id, generation_time in self.initial_packages
            ]
        )
        self.satellites[self.start].set_queue_length(
            len(self.packages)
        )  # Set initial queue length for start satellite

        # Initialize the buffer for storing experiences
        self.buffer = ExperienceBuffer()

        # State related variables (to be updated with handle_event)
        self.state = None
        self.time = None
        self.cur_package = None
        self.cur_sat = None
        self.neighbours = []

    def handle_event(self):
        """
        The state at each step includes:
        - The current satellite and the destination positions
        - State of each neighbour, including its position, processing rate, and queue length
        """
        # Unpack event attributes
        event = self.event_queue.pop()
        package_id = event.package_id
        event_time = event.event_time
        src = event.src

        # Current and destination positions
        cur_pos = self.satellite_positions[src]
        dst_pos = self.satellite_positions[self.end]
        rel_pos = dst_pos - cur_pos

        # Find all neighbours
        neighbour_indices = find_neighbours(
            cur_idx=src,
            dst_pos=dst_pos,
            satellite_positions=self.satellite_positions,
        )
        neighbours = [self.satellites[i] for i in neighbour_indices]
        n_neighbours = len(neighbours)

        # State of each neighbour
        neighbour_states = np.array([])
        for sat in neighbours:
            rel_sat_pos = dst_pos - sat.position
            neighbour_states = np.hstack(
                [neighbour_states, rel_sat_pos, sat.processing_rate, sat.queue_length]
            )
        n_neighbours_max = self.system_config["satellite"]["n_neighbours"]
        paddings = np.zeros(5 * (n_neighbours_max - n_neighbours))
        neighbour_states = np.hstack([neighbour_states, paddings])

        # Concatenate everything and flatten into an 1-d array
        self.state = np.hstack([rel_pos, neighbour_states])
        self.time = event_time
        self.cur_package = package_id
        self.cur_sat = src
        self.neighbours = neighbour_indices

        # Update the buffer
        experience = None
        if package_id in self.buffer.buffer:
            self.buffer.update_experience(
                package_id=package_id, new_experience={"next_state": self.state}
            )
            experience = self.buffer.pop_experience(package_id)

        # Overwrite the experience in the buffer
        self.buffer.add_experience(
            package_id=package_id, experience={"state": self.state}
        )
        return experience

    def step(self, action):
        """
        After each event, take an action of transferring a package to another satellite.
        Note that the state right after the action IS NOT s'.
        By definition, s' should be the state at which we are ready to take the next action.
        Thus, we need to wait for the next event of the SAME package to update the state.
        """
        # Get the destination satellite index based on the action
        dst = self.neighbours[action]

        # Dequeue the package from the source satellite
        self.satellites[self.cur_sat].dequeue_package()

        # Calculate transmission time
        c = self.system_config["physics"]["c"]
        dist = np.linalg.norm(
            self.satellite_positions[self.cur_sat] - self.satellite_positions[dst]
        )
        arrival_time = self.time + dist / c

        # Push new event into the event queue if the package hasn't reached the end
        if dst != self.end:
            # Calculate processing time
            sat: Satellite = self.satellites[dst]
            start_time = max(
                sat.busy_time, arrival_time
            )  # Start processing when the package arrives and the satellite is ready
            processing_time = generate_processing_time(mu=sat.processing_rate)
            sat.enqueue_package(processing_time=processing_time)
            new_event = Event(
                package_id=self.cur_package,
                event_time=start_time + processing_time,
                src=dst,
            )
            self.event_queue.push(new_event)
            # Calculate reward
            reward_time = start_time + processing_time - self.time
            reward_dist = compute_arc_length(
                self.satellite_positions[dst], self.satellite_positions[self.end]
            )
            alpha = self.system_config["baseline_reward"]["alpha"]
            reward = -(reward_time + alpha * reward_dist)
        else:
            # Record the end time otherwise
            self.packages[self.cur_package].record_end_time(arrival_time)
            # Calculate reward time
            reward_time = arrival_time - self.time
            arrival_bonus = self.system_config["baseline_reward"]["arrival_bonus"]
            reward = -reward_time + arrival_bonus

        done = len(self.event_queue.events) == 0  # Check if all events are done
        # Calculate the average AoI once done
        info = {}
        if done:
            avg_aoi = np.mean(
                [pkg.end_time - pkg.generation_time for pkg in self.packages]
            )
            info["average_aoi"] = avg_aoi

        # Update the experience in the buffer with action, reward, and done
        self.buffer.update_experience(
            self.cur_package,
            new_experience={"action": action, "reward": reward, "done": done},
        )

        return done, info
