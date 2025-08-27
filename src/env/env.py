import numpy as np
from src.utils.generators import *
from src.utils.geometry import *
from src.utils.get_config import get_system_config
from env_classes import *

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)


class SatelliteEnv:
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

        self.packages = [Package(gen_time) for _, gen_time in packages]

        self.event_queue = EventQueue(
            [
                Event(pkg_id, generation_time, self.start)
                for pkg_id, generation_time in packages
            ]
        )
        self.satellites[self.start].set_queue_length(
            len(self.packages)
        )  # Set initial queue length for start satellite

        # Initialize the buffer for storing experiences
        self.buffer = ExperienceBuffer()

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

        # Generate all packages at the start satellite
        mu = self.satellites[self.start].processing_rate
        simulation_time = self.system_config["simulation_time"]
        packages = generate_all_packages(mu, simulation_time)

        self.packages = [Package(gen_time) for _, gen_time in packages]

        self.event_queue = EventQueue(
            [
                Event(pkg_id, generation_time, self.start)
                for pkg_id, generation_time in packages
            ]
        )
        self.satellites[self.start].set_queue_length(
            len(self.packages)
        )  # Set initial queue length for start satellite

        # Initialize the buffer for storing experiences
        self.buffer = ExperienceBuffer()

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

    def step(self, action, event: Event):
        """
        After each event, take an action of transferring a package to another satellite.
        Note that the state right after the action IS NOT s'.
        By definition, s' should be the state at which we are ready to take the next action.
        Thus, we need to wait for the next event of the SAME package to update the state.
        """
        # Unpack event attributes
        package_id = event.package_id
        event_time = event.event_time
        src = event.src

        # Find the list of neighbours for the current satellite
        neighbours = find_neighbours(
            cur_idx=src,
            dst_pos=self.satellite_positions[self.end],
            satellite_positions=self.satellite_positions,
        )
        # Get the destination satellite index based on the action
        dst = neighbours[action]

        # Dequeue the package from the source satellite
        self.satellites[src].dequeue_package()

        # Calculate transmission time
        c = self.system_config["physics"]["c"]
        dist = np.linalg.norm(
            self.satellite_positions[src] - self.satellite_positions[dst]
        )
        arrival_time = event_time + dist / c

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
                package_id=package_id,
                event_time=start_time + processing_time,
                src=dst,
            )
            self.event_queue.push(new_event)
            # Calculate reward
            reward_time = start_time + processing_time - event_time
            reward_dist = compute_arc_length(
                self.satellite_positions[dst], self.satellite_positions[self.end]
            )
            alpha = self.system_config["baseline_reward"]["alpha"]
            reward = -(reward_time + alpha * reward_dist)
        else:
            # Record the end time otherwise
            self.packages[package_id].record_end_time(arrival_time)
            # Calculate reward time
            reward_time = arrival_time - event_time
            arrival_bonus = self.system_config["baseline_reward"]["arrival_bonus"]
            reward = -reward_time + arrival_bonus

        # Add the experience to the buffer
        self.buffer.add_experience(
            package_id,
            experience={
                "action": action,
                "reward": reward,
            },
        )

        done = len(self.event_queue.events) == 0  # Check if all events are done
        # Calculate the average AoI once done
        info = {}
        if done:
            avg_aoi = np.mean(
                [pkg.end_time - pkg.generation_time for pkg in self.packages]
            )
            info["average_aoi"] = avg_aoi

        return reward, done, info
