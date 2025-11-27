import numpy as np
from src.utils.generators import *
from src.utils.geometry import *
from src.utils.get_config import get_system_config, get_agent_config
from src.env.env_classes import *
from src.utils.logger import get_logger
from src.env.state_models import NeighbourState, EnvironmentState

logger = get_logger(__name__)
system_config = get_system_config()
agent_config = get_agent_config()


class SatelliteEnv:
    """
    Custom Gym environment for satellite operations.
    """

    def __init__(self):
        self.n_satellites = system_config["satellite"]["n_satellites"]
        self.radius = (
            system_config["physics"]["r_earth"] + system_config["satellite"]["height"]
        )  # Radius of the satellites' orbit

        # Initialize satellite positions
        satellite_positions = generate_satellite_positions(self.n_satellites, self.radius)
        self.satellite_positions = satellite_positions  # Keep persistent for neighbour calculations

        # Choose start and end satellites
        start, end = choose_start_end(satellite_positions)
        self.start = start
        self.end = end

        # Initialize satellite processing rates
        mu_lower = system_config["satellite"]["mu"]["lower"]
        mu_upper = system_config["satellite"]["mu"]["upper"]
        satellite_processing_rates = generate_satellite_processing_rates(
            self.n_satellites, lower=mu_lower, upper=mu_upper
        )
        # Scale processing rates for start satellite to intentionally create congestion in the network,
        # which forces the agent to learn dynamic routing strategies
        scale = system_config["satellite"]["mu"]["start_scale"]
        satellite_processing_rates[start] *= scale

        # Create satellite objects
        self.satellites = [
            Satellite(
                position=satellite_positions[i],
                processing_rate=satellite_processing_rates[i],
            )
            for i in range(self.n_satellites)
        ]

        # Generate all packets at the start satellite
        mu = self.satellites[self.start].processing_rate
        simulation_time = system_config["physics"]["simulation_time"]
        # Generate all packets at the start satellite
        self.initial_packets = generate_all_packets(mu, simulation_time)
        logger.info(f"Number of packets generated: {len(self.initial_packets)}")

    def reset(self):
        """
        Empty satellite queues and reset packets.
        Args:
            seed (int, optional): Random seed.
            options (dict, optional): Additional options.
        Returns:
            Tuple[state, info]: The initial state and info dictionary.
        """
        # Empty satellite queues
        for satellite in self.satellites:
            satellite.reset_queue()

        # Reset packets and event queue
        self.packets = [Packet(gen_time) for _, gen_time in self.initial_packets]

        self.event_queue = EventQueue(
            [
                Event(pkg_id, generation_time, EventType.PROCESSED.value, self.start)
                for pkg_id, generation_time in self.initial_packets
            ]
        )
        self.satellites[self.start].set_queue_length(len(self.packets))  # Set initial queue length for start satellite

        # Initialize the buffer for storing experiences
        self.buffer = ExperienceBuffer()

        # State related variables (to be updated with handle_events)
        self.state = None
        self.time = None
        self.cur_packet = None
        self.cur_sat = None
        self.neighbours = []

    def handle_events(self):
        """
        The state at each step includes:
        - The current satellite and the destination positions
        - State of each neighbour, including its position, processing rate, and queue length
        """
        # Unpack event attributes
        event = self.event_queue.pop()

        # Handle ARRIVAL events
        while event.event_type == EventType.ARRIVAL.value:
            packet_id = event.packet_id
            event_time = event.event_time
            src = event.sat
            # Start processing
            # Check if the next satellite queue is full at arrival time
            sat: Satellite = self.satellites[src]
            if sat.busy_time > event_time and sat.queue_length >= system_config["satellite"]["queue_limit"]:
                self.packets[packet_id].drop()
                reward = -agent_config["train"]["reward"]["drop_penalty"]
                self.buffer.update_experience(
                    packet_id=packet_id,
                    new_experience={
                        "reward": reward,
                        "next_state": self.state,  # Doesn't matter as the packet is done
                        "info": "dropped",  # Done packet, not done episode
                    },
                )
                logger.info(f"Packet ID {packet_id} dropped by satellite {src}.")
                self.buffer.complete_experience(packet_id=packet_id)
            else:
                # Start processing when the packet arrives and the satellite is ready
                start_time = max(sat.busy_time, event_time)
                processing_time = generate_processing_time(mu=sat.processing_rate)
                sat.enqueue_packet(start_time=start_time, processing_time=processing_time)
                new_event = Event(
                    packet_id=packet_id,
                    event_time=start_time + processing_time,
                    event_type=EventType.PROCESSED.value,
                    sat=src,
                )
                self.event_queue.push(new_event)
                # Calculate reward
                reward_time = start_time + processing_time - self.packets[packet_id].sent_time
                reward_dist = compute_arc_length(self.satellite_positions[src], self.satellite_positions[self.end])
                c = system_config["physics"]["c"]
                alpha = agent_config["train"]["reward"]["alpha"]
                reward = -(reward_time + alpha * reward_dist / c)
                self.buffer.update_experience(
                    packet_id=packet_id,
                    new_experience={"reward": reward},
                )
            # Continue resolving ARRIVAL events
            event = self.event_queue.pop()

        # Handle a single PROCESSED event
        packet_id = event.packet_id
        event_time = event.event_time
        src = event.sat

        # Current and destination positions
        cur_pos = self.satellite_positions[src]
        dst_pos = self.satellite_positions[self.end]

        # Find all neighbours
        neighbour_indices = find_neighbours(
            cur_idx=src,
            dst_pos=dst_pos,
            satellite_positions=self.satellite_positions,
        )
        neighbours = [self.satellites[i] for i in neighbour_indices]
        n_neighbours = len(neighbours)

        # State of each neighbour
        neighbour_states = []
        for sat in neighbours:
            euc_dist = compute_euclidean_distance(cur_pos, sat.position)
            arc_len = compute_arc_length(sat.position, dst_pos)
            sat_state = NeighbourState(
                distance=euc_dist,
                arc_length=arc_len,
                processing_rate=sat.processing_rate,
                queue_length=sat.queue_length,
            )
            neighbour_states.append(sat_state)
        n_neighbours_max = system_config["satellite"]["n_neighbours"]
        paddings = [NeighbourState(distance=0, arc_length=0, processing_rate=0, queue_length=0)] * (
            n_neighbours_max - n_neighbours
        )
        neighbour_states.extend(paddings)

        # Update the environment state
        self.state = EnvironmentState(neighbours=neighbour_states).to_numpy()

        # Other variable updates
        self.time = event_time
        self.cur_packet = packet_id
        self.cur_sat = src
        self.neighbours = neighbour_indices

        # Update the buffer
        if packet_id in self.buffer.buffer:
            self.buffer.update_experience(packet_id=packet_id, new_experience={"next_state": self.state})
            self.buffer.complete_experience(packet_id)

        # Overwrite the experience in the buffer
        self.buffer.add_experience(packet_id=packet_id, experience={"state": self.state})

    def step(self, action):
        """
        After each event, take an action of transferring a packet to another satellite.
        Note that the state right after the action IS NOT s'.
        By definition, s' should be the state at which we are ready to take the next action.
        Thus, we need to wait for the next PROCESS event of the SAME packet to update the state.
        """
        # Get the destination satellite index based on the action
        dst = self.neighbours[action]

        # Dequeue the packet from the source satellite
        self.satellites[self.cur_sat].dequeue_packet()

        # Calculate transmission time
        c = system_config["physics"]["c"]
        dist = np.linalg.norm(self.satellite_positions[self.cur_sat] - self.satellite_positions[dst])
        arrival_time = self.time + dist / c

        # Push new event into the event queue if the packet hasn't reached the end
        if dst != self.end:
            self.packets[self.cur_packet].record_sent_time(self.time)
            self.buffer.update_experience(
                self.cur_packet,
                new_experience={"action": action},
            )
            # Push new ARRIVAL event
            arrival_event = Event(
                packet_id=self.cur_packet,
                event_time=arrival_time,
                event_type=EventType.ARRIVAL.value,
                sat=dst,  # destination satellite
            )
            self.event_queue.push(arrival_event)
            return False, {}
        else:
            # Record the end time otherwise
            self.packets[self.cur_packet].record_end_time(arrival_time)
            # Calculate reward time
            reward_time = arrival_time - self.time
            arrival_bonus = agent_config["train"]["reward"]["arrival_bonus"]
            reward = -reward_time + arrival_bonus

            # Update the experience in the buffer with action, reward, and done
            self.buffer.update_experience(
                self.cur_packet,
                new_experience={
                    "action": action,
                    "reward": reward,
                    "next_state": self.state,  # Doesn't matter as the packet is done
                    "info": "arrived",  # Done packet, not done episode
                },
            )
            logger.info(
                f"Packet ID {self.cur_packet} reached the destination. "
                f"AoI: {arrival_time - self.packets[self.cur_packet].generation_time:.4}"
            )
            self.buffer.complete_experience(packet_id=self.cur_packet)

            # Calculate the average AoI and packet drop ratio once done
            episode_done = len(self.event_queue.events) == 0
            episode_info = {}
            if episode_done:
                avg_aoi = np.mean([pkg.end_time - pkg.generation_time for pkg in self.packets if not pkg.dropped])
                n_dropped = len([pkg for pkg in self.packets if pkg.dropped])
                dropped_ratio = n_dropped / len(self.packets)
                episode_info["average_aoi"] = avg_aoi
                episode_info["dropped_ratio"] = dropped_ratio

            return episode_done, episode_info
