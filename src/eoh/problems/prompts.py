class GetPrompts:
    def __init__(self):
        self.prompt_task = """
        You are improving a Reinforcement Learning agent for satellite packet routing.
        You'll be given the problem description and data samples from the agent's experience.
        Based on the given information, design a function to compute offset values that adjust the agent's Q-values for better routing decisions.
        Your offset values will be added to the agent's Q-values to produce the final action selection.

        ENVIRONMENT:
        - Multiple satellites form a network, each modeled as an M/M/1/c queue (Poisson arrivals, exponential service, finite capacity c)
        - Agent routes packets from source satellite to destination satellite through intermediate hops
        - Goal: Minimize the average Age of Information (AoI), and packet drop rate (due to queue overflows)

        ROUTING DECISION:
        At each hop, current satellite chooses next hop from up to {max_neighbours} visible neighbours.
        Each neighbour characterized by:
        1. distance: Physical distance from current satellite (km)
        2. arc_length: Great circle distance from neighbour to destination (km)
        3. processing_rate: Service rate μ in M/M/1/c model (packets/second)
        4. queue_length: Current number of packets waiting in queue (0 to capacity c)
        The possible actions are the indices of the selectable neighbours, each has a Q-value estimated by the agent.
        If there are fewer than {max_neighbours} neighbours, missing neighbours are zero-padded and corresponding Q-values are masked with -np.inf

        YOUR TASK:
        Design function to compute offset values added to agent's Q-values.
        - Input: 4 arrays of length {max_neighbours}, one per feature
        - Output: offset array of length {max_neighbours}
        - Routing decision: argmax(Q_agent + offset)

        REASONING GUIDELINES:
        - High queue_length → congestion/drop risk → negative offset
        - Low arc_length → closer to destination → positive offset
        - High processing_rate → fast service → positive offset
        - Distance impacts propagation delay (at the speed of light)
        - Offsets should refine (not overpower) the learned Q-values; respect the scale observed in the examples.

        DATA FORMAT LEGEND:
        Each line represents one sample, containing state arrays and contextual information.
        Arrays corresponding to the {max_neighbours} neighbours:
        - D: Distances (km)
        - A: Arc Lengths (km)
        - P: Processing Rates (pkts/sec)
        - Q: Queue Lengths (pkts)
        
        Contextual info per sample:
        - QV: Array of base Q-Values (from agent)
        - Act: Selected Action Index
        - R: Reward received after action
        - Info: The packet outcomes after routing decisions:
            * 'dropped': dropped
            * 'arrived': arrived
            * None otherwise (still in an intermediate satellite)

        DATA SAMPLES:
        {sample_data}

        Design offset function that improves routing performance.
        """
        self.prompt_func_name = "compute_offset"
        self.prompt_func_inputs = [
            "distances",
            "arc_lengths",
            "processing_rates",
            "queue_lengths",
        ]
        self.prompt_func_outputs = ["offset_values"]
        self.prompt_inout_inf = "'distances', 'arc_lengths', 'processing_rates', 'queue_lengths', and 'offset_values' are arrays of shape ({max_neighbours},)."
        self.prompt_other_inf = "All inputs and outputs should be NumPy arrays."

    def _format_prompt_task(self, max_neighbours, sample_data):
        return self.prompt_task.format(max_neighbours=max_neighbours, sample_data=sample_data)

    def get_task(self, max_neighbours, sample_data):
        return self._format_prompt_task(max_neighbours, sample_data)

    def get_func_name(self):
        return self.prompt_func_name

    def get_func_inputs(self):
        return self.prompt_func_inputs

    def get_func_outputs(self):
        return self.prompt_func_outputs

    def get_inout_inf(self):
        return self.prompt_inout_inf

    def get_other_inf(self):
        return self.prompt_other_inf
