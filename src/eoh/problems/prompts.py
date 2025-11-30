from src.utils.others import generate_prompt_data
from src.utils.get_config import get_system_config
import tiktoken

MAX_NEIGHBOURS = get_system_config()["satellite"]["n_neighbours"]
DATA_SAMPLE_PATH = "data/dqn_data_samples_20251127.csv"


class GetPrompts:
    def __init__(self):
        self.prompt_task = """You are improving a Reinforcement Learning agent for packet routing.

        PROBLEM DESCRIPTION:
        - Multiple satellites form a network, each modeled as an M/M/1/c queue
        - Agent routes packets from source satellite to destination satellite through intermediate hops
        - Goal: Minimize the average Age of Information and packet drop rate (due to queue overflows)

        ROUTING DECISION:
        Each step, current satellite chooses next hop from {max_neighbours} visible neighbours.
        Each neighbour features:
        1. distance: Physical distance from current satellite (km)
        2. arc_length: Great circle distance from neighbour to destination (km)
        3. processing_rate: Service rate μ in M/M/1/c model (packets/second)
        4. queue_length: Current number of packets waiting in queue
        The possible actions are the indices of the selectable neighbours, each has a Q-value estimated by the agent.
        If there are fewer than {max_neighbours} neighbours, missing neighbours are zero-padded and corresponding Q-values are -np.inf

        EXAMPLE DATA FORMAT:
        You're given some examples, each written in one line. Each example contains state arrays and contextual information.
        State arrays:
        - D: Distances
        - A: Arc Lengths
        - P: Processing Rates
        - Q: Queue Lengths

        Contextual information:
        - QV: Agent estimated Q-Values
        - Act: Selected Action Index (0-indexed)
        - R: Reward received
        - Info: The packet outcome:
            * 'dropped': dropped
            * 'arrived': arrived
            * None otherwise (still in transit)

        DATA SAMPLES:
        {data_samples}

        YOUR TASK:
        Design function computing offset values added to agent's Q-values.
        - Input: 4 arrays of length {max_neighbours}, one per feature
        - Output: offset array of length {max_neighbours}
        - Routing decision: argmax(Q_agent + Q_offset)

        REASONING GUIDELINES:
        - High queue_length → congestion/drop risk → negative offset
        - Low arc_length → closer to destination → positive offset
        - High processing_rate → fast service → positive offset
        - Distance impacts propagation delay (at the speed of light)
        - Offsets should refine (not overpower) the learned Q-values; respect the scale observed in the examples.

        Design a function that improves routing performance."""
        self.prompt_func_name = "compute_offset"
        self.prompt_func_inputs = [
            "distances",
            "arc_lengths",
            "processing_rates",
            "queue_lengths",
        ]
        self.prompt_func_outputs = ["offsets"]
        self.prompt_inout_inf = (
            "'distances', 'arc_lengths', 'processing_rates', 'queue_lengths', and 'offsets' are of the same shape."
        )
        self.prompt_other_inf = "All inputs and outputs should be NumPy arrays."

    def _format_prompt_task(self, data_path):
        N_SAMPLES = {"num_dropped": 2, "num_arrived": 1, "num_none": 2}
        data_samples = generate_prompt_data(csv_path=data_path, **N_SAMPLES)
        return self.prompt_task.format(max_neighbours=MAX_NEIGHBOURS, data_samples=data_samples)

    def get_task(self, data_path=DATA_SAMPLE_PATH):
        return self._format_prompt_task(data_path)

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


if __name__ == "__main__":
    getprompts = GetPrompts()
    task_prompt = getprompts.get_task()
    print(task_prompt)

    # Calculate number of tokens
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    print(f"\nNumber of tokens in task prompt: {len(enc.encode(task_prompt))}")
