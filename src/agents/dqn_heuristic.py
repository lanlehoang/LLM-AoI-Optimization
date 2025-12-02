import json
from .dqn import DqnAgent
from src.utils.others import state_to_arrays
import numpy as np


class DqnHeuristicAgent:
    def __init__(self, dqn_model_path: str, heuristic_config_path: str):
        # Load the pre-trained DQN model
        self.dqn_agent = DqnAgent()
        self.dqn_agent.load_model(dqn_model_path)

        # Load heuristic configuration
        with open(heuristic_config_path, "r") as f:
            heuristic_config = json.load(f)

        # Extract the code from heuristic configuration
        self.code = heuristic_config.get("code")
        if not self.code:
            raise ValueError("Heuristic configuration must contain 'code' key.")
        self.offset_function = self._exec_code_string(self.code)

    def _exec_code_string(self, code_string):
        namespace = {"np": np, "numpy": np}
        exec(code_string, namespace)

        # Load directly because the code was validated during heuristic generation
        return namespace["compute_offset"]

    def choose_action(self, state):
        # Convert state to arrays suitable for DQN input
        distances, arc_lengths, processing_rates, queue_lengths = state_to_arrays(state)
        q_offset = self.offset_function(distances, arc_lengths, processing_rates, queue_lengths)
        action = self.dqn_agent.choose_action_with_offset(state, q_offset)
        return action
