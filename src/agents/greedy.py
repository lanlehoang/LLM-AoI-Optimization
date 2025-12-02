"""
Greedy methods for comparison with the proposed approach.
"""

import numpy as np
from src.utils.others import state_to_arrays


class GreedyAgent:
    @staticmethod
    def choose_nearest_to_destination(state):
        _, arc_lengths, _, _ = state_to_arrays(state)
        action = int(np.argmin(arc_lengths))
        return action

    @staticmethod
    def choose_best_expected_time(state):
        _, arc_lengths, processing_rates, queue_lengths = state_to_arrays(state)

        # Check if the destination is reachable
        destination_mask = (arc_lengths == 0) & (processing_rates > 0)  # Differentiate from zero padding
        if np.any(destination_mask):
            action = int(np.where(destination_mask)[0][0])
        else:
            # If the destination is not immediately reachable, choose the best expected time
            expected_times = (queue_lengths + 1) / processing_rates
            action = int(np.argmin(expected_times))
        return action
