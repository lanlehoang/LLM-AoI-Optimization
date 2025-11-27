import numpy as np
from src.agents.dqn import Agent
from src.utils.get_config import get_system_config
from src.utils.generators import RANDOM_SEED
from src.utils.others import state_to_arrays
from src.env.env import SatelliteEnv

N_NEIGHBOURS = get_system_config()["satellite"]["n_neighbours"]


class SatelliteRouting:
    def __init__(self, model_path) -> None:
        self.n_instance = 10
        self.agent = Agent()
        self.agent.load_model(model_path)

    def evaluate(self, code_string):
        # Initialize environment and run simulations
        np.random.seed(RANDOM_SEED)
        env = SatelliteEnv()

        aois = []
        dropped_ratios = []
        for _ in range(self.n_instance):
            env.reset()
            episode_done = False
            while not episode_done:
                env.handle_events()
                dis, arc, process, queue = state_to_arrays(env.state)
                # Execute LLM-generated code to compute offsets
                try:
                    namespace = {"np": np, "numpy": np}
                    exec(code_string, namespace)
                    # Check if function exists
                    if "compute_offset" not in namespace:
                        return None
                    compute_offset = namespace["compute_offset"]
                    q_offset = compute_offset(dis, arc, process, queue)
                except Exception as e:
                    # Code failed to execute or runtime error
                    return None

                action = self.agent.choose_action_with_offset(env.state, q_offset)
                episode_done, episode_info = env.step(action)
            # Get metrics
            aois.append(episode_info["average_aoi"])
            dropped_ratios.append(episode_info["dropped_ratio"])

        avg_aoi = np.mean(aois).item()
        avg_dropped_ratio = np.mean(dropped_ratios).item()

        fitness = (1 - avg_dropped_ratio) / avg_aoi if avg_dropped_ratio < 0.1 else 0
        return fitness
