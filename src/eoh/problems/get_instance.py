import numpy as np
from src.utils.generators import RANDOM_SEED
from src.env.env import SatelliteEnv


class GetData:
    def __init__(self, n_instance):
        self.n_instance = n_instance

    def generate_instances(self):
        np.random.seed(RANDOM_SEED)
        instance_data = []
        for _ in range(self.n_instance):
            env = SatelliteEnv()
            # TODO: Continue the implementation to generate and return instances
        return instance_data
