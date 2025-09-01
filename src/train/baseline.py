from src.env.env import SatelliteEnv
from src.utils.logger import get_logger
from src.utils.get_config import get_agent_config, get_system_config
from src.utils.generators import RANDOM_SEED
from src.agents.dqn import Agent
import numpy as np
import pandas as pd

logger = get_logger(__name__)
system_config = get_system_config()
agent_config = get_agent_config()


def main():
    logger.info("Initializing the environment with random seed {RANDOM_SEED}")
    np.random.seed(RANDOM_SEED)
    env = SatelliteEnv()

    logger.info("Initializing the agent")
    agent = Agent()
    steps = 0  # Count steps to decay epsilon
    decay_interval = agent_config["epsilon"]["decay_interval"]

    aois = []  # AoI

    logger.info("Training the agent")
    for i in range(agent_config["train"]["epochs"]):
        env.reset()
        done = False
        while not done:
            experience = env.handle_event()
            # Push previously updated experience into ReplayBuffer
            if experience:
                state = experience["state"]
                action = experience["action"]
                reward = experience["reward"]
                next_state = experience["next_state"]
                done = experience["done"]
                agent.remember(state, action, reward, next_state, done)
                agent.learn()
            action = agent.choose_action(env.state, len(env.neighbours))
            info = env.step(action)

            # Epsilon decay
            steps = (steps + 1) % decay_interval
            if steps == 0:
                logger.info(f"Decaying epsilon. Current epsilon: {agent.epsilon}")
                agent.decay_epsilon()

        logger.info(f"Average AoI of episode {i + 1}: {info['average_aoi']:.6f}")
        aois.append(info["average_aoi"])

    logger.info("Saving the results")
    df = pd.DataFrame({"AoI": aois})
    df.to_csv("results.csv", index=False)


if __name__ == "__main__":
    main()
