from src.env.env import SatelliteEnv
from src.utils.logger import get_logger
from src.utils.get_config import get_agent_config, get_system_config
from src.utils.generators import RANDOM_SEED
from src.agents.dqn import Agent
import numpy as np

# import pandas as pd

logger = get_logger(__name__)
system_config = get_system_config()
agent_config = get_agent_config()


def main():
    logger.info(f"Initializing the environment with random seed {RANDOM_SEED}")
    np.random.seed(RANDOM_SEED)
    env = SatelliteEnv()

    logger.info("Initializing the agent")
    agent = Agent(
        input_dims=53
    )  # TODO: Convert input_dims to variable instead of hardcode
    steps = 0  # Count steps to decay epsilon
    decay_interval = agent_config["train"]["epsilon"]["decay_interval"]

    aois = []  # AoI
    dropped_ratios = []

    logger.info("Training the agent")
    for i in range(agent_config["train"]["epochs"]):
        logger.info(f"\n\n---Episode {i + 1} begins---")
        logger.info(f"Epsilon: {agent.epsilon:.3}")
        env.reset()
        done = False
        while not done:
            env.handle_events()
            # Push all complete experiences into ReplayBuffer
            experiences = env.buffer.get_all_complete_experiences()
            for experience in experiences:
                state = experience["state"]
                action = experience["action"]
                reward = experience["reward"]
                next_state = experience["next_state"]
                done = experience["done"]
                agent.remember(state, action, reward, next_state, done)
            agent.learn()
            action, _, _ = agent.choose_action(env.state)
            done, info = env.step(action)

            # Epsilon decay
            steps = (steps + 1) % decay_interval
            if steps == 0:
                agent.decay_epsilon()

        logger.info(f"Average AoI of episode {i + 1}: {info['average_aoi']:.4f}")
        aois.append(info["average_aoi"])
        logger.info(f"Dropped ratio of episode {i + 1}: {info['dropped_ratio']:.4f}")
        dropped_ratios.append(info["dropped_ratio"])
        logger.info(f"---Episode {i + 1} ends---")

    logger.info(f"AoI: {[np.round(aois, 4)]}")
    logger.info(f"Dropped ratios: {[ratio for ratio in np.round(dropped_ratios, 4)]}")
    logger.info("Saving the results")
    # df = pd.DataFrame({"AoI": aois})
    # df.to_csv("results.csv", index=False)


if __name__ == "__main__":
    main()
