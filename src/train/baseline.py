from src.env.env import SatelliteEnv
from src.utils.logger import get_logger
from src.utils.get_config import get_agent_config, get_system_config
from src.utils.generators import RANDOM_SEED
from src.agents.dqn import Agent
import numpy as np
from src.env.state_models import EnvironmentState
from datetime import datetime
import os
from pathlib import Path
import pandas as pd

logger = get_logger(__name__)
system_config = get_system_config()
agent_config = get_agent_config()

ENVIRONMENT_SHAPE = EnvironmentState.STATE_DIM

# Define paths
BASE_DIR = Path.resolve(Path(__file__)).parent.parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"


def main():
    logger.info(f"Initializing the environment with random seed {RANDOM_SEED}")
    np.random.seed(RANDOM_SEED)
    env = SatelliteEnv()

    logger.info("Initializing the agent")
    agent = Agent(input_dims=ENVIRONMENT_SHAPE)
    steps = 0  # Count steps to decay epsilon
    decay_interval = agent_config["train"]["epsilon"]["decay_interval"]

    aois = []  # AoI
    dropped_ratios = []

    logger.info("Training the agent")
    for i in range(agent_config["train"]["epochs"]):
        logger.info(f"\n\n---Episode {i + 1} begins---")
        logger.info(f"Epsilon: {agent.epsilon:.3}")
        env.reset()
        episode_done = False
        while not episode_done:
            env.handle_events()
            # Push all complete experiences into ReplayBuffer
            experiences = env.buffer.get_all_complete_experiences()
            for experience in experiences:
                state = experience["state"]
                action = experience["action"]
                reward = experience["reward"]
                next_state = experience["next_state"]
                info = experience.get("info")
                done = True if info else False  # Packet is done if info is not None
                agent.remember(state, action, reward, next_state, done)
            # TODO: Combine experiences and Q-values logging to save data samples
            agent.learn()
            action, _, _ = agent.choose_action(env.state)
            episode_done, episode_info = env.step(action)

            # Epsilon decay
            steps = (steps + 1) % decay_interval
            if steps == 0:
                agent.decay_epsilon()

        logger.info(f"Average AoI of episode {i + 1}: {episode_info['average_aoi']:.4f}")
        aois.append(episode_info["average_aoi"])
        logger.info(f"Dropped ratio of episode {i + 1}: {episode_info['dropped_ratio']:.4f}")
        dropped_ratios.append(episode_info["dropped_ratio"])
        logger.info(f"---Episode {i + 1} ends---")

    logger.info(f"AoI: {[aoi.item() for aoi in np.round(aois, 4)]}")
    logger.info(f"Dropped ratios: {[r.item() for r in np.round(dropped_ratios, 4)]}")
    logger.info("Saving the results")

    # Save the model
    os.makedirs(MODEL_DIR, exist_ok=True)
    MODEL_PATH = f"{MODEL_DIR}/dqn_{datetime.now().strftime("%Y-%m-%d")}.pth"
    agent.save_model(MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")

    # Save the results
    os.makedirs(DATA_DIR, exist_ok=True)
    DATA_PATH = f"{DATA_DIR}/dqn_results_{datetime.now().strftime("%Y-%m-%d")}.csv"
    df = pd.DataFrame({
        "AoI": [aoi.item() for aoi in np.round(aois, 4)],
        "Dropped_Ratio": [r.item() for r in np.round(dropped_ratios, 4)],
    })
    df.to_csv(DATA_PATH, index=False)
    logger.info(f"Results saved to {DATA_PATH}")

if __name__ == "__main__":
    main()
