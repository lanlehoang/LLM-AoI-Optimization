from src.eoh import evol
from src.eoh.utils.get_params import Params
import os
from pathlib import Path

# Load environment variables
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o-mini"
API_ENDPOINT = "api.openai.com"

# Get the pretrained DQN model path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "dqn_baseline_20251201.pth"


def run_evolution():
    # Parameter initilization
    params = Params()

    # Set parameters
    params.set_params(
        llm_api_endpoint=API_ENDPOINT,
        llm_api_key=API_KEY,
        llm_model=MODEL,
        ec_pop_size=3,  # Number of individuals per population
        ec_n_pop=5,  # Number of populations (iterations)
        exp_n_proc=1,  # Multi-core parallel
        exp_debug_mode=False,
    )

    evolution = evol.Evol(params, model_path=str(MODEL_PATH))
    evolution.run()


if __name__ == "__main__":
    run_evolution()
