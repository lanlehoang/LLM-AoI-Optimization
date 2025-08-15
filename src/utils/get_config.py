import yaml
import pathlib
from enum import Enum
import dotenv

CONFIG_DIR = pathlib.Path(__file__).parent.parent.parent / "config"
ENV_DIR = pathlib.Path(__file__).parent.parent.parent / ".env"


class ConfigPaths(Enum):
    SYSTEM_CONFIG = CONFIG_DIR / "system_config.yaml"


def get_system_config():
    with open(ConfigPaths.SYSTEM_CONFIG.value, "r") as file:
        config = yaml.safe_load(file)
    return config


def get_env():
    dotenv.load_dotenv(dotenv_path=ENV_DIR)
