import os
from functools import cache
import json
from bias import CONFIG_PATH
from utils import get_logger

futures_config_path = os.path.join(CONFIG_PATH, "futures.json")
logger = get_logger()

@cache
def get_futures_config():
    json_data = {}
    if os.path.exists(futures_config_path):
        with open(futures_config_path) as f:
            json_data = json.load(f)
    return json_data

def update_leverage(pair: str = "default", leverage: float = 5.0):
    config = get_futures_config()
    config[pair] = leverage
    with open(futures_config_path, "w") as f:
        json.dump(config, f)
    get_futures_config.cache_clear()

def get_leverage(pair: str = "default"):
    config = get_futures_config()
    logger.info(f"Config: {config}")
    fallback_leverage = config.get(pair, config.get("default", 5.0))
    return fallback_leverage

