import os
from functools import cache
import json
from bias.interface import BiasType
from utils import TimeBasedDeque
from bias import CONFIG_PATH, get_config
from tinydb import TinyDB, Query
from utils import get_logger
from collections import Counter

futures_config_path = os.path.join(CONFIG_PATH, "futures.json")
logger = get_logger()

#Use in memory store for real sentiments in the last 3 hours
real_sentiments = {}

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

def update_sentiment(symbol: str, sentiment: str):
    if symbol not in real_sentiments:
        real_sentiments[symbol] = TimeBasedDeque(60*60*2)
    real_sentiments[symbol].add(sentiment)

def get_real_sentiment(symbol: str):
    if symbol not in real_sentiments:
        return []
    seconds_to_check = int(get_config("ReverseTrendCheckBackSeconds", 60*10))
    sentiments = real_sentiments[symbol].get_items_last_x_seconds(seconds_to_check)
    total = len(sentiments)
    if total < int(get_config("ReverseTrendCheckMinCount", 60)):
        logger.info(f"Insufficient data {total} for {symbol} to check reverse trend, returning NEUTRAL")
        return BiasType.NEUTRAL
    count = Counter(sentiments)
    percentages = {key: (value / total) * 100 for key, value in count.items()}
    percentageThreshold = int(get_config("ReverseTrendCheckPercentageAgreeThreshold", 100))
    if percentages[BiasType.LONG.value] >= percentageThreshold:
        logger.info(f"Real sentiment for {symbol}: LONG")
        return BiasType.LONG
    if percentages[BiasType.SHORT.value] >= percentageThreshold:
        logger.info(f"Real sentiment for {symbol}: SHORT")
        return BiasType.SHORT
    logger.info(f"Real sentiment for {symbol}: NEUTRAL, percentage not meeting threshold")
    return BiasType.NEUTRAL
