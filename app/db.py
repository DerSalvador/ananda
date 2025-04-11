import os
from functools import cache
import json
from bias.interface import BiasType
from utils import TimeBasedDeque
from bias import CONFIG_PATH, get_config
from tinydb import TinyDB, Query
from utils import get_logger
from collections import Counter
import numpy as np

futures_config_path = os.path.join(CONFIG_PATH, "futures.json")
logger = get_logger()

#Use in memory store for real sentiments in the last 3 hours
real_sentiments = {}
profits = {}

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

def update_profit(symbol: str, profit: float, is_short: bool = False):
    if symbol not in profits:
        profits[symbol] = TimeBasedDeque(60*60*2)
    profits[symbol].add({"profit": profit, "is_short": is_short})

def get_profits(symbol: str):
    if symbol not in profits:
        return []
    return profits[symbol].get_items()

def current_position(symbol: str):
    if symbol not in profits:
        return BiasType.NEUTRAL
    seconds_to_check = int(get_config("ReverseTrendCheckBackSeconds", 60*10))
    last_bias = [x.get("is_short") for x in profits[symbol].get_items_last_x_seconds(seconds_to_check)]
    if not last_bias:
        return BiasType.NEUTRAL
    if last_bias[-1]:
        return BiasType.SHORT
    else:
        return BiasType.LONG

def is_linear_decreasing(profits, threshold = 95.0):
    if len(profits) < 2:
        return False  # not enough data to compute a line

    x = np.arange(len(profits))
    y = np.array(profits, dtype=np.float64)

    # Fit linear regression line: y = mx + c
    m, c = np.polyfit(x, y, 1)
    y_fit = m * x + c

    # Check if slope is negative (i.e., decreasing)
    m = round(m,2)
    if m >= 0:
        return False

    # Calculate deviation from the line
    deviation = y - y_fit
    deviation = np.round(deviation, 2)
    std_dev = np.std(deviation)

    if std_dev == 0:
        return True  # perfectly linear

    # Count how many points deviate more than one std deviation
    outliers = np.abs(deviation) > std_dev
    percent_outliers = np.sum(outliers) / len(profits) * 100

    return percent_outliers <= 100 - threshold

def should_reverse(symbol: str):
    if symbol not in profits:
        return False
    seconds_to_check = int(get_config("ReverseTrendCheckBackSeconds", 60*10))
    last_profits = [x.get("profit") for x in profits[symbol].get_items_last_x_seconds(seconds_to_check)]
    min_length = int(get_config("ReverseTrendCheckMinCount", 60))
    if len(last_profits) < min_length:
        logger.info(f"Insufficient data {len(last_profits)} for {symbol} to check reverse trend")
        return False

    true_so_far = False
    reverse_trend_negative_percent = float(get_config("ReverseTrendShouldBeNegativePercent", 100))
    negative_count = sum(1 for profit in last_profits if profit < 0)
    total_count = len(last_profits)
    negative_percent = round((negative_count / total_count) * 100, 2)
    logger.info(f"Negative percent for {symbol}: {negative_percent:.2f}%")
    if negative_percent < reverse_trend_negative_percent:
        logger.info(f"Negative percent {negative_percent:.2f}% is less than threshold {reverse_trend_negative_percent}% for {symbol}")
        return False
    else:
        true_so_far = True

    check_first_greater_than_last = get_config("ReverseTrendCheckFirstGreater", "true") == "true"
    if check_first_greater_than_last:
        first_profit = last_profits[0]
        last_profit = last_profits[-1]
        if first_profit < last_profit:
            logger.info(f"First profit is lesser than last for {symbol}")
            return False
        else:
            logger.info(f"First profit is greater than last for {symbol}")
            true_so_far = True

    check_linear_decreasing = get_config("ReverseTrendCheckLinearDecreasing", "true") == "true"
    if check_linear_decreasing:
        linear_decreasing_threshold_percent = float(get_config("ReverseTrendLinearDecreasingThresholdPercent", 95))
        if is_linear_decreasing(last_profits, linear_decreasing_threshold_percent):
            logger.info(f"Profits are linearly decreasing for {symbol}")
            true_so_far = True
        else:
            logger.info(f"Profits are not linearly decreasing for {symbol}")
            return False

    return true_so_far
