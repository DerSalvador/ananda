# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple
import requests
import os
import logging

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,  # @informative decorator
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
import requests
from technical import qtpylib


bias_endpoint = os.getenv("BIAS_ENDPOINT", "")
class AnandaStrategySplit(IStrategy):
    """
    This is a strategy template to get you started.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_entry_trend, populate_exit_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = "5m"

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "60": 0.01,
        "30": 0.02,
        "0": 0.04
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.10

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # Strategy parameters
    buy_rsi = IntParameter(10, 40, default=30, space="buy")
    sell_rsi = IntParameter(60, 90, default=70, space="sell")

    def get_bias(self, pair):
        market_bias = {}
        try:
            response = requests.get(f"{bias_endpoint}/sentiment/{pair}")
            response.raise_for_status()
            market_bias = response.json()
        except Exception as e:
            logging.error(f"Error getting market bias: {e}")

        market_bias = market_bias.get("final", {}).get("bias", "neutral")
        return market_bias

    def get_leverage(self, pair, proposed_leverage):
        leverage = proposed_leverage
        try:
            response = requests.get(f"{bias_endpoint}/leverage?pair={pair}")
            response.raise_for_status()
            leverage = response.json().get("leverage", proposed_leverage)
        except Exception as e:
            logging.error(f"Error getting leverage: {e}")

        return leverage

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """
        logging.warn("Ignore populate indicators")
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """
        logging.warn("Populate entry with new api")
        symbol = metadata['pair'].replace("/USDT:USDT", "")
        market_bias = self.get_bias(symbol)

        if market_bias == "neutral":
            logging.info(f"Market bias is {market_bias} for {symbol}, skipping order.")
        if market_bias == "long":       
            logging.info(f"Market bias is {market_bias} for {symbol}, going long.")
            dataframe.loc[:, ['enter_long', 'enter_tag']] = (1, 'entry_reason')
        if market_bias == "short":             
            logging.info(f"Market bias is {market_bias} for {symbol}, going short.")
            dataframe.loc[:, ['enter_short', 'enter_tag']] = (1, 'entry_reason')

        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str,
                 **kwargs) -> float:
        """
        Customize leverage for each new trade. This method is only called in futures mode.

        :param pair: Pair that's currently analyzed
        :param current_time: Current time
        :param current_rate: Current rate
        :param proposed_leverage: Proposed leverage
        :param max_leverage: Max leverage
        :param entry_tag: Entry tag
        :param side: Side
        :return: Leverage
        """
        proposed_leverage = self.get_leverage(pair, proposed_leverage)

        logging.info(f"Using leverage {proposed_leverage} for {pair}")
        return proposed_leverage

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """
        logging.warn("Ignore exit, using roi and stoploss")
        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        is_short = trade.is_short
        symbol = pair.replace("/USDT:USDT", "")
        market_bias = self.get_bias(symbol)
        logging.info(f"Market bias for {symbol} is {market_bias}")

        if market_bias == "long" and is_short:
            logging.info(f"Trade is short but bias is long, selling short")
            return True
            
        if market_bias == "short" and not is_short:
            logging.info(f"Trade is long but bias is short, selling long")
            return True
                
