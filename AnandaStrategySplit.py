# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import time
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
from collections import deque

class TimeBasedDeque:
    def __init__(self, max_age=3600):
        self.max_age = max_age
        self.queue = deque()

    def add(self, item):
        """Insert a new item with the current timestamp."""
        timestamp = time.time()
        self.queue.append((timestamp, item))
        self.cleanup()

    def cleanup(self):
        now = time.time()
        while self.queue and (now - self.queue[0][0] > self.max_age):
            self.queue.popleft()

    def get_items_last_x_seconds(self, seconds):
        """Retrieve items from the last X seconds."""
        threshold = time.time() - (seconds)
        return [item for t, item in self.queue if t >= threshold]


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
    api_profit: "http://freqtrade:'&&Liz270117!!'@freqtrade-clusterip-bot-ananda.bot-ananda.svc.cluster.local:8084/api/v1/profit"

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = "1m"

    # Can this strategy go short?
    can_short: bool = True

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
        reason = ""
        try:
            response = requests.get(f"{bias_endpoint}/sentiment/{pair}")
            response.raise_for_status()
            market_bias = response.json()
            reason = market_bias.get("final", {}).get("reason", "")
            market_bias = market_bias.get("final", {}).get("bias", "neutral")
        except Exception as e:
            s = f"Error getting market bias: {e}"
            logging.error(s)
            market_bias = "neutral"
            raise e
        finally:
            return market_bias, reason

    def set_sentiment(self, pair, sentiment):
        try:
            response = requests.post(f"{bias_endpoint}/sentiment/{pair}", json={"bias": sentiment})
            response.raise_for_status()
        except Exception as e:
            logging.error(f"Error setting sentiment: {e}")

    def get_leverage(self, pair, proposed_leverage):
        leverage = 5 # proposed_leverage
        try:
            response = requests.get(f"{bias_endpoint}/leverage?pair={pair}")
            response.raise_for_status()
            leverage = response.json().get("leverage", proposed_leverage)
        except Exception as e:
            logging.error(f"Error getting leverage: {e}")
            raise e
        finally:
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
        # dataframe.drop(dataframe.index, inplace=True)
        logging.warn("Populate entry with new api")
        symbol = metadata['pair'].split("/")[0]
        market_bias, reason = self.get_bias(symbol)

        logging.info(f"Market bias is {market_bias} for {symbol}, reason: {reason}")
        if market_bias == "neutral":
            logging.info(f"Market bias is {market_bias} for {symbol}, skipping order.")
            dataframe.loc[:, ['enter_long', 'enter_tag']] = (0, reason)
            dataframe.loc[:, ['enter_short', 'enter_tag']] = (0, reason)
        elif market_bias == "long":       
            dataframe.loc[:, ['enter_long', 'enter_tag']] = (1, reason)
            s = f"Market bias is {market_bias} for {symbol}, going long."
            logging.info(s)
            self.dp.send_msg(s)
        elif market_bias == "short":
            dataframe.loc[:, ['enter_short', 'enter_tag']] = (1, reason)
            s = f"Market bias is {market_bias} for {symbol}, going short."
            logging.info(s)
            self.dp.send_msg(s)
        else:
            raise Exception(f"Market Bias unknown {market_bias}")

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

    def is_linear_decreasing(self, profits, threshold=0.05):
        period = len(profits)
        y = np.array(profits, dtype=np.float64)
        slope = ta.LINEARREG_SLOPE(y, timeperiod=period)

        if slope is None:
            return False

        latest_slope = slope[-1]

        return latest_slope < 0 and abs(latest_slope) < threshold

    reverse_dict = {}
    def reverse_logic(self, symbol: str, profit: float):
        if symbol not in self.reverse_dict:
            self.reverse_dict[symbol] = {}
        if "start" not in self.reverse_dict[symbol]:
            self.reverse_dict[symbol]["start"] = int(time.time())
        minutes_past = (int(time.time()) - self.reverse_dict[symbol]["start"]) // 60

        # append profit
        if not "profits" in self.reverse_dict[symbol]:
            self.reverse_dict[symbol]["profits"] = TimeBasedDeque(max_age=3600)
        self.reverse_dict[symbol]["profits"].add(profit)

        # get profits from last ten minutes
        profits = self.reverse_dict[symbol]["profits"].get_items_last_x_seconds(600)
        if len(profits) > 60:
            # is profits all negative
            all_negative = all(p < 0 for p in profits)
            if not all_negative:
                return False
            # is earlier profit greater than current profit
            first_profit = profits[0]
            current_profit = profits[-1]
            first_profit_greater = first_profit > current_profit
            if not first_profit_greater:
                return False
            if self.is_linear_decreasing(profits, threshold=0.1):
                logging.info(f"Reversing logic for {symbol}, profits: {profits}")
                return True
        return False

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        is_short = trade.is_short
        is_long = not trade.is_short
        symbol = pair.split("/")[0]

        if self.reverse_logic(pair, current_profit):
            if is_long:
                self.set_sentiment(symbol, "short")
                logging.info(f"Trade is long, but profits are consistently negative. Reverse logic applies. Marking sentiment as short.")
            if is_short:
                self.set_sentiment(symbol, "long")
                logging.info(f"Trade is short, but profits are consistently negative. Reverse logic applies. Marking sentiment as long.")
            return True

#############################################################

    def make_get_request_with_retry(self, url, auth, retries=20, sleep_time=10):
        lock = threading.Lock()
        lock.acquire()
        response = None
        time.sleep(self.spot_pair_delay)
        s = ""
        # Acquire a lock for the label to ensure only one thread processes it
        try:
            attempt = 0
            for attempt in range(0, retries):
                try:
                    time.sleep(self.spot_pair_delay)
                    requests.adapters.HTTPAdapter(pool_maxsize = 500, pool_connections=100, max_retries=10)
                    response = requests.get(url, auth=auth, timeout=120)
                    response.raise_for_status()  # Ensure we notice bad responses
                    s = f"Successful request to endpoint {url}"
                    attempt = 0
                    return response  # Break the loop and return response if successful
                except Exception as ex:
                    s = f"Retries exhausted, giving up on request {url}"
                    self.logme(f"Attempt {attempt + 1} failed. Error: {ex}, {s}", telegram=True)
                    time.sleep(sleep_time)
        except Exception as e:
            s = f"Exception in make request: {e}"
            # self.dp.send_msg(s)
            traceback.print_exc()
            logging.error(s)
        finally:
            lock.release()
            logging.error(s)

    def getJsonFromAPI(self, endpoint):
        url = endpoint
        auth = (self.bot_username, self.bot_password)
        time.sleep(self.spot_pair_delay)
        response = None
        try:
            response = self.make_get_request_with_retry(self, url, auth)
        except Exception as e:
            s = f"self.make_get_request_with_retry exception: {e}"
            logging.error(s, telegram=True)
            traceback.print_exc()

        finally:
            if response is None:
                data = {}
            else:
                data = response.json()
            return data

    def getWinrate(self):
        json = None
        try:
            json = self.getJsonFromAPI(self.api_profit)
        except Exception as e:
            self.logme(f"could not get profit from hedge_bot {self.hedge_bot_api_profit}, continuing")
            traceback.print_exc()

        if json:
            # profit_all_coin = json['profit_all_coin']
            # winning_trades = json['winning_trades']
            # losing_trades = json['losing_trades']
            winrate = json['winrate']
        else:
            raise Exception(f"No Data returned from Profit endpint {api_profit}")
        return winrate

