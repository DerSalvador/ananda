# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import time, traceback, threading
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple
import os
from enum import Enum
import re
from html import unescape
from functools import cache

import logging
freqtradelog = logging.getLogger("freqtrade")
from typing import Any
# from utils.coinGeckoAPI import CoinGeckoAPI
# coinGeckoAPI = CoinGeckoAPI()
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
        # self.cleanup()

    def length(self):
        return len(self.queue)
        
    def last(self):
        if self.length() > 0:
            return self.queue[-1]
        else:
            return None

    def clear(self):
        self.queue = []
        
    def cleanup(self):
        now = time.time()
        while self.queue and (now - self.queue[0][0] > self.max_age):
            self.queue.popleft()

    def get_items_last_x_seconds(self, seconds):
        """Retrieve items from the last X seconds."""
        threshold = time.time() - (seconds)
        return [item for t, item in self.queue ] # if t >= threshold]

class Constants(Enum):
    LINEAR_INCREASING = 1
    LINEAR_DECREASING = 2
    LINEAR_STABLE = 3
    PAIR = 4
    MARKET = 5
    BULLISH = 6
    BEARISH = 7 
    
bias_endpoint = os.getenv("BIAS_ENDPOINT", "")
BIAS_CONFIG_UPDATE_SECONDS = 60

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
    api_profit = None
    bot_username = None
    bot_password = None
    reverse_dict: dict = {}
    macd_short_window = 6
    macd_long_window = 26
    macd_signal_window = 9
    bullish_threshold_pct = 0.7
    bearish_threshold_pct = 0.7
    candle_divisor = 3.0
    bullish_partial_threshold_pct = 0.75
    bearish_partial_threshold_pct = 0.75
    custom_stake = 100
    custom_leverage = 5
    default_stake = 100
    last_trend_entries = 3
    
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = "1m"

    # Can this strategy go short?
    can_short: bool = True

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    # minimal_roi = {
    #     "0": 0.25
    # }
    # disable minimal_roi, will be configurable as well
    minimal_roi = {}
    return_on_invest = 0.1
    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    default_stoploss = -0.12
    stoploss = default_stoploss
    stoploss_win = -0.06
    stoploss_loss = -0.15

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
    banned_minutes: int = 60
    # Strategy parameters
    buy_rsi = IntParameter(10, 40, default=30, space="buy")
    sell_rsi = IntParameter(60, 90, default=70, space="sell")

    def get_roi(self):
        logging.info(f"getting roi in float percent (0.2 = 20%) from bias endpoint")
        # try:
        #     response = requests.get(f"{bias_endpoint}/sentiment/roi")
        #     response.raise_for_status()
        #     market_bias = response.json()
        #     self.return_on_invest = market_bias.get("roi", {}).get("roi")
        # except Exception as e:
        #     s = f"Error getting roi: {e}"
        #     logging.error(s)
        #     raise e
        # finally:
        #     return roi
        return self.return_on_invest

    def get_bias(self, pair, returnFull = False):
        # apikey = self.config['bias']['binance_api_key'] # 
        # apisecret = self.config['bias']['binance_secret_key'] # 
        # market_bias = coinGeckoAPI.get_file_sentiment(None, apikey, apisecret)
        market_bias = {}
        reason = ""
        try:
            response = requests.get(f"{bias_endpoint}/sentiment/{pair}")
            response.raise_for_status()
            market_bias = response.json()
            if returnFull:
                return market_bias, ""
            reason = market_bias.get("final", {}).get("reason", "")
            market_bias = market_bias.get("final", {}).get("bias", "neutral")
        except Exception as e:
            s = f"Error getting market bias: {e}"
            logging.error(s)
            market_bias = "neutral"
            raise e
        finally:
            return market_bias, reason
        return market_bias, f"All endpoints agreed on {market_bias}"
    
    # def get_bias(self, pair):
    #     # apikey = self.config['bias']['binance_api_key'] # 
    #     # apisecret = self.config['bias']['binance_secret_key'] # 
    #     # market_bias = coinGeckoAPI.get_file_sentiment(None, apikey, apisecret)
    #     market_bias = {}
    #     reason = ""
    #     try:
    #         response = requests.get(f"{bias_endpoint}/sentiment/{pair}")
    #         response.raise_for_status()
    #         market_bias = response.json()
    #         reason = market_bias.get("final", {}).get("reason", "")
    #         market_bias = market_bias.get("final", {}).get("bias", "neutral")
    #     except Exception as e:
    #         s = f"Error getting market bias: {e}"
    #         logging.error(s)
    #         market_bias = "neutral"
    #         raise e
    #     finally:
    #         return market_bias, reason
    #     return market_bias, f"All endpoints agreed on {market_bias}"

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

    def get_stake_amount(self):
        return self.default_stake
        # stake_amount = -1
        # try:
        #     response = requests.get(f"{bias_endpoint}/stake_amount")
        #     response.raise_for_status()
        #     stake_amount = response.json().get("stake_amount")
        # except Exception as e:
        #     logging.error(f"Error getting leverage: {e}")
        #     raise e
        # finally:
        #     return stake_amount

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
        pair = metadata['pair']
        try:
            logging.info("Populate entry with dataframe and dict")
            logging.info(dataframe)
            logging.info(metadata)
            # Initialize required columns
            dataframe['enter_long'] = 0       # Required even if not used
            dataframe['enter_short'] = 0
            dataframe['enter_tag'] = None            
            # dataframe.drop(dataframe.index, inplace=True)
            logging.warn("Populate entry with new api")
            symbol = metadata['pair'].split("/")[0]
            market_bias, reason = self.get_bias(symbol)
            org_market_bias = market_bias
            logging.info(f"Check if need to overrule bias={market_bias} after reverse logic has processed in custom_exit")
            logging.info(self.reverse_dict)
            if symbol in self.reverse_dict and "reverse" in self.reverse_dict[symbol]: 
                if self.reverse_dict[symbol]["reverse"] is not None:
                    market_bias = self.reverse_dict[symbol]["reverse"]
                    s = f"{symbol} Overruling market_bias original {org_market_bias} with {market_bias} by reverse logic"
                    logging.info(s)
                    self.sendTelegram(s)
                    reason = s
                else:
                    s = f"reverse logic for {symbol} not overruling market bias {org_market_bias}"
                    logging.info(s)
                    self.sendTelegram(s)
            else:
                logging.info(f"{symbol} not found in reverse_dict")
            logging.info(self.reverse_dict)
                    
            logging.info(f"Market bias is {market_bias} for {symbol}, reason: {reason}")
            dataframe.loc[:, ['enter_long', 'enter_tag']] = (0, "reason")
            dataframe.loc[:, ['enter_short', 'enter_tag']] = (0, "reason")
            if market_bias == "neutral":
                logging.info(f"Market bias is {market_bias} for {symbol}, skipping order.")
                dataframe.loc[:, ['enter_long', 'enter_tag']] = (0, reason)
                dataframe.loc[:, ['enter_short', 'enter_tag']] = (0, reason)
            elif market_bias == "long":
                dataframe.loc[:, ['enter_long', 'enter_tag']] = (1, reason)
                s = f"Market bias is {market_bias} for {symbol}, going long, reason: {reason}"
                logging.info(s)
                # self.sendTelegram(s)
            elif market_bias == "short":
                dataframe.loc[:, ['enter_short', 'enter_tag']] = (1, reason)
                s = f"Market bias is {market_bias} for {symbol}, going short, reason: {reason}"
                logging.info(s)
                # self.sendTelegram(s)
            else:
                raise Exception(f"Market Bias unknown {market_bias}")
        finally:
            if self.is_pair_in_open_trades(pair):
                if symbol in self.reverse_dict and "reverse" in self.reverse_dict[symbol]: 
                    logging.warn(f"{pair} is in open trades, reverse marker has been cleared")
                    self.reverse_dict[symbol]["reverse"] = None
            else:
                logging.warn(f"{pair} is not yet in open trades, reverse marker not cleared")
            
        return dataframe

    def is_pair_in_open_trades(self, pair: str) -> bool:
        """
        Returns True if the given pair is currently in open trades, else False.
        """
        open_trades = Trade.get_trades_proxy(is_open=True)
        logging.info("Open Trades")
        logging.info(open_trades)
        return any(trade.pair == pair for trade in open_trades)

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

        self.custom_leverage = self.get_leverage(pair, proposed_leverage)

        logging.info(f"Using leverage {self.custom_leverage} for {pair}")
        return self.custom_leverage

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """
        logging.warn("Ignore exit, using roi and stoploss")
        return dataframe

    def is_linear_decreasing(self, profits, symbol, threshold=0.05):
        logging.info(f"Checking if profits are linear decreasing {symbol}...")
        # logging.info(profits)
        period = len(profits)
        df = pd.DataFrame(profits, columns=["current_profit"])
        trend = self.detectBullishOrBearishCandle(df)
        jupiterString = df['current_profit'].to_string(header=False, index=False).replace('\n', ',')
        logging.info(f"{symbol} Jupyter Array")
        logging.info(jupiterString)
        self.sendTelegram(jupiterString)
        s = f"{symbol}: Trend for profit array is {trend}..."
        logging.info(s)
        self.sendTelegram(s)
        if trend == Constants.BULLISH or trend == Constants.LINEAR_STABLE:
            return False
        elif trend == Constants.BEARISH:
            return True
        else:
            raise Exception(f"cannot determine trend for {symbol} in is_linear_decreasing")
        # y = np.array(profits, dtype=np.float64)
        # slope = ta.LINEARREG_SLOPE(y, timeperiod=period)

        # if slope is None:
        #     return False

        # latest_slope = slope[-1]

        # return latest_slope < 0 and abs(latest_slope) < threshold

    # reverse_dict = {}
    def addProfit(self, symbol, profit: float):
        o = ""
        if symbol not in self.reverse_dict:
            self.reverse_dict[symbol] = {}
        # append profit
        if not "profits" in self.reverse_dict[symbol]:
            self.reverse_dict[symbol]["profits"] = TimeBasedDeque(max_age=600)
        profits = self.reverse_dict[symbol]["profits"].get_items_last_x_seconds(600)
        o = ', '.join([str(x) for x in profits[:3]])
        output = o + f"..., whole length: {len(profits)}"
        logging.info(output)
        self.sendTelegram(output)
        if len(profits) >= 3:
            trend = self.is_linear_decreasing(profits, symbol, threshold=0.1)
            s = f"{symbol}: current_profit Trend for {symbol} is {trend} with length {len(profits)}"
            logging.info(s)
            self.sendTelegram(s)
        if "start" not in self.reverse_dict[symbol]:
            self.reverse_dict[symbol]["start"] = int(time.time())
        minutes_past = (int(time.time()) - self.reverse_dict[symbol]["start"]) // 60
        
        # self.reverse_dict[symbol]["profits"].add(profit)
        profits_deque = self.reverse_dict[symbol]["profits"]
        last_profit = profits_deque.last()
        
        # Only add if different from the last one
        if (last_profit == None or last_profit != profit):
            profits_deque.add(profit)
        # get profits from last ten minutes
        logging.info(f"{symbol}: Profits array length: {len(profits)}")
        # logging.info(profits[:3])
        logging.info(f"{symbol}: Profits array length: {len(profits)}, minutes_past={minutes_past}")

    def reverse_logic(self, symbol: str):
        profitArrCount = 480
        minutesPastLimit = 240
        try:
            logging.info(f"Entering reverse logic {symbol}")
            profits = self.reverse_dict[symbol]["profits"].get_items_last_x_seconds(600)
            minutes_past = (int(time.time()) - self.reverse_dict[symbol]["start"]) // 60
            if len(profits) > profitArrCount or minutes_past > minutesPastLimit:
                deque = self.reverse_dict[symbol]["profits"]
                logging.info(f"{symbol}: Profits array length: {len(profits)} is greater 60 or minutes_past={minutes_past} > 20")
                self.reverse_dict[symbol]["start"] = int(time.time())
                # is profits all negative
                all_negative = all(p < 0 for p in profits)
                if not all_negative:
                    s = f"{symbol}: Profits array not all negative, not reversing"
                    logging.info(s)
                    self.sendTelegram(s)
                    logging.info(f"Leaving reverse logic {symbol}")
                    return False
                else:
                    s = f"{symbol}: Profits array all negative, reversing"
                    logging.info(s)
                    self.sendTelegram(s)
                p = self.reverse_dict[symbol]["profits"].get_items_last_x_seconds(20 * 60)
                output = ', '.join([str(x) for x in p[:3]])
                logging.info(output)
                output += f"..., whole length: {len(p)}"
                self.sendTelegram(output)
                # disabled for now, resp. make it configurable 
                # is earlier profit greater than current profit
                first_profit = profits[0]
                last_profit = profits[-1]
                first_profit_greater = first_profit > last_profit
                if not first_profit_greater:
                    logging.info(f"{symbol}: first_profit {first_profit} is greater current_profit {last_profit}, not reversing")
                    return False
                if self.is_linear_decreasing(profits, symbol, threshold=0.1):
                    logging.info(f"Profits are negative and decreasing, Reversing logic for {symbol}")
                    # logging.info(profits)
                    return True
        except Exception as e:
            s = f"Exception in reverse_logic {e}"
            logging.error(s)
            raise Exception(s)
        finally:
            if len(profits) > profitArrCount or minutes_past > minutesPastLimit:
                self.reverse_dict[symbol]["profits"].clear()

        return False

    def reverse_direction(self, is_long, symbol, is_short, trade):
        s = None
        if "count" not in self.reverse_dict[symbol]:
            self.reverse_dict[symbol]["count"] = 0
            logging.warn(f"Set Reverse count {symbol} = {self.reverse_dict[symbol]['count']}")
        if self.reverse_dict[symbol]["count"] < 3: 
            self.reverse_dict[symbol]["count"] += 1
            logging.warn(f"Reverse count {symbol} = {self.reverse_dict[symbol]['count']}")
        else:
            s = f"Max reverse count reached, banning {symbol} for {self.banned_minutes}min"
            logging.warn(s)
            if "bannedAt" not in self.reverse_dict[symbol] or self.reverse_dict[symbol]["bannedAt"] == None:
                self.reverse_dict[symbol]["bannedAt"] = int(time.time())
        if not self.isBanned(symbol):
            if is_long:
                self.set_sentiment(symbol, "short")
                self.reverse_dict[symbol]["reverse"] = "short"
                s = f"{symbol}: Trade is long, but profits are consistently negative or stoploss reached. Reverse logic applies. Marking sentiment as short."
                logging.info(s)
                self.sendTelegram(s)
            
            elif is_short:
                self.set_sentiment(symbol, "long")
                self.reverse_dict[symbol]["reverse"] = "long"
                s = f"{symbol}: Trade is short, but profits are consistently negative or stoploss reached. Reverse logic applies. Marking sentiment as long."
                logging.info(s)
                self.sendTelegram(s)            
            else:
                raise Exception(f"Trade has no direction {trade}")
        else:
            s = f"{symbol} is still banned ({self.banned_minutes}min), not reversing although all conditions met"
            logging.warn(s)
        return s

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        is_short = trade.is_short
        is_long = not trade.is_short
        symbol = pair.split("/")[0]
        self.addProfit(symbol, current_profit)
        logging.info(f"Entering custom_exit with {symbol} is_short={is_short} is_long={is_long}, current_profit={current_profit}, self.stoploss={self.stoploss}")
        if current_profit >= self.return_on_invest:
            logging.info(f"current_profit={current_profit} is greater or equal {self.return_on_invest}, realizing profit ")
            return f"profit_{self.return_on_invest}_for_{symbol}"
        # self.adjust_stoploss()        
        # leverage = trade.leverage
        # stake_amount = trade.stake_amount
        # current_profit_pct = current_profit * 100 / (leverage * stake_amount)
        freqtradelog.info(f"current_profit={current_profit}, self.stoploss={self.stoploss}")
        logging.info(f"current_profit={current_profit}, self.stoploss={self.stoploss}")
        if current_profit <= self.stoploss:
            self.reverse_direction(is_long, symbol, is_short, trade)     
            s = f"Stoploss_reached_{self.stoploss}_current_profit_{current_profit}"
            logging.info(s)
            self.sendTelegram(s)
            return s
            
        # check if banned from reverse logic
        if symbol in self.reverse_dict:
            self.reverse_dict.setdefault(symbol, {})["reverse"] = None
            # logging.info(f"Check if {symbol} is banned from reverse logic")
            # if "bannedAt" in self.reverse_dict[symbol]:
            #     minutes_past = (int(time.time()) - self.reverse_dict[symbol]["bannedAt"]) // 60
            #     if minutes_past >= self.banned_minutes: 
            #         s = f"Ban for {symbol} is over, trying to reverse if necessary"
            #         logging.info(s)
            #         self.sendTelegram(s)
            #         self.reverse_dict[symbol]["bannedAt"] = None
            #         self.reverse_dict[symbol]["count"] = 0
            #     else:
            #         s = f"Ban for  {symbol}  reverse logic still active"
            #         logging.info(s)
            #         self.sendTelegram(s)
            #         return False
                        
            # check count of reverses to ban symbol
            logging.info(f"{symbol} Check Count of reverses")        
            profits = []
            if "profits" in self.reverse_dict[symbol]:
                profits = self.reverse_dict[symbol]["profits"].get_items_last_x_seconds(600)        
            if self.reverse_logic(symbol):
                self.reverse_direction(is_long, symbol, is_short, trade)
                direction = not trade.is_short   
                return f"{symbol}_is_short={trade.is_short}_overruled_by_reverse_logic_going_is_short={direction}"
        else:
            s = f"{symbol} not yet in reverse_dict"
            logging.warn(s)
            self.sendTelegram(s)
        return False

    def isBanned(self, symbol):
        logging.info(f"Check if {symbol} is banned from reverse logic")
        if "bannedAt" in self.reverse_dict[symbol] and self.reverse_dict[symbol]["bannedAt"] != None:
            minutes_past = (int(time.time()) - self.reverse_dict[symbol]["bannedAt"]) // 60
            if minutes_past >= self.banned_minutes: 
                s = f"Reverse Logic Ban for {symbol} is over, trying to reverse if necessary"
                logging.info(s)
                self.sendTelegram(s)
                self.reverse_dict[symbol]["bannedAt"] = None
                self.reverse_dict[symbol]["count"] = 0
            else:
                s = f"Reverse Logic Ban for  {symbol}  reverse logic still active"
                logging.info(s)
                self.sendTelegram(s)
                return True
        else:
            s = f"BannedAt for  {symbol} not found in reverse_dict of is None"
            logging.info(s)
            
        return False
        
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, after_fill: bool, **kwargs: Any) -> float:       
        logging.info(f"Entering custom_stoploss with {pair} current_profit={current_profit}, self.stoploss={self.stoploss}")
        return self.default_stoploss

    def summarize_biases(self, response_json):
        summary = []
        for source, data in response_json.items():
            if source == "final":
                continue  # Skip the 'final' key
            bias = data.get("bias", "unknown")
            error = data.get("error")
            reason = data.get("reason", "")
            
            details = f"{source}={bias}"
            if error:
                details += f" ({error})"
            elif reason:
                details += f" ({reason})"
            
            summary.append(details)
    
        return " ".join(summary)
        
    def bot_start(self, **kwargs) -> None:
        market_bias, reason = self.get_bias("BTC")
        s = f"bot_start: Market bias is {market_bias}"
        logging.info(s)
        self.sendTelegram(s)
        market_bias_full, _ = self.get_bias("BTC", returnFull=True)
        logging.info(market_bias_full)
        reason_string = self.summarize_biases(market_bias_full)
        self.sendTelegram(reason_string)

        self.biasConfig = self.get_config()

        self.api_profit = self.config['bias']['api_profit']
        self.bot_username =  self.config['api_server']['username']
        self.bot_password =  self.config['api_server']['password']
        freqtradelog.info(f"self.api_profit {self.api_profit}")
        logging.info(f"Bot or bot loop started...")

    # def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
    #     self.bot_start()
#########################
# Protections
# See https://www.freqtrade.io/en/stable/plugins/#full-example-of-protections
    @property
    def protections(self):
        return []
            # {
            #     "method": "CooldownPeriod",
            #     # "stop_duration_candles": 1
            #     "stop_duration": 1 # stop_duration in minutes
            # },
            # {
            #     "method": "MaxDrawdown",
            #     "lookback_period_candles": 48,
            #     "trade_limit": 20,
            #     # "stop_duration_candles": 20,
            #     "stop_duration": 120, # stop_duration in minutes
            #     "max_allowed_drawdown": 0.1
            # },
            # {
            #     "method": "StoplossGuard",
            #     "lookback_period_candles": 20,
            #     "trade_limit": 4,
            #     # "stop_duration_candles": 2,
            #     "stop_duration": 120,
            #     "only_per_pair": True
            # },
            # {
            #     "method": "LowProfitPairs",
            #     "lookback_period_candles": 60,
            #     "trade_limit": 2,
            #     # "stop_duration_candles": 60,
            #     "stop_duration": 120,
            #     "required_profit": 0.01
            # },
            # {
            #     "method": "LowProfitPairs",
            #     "lookback_period_candles": 60,
            #     "trade_limit": 4,
            #     # "stop_duration_candles": 2,
            #     "stop_duration": 20,
            #     "required_profit": 0.01
            # }
        # ]
#############################################################
# Winrate
    def adjust_stoploss(self):
        win_rate = self.getWinrate()
        # Adjust stop-loss based on win/loss performance
        if win_rate > 0.7:  # If win rate is high, tighten stop-loss
            s = f"Adjusting stoploss to -0.12 because winrate {win_rate} is higher than 0.7"
            logging.info(s)
            self.sendTelegram(s)
            self.stoploss = self.stoploss_win
            return self.stoploss  # Reduce stop-loss (secure profits)
        elif win_rate < 0.3:  # If losing a lot, loosen stop-loss
            s = f"Adjusting stoploss to -0.18 because winrate {win_rate} is lower than 0.3"
            logging.info(s)
            self.sendTelegram(s)
            self.stoploss = self.stoploss_loss
            return self.stoploss  # Allow more room for trade recovery
        else:
            logging.info(f"Adjusting stoploss to default {self.default_stoploss}")
            self.stoploss = self.default_stoploss
            return self.default_stoploss  # Keep default stop-loss
                
    def make_get_request_with_retry(self, url, auth, retries=20, sleep_time=10):
        # lock = threading.Lock()
        # lock.acquire()
        response = None
        s = ""
        # Acquire a lock for the label to ensure only one thread processes it
        try:
            attempt = 0
            for attempt in range(0, retries):
                try:
                    requests.adapters.HTTPAdapter(pool_maxsize = 500, pool_connections=100, max_retries=10)
                    response = requests.get(url, auth=auth, timeout=120)
                    response.raise_for_status()  # Ensure we notice bad responses
                    s = f"Successful request to endpoint {url}"
                    attempt = 0
                    return response  # Break the loop and return response if successful
                except Exception as ex:
                    s = f"Retries exhausted, giving up on request {url}"
                    logging.error(f"Attempt {attempt + 1} failed. Error: {ex}, {s}")
        except Exception as e:
            s = f"Exception in make request: {e}"
            # self.sendTelegram(s)
            traceback.print_exc()
            logging.error(s)
        finally:
            # lock.release()
            logging.info(s)

    def getJsonFromAPI(self, endpoint):
        url = endpoint
        auth = (self.bot_username, self.bot_password)
        response = None
        try:
            response = self.make_get_request_with_retry(url, auth)
        except Exception as e:
            s = f"self.make_get_request_with_retry exception: {e}"
            logging.error(s)
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
            logging.error(f"could not get profit from hedge_bot {self.api_profit}, continuing")
            traceback.print_exc()

        if json:
            # profit_all_coin = json['profit_all_coin']
            # winning_trades = json['winning_trades']
            # losing_trades = json['losing_trades']
            winrate = json['winrate']
        else:
            raise Exception(f"No Data returned from Profit endpint {self.api_profit}")
        return winrate

    def populateTrend(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            short_window=self.macd_short_window # 6
            long_window=self.macd_long_window # 26
            signal_window=self.macd_signal_window # 9
            # Calculate the short term exponential moving average (EMA)
            df['EMA_short'] = df['current_profit'].ewm(span=short_window, adjust=True).mean()   
            # Calculate the long term exponential moving average (EMA)
            df['EMA_long'] = df['current_profit'].ewm(span=long_window, adjust=True).mean()    
            # Calculate the MACD line
            df['MACD'] = df['EMA_short'] - df['EMA_long']   
            # Calculate the Signal line
            df['Signal_Line'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()   
            # Determine the trend
            df['Trend'] = df.apply(lambda row: 'Bullish' if row['MACD'] > row['Signal_Line'] else 'Bearish', axis=1)
        except Exception as e:
            logging.info(f"Exception in populateTrend {e}")
            raise e
        return df
    
    def detectBullishOrBearishCandle(self, df: pd.DataFrame) -> Constants:
        logging.info(f"Start Detect Overall Trend")
        if df.empty is True:
            logging.info("Warning: Dataframe is empty in detectBullishOrBearishCandle")
            return Constants.LINEAR_STABLE   
        logging.info(df)        
        df = self.populateTrend(df)
        bullish_count = df['Trend'].value_counts().get('Bullish', 0)
        bearish_count = df['Trend'].value_counts().get('Bearish', 0)
        total_rows = len(df)
        if bullish_count / total_rows >= self.bullish_threshold_pct:
            overall_result = Constants.BULLISH
        elif bearish_count / total_rows >= self.bearish_threshold_pct:
            overall_result = Constants.BEARISH
        else:
            overall_result = Constants.LINEAR_STABLE
        logging.info(f"Overall Result (whole dataframe) Trend: {overall_result}")
        partial_length = round(float(total_rows) // self.candle_divisor)
        if partial_length > 0 and len(df) > partial_length:
            last_partial_df = df[-partial_length:]
            first_partial_df = df[:-partial_length]
            first_partial_result = self.determineTrend(overall_result, partial_length, first_partial_df)
            last_partial_result = self.determineTrend(overall_result, partial_length, last_partial_df)
            logging.info(first_partial_df)
            logging.info(f"first_partial_result={first_partial_result}")
            logging.info(last_partial_df)
            logging.info(f"last_partial_result={last_partial_result}")
            # Check the last 3 rows
            if self.last_trend_entries > len(df):
                logging.info(f"Last trends entries {self.last_trend_entries} is greater than candle length {len(df)} adjusting to candle length")            
                self.last_trend_entries = len(df)
            last_trends = df['Trend'].tail(self.last_trend_entries)
            if all(last_trends == 'Bullish') and overall_result == Constants.BULLISH and last_partial_result == Constants.BULLISH and first_partial_result == Constants.BULLISH:
                overall_result = Constants.BULLISH
            elif all(last_trends == 'Bearish') and overall_result == Constants.BEARISH and last_partial_result == Constants.BEARISH and first_partial_result == Constants.BEARISH:
                overall_result = Constants.BEARISH
            else:
                overall_result = Constants.LINEAR_STABLE
            logging.info(f"End Detect Overall Trend for Dataframe: overall_result={overall_result}")        
            logging.info(f"all(last_trends)={all(last_trends == 'Bullish')}, overall_result={overall_result}, last_partial_result=last_partial_result={last_partial_result}")
            return overall_result    
        else:
            logging.info(f"partial_length={partial_length} > 0 and len(df)={len(df)} > partial_length={partial_length} is false, no detectBullishOrBearishCandle")
            return Constants.LINEAR_STABLE 

    def determineTrend(self, overall_result, partial_length, df):
        bullish_count = df['Trend'].value_counts().get('Bullish', 0)
        bearish_count = df['Trend'].value_counts().get('Bearish', 0)
        if bullish_count / partial_length >= self.bullish_partial_threshold_pct or overall_result == Constants.BULLISH:
            return Constants.BULLISH
        elif bearish_count / partial_length >= self.bearish_partial_threshold_pct or overall_result == Constants.BEARISH:
            return Constants.BEARISH
        else:
            return Constants.LINEAR_STABLE

    def strip_formatting(self, text):
        # Remove Markdown (basic)
        text = re.sub(r'(\*|_|~|`|\[.*?\]\(.*?\))', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Unescape HTML entities like &amp; -> &
        text = unescape(text)
        
        return text

    def sendTelegram(self, msg: str):
        try:
            truncated = msg[:160] if len(msg) > 160 else msg
            if len(msg) < 160:
                truncated = self.strip_formatting(truncated)
                time.sleep(3)
                self.dp.send_msg(truncated)
            else:
                logging.info(f"Message not send to telegram, greater 160: {msg}...")            
        except Exception:
            logging.warn(f"msg not send to telegram: {msg}")

    @cache
    def get_config_internal(self, cache_time):
        response = requests.get(f"{bias_endpoint}/configs")
        response.raise_for_status()
        config_json = response.json()
        config_dict = {item['name']: item['value'] for item in config_json}
        return config_dict

    def get_config(self):
        cache_key = int(time.time() // BIAS_CONFIG_UPDATE_SECONDS)
        return self.get_config_internal(cache_key)
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                                proposed_stake: float, min_stake: float | None, max_stake: float,
                                leverage: float, entry_tag: str | None, side: str,
                                **kwargs) -> float:
        # win_rate = self.calculate_winrate()
        config = self.get_config()
        # win_rate_high = float(config.get("WinrateHigh", 0.7))
        # win_rate_low = float(config.get("WinrateLow", 0.3))
        # max_stake = float(config.get("MaxStake", 1000))
        # min_stake = float(config.get("MinStake", 10))
        self.custom_stake = default_stake = float(config.get("DefaultStake", 100))
        logging.info(f"Setting default stake to {self.custom_stake}")
        # stake_increment_step = float(config.get("StakeIncrementStep", 100))
        # if win_rate >= win_rate_high and self.custom_stake < max_stake:
        #     self.custom_stake += stake_increment_step
        #     logging.info(f"Winrate is high, increasing stake to {self.custom_stake}")
        #     self.dp.send_msg(f"Winrate is high, increasing stake to {self.custom_stake}")
        # elif win_rate <= win_rate_low and self.custom_stake > min_stake:
        #     self.custom_stake -= stake_increment_step
        #     logging.info(f"Winrate is low, decreasing stake to {self.custom_stake}")
        #     self.dp.send_msg(f"Winrate is low, decreasing stake to {self.custom_stake}")
        # elif win_rate > win_rate_low and win_rate < win_rate_high and self.custom_stake != default_stake:
        #     self.custom_stake = default_stake
        #     logging.info(f"Winrate is neutral, using default stake {self.custom_stake}")
        #     self.dp.send_msg(f"Winrate is neutral, using default stake {self.custom_stake}")
        # self.custom_stake = min(self.custom_stake, max_stake)
        # self.custom_stake = max(self.custom_stake, min_stake)
        return self.custom_stake        
        # win_rate = self.getWinrate()
        # # Adjust custom_stake and leverage based on win/loss performance
        # if win_rate > 0.7:  # If win rate is high, tighten increase stake and leverage
        #     if self.custom_stake <= 1000:
        #         self.custom_stake += 100
        #         self.custom_leverage += 1
        #         s = f"Adjusting custom_stake to {self.custom_stake} and custom_leverage {self.custom_leverage}"
        #         logging.info(s)
        #         self.sendTelegram(s)
        #     return self.custom_stake  # Reduce stop-loss (secure profits)
        # elif win_rate < 0.3:  # If losing a lot, decrease stake and leverge
        #     s = f"Adjusting custom_stake to {self.custom_stake} and custom_leverage {self.custom_leverage}"
        #     logging.info(s)
        #     self.sendTelegram(s)
        #     if self.custom_stake >= 200:
        #         self.custom_stake -= 100
        #         if self.custom_leverage > 5:
        #             self.custom_leverage -= 1
        #         s = f"Adjusting custom_stake to {self.custom_stake} and custom_leverage {self.custom_leverage}"
        #         logging.info(s)
        #         self.sendTelegram(s)
        #     return self.custom_stake  # Allow more room for trade recovery
        # else:
        #     logging.info(f"Adjusting custom_stake to default {self.default_stake}")
        #     self.custom_stake = self.get_stake_amount()
        #     # Use default stake amount.
        #     return self.custom_stake
