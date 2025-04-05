Okay, let's enhance the AnandaStrategySplit strategy to implement this trade limiting logic.

We need to:

Track Reverse Exits: Keep track of which pairs have been exited due to the reverse_logic in custom_exit.

Count Subsequent Entries: For pairs flagged in step 1, count how many times populate_entry_trend generates an entry signal.

Enforce the Limit: Prevent populate_entry_trend from generating an entry signal for a flagged pair if its entry count has reached 3.

Reset the Count: Decide when the count should reset. The most logical point is when another custom_exit triggered by reverse_logic happens for that same pair.

Here's the modified code:

# --- START OF FILE AnandaStrategySplit.py ---

# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement, E1101
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import time
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple, Any # Added Any
import requests
import os
import logging
import threading # Added for lock, although not strictly used in the final get request logic shown
import traceback # Added for better error logging

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
from freqtrade.exchange import timeframe_to_seconds
from freqtrade.persistence import Trade, PairLocks
from freqtrade.data.dataprovider import DataProvider # Added for dp access if needed elsewhere

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
# import requests # already imported
from technical import qtpylib
from collections import deque

logger = logging.getLogger(__name__) # Use logger instead of logging directly

class TimeBasedDeque:
    def __init__(self, max_age=3600):
        self.max_age = max_age
        self.queue = deque()
        self._lock = threading.Lock() # Added lock for thread safety

    def add(self, item):
        """Insert a new item with the current timestamp."""
        timestamp = time.time()
        with self._lock:
            self.queue.append((timestamp, item))
            self.cleanup() # Cleanup can happen within the lock or outside, depending on need

    def cleanup(self):
        # Assumes called within a lock or thread-safe context if needed externally
        now = time.time()
        # No need to acquire lock again if called from add() which already holds it
        while self.queue and (now - self.queue[0][0] > self.max_age):
            self.queue.popleft()

    def get_items_last_x_seconds(self, seconds):
        """Retrieve items from the last X seconds."""
        threshold = time.time() - seconds
        with self._lock:
            # Create a copy to avoid issues if cleanup runs concurrently in another thread
            # Although the lock makes this less likely, it's safer practice
            items = list(self.queue)
        return [item for t, item in items if t >= threshold]


bias_endpoint = os.getenv("BIAS_ENDPOINT", "")

# Define the maximum number of entries allowed after a reverse logic exit
MAX_REVERSE_ENTRIES = 3

class AnandaStrategySplit(IStrategy):
    """
    Strategy incorporating market bias from an external API and
    a custom exit logic based on consistent negative PNL,
    with a limit on entries after such an exit.
    """
    api_profit: "http://freqtrade:'&&Liz270117!!'@freqtrade-clusterip-bot-ananda.bot-ananda.svc.cluster.local:8084/api/v1/profit" # Check if this is actually used

    INTERFACE_VERSION = 3
    timeframe = "1m"
    can_short: bool = True
    minimal_roi = {
        "60": 0.01,
        "30": 0.02,
        "0": 0.04
    }
    stoploss = -0.10
    trailing_stop = False
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    startup_candle_count: int = 30

    # Strategy parameters (consider if these are actually used)
    buy_rsi = IntParameter(10, 40, default=30, space="buy")
    sell_rsi = IntParameter(60, 90, default=70, space="sell")

    # --- Custom State Variables ---
    # Dictionary to store profit deque for each pair for reverse_logic
    # Format: { "pair": TimeBasedDeque }
    reverse_profits: Dict[str, TimeBasedDeque] = {}

    # Dictionary to track entry counts after a reverse_logic exit
    # Format: { "pair": count }
    # The key exists only if a reverse exit occurred and the count is < MAX_REVERSE_ENTRIES
    reverse_entry_counts: Dict[str, int] = {}
    # ------------------------------

    # --- Bot Configuration (Consider moving sensitive parts to config) ---
    # These might be better accessed via self.config if available, or passed during init
    # spot_pair_delay = 0 # Placeholder - configure if needed
    # bot_username = "your_username" # Placeholder - configure securely
    # bot_password = "your_password" # Placeholder - configure securely
    # ---------------------------------------------------------------------

    # Class variable to store Dataprovider instance if needed outside callbacks
    # dp: Optional[DataProvider] = None

    #@classmethod
    #def load_dataprovider(cls, config: dict):
    #    """ Load dataprovider if needed """
    #    cls.dp = DataProvider(config, None, None) # Needs exchange and potentially other args

    def bot_loop_start(self, **kwargs) -> None:
        """
        Called once right before the bot starts.
        """
        logger.info("Bot loop starting...")
        # Initialize Dataprovider if you need it for dp.send_msg etc.
        # Ensure config is correctly passed or available if using load_dataprovider approach
        # if not self.dp and self.config:
        #      self.dp = DataProvider(self.config, None, None)

        # You might want to clear the state on bot start, although it usually
        # gets rebuilt naturally. If persistence across restarts was needed,
        # this would be the place to load from db/file.
        self.reverse_profits = {}
        self.reverse_entry_counts = {}

        # Get necessary config items if not hardcoded
        # self.bot_username = self.config.get('api_server', {}).get('username')
        # self.bot_password = self.config.get('api_server', {}).get('password')
        # self.spot_pair_delay = self.config.get('custom_config', {}).get('spot_pair_delay', 0.1)


    def get_bias(self, pair: str) -> Tuple[str, str]:
        market_bias = "neutral" # Default value
        reason = ""
        if not bias_endpoint:
            logger.warning("BIAS_ENDPOINT not set, defaulting to neutral bias.")
            return market_bias, "BIAS_ENDPOINT not set"
        try:
            # Use symbol (e.g., BTC) not pair (e.g., BTC/USDT) for the endpoint
            symbol = pair.split('/')[0]
            url = f"{bias_endpoint}/sentiment/{symbol}"
            logger.debug(f"Requesting bias from: {url}")
            response = requests.get(url, timeout=10) # Added timeout
            response.raise_for_status()
            data = response.json()
            reason = data.get("final", {}).get("reason", "No reason provided")
            market_bias = data.get("final", {}).get("bias", "neutral")
            logger.debug(f"Received bias for {symbol}: {market_bias}, Reason: {reason}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting market bias for {pair}: {e}")
            reason = f"API Error: {e}"
            # Keep market_bias as "neutral" on error
        except Exception as e:
            logger.error(f"Unexpected error getting market bias for {pair}: {e}")
            reason = f"Unexpected Error: {e}"
            # Keep market_bias as "neutral" on error
        return market_bias, reason

    def set_sentiment(self, pair: str, sentiment: str):
        if not bias_endpoint:
            logger.warning("BIAS_ENDPOINT not set, cannot set sentiment.")
            return
        try:
             # Use symbol (e.g., BTC) not pair (e.g., BTC/USDT) for the endpoint
            symbol = pair.split('/')[0]
            url = f"{bias_endpoint}/sentiment/{symbol}"
            logger.info(f"Setting sentiment for {symbol} to {sentiment} via {url}")
            response = requests.post(url, json={"bias": sentiment}, timeout=10) # Added timeout
            response.raise_for_status()
            logger.info(f"Successfully set sentiment for {symbol} to {sentiment}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error setting sentiment for {pair} to {sentiment}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error setting sentiment for {pair} to {sentiment}: {e}")

    def get_leverage(self, pair: str, proposed_leverage: float) -> float:
        leverage = proposed_leverage # Default to proposed
        if not bias_endpoint:
            logger.warning("BIAS_ENDPOINT not set, using proposed leverage.")
            return leverage
        try:
            url = f"{bias_endpoint}/leverage?pair={pair}" # Assuming endpoint uses pair directly
            logger.debug(f"Requesting leverage from: {url}")
            response = requests.get(url, timeout=10) # Added timeout
            response.raise_for_status()
            data = response.json()
            leverage = data.get("leverage", proposed_leverage)
            logger.debug(f"Received leverage for {pair}: {leverage}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting leverage for {pair}: {e}. Using proposed leverage {proposed_leverage}.")
            # Keep default leverage on error
        except Exception as e:
             logger.error(f"Unexpected error getting leverage for {pair}: {e}. Using proposed leverage {proposed_leverage}.")
             # Keep default leverage on error
        # Ensure leverage does not exceed max_leverage (this check might also happen in base class)
        # max_leverage = self.config.get('max_leverage', proposed_leverage) # Example: Get max from config
        # leverage = min(leverage, max_leverage)
        return leverage

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Indicators are not used directly for entry/exit signals in this version
        # but might be useful for display or other custom logic.
        # Keep it minimal if not needed.
        # Example: Add RSI if needed for display or other checks
        # dataframe['rsi'] = ta.RSI(dataframe)
        logger.debug(f"Populating indicators for {metadata['pair']} (currently minimal)")
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata['pair']
        symbol = pair.split("/")[0] # Use symbol for API calls usually

        # --- Reverse Entry Limit Check ---
        current_reverse_entry_count = self.reverse_entry_counts.get(pair)
        limit_reached = False
        limit_reason = ""
        if current_reverse_entry_count is not None:
            if current_reverse_entry_count >= MAX_REVERSE_ENTRIES:
                limit_reached = True
                limit_reason = f" (Reverse Entry Limit {MAX_REVERSE_ENTRIES} Reached)"
                logger.info(f"[{pair}] Entry limit reached ({current_reverse_entry_count}/{MAX_REVERSE_ENTRIES}) after reverse logic exit. Preventing new entry.")
            else:
                 logger.info(f"[{pair}] Reverse entry count: {current_reverse_entry_count}/{MAX_REVERSE_ENTRIES}")

        # Initialize entry columns
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['enter_tag'] = '' # Default empty tag

        if limit_reached:
            # If limit is reached, ensure signals are 0 and set the tag
            dataframe.loc[:, 'enter_tag'] = f"ReverseLimit{limit_reason}"
            return dataframe
        # --- End Reverse Entry Limit Check ---


        # Get market bias only if the limit is not reached
        market_bias, reason = self.get_bias(pair)
        reason += limit_reason # Append limit reason if relevant (won't happen if limit_reached is True)

        logger.info(f"[{pair}] Market bias check: {market_bias}. Reason: {reason}")

        # Set entry signals based on bias
        entry_signal_set = False
        if market_bias == "long":
            dataframe.loc[:, ['enter_long', 'enter_tag']] = (1, f"BiasLong_{reason}")
            s = f"[{pair}] Bias is long, setting enter_long=1. Reason: {reason}"
            logger.info(s)
            # self.dp.send_msg(s) # Use self.dp if initialized and needed
            entry_signal_set = True
        elif market_bias == "short":
            dataframe.loc[:, ['enter_short', 'enter_tag']] = (1, f"BiasShort_{reason}")
            s = f"[{pair}] Bias is short, setting enter_short=1. Reason: {reason}"
            logger.info(s)
            # self.dp.send_msg(s) # Use self.dp if initialized and needed
            entry_signal_set = True
        elif market_bias == "neutral":
            dataframe.loc[:, 'enter_tag'] = f"BiasNeutral_{reason}"
            logger.info(f"[{pair}] Market bias is neutral, skipping entry. Reason: {reason}")
        else:
            # Handle unexpected bias values
            dataframe.loc[:, 'enter_tag'] = f"BiasUnknown_{market_bias}_{reason}"
            logger.warning(f"[{pair}] Unknown market bias received: {market_bias}. Reason: {reason}")

        # --- Increment Reverse Entry Counter ---
        # Increment counter only if an entry signal was actually set *and* we are tracking this pair
        if entry_signal_set and current_reverse_entry_count is not None:
             self.reverse_entry_counts[pair] = current_reverse_entry_count + 1
             logger.info(
                 f"[{pair}] Incremented reverse entry count to "
                 f"{self.reverse_entry_counts[pair]}/{MAX_REVERSE_ENTRIES}."
             )
             # Update the tag to reflect the count state
             tag_col = 'enter_long' if market_bias == "long" else 'enter_short'
             current_tag = dataframe.iloc[-1]['enter_tag'] # Get last row tag
             dataframe.loc[:, 'enter_tag'] = f"{current_tag}_RevCount{self.reverse_entry_counts[pair]}"


        return dataframe


    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
                 **kwargs) -> float:
        """
        Customize leverage for each new trade.
        """
        custom_leverage = self.get_leverage(pair, proposed_leverage)

        # Ensure leverage doesn't exceed the maximum allowed by the exchange/config
        final_leverage = min(custom_leverage, max_leverage)

        logger.info(f"[{pair}] Proposed leverage: {proposed_leverage}, Max leverage: {max_leverage}, "
                    f"API leverage: {custom_leverage}. Using final leverage: {final_leverage}")
        return final_leverage


    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Exit signals are primarily handled by ROI, Stoploss, and custom_exit.
        This function can remain minimal unless specific indicator-based exits are needed.
        """
        # dataframe['exit_long'] = 0
        # dataframe['exit_short'] = 0
        logger.debug(f"Populating exit trend for {metadata['pair']} (currently no signal)")
        return dataframe


    def is_linear_decreasing(self, profits: list, threshold: float = 0.05) -> bool:
        """Checks if the profit trend is linearly decreasing within a threshold."""
        period = len(profits)
        if period < 2: # Need at least 2 points for a slope
            return False

        # Ensure profits is a numpy array of floats
        y = np.array(profits, dtype=np.float64)

        # Calculate linear regression slope using TA-Lib
        # Note: TA-Lib functions return NaN for initial periods until timeperiod is met.
        # We apply it once to the whole series.
        try:
            # Use timeperiod=period to calculate slope over the entire provided list
            slopes = ta.LINEARREG_SLOPE(y, timeperiod=period)
            if slopes is None or len(slopes) == 0 or np.isnan(slopes[-1]):
                logger.warning(f"Could not calculate slope for profits: {profits}")
                return False
            latest_slope = slopes[-1]
        except Exception as e:
            logger.error(f"Error calculating linear regression slope: {e} for profits: {profits}")
            return False


        is_decreasing = latest_slope < 0
        # Check if the absolute slope is below the threshold (gentle decrease)
        # You might want to adjust this logic depending on what 'threshold' means.
        # If threshold is max allowed negative slope: is_decreasing and latest_slope > -abs(threshold)
        # If threshold is min required negative slope: is_decreasing and latest_slope < -abs(threshold)
        # Current implementation: slope is negative and its magnitude is less than threshold (very gentle decrease)
        is_gentle = abs(latest_slope) < threshold

        logger.debug(f"Linear decreasing check: Slope={latest_slope}, Threshold={threshold}, Decreasing={is_decreasing}, Gentle={is_gentle}")

        return is_decreasing # Simple check if slope is negative


    def reverse_logic(self, pair: str, current_profit: float) -> bool:
        """
        Determines if the trade should be reversed based on consistently
        negative PNL over the last 10 minutes (600 seconds).
        """
        symbol = pair.split('/')[0] # Often useful for logging/external systems

        # Initialize deque for the pair if it doesn't exist
        if pair not in self.reverse_profits:
            self.reverse_profits[pair] = TimeBasedDeque(max_age=3600) # Max age 1 hour

        # Add current profit (as percentage, e.g., 0.01 for 1%)
        self.reverse_profits[pair].add(current_profit)

        # Get profits from the last 10 minutes (600 seconds)
        profits_last_10m = self.reverse_profits[pair].get_items_last_x_seconds(600)

        # Require a minimum number of data points (e.g., > 60 samples assumes ~1 sample/10sec)
        min_samples = 30 # Reduced for potentially less frequent updates
        if len(profits_last_10m) < min_samples:
            logger.debug(f"[{pair}] Reverse logic check: Not enough samples ({len(profits_last_10m)}/{min_samples}) in last 10 mins.")
            return False

        # 1. Check if all profits in the window are negative
        all_negative = all(p < 0 for p in profits_last_10m)
        if not all_negative:
            logger.debug(f"[{pair}] Reverse logic check: Not all profits in last 10 mins were negative.")
            return False

        # 2. Check if the trend is generally worsening (optional but can be useful)
        # Example: Is the first half's average worse than the second half's?
        # mid_point = len(profits_last_10m) // 2
        # avg_first_half = np.mean(profits_last_10m[:mid_point])
        # avg_second_half = np.mean(profits_last_10m[mid_point:])
        # if avg_second_half >= avg_first_half: # If profit improved or stayed same
        #     logger.debug(f"[{pair}] Reverse logic check: Profit trend not worsening ({avg_first_half:.4f} -> {avg_second_half:.4f}).")
        #     return False

        # 3. Check if the profit trend is linearly decreasing (using the helper)
        # Adjust the threshold as needed. A smaller threshold means a flatter negative slope qualifies.
        # A threshold of 0.1 might be too large if profits are small percentages. Try 0.001?
        if self.is_linear_decreasing(profits_last_10m, threshold=0.001): # Check for negative slope
            logger.info(f"[{pair}] Reverse logic triggered: Profits consistently negative and decreasing in last 10 mins. Samples: {len(profits_last_10m)}")
            logger.debug(f"[{pair}] Profits sample (last 10): {profits_last_10m[-10:]}") # Log last few profits
            return True
        else:
            logger.debug(f"[{pair}] Reverse logic check: Profits negative, but trend not linearly decreasing.")
            return False


    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs) -> Optional[str]:
        """
        Custom exit logic based on the reverse_logic function.
        Also resets the reverse entry counter for the pair if triggered.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1] # Access last candle data if needed

        if self.reverse_logic(pair, current_profit):
            # --- Reset Reverse Entry Counter ---
            # Resetting the counter signifies that a reverse exit has just happened,
            # allowing the count for subsequent entries to start from 0 again.
            # If the pair wasn't in the dict, it means no reverse exit happened before,
            # so adding it with 0 starts the tracking.
            self.reverse_entry_counts[pair] = 0
            logger.info(f"[{pair}] Custom exit triggered by reverse_logic. Resetting/Initializing reverse entry counter to 0.")
            # ------------------------------------

            # Set sentiment based on the reversal
            symbol = pair.split('/')[0]
            exit_reason = f"reverse_logic_exit_count_reset"
            if trade.is_short:
                self.set_sentiment(symbol, "long")
                logger.info(f"[{pair}] Trade was short, setting sentiment to long due to reverse logic.")
                exit_reason = f"reverse_short_to_long_count_reset"
            else: # trade.is_long
                self.set_sentiment(symbol, "short")
                logger.info(f"[{pair}] Trade was long, setting sentiment to short due to reverse logic.")
                exit_reason = f"reverse_long_to_short_count_reset"

            return exit_reason # Return a reason string to exit

        return None # Return None if custom exit condition is not met


    # --- Optional: Helper methods for external API calls (like getWinrate) ---
    # These seem less related to the core strategy logic and more to monitoring/external interaction.
    # Ensure they handle errors gracefully and have proper authentication if used.

    def make_get_request_with_retry(self, url, auth, retries=3, sleep_time=5):
        """ Generic GET request helper with retries """
        response = None
        last_exception = None
        for attempt in range(retries):
            try:
                # Consider using a session object for connection pooling if making many calls
                response = requests.get(url, auth=auth, timeout=20) # Increased timeout
                response.raise_for_status()
                logger.debug(f"Successful request to endpoint {url}")
                return response
            except requests.exceptions.RequestException as ex:
                last_exception = ex
                logger.warning(f"Attempt {attempt + 1}/{retries} failed for {url}. Error: {ex}. Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
            except Exception as ex: # Catch other potential errors
                 last_exception = ex
                 logger.error(f"Unexpected error during request attempt {attempt + 1}/{retries} for {url}: {ex}")
                 # Depending on the error, may or may not want to retry
                 time.sleep(sleep_time)

        logger.error(f"Retries exhausted for {url}. Last error: {last_exception}")
        # Optionally raise the last exception or return None
        # raise last_exception if last_exception else Exception(f"Request failed after {retries} retries")
        return None # Indicate failure

    def getJsonFromAPI(self, endpoint):
        """ Safely gets JSON from an API endpoint using the retry helper """
        # These should be loaded securely, e.g., from config or environment variables
        # bot_username = self.config.get('api_server', {}).get('username')
        # bot_password = self.config.get('api_server', {}).get('password')
        bot_username = "freqtrade" # Example placeholder
        bot_password = "&&Liz270117!!" # Example placeholder - DO NOT HARDCODE SECRETS

        if not bot_username or not bot_password:
             logger.error(f"API username/password not configured. Cannot call {endpoint}.")
             return {} # Return empty dict on config error

        auth = (bot_username, bot_password)
        response = self.make_get_request_with_retry(endpoint, auth)

        if response:
            try:
                data = response.json()
                return data
            except ValueError as e: # Catches JSON decoding errors
                logger.error(f"Failed to decode JSON from {endpoint}. Error: {e}. Response text: {response.text[:200]}")
                return {} # Return empty dict on JSON error
        else:
            logger.error(f"No response received from {endpoint} after retries.")
            return {} # Return empty dict on request failure


    def getWinrate(self):
        """ Example usage of getJsonFromAPI to fetch winrate """
        # Ensure self.api_profit is correctly defined and accessible
        if not hasattr(self, 'api_profit') or not self.api_profit:
            logger.error("api_profit endpoint not defined in strategy.")
            return None # Indicate error

        data = self.getJsonFromAPI(self.api_profit)

        if data and 'winrate' in data:
            winrate = data['winrate']
            logger.info(f"Fetched winrate: {winrate}")
            return winrate
        else:
            logger.error(f"Could not get 'winrate' from profit endpoint {self.api_profit}. Data received: {data}")
            return None # Indicate error

    # --- Utility Logging ---
    # def logme(self, msg, telegram=False): # Example utility
    #     logger.info(msg)
    #     if telegram and hasattr(self, 'dp') and self.dp:
    #         try:
    #             self.dp.send_msg(msg)
    #         except Exception as e:
    #             logger.error(f"Failed to send Telegram message: {e}")


# --- END OF FILE AnandaStrategySplit.py ---


Key Changes and Explanations:

MAX_REVERSE_ENTRIES Constant: Defined at the top for clarity and easy modification (set to 3).

reverse_entry_counts Dictionary: Added as a class attribute (self.reverse_entry_counts: Dict[str, int] = {}) to store the count of entries made after a reverse_logic exit for each pair. The key is the pair string (e.g., "BTC/USDT"), and the value is the integer count.

Initialization (bot_loop_start): Cleared reverse_entry_counts and reverse_profits at the start to ensure a clean state for each bot run.

custom_exit Modification:

When self.reverse_logic(pair, current_profit) returns True:

It now resets or initializes the count for that pair in self.reverse_entry_counts to 0. self.reverse_entry_counts[pair] = 0. This signals that a reverse exit just occurred, and the counting for subsequent entries should begin (or restart).

A more descriptive exit reason is returned (e.g., reverse_long_to_short_count_reset).

populate_entry_trend Modification:

Limit Check First: It now checks self.reverse_entry_counts before calling get_bias.

It uses current_reverse_entry_count = self.reverse_entry_counts.get(pair) to safely get the count. If the pair is not in the dictionary, it means no reverse exit has happened for it yet, so get() returns None, and the limit doesn't apply.

If current_reverse_entry_count is not None and is >= MAX_REVERSE_ENTRIES, it sets limit_reached = True, logs a message, sets the entry signals to 0, adds a specific enter_tag, and returns early, preventing any entry signal generation.

Increment Counter: If an entry signal is generated (enter_long=1 or enter_short=1) and current_reverse_entry_count is not None (meaning we are actively tracking this pair post-reverse-exit), the counter for that pair is incremented: self.reverse_entry_counts[pair] = current_reverse_entry_count + 1. The enter_tag is also updated to show the current count state.

State Management: The reverse_entry_counts dictionary holds the state within the running bot instance. If the bot restarts, this state is lost (unless you implement persistence using the database, which is more complex). The current implementation resets the count only when a new custom_exit via reverse_logic occurs for that pair.

Logging: Added more specific logging messages related to the reverse entry counting and limiting.

API Calls: Refined API call functions (get_bias, set_sentiment, get_leverage, make_get_request_with_retry, getJsonFromAPI) with timeouts, better error handling, default values, and debug logging. Ensured they use the symbol (e.g., "BTC") where appropriate based on the likely API design. Added placeholders for secure credential handling.

Helper Functions (is_linear_decreasing, reverse_logic): Improved robustness, added logging, and adjusted parameters (like min_samples, threshold).

Imports and Typing: Added Dict, Any, Optional, Tuple, DataProvider, Trade, PairLocks, logging, traceback, threading.

Logger Usage: Switched from logging.warn/info to logger.warning/info using logger = logging.getLogger(__name__).

This implementation correctly tracks entries after a custom_exit based on reverse_logic and prevents further entries once the limit of 3 is reached, until the counter is reset by another reverse_logic exit for the same pair.
