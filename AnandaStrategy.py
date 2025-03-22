from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import argparse
import logging
import time
from utils.coinGeckoAPI import CoinGeckoAPI
import pandas as pd
from tabulate import tabulate
from binance.client import Client
from binance.exceptions import BinanceAPIException
import requests, decimal


ATR_PERIOD = 14  # Number of periods for ATR calculation
COOLDOWN_PERIOD = 300  # 5 minutes

# Dictionary to track last reversal times
last_reversal_time = {}
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Global variables
COOLDOWN_PERIOD = 300  # 5 minutes
last_reversal_time = {}
sleep = 5
SLIPPAGE_PERCENT = 0.001  # 0.1% slippage for limit orders
precision_cache = {}
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
coinGeckoAPI = CoinGeckoAPI()


class AnandaStrategy(IStrategy):
    INTERFACE_VERSION = 3
    # ROI table:
    # fmt: off
    minimal_roi = {'0': 1, '100': 2, '200': 3, '300': -1}
    # fmt: on
    # Stoploss:
    stoploss = -0.2
    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True
    # Buy hypers
    timeframe = '5m'
    use_exit_signal = False
    # #################### END OF RESULT PLACE ####################

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Get market bias
        symbol = metadata['pair'].replace("/USDT:USDT", "")
        market_bias = coinGeckoAPI.get_file_sentiment(symbol, coinGeckoAPI.apikey, coinGeckoAPI.apisecret)
        logging.info(f"Market bias for {symbol} is {market_bias}")
        if market_bias == "neutral":
            logging.info(f"Market bias is {market_bias} for {symbol}, skipping order.")
        if market_bias == "long":       
            logging.info(f"Market bias is {market_bias} for {symbol}, skipping order.")
            dataframe.loc[:, ['enter_long', 'enter_tag']] = (1, 'entry_reason')
        if market_bias == "short":             
            logging.info(f"Market bias is {market_bias} for {symbol}, skipping order.")
            dataframe.loc[:, ['enter_short', 'enter_tag']] = (1, 'entry_reason')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe
    
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Binance Futures Position Manager")
#     parser.add_argument("--apikey", required=True, help="Your Binance API Key")
#     parser.add_argument("--apisecret", required=True, help="Your Binance API Secret")
#     parser.add_argument("--loss", type=float, default=5.0, help="Loss percentage to trigger position reversal (default: 5%)")
#     parser.add_argument("--profit", type=float, default=6.0, help="Profit percentage to trigger position closure (default: 6%)")
#     parser.add_argument("--sleep", type=float, default=5.0, help="sleep in seconds before api call")
#     parser.add_argument("--leverage", type=float, default=5.0, help="leverage")
#     parser.add_argument("--stake", type=float, default=50.0, help="Stake amount per trade (default: 500 USDT)")

#     args = parser.parse_args()

#     coinGeckoAPI.coinGeckoAPIKey = "CG-AgEZRgMf3iLk1S8CwyCKp7N3"
#     coinGeckoAPI.apikey = args.apikey
#     coinGeckoAPI.apisecret = args.apisecret
#     sleep = args.sleep
#     leverage = args.leverage
#     stake_amount = args.stake

#     client = Client(args.apikey, args.apisecret)

#     logging.info(f"Monitoring positions for {args.loss}% loss threshold and {args.profit}% profit threshold...")
#     # monitor_and_manage(client, args.loss, args.profit, leverage, stake_amount)
