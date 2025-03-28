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
import time
from freqtrade.strategy import IStrategy
from freqtrade.strategy import CategoricalParameter, IntParameter
from functools import reduce
from pandas import DataFrame
from datetime import datetime
from dateutil import parser
import os 
import sys
# import timeit
from freqtrade.persistence import Trade
import numpy as np
# Get rid of pandas warnings during backtesting
import pandas as pd
from attrs import exceptions
pd.options.display.float_format = '{:f}'.format
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)
from sqlalchemy import create_engine

import sqlite3
import urllib3

from pandas import DataFrame, Series
# --------------------------------
import talib.abstract as ta
import ta as taa
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy  # noqa

import requests
import json
import pywt
import talib.abstract as ta
from utils.DataframeUtils import DataframeUtils, ScalerType

from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from utils.FuturesPositionsFetcher import FuturesPositionsFetcher
# from utils.coinGeckoAPI import CoinGeckoAPI
from typing import Dict, List, Optional, Tuple, Union
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import threading

import logging
import warnings
from scipy.stats import linregress

import threading
from contextvars import ContextVar
from typing import Any, Dict, Final, Optional

from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import NoSuchModuleError
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.pool import StaticPool
# from binance.client import Client
from freqtrade.exceptions import OperationalException
from freqtrade.persistence.base import ModelBase
from freqtrade.persistence.custom_data import _CustomData
from freqtrade.persistence.key_value_store import _KeyValueStoreModel
from freqtrade.persistence.migrations import check_migrate
from freqtrade.persistence.pairlock import PairLock
from freqtrade.persistence.trade_model import Order, Trade 
import traceback
from utils.DataframeUtils import DataframeUtils, ScalerType
import pywt
import talib.abstract as ta

from freqtrade.rpc import RPCManager
from freqtrade.rpc.external_message_consumer import ExternalMessageConsumer
from freqtrade.rpc.rpc_types import (ProfitLossStr, RPCCancelMsg, RPCEntryMsg, RPCExitCancelMsg,
                                     RPCExitMsg, RPCProtectionMsg, RPCMessageType)

from utils.dsHedging import dsHedging
import numpy as np
from enum import Enum


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
        else:
            dataframe.loc[:, ['enter_long', 'enter_tag']] = (0, 'entry_reason')
            
        if market_bias == "short":             
            logging.info(f"Market bias is {market_bias} for {symbol}, skipping order.")
            dataframe.loc[:, ['enter_short', 'enter_tag']] = (1, 'entry_reason')
        else:
            dataframe.loc[:, ['enter_short', 'enter_tag']] = (0, 'entry_reason')
            

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        is_short = trade.is_short
        market_bias = coinGeckoAPI.get_file_sentiment(symbol, coinGeckoAPI.apikey, coinGeckoAPI.apisecret)
        logging.info(f"Market bias for {symbol} is {market_bias}")
        if market_bias == "long" and is_short:       
            logging.info(f"Trade is short but bias is long, selling short")
            return true
            
        if market_bias == "short" and not is_short:             
            logging.info(f"Trade is long but bias is short, selling long")
            return true
                
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str,
                 **kwargs) -> float:
        """
        Customize leverage for each new trade. This method is only called in futures mode.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on this pair
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: "long" or "short" - indicating the direction of the proposed trade
        :return: A leverage amount, which is between 1.0 and max_leverage.
        """
        return 5 # Should be configurable
    
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
