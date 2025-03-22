#pragma pylint: disable=W0105, C0103, C0301, W1203

from datetime import datetime
from functools import reduce
# import timeit
import requests
import argparse
import logging
import time
from coinGeckoAPI import CoinGeckoAPI
import pandas as pd
from tabulate import tabulate
from binance.client import Client
from binance.exceptions import BinanceAPIException
import requests
import time
import hmac
import hashlib
import requests
# from freqtrade.rpc import RPCManager
# from freqtrade.rpc.external_message_consumer import ExternalMessageConsumer
# from freqtrade.rpc.rpc_types import (ProfitLossStr, RPCCancelMsg, RPCEntryMsg, RPCExitCancelMsg,
#                                      RPCExitMsg, RPCProtectionMsg)
# from freqtrade import rpc_singleton
from urllib.parse import urlencode
from binance.client import Client

import numpy as np
# Get rid of pandas warnings during backtesting
import pandas as pd
from pandas import DataFrame, Series
import scipy
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

# from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# import freqtrade.vendor.qtpylib.indicators as qtpylib

from freqtrade.strategy import (IStrategy, DecimalParameter, CategoricalParameter)
from freqtrade.persistence import Trade

pd.options.mode.chained_assignment = None  # default='warn'

# Strategy specific imports, files must reside in same folder as strategy
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import logging
import warnings

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

class dsHedging():
    
    @staticmethod
    def logme(msg: str):
        print(f"{msg}")
        log.info(f"{msg}")
            
    @staticmethod
    def hedge_me(strategy: IStrategy, is_short: bool, pair, position, endpoint=None, label=None):
        if position is None or float(position[0]['positionAmt']) == 0.0:
            dsHedging.logme("No position exist, trying to hedge")
            if is_short:
                is_short = False # take as is becomes long trade
                dsHedging.place_order_in_binance(strategy, is_short, pair, position, endpoint, label)
                # return dsHedging.hedge(strategy, pair, "long", endpoint, label)
            else:
                is_short = True # take as is becomes short trade
                dsHedging.place_order_in_binance(strategy, is_short, pair, position, endpoint, label)
                # return dsHedging.hedge(strategy, pair, "short", endpoint, label)
        else:
            dsHedging.logme(f"NOT hedging because position already exists for {position} with amount {float(position[0]['positionAmt'])}")
            return False
       
    BASE_URL = 'https://fapi.binance.com'

    base_url = 'https://api.binance.com'
    fbase_url = 'https://fapi.binance.com'

    headers = {
        'X-MBX-APIKEY': ''
    }

    def get_timestamp():
        return int(time.time() * 1000)

    def sign_request(strategy, data, api_secret):
        query_string = urlencode(data)
        signature = hmac.new(strategy.hedging_apisecret.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
        return signature

    # Get server time from Binance
    def get_server_time():
        response = requests.get(f"https://fapi.binance.com/fapi/v1/time")
        return response.json()['serverTime']
                   
    @staticmethod
    def place_order_in_binance(strategy: IStrategy, is_short: bool, pair, position, endpoint=None, label=None):
        try:
            symbol = pair.replace("/USDT:USDT", "")
            symbol = symbol.replace("/USDT", "USDT")
            side = None
            if is_short == True:
                side = 'SELL'
            else:
                side = 'BUY'
            tag=f'Hedging {pair}: {datetime.now()}, leverage {strategy.hedging_leverage}, '
            tag+=f'amount {strategy.hedging_stake_amount}, ' 
            tag+=f'direction {side} on binance directly'
            dsHedging.logme(tag)
            dsHedging.logme(f"Haleluja.... start finally hedging for pair {pair} with direction {side}")
            dsHedging.logme(f"Entering Hedge Modus with direction {side}")
            client = Client(strategy.hedging_apikey, strategy.hedging_apisecret)
            ticker = client.futures_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])
            quantity = round(strategy.hedging_stake_amount / current_price, dsHedging.getQuantityPrecision(symbol))  # Adjust the USDT amount as needed

            # client.futures_change_leverage(symbol=symbol, leverage=strategy.hedging_leverage, recvWindow=20000 )
            q = round(quantity, dsHedging.getPricePrecision(symbol))
            # order = client.futures_create_order(
            #     symbol=symbol,
            #     side=side,
            #     type="MARKET",
            #     newClientOrderId=tag,
            #     recvWindow=20000,
            #     quantity=q
            # )
            # Timestamp and recvWindow to prevent signature issues
            timestamp = int(time.time())
            recv_window = 10000  # 10 seconds
            # Calculate time offset
            server_time = dsHedging.get_server_time()
            local_time = int(time.time() * 1000)
            time_offset = server_time - local_time

            # Parameters to be signed
            # params = {
            #     'symbol': symbol,
            #     'side': side,
            #     'type': "MARKET",
            #     'quantity': q,
            #     'newClientOrderId': tag,
            #     'timestamp': int(timestamp) * 1000,
            #     'recvWindow': 10000
            # }

            # Create query string
            # query_string = '&'.join([f"{key}={value}" for key, value in params.items()])

            # Generate the signature using HMAC SHA256
            # signature = hmac.new(
            #     strategy.hedging_apisecret.encode('utf-8'),
            #     query_string.encode('utf-8'),
            #     hashlib.sha256
            # ).hexdigest()

            # Add the signature to the parameters
            # params['signature'] = signature

            # Headers with API Key
            headers = {
                'X-MBX-APIKEY': strategy.hedging_apikey
            }
            # Send the POST request to create the order
            current_timestamp_ms = int(round(time.time() * 1000))
            # path = " /sapi/v2/sub-account/futures/positionRisk" # "/fapi/v2/positionRisk" #  "/sapi/v2/futures/positionRisk" # '/fapi/v2/positionRisk'
            params = {
                'symbol': symbol,
                'side': side,
                'type': "MARKET",
                'quantity': q,
                # 'newClientOrderId': tag[:35],
                'recvWindow': 60000,
                'timestamp': current_timestamp_ms + abs(time_offset)
            }
            params['signature'] = dsHedging.sign_request(strategy, params, strategy.hedging_apisecret)
            headers['X-MBX-APIKEY'] = strategy.hedging_apikey            
            response = requests.post(f"https://fapi.binance.com/fapi/v1/order", headers=headers, params=params, timeout=60000)

            # Output the response
            print(response.json())
            
            # dsHedging.logme(f"Order placed for {symbol}: {order}")
        except Exception as e:
            dsHedging.logme(f"Error in place_order_in_binance in dsHedging {e}")
            raise e
            
    @staticmethod
    def getPricePrecision(symbol):
            url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
            response = requests.get(url)
            data = response.json()
            for pos in data['symbols']:
                print(f"Price Precision: {pos['symbol']} {pos['pricePrecision']}")
                if pos['symbol'] == symbol:
                    return pos['pricePrecision']

    @staticmethod
    def getQuantityPrecision(symbol):
            url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
            response = requests.get(url)
            data = response.json()
            for pos in data['symbols']:
                print(f"Quantity Precision: {pos['symbol']} {pos['quantityPrecision']}")            
                if pos['symbol'] == symbol:
                    return pos['quantityPrecision']

    # Do *not* hyperopt for the roi and stoploss spaces
    @staticmethod
    def hedge(strategy: IStrategy, pair, direction, endpoint=None, label=None):
        # Get the current timestamp in UTC timezone
        current_timestamp = datetime.now()
        if endpoint is None:
            endpoint = strategy.hedging_url
        tag=f'{label} Hedging {pair}: {current_timestamp}, leverage {strategy.hedging_leverage}, '
        tag+=f'amount {strategy.hedging_stake_amount}, ' 
        tag+=f'direction {direction} on bot {endpoint}'
        dsHedging.logme(f"Haleluja.... start finally hedging for pair {pair} with direction {direction}")
        dsHedging.logme(f"Entering Hedge Modus with direction {direction}")
        if ':' not in pair:
            pair += ':USDT'
        dsHedging.logme(f"Entering hedging: {tag}")
        payload = {
            "pair": pair,
            "side": direction,
            "ordertype": "market",
            "stakeamount": strategy.hedging_stake_amount,
            "entry_tag": tag,
            "leverage": strategy.hedging_leverage
        }
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }
        username = strategy.config['api_server']['username']
        password = strategy.config['api_server']['password']
        auth = (username, password)

        try:
            response = requests.post(endpoint, json=payload, headers=headers, auth=auth)
            response.raise_for_status()  # Check for HTTP errors
            if response.status_code == 200:
                dsHedging.logme("Hedging successful. Response:")
                dsHedging.logme(response.json())
                return True
            else:
                dsHedging.logme(f"Request returned status code: {response.status_code} with response {response}")
                return False
        except Exception as e:
            dsHedging.logme(f"An error occurred during hedging for pair {pair}: {e}")
            log.error(f"An error occurred during hedging for pair {pair}: {e}")
            raise e
        finally:
            dsHedging.logme("Leaving Hedge Modus")
