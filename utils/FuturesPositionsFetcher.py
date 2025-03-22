import requests
from requests.adapters import HTTPAdapter
import logging
import sys
import argparse
# from binance.spot import Spot as Client
from binance.client import Client
from binance.lib.utils import config_logging
import time
# from examples.utils.prepare_env import get_api_key
import sys
import argparse
import requests
import time
import hmac
import hashlib
from urllib.parse import urlencode


config_logging(logging, logging.INFO)
log = logging.getLogger(__name__)

class FuturesPositionsFetcher:
    client = None
    def __init__(self, api_key, api_secret):
        config_logging(logging, logging.INFO)
        # Step 1: Create a session
        session = requests.Session()

        # session.mount('https://', adapter)
        self.client = Client(api_key, api_secret)

    def get_futures_position_information(self, symbol):
        try:
            if not symbol:
                raise ValueError("Symbol must be specified.")
            # Step 2: Configure the session adapter to increase pool size
            adapter = HTTPAdapter(pool_connections=100, pool_maxsize=100)  # You can adjust the pool size as needed
            self.client.session.mount('https://', adapter)
            pos = self.client.futures_position_information(symbol=symbol, recvWindow=60000)
            # log.info(f"Found Position in Binance: {pos}")
        except Exception as e:
            log.error (f"Exception in Binance Get Future Position: {e}")
            raise e
        return pos
    
    base_url = 'https://api.binance.com'
    fbase_url = 'https://fapi.binance.com'

    headers = {
        'X-MBX-APIKEY': ''
    }

    def get_timestamp():
        return int(time.time() * 1000)

    def getPosition(self, api_key, api_secret, symbol) -> dict:
        response = None
        try:
            # get_account_balances(api_key, api_secret, wallet)
            current_timestamp_ms = int(round(time.time() * 1000))
            # path = " /sapi/v2/sub-account/futures/positionRisk" # "/fapi/v2/positionRisk" #  "/sapi/v2/futures/positionRisk" # '/fapi/v2/positionRisk'
            path = '/fapi/v2/positionRisk'
            params = {
                'symbol': symbol, 
                'recvWindow': 60000,
                'timestamp': current_timestamp_ms
            }
            params['signature'] = self.sign_request(params, api_secret)
            self.headers['X-MBX-APIKEY'] = api_key
            url = self.fbase_url + path
            response = requests.get(url, headers=self.headers, params=params)
        except Exception as e:
            log.error(f"Error/Exception in FuturesPositionsFetcher.getPosition {e}, api_key={api_key}, api_secret={api_secret}, symbol={symbol}")
            raise e
        return response.json()

    def sign_request(self, data, api_secret):
        query_string = urlencode(data)
        signature = hmac.new(api_secret.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
        return signature