import os
import requests
from utils import get_logger
from bias.interface import BiasInterface, BiasRequest, BiasResponse, BiasType

logger = get_logger()

class CoinGeckoMarket(BiasInterface):
    def get_crypto_market_data(self):
        api_key = os.getenv("COINGECKO_API_KEY", "")
        url = f"https://pro-api.coingecko.com/api/v3/coins/markets?x_cg_pro_api_key={api_key}"
        params = {
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'per_page': 10,
            'page': 1,
            'sparkline': False
        }
        response = requests.get(url, params=params)
        data = response.json()
        return data

    def calculate_trend_bias(self, market_data):
        total_positive = 0
        total_negative = 0
        for crypto in market_data:
            if crypto['price_change_percentage_24h'] is not None:
                if crypto['price_change_percentage_24h'] >= 0:
                    total_positive += 1
                else:
                    total_negative += 1
        
        if total_positive > total_negative:
            return BiasType.LONG
        elif total_positive < total_negative:
            return BiasType.SHORT
        else:
            return BiasType.NEUTRAL


    def bias(self, biasRequest: BiasRequest) -> BiasResponse:
        market_data = self.get_crypto_market_data()
        if market_data.get("status", {}).get("error_code"):
            pass
            raise Exception(market_data)
        bias = self.calculate_trend_bias(market_data)
        return BiasResponse(bias=bias)
