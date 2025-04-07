import os

import requests
from bias.interface import BiasInterface, BiasRequest, BiasResponse, BiasType


class CoinGeckoBTC(BiasInterface):
    def get_bitcoin_data(self):
        API_URL = "https://pro-api.coingecko.com/api/v3"
        COINGECKO_API_KEY=os.getenv("COINGECKO_API_KEY", "")
        endpoint = f"{API_URL}/coins/markets"
        params = {
            "vs_currency": "usd",
            "ids": "bitcoin",
            "order": "market_cap_desc",
            "per_page": 1,
            "page": 1,
            "sparkline": "false",
            "price_change_percentage": "24h,7d",
            "x_cg_pro_api_key": COINGECKO_API_KEY,
        }
        response = requests.get(endpoint, params=params, timeout=10)
        response.raise_for_status()  # Raise an error for bad responses
        data = response.json()
        return data[0] if data else None

    def analyze_trend_day(self, bitcoin_data):
        change_24h = bitcoin_data.get("price_change_percentage_24h", 0)
        # change_7d = bitcoin_data.get("price_change_percentage_7d_in_currency", 0)
        volume = bitcoin_data.get("total_volume", 0)

        # Determine bias based on 24-hour and 7-day changes
        if change_24h > 0: # and change_7d > 0:
            trend = BiasType.LONG
        elif change_24h < 0: # and change_7d < 0:
            trend = BiasType.SHORT
        else:
            trend = BiasType.NEUTRAL

        return trend


    def bias(self, biasRequest: BiasRequest) -> BiasResponse:
        bitcoin_data = self.get_bitcoin_data()
        trend_bias = self.analyze_trend_day(bitcoin_data)
        return BiasResponse(bias=trend_bias)
