import os
import requests
from bias.coin_gecko_btc import COIN_GECKO_SYMBOLS
from utils import get_logger
from bias.interface import BiasInterface, BiasRequest, BiasResponse, BiasType

logger = get_logger()

class CoinGeckoMarket(BiasInterface):
    paid = True
    def get_crypto_market_data(self, biasRequest: BiasRequest):
        api_key = os.getenv("COINGECKO_API_KEY", "")
        url = f"https://pro-api.coingecko.com/api/v3/coins/markets?x_cg_pro_api_key={api_key}"
        params = {
            'vs_currency': 'usd',
            # "ids": self.getidfromsymbol(biasRequest.symbol),
            'order': 'market_cap_desc',
            'per_page': 10,
            'page': 1,
            'sparkline': False
        }
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an error for bad responses
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
            return BiasType.LONG, f"positive: {total_positive}, negative: {total_negative}"
        elif total_positive < total_negative:
            return BiasType.SHORT, f"positive: {total_positive}, negative: {total_negative}"
        else:
            return BiasType.NEUTRAL, f"positive: {total_positive}, negative: {total_negative}"


    def bias(self, biasRequest: BiasRequest) -> BiasResponse:
        market_data = self.get_crypto_market_data(biasRequest)
        bias, reason = self.calculate_trend_bias(market_data)
        return BiasResponse(bias=bias, usedSymbol=False, reason=reason)

    def getidfromsymbol(self, symbol: str) -> str:
        # Placeholder for the actual implementation to get the ID from the symbol
        # This should be replaced with the actual logic to map symbols to IDs
        for coin in COIN_GECKO_SYMBOLS:
            if coin['symbol'].lower() == symbol.lower():
                return coin['id']
        logger.warning(f"Symbol {symbol} not found in COIN_GECKO_SYMBOLS.")
        return "bitcoin"
