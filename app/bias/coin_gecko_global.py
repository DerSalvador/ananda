import os

import requests
from utils import get_logger
from bias.interface import BiasInterface, BiasRequest, BiasResponse, BiasType

logger = get_logger()

class CoinGeckoGlobal(BiasInterface):
    paid = True
    def bias(self, biasRequest: BiasRequest) -> BiasResponse:
        COINGECKO_API_KEY=os.getenv("COINGECKO_API_KEY", "")
        url = f"https://pro-api.coingecko.com/api/v3/global?x_cg_pro_api_key={COINGECKO_API_KEY}"
        response = requests.get(url)

        data = response.json()
        if data.get("status", {}).get("error_code"):
            raise Exception(data)
        
        market_cap_change_percentage_24h_usd = data["data"]["market_cap_change_percentage_24h_usd"]
        logger.info(f"Market Cap Change Percentage 24h USD: {market_cap_change_percentage_24h_usd}")
        reason = f"Market Cap Change Percentage 24h USD: {round(market_cap_change_percentage_24h_usd, 2)}"
        
        market_trend_bias = BiasType.NEUTRAL
        if market_cap_change_percentage_24h_usd > 0:
            market_trend_bias = BiasType.LONG 
        elif  market_cap_change_percentage_24h_usd < 0:
            market_trend_bias = BiasType.SHORT 
        return BiasResponse(bias=market_trend_bias, usedSymbol=False, reason=reason)
