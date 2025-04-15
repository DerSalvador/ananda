import requests
import os
from bias import get_config
from utils import get_logger
from bias.interface import BiasInterface, BiasRequest, BiasResponse, BiasType

logger = get_logger()

class GreedAndFear(BiasInterface):
    def bias(self, biasRequest: BiasRequest) -> BiasResponse:
        limit = int(get_config("GreedAndFearLimit", 10))
        url = f"https://api.alternative.me/fng/?limit={limit}"
        logger.info(f"Requesting {url}")

        response = requests.get(url)
        data = response.json()
        if 'data' not in data:
            raise Exception(data)
        
        classifications = [entry['value_classification'].lower() for entry in data['data']]
        
        greed_count = sum(1 for c in classifications if 'greed' in c)
        fear_count = sum(1 for c in classifications if 'fear' in c)
        logger.info(f"Greed count: {greed_count}")
        logger.info(f"Fear count: {fear_count}")
        total = len(classifications)
        
        if greed_count / total >= 0.8:
            ret = BiasType.LONG
            reason = f"Greed count: {greed_count}, Fear count: {fear_count}, Percentage: {greed_count / total:.2%}"
        elif fear_count / total >= 0.8:
            ret = BiasType.SHORT
            reason = f"Fear count: {fear_count}, Greed count: {greed_count}, Percentage: {fear_count / total:.2%}"
        else:
            ret = BiasType.NEUTRAL
            reason = f"Greed count: {greed_count}, Fear count: {fear_count}, Percentage: {greed_count / total:.2%}"

        return BiasResponse(bias=ret, usedSymbol=False, reason=reason)

