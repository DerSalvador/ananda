import requests
from utils import get_logger
from bias.interface import BiasInterface, BiasRequest, BiasResponse, BiasType

logger = get_logger()

class GreedAndFear(BiasInterface):
    def bias(self, biasRequest: BiasRequest) -> BiasResponse:
        url = "https://api.alternative.me/fng/?limit=100"

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
        elif fear_count / total >= 0.8:
            ret = BiasType.SHORT
        else:
            ret = BiasType.NEUTRAL

        return BiasResponse(bias=ret)

