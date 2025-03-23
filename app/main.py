from dotenv import load_dotenv
load_dotenv()

from bias import INTERFACES
from bias.interface import BiasRequest, BiasResponse, BiasType
from utils import get_logger
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI
import uvicorn
from datetime import datetime

app = FastAPI()
logger = get_logger()

@app.get("/status")
async def get_status():
    return {"timestamp": datetime.now().isoformat()}

@app.get("/sentiment/{symbol}")
async def get_sentiment(symbol: str) -> dict[str, BiasResponse]:
    request = BiasRequest(symbol=symbol)
    return post_sentiment(request)

def post_sentiment(request: BiasRequest) -> dict[str, BiasResponse]:
    sentiments = {}
    executor = ThreadPoolExecutor(max_workers=len(INTERFACES))
    futures = {executor.submit(interface.bias, request): name for name, interface in INTERFACES.items()}

    for future in futures:
        interface_name = futures[future]
        try:
            sentiment = future.result()
            sentiments[interface_name] = sentiment
        except Exception as e:
            sentiments[interface_name] = BiasResponse(bias=BiasType.NEUTRAL, error=str(e))
            logger.error(f"Error: {e} for {interface_name} and {request.symbol}.")

    # Always restrictive logic
    sentiment_values = [result.bias for result in sentiments.values()]
    if sentiment_values and all(s == sentiment_values[0] for s in sentiment_values):
        logger.info(f"Final sentiment agreed on {sentiment_values[0]}")
        sentiments["final"] = {"bias": sentiment_values[0]}
    else:
        logger.info(f"Final sentiment is NEUTRAL")
        sentiments["final"] = {"bias": BiasType.NEUTRAL, "error": "Sentiments do not agree"}
    return sentiments

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9000, reload=True)

