from dotenv import load_dotenv
load_dotenv()

from db import get_leverage, update_leverage
from bias import get_biases, getInterfaces, update_bias
from pydantic import BaseModel
from bias.interface import BiasRequest, BiasResponse, BiasType
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from utils import get_logger
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import uvicorn
from datetime import datetime

app = FastAPI()
templates = Jinja2Templates(directory="templates")
# app.mount("/static", StaticFiles(directory="static"), name="static")
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
    interfaces = getInterfaces()
    executor = ThreadPoolExecutor(max_workers=len(interfaces))
    futures = {executor.submit(interface.bias, request): name for name, interface in interfaces.items()}

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
        logger.warn(f"Final sentiment is NEUTRAL")
        sentiments["final"] = {"bias": BiasType.NEUTRAL, "error": "Sentiments do not agree"}
    return sentiments

class UpdateBiasRequest(BaseModel):
    name: str
    active: bool

@app.get("/", response_class=HTMLResponse)
async def get_config_page(request: Request):
    return templates.TemplateResponse("config.html", {"request": request, "biases": get_biases(), "leverage": get_leverage()})

class UpdateLeverageRequest(BaseModel):
    pair: str = "default"
    leverage: float

@app.get("/leverage")
async def _get_leverage(pair: str = "default"):
    return {"leverage": get_leverage(pair)}

@app.post("/update-leverage")
async def _update_leverage(data: UpdateLeverageRequest):
    update_leverage(data.pair, data.leverage)
    return {"status": "success", "message": "Leverage updated"}

@app.post("/update-bias")
async def _update_bias(data: UpdateBiasRequest):
    update_bias(data.name, data.active)
    return {"status": "success", "message": "Bias updated"}



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9000, reload=True)

