from dotenv import load_dotenv
load_dotenv()

from db import current_position, get_profits, should_reverse, update_profit, update_sentiment
from bias import get_all_configs, get_biases, get_config, getInterfaces, update_bias, update_config
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
def get_sentiment(symbol: str, all_sentiments: bool = False) -> dict[str, BiasResponse]:
    request = BiasRequest(symbol=symbol)
    return post_sentiment(request, all_sentiments)

def post_sentiment(request: BiasRequest, all_sentiments: bool = False) -> dict[str, BiasResponse]:
    sentiments = {}
    interfaces = getInterfaces(all = all_sentiments)
    executor = ThreadPoolExecutor(max_workers=len(interfaces))
    futures = {executor.submit(interface.bias, request): name for name, interface in interfaces.items()}

    for future in futures:
        interface_name = futures[future]
        try:
            sentiment = future.result()
            sentiments[interface_name] = sentiment
        except Exception as e:
            sentiments[interface_name] = BiasResponse(bias=BiasType.NEUTRAL, error=str(e))
            logger.exception(f"Error: {e} for {interface_name} and {request.symbol}.")

    # Always restrictive logic
    sentiment_values = [result.bias for result in sentiments.values()]
    if sentiment_values and all(s == sentiment_values[0] for s in sentiment_values):
        reason = f"Final sentiment agreed on {sentiment_values[0]}"
        logger.info(reason)
        sentiments["final"] = {"bias": sentiment_values[0], "reason": reason}
    else:
        logger.warn(f"Final sentiment is NEUTRAL")
        sentiments["final"] = {"bias": BiasType.NEUTRAL, "error": "Sentiments do not agree", "reason": "Sentiments do not agree"}

    # Check back on real values from custom_exit and reverse trend if needed
    should_reverse_bool = should_reverse(request.symbol)
    logger.info(f"Should reverse: {should_reverse_bool}")
    if should_reverse_bool:
        cur_position = current_position(request.symbol)
        if cur_position == sentiments["final"]["bias"]:
            logger.info(f"Reversing position for {request.symbol} to LONG")
            if cur_position == BiasType.SHORT:
                sentiments["final"] = {"bias": BiasType.LONG, "reason": "Sentiments agreed on SHORT, reversing position"}
            elif cur_position == BiasType.LONG:
                sentiments["final"] = {"bias": BiasType.SHORT, "reason": "Sentiments agreed on LONG, reversing position"}

    return sentiments

class UpdateBiasRequest(BaseModel):
    name: str
    active: bool

@app.get("/", response_class=HTMLResponse)
def get_config_page(request: Request):
    return templates.TemplateResponse("config.html", {"request": request, "biases": get_biases(), "configs": get_all_configs()})

class UpdateLeverageRequest(BaseModel):
    pair: str = "default"
    leverage: float

@app.get("/leverage")
def _get_leverage(pair: str = "default"):
    leverage = get_config("Leverage")
    if not leverage:
        leverage = 5.0
    else:
        leverage = float(leverage)
    return {"leverage": leverage}

@app.get("/currentsentiment")
def _get_current_sentiment():
    symbols = get_config("BiasSymbols")
    show_all = get_config("BiasShowAll") == "true"
    if not symbols:
        raise ValueError("No symbols found in config")
    sentiments = {}
    for symbol in symbols.split(","):
        symbol = symbol.strip()
        sentiment = get_sentiment(symbol, show_all)
        logger.info(sentiment)
        sentiments[symbol] = sentiment
    return sentiments

@app.post("/update-bias")
def _update_bias(data: UpdateBiasRequest):
    update_bias(data.name, data.active)
    return {"status": "success", "message": "Bias updated"}

@app.get("/configs")
def _get_configs():
    return get_all_configs()

class UpdateConfigRequest(BaseModel):
    name: str
    value: str
@app.post("/config")
def _update_config(data: UpdateConfigRequest):
    update_config(data.name, data.value)
    return {"status": "success", "message": "Config updated"}

class AddProfitRequest(BaseModel):
    symbol: str
    profit: float
    is_short: bool = False
@app.post("/profit")
def _update_profit(addProfitRequest: AddProfitRequest):
    update_profit(addProfitRequest.symbol, addProfitRequest.profit, addProfitRequest.is_short)

@app.get("/profit/{symbol}")
def _get_profit(symbol: str):
    profits = get_profits(symbol)
    return profits

@app.get("/customexit/{symbol}")
def custom_exit(symbol: str):
    should_reverse_bool = should_reverse(symbol)
    return { "exit": should_reverse_bool, "position": current_position(symbol) }

@app.post("/sentiment/{symbol}")
def _update_sentiment(symbol: str, updateRequest: BiasResponse):
    logger.info(f"Updating sentiment for {symbol} to {updateRequest.bias}")
    update_sentiment(symbol, updateRequest.bias)
    return {"status": "success", "message": "Sentiment updated"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9000, reload=True)

