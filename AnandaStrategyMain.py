import argparse
import logging
import time
from utils.coinGeckoAPI import CoinGeckoAPI
import pandas as pd
from tabulate import tabulate
from binance.client import Client
from binance.exceptions import BinanceAPIException
import requests, decimal

ATR_PERIOD = 14  # Number of periods for ATR calculation
COOLDOWN_PERIOD = 300  # 5 minutes

# Dictionary to track last reversal times
last_reversal_time = {}
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Global variables
COOLDOWN_PERIOD = 300  # 5 minutes
last_reversal_time = {}
sleep = 5
SLIPPAGE_PERCENT = 0.001  # 0.1% slippage for limit orders
precision_cache = {}
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
coinGeckoAPI = CoinGeckoAPI()

def getPricePrecision(symbol):
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    response = requests.get(url)
    data = response.json()
    for pos in data['symbols']:
        if pos['symbol'] == symbol:
            return pos['pricePrecision']

def getQuantityPrecision(symbol):
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    response = requests.get(url)
    data = response.json()
    for pos in data['symbols']:
        if pos['symbol'] == symbol:
            return pos['quantityPrecision']

def get_open_positions(client):
    try:
        time.sleep(sleep)
        positions = client.futures_position_information()
        open_positions = [pos for pos in positions if float(pos["positionAmt"]) != 0]
        return open_positions
    except BinanceAPIException as e:
        logging.error(f"API Error: {e}")
        return []

# Caching symbol precisions to reduce API calls
def get_symbol_precision(client, symbol):
    if symbol in precision_cache:
        return precision_cache[symbol]

    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    response = requests.get(url)
    data = response.json()
    
    for pos in data['symbols']:
        if pos['symbol'] == symbol:
            precision_cache[symbol] = {
                "pricePrecision": pos['pricePrecision'],
                "quantityPrecision": pos['quantityPrecision']
            }
            return precision_cache[symbol]
    return None

def calculate_pnl(entry_price, mark_price):
    return ((mark_price - entry_price) / entry_price) * 100

def get_tick_size(symbol):
    """Retrieve the tick size for a given symbol from Binance Futures API."""
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    response = requests.get(url)
    data = response.json()

    for s in data['symbols']:
        if s['symbol'] == symbol:
            for f in s['filters']:
                if f['filterType'] == 'PRICE_FILTER':
                    return float(f['tickSize'])  # Convert to float for calculations
    return None  # Return None if symbol not found

def close_position(client, position):
    symbol = position["symbol"]
    qty = abs(float(position["positionAmt"]))
    if qty == 0:
        logging.info(f"No open position for {symbol}")
        return
    
    side = "SELL" if float(position["positionAmt"]) > 0 else "BUY"
    
    # Use last traded price with slippage for limit order
    ticker = client.futures_symbol_ticker(symbol=symbol)
    last_price = float(ticker['price'])
    limit_price = last_price * (1 - SLIPPAGE_PERCENT) if side == "SELL" else last_price * (1 + SLIPPAGE_PERCENT)
    symbol_info = get_symbol_info(client, symbol)

    limit_price = round_to_tick_size(limit_price, get_tick_size(symbol))

    logging.info(f"Closing position: {side} {qty} {symbol} with Limit Order at {limit_price}")
    
    try:
        time.sleep(sleep)
        order = client.futures_create_order(
            symbol=symbol,
            side=side,
            type="LIMIT",
            price=round(limit_price, get_symbol_precision(client, symbol)["pricePrecision"]),
            quantity=qty,
            timeInForce="GTC"
        )
        return order
    except BinanceAPIException as e:
        logging.error(f"Error closing position: {e}")
        
# round price to the nearest tick size
def round_to_tick_size(price, tick_size):
    return float(decimal.Decimal(price).quantize(decimal.Decimal(str(tick_size)), rounding=decimal.ROUND_DOWN))

def reverse_position(client, position):
    global last_reversal_time
    symbol = position["symbol"]
    qty = abs(float(position["positionAmt"]))

    if qty == 0:
        logging.info(f"No open position for {symbol}")
        return

    current_time = time.time()
    if symbol in last_reversal_time and (current_time - last_reversal_time[symbol]) < COOLDOWN_PERIOD:
        remaining_time = COOLDOWN_PERIOD - (current_time - last_reversal_time[symbol])
        logging.info(f"Skipping reversal for {symbol}, cooldown active. Try again in {remaining_time:.2f} seconds.")
        return

    close_position(client, position)

    new_side = "BUY" if float(position["positionAmt"]) < 0 else "SELL"

    ticker = client.futures_symbol_ticker(symbol=symbol)
    last_price = float(ticker['price'])
    limit_price = last_price * (1 - SLIPPAGE_PERCENT) if new_side == "SELL" else last_price * (1 + SLIPPAGE_PERCENT)
    symbol_info = get_symbol_info(client, symbol)
    
    limit_price = round_to_tick_size(limit_price, get_tick_size(symbol))
    logging.info(f"Reversing position: {new_side} {qty} {symbol} with Limit Order at {limit_price}")

    try:
        time.sleep(sleep)
        order = client.futures_create_order(
            symbol=symbol,
            side=new_side,
            type="LIMIT",
            price=round(limit_price, get_symbol_precision(client, symbol)["pricePrecision"]),
            quantity=qty,
            timeInForce="GTC"
        )
        last_reversal_time[symbol] = current_time  # Update last reversal time
        return order
    except BinanceAPIException as e:
        logging.error(f"Error reversing position: {e}")
                
def get_trade_history(client, symbol):
    try:
        time.sleep(sleep)
        trades = client.futures_account_trades(symbol=symbol)
        return trades
    except BinanceAPIException as e:
        logging.error(f"Error fetching trade history for {symbol}: {e}")
        return []

def has_recent_loss(client, symbol, hours=0.2):
    try:
        trades = get_trade_history(client, symbol)
    except Exception as e:
        logging.error(f"Failed to retrieve trade history for {symbol}: {e}")
        return False

    if not trades:
        logging.info(f"No trades found for {symbol}.")
        return False

    trades = sorted(trades, key=lambda x: x['time'], reverse=True)
    latest_trade = trades[0]

    if float(latest_trade['realizedPnl']) > 0:
        logging.info(f"Latest trade for {symbol} is profitable with PnL {latest_trade['realizedPnl']}. No recent loss reported.")
        return False

    current_time = time.time() * 1000
    cutoff_time = current_time - (hours * 3600 * 1000)

    for trade in trades:
        if trade['time'] >= cutoff_time and float(trade['realizedPnl']) < 0 and trade['symbol'] == symbol:
            logging.info(f"Recent loss detected for {symbol} in the last {hours} hours, loss {trade['realizedPnl']}.")
            return True

    return False
def place_order_if_no_position(client, symbol, lev, stake_amount):
    """Places a limit order if there's no open position"""
    open_positions = [pos["symbol"] for pos in client.futures_position_information() if float(pos["positionAmt"]) != 0]
    market_bias = coinGeckoAPI.get_file_sentiment(symbol, coinGeckoAPI.apikey, coinGeckoAPI.apisecret)
    logging.info(f"Market bias for {symbol} is {market_bias}")
    coinGeckoAPI.bias_determination = coinGeckoAPI.getBiasDetermination()
    logging.info(f"Bias Determination is {coinGeckoAPI.bias_determination}")
    
    if symbol in open_positions:
        logging.info(f"Position already open for {symbol}, skipping order.")
        return

    symbol_info = get_symbol_info(client, symbol)
    if not symbol_info:
        logging.error(f"Could not fetch precision data for {symbol}. Skipping order.")
        return

    client.futures_change_leverage(symbol=symbol, leverage=int(lev))
    time.sleep(sleep)

    ticker = client.futures_symbol_ticker(symbol=symbol)
    current_price = float(ticker['price'])
    
    
    # Get market bias
    market_bias = coinGeckoAPI.get_file_sentiment(symbol, coinGeckoAPI.apikey, coinGeckoAPI.apisecret)
    logging.info(f"Market bias for {symbol} is {market_bias}")

    if market_bias == "neutral":
        logging.info(f"Market bias is neutral for {symbol}, skipping order.")
        return

    # Determine order side based on market bias
    order_side = "BUY" if market_bias == "long" else "SELL"
    logging.info(f"Placing {order_side} order for {symbol}")
    symbol_info_qty = getQuantityPrecision(symbol)
    quantity = round(stake_amount / current_price, symbol_info_qty)
    # Calculate the correct limit price for closing the position
    limit_price = current_price * (1 - SLIPPAGE_PERCENT) if order_side == "BUY" else current_price * (1 + SLIPPAGE_PERCENT)

    logging.info(f"Placing {order_side} order for {symbol} at {limit_price}")
    if has_recent_loss(client, symbol):
        market_bias = coinGeckoAPI.get_file_sentiment(symbol, coinGeckoAPI.apikey, coinGeckoAPI.apisecret)
        logging.info(f"Market bias for {symbol} is {market_bias}")
        coinGeckoAPI.bias_determination = coinGeckoAPI.getBiasDetermination()
        logging.info(f"Bias Determination is {coinGeckoAPI.bias_determination}")
        logging.info(f"Skipping order for {symbol} due to recent loss.")
        return
    try:
        order = client.futures_create_order(
            symbol=symbol,
            side=order_side,
            type="LIMIT",
            price=round(limit_price, symbol_info["pricePrecision"]),
            quantity=quantity,
            timeInForce="GTC"
        )
        logging.info(f"Order placed: {order}")
    except BinanceAPIException as e:
        logging.error(f"Error placing buy order for {symbol}: {e}")
        

def getWhitelist():
    symbols = [
        #"BTCUSDT",
        #"ETHUSDT",
        #"SOLUSDT",
        "JTOUSDT",
        "DOTUSDT",
        "XRPUSDT",
        "XLMUSDT",
        "IOTAUSDT",
        "ALGOUSDT",
        "BCHUSDT",
        "DOGEUSDT",
        "GMXUSDT",
        "ADAUSDT"
    ]
    return [{"symbol": symbol} for symbol in symbols]    

def get_top_5_futures_pairs(client):
    return getWhitelist()

    url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
    try:
        response = requests.get(url)
        data = response.json()

        time.sleep(sleep)
        exchange_info = client.futures_exchange_info()
        valid_symbols = {s['symbol'] for s in exchange_info['symbols'] if s['contractType'] == 'PERPETUAL' and s['status'] != 'SETTLING'}
        filtered_data = [d for d in data if d['symbol'] in valid_symbols]

        sorted_pairs = sorted(filtered_data, key=lambda x: float(x['priceChangePercent']), reverse=True)
        return sorted_pairs[:5]
    except Exception as e:
        logging.error(f"Error fetching top futures pairs: {e}")
        return []

def get_symbol_info(client, symbol):
    try:
        time.sleep(sleep)
        exchange_info = client.futures_exchange_info()
        for s in exchange_info['symbols']:
            if s['symbol'] == symbol:
                return s
        logging.error(f"Symbol {symbol} not found.")
        return None
    except BinanceAPIException as e:
        logging.error(f"Error fetching symbol info: {e}")
        return None

def get_historical_klines(client, symbol, interval, limit=100):
    try:
        time.sleep(sleep)
        klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=["open_time", "open", "high", "low", "close", "volume", "close_time", "qav", "num_trades", "taker_base_vol", "taker_quote_vol", "ignore"])
        df = df[["high", "low", "close"]].astype(float)
        return df
    except BinanceAPIException as e:
        logging.error(f"Error fetching historical klines for {symbol}: {e}")
        return None

def calculate_atr(df, period=ATR_PERIOD):
    df['tr'] = df[['high', 'low', 'close']].apply(lambda x: max(x[0] - x[1], abs(x[0] - x[2]), abs(x[1] - x[2])), axis=1)
    return df['tr'].rolling(window=period).mean().iloc[-1]

def calculate_pnl_percentage(entry_price, mark_price, position_amt):
    pnl_percentage = ((mark_price - entry_price) / entry_price) * 100
    if position_amt < 0:  # Short position
        pnl_percentage *= -1
    return pnl_percentage

def monitor_and_manage(client, loss_threshold, profit_threshold, leverage, stake_amount):
    market_bias = coinGeckoAPI.get_file_sentiment(None, client.API_KEY, client.API_SECRET)
    logging.info(f"Market bias is {market_bias}")

    while True:
        time.sleep(sleep)
        coinGeckoAPI.bias_determination = coinGeckoAPI.getBiasDetermination()
        print(f"Bias Determination is {coinGeckoAPI.bias_determination}")
        open_positions = get_open_positions(client)
        top_5_pairs = get_top_5_futures_pairs(client)

        position_data = []

        for position in open_positions:
            symbol = position["symbol"]
            entry_price = float(position["entryPrice"])
            mark_price = float(position["markPrice"])
            position_amt = float(position["positionAmt"])
            df = get_historical_klines(client, symbol, "15m")
            
            if df is not None:
                atr = calculate_atr(df)
                loss_threshold = atr * loss_threshold  # ATR-based loss threshold
                profit_threshold = atr * profit_threshold  # ATR-based profit threshold
                pnl_percentage = calculate_pnl_percentage(entry_price, mark_price, position_amt)
                if pnl_percentage <= -loss_threshold:
                    logging.warning(f"Loss reached {pnl_percentage:.2f}% for {symbol}. Reversing position!")
                    reverse_position(client, position)
                elif pnl_percentage >= profit_threshold:
                    logging.info(f"Profit long reached {pnl_percentage:.2f}% for {symbol}. Closing position!")
                    close_position(client, position)
                elif market_bias == "long" and position_amt < 0:
                    logging.info(f"Market bias is long, reverting short position for {symbol} to long.")
                    reverse_position(client, position)
                elif market_bias == "short" and position_amt > 0:
                    logging.info(f"Market bias is short, reverting long position for {symbol} to short.")
                    reverse_position(client, position)
                    
        df = pd.DataFrame(position_data, columns=["Symbol", "Entry Price", "Current Price", "PnL (%)", f"{loss_threshold}% Loss Price", f"{profit_threshold}% Profit Price"])
        print(tabulate(df, headers="keys", tablefmt="pretty"))
        for pair in top_5_pairs:
            place_order_if_no_position(client, pair['symbol'], leverage, stake_amount)
            time.sleep(sleep)

        time.sleep(sleep)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Binance Futures Position Manager")
#     parser.add_argument("--apikey", required=True, help="Your Binance API Key")
#     parser.add_argument("--apisecret", required=True, help="Your Binance API Secret")
#     parser.add_argument("--loss", type=float, default=5.0, help="Loss percentage to trigger position reversal (default: 5%)")
#     parser.add_argument("--profit", type=float, default=6.0, help="Profit percentage to trigger position closure (default: 6%)")
#     parser.add_argument("--sleep", type=float, default=5.0, help="sleep in seconds before api call")
#     parser.add_argument("--leverage", type=float, default=5.0, help="leverage")
#     parser.add_argument("--stake", type=float, default=50.0, help="Stake amount per trade (default: 500 USDT)")

#     args = parser.parse_args()

#     coinGeckoAPI.coinGeckoAPIKey = "CG-AgEZRgMf3iLk1S8CwyCKp7N3"
#     coinGeckoAPI.apikey = args.apikey
#     coinGeckoAPI.apisecret = args.apisecret
#     sleep = args.sleep
#     leverage = args.leverage
#     stake_amount = args.stake

#     client = Client(args.apikey, args.apisecret)

#     logging.info(f"Monitoring positions for {args.loss}% loss threshold and {args.profit}% profit threshold...")
#     monitor_and_manage(client, args.loss, args.profit, leverage, stake_amount)


