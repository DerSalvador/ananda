import os
from bias.interface import BiasInterface, BiasRequest, BiasResponse, BiasType
from binance.client import Client
import pandas as pd


class BinanceTrend(BiasInterface):
    def get_candlestick_data(self, client, symbol, interval, limit=100):
        """
        Fetch candlestick (OHLC) data for a given symbol and interval.

        Args:
            client (Client): Binance API client.
            symbol (str): The trading pair (e.g., 'BTCUSDT').
            interval (str): Timeframe (e.g., '1d' for daily, '1h' for hourly).
            limit (int): Number of candles to fetch.

        Returns:
            pandas.DataFrame: Candlestick data.
        """
        candles = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        
        # Convert data to DataFrame
        df = pd.DataFrame(candles, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'num_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Keep only relevant columns and convert to numeric
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def detect_trend(self, df):
        df['SMA_20'] = df['close'].rolling(window=20).mean()  # 20-day SMA
        df['SMA_50'] = df['close'].rolling(window=50).mean()  # 50-day SMA

        # Determine trend
        def trend_logic(row):
            if row['SMA_20'] > row['SMA_50']:
                return BiasType.LONG
            elif row['SMA_20'] < row['SMA_50']:
                return BiasType.SHORT
            else:
                return BiasType.NEUTRAL

        df['trend'] = df.apply(trend_logic, axis=1)
        return df



    def bias(self, biasRequest: BiasRequest) -> BiasResponse:
        latest_trend = BiasType.NEUTRAL
        # Initialize Binance API client
        apikey = os.getenv("BINANCE_API_KEY", "")
        apisecret = os.getenv("BINANCE_API_SECRET", "")
        client = Client(apikey, apisecret)

        # Fetch BTC/USDT daily data
        symbol = biasRequest.symbol + "USDT"
        interval = "1d"
        df = self.get_candlestick_data(client, symbol, interval)

        # Calculate SMA and detect trends
        df = self.detect_trend(df)
        latest_trend = df.iloc[-1]['trend']
        return BiasResponse(bias=latest_trend, usedSymbol=True)
