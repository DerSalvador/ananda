from bias.greed_and_fear import GreedAndFear
from bias.binance_trend import BinanceTrend
from bias.coin_gecko_btc import CoinGeckoBTC
from bias.coin_gecko_global import CoinGeckoGlobal
from bias.coin_gecko_market import CoinGeckoMarket


INTERFACES = {
    "coin_gecko_market": CoinGeckoMarket(),
    "coin_gecko_global": CoinGeckoGlobal(),
    "coin_gecko_BTC": CoinGeckoBTC(),
    "binance_trend": BinanceTrend(),
    "greed_and_fear": GreedAndFear()
}
