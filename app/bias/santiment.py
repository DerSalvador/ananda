from functools import cache
import os
import json
import requests
from datetime import datetime, timedelta
from typing import Optional
from bias import get_config
from utils import get_logger
from bias.interface import BiasInterface, BiasRequest, BiasResponse, BiasType

logger = get_logger()


class SantimentBias(BiasInterface):
    paid = True

    def __init__(self):
        self.api_key = os.getenv("SANTIMENT_API_KEY", "")
        self.metric = "sentiment_weighted_total_1d_v2"

    @cache
    def get_slug(self, ticker: str) -> str:
        for asset in SANTIMENT_ASSETS["data"]["allProjects"]:
            if asset["ticker"].lower() == ticker.lower():
                return asset["slug"]
        logger.warning(f"Ticker '{ticker}' not found in Santiment assets.")
        return "bitcoin"

    def iso_date(self, days_ago: int = 0) -> str:
        dt = datetime.utcnow() - timedelta(days=days_ago)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    def fetch_sentiment_data(self, symbol: str) -> Optional[list[float]]:
        headers = {
            "Content-Type": "application/graphql",
            "Authorization": f"Apikey {self.api_key}",
        }

        SantimentFromTimeDaysAgo = int(get_config("SantimentFromTimeDaysAgo", 7))
        SantimentToTimeDaysAgo = int(get_config("SantimentToTimeDaysAgo", 0))
        from_time = self.iso_date(SantimentFromTimeDaysAgo)
        to_time = self.iso_date(SantimentToTimeDaysAgo)

        query = f"""
        {{
            getMetric(metric: "{self.metric}") {{
                timeseriesData(
                    slug: "{self.get_slug(symbol)}",
                    from: "{from_time}",
                    to: "{to_time}",
                    interval: "1d"
                ) {{
                    datetime
                    value
                }}
            }}
        }}
        """

        try:
            logger.info(f"Requesting sentiment data for '{symbol}' from {from_time} to {to_time}")
            response = requests.post(
                "https://api.santiment.net/graphql",
                headers=headers,
                data=query,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            logger.debug(data)

            points = data["data"]["getMetric"]["timeseriesData"]
            values = [point["value"] for point in points if point["value"] is not None]

            logger.info(f"Fetched {len(values)} sentiment values")
            return values[-6:]  # Last 6: previous 5 + latest
        except Exception as e:
            logger.exception(f"Error fetching data from Santiment: {e}")
            return None

    def bias(self, biasRequest: BiasRequest) -> BiasResponse:
        values = self.fetch_sentiment_data(biasRequest.symbol)

        if values is None or len(values) < 6:
            return BiasResponse(
                bias=BiasType.NEUTRAL,
                error="Not enough data to determine signal.",
                usedSymbol=True,
                reason="Santiment API returned insufficient or invalid data."
            )

        *prev5, latest = values[-6:]  # Unpack last 6 into prev5 and latest
        average = sum(prev5) / len(prev5)
        diff = latest - average
        threshold = float(get_config("SantimentThreshold", 0.25))

        logger.info(f"Latest sentiment: {latest:.2f}")
        logger.info(f"Previous 5-day average: {average:.2f}")
        logger.info(f"Difference: {diff:.4f} | Threshold: ±{threshold}")

        if diff > threshold:
            signal = BiasType.LONG
            reason = f"Signal: LONG — Diff {diff:.4f} > Threshold {threshold}"
        elif diff < -threshold:
            signal = BiasType.SHORT
            reason = f"Signal: SHORT — Diff {diff:.4f} < -Threshold {-threshold}"
        else:
            signal = BiasType.NEUTRAL
            reason = f"Signal: HOLD — Diff {diff:.4f} within ±{threshold}"

        return BiasResponse(
            bias=signal,
            usedSymbol=True,
            reason=reason
        )

SANTIMENT_ASSETS = {
    "data": {
      "allProjects": [
        {
          "name": "FACT0RN",
          "slug": "fact0rn",
          "ticker": "FACT"
        },
        {
          "name": "Enecuum",
          "slug": "enecuum",
          "ticker": "ENQ"
        },
        {
          "name": "Cosmos",
          "slug": "cosmos",
          "ticker": "ATOM"
        },
        {
          "name": "ZeuxCoin",
          "slug": "zeuxcoin",
          "ticker": "ZUC"
        },
        {
          "name": "Treelion",
          "slug": "treelion",
          "ticker": "TRN"
        },
        {
          "name": "Kava",
          "slug": "kava",
          "ticker": "KAVA"
        },
        {
          "name": "Synth sLINK",
          "slug": "slink",
          "ticker": "sLINK"
        },
        {
          "name": "Levana Protocol",
          "slug": "levana-protocol",
          "ticker": "LVN"
        },
        {
          "name": "Wojak",
          "slug": "wojak",
          "ticker": "WOJAK"
        },
        {
          "name": "TrueFi",
          "slug": "truefi-token",
          "ticker": "TRU"
        },
        {
          "name": "HyperExchange",
          "slug": "hyperexchange",
          "ticker": "HX"
        },
        {
          "name": "MAGA",
          "slug": "maga-ethereum",
          "ticker": "MAGA"
        },
        {
          "name": "Euro Tether",
          "slug": "tether-eurt",
          "ticker": "EURt"
        },
        {
          "name": "MargiX",
          "slug": "margix",
          "ticker": "MGX"
        },
        {
          "name": "EOS Force",
          "slug": "eos-force",
          "ticker": "EOSC"
        },
        {
          "name": "YottaChain",
          "slug": "yottachain",
          "ticker": "YTA"
        },
        {
          "name": "Anchor",
          "slug": "anchor",
          "ticker": "ANCT"
        },
        {
          "name": "SEED Pool Token",
          "slug": "pseed",
          "ticker": "pSEED"
        },
        {
          "name": "AllianceBlock",
          "slug": "allianceblock",
          "ticker": "ALBT"
        },
        {
          "name": "0x Tracker",
          "slug": "0xtracker",
          "ticker": "ZRT"
        },
        {
          "name": "X World Games",
          "slug": "x-world-games",
          "ticker": "XWG"
        },
        {
          "name": "Electra Protocol",
          "slug": "electra-protocol",
          "ticker": "XEP"
        },
        {
          "name": "7Eleven",
          "slug": "7eleven",
          "ticker": "7E"
        },
        {
          "name": "Yap Stone",
          "slug": "yap-stone",
          "ticker": "YAP"
        },
        {
          "name": "Dash Cash",
          "slug": "dash-cash",
          "ticker": "DSC"
        },
        {
          "name": "Hintchain",
          "slug": "hintchain",
          "ticker": "HINT"
        },
        {
          "name": "Azbit",
          "slug": "azbit",
          "ticker": "AZ"
        },
        {
          "name": "Project WITH",
          "slug": "project-with",
          "ticker": "WIKEN"
        },
        {
          "name": "Chiliz",
          "slug": "chiliz",
          "ticker": "CHZ"
        },
        {
          "name": "DeepCloud AI",
          "slug": "deepcloud-ai",
          "ticker": "DEEP"
        },
        {
          "name": "Mint Club",
          "slug": "mint-club",
          "ticker": "MINT"
        },
        {
          "name": "Pokeball",
          "slug": "pokeball",
          "ticker": "POKE"
        },
        {
          "name": "Uniswap V2: DAI-USDC",
          "slug": "uniswap_dai_usdc_lp",
          "ticker": "UNI-V2 DAI-USDC LP"
        },
        {
          "name": "Uniswap V2: AAVE",
          "slug": "uniswap_aave_eth_lp",
          "ticker": "UNI-V2 AAVE/ETH LP"
        },
        {
          "name": "Sessia",
          "slug": "sessia",
          "ticker": "KICKS"
        },
        {
          "name": "Polkacover",
          "slug": "polkacover",
          "ticker": "CVR"
        },
        {
          "name": "Helena",
          "slug": "helena",
          "ticker": "HLN"
        },
        {
          "name": "Bitfex",
          "slug": "bitfex",
          "ticker": "BFX"
        },
        {
          "name": "HUSD",
          "slug": "husd",
          "ticker": "HUSD"
        },
        {
          "name": "Fei USD",
          "slug": "fei-protocol",
          "ticker": "FEI"
        },
        {
          "name": "DAD",
          "slug": "dad-chain",
          "ticker": "DAD"
        },
        {
          "name": "MOG Coin",
          "slug": "mog-coin",
          "ticker": "MOG"
        },
        {
          "name": "Helium ",
          "slug": "helium",
          "ticker": "HNT"
        },
        {
          "name": "Augmint",
          "slug": "augmint",
          "ticker": "AUG"
        },
        {
          "name": "Uniswap V2: SNX",
          "slug": "uniswap_snx_eth_lp",
          "ticker": "UNI-V2 SNX/ETH LP "
        },
        {
          "name": "Define",
          "slug": "define",
          "ticker": "DFA"
        },
        {
          "name": "Penpie",
          "slug": "penpie",
          "ticker": "PNP"
        },
        {
          "name": "MB8 Coin",
          "slug": "mb8-coin",
          "ticker": "MB8"
        },
        {
          "name": "ChronoCoin",
          "slug": "chronocoin",
          "ticker": "CRN"
        },
        {
          "name": "Nitrogen Network",
          "slug": "nitrogen",
          "ticker": "NGN"
        },
        {
          "name": "Totle",
          "slug": "totle",
          "ticker": "TTL"
        },
        {
          "name": "Ash",
          "slug": "ash",
          "ticker": "ASH"
        },
        {
          "name": "LINKA",
          "slug": "linka",
          "ticker": "LINKA"
        },
        {
          "name": "Karura",
          "slug": "karura",
          "ticker": "KAR"
        },
        {
          "name": "Poketto",
          "slug": "poketto",
          "ticker": "PKT"
        },
        {
          "name": "Tesra",
          "slug": "tesra",
          "ticker": "TSR"
        },
        {
          "name": "INLOCK",
          "slug": "inlock",
          "ticker": "ILK"
        },
        {
          "name": "SuperRare",
          "slug": "superrare",
          "ticker": "RARE"
        },
        {
          "name": "XVIX",
          "slug": "xvix",
          "ticker": "XVIX"
        },
        {
          "name": "Zero1 Labs",
          "slug": "zero1-labs",
          "ticker": "DEAI"
        },
        {
          "name": "CryptoBossCoin",
          "slug": "cryptobosscoin",
          "ticker": "CBC"
        },
        {
          "name": "CoinHe Token",
          "slug": "coinhe-token",
          "ticker": "CHT"
        },
        {
          "name": "Flow",
          "slug": "flow",
          "ticker": "FLOW"
        },
        {
          "name": "Xensor",
          "slug": "xensor",
          "ticker": "XSR"
        },
        {
          "name": "Stake DAO",
          "slug": "stake-dao",
          "ticker": "SDT"
        },
        {
          "name": "FiatDex Gateway",
          "slug": "fiatdex-gateway",
          "ticker": "FDG"
        },
        {
          "name": "Edgecoin",
          "slug": "edgecoin",
          "ticker": "EDGT"
        },
        {
          "name": "Zerion",
          "slug": "zerion",
          "ticker": "ZRN"
        },
        {
          "name": "ALP Coin",
          "slug": "alp-coin",
          "ticker": "ALP"
        },
        {
          "name": "Balancer [on Ethereum]",
          "slug": "balancer",
          "ticker": "BAL"
        },
        {
          "name": "Nuo Network",
          "slug": "nuo",
          "ticker": "NUO"
        },
        {
          "name": "UMA",
          "slug": "uma",
          "ticker": "UMA"
        },
        {
          "name": "BenePit Protocol",
          "slug": "benepit-protocol",
          "ticker": "BNP"
        },
        {
          "name": "SSV Network",
          "slug": "ssv-network",
          "ticker": "SSV"
        },
        {
          "name": "Set Protocol",
          "slug": "set-protocol",
          "ticker": "SPX"
        },
        {
          "name": "Litecoin",
          "slug": "litecoin",
          "ticker": "LTC"
        },
        {
          "name": "Typhoon Cash",
          "slug": "typhoon-cash",
          "ticker": "PHOON"
        },
        {
          "name": "Uniswap V2: WBTC-USDC",
          "slug": "uniswap_wbtc_usdc_lp",
          "ticker": "UNI-V2 WBTC/USDC LP"
        },
        {
          "name": "MimbleWimbleCoin",
          "slug": "mimblewimblecoin",
          "ticker": "MWC"
        },
        {
          "name": "Razor Network",
          "slug": "razor",
          "ticker": "RZR"
        },
        {
          "name": "Uniswap V2: BAT",
          "slug": "uniswap_bat_eth_lp",
          "ticker": "UNI-V2 BAT/ETH LP"
        },
        {
          "name": "Snax",
          "slug": "snax",
          "ticker": "SNAX"
        },
        {
          "name": "PegaSys",
          "slug": "pegasys",
          "ticker": "PSS"
        },
        {
          "name": "Chintai",
          "slug": "chintai",
          "ticker": "CHN"
        },
        {
          "name": "Bitcoin Diamond",
          "slug": "bitcoin-diamond",
          "ticker": "BCD"
        },
        {
          "name": "Coinzo Token",
          "slug": "coinzo",
          "ticker": "CNZ"
        },
        {
          "name": "Seedworld",
          "slug": "seedworld",
          "ticker": "SWORLD"
        },
        {
          "name": "MARKET Protocol",
          "slug": "market-protocol",
          "ticker": "MPX"
        },
        {
          "name": "JulSwap",
          "slug": "julswap",
          "ticker": "JULD"
        },
        {
          "name": "Synth sBCH",
          "slug": "sbch",
          "ticker": "sBCH"
        },
        {
          "name": "Transledger",
          "slug": "transledger",
          "ticker": "TRN"
        },
        {
          "name": "Punk",
          "slug": "punk",
          "ticker": "PUNK"
        },
        {
          "name": "Kambria",
          "slug": "kambria",
          "ticker": "KAT"
        },
        {
          "name": "Trog",
          "slug": "trog",
          "ticker": "TROG"
        },
        {
          "name": "X-power Chain",
          "slug": "xpower",
          "ticker": "XPO"
        },
        {
          "name": "Eminer",
          "slug": "eminer",
          "ticker": "EM"
        },
        {
          "name": "Pledgecamp",
          "slug": "pledge-coin",
          "ticker": "PLG"
        },
        {
          "name": "Basis Cash",
          "slug": "basis-cash",
          "ticker": "BAC"
        },
        {
          "name": "inSure DeFi",
          "slug": "insure",
          "ticker": "SURE"
        },
        {
          "name": "Foxsy AI",
          "slug": "foxsy-ai",
          "ticker": "FOXSY"
        },
        {
          "name": "Simone",
          "slug": "simone",
          "ticker": "SON"
        },
        {
          "name": "Celo Dollar",
          "slug": "celo-dollar",
          "ticker": "CUSD"
        },
        {
          "name": "Game.com",
          "slug": "game",
          "ticker": "GTC"
        },
        {
          "name": "EMOGI Network",
          "slug": "emogi-network",
          "ticker": "LOL"
        },
        {
          "name": "Insured Finance",
          "slug": "insured-finance",
          "ticker": "INFI"
        },
        {
          "name": "Ref Finance",
          "slug": "ref-finance",
          "ticker": "REF"
        },
        {
          "name": "12Ships",
          "slug": "12ships",
          "ticker": "TSHP"
        },
        {
          "name": "PLANET",
          "slug": "planet",
          "ticker": "PLA"
        },
        {
          "name": "Lightning Network",
          "slug": "lightning-network",
          "ticker": "LNN"
        },
        {
          "name": "YAWN",
          "slug": "yawn",
          "ticker": "$YAWN"
        },
        {
          "name": "Tokoin",
          "slug": "tokoin",
          "ticker": "TOKO"
        },
        {
          "name": "FIBOS",
          "slug": "fibos",
          "ticker": "FO"
        },
        {
          "name": "CryptoAutos",
          "slug": "cryptoautos",
          "ticker": "AUTOS"
        },
        {
          "name": "Cryptocean",
          "slug": "cryptocean",
          "ticker": "CRON"
        },
        {
          "name": "VideoCoin",
          "slug": "videocoin",
          "ticker": "VID"
        },
        {
          "name": "MixMarvel",
          "slug": "mixmarvel",
          "ticker": "MIX"
        },
        {
          "name": "Sealchain",
          "slug": "sealchain",
          "ticker": "SEAL"
        },
        {
          "name": "WALL STREET BABY",
          "slug": "wall-street-bet",
          "ticker": "WSB"
        },
        {
          "name": "CryptoBonusMiles",
          "slug": "cryptobonusmiles",
          "ticker": "CBM"
        },
        {
          "name": "VIDY",
          "slug": "vidy",
          "ticker": "VIDY"
        },
        {
          "name": "Vodi X",
          "slug": "vodi-x",
          "ticker": "VDX"
        },
        {
          "name": "Wavesbet",
          "slug": "wavesbet",
          "ticker": "WBET"
        },
        {
          "name": "Splintershards",
          "slug": "splintershards",
          "ticker": "SPS"
        },
        {
          "name": "Yobit Token",
          "slug": "yobit",
          "ticker": "YO"
        },
        {
          "name": "Rotten",
          "slug": "rotten",
          "ticker": "ROT"
        },
        {
          "name": "Synth sDEFI",
          "slug": "sdefi",
          "ticker": "sDEFI"
        },
        {
          "name": "NexDAX Chain",
          "slug": "nexdax-chain",
          "ticker": "NT"
        },
        {
          "name": "UGOLD Inc.",
          "slug": "ugold-inc",
          "ticker": "UGOLD"
        },
        {
          "name": "QLC Chain",
          "slug": "qlink",
          "ticker": "QLC"
        },
        {
          "name": "DeFi Socks",
          "slug": "defisocks",
          "ticker": "DEFISOCKS"
        },
        {
          "name": "Plugin Decentralized Oracle",
          "slug": "plugin",
          "ticker": "PLI"
        },
        {
          "name": "Amino Network",
          "slug": "amino",
          "ticker": "AMIO"
        },
        {
          "name": "Synth sXAG",
          "slug": "sxag",
          "ticker": "sXAG"
        },
        {
          "name": "NSS Coin",
          "slug": "nss",
          "ticker": "NSS"
        },
        {
          "name": "Valobit",
          "slug": "valobit",
          "ticker": "VBIT"
        },
        {
          "name": "Aryacoin",
          "slug": "aryacoin",
          "ticker": "AYA"
        },
        {
          "name": "FU Coin",
          "slug": "fu-coin",
          "ticker": "FU"
        },
        {
          "name": "Book.io",
          "slug": "book-io",
          "ticker": "STUFF"
        },
        {
          "name": "En-Tan-Mo",
          "slug": "en-tan-mo",
          "ticker": "ETM"
        },
        {
          "name": "Decentralized Vulnerability Platform",
          "slug": "decentralized-vulnerability-platform",
          "ticker": "DVP"
        },
        {
          "name": "Seedify.fund",
          "slug": "seedify-fund",
          "ticker": "SFUND"
        },
        {
          "name": "ChatCoin",
          "slug": "chatcoin",
          "ticker": "CHAT"
        },
        {
          "name": "Alltoscan",
          "slug": "alltoscan",
          "ticker": "ATS"
        },
        {
          "name": "CNNS",
          "slug": "cnns",
          "ticker": "CNNS"
        },
        {
          "name": "Molecular Future",
          "slug": "molecular-future",
          "ticker": "MOF"
        },
        {
          "name": "Metaplex",
          "slug": "metaplex",
          "ticker": "MPLX"
        },
        {
          "name": "Solvex Network",
          "slug": "privapp-network",
          "ticker": "SOLVEX"
        },
        {
          "name": "FLock.io",
          "slug": "flock-io",
          "ticker": "FLOCK"
        },
        {
          "name": "EUR CoinVertible",
          "slug": "eur-coinvertible",
          "ticker": "EURCV"
        },
        {
          "name": "Artfinity",
          "slug": "artfinity",
          "ticker": "AT"
        },
        {
          "name": "Autonio",
          "slug": "autonio",
          "ticker": "NIOX"
        },
        {
          "name": "WHEN Token",
          "slug": "when-token",
          "ticker": "WHEN"
        },
        {
          "name": "Safe Exchange",
          "slug": "safe-exchange-coin",
          "ticker": "SAFEX"
        },
        {
          "name": "ZeroNet",
          "slug": "zeronet",
          "ticker": "ZNT"
        },
        {
          "name": "Spiking",
          "slug": "spiking",
          "ticker": "SPIKE"
        },
        {
          "name": "GoPower",
          "slug": "gopower",
          "ticker": "GPT"
        },
        {
          "name": "Dai [on Optimism]",
          "slug": "o-multi-collateral-dai",
          "ticker": "DAI"
        },
        {
          "name": "Locus Chain",
          "slug": "locus-chain",
          "ticker": "LOCUS"
        },
        {
          "name": "Moca Network",
          "slug": "mocaverse",
          "ticker": "MOCA"
        },
        {
          "name": "NikolAI",
          "slug": "nikolai",
          "ticker": "NIKO"
        },
        {
          "name": "Flowchain",
          "slug": "flowchain",
          "ticker": "FLC"
        },
        {
          "name": "Trias",
          "slug": "trias",
          "ticker": "TRY"
        },
        {
          "name": "QuickX Protocol",
          "slug": "quickx-protocol",
          "ticker": "QCX"
        },
        {
          "name": "ChartEx",
          "slug": "chartex",
          "ticker": "CHART"
        },
        {
          "name": "PATHHIVE",
          "slug": "phv",
          "ticker": "PHV"
        },
        {
          "name": "BOLT",
          "slug": "bolt",
          "ticker": "BOLT"
        },
        {
          "name": "PulsePad",
          "slug": "pulsepad",
          "ticker": "PLSPAD"
        },
        {
          "name": "TOKOK",
          "slug": "tokok",
          "ticker": "TOK"
        },
        {
          "name": "INMAX",
          "slug": "inmax",
          "ticker": "INX"
        },
        {
          "name": "Lambda",
          "slug": "lambda",
          "ticker": "LAMB"
        },
        {
          "name": "Birdchain",
          "slug": "birdchain",
          "ticker": "BIRD"
        },
        {
          "name": "Yearn Finance DOT",
          "slug": "yearn-finance-dot",
          "ticker": "YFDOT"
        },
        {
          "name": "Contents Protocol",
          "slug": "contents-protocol",
          "ticker": "CPT"
        },
        {
          "name": "AdEx",
          "slug": "adx-net",
          "ticker": "ADX"
        },
        {
          "name": "Tribe",
          "slug": "tribe",
          "ticker": "TRIBE"
        },
        {
          "name": "P2P Global Network",
          "slug": "p2p-global-network",
          "ticker": "P2PX"
        },
        {
          "name": "Paragon",
          "slug": "paragon",
          "ticker": "PRG"
        },
        {
          "name": "Asian Fintech",
          "slug": "asian-fintech",
          "ticker": "AFIN"
        },
        {
          "name": "Synth sCHF",
          "slug": "schf",
          "ticker": "sCHF"
        },
        {
          "name": "Synth sADA",
          "slug": "sada",
          "ticker": "sADA"
        },
        {
          "name": "Synth sDASH",
          "slug": "sdash",
          "ticker": "sDASH"
        },
        {
          "name": "HyperGPT",
          "slug": "hypergpt",
          "ticker": "HGPT"
        },
        {
          "name": "S4FE",
          "slug": "s4fe",
          "ticker": "S4F"
        },
        {
          "name": "GMB",
          "slug": "gmb",
          "ticker": "GMB"
        },
        {
          "name": "WAX",
          "slug": "wax",
          "ticker": "WAXP"
        },
        {
          "name": "Zero Collateral",
          "slug": "zero-collateral",
          "ticker": "ZCT"
        },
        {
          "name": "Online",
          "slug": "online",
          "ticker": "OIO"
        },
        {
          "name": "CYCLUB",
          "slug": "cyclub",
          "ticker": "CYCLUB"
        },
        {
          "name": "Bitrue Coin",
          "slug": "bitrue-coin",
          "ticker": "BTR"
        },
        {
          "name": "Lisk Machine Learning",
          "slug": "lisk-machine-learning",
          "ticker": "LML"
        },
        {
          "name": "Wrapped Matic",
          "slug": "wmatic",
          "ticker": "WMATIC"
        },
        {
          "name": "Xriba",
          "slug": "xriba",
          "ticker": "XRA"
        },
        {
          "name": "Almeela ",
          "slug": "almeela",
          "ticker": "KZE"
        },
        {
          "name": "RENEC",
          "slug": "renec",
          "ticker": "RENEC"
        },
        {
          "name": "Coineal Token",
          "slug": "coineal-token",
          "ticker": "NEAL"
        },
        {
          "name": "BUDDY",
          "slug": "buddy",
          "ticker": "BUD"
        },
        {
          "name": "BuySell",
          "slug": "buysell",
          "ticker": "BULL"
        },
        {
          "name": "Cryptopay",
          "slug": "cryptopay",
          "ticker": "CPAY"
        },
        {
          "name": "DIGG",
          "slug": "digg",
          "ticker": "DIGG"
        },
        {
          "name": "Synth iOIL",
          "slug": "ioil",
          "ticker": "iOIL"
        },
        {
          "name": "NKN",
          "slug": "nkn",
          "ticker": "NKN"
        },
        {
          "name": "Synth sETC",
          "slug": "setc",
          "ticker": "sETC"
        },
        {
          "name": "HashNet BitEco",
          "slug": "hashnet-biteco",
          "ticker": "HNB"
        },
        {
          "name": "StackOs",
          "slug": "stackos",
          "ticker": "STACK"
        },
        {
          "name": "Carbon browser",
          "slug": "carbon-browser",
          "ticker": "CSIX"
        },
        {
          "name": "Ozone Chain",
          "slug": "ozone-chain",
          "ticker": "OZO"
        },
        {
          "name": "FOGNET",
          "slug": "fognet",
          "ticker": "FOG"
        },
        {
          "name": "Capitalrock",
          "slug": "capitalrock",
          "ticker": "CR"
        },
        {
          "name": "QANplatform",
          "slug": "qanplatform",
          "ticker": "QANX"
        },
        {
          "name": "EveryCoin",
          "slug": "everycoin",
          "ticker": "EVY"
        },
        {
          "name": "IZIChain",
          "slug": "izichain",
          "ticker": "IZI"
        },
        {
          "name": "InnovaMinex",
          "slug": "innovaminex",
          "ticker": "MINX"
        },
        {
          "name": "Quadrant Protocol",
          "slug": "quadrantprotocol",
          "ticker": "EQUAD"
        },
        {
          "name": "SpiderDAO",
          "slug": "spiderdao",
          "ticker": "SPDR"
        },
        {
          "name": "Q DAO Governance token v1.0",
          "slug": "q-dao-governance-token",
          "ticker": "QDAO"
        },
        {
          "name": "RWA Inc.",
          "slug": "rwa-inc",
          "ticker": "RWA"
        },
        {
          "name": "MOBOX",
          "slug": "mobox",
          "ticker": "MBOX"
        },
        {
          "name": "ARMOR",
          "slug": "armor",
          "ticker": "ARMOR"
        },
        {
          "name": "Sharpe AI",
          "slug": "sharpe-ai",
          "ticker": "SAI"
        },
        {
          "name": "r/FortNiteBR Bricks",
          "slug": "bricks",
          "ticker": "BRICK"
        },
        {
          "name": "Orbitt Token",
          "slug": "orbitt-token",
          "ticker": "ORBT"
        },
        {
          "name": "Hydranet",
          "slug": "hydranet",
          "ticker": "HDN"
        },
        {
          "name": "DEX",
          "slug": "dex",
          "ticker": "DEX"
        },
        {
          "name": "Hxro",
          "slug": "hxro",
          "ticker": "HXRO"
        },
        {
          "name": "PTON",
          "slug": "pton",
          "ticker": "PTON"
        },
        {
          "name": "Bonpay",
          "slug": "bonpay",
          "ticker": "BON"
        },
        {
          "name": "SwftCoin",
          "slug": "swftcoin",
          "ticker": "SWFTC"
        },
        {
          "name": "Levolution",
          "slug": "levolution",
          "ticker": "LEVL"
        },
        {
          "name": "IOU",
          "slug": "iou",
          "ticker": "IOUX"
        },
        {
          "name": "SpaceChain",
          "slug": "spacechain",
          "ticker": "SPC"
        },
        {
          "name": "Multichain [on Ethereum]",
          "slug": "multichain",
          "ticker": "MULTI"
        },
        {
          "name": "ParkinGo",
          "slug": "parkingo",
          "ticker": "GOT"
        },
        {
          "name": "Qredo",
          "slug": "qredo",
          "ticker": "OPEN"
        },
        {
          "name": "StakeWise Staked ETH",
          "slug": "staked-eth",
          "ticker": "osETH"
        },
        {
          "name": "Hercules",
          "slug": "hercules",
          "ticker": "HERC"
        },
        {
          "name": "CoinUs",
          "slug": "coinus",
          "ticker": "CNUS"
        },
        {
          "name": "Assemble Protocol",
          "slug": "assemble-protocol",
          "ticker": "ASM"
        },
        {
          "name": "Skillful AI",
          "slug": "skillful-ai",
          "ticker": "SKAI"
        },
        {
          "name": "BOSAGORA",
          "slug": "bosagora",
          "ticker": "BOA"
        },
        {
          "name": "Skychain",
          "slug": "skychain",
          "ticker": "SCH"
        },
        {
          "name": "First Digital USD [on Binance]",
          "slug": "bnb-first-digital-usd",
          "ticker": "FDUSD"
        },
        {
          "name": "Shiba Predator",
          "slug": "shiba-predator",
          "ticker": "QOM"
        },
        {
          "name": "FLETA",
          "slug": "fleta",
          "ticker": "FLETA"
        },
        {
          "name": "EXRNchain",
          "slug": "exrnchain",
          "ticker": "EXRN"
        },
        {
          "name": "Internxt",
          "slug": "internxt",
          "ticker": "INXT"
        },
        {
          "name": "Fountain",
          "slug": "fountain",
          "ticker": "FTN"
        },
        {
          "name": "Silly Dragon",
          "slug": "silly-dragon",
          "ticker": "SILLY"
        },
        {
          "name": "Carry",
          "slug": "carry",
          "ticker": "CRE"
        },
        {
          "name": "AIDUS TOKEN",
          "slug": "aidus-token",
          "ticker": "AID"
        },
        {
          "name": "Shyft Network",
          "slug": "shyft-network",
          "ticker": "SHFT"
        },
        {
          "name": "Meta Games Coin",
          "slug": "meta-games-coin",
          "ticker": "MGC"
        },
        {
          "name": "BlackFort Exchange Network",
          "slug": "blackfort-exchange-network",
          "ticker": "BXN"
        },
        {
          "name": "IntelliShare",
          "slug": "intellishare",
          "ticker": "INE"
        },
        {
          "name": "B2BX",
          "slug": "b2bx",
          "ticker": "B2B"
        },
        {
          "name": "SOLVE",
          "slug": "solve",
          "ticker": "SOLVE"
        },
        {
          "name": "lisUSD",
          "slug": "lisusd",
          "ticker": "lisUSD"
        },
        {
          "name": "Tchain",
          "slug": "tchain",
          "ticker": "TCH"
        },
        {
          "name": "CryptoFranc",
          "slug": "cryptofranc",
          "ticker": "XCHF"
        },
        {
          "name": "Bitcoin Atom",
          "slug": "bitcoin-atom",
          "ticker": "BCA"
        },
        {
          "name": "Solana",
          "slug": "solana",
          "ticker": "SOL"
        },
        {
          "name": "DuckDaoDime",
          "slug": "duckdaodime",
          "ticker": "DDIM"
        },
        {
          "name": "SUNDOG",
          "slug": "sundog",
          "ticker": "SUNDOG"
        },
        {
          "name": "Galeon",
          "slug": "galeon",
          "ticker": "GALEON"
        },
        {
          "name": "Evedo",
          "slug": "evedo",
          "ticker": "EVED"
        },
        {
          "name": "DREP",
          "slug": "drep",
          "ticker": "DREP"
        },
        {
          "name": "TigerCash",
          "slug": "tigercash",
          "ticker": "TCH"
        },
        {
          "name": "Clearpool",
          "slug": "clearpool",
          "ticker": "CPOOL"
        },
        {
          "name": "COVA",
          "slug": "cova",
          "ticker": "COVA"
        },
        {
          "name": "Bodhi",
          "slug": "bodhi",
          "ticker": "BOT"
        },
        {
          "name": "FOAM Protocol",
          "slug": "foam",
          "ticker": "FOAM"
        },
        {
          "name": "Ctomorrow Platform",
          "slug": "ctomorrow-platform",
          "ticker": "CTP"
        },
        {
          "name": "Vertex Protocol",
          "slug": "vertex-protocol",
          "ticker": "VRTX"
        },
        {
          "name": "Tidalflats",
          "slug": "tidalflats",
          "ticker": "TIDE"
        },
        {
          "name": "AIT Protocol",
          "slug": "ait-protocol",
          "ticker": "AIT"
        },
        {
          "name": "zkLink",
          "slug": "zklink",
          "ticker": "ZKL"
        },
        {
          "name": "Suzuverse",
          "slug": "suzuverse",
          "ticker": "SGT"
        },
        {
          "name": "Turtlecoin",
          "slug": "turtlecoin",
          "ticker": "TRTL"
        },
        {
          "name": "Binemon",
          "slug": "binemon",
          "ticker": "BIN"
        },
        {
          "name": "Sharder",
          "slug": "sharder",
          "ticker": "SS"
        },
        {
          "name": "The Currency Analytics",
          "slug": "the-currency-analytics",
          "ticker": "TCAT"
        },
        {
          "name": "Upfiring",
          "slug": "upfiring",
          "ticker": "UFR"
        },
        {
          "name": "Tratok",
          "slug": "tratok",
          "ticker": "TRAT"
        },
        {
          "name": "Akash Network",
          "slug": "akash-network",
          "ticker": "AKT"
        },
        {
          "name": "Numeraire",
          "slug": "numeraire",
          "ticker": "NMR"
        },
        {
          "name": "Unicly CryptoPunks Collection",
          "slug": "unicly-cryptopunks-collection",
          "ticker": "UPUNK"
        },
        {
          "name": "DigitalBits",
          "slug": "digitalbits",
          "ticker": "XDB"
        },
        {
          "name": "Gomics",
          "slug": "gomics",
          "ticker": "GOM"
        },
        {
          "name": "IQ",
          "slug": "iq",
          "ticker": "IQ"
        },
        {
          "name": "Gnosis",
          "slug": "gnosis-gno",
          "ticker": "GNO"
        },
        {
          "name": "Habibi",
          "slug": "habibi-cat",
          "ticker": "HABIBI"
        },
        {
          "name": "Electric Vehicle Zone",
          "slug": "electric-vehicle-zone",
          "ticker": "EVZ"
        },
        {
          "name": "sudeng",
          "slug": "sudeng",
          "ticker": "HIPPO"
        },
        {
          "name": "Playkey",
          "slug": "playkey",
          "ticker": "PKT"
        },
        {
          "name": "PolkaBridge",
          "slug": "polkabridge",
          "ticker": "PBR"
        },
        {
          "name": "FirmaChain",
          "slug": "firmachain",
          "ticker": "FCT"
        },
        {
          "name": "O3Swap",
          "slug": "o3swap",
          "ticker": "O3"
        },
        {
          "name": "Huobi Token",
          "slug": "huobi-token",
          "ticker": "HT"
        },
        {
          "name": "Frax [on Optimism]",
          "slug": "o-frax",
          "ticker": "FRAX"
        },
        {
          "name": "nomnom",
          "slug": "nomnom",
          "ticker": "NOMNOM"
        },
        {
          "name": "Lattice Token",
          "slug": "lattice-token",
          "ticker": "LTX"
        },
        {
          "name": "Coinvest",
          "slug": "coinvest",
          "ticker": "COIN"
        },
        {
          "name": "Cherry Swap",
          "slug": "cherry-swap",
          "ticker": "CSW"
        },
        {
          "name": "Bisq",
          "slug": "bisq",
          "ticker": "BISQ"
        },
        {
          "name": "pEOS",
          "slug": "peos",
          "ticker": "PEOS"
        },
        {
          "name": "Band Protocol",
          "slug": "band-protocol",
          "ticker": "BAND"
        },
        {
          "name": "Shuffle",
          "slug": "shuffle",
          "ticker": "SHFL"
        },
        {
          "name": "SnowSwap",
          "slug": "snowswap",
          "ticker": "SNOW"
        },
        {
          "name": "Gods Unchained",
          "slug": "gods-unchained",
          "ticker": "GODS"
        },
        {
          "name": "LGCY Network",
          "slug": "lgcy-network",
          "ticker": "LGCY"
        },
        {
          "name": "Homeros",
          "slug": "homeros",
          "ticker": "HMR"
        },
        {
          "name": "The First Youtube Cat",
          "slug": "the-first-youtube-cat",
          "ticker": "PAJAMAS"
        },
        {
          "name": "BlazeStake Staked SOL",
          "slug": "blazestake-staked-sol",
          "ticker": "BSOL"
        },
        {
          "name": "Binance USD [on Ethereum]",
          "slug": "binance-usd",
          "ticker": "BUSD"
        },
        {
          "name": "Hyper Speed Network",
          "slug": "hyper-speed-network",
          "ticker": "HSN"
        },
        {
          "name": "Staked",
          "slug": "staked",
          "ticker": "STK"
        },
        {
          "name": "KUN",
          "slug": "kun",
          "ticker": "KUN"
        },
        {
          "name": "Alpaca Finance",
          "slug": "alpaca-finance",
          "ticker": "ALPACA"
        },
        {
          "name": "GYEN",
          "slug": "gyen",
          "ticker": "GYEN"
        },
        {
          "name": "AMATERASU OMIKAMI",
          "slug": "amaterasu-omikami",
          "ticker": "OMIKAMI"
        },
        {
          "name": "Curate",
          "slug": "curate",
          "ticker": "XCUR"
        },
        {
          "name": "SX Network",
          "slug": "p-sportx",
          "ticker": "SX"
        },
        {
          "name": "PAID Network",
          "slug": "paid-network",
          "ticker": "PAID"
        },
        {
          "name": "Tidal Finance",
          "slug": "tidal-finance",
          "ticker": "TIDAL"
        },
        {
          "name": "Wall Street Games",
          "slug": "wall-street-games",
          "ticker": "WSG"
        },
        {
          "name": "Medieval Empires",
          "slug": "medieval-empires",
          "ticker": "MEE"
        },
        {
          "name": "Venus Reward Token",
          "slug": "venus-reward-token",
          "ticker": "VRT"
        },
        {
          "name": "Avalanche",
          "slug": "avalanche",
          "ticker": "AVAX"
        },
        {
          "name": "CEEK",
          "slug": "ceek-vr",
          "ticker": "CEEK"
        },
        {
          "name": "BICONOMY",
          "slug": "biconomy",
          "ticker": "BICO"
        },
        {
          "name": "TrueFeedBack",
          "slug": "truefeedback",
          "ticker": "TFB"
        },
        {
          "name": "PAX Gold",
          "slug": "pax-gold",
          "ticker": "PAXG"
        },
        {
          "name": "Abra",
          "slug": "abra",
          "ticker": "ABR"
        },
        {
          "name": "MetaMoneyMarket",
          "slug": "metamoneymarket",
          "ticker": "MMM"
        },
        {
          "name": "Ariva",
          "slug": "ariva",
          "ticker": "ARV"
        },
        {
          "name": "Opus",
          "slug": "opus-2",
          "ticker": "OPUS"
        },
        {
          "name": "Elitium",
          "slug": "elitium",
          "ticker": "EUM"
        },
        {
          "name": "Agrello",
          "slug": "agrello-delta",
          "ticker": "DLT"
        },
        {
          "name": "AlphaWallet",
          "slug": "alphawallet",
          "ticker": "AWL"
        },
        {
          "name": "Ambo",
          "slug": "ambo",
          "ticker": "AMBO"
        },
        {
          "name": "Argent",
          "slug": "argent",
          "ticker": "ARG"
        },
        {
          "name": "Betoken",
          "slug": "betoken",
          "ticker": "BTK"
        },
        {
          "name": "LiquidApps",
          "slug": "liquid-apps",
          "ticker": "DAPP"
        },
        {
          "name": "AgriDex",
          "slug": "agridex",
          "ticker": "AGRI"
        },
        {
          "name": "MangoMan Intelligent",
          "slug": "mangoman-intelligent",
          "ticker": "MMIT"
        },
        {
          "name": "Cookie",
          "slug": "cookie",
          "ticker": "COOKIE"
        },
        {
          "name": "Colletrix",
          "slug": "colletrix",
          "ticker": "CIPX"
        },
        {
          "name": "Howdoo",
          "slug": "howdoo",
          "ticker": "UDOO"
        },
        {
          "name": "AmonD",
          "slug": "amond",
          "ticker": "AMON"
        },
        {
          "name": "Bob’s Repair",
          "slug": "bobs-repair",
          "ticker": "BOB"
        },
        {
          "name": "Business Credit Substitute",
          "slug": "business-credit-substitute",
          "ticker": "BCS"
        },
        {
          "name": "Bitpie",
          "slug": "bitpie",
          "ticker": "BTP"
        },
        {
          "name": "ClearPoll",
          "slug": "clearpoll",
          "ticker": "POLL"
        },
        {
          "name": "BUTTON Wallet",
          "slug": "button-wallet",
          "ticker": "BUTT"
        },
        {
          "name": "Cobo Wallet",
          "slug": "cobo-wallet",
          "ticker": "COBO"
        },
        {
          "name": "Blockzero Labs",
          "slug": "blockzerolabs",
          "ticker": "XIO"
        },
        {
          "name": "Synth iLINK",
          "slug": "ilink",
          "ticker": "iLINK"
        },
        {
          "name": "Polis",
          "slug": "polis",
          "ticker": "POLIS"
        },
        {
          "name": "NFTb",
          "slug": "nftb",
          "ticker": "NFTB"
        },
        {
          "name": "Sin City Metaverse",
          "slug": "sincity-token",
          "ticker": "SIN"
        },
        {
          "name": "Blockchain Quotations Index Token",
          "slug": "blockchain-quotations-index-token",
          "ticker": "BQT"
        },
        {
          "name": "Coinbase Wallet",
          "slug": "coinbase-wallet",
          "ticker": "CBW"
        },
        {
          "name": "KEY",
          "slug": "key",
          "ticker": "KEY"
        },
        {
          "name": "Fetch",
          "slug": "fetch-wallet",
          "ticker": "FTW"
        },
        {
          "name": "imToken",
          "slug": "imtoken",
          "ticker": "IMT"
        },
        {
          "name": "MMX",
          "slug": "mmx",
          "ticker": "MMX"
        },
        {
          "name": "SolFarm",
          "slug": "solfarm",
          "ticker": "TULIP"
        },
        {
          "name": "Synth iTRX",
          "slug": "itrx",
          "ticker": "iTRX"
        },
        {
          "name": "DeFi Saver",
          "slug": "defi-saver",
          "ticker": "DFS"
        },
        {
          "name": "DexWallet",
          "slug": "dexwallet",
          "ticker": "DWL"
        },
        {
          "name": "Condominium",
          "slug": "condominium",
          "ticker": "CDM"
        },
        {
          "name": "Synth sXMR",
          "slug": "sxmr",
          "ticker": "sXMR"
        },
        {
          "name": "Solanium",
          "slug": "solanium",
          "ticker": "SLIM"
        },
        {
          "name": "CENTERCOIN",
          "slug": "centercoin",
          "ticker": "CENT"
        },
        {
          "name": "LemoChain",
          "slug": "lemochain",
          "ticker": "LEMO"
        },
        {
          "name": "HYTOPIA",
          "slug": "hytopia",
          "ticker": "TOPIA"
        },
        {
          "name": "Beam",
          "slug": "beam",
          "ticker": "BEAM"
        },
        {
          "name": "SaluS",
          "slug": "salus",
          "ticker": "SLS"
        },
        {
          "name": "SENATE",
          "slug": "senate",
          "ticker": "SENATE"
        },
        {
          "name": "MetaTrace",
          "slug": "metatrace",
          "ticker": "TRC"
        },
        {
          "name": "Solrise Finance",
          "slug": "solrise-finance",
          "ticker": "SLRS"
        },
        {
          "name": "Synth sETH",
          "slug": "seth",
          "ticker": "sETH"
        },
        {
          "name": "Nibiru Chain",
          "slug": "nibiru-chain",
          "ticker": "NIBI"
        },
        {
          "name": "DeHive",
          "slug": "dehive",
          "ticker": "DHV"
        },
        {
          "name": "Rhenium",
          "slug": "rhenium",
          "ticker": "XRH"
        },
        {
          "name": "Block Array",
          "slug": "block-array",
          "ticker": "ARY"
        },
        {
          "name": "Synth sXRP",
          "slug": "sxrp",
          "ticker": "sXRP"
        },
        {
          "name": "POOH",
          "slug": "pooh",
          "ticker": "POOH"
        },
        {
          "name": "Freysa",
          "slug": "freysa-ai",
          "ticker": "FAI"
        },
        {
          "name": "Tarot",
          "slug": "tarot",
          "ticker": "TAROT"
        },
        {
          "name": "Alpha Token",
          "slug": "alpha-token",
          "ticker": "A"
        },
        {
          "name": "Synth iXRP",
          "slug": "ixrp",
          "ticker": "iXRP"
        },
        {
          "name": "Synth sEOS",
          "slug": "seos",
          "ticker": "sEOS"
        },
        {
          "name": "Olyseum",
          "slug": "olyseum",
          "ticker": "OLY"
        },
        {
          "name": "Manchester City Fan Token",
          "slug": "manchester-city-fan-token",
          "ticker": "CITY"
        },
        {
          "name": "Jigstack",
          "slug": "jigstack",
          "ticker": "STAK"
        },
        {
          "name": "Blackmoon",
          "slug": "blackmoon",
          "ticker": "BMC"
        },
        {
          "name": "Bibox Token",
          "slug": "bibox-token",
          "ticker": "BIX"
        },
        {
          "name": "TopGoal",
          "slug": "topgoal",
          "ticker": "GOAL"
        },
        {
          "name": "Synth iADA",
          "slug": "iada",
          "ticker": "iADA"
        },
        {
          "name": "Synth iCEX",
          "slug": "icex",
          "ticker": "iCEX"
        },
        {
          "name": "Banano",
          "slug": "banano",
          "ticker": "BAN"
        },
        {
          "name": "Synth iDEFI",
          "slug": "idefi",
          "ticker": "iDEFI"
        },
        {
          "name": "CONTRACOIN",
          "slug": "contracoin",
          "ticker": "CTCN"
        },
        {
          "name": "KeeperDAO",
          "slug": "keeperdao",
          "ticker": "ROOK"
        },
        {
          "name": "Modefi",
          "slug": "modefi",
          "ticker": "MOD"
        },
        {
          "name": "Synth iBCH",
          "slug": "ibch",
          "ticker": "iBCH"
        },
        {
          "name": "ICHI",
          "slug": "ichi",
          "ticker": "ICHI"
        },
        {
          "name": "B.Protocol",
          "slug": "b-protocol",
          "ticker": "BPRO"
        },
        {
          "name": "Sumokoin",
          "slug": "sumokoin",
          "ticker": "SUMO"
        },
        {
          "name": "Pure",
          "slug": "purex",
          "ticker": "PUREX"
        },
        {
          "name": "Synth sEUR",
          "slug": "seur",
          "ticker": "sEUR"
        },
        {
          "name": "Synth iDASH",
          "slug": "idash",
          "ticker": "iDASH"
        },
        {
          "name": "Niza Global",
          "slug": "niza-global",
          "ticker": "NIZA"
        },
        {
          "name": "Coinbase Wrapped Staked ETH",
          "slug": "coinbase-wrapped-staked-eth",
          "ticker": "cbETH"
        },
        {
          "name": "NNB Token",
          "slug": "nnb-token",
          "ticker": "NNB"
        },
        {
          "name": "Invacio",
          "slug": "invacio",
          "ticker": "INV"
        },
        {
          "name": "BUX Token",
          "slug": "bux-token",
          "ticker": "BUX"
        },
        {
          "name": "Mirrored Netflix",
          "slug": "mirrored-netflix",
          "ticker": "mNFLX"
        },
        {
          "name": "Artrade",
          "slug": "artrade",
          "ticker": "ATR"
        },
        {
          "name": "Spendcoin",
          "slug": "spendcoin",
          "ticker": "SPND"
        },
        {
          "name": "Synth sBNB",
          "slug": "sbnb",
          "ticker": "sBNB"
        },
        {
          "name": "Smartlands",
          "slug": "smartlands",
          "ticker": "SLT"
        },
        {
          "name": "Bitcoin Cash",
          "slug": "bitcoin-cash",
          "ticker": "BCH"
        },
        {
          "name": "Manifold Finance",
          "slug": "manifold-finance",
          "ticker": "FOLD"
        },
        {
          "name": "Bitcoin Cash ABC",
          "slug": "bitcoin-cash-abc-2",
          "ticker": "BCHA"
        },
        {
          "name": "Cajutel",
          "slug": "cajutel",
          "ticker": "CAJ"
        },
        {
          "name": "VIBE",
          "slug": "vibe",
          "ticker": "VIBE"
        },
        {
          "name": "Synth iLTC",
          "slug": "iltc",
          "ticker": "iLTC"
        },
        {
          "name": "Animecoin",
          "slug": "anime",
          "ticker": "ANIME"
        },
        {
          "name": "AI Rig Complex",
          "slug": "ai-rig-complex",
          "ticker": "ARC"
        },
        {
          "name": "Lamden",
          "slug": "lamden",
          "ticker": "TAU"
        },
        {
          "name": "Synth sLTC",
          "slug": "sltc",
          "ticker": "sLTC"
        },
        {
          "name": "Portugal National Team Fan Token",
          "slug": "portugal-national-team-fan-token",
          "ticker": "POR"
        },
        {
          "name": "Opacity",
          "slug": "opacity",
          "ticker": "OPCT"
        },
        {
          "name": "Everclear",
          "slug": "everclear",
          "ticker": "CLEAR"
        },
        {
          "name": "Beta Finance",
          "slug": "beta-finance",
          "ticker": "BETA"
        },
        {
          "name": "DEW",
          "slug": "dew",
          "ticker": "DEW"
        },
        {
          "name": "Cannation",
          "slug": "cannation",
          "ticker": "BTCP"
        },
        {
          "name": "QUINT",
          "slug": "quint",
          "ticker": "QUINT"
        },
        {
          "name": "Synth sXTZ",
          "slug": "sxtz",
          "ticker": "sXTZ"
        },
        {
          "name": "HyperCash",
          "slug": "hypercash",
          "ticker": "HC"
        },
        {
          "name": "Zilla",
          "slug": "zilla",
          "ticker": "ZLA"
        },
        {
          "name": "MenaPay",
          "slug": "menapay",
          "ticker": "MPAY"
        },
        {
          "name": "Polkadex",
          "slug": "polkadex",
          "ticker": "PDEX"
        },
        {
          "name": "SEER",
          "slug": "seer",
          "ticker": "SEER"
        },
        {
          "name": "MaidSafeCoin",
          "slug": "maidsafecoin",
          "ticker": "MAID"
        },
        {
          "name": "CashBet",
          "slug": "cashbet-coin",
          "ticker": "CBC"
        },
        {
          "name": "Sav3Token",
          "slug": "sav3token",
          "ticker": "SAV3"
        },
        {
          "name": "Unobtanium",
          "slug": "unobtanium",
          "ticker": "UNO"
        },
        {
          "name": "Bytom",
          "slug": "bytom",
          "ticker": "BTM"
        },
        {
          "name": "Synth iBTC",
          "slug": "ibtc-synthetix",
          "ticker": "iBTC"
        },
        {
          "name": "Cloudbric",
          "slug": "cloudbric",
          "ticker": "CLBK"
        },
        {
          "name": "Ink",
          "slug": "ink",
          "ticker": "INK"
        },
        {
          "name": "Komodo",
          "slug": "komodo",
          "ticker": "KMD"
        },
        {
          "name": "Blockcloud",
          "slug": "blockcloud",
          "ticker": "IOT"
        },
        {
          "name": "Sonic The Goat",
          "slug": "sonic-the-goat",
          "ticker": "GOAT"
        },
        {
          "name": "Synth iETH",
          "slug": "ieth",
          "ticker": "iETH"
        },
        {
          "name": "Insureum",
          "slug": "insureum",
          "ticker": "ISR"
        },
        {
          "name": "Zeepin",
          "slug": "zeepin",
          "ticker": "ZPT"
        },
        {
          "name": "Synth iEOS",
          "slug": "ieos",
          "ticker": "iEOS"
        },
        {
          "name": "Synth sAUD",
          "slug": "saud",
          "ticker": "sAUD"
        },
        {
          "name": "Zippie",
          "slug": "zippie",
          "ticker": "ZIPT"
        },
        {
          "name": "Decred",
          "slug": "decred",
          "ticker": "DCR"
        },
        {
          "name": "Global Currency Reserve",
          "slug": "global-currency-reserve",
          "ticker": "KWD"
        },
        {
          "name": "Milady Meme Coin",
          "slug": "milady-meme-coin",
          "ticker": "LADYS"
        },
        {
          "name": "Synth iETC",
          "slug": "ietc",
          "ticker": "iETC"
        },
        {
          "name": "Intelligent Investment Chain",
          "slug": "intelligent-investment-chain",
          "ticker": "IIC"
        },
        {
          "name": "Global Token",
          "slug": "global-token",
          "ticker": "GBL"
        },
        {
          "name": "Lido Staked Matic",
          "slug": "lido-staked-matic",
          "ticker": "stMATIC"
        },
        {
          "name": "FREE Coin",
          "slug": "free-coin",
          "ticker": "FREE"
        },
        {
          "name": "Ordinals",
          "slug": "ordinals",
          "ticker": "ORDI"
        },
        {
          "name": "My Lovely Planet",
          "slug": "my-lovely-planet",
          "ticker": "MLC"
        },
        {
          "name": "Escroco Emerald",
          "slug": "escroco-emerald",
          "ticker": "ESCE"
        },
        {
          "name": "HyperChainX",
          "slug": "hyperchainx",
          "ticker": "HYPER"
        },
        {
          "name": "Rasta Kitty Token",
          "slug": "rasta-kitty-token",
          "ticker": "RAS"
        },
        {
          "name": "RedFOX Labs",
          "slug": "redfox-labs",
          "ticker": "RFOX"
        },
        {
          "name": "Synth iXMR",
          "slug": "ixmr",
          "ticker": "iXMR"
        },
        {
          "name": "International Crypto X",
          "slug": "internationalcryptox",
          "ticker": "INCX"
        },
        {
          "name": "Quasarcoin",
          "slug": "quasarcoin",
          "ticker": "QAC"
        },
        {
          "name": "MediBloc",
          "slug": "medibloc",
          "ticker": "MED"
        },
        {
          "name": "Synth iXTZ",
          "slug": "ixtz",
          "ticker": "iXTZ"
        },
        {
          "name": "LIF3",
          "slug": "lif3",
          "ticker": "LIF3"
        },
        {
          "name": "Shiba Saga",
          "slug": "shiba-saga",
          "ticker": "SHIA"
        },
        {
          "name": "Opulous",
          "slug": "opulous",
          "ticker": "OPUL"
        },
        {
          "name": "A3S Protocol",
          "slug": "arb-a3s-protocol",
          "ticker": "AA"
        },
        {
          "name": "Calcium",
          "slug": "calcium",
          "ticker": "CAL"
        },
        {
          "name": "Wrapped Astar",
          "slug": "wrapped-astar",
          "ticker": "WASTR"
        },
        {
          "name": "Billion Happiness",
          "slug": "billionhappiness",
          "ticker": "BHC"
        },
        {
          "name": "Gather",
          "slug": "gather",
          "ticker": "GTH"
        },
        {
          "name": "TenX",
          "slug": "tenx",
          "ticker": "PAY"
        },
        {
          "name": "Elite",
          "slug": "1337coin",
          "ticker": "1337"
        },
        {
          "name": "SPINDLE",
          "slug": "spindle",
          "ticker": "SPD"
        },
        {
          "name": "ONOToken",
          "slug": "onotoken",
          "ticker": "ONOT"
        },
        {
          "name": "Synth iBNB",
          "slug": "ibnb",
          "ticker": "iBNB"
        },
        {
          "name": "Bifrost (BNC)",
          "slug": "bifrost-bnc",
          "ticker": "BNC"
        },
        {
          "name": "Dogs Of Elon",
          "slug": "dogs-of-elon",
          "ticker": "DOE"
        },
        {
          "name": "Payment Swap Utility Board",
          "slug": "psub",
          "ticker": "PSUB"
        },
        {
          "name": "BeeKan",
          "slug": "beekan",
          "ticker": "BKBT"
        },
        {
          "name": "Gas",
          "slug": "gas",
          "ticker": "GAS"
        },
        {
          "name": "COPYTRACK",
          "slug": "copytrack",
          "ticker": "CPY"
        },
        {
          "name": "ISKRA Token",
          "slug": "iskra",
          "ticker": "ISK"
        },
        {
          "name": "Minted",
          "slug": "minted",
          "ticker": "MTD"
        },
        {
          "name": "Burst",
          "slug": "burst",
          "ticker": "BURST"
        },
        {
          "name": "Staked WEMIX",
          "slug": "staked-wemix",
          "ticker": "stWEMIX"
        },
        {
          "name": "Wise Monkey",
          "slug": "wise-monkey",
          "ticker": "MONKY"
        },
        {
          "name": "Synth sTRX",
          "slug": "strx",
          "ticker": "sTRX"
        },
        {
          "name": "Equilibria Finance",
          "slug": "equilibria-finance",
          "ticker": "EQB"
        },
        {
          "name": "BIG",
          "slug": "big",
          "ticker": "BIG"
        },
        {
          "name": "Dignity",
          "slug": "dignity",
          "ticker": "DIG"
        },
        {
          "name": "Odyssey",
          "slug": "odyssey",
          "ticker": "OCN"
        },
        {
          "name": "Berry Data",
          "slug": "berry-data",
          "ticker": "BRY"
        },
        {
          "name": "BearAI",
          "slug": "bearai",
          "ticker": "BAI"
        },
        {
          "name": "Weecoins",
          "slug": "weecoins",
          "ticker": "WCS"
        },
        {
          "name": "Themis",
          "slug": "themis",
          "ticker": "GET"
        },
        {
          "name": "CanonChain",
          "slug": "cononchain",
          "ticker": "CZR"
        },
        {
          "name": "Experience Token",
          "slug": "experience-token",
          "ticker": "EXT"
        },
        {
          "name": "On-Chain Dynamics",
          "slug": "on-chain-dynamics",
          "ticker": "OCD"
        },
        {
          "name": "Ether Zero",
          "slug": "ether-zero",
          "ticker": "ETZ"
        },
        {
          "name": "Namecoin",
          "slug": "namecoin",
          "ticker": "NMC"
        },
        {
          "name": "Mirrored Microsoft",
          "slug": "mirrored-microsoft",
          "ticker": "mMSFT"
        },
        {
          "name": "Invictus Hyperion Fund",
          "slug": "invictus-hyperion-fund",
          "ticker": "IHF"
        },
        {
          "name": "REVV",
          "slug": "revv",
          "ticker": "REVV"
        },
        {
          "name": "PEAKDEFI",
          "slug": "peakdefi",
          "ticker": "PEAK"
        },
        {
          "name": "Only1",
          "slug": "only1",
          "ticker": "LIKE"
        },
        {
          "name": "HDAC",
          "slug": "hdac",
          "ticker": "DAC"
        },
        {
          "name": "Primas",
          "slug": "primas",
          "ticker": "PST"
        },
        {
          "name": "GameFi",
          "slug": "gamefi",
          "ticker": "GAFI"
        },
        {
          "name": "VikkyToken",
          "slug": "vikkytoken",
          "ticker": "VIKKY"
        },
        {
          "name": "X",
          "slug": "xerc20-pro",
          "ticker": "X"
        },
        {
          "name": "MonaCoin",
          "slug": "monacoin",
          "ticker": "MONA"
        },
        {
          "name": "DACC",
          "slug": "dacc",
          "ticker": "DACC"
        },
        {
          "name": "Netvrk",
          "slug": "netvrk",
          "ticker": "NETVR"
        },
        {
          "name": "Paytomat",
          "slug": "paytomat",
          "ticker": "PTI"
        },
        {
          "name": "Time New Bank",
          "slug": "time-new-bank",
          "ticker": "TNB"
        },
        {
          "name": "ShipChain",
          "slug": "shipchain",
          "ticker": "SHIP"
        },
        {
          "name": "Liquid ASTR",
          "slug": "liquid-astr",
          "ticker": "NASTR"
        },
        {
          "name": "Effect.AI",
          "slug": "effect-ai",
          "ticker": "EFX"
        },
        {
          "name": "Zipper",
          "slug": "zip",
          "ticker": "ZIP"
        },
        {
          "name": "Everus",
          "slug": "everus",
          "ticker": "EVR"
        },
        {
          "name": "Venus XVS",
          "slug": "venus-xvs",
          "ticker": "vXVS"
        },
        {
          "name": "Vara Network",
          "slug": "vara-network",
          "ticker": "VARA"
        },
        {
          "name": "Connex",
          "slug": "connex",
          "ticker": "CONX"
        },
        {
          "name": "Zen Protocol",
          "slug": "zen-protocol",
          "ticker": "ZP"
        },
        {
          "name": "Yuan Chain Coin",
          "slug": "yuan-chain-coin",
          "ticker": "YCC"
        },
        {
          "name": "MediShares",
          "slug": "medishares",
          "ticker": "MDS"
        },
        {
          "name": "Venus USDC",
          "slug": "venus-usdc",
          "ticker": "vUSDC"
        },
        {
          "name": "Frax Finance - Frax Ether",
          "slug": "frax-finance-frax-ether",
          "ticker": "FRXETH"
        },
        {
          "name": "Holdstation",
          "slug": "holdstation",
          "ticker": "HOLD"
        },
        {
          "name": "Cellula",
          "slug": "cellula",
          "ticker": "CELA"
        },
        {
          "name": "trumpwifhat",
          "slug": "trump-wif-hat",
          "ticker": "TRUMP"
        },
        {
          "name": "Mothership",
          "slug": "mothership",
          "ticker": "MSP"
        },
        {
          "name": "Step Hero",
          "slug": "step-hero",
          "ticker": "HERO"
        },
        {
          "name": "Wagerr",
          "slug": "wagerr",
          "ticker": "WGR"
        },
        {
          "name": "Connect Coin",
          "slug": "connect-coin",
          "ticker": "XCON"
        },
        {
          "name": "Ondo US Dollar Yield",
          "slug": "ondo-us-dollar-yield",
          "ticker": "USDY"
        },
        {
          "name": "Wombat Web 3 Gaming Platform",
          "slug": "wombat-web-3-gaming-platform",
          "ticker": "WOMBAT"
        },
        {
          "name": "MainnetZ",
          "slug": "mainnetz",
          "ticker": "NetZ"
        },
        {
          "name": "StockChain",
          "slug": "stockchain",
          "ticker": "SCC"
        },
        {
          "name": "OWL Token (StealthSwap)",
          "slug": "owl-token-stealthswap",
          "ticker": "OWL"
        },
        {
          "name": "Scorum",
          "slug": "scorum-coins",
          "ticker": "SCR"
        },
        {
          "name": "qiibee",
          "slug": "qiibee",
          "ticker": "QBX"
        },
        {
          "name": "reflect.finance",
          "slug": "reflect-finance",
          "ticker": "RFI"
        },
        {
          "name": "Suiswap",
          "slug": "suiswap",
          "ticker": "SSWP"
        },
        {
          "name": "UX Chain",
          "slug": "ux",
          "ticker": "UX"
        },
        {
          "name": "EUNOMIA",
          "slug": "eunomia",
          "ticker": "ENTS"
        },
        {
          "name": "Jiyo",
          "slug": "jiyo",
          "ticker": "JIYOX"
        },
        {
          "name": "Moeda Loyalty Points",
          "slug": "moeda-loyalty-points",
          "ticker": "MDA"
        },
        {
          "name": "Passage",
          "slug": "passage",
          "ticker": "PASG"
        },
        {
          "name": "Request",
          "slug": "request",
          "ticker": "REQ"
        },
        {
          "name": "Sun (New)",
          "slug": "sun-token",
          "ticker": "SUN"
        },
        {
          "name": "aixCB by Virtuals",
          "slug": "aixcb-by-virtuals",
          "ticker": "AIXCB"
        },
        {
          "name": "Pizza (Ordinals)",
          "slug": "pizza-ordinals",
          "ticker": "PIZZA"
        },
        {
          "name": "QuantixAI",
          "slug": "quantixai",
          "ticker": "QAI"
        },
        {
          "name": "Divi",
          "slug": "divi",
          "ticker": "DIVI"
        },
        {
          "name": "UFC Fan Token",
          "slug": "ufc-fan-token",
          "ticker": "UFC"
        },
        {
          "name": "Content Value Network",
          "slug": "content-value-network",
          "ticker": "CVNT"
        },
        {
          "name": "Aura Finance",
          "slug": "aura-finance",
          "ticker": "AURA"
        },
        {
          "name": "Hurify",
          "slug": "hurify",
          "ticker": "HUR"
        },
        {
          "name": "FairCoin",
          "slug": "faircoin",
          "ticker": "FAIR"
        },
        {
          "name": "DigitalNote",
          "slug": "digitalnote",
          "ticker": "XDN"
        },
        {
          "name": "Bitcore",
          "slug": "bitcore",
          "ticker": "BTX"
        },
        {
          "name": "Reboot",
          "slug": "reboot",
          "ticker": "GG"
        },
        {
          "name": "bitsCrunch",
          "slug": "bitscrunch",
          "ticker": "BCUT"
        },
        {
          "name": "Tiger King Coin",
          "slug": "tiger-king-coin",
          "ticker": "TKING"
        },
        {
          "name": "Primecoin",
          "slug": "primecoin",
          "ticker": "XPM"
        },
        {
          "name": "Aeternity",
          "slug": "aeternity",
          "ticker": "AE"
        },
        {
          "name": "Snetwork",
          "slug": "snetwork",
          "ticker": "SNET"
        },
        {
          "name": "Klever",
          "slug": "klever",
          "ticker": "KLV"
        },
        {
          "name": "L7 DEX",
          "slug": "l7-dex",
          "ticker": "LSD"
        },
        {
          "name": "BlitzPredict",
          "slug": "blitzpredict",
          "ticker": "XBP"
        },
        {
          "name": "Bitrock",
          "slug": "bitrock",
          "ticker": "BROCK"
        },
        {
          "name": "Hermez Network",
          "slug": "hermez-network",
          "ticker": "HEZ"
        },
        {
          "name": "Sashimi",
          "slug": "sashimi",
          "ticker": "SASHIMI"
        },
        {
          "name": "0x0.ai",
          "slug": "0x0-ai-ai-smart-contract",
          "ticker": "0x0"
        },
        {
          "name": "Steem Dollars",
          "slug": "steem-dollars",
          "ticker": "SBD"
        },
        {
          "name": "Shark Cat",
          "slug": "shark-cat",
          "ticker": "SC"
        },
        {
          "name": "Maple",
          "slug": "maple",
          "ticker": "MPL"
        },
        {
          "name": "Jesus Coin",
          "slug": "jesus-toys",
          "ticker": "JESUS"
        },
        {
          "name": "Eligma",
          "slug": "eligma-token",
          "ticker": "ELI"
        },
        {
          "name": "Massnet",
          "slug": "massnet",
          "ticker": "MASS"
        },
        {
          "name": "Metaverser",
          "slug": "metaverser",
          "ticker": "MTVT"
        },
        {
          "name": "Acorn Protocol",
          "slug": "acorn-protocol",
          "ticker": "ACN"
        },
        {
          "name": "Qrkita Token",
          "slug": "qrkita-token",
          "ticker": "QRT"
        },
        {
          "name": "Edelcoin",
          "slug": "edelcoin",
          "ticker": "EDLC"
        },
        {
          "name": "ValueChain",
          "slug": "valuechain",
          "ticker": "VLC"
        },
        {
          "name": "Graviocoin",
          "slug": "graviocoin",
          "ticker": "GIO"
        },
        {
          "name": "Bitsum",
          "slug": "bitsum",
          "ticker": "BSM"
        },
        {
          "name": "SHIELD",
          "slug": "shield-xsh",
          "ticker": "XSH"
        },
        {
          "name": "Pivot Token",
          "slug": "pivot-token",
          "ticker": "PVT"
        },
        {
          "name": "Hatom",
          "slug": "hatom",
          "ticker": "HTM"
        },
        {
          "name": "KILT Protocol",
          "slug": "kiltprotocol",
          "ticker": "KILT"
        },
        {
          "name": "ADAPad",
          "slug": "adapad",
          "ticker": "ADAPAD"
        },
        {
          "name": "Pkoin",
          "slug": "pocketnet",
          "ticker": "PKOIN"
        },
        {
          "name": "XP NETWORK",
          "slug": "xp-network",
          "ticker": "XPNET"
        },
        {
          "name": "MinexCoin",
          "slug": "minexcoin",
          "ticker": "MNX"
        },
        {
          "name": "TokenPay",
          "slug": "tokenpay",
          "ticker": "TPAY"
        },
        {
          "name": "Faceter",
          "slug": "faceter",
          "ticker": "FACE"
        },
        {
          "name": "VITE",
          "slug": "vite",
          "ticker": "VITE"
        },
        {
          "name": "vEmpire DDAO",
          "slug": "vempire-ddao",
          "ticker": "VEMP"
        },
        {
          "name": "Litecoin Cash",
          "slug": "litecoin-cash",
          "ticker": "LCC"
        },
        {
          "name": "Pura",
          "slug": "pura",
          "ticker": "PURA"
        },
        {
          "name": "Tectonic",
          "slug": "tectonic",
          "ticker": "TONIC"
        },
        {
          "name": "Bee Token",
          "slug": "bee-token",
          "ticker": "BEE"
        },
        {
          "name": "Nebulas",
          "slug": "nebulas-token",
          "ticker": "NAS"
        },
        {
          "name": "GoChain",
          "slug": "gochain",
          "ticker": "GO"
        },
        {
          "name": "Router Protocol",
          "slug": "router-protocol",
          "ticker": "ROUTE"
        },
        {
          "name": "Wrapped CRO",
          "slug": "wrapped-cro",
          "ticker": "WCRO"
        },
        {
          "name": "Etho Protocol",
          "slug": "etho-protocol",
          "ticker": "ETHO"
        },
        {
          "name": "LINK",
          "slug": "link",
          "ticker": "FNSA"
        },
        {
          "name": "Experty",
          "slug": "experty",
          "ticker": "EXY"
        },
        {
          "name": "CPChain",
          "slug": "cpchain",
          "ticker": "CPC"
        },
        {
          "name": "Delysium [on Ethereum]",
          "slug": "delysium",
          "ticker": "AGI"
        },
        {
          "name": "MyToken",
          "slug": "mytoken",
          "ticker": "MT"
        },
        {
          "name": "Monero Classic",
          "slug": "monero-classic",
          "ticker": "XMC"
        },
        {
          "name": "BrahmaOS",
          "slug": "brahmaos",
          "ticker": "BRM"
        },
        {
          "name": "Morpheus Labs",
          "slug": "morpheus-labs",
          "ticker": "MITX"
        },
        {
          "name": "Narrative Network",
          "slug": "narrative",
          "ticker": "NRV"
        },
        {
          "name": "Creo Engine",
          "slug": "creo-engine",
          "ticker": "CREO"
        },
        {
          "name": "U Network",
          "slug": "u-network",
          "ticker": "UUU"
        },
        {
          "name": "ParaSwap",
          "slug": "paraswap",
          "ticker": "PSP"
        },
        {
          "name": "Polybius",
          "slug": "polybius",
          "ticker": "PLBT"
        },
        {
          "name": "Sp8de",
          "slug": "sp8de",
          "ticker": "SPX"
        },
        {
          "name": "SALT",
          "slug": "salt",
          "ticker": "SALT"
        },
        {
          "name": "HEROcoin",
          "slug": "herocoin",
          "ticker": "PLAY"
        },
        {
          "name": "GoWithMi",
          "slug": "gowithmi",
          "ticker": "GMAT"
        },
        {
          "name": "YooShi",
          "slug": "yooshi",
          "ticker": "YOOSHI"
        },
        {
          "name": "DerivaDAO",
          "slug": "derivadao",
          "ticker": "DDX"
        },
        {
          "name": "Octokn",
          "slug": "octo-gaming",
          "ticker": "OTK"
        },
        {
          "name": "CyberVein",
          "slug": "cybervein",
          "ticker": "CVT"
        },
        {
          "name": "Lendingblock",
          "slug": "lendingblock",
          "ticker": "LND"
        },
        {
          "name": "Speed Mining Service",
          "slug": "speed-mining-service",
          "ticker": "SMS"
        },
        {
          "name": "TosDis",
          "slug": "tosdis",
          "ticker": "DIS"
        },
        {
          "name": "Stratis",
          "slug": "stratis",
          "ticker": "STRAX"
        },
        {
          "name": "Particl",
          "slug": "particl",
          "ticker": "PART"
        },
        {
          "name": "Amazy",
          "slug": "amazy",
          "ticker": "AZY"
        },
        {
          "name": "Ice Open Network",
          "slug": "ice-decentralized-future",
          "ticker": "ICE"
        },
        {
          "name": "Wrapped XDC",
          "slug": "wrapped-xdc-network",
          "ticker": "WXDC"
        },
        {
          "name": "Nash Exchange",
          "slug": "nash-exchange",
          "ticker": "NEX"
        },
        {
          "name": "Blink Galaxy",
          "slug": "outer-ring-mmo-gq",
          "ticker": "GQ"
        },
        {
          "name": "The Midas Touch Gold",
          "slug": "the-midas-touch-gold",
          "ticker": "TMTG"
        },
        {
          "name": "OriginTrail",
          "slug": "origintrail",
          "ticker": "TRAC"
        },
        {
          "name": "Opium",
          "slug": "opium",
          "ticker": "OPIUM"
        },
        {
          "name": "Expanse",
          "slug": "expanse",
          "ticker": "EXP"
        },
        {
          "name": "Sentinel",
          "slug": "sentinel",
          "ticker": "DVPN"
        },
        {
          "name": "AME Chain",
          "slug": "amepay",
          "ticker": "AME"
        },
        {
          "name": "Integral",
          "slug": "integral",
          "ticker": "ITGR"
        },
        {
          "name": "Blockport",
          "slug": "blockport",
          "ticker": "BPT"
        },
        {
          "name": "DAO.Casino",
          "slug": "dao-casino",
          "ticker": "BET"
        },
        {
          "name": "DeFi Bids",
          "slug": "defi-bids",
          "ticker": "BID"
        },
        {
          "name": "eosDAC",
          "slug": "eosdac",
          "ticker": "EOSDAC"
        },
        {
          "name": "No BS Crypto",
          "slug": "no-bs-crypto",
          "ticker": "NOBS"
        },
        {
          "name": "OXBT",
          "slug": "oxbt",
          "ticker": "OXBT"
        },
        {
          "name": "Thrupenny",
          "slug": "thrupenny",
          "ticker": "TPY"
        },
        {
          "name": "Polkastarter",
          "slug": "polkastarter",
          "ticker": "POLS"
        },
        {
          "name": "Ravencoin Classic",
          "slug": "ravencoin-classic",
          "ticker": "RVC"
        },
        {
          "name": "Venus BTC",
          "slug": "venus-btc",
          "ticker": "vBTC"
        },
        {
          "name": "Hourglass",
          "slug": "hourglass",
          "ticker": "WAIT"
        },
        {
          "name": "Reflexer Ungovernance Token",
          "slug": "reflexer-ungovernance-token",
          "ticker": "FLX"
        },
        {
          "name": "Asura Coin",
          "slug": "asura-coin",
          "ticker": "ASA"
        },
        {
          "name": "HYCON ",
          "slug": "hycon",
          "ticker": "HYC"
        },
        {
          "name": "Steem",
          "slug": "steem",
          "ticker": "STEEM"
        },
        {
          "name": "Venus BCH",
          "slug": "venus-bch",
          "ticker": "vBCH"
        },
        {
          "name": "Sharpe Platform Token",
          "slug": "sharpe-platform-token",
          "ticker": "SHP"
        },
        {
          "name": "Doge Killer",
          "slug": "doge-killer",
          "ticker": "LEASH"
        },
        {
          "name": "ALBOS",
          "slug": "albos",
          "ticker": "ALB"
        },
        {
          "name": "Waletoken",
          "slug": "waletoken",
          "ticker": "WTN"
        },
        {
          "name": "Eden",
          "slug": "archer-dao-governance-token",
          "ticker": "EDEN"
        },
        {
          "name": "Biswap",
          "slug": "biswap",
          "ticker": "BSW"
        },
        {
          "name": "SPECTRE AI",
          "slug": "spectre-ai",
          "ticker": "SPECTRE"
        },
        {
          "name": "τBitcoin",
          "slug": "t-bitcoin",
          "ticker": "ΤBTC"
        },
        {
          "name": "Venus DAI",
          "slug": "venus-dai",
          "ticker": "vDAI"
        },
        {
          "name": "BABB",
          "slug": "babb",
          "ticker": "BAX"
        },
        {
          "name": "BioPassport Token",
          "slug": "biopassport-token",
          "ticker": "BIOT"
        },
        {
          "name": "XMax",
          "slug": "xmax",
          "ticker": "XMX"
        },
        {
          "name": "ONUS",
          "slug": "onus",
          "ticker": "ONUS"
        },
        {
          "name": "Revomon",
          "slug": "revomon",
          "ticker": "REVO"
        },
        {
          "name": "Vivid Coin",
          "slug": "vivid-coin",
          "ticker": "VIVID"
        },
        {
          "name": "Bridge Mutual",
          "slug": "bridge-mutual",
          "ticker": "BMI"
        },
        {
          "name": "Refereum",
          "slug": "refereum",
          "ticker": "RFR"
        },
        {
          "name": "Memecoin",
          "slug": "meme",
          "ticker": "MEME"
        },
        {
          "name": "CoinEx Token",
          "slug": "coinex-token",
          "ticker": "CET"
        },
        {
          "name": "VIMworld",
          "slug": "vimworld",
          "ticker": "VEED"
        },
        {
          "name": "Genesis Vision",
          "slug": "genesis-vision",
          "ticker": "GVT"
        },
        {
          "name": "THEKEY",
          "slug": "thekey",
          "ticker": "TKY"
        },
        {
          "name": "Sleepless AI",
          "slug": "bnb-sleepless-ai",
          "ticker": "AI"
        },
        {
          "name": "Sakura Bloom",
          "slug": "sakura-bloom",
          "ticker": "SKB"
        },
        {
          "name": "Nano",
          "slug": "nano",
          "ticker": "XNO"
        },
        {
          "name": "IQ.cash",
          "slug": "iqcash",
          "ticker": "IQ"
        },
        {
          "name": "ConstitutionDAO",
          "slug": "constitutiondao",
          "ticker": "PEOPLE"
        },
        {
          "name": "Groestlcoin",
          "slug": "groestlcoin",
          "ticker": "GRS"
        },
        {
          "name": "Lightning Bitcoin",
          "slug": "lightning-bitcoin",
          "ticker": "LBTC"
        },
        {
          "name": "XSwap",
          "slug": "xswap",
          "ticker": "XSWAP"
        },
        {
          "name": "More Coin",
          "slug": "more-coin",
          "ticker": "MORE"
        },
        {
          "name": "Kalkulus",
          "slug": "kalkulus",
          "ticker": "KLKS"
        },
        {
          "name": "GameZone",
          "slug": "gamezone",
          "ticker": "GZONE"
        },
        {
          "name": "Drep [new]",
          "slug": "drep-new",
          "ticker": "DREP"
        },
        {
          "name": "Basenji",
          "slug": "basenjibase",
          "ticker": "BENJI"
        },
        {
          "name": "LandWolf (SOL)",
          "slug": "landwolf-sol",
          "ticker": "WOLF"
        },
        {
          "name": "Mistery On Cro",
          "slug": "mistery-on-cro",
          "ticker": "MERY"
        },
        {
          "name": "QASH",
          "slug": "qash",
          "ticker": "QASH"
        },
        {
          "name": "Xai",
          "slug": "arb-xai-games",
          "ticker": "XAI"
        },
        {
          "name": "Stellar",
          "slug": "stellar",
          "ticker": "XLM"
        },
        {
          "name": "Lido wstETH",
          "slug": "lido-finance-wsteth",
          "ticker": "WSTETH"
        },
        {
          "name": "Doge Dash",
          "slug": "doge-dash",
          "ticker": "DOGEDASH"
        },
        {
          "name": "Venus USDT",
          "slug": "venus-usdt",
          "ticker": "vUSDT"
        },
        {
          "name": "indaHash",
          "slug": "indahash",
          "ticker": "IDH"
        },
        {
          "name": "Aerobud",
          "slug": "aerobud",
          "ticker": "AEROBUD"
        },
        {
          "name": "Furucombo",
          "slug": "furucombo",
          "ticker": "COMBO"
        },
        {
          "name": "Aurory",
          "slug": "aurory",
          "ticker": "AURY"
        },
        {
          "name": "Bottos",
          "slug": "bottos",
          "ticker": "BTO"
        },
        {
          "name": "Bytecoin",
          "slug": "bytecoin-bcn",
          "ticker": "BCN"
        },
        {
          "name": "Cetus Protocol",
          "slug": "cetus-protocol",
          "ticker": "CETUS"
        },
        {
          "name": "Trinity Network Credit",
          "slug": "trinity-network-credit",
          "ticker": "TNC"
        },
        {
          "name": "Globalvillage Ecosystem",
          "slug": "globalvillage-ecosystem",
          "ticker": "GVE"
        },
        {
          "name": "Name Changing Token",
          "slug": "name-changing-token",
          "ticker": "NCT"
        },
        {
          "name": "Bitcoin File",
          "slug": "bitcoin-file",
          "ticker": "BIFI"
        },
        {
          "name": "IRISnet",
          "slug": "irisnet",
          "ticker": "IRIS"
        },
        {
          "name": "T.OS",
          "slug": "t-os",
          "ticker": "TOSC"
        },
        {
          "name": "Spectiv",
          "slug": "signal-token",
          "ticker": "SIG"
        },
        {
          "name": "YUI Token",
          "slug": "yui-token",
          "ticker": "YUI"
        },
        {
          "name": "WagyuSwap",
          "slug": "wagyuswap",
          "ticker": "WAG"
        },
        {
          "name": "Futureswap",
          "slug": "futureswap",
          "ticker": "FST"
        },
        {
          "name": "Uniswap V2: USDC",
          "slug": "uniswap_usdc_eth_lp",
          "ticker": "UNI-V2 USDC/ETH LP"
        },
        {
          "name": "Sentinel Chain",
          "slug": "sentinel-chain",
          "ticker": "SENC"
        },
        {
          "name": "MAPS",
          "slug": "maps",
          "ticker": "MAPS"
        },
        {
          "name": "LockTrip",
          "slug": "lockchain",
          "ticker": "LOC"
        },
        {
          "name": "HorusPay",
          "slug": "horuspay",
          "ticker": "HORUS"
        },
        {
          "name": "Electroneum",
          "slug": "electroneum",
          "ticker": "ETN"
        },
        {
          "name": "Biotron",
          "slug": "biotron",
          "ticker": "BTRN"
        },
        {
          "name": "Nxt",
          "slug": "nxt",
          "ticker": "NXT"
        },
        {
          "name": "Convergence",
          "slug": "convergence",
          "ticker": "CONV"
        },
        {
          "name": "Stronghold Token",
          "slug": "stronghold-token",
          "ticker": "SHX"
        },
        {
          "name": "Electrify.Asia",
          "slug": "electrifyasia",
          "ticker": "ELEC"
        },
        {
          "name": "Nerves",
          "slug": "nerves",
          "ticker": "NER"
        },
        {
          "name": "Lykke",
          "slug": "lykke",
          "ticker": "LKK"
        },
        {
          "name": "BOMB",
          "slug": "bomb",
          "ticker": "BOMB"
        },
        {
          "name": "Viacoin",
          "slug": "viacoin",
          "ticker": "VIA"
        },
        {
          "name": "ION",
          "slug": "ion",
          "ticker": "ION"
        },
        {
          "name": "Uniswap V2: UNI 6",
          "slug": "uniswap_uni_eth_lp",
          "ticker": "Uniswap UNI/ETH LP"
        },
        {
          "name": "NetMind Token",
          "slug": "netmind-token",
          "ticker": "NMT"
        },
        {
          "name": "Platypus Finance",
          "slug": "platypus-finance",
          "ticker": "PTP"
        },
        {
          "name": "PiP (Hyperliquid)",
          "slug": "pip-on-hl",
          "ticker": "PIP"
        },
        {
          "name": "4EVERLAND",
          "slug": "4everland",
          "ticker": "4EVER"
        },
        {
          "name": "Dreamcoins",
          "slug": "dreamcoins",
          "ticker": "DREAM"
        },
        {
          "name": "GPU ai Rich",
          "slug": "gpu-ai-rich",
          "ticker": "RICH"
        },
        {
          "name": "Atlantis Blue Digital Token",
          "slug": "atlantis-blue-digital-token",
          "ticker": "ABDT"
        },
        {
          "name": "Metahero",
          "slug": "metahero",
          "ticker": "HERO"
        },
        {
          "name": "Nebula AI",
          "slug": "nebula-ai",
          "ticker": "NBAI"
        },
        {
          "name": "Decentralized Machine Learning",
          "slug": "decentralized-machine-learning",
          "ticker": "DML"
        },
        {
          "name": "Distributed Credit Chain",
          "slug": "distributed-credit-chain",
          "ticker": "DCC"
        },
        {
          "name": "PayPie",
          "slug": "paypie",
          "ticker": "PPP"
        },
        {
          "name": "Ardor",
          "slug": "ardor",
          "ticker": "ARDR"
        },
        {
          "name": "analoS",
          "slug": "analos",
          "ticker": "ANALOS"
        },
        {
          "name": "Jesus Coin",
          "slug": "jesus-coin",
          "ticker": "JC"
        },
        {
          "name": "MongCoin",
          "slug": "mongcoin",
          "ticker": "$MONG"
        },
        {
          "name": "UNUS SED LEO",
          "slug": "unus-sed-leo",
          "ticker": "LEO"
        },
        {
          "name": "NFT Art Finance",
          "slug": "nft-art-finance",
          "ticker": "NFTART"
        },
        {
          "name": "SportyCo",
          "slug": "sportyco",
          "ticker": "SPF"
        },
        {
          "name": "Centrifuge",
          "slug": "centrifuge",
          "ticker": "CFG"
        },
        {
          "name": "Pundi X NEM",
          "slug": "pundi-x-nem",
          "ticker": "NPXSXEM"
        },
        {
          "name": "Educoin",
          "slug": "edu-coin",
          "ticker": "EDU"
        },
        {
          "name": "CatCoin",
          "slug": "catcoin",
          "ticker": "CAT"
        },
        {
          "name": "Influence Chain",
          "slug": "influence-chain",
          "ticker": "IFC"
        },
        {
          "name": "Arcblock",
          "slug": "arcblock",
          "ticker": "ABT"
        },
        {
          "name": "Ignis",
          "slug": "ignis",
          "ticker": "IGNIS"
        },
        {
          "name": "Uniswap V2: DAI",
          "slug": "uniswap_dai_eth_lp",
          "ticker": "UNI-V2 DAI/ETH LP"
        },
        {
          "name": "Uniswap V2: YFI",
          "slug": "uniswap_yfi_eth_lp",
          "ticker": "UNI-V2 YFI/ETH LP"
        },
        {
          "name": "Balancer: BAL/ETH 80/20 #2",
          "slug": "balancer_pool_token",
          "ticker": "BPT-BAL/ETH-80/20"
        },
        {
          "name": "Crypto Asset Governance Alliance",
          "slug": "crypto-asset-governance-alliance",
          "ticker": "CAGA"
        },
        {
          "name": "AIPAD",
          "slug": "aipad",
          "ticker": "AIPAD"
        },
        {
          "name": "Omchain",
          "slug": "omchain",
          "ticker": "OMC"
        },
        {
          "name": "Token IN",
          "slug": "token-in",
          "ticker": "TIN"
        },
        {
          "name": "Pleasure Coin",
          "slug": "pleasure-coin",
          "ticker": "NSFW"
        },
        {
          "name": "Skeb Coin",
          "slug": "skeb-coin",
          "ticker": "SKEB"
        },
        {
          "name": "BetterBetting",
          "slug": "betterbetting",
          "ticker": "BETR"
        },
        {
          "name": "The ChampCoin",
          "slug": "the-champcoin",
          "ticker": "TCC"
        },
        {
          "name": "TrueUSD [on Ethereum]",
          "slug": "trueusd",
          "ticker": "TUSD"
        },
        {
          "name": "Dent",
          "slug": "dent",
          "ticker": "DENT"
        },
        {
          "name": "ARPA Chain",
          "slug": "arpa-chain",
          "ticker": "ARPA"
        },
        {
          "name": "IONChain",
          "slug": "ionchain",
          "ticker": "IONC"
        },
        {
          "name": "OAX",
          "slug": "oax",
          "ticker": "OAX"
        },
        {
          "name": "Oasis Network",
          "slug": "oasis-network",
          "ticker": "ROSE"
        },
        {
          "name": "Maker",
          "slug": "maker",
          "ticker": "MKR"
        },
        {
          "name": "Maro",
          "slug": "maro",
          "ticker": "MARO"
        },
        {
          "name": "Blockchain Monster Hunt",
          "slug": "blockchain-monster-hunt",
          "ticker": "BCMC"
        },
        {
          "name": "Rublix",
          "slug": "rublix",
          "ticker": "RBLX"
        },
        {
          "name": "Compound Uniswap",
          "slug": "compound-uniswap",
          "ticker": "cUNI"
        },
        {
          "name": "DAEX",
          "slug": "daex",
          "ticker": "DAX"
        },
        {
          "name": "MktCoin",
          "slug": "mktcoin",
          "ticker": "MLM"
        },
        {
          "name": "Gemini Dollar",
          "slug": "gemini-dollar",
          "ticker": "GUSD"
        },
        {
          "name": "Valuto",
          "slug": "valuto",
          "ticker": "VLU"
        },
        {
          "name": "FTX Token",
          "slug": "ftx-token",
          "ticker": "FTT"
        },
        {
          "name": "Veloce",
          "slug": "o-veloce-vext",
          "ticker": "VEXT"
        },
        {
          "name": "Endorsit",
          "slug": "endorsit",
          "ticker": "EDS"
        },
        {
          "name": "Reef",
          "slug": "reef",
          "ticker": "REEF"
        },
        {
          "name": "Penta",
          "slug": "penta",
          "ticker": "PNT"
        },
        {
          "name": "RAMP",
          "slug": "ramp",
          "ticker": "RAMP"
        },
        {
          "name": "Krypton DAO",
          "slug": "krypton-dao",
          "ticker": "KRD"
        },
        {
          "name": "Polygon [on Polygon]",
          "slug": "p-matic-network",
          "ticker": "MATIC"
        },
        {
          "name": "Compound USDT",
          "slug": "compound-usdt",
          "ticker": "cUSDT"
        },
        {
          "name": "Wrapped AVAX",
          "slug": "wavax",
          "ticker": "WAVAX"
        },
        {
          "name": "Ultra",
          "slug": "ultra",
          "ticker": "UOS"
        },
        {
          "name": "Dingocoin",
          "slug": "dingocoin",
          "ticker": "DINGO"
        },
        {
          "name": "BOME TRUMP",
          "slug": "bome-trump",
          "ticker": "TRUMP"
        },
        {
          "name": "Catboy",
          "slug": "cat-boy",
          "ticker": "CATBOY"
        },
        {
          "name": "Dascoin",
          "slug": "dascoin",
          "ticker": "DASC"
        },
        {
          "name": "Font",
          "slug": "font",
          "ticker": "FONT"
        },
        {
          "name": "Peercoin",
          "slug": "peercoin",
          "ticker": "PPC"
        },
        {
          "name": "Aave LINK",
          "slug": "aave-link",
          "ticker": "aLINK"
        },
        {
          "name": "Cellframe",
          "slug": "cellframe",
          "ticker": "CELL"
        },
        {
          "name": "CoinPoker",
          "slug": "coinpoker",
          "ticker": "CHP"
        },
        {
          "name": "Gelato",
          "slug": "gelato",
          "ticker": "GEL"
        },
        {
          "name": "Bigbom",
          "slug": "bigbom",
          "ticker": "BBO"
        },
        {
          "name": "Kind Ads Token",
          "slug": "kind-ads-token",
          "ticker": "KIND"
        },
        {
          "name": "bitCNY",
          "slug": "bitcny",
          "ticker": "BITCNY"
        },
        {
          "name": "Polkamarkets",
          "slug": "polkamarkets",
          "ticker": "POLK"
        },
        {
          "name": "TitanSwap",
          "slug": "titanswap",
          "ticker": "TITAN"
        },
        {
          "name": "Ryo Currency",
          "slug": "ryo-currency",
          "ticker": "RYO"
        },
        {
          "name": "AirDAO",
          "slug": "airdao",
          "ticker": "AMB"
        },
        {
          "name": "BoutsPro",
          "slug": "boutspro",
          "ticker": "BOUTS"
        },
        {
          "name": "ATN",
          "slug": "atn",
          "ticker": "ATN"
        },
        {
          "name": "Business Credit Alliance Chain",
          "slug": "business-credit-alliance-chain",
          "ticker": "BCAC"
        },
        {
          "name": "Storm",
          "slug": "storm",
          "ticker": "STORM"
        },
        {
          "name": "Aave ETH",
          "slug": "aave-eth",
          "ticker": "aETH"
        },
        {
          "name": "Linda",
          "slug": "linda",
          "ticker": "LINDA"
        },
        {
          "name": "Brazil National Fan Token",
          "slug": "brazil-national-football-team-fan-token",
          "ticker": "BFT"
        },
        {
          "name": "RedStone",
          "slug": "redstone",
          "ticker": "RED"
        },
        {
          "name": "Synthetix [on Ethereum]",
          "slug": "synthetix-network-token",
          "ticker": "SNX"
        },
        {
          "name": "Calamari Network",
          "slug": "calamari-network",
          "ticker": "KMA"
        },
        {
          "name": "Helbiz",
          "slug": "helbiz",
          "ticker": "HBZ"
        },
        {
          "name": "Bodhi [ETH]",
          "slug": "bodhi-eth",
          "ticker": "BOE"
        },
        {
          "name": "Kuende",
          "slug": "kuende",
          "ticker": "KUE"
        },
        {
          "name": "BuckHathCoin",
          "slug": "buck-hath-coin",
          "ticker": "BHIG"
        },
        {
          "name": "Aave TUSD",
          "slug": "aave-tusd",
          "ticker": "aTUSD"
        },
        {
          "name": "Freeway Token",
          "slug": "freeway-token",
          "ticker": "FWT"
        },
        {
          "name": "Nodle",
          "slug": "nodleiot",
          "ticker": "NODL"
        },
        {
          "name": "Aki Network",
          "slug": "aki-network",
          "ticker": "AKI"
        },
        {
          "name": "Kalao",
          "slug": "kalao",
          "ticker": "KLO"
        },
        {
          "name": "DigiFinexToken",
          "slug": "digifinextoken",
          "ticker": "DFT"
        },
        {
          "name": "WaykiChain",
          "slug": "waykichain",
          "ticker": "WICC"
        },
        {
          "name": "FuturoCoin",
          "slug": "futurocoin",
          "ticker": "FTO"
        },
        {
          "name": "Patron",
          "slug": "patron",
          "ticker": "PAT"
        },
        {
          "name": "Docademic",
          "slug": "docademic",
          "ticker": "MTC"
        },
        {
          "name": "Aave DAI",
          "slug": "aave-dai",
          "ticker": "aDAI"
        },
        {
          "name": "Aave MKR",
          "slug": "aave-mkr",
          "ticker": "aMKR"
        },
        {
          "name": "Aave BAT",
          "slug": "aave-bat",
          "ticker": "aBAT"
        },
        {
          "name": "Wrapped Ampleforth",
          "slug": "wrapped-ampleforth",
          "ticker": "WAMPL"
        },
        {
          "name": "Fulcrom Finance",
          "slug": "fulcrom-finance",
          "ticker": "FUL"
        },
        {
          "name": "FARM",
          "slug": "farm-2",
          "ticker": "FARM"
        },
        {
          "name": "Avatly (New)",
          "slug": "avatly-new",
          "ticker": "AVATLY"
        },
        {
          "name": "FuzeX",
          "slug": "fuzex",
          "ticker": "FXT"
        },
        {
          "name": "Aave USDT",
          "slug": "aave-usdt",
          "ticker": "aUSDT"
        },
        {
          "name": "Oraichain Token",
          "slug": "oraichain-token",
          "ticker": "ORAI"
        },
        {
          "name": "Quanta Utility Token",
          "slug": "quanta-utility-token",
          "ticker": "QNTU"
        },
        {
          "name": "Presearch",
          "slug": "presearch",
          "ticker": "PRE"
        },
        {
          "name": "Bitex Global XBX Coin",
          "slug": "bitex-global-xbx-coin",
          "ticker": "XBX"
        },
        {
          "name": "Valor Token",
          "slug": "valor-token",
          "ticker": "VALOR"
        },
        {
          "name": "ODUWA",
          "slug": "oduwa",
          "ticker": "OWC"
        },
        {
          "name": "Civic",
          "slug": "civic",
          "ticker": "CVC"
        },
        {
          "name": "Based Turbo",
          "slug": "based-turbo",
          "ticker": "TURBO"
        },
        {
          "name": "Stafi",
          "slug": "stafi",
          "ticker": "FIS"
        },
        {
          "name": "TRUMP MEME (trumpmeme.net)",
          "slug": "trumpmeme-net",
          "ticker": "MEME"
        },
        {
          "name": "wstUSDT",
          "slug": "wstusdt",
          "ticker": "WSTUSDT"
        },
        {
          "name": "NOTAI",
          "slug": "notai",
          "ticker": "NOTAI"
        },
        {
          "name": "Elysian",
          "slug": "elysian",
          "ticker": "ELY"
        },
        {
          "name": "Decentralized Social",
          "slug": "deso",
          "ticker": "DESO"
        },
        {
          "name": "ALLUVA",
          "slug": "alluva",
          "ticker": "ALV"
        },
        {
          "name": "dKargo",
          "slug": "dkargo",
          "ticker": "DKA"
        },
        {
          "name": "BnkToTheFuture",
          "slug": "bnktothefuture",
          "ticker": "BFT"
        },
        {
          "name": "Silent Notary",
          "slug": "silent-notary",
          "ticker": "SNTR"
        },
        {
          "name": "Verum Coin",
          "slug": "verum-coin",
          "ticker": "VERUM"
        },
        {
          "name": "Cybereits",
          "slug": "cybereits",
          "ticker": "CRE"
        },
        {
          "name": "POPCHAIN",
          "slug": "popchain",
          "ticker": "PCH"
        },
        {
          "name": "OneLedger",
          "slug": "oneledger",
          "ticker": "OLT"
        },
        {
          "name": "Tottenham Hotspur Fan Token",
          "slug": "tottenham-hotspur-fan-token",
          "ticker": "SPURS"
        },
        {
          "name": "Aave SUSD",
          "slug": "aave-susd",
          "ticker": "aSUSD"
        },
        {
          "name": "Islamic Coin",
          "slug": "islamic-coin",
          "ticker": "ISLM"
        },
        {
          "name": "SELF Crypto",
          "slug": "self-crypto",
          "ticker": "SELF"
        },
        {
          "name": "Aurora",
          "slug": "aurora",
          "ticker": "AOA"
        },
        {
          "name": "CryCash",
          "slug": "crycash",
          "ticker": "CRC"
        },
        {
          "name": "EDUCare",
          "slug": "educare",
          "ticker": "EKT"
        },
        {
          "name": "Wrapped Bitcoin [on Polygon]",
          "slug": "p-wrapped-bitcoin",
          "ticker": "WBTC"
        },
        {
          "name": "FootballCoin",
          "slug": "footballcoin",
          "ticker": "XFC"
        },
        {
          "name": "Skrumble Network",
          "slug": "skrumble-network",
          "ticker": "SKM"
        },
        {
          "name": "Aave MANA",
          "slug": "aave-mana",
          "ticker": "aMANA"
        },
        {
          "name": "Electronic USD",
          "slug": "electronic-usd",
          "ticker": "eUSD"
        },
        {
          "name": "UpOnly",
          "slug": "uponly",
          "ticker": "UPO"
        },
        {
          "name": "HEX (PulseChain)",
          "slug": "hex-pulsechain",
          "ticker": "HEX"
        },
        {
          "name": "Lunyr",
          "slug": "lunyr",
          "ticker": "LUN"
        },
        {
          "name": "Debitum",
          "slug": "debitum-network",
          "ticker": "DEB"
        },
        {
          "name": "Aeon",
          "slug": "aeon",
          "ticker": "AEON"
        },
        {
          "name": "Soarcoin",
          "slug": "soarcoin",
          "ticker": "SOAR"
        },
        {
          "name": "DeepBrain Chain",
          "slug": "deepbrain-chain",
          "ticker": "DBC"
        },
        {
          "name": "Crypterium",
          "slug": "crpt",
          "ticker": "CRPT"
        },
        {
          "name": "Echoin",
          "slug": "echoin",
          "ticker": "ECH"
        },
        {
          "name": "U.CASH",
          "slug": "ucash",
          "ticker": "UCASH"
        },
        {
          "name": "Aave KNC",
          "slug": "aave-knc",
          "ticker": "aKNC"
        },
        {
          "name": "Paribus",
          "slug": "paribus",
          "ticker": "PBX"
        },
        {
          "name": "MOG CAT",
          "slug": "mog-cat",
          "ticker": "MOG"
        },
        {
          "name": "APF coin",
          "slug": "apf-coin",
          "ticker": "APFC"
        },
        {
          "name": "Telcoin",
          "slug": "telcoin",
          "ticker": "TEL"
        },
        {
          "name": "VinDax Coin",
          "slug": "vindax-coin",
          "ticker": "VD"
        },
        {
          "name": "OLXA",
          "slug": "olxa",
          "ticker": "OLXA"
        },
        {
          "name": "BoringDAO",
          "slug": "boringdao",
          "ticker": "BOR"
        },
        {
          "name": "Shekel",
          "slug": "shekel",
          "ticker": "JEW"
        },
        {
          "name": "Aave ZRX",
          "slug": "aave-zrx",
          "ticker": "aZRX"
        },
        {
          "name": "Blind Boxes",
          "slug": "blind-boxes",
          "ticker": "BLES"
        },
        {
          "name": "Kava Swap",
          "slug": "kava-swap",
          "ticker": "SWP"
        },
        {
          "name": "Ancient8",
          "slug": "ancient8",
          "ticker": "A8"
        },
        {
          "name": "Waffles Davincij15's Cat",
          "slug": "waffles-davincij15s-cat",
          "ticker": "$WAFFLES"
        },
        {
          "name": "Pollux Coin",
          "slug": "pollux-coin",
          "ticker": "POX"
        },
        {
          "name": "Milady Cult Coin",
          "slug": "milady-cult-coin",
          "ticker": "CULT"
        },
        {
          "name": "Paycoin",
          "slug": "payprotocol",
          "ticker": "PCI"
        },
        {
          "name": "Pixie Coin",
          "slug": "pixie-coin",
          "ticker": "PXC"
        },
        {
          "name": "Pal Network",
          "slug": "pal-network",
          "ticker": "PAL"
        },
        {
          "name": "PitisCoin",
          "slug": "pitiscoin",
          "ticker": "PTS"
        },
        {
          "name": "Nexo",
          "slug": "nexo",
          "ticker": "NEXO"
        },
        {
          "name": "Proton Token",
          "slug": "proton-token",
          "ticker": "PTT"
        },
        {
          "name": "Kora Network Token",
          "slug": "kora-network-token",
          "ticker": "KNT"
        },
        {
          "name": "Kryll.io",
          "slug": "kryll",
          "ticker": "KRL"
        },
        {
          "name": "E-Dinar Coin",
          "slug": "e-dinar-coin",
          "ticker": "EDR"
        },
        {
          "name": "Nasdacoin",
          "slug": "nasdacoin",
          "ticker": "NSD"
        },
        {
          "name": "CROAT",
          "slug": "croat",
          "ticker": "CROAT"
        },
        {
          "name": "Orion",
          "slug": "orion-xyz",
          "ticker": "ORN"
        },
        {
          "name": "Decubate",
          "slug": "decubate",
          "ticker": "DCB"
        },
        {
          "name": "XRP Healthcare",
          "slug": "xrp-healthcare",
          "ticker": "XRPH"
        },
        {
          "name": "Alphabet",
          "slug": "alphabet-erc404",
          "ticker": "ALPHABET"
        },
        {
          "name": "Matchpool",
          "slug": "guppy",
          "ticker": "GUP"
        },
        {
          "name": "Comtech Gold",
          "slug": "comtech-gold",
          "ticker": "CGO"
        },
        {
          "name": "Amp",
          "slug": "amp",
          "ticker": "AMP"
        },
        {
          "name": "Jury.Online Token",
          "slug": "jury-online-token",
          "ticker": "JOT"
        },
        {
          "name": "Friendz",
          "slug": "friends",
          "ticker": "FDZ"
        },
        {
          "name": "Cosmo Coin",
          "slug": "cosmo-coin",
          "ticker": "COSM"
        },
        {
          "name": "ModulTrade",
          "slug": "modultrade",
          "ticker": "MTRC"
        },
        {
          "name": "ThetaDrop",
          "slug": "thetadrop",
          "ticker": "TDROP"
        },
        {
          "name": "Aave BUSD",
          "slug": "aave-busd",
          "ticker": "aBUSD"
        },
        {
          "name": "Choise.com",
          "slug": "choise",
          "ticker": "CHO"
        },
        {
          "name": "Solend",
          "slug": "solend",
          "ticker": "SLND"
        },
        {
          "name": "XELS",
          "slug": "xels",
          "ticker": "XELS"
        },
        {
          "name": "IBStoken",
          "slug": "ibstoken",
          "ticker": "IBS"
        },
        {
          "name": "CZ THE GOAT",
          "slug": "cz-the-goat",
          "ticker": "CZGOAT"
        },
        {
          "name": "Don-key",
          "slug": "don-key",
          "ticker": "DON"
        },
        {
          "name": "Forty Seven Bank (47 Bank)",
          "slug": "forty-seven-bank",
          "ticker": "FSBT"
        },
        {
          "name": "Wrapped NCG (Nine Chronicles Gold)",
          "slug": "wrapped-ncg",
          "ticker": "WNCG"
        },
        {
          "name": "empowr coin",
          "slug": "empowr-coin",
          "ticker": "EMPR"
        },
        {
          "name": "Exosis",
          "slug": "exosis",
          "ticker": "EXO"
        },
        {
          "name": "Bitcoin BEP2",
          "slug": "bitcoin-bep2",
          "ticker": "BTCB"
        },
        {
          "name": "Temco",
          "slug": "temco",
          "ticker": "TEMCO"
        },
        {
          "name": "0xcert Protocol",
          "slug": "0xcert",
          "ticker": "ZXC"
        },
        {
          "name": "Aave REP",
          "slug": "aave-rep",
          "ticker": "aREP"
        },
        {
          "name": "Verified USD",
          "slug": "verified-usd",
          "ticker": "USDV"
        },
        {
          "name": "Aave LEND",
          "slug": "aave-lend",
          "ticker": "aLEND"
        },
        {
          "name": "DAV Token",
          "slug": "dav-coin",
          "ticker": "DAV"
        },
        {
          "name": "Cashberry Coin",
          "slug": "cashberry-coin",
          "ticker": "CBC"
        },
        {
          "name": "HitChain",
          "slug": "hitchain",
          "ticker": "HIT"
        },
        {
          "name": "Pillar",
          "slug": "pillar",
          "ticker": "PLR"
        },
        {
          "name": "PumaPay",
          "slug": "pumapay",
          "ticker": "PMA"
        },
        {
          "name": "LocalCoinSwap",
          "slug": "local-coin-swap",
          "ticker": "LCS"
        },
        {
          "name": "LEOcoin",
          "slug": "leocoin",
          "ticker": "LC4"
        },
        {
          "name": "OKB",
          "slug": "okb",
          "ticker": "OKB"
        },
        {
          "name": "MIR COIN",
          "slug": "mir-coin",
          "ticker": "MIR"
        },
        {
          "name": "Timicoin",
          "slug": "timicoin",
          "ticker": "TMC"
        },
        {
          "name": "Mango Markets",
          "slug": "mango-markets",
          "ticker": "MNGO"
        },
        {
          "name": "Honest",
          "slug": "honest",
          "ticker": "HNST"
        },
        {
          "name": "Consentium",
          "slug": "consentium",
          "ticker": "CSM"
        },
        {
          "name": "ATC Coin",
          "slug": "atc-coin",
          "ticker": "ATCC"
        },
        {
          "name": "Theta Fuel",
          "slug": "theta-fuel",
          "ticker": "TFUEL"
        },
        {
          "name": "Finblox",
          "slug": "finblox",
          "ticker": "FBX"
        },
        {
          "name": "CUDOS",
          "slug": "cudos",
          "ticker": "CUDOS"
        },
        {
          "name": "Airbloc",
          "slug": "airbloc",
          "ticker": "ABL"
        },
        {
          "name": "DxChain",
          "slug": "dxchain-token",
          "ticker": "DX"
        },
        {
          "name": "APIS",
          "slug": "apis",
          "ticker": "APIS"
        },
        {
          "name": "LogisCoin",
          "slug": "logiscoin",
          "ticker": "LGS"
        },
        {
          "name": "Insolar",
          "slug": "insolar",
          "ticker": "XNS"
        },
        {
          "name": "Davinci Coin",
          "slug": "davinci-coin",
          "ticker": "DAC"
        },
        {
          "name": "Coldstack",
          "slug": "coldstack",
          "ticker": "CLS"
        },
        {
          "name": "NFTX Hashmasks Index",
          "slug": "nftx-hashmasks-index",
          "ticker": "MASK"
        },
        {
          "name": "Indigo Protocol",
          "slug": "indigo-protocol",
          "ticker": "INDY"
        },
        {
          "name": "Olive",
          "slug": "olive",
          "ticker": "OLE"
        },
        {
          "name": "OceanEx",
          "slug": "oceanex-token",
          "ticker": "OCE"
        },
        {
          "name": "Ark",
          "slug": "ark",
          "ticker": "ARK"
        },
        {
          "name": "bitCEO",
          "slug": "bitceo",
          "ticker": "BCEO"
        },
        {
          "name": "Falconswap",
          "slug": "fsw-token",
          "ticker": "FSW"
        },
        {
          "name": "AiLink Token",
          "slug": "ailink-token",
          "ticker": "ALI"
        },
        {
          "name": "Index Cooperative",
          "slug": "index-cooperative",
          "ticker": "INDEX"
        },
        {
          "name": "Stobox Token",
          "slug": "stobox-token",
          "ticker": "STBU"
        },
        {
          "name": "ANyONe Protocol",
          "slug": "anyone-protocol",
          "ticker": "ANYONE"
        },
        {
          "name": "Wabi",
          "slug": "tael",
          "ticker": "WABI"
        },
        {
          "name": "3space Art",
          "slug": "3space-art",
          "ticker": "PACE"
        },
        {
          "name": "MarsX",
          "slug": "marsx",
          "ticker": "MX"
        },
        {
          "name": "DOGAMÍ",
          "slug": "dogami",
          "ticker": "DOGA"
        },
        {
          "name": "KWHCoin",
          "slug": "kwhcoin",
          "ticker": "KWH"
        },
        {
          "name": "Six Domain Chain",
          "slug": "six-domain-chain",
          "ticker": "SDA"
        },
        {
          "name": "CariNet",
          "slug": "carinet",
          "ticker": "CIT"
        },
        {
          "name": "FarmaTrust",
          "slug": "farmatrust",
          "ticker": "FTT"
        },
        {
          "name": "MGC Token",
          "slug": "mgc-token",
          "ticker": "MGC"
        },
        {
          "name": "Knoxstertoken",
          "slug": "knoxstertoken",
          "ticker": "FKX"
        },
        {
          "name": "Blocery",
          "slug": "blocery",
          "ticker": "BLY"
        },
        {
          "name": "Kin",
          "slug": "kin",
          "ticker": "KIN"
        },
        {
          "name": "TFL.io",
          "slug": "trueflip",
          "ticker": "TFL"
        },
        {
          "name": "Govi",
          "slug": "govi",
          "ticker": "GOVI"
        },
        {
          "name": "Curio Governance",
          "slug": "curio-governance",
          "ticker": "CGT"
        },
        {
          "name": "UnMarshal",
          "slug": "unmarshal",
          "ticker": "MARSH"
        },
        {
          "name": "HOLD",
          "slug": "hold",
          "ticker": "HOLD"
        },
        {
          "name": "WPP",
          "slug": "wpp-token",
          "ticker": "WPP"
        },
        {
          "name": "BUMO",
          "slug": "bumo",
          "ticker": "BU"
        },
        {
          "name": "HeartBout",
          "slug": "heartbout",
          "ticker": "HB"
        },
        {
          "name": "Auto",
          "slug": "auto",
          "ticker": "AUTO"
        },
        {
          "name": "SEEN",
          "slug": "seen",
          "ticker": "SEEN"
        },
        {
          "name": "Ti-Value",
          "slug": "ti-value",
          "ticker": "TV"
        },
        {
          "name": "district0x",
          "slug": "district0x",
          "ticker": "DNT"
        },
        {
          "name": "Gems ",
          "slug": "gems-protocol",
          "ticker": "GEM"
        },
        {
          "name": "EosBLACK",
          "slug": "eosblack",
          "ticker": "BLACK"
        },
        {
          "name": "Quantstamp",
          "slug": "quantstamp",
          "ticker": "QSP"
        },
        {
          "name": "Bitmart",
          "slug": "bitmart-token",
          "ticker": "BMX"
        },
        {
          "name": "Global Social Chain",
          "slug": "global-social-chain",
          "ticker": "GSC"
        },
        {
          "name": "DWS",
          "slug": "dws",
          "ticker": "DWS"
        },
        {
          "name": "Formosa Financial",
          "slug": "formosa-financial",
          "ticker": "FMF"
        },
        {
          "name": "HashCoin",
          "slug": "hashcoin",
          "ticker": "HSC"
        },
        {
          "name": "Contentos",
          "slug": "contentos",
          "ticker": "COS"
        },
        {
          "name": "V-Dimension",
          "slug": "v-dimension",
          "ticker": "VOLLAR"
        },
        {
          "name": "Helpico",
          "slug": "helpico",
          "ticker": "HELP"
        },
        {
          "name": "Gourmet Galaxy",
          "slug": "gourmet-galaxy",
          "ticker": "GUM"
        },
        {
          "name": "WaifuAI",
          "slug": "waifuai",
          "ticker": "WFAI"
        },
        {
          "name": "PolkaDomain",
          "slug": "polkadomain",
          "ticker": "NAME"
        },
        {
          "name": "Zeusshield",
          "slug": "zeusshield",
          "ticker": "ZSC"
        },
        {
          "name": "DPRating",
          "slug": "dprating",
          "ticker": "RATING"
        },
        {
          "name": "MyBit",
          "slug": "mybit",
          "ticker": "MYB"
        },
        {
          "name": "Ice Rock Mining",
          "slug": "ice-rock-mining",
          "ticker": "ROCK2"
        },
        {
          "name": "BitcoiNote",
          "slug": "bitcoinote",
          "ticker": "BTCN"
        },
        {
          "name": "Webcoin",
          "slug": "webcoin",
          "ticker": "WEB"
        },
        {
          "name": "Relex",
          "slug": "relex",
          "ticker": "RLX"
        },
        {
          "name": "MEX",
          "slug": "mex",
          "ticker": "MEX"
        },
        {
          "name": "Remme",
          "slug": "remme",
          "ticker": "REM"
        },
        {
          "name": "Aave USDC",
          "slug": "aave-usdc",
          "ticker": "aUSDC"
        },
        {
          "name": "Ziktalk",
          "slug": "ziktalk",
          "ticker": "ZIK"
        },
        {
          "name": "$REKT",
          "slug": "rektcoin",
          "ticker": "REKT"
        },
        {
          "name": "Alphacat",
          "slug": "alphacat",
          "ticker": "ACAT"
        },
        {
          "name": "YOU COIN",
          "slug": "you-coin",
          "ticker": "YOU"
        },
        {
          "name": "Aave WBTC",
          "slug": "aave-wbtc",
          "ticker": "aWBTC"
        },
        {
          "name": "ClearCoin",
          "slug": "clearcoin",
          "ticker": "XCLR"
        },
        {
          "name": "Starta",
          "slug": "starta",
          "ticker": "STA"
        },
        {
          "name": "GSENetwork",
          "slug": "gsenetwork",
          "ticker": "GSE"
        },
        {
          "name": "MoneyToken",
          "slug": "moneytoken",
          "ticker": "IMT"
        },
        {
          "name": "SuperTrust",
          "slug": "supertrust",
          "ticker": "SUT"
        },
        {
          "name": "Rogue West",
          "slug": "rogue-west",
          "ticker": "ROGUE"
        },
        {
          "name": "APYSwap",
          "slug": "apyswap",
          "ticker": "APYS"
        },
        {
          "name": "Solana Name Service",
          "slug": "solana-name-service",
          "ticker": "FIDA"
        },
        {
          "name": "ugChain",
          "slug": "ugchain",
          "ticker": "UGC"
        },
        {
          "name": "United Traders Token",
          "slug": "uttoken",
          "ticker": "UTT"
        },
        {
          "name": "ProximaX",
          "slug": "proximax",
          "ticker": "XPX"
        },
        {
          "name": "OneRoot Network",
          "slug": "oneroot-network",
          "ticker": "RNT"
        },
        {
          "name": "LinkEye",
          "slug": "linkeye",
          "ticker": "LET"
        },
        {
          "name": "Etherparty",
          "slug": "etherparty",
          "ticker": "FUEL"
        },
        {
          "name": "Taiko",
          "slug": "taiko",
          "ticker": "TAIKO"
        },
        {
          "name": "Umee",
          "slug": "umee",
          "ticker": "UX"
        },
        {
          "name": "Beyondfi",
          "slug": "beyond-finance",
          "ticker": "BYN"
        },
        {
          "name": "DogeBonk",
          "slug": "dogebonk",
          "ticker": "DOBO"
        },
        {
          "name": "Lendroid Support Token",
          "slug": "lendroid-support-token",
          "ticker": "LST"
        },
        {
          "name": "Iconomi",
          "slug": "iconomi",
          "ticker": "ICN"
        },
        {
          "name": "WHALE",
          "slug": "whale",
          "ticker": "WHALE"
        },
        {
          "name": "AirSwap",
          "slug": "airswap",
          "ticker": "AST"
        },
        {
          "name": "Medicalchain",
          "slug": "medical-chain",
          "ticker": "MTN"
        },
        {
          "name": "Chronologic",
          "slug": "chronologic",
          "ticker": "DAY"
        },
        {
          "name": "ZeroSwap",
          "slug": "zeroswap",
          "ticker": "ZEE"
        },
        {
          "name": "GoHelpFund",
          "slug": "gohelpfund",
          "ticker": "HELP"
        },
        {
          "name": "Metal",
          "slug": "metal",
          "ticker": "MTL"
        },
        {
          "name": "Beefy.Finance",
          "slug": "beefy-finance",
          "ticker": "BIFI"
        },
        {
          "name": "Eva Cash",
          "slug": "eva-cash",
          "ticker": "EVC"
        },
        {
          "name": "WeShow Token",
          "slug": "weshow-token",
          "ticker": "WET"
        },
        {
          "name": "Fellaz",
          "slug": "fellaz",
          "ticker": "FLZ"
        },
        {
          "name": "PLAYA3ULL GAMES",
          "slug": "playa3ull",
          "ticker": "3ULL"
        },
        {
          "name": "Inspect",
          "slug": "inspect",
          "ticker": "INSP"
        },
        {
          "name": "OTCBTC Token",
          "slug": "otcbtc-token",
          "ticker": "OTB"
        },
        {
          "name": "Couchain",
          "slug": "couchain",
          "ticker": "COU"
        },
        {
          "name": "Razor Network",
          "slug": "razor-network",
          "ticker": "RAZOR"
        },
        {
          "name": "Bitcoin SV",
          "slug": "bitcoin-sv",
          "ticker": "BSV"
        },
        {
          "name": "DreamTeam",
          "slug": "dreamteam-token",
          "ticker": "DTT"
        },
        {
          "name": "W3Coin",
          "slug": "w3coin",
          "ticker": "W3C"
        },
        {
          "name": "Earth Token",
          "slug": "earth-token",
          "ticker": "EARTH"
        },
        {
          "name": "DATx",
          "slug": "datx",
          "ticker": "DATX"
        },
        {
          "name": "First Digital USD [on Ethereum]",
          "slug": "first-digital-usd",
          "ticker": "FDUSD"
        },
        {
          "name": "Moviebloc",
          "slug": "moviebloc",
          "ticker": "MBL"
        },
        {
          "name": "Binamon",
          "slug": "binamon",
          "ticker": "BMON"
        },
        {
          "name": "Project TXA",
          "slug": "project-txa",
          "ticker": "TXA"
        },
        {
          "name": "StakeCubeCoin",
          "slug": "stakecubecoin",
          "ticker": "SCC"
        },
        {
          "name": "Decent",
          "slug": "decent",
          "ticker": "DCT"
        },
        {
          "name": "Dragon Coins",
          "slug": "dragon-coins",
          "ticker": "DRG"
        },
        {
          "name": "ZMINE",
          "slug": "zmine",
          "ticker": "ZMN"
        },
        {
          "name": "Graphlinq Protocol",
          "slug": "graphlinq-protocol",
          "ticker": "GLQ"
        },
        {
          "name": "SIX",
          "slug": "six",
          "ticker": "SIX"
        },
        {
          "name": "Telos",
          "slug": "telos",
          "ticker": "TLOS"
        },
        {
          "name": "Lido Staked ETH",
          "slug": "steth",
          "ticker": "stETH"
        },
        {
          "name": "Bobo Cash",
          "slug": "bobo-cash",
          "ticker": "BOBO"
        },
        {
          "name": "Unvest",
          "slug": "unvest",
          "ticker": "UNV"
        },
        {
          "name": "Wonderman Nation",
          "slug": "wonderman-nation",
          "ticker": "WNDR"
        },
        {
          "name": "DAPS Token",
          "slug": "daps-token",
          "ticker": "DAPS"
        },
        {
          "name": "Metronome",
          "slug": "metronome",
          "ticker": "MET"
        },
        {
          "name": "Humanscape",
          "slug": "humanscape",
          "ticker": "HUM"
        },
        {
          "name": "Nuggets",
          "slug": "nuggets",
          "ticker": "NUG"
        },
        {
          "name": "Smartshare",
          "slug": "smartshare",
          "ticker": "SSP"
        },
        {
          "name": "Eristica",
          "slug": "eristica",
          "ticker": "ERT"
        },
        {
          "name": "Non-Fungible Yearn",
          "slug": "non-fungible-yearn",
          "ticker": "NFY"
        },
        {
          "name": "Clover Finance",
          "slug": "clover",
          "ticker": "CLV"
        },
        {
          "name": "VeriME",
          "slug": "verime",
          "ticker": "VME"
        },
        {
          "name": "Glitch",
          "slug": "glitch",
          "ticker": "GLCH"
        },
        {
          "name": "Measurable Data Token",
          "slug": "measurable-data-token",
          "ticker": "MDT"
        },
        {
          "name": "Pibble",
          "slug": "pibble",
          "ticker": "PIB"
        },
        {
          "name": "CARAT",
          "slug": "carat",
          "ticker": "CARAT"
        },
        {
          "name": "Daneel",
          "slug": "daneel",
          "ticker": "DAN"
        },
        {
          "name": "Insights Network",
          "slug": "insights-network",
          "ticker": "INSTAR"
        },
        {
          "name": "Fractal",
          "slug": "fractal",
          "ticker": "FCL"
        },
        {
          "name": "Zuki",
          "slug": "zuki-moba",
          "ticker": "ZUKI"
        },
        {
          "name": "VinuChain",
          "slug": "vinuchain",
          "ticker": "VC"
        },
        {
          "name": "cheqd",
          "slug": "cheqd",
          "ticker": "CHEQ"
        },
        {
          "name": "Papi",
          "slug": "papi",
          "ticker": "PAPI"
        },
        {
          "name": "CyberMiles",
          "slug": "cybermiles",
          "ticker": "CMT"
        },
        {
          "name": "GIGA",
          "slug": "giga",
          "ticker": "XG"
        },
        {
          "name": "DEXTools",
          "slug": "dextools",
          "ticker": "DEXT"
        },
        {
          "name": "SDChain",
          "slug": "sdchain",
          "ticker": "SDA"
        },
        {
          "name": "Wixlar",
          "slug": "wixlar",
          "ticker": "WIX"
        },
        {
          "name": "Metadium",
          "slug": "metadium",
          "ticker": "META"
        },
        {
          "name": "ADAMANT Messenger",
          "slug": "adamant-messenger",
          "ticker": "ADM"
        },
        {
          "name": "ELA Coin",
          "slug": "ela-coin",
          "ticker": "ELAC"
        },
        {
          "name": "Yearn Secure",
          "slug": "yearn-secure",
          "ticker": "YSEC"
        },
        {
          "name": "Waifu Token",
          "slug": "waifu-token",
          "ticker": "WAIF"
        },
        {
          "name": "MetaMorph",
          "slug": "metamorph",
          "ticker": "METM"
        },
        {
          "name": "Nafter",
          "slug": "nafter",
          "ticker": "NAFT"
        },
        {
          "name": "Leeds United Fan Token",
          "slug": "leeds-united-fan-token",
          "ticker": "LUFC"
        },
        {
          "name": "BHO Network",
          "slug": "bholdus",
          "ticker": "BHO"
        },
        {
          "name": "IceChain",
          "slug": "icechain",
          "ticker": "ICHX"
        },
        {
          "name": "VegaWallet Token",
          "slug": "vegawallet-token",
          "ticker": "VGW"
        },
        {
          "name": "NAGA",
          "slug": "naga",
          "ticker": "NGC"
        },
        {
          "name": "HashBX",
          "slug": "hashsbx",
          "ticker": "HBX"
        },
        {
          "name": "AMO Coin",
          "slug": "amo-coin",
          "ticker": "AMO"
        },
        {
          "name": "LTO Network",
          "slug": "lto-network",
          "ticker": "LTO"
        },
        {
          "name": "Fortuna",
          "slug": "fortuna",
          "ticker": "FOTA"
        },
        {
          "name": "Akropolis",
          "slug": "akropolis",
          "ticker": "AKRO"
        },
        {
          "name": "Hypr Network",
          "slug": "hypr-network",
          "ticker": "HYPR"
        },
        {
          "name": "OneArt",
          "slug": "artwallet",
          "ticker": "1ART"
        },
        {
          "name": "$LONDON",
          "slug": "london",
          "ticker": "LONDON"
        },
        {
          "name": "NEXT",
          "slug": "next-coin",
          "ticker": "NEXT"
        },
        {
          "name": "Fluz Fluz",
          "slug": "fluz-fluz",
          "ticker": "FLUZ"
        },
        {
          "name": "Reserve",
          "slug": "reserve-rights",
          "ticker": "RSR"
        },
        {
          "name": "TE-FOOD",
          "slug": "te-food",
          "ticker": "TONE"
        },
        {
          "name": "Efinity",
          "slug": "efinity",
          "ticker": "EFI"
        },
        {
          "name": "Ink Protocol",
          "slug": "ink-protocol",
          "ticker": "XNK"
        },
        {
          "name": "Friend.tech",
          "slug": "friend-tech",
          "ticker": "FRIEND"
        },
        {
          "name": "Teloscoin",
          "slug": "teloscoin",
          "ticker": "TELOS"
        },
        {
          "name": "LYNC Network",
          "slug": "lync-network",
          "ticker": "LYNC"
        },
        {
          "name": "SOAR.FI",
          "slug": "soar-fi",
          "ticker": "SOAR"
        },
        {
          "name": "8PAY",
          "slug": "8pay",
          "ticker": "8PAY"
        },
        {
          "name": "Realm",
          "slug": "realm",
          "ticker": "REALM"
        },
        {
          "name": "BounceBit",
          "slug": "bouncebit",
          "ticker": "BB"
        },
        {
          "name": "Ormeus Coin",
          "slug": "ormeus-coin",
          "ticker": "ORME"
        },
        {
          "name": "Atlas Protocol",
          "slug": "atlas-protocol",
          "ticker": "ATP"
        },
        {
          "name": "Hashgard",
          "slug": "hashgard",
          "ticker": "GARD"
        },
        {
          "name": "Blockchain Certified Data Token",
          "slug": "blockchain-certified-data-token",
          "ticker": "BCDT"
        },
        {
          "name": "Darico Ecosystem Coin",
          "slug": "darcio-ecosystem-coin",
          "ticker": "DEC"
        },
        {
          "name": "HyperQuant",
          "slug": "hyperquant",
          "ticker": "HQT"
        },
        {
          "name": "ILCoin",
          "slug": "ilcoin",
          "ticker": "ILC"
        },
        {
          "name": "Orion Money",
          "slug": "orion-money",
          "ticker": "ORION"
        },
        {
          "name": "Tolar",
          "slug": "tolar",
          "ticker": "TOL"
        },
        {
          "name": "Etheera",
          "slug": "etheera",
          "ticker": "ETA"
        },
        {
          "name": "Safe Haven",
          "slug": "safe-haven",
          "ticker": "SHA"
        },
        {
          "name": "Zilliqa",
          "slug": "zilliqa",
          "ticker": "ZIL"
        },
        {
          "name": "OnX Finance",
          "slug": "onx-finance",
          "ticker": "ONX"
        },
        {
          "name": "Substratum",
          "slug": "substratum",
          "ticker": "SUB"
        },
        {
          "name": "LinkToken",
          "slug": "linktoken",
          "ticker": "LTK"
        },
        {
          "name": "Lition",
          "slug": "lition",
          "ticker": "LIT"
        },
        {
          "name": "BHPCash",
          "slug": "bhpcash",
          "ticker": "BHPC"
        },
        {
          "name": "Unibright",
          "slug": "unibright",
          "ticker": "UBT"
        },
        {
          "name": "XDC Network",
          "slug": "xdc-network",
          "ticker": "XDC"
        },
        {
          "name": "aXpire",
          "slug": "axpire",
          "ticker": "AXP"
        },
        {
          "name": "Duck DAO",
          "slug": "duck-dao",
          "ticker": "DUCK"
        },
        {
          "name": "Blockpass",
          "slug": "blockpass",
          "ticker": "PASS"
        },
        {
          "name": "Streamr",
          "slug": "streamr-datacoin",
          "ticker": "DATA"
        },
        {
          "name": "Muhdo Hub",
          "slug": "muhdo-hub",
          "ticker": "DNA"
        },
        {
          "name": "Cryptify AI",
          "slug": "cryptify-ai-ethereum",
          "ticker": "CRAI"
        },
        {
          "name": "888",
          "slug": "888-meme",
          "ticker": "888"
        },
        {
          "name": "Everyworld",
          "slug": "everyworld",
          "ticker": "EVERY"
        },
        {
          "name": "Trinity Protocol",
          "slug": "trinity-protocol",
          "ticker": "TRI"
        },
        {
          "name": "XYO Network",
          "slug": "xyo",
          "ticker": "XYO"
        },
        {
          "name": "Radium",
          "slug": "radium",
          "ticker": "RADS"
        },
        {
          "name": "PlayGame Token",
          "slug": "playgame-token",
          "ticker": "PXG"
        },
        {
          "name": "Canya",
          "slug": "canyacoin",
          "ticker": "CAN"
        },
        {
          "name": "Euler",
          "slug": "euler-finance",
          "ticker": "EUL"
        },
        {
          "name": "VeriBlock",
          "slug": "veriblock",
          "ticker": "VBK"
        },
        {
          "name": "Scanet World Coin",
          "slug": "scanet-world-coin",
          "ticker": "SWC"
        },
        {
          "name": "MetaHash",
          "slug": "metahash",
          "ticker": "MHC"
        },
        {
          "name": "DECOIN",
          "slug": "decoin",
          "ticker": "DTEP"
        },
        {
          "name": "Metacraft",
          "slug": "metacraft",
          "ticker": "MCT"
        },
        {
          "name": "Serve",
          "slug": "serve",
          "ticker": "SERV"
        },
        {
          "name": "Nectar",
          "slug": "nectar",
          "ticker": "NEC"
        },
        {
          "name": "ICON",
          "slug": "icon",
          "ticker": "ICX"
        },
        {
          "name": "SakeToken",
          "slug": "sake-token",
          "ticker": "SAKE"
        },
        {
          "name": "BitDAO",
          "slug": "bitdao",
          "ticker": "BIT"
        },
        {
          "name": "Manta Network",
          "slug": "manta-network",
          "ticker": "MANTA"
        },
        {
          "name": "AurusX",
          "slug": "aurusx",
          "ticker": "AX"
        },
        {
          "name": "BLOX",
          "slug": "blox-pro",
          "ticker": "BLOX"
        },
        {
          "name": "Hyperion",
          "slug": "hyperion",
          "ticker": "HYN"
        },
        {
          "name": "SmarDex",
          "slug": "smardex",
          "ticker": "SDEX"
        },
        {
          "name": "IHT Real Estate Protocol",
          "slug": "iht-real-estate-protocol",
          "ticker": "IHT"
        },
        {
          "name": "Benz",
          "slug": "benz",
          "ticker": "BENZ"
        },
        {
          "name": "Chain Guardians",
          "slug": "chain-guardians",
          "ticker": "CGG"
        },
        {
          "name": "Hawksight",
          "slug": "hawksight",
          "ticker": "HAWK"
        },
        {
          "name": "Geojam Token",
          "slug": "geojam-token",
          "ticker": "JAM"
        },
        {
          "name": "Bitcoin Gold",
          "slug": "bitcoin-gold",
          "ticker": "BTG"
        },
        {
          "name": "Simple Token (OST)",
          "slug": "ost",
          "ticker": "OST"
        },
        {
          "name": "Matryx",
          "slug": "matryx",
          "ticker": "MTX"
        },
        {
          "name": "AidCoin",
          "slug": "aidcoin",
          "ticker": "AID"
        },
        {
          "name": "Adventure Gold",
          "slug": "adventure-gold",
          "ticker": "AGLD"
        },
        {
          "name": "Bitcoin Interest",
          "slug": "bitcoin-interest",
          "ticker": "BCI"
        },
        {
          "name": "MultiVAC",
          "slug": "multivac",
          "ticker": "MTV"
        },
        {
          "name": "Atari Token",
          "slug": "atari-token",
          "ticker": "ATRI"
        },
        {
          "name": "OneRare",
          "slug": "onerare",
          "ticker": "ORARE"
        },
        {
          "name": "Websea",
          "slug": "websea",
          "ticker": "WBS"
        },
        {
          "name": "Cloud",
          "slug": "sanctum-cloud",
          "ticker": "CLOUD"
        },
        {
          "name": "Ondo",
          "slug": "ondo-finance",
          "ticker": "ONDO"
        },
        {
          "name": "Callisto Network",
          "slug": "callisto-network",
          "ticker": "CLO"
        },
        {
          "name": "Auctus",
          "slug": "auctus",
          "ticker": "AUC"
        },
        {
          "name": "Magic Cube Coin",
          "slug": "magic-cube-coin",
          "ticker": "MCC"
        },
        {
          "name": "Aladdin",
          "slug": "aladdin",
          "ticker": "ADN"
        },
        {
          "name": "DAOventures",
          "slug": "daoventures",
          "ticker": "DVD"
        },
        {
          "name": "Jarvis+",
          "slug": "jarvis",
          "ticker": "JAR"
        },
        {
          "name": "The Dons",
          "slug": "the-dons",
          "ticker": "DONS"
        },
        {
          "name": "Cheems Inu (new)",
          "slug": "cheems-inu-new",
          "ticker": "CINU"
        },
        {
          "name": "Dragonchain",
          "slug": "dragonchain",
          "ticker": "DRGN"
        },
        {
          "name": "Holo",
          "slug": "holo",
          "ticker": "HOT"
        },
        {
          "name": "Incent",
          "slug": "incent",
          "ticker": "INCNT"
        },
        {
          "name": "Content Neutrality Network",
          "slug": "content-neutrality-network",
          "ticker": "CNN"
        },
        {
          "name": "Covesting",
          "slug": "covesting",
          "ticker": "COV"
        },
        {
          "name": "Atonomi",
          "slug": "atonomi",
          "ticker": "ATMI"
        },
        {
          "name": "Hegic [on Ethereum]",
          "slug": "hegic",
          "ticker": "HEGIC"
        },
        {
          "name": "Woodcoin",
          "slug": "woodcoin",
          "ticker": "LOG"
        },
        {
          "name": "Fundamenta",
          "slug": "fundamenta",
          "ticker": "FMTA"
        },
        {
          "name": "Misbloc",
          "slug": "misbloc",
          "ticker": "MSB"
        },
        {
          "name": "Bistroo",
          "slug": "bistroo",
          "ticker": "BIST"
        },
        {
          "name": "KIKICat",
          "slug": "kikicat",
          "ticker": "KIKI"
        },
        {
          "name": "Pirate Nation",
          "slug": "pirate-nation",
          "ticker": "PIRATE"
        },
        {
          "name": "Byte",
          "slug": "byte",
          "ticker": "BYTE"
        },
        {
          "name": "PeiPei",
          "slug": "peipei-coin",
          "ticker": "PEIPEI"
        },
        {
          "name": "XT Stablecoin XTUSD",
          "slug": "xtusd",
          "ticker": "XTUSD"
        },
        {
          "name": "LOBO•THE•WOLF•PUP",
          "slug": "lobo-the-wolf-pup",
          "ticker": "LOBO"
        },
        {
          "name": "Maecenas",
          "slug": "maecenas",
          "ticker": "ART"
        },
        {
          "name": "AVINOC",
          "slug": "avinoc",
          "ticker": "AVINOC"
        },
        {
          "name": "WinStars.live",
          "slug": "winstars-live",
          "ticker": "WNL"
        },
        {
          "name": "PalletOne",
          "slug": "palletone",
          "ticker": "PTN"
        },
        {
          "name": "Coinsuper Ecosystem Network",
          "slug": "coinsuper-ecosystem-network",
          "ticker": "CEN"
        },
        {
          "name": "Brazilian Digital Token",
          "slug": "brz",
          "ticker": "BRZ"
        },
        {
          "name": "WINk",
          "slug": "wink",
          "ticker": "WIN"
        },
        {
          "name": "GAMEE",
          "slug": "gamee",
          "ticker": "GMEE"
        },
        {
          "name": "crow with knife",
          "slug": "crow-with-knife",
          "ticker": "CAW"
        },
        {
          "name": "ArGo",
          "slug": "argoapp",
          "ticker": "ARGO"
        },
        {
          "name": "UniMex Network",
          "slug": "unimex-network",
          "ticker": "UMX"
        },
        {
          "name": "Forefront",
          "slug": "forefront",
          "ticker": "FF"
        },
        {
          "name": "BaaSid",
          "slug": "baasid",
          "ticker": "BAAS"
        },
        {
          "name": "Ethereum",
          "slug": "ethereum",
          "ticker": "ETH"
        },
        {
          "name": "NPER",
          "slug": "nper",
          "ticker": "NPER"
        },
        {
          "name": "Zentry",
          "slug": "zentry",
          "ticker": "ZENT"
        },
        {
          "name": "BOOM",
          "slug": "boom",
          "ticker": "BOOM"
        },
        {
          "name": "Webflix Token",
          "slug": "webflix-token",
          "ticker": "WFX"
        },
        {
          "name": "UTRUST",
          "slug": "utrust",
          "ticker": "UTK"
        },
        {
          "name": "XMON",
          "slug": "xmon",
          "ticker": "XMON"
        },
        {
          "name": "YOUR AI",
          "slug": "your",
          "ticker": "YOURAI"
        },
        {
          "name": "Art de Finance",
          "slug": "art-de-finance",
          "ticker": "ADF"
        },
        {
          "name": "NEM",
          "slug": "nem",
          "ticker": "XEM"
        },
        {
          "name": "Santiment",
          "slug": "santiment",
          "ticker": "SAN"
        },
        {
          "name": "Elastos",
          "slug": "elastos",
          "ticker": "ELA"
        },
        {
          "name": "Boba Network",
          "slug": "boba-network",
          "ticker": "BOBA"
        },
        {
          "name": "Datawallet",
          "slug": "datawallet",
          "ticker": "DXT"
        },
        {
          "name": "GoNetwork",
          "slug": "gonetwork",
          "ticker": "GOT"
        },
        {
          "name": "zkSync",
          "slug": "zksync",
          "ticker": "ZK"
        },
        {
          "name": "SatoshiVM",
          "slug": "satoshivm",
          "ticker": "SAVM"
        },
        {
          "name": "Loom Network",
          "slug": "loom-network",
          "ticker": "LOOM"
        },
        {
          "name": "CHEWY",
          "slug": "chewy-token",
          "ticker": "CHWY"
        },
        {
          "name": "Brokoli Network",
          "slug": "brokoli-network",
          "ticker": "BRKL"
        },
        {
          "name": "Luigi Inu",
          "slug": "luigi-inu",
          "ticker": "LUIGI"
        },
        {
          "name": "OX Coin [on Arbitrum]",
          "slug": "arb-ox-coin",
          "ticker": "OX"
        },
        {
          "name": "Katalyo",
          "slug": "katalyo",
          "ticker": "KTLYO"
        },
        {
          "name": "Ivy",
          "slug": "ivy",
          "ticker": "IVY"
        },
        {
          "name": "ChainX",
          "slug": "chainx",
          "ticker": "PCX"
        },
        {
          "name": "Lympo",
          "slug": "lympo",
          "ticker": "LYM"
        },
        {
          "name": "Lith Token",
          "slug": "lith-token",
          "ticker": "LITH"
        },
        {
          "name": "Public Mint",
          "slug": "public-mint",
          "ticker": "MINT"
        },
        {
          "name": "Sarcophagus",
          "slug": "sarcophagus",
          "ticker": "SARCO"
        },
        {
          "name": "Birake",
          "slug": "birake",
          "ticker": "BIR"
        },
        {
          "name": "Caspian",
          "slug": "caspian",
          "ticker": "CSP"
        },
        {
          "name": "Function X",
          "slug": "function-x",
          "ticker": "FX"
        },
        {
          "name": "YEE",
          "slug": "yee",
          "ticker": "YEE"
        },
        {
          "name": "Kleros",
          "slug": "kleros",
          "ticker": "PNK"
        },
        {
          "name": "Ulord",
          "slug": "ulord",
          "ticker": "UT"
        },
        {
          "name": "NvirWorld",
          "slug": "nvirworld",
          "ticker": "NVIR"
        },
        {
          "name": "DecentraWeb",
          "slug": "decentraweb",
          "ticker": "DWEB"
        },
        {
          "name": "Agoras: Currency of Tau",
          "slug": "agoras-tokens",
          "ticker": "AGRS"
        },
        {
          "name": "Thingschain",
          "slug": "thingschain",
          "ticker": "TIC"
        },
        {
          "name": "CWV Chain",
          "slug": "cwv-chain",
          "ticker": "CWV"
        },
        {
          "name": "SingularityDAO",
          "slug": "singularitydao",
          "ticker": "SDAO"
        },
        {
          "name": "Propy",
          "slug": "propy",
          "ticker": "PRO"
        },
        {
          "name": "HTMLCOIN",
          "slug": "html-coin",
          "ticker": "HTML"
        },
        {
          "name": "CyberMusic",
          "slug": "cybermusic",
          "ticker": "CYMT"
        },
        {
          "name": "Fivebalance ",
          "slug": "fivebalance",
          "ticker": "FBN"
        },
        {
          "name": "Splyt",
          "slug": "splyt",
          "ticker": "SHOPX"
        },
        {
          "name": "BobaCat",
          "slug": "bobacat",
          "ticker": "PSPS"
        },
        {
          "name": "Blendr Network",
          "slug": "blendr-network",
          "ticker": "BLENDR"
        },
        {
          "name": "PulseX",
          "slug": "pulsex",
          "ticker": "PLSX"
        },
        {
          "name": "Smartup",
          "slug": "smartup",
          "ticker": "SMARTUP"
        },
        {
          "name": "carVertical",
          "slug": "carvertical",
          "ticker": "CV"
        },
        {
          "name": "Rage Fan",
          "slug": "rage-fan",
          "ticker": "RAGE"
        },
        {
          "name": "Boltt Coin",
          "slug": "boltt-coin",
          "ticker": "BOLTT"
        },
        {
          "name": "Dapp Token",
          "slug": "dapp-token",
          "ticker": "DAPPT"
        },
        {
          "name": "MahaDAO",
          "slug": "mahadao",
          "ticker": "MAHA"
        },
        {
          "name": "ZClassic",
          "slug": "zclassic",
          "ticker": "ZCL"
        },
        {
          "name": "Dogey-Inu",
          "slug": "dogey-inu",
          "ticker": "DINU"
        },
        {
          "name": "BoatPilot Token",
          "slug": "boat-pilot-token",
          "ticker": "NAVY"
        },
        {
          "name": "Qredit",
          "slug": "qredit",
          "ticker": "QRT"
        },
        {
          "name": "Pluton",
          "slug": "pluton",
          "ticker": "PLU"
        },
        {
          "name": "UGAS",
          "slug": "ugas",
          "ticker": "UGAS"
        },
        {
          "name": "IterationSyndicate",
          "slug": "iterationsyndicate",
          "ticker": "ITS"
        },
        {
          "name": "APY.Finance",
          "slug": "apy-finance",
          "ticker": "APY"
        },
        {
          "name": "Hiveterminal Token",
          "slug": "hiveterminal-token",
          "ticker": "HVN"
        },
        {
          "name": "Sylo",
          "slug": "sylo",
          "ticker": "SYLO"
        },
        {
          "name": "Quantum Resistant Ledger",
          "slug": "quantum-resistant-ledger",
          "ticker": "QRL"
        },
        {
          "name": "MassGrid",
          "slug": "massgrid",
          "ticker": "MGD"
        },
        {
          "name": "Dexter G",
          "slug": "dexter-g",
          "ticker": "DXG"
        },
        {
          "name": "Apex",
          "slug": "apex",
          "ticker": "CPX"
        },
        {
          "name": "PRIA",
          "slug": "pria",
          "ticker": "PRIA"
        },
        {
          "name": "Power Ledger",
          "slug": "power-ledger",
          "ticker": "POWR"
        },
        {
          "name": "Lisk",
          "slug": "lisk",
          "ticker": "LSK"
        },
        {
          "name": "HedgeTrade",
          "slug": "hedgetrade",
          "ticker": "HEDG"
        },
        {
          "name": "WEMIX",
          "slug": "wemix",
          "ticker": "WEMIX"
        },
        {
          "name": "nOS",
          "slug": "nos",
          "ticker": "NOS"
        },
        {
          "name": "SLG.GAMES",
          "slug": "land-of-conquest",
          "ticker": "SLG"
        },
        {
          "name": "The HUSL",
          "slug": "the-husl",
          "ticker": "HUSL"
        },
        {
          "name": "Coreto",
          "slug": "coreto",
          "ticker": "COR"
        },
        {
          "name": "Banyan Network",
          "slug": "banyan-network",
          "ticker": "BBN"
        },
        {
          "name": "Mossland",
          "slug": "moss-coin",
          "ticker": "MOC"
        },
        {
          "name": "Bluzelle",
          "slug": "bluzelle",
          "ticker": "BLZ"
        },
        {
          "name": "ShineChain",
          "slug": "shinechain",
          "ticker": "SHE"
        },
        {
          "name": "CGC Token",
          "slug": "cgc-token",
          "ticker": "CGC"
        },
        {
          "name": "CargoX",
          "slug": "cargox",
          "ticker": "CXO"
        },
        {
          "name": "Oxen",
          "slug": "loki",
          "ticker": "LOKI"
        },
        {
          "name": "Enjin Coin",
          "slug": "enjin-coin",
          "ticker": "ENJ"
        },
        {
          "name": "STATERA",
          "slug": "statera",
          "ticker": "STA"
        },
        {
          "name": "CRD Network",
          "slug": "crdnetwork",
          "ticker": "CRD"
        },
        {
          "name": "UBIX.Network",
          "slug": "ubix-network",
          "ticker": "UBX"
        },
        {
          "name": "Collateral Pay",
          "slug": "collateral-pay",
          "ticker": "COLL"
        },
        {
          "name": "Vexanium",
          "slug": "vexanium",
          "ticker": "VEX"
        },
        {
          "name": "DDKoin",
          "slug": "ddkoin",
          "ticker": "DDK"
        },
        {
          "name": "DADI",
          "slug": "dadi",
          "ticker": "DADI"
        },
        {
          "name": "Credits",
          "slug": "credits",
          "ticker": "CS"
        },
        {
          "name": "CommerceBlock",
          "slug": "commerceblock",
          "ticker": "CBT"
        },
        {
          "name": "Balancer [on Optimism]",
          "slug": "o-balancer",
          "ticker": "BAL"
        },
        {
          "name": "Kcash",
          "slug": "kcash",
          "ticker": "KCASH"
        },
        {
          "name": "Tcash",
          "slug": "tcash",
          "ticker": "TCASH"
        },
        {
          "name": "OWNDATA ",
          "slug": "owndata",
          "ticker": "OWN"
        },
        {
          "name": "Flash",
          "slug": "flash",
          "ticker": "FLASH"
        },
        {
          "name": "Zero Utility Token",
          "slug": "zero-utility-token",
          "ticker": "ZUT"
        },
        {
          "name": "Warp Finance",
          "slug": "warp-finance",
          "ticker": "WARP"
        },
        {
          "name": "ZELIX",
          "slug": "zelix",
          "ticker": "ZELIX"
        },
        {
          "name": "MongolNFT Coin",
          "slug": "mongolnft-coin",
          "ticker": "MNFT"
        },
        {
          "name": "Hydro Protocol",
          "slug": "hydro-protocol",
          "ticker": "HOT"
        },
        {
          "name": "EOS",
          "slug": "eos",
          "ticker": "EOS"
        },
        {
          "name": "Loopring",
          "slug": "loopring",
          "ticker": "LRC"
        },
        {
          "name": "MobileGo",
          "slug": "mobilego",
          "ticker": "MGO"
        },
        {
          "name": "VeChain",
          "slug": "vechain",
          "ticker": "VET"
        },
        {
          "name": "Polymath",
          "slug": "polymath-network",
          "ticker": "POLY"
        },
        {
          "name": "Ripio Credit Network",
          "slug": "ripio-credit-network",
          "ticker": "RCN"
        },
        {
          "name": "saffron.finance",
          "slug": "saffron-finance",
          "ticker": "SFI"
        },
        {
          "name": "APR Coin",
          "slug": "apr-coin",
          "ticker": "APR"
        },
        {
          "name": "BarnBridge",
          "slug": "barnbridge",
          "ticker": "BOND"
        },
        {
          "name": "Ultima",
          "slug": "ultima",
          "ticker": "ULTIMA"
        },
        {
          "name": "Tapmydata",
          "slug": "tapmydata",
          "ticker": "TAP"
        },
        {
          "name": "Chonk",
          "slug": "chonk",
          "ticker": "CHONK"
        },
        {
          "name": "OptionRoom",
          "slug": "optionroom",
          "ticker": "ROOM"
        },
        {
          "name": "Artificial Liquid Intelligence",
          "slug": "alethea-artificial-liquid-intelligence-token",
          "ticker": "ALI"
        },
        {
          "name": "iExec RLC",
          "slug": "rlc",
          "ticker": "RLC"
        },
        {
          "name": "ArchLoot",
          "slug": "archloot",
          "ticker": "AL"
        },
        {
          "name": "Storj",
          "slug": "storj",
          "ticker": "STORJ"
        },
        {
          "name": "HOQU",
          "slug": "hoqu",
          "ticker": "HQX"
        },
        {
          "name": "Umbrella Network",
          "slug": "umbrella-network",
          "ticker": "UMB"
        },
        {
          "name": "Moonriver",
          "slug": "moonriver",
          "ticker": "MOVR"
        },
        {
          "name": "Qtum",
          "slug": "qtum",
          "ticker": "QTUM"
        },
        {
          "name": "Fera",
          "slug": "fera",
          "ticker": "FERA"
        },
        {
          "name": "BITTO",
          "slug": "bitto",
          "ticker": "BITTO"
        },
        {
          "name": "trac (Ordinals)",
          "slug": "trac",
          "ticker": "TRAC"
        },
        {
          "name": "Monsterra (MSTR)",
          "slug": "monsterra",
          "ticker": "MSTR"
        },
        {
          "name": "ETHPad",
          "slug": "ethpad",
          "ticker": "ETHPAD"
        },
        {
          "name": "SmartWorld Global",
          "slug": "smartworld-global",
          "ticker": "SWGT"
        },
        {
          "name": "KANDO AI",
          "slug": "kando-ai",
          "ticker": "KANDO"
        },
        {
          "name": "Consensus",
          "slug": "consensus",
          "ticker": "SEN"
        },
        {
          "name": "Monero",
          "slug": "monero",
          "ticker": "XMR"
        },
        {
          "name": "YGGDRASH",
          "slug": "yeed",
          "ticker": "YEED"
        },
        {
          "name": "Orbs",
          "slug": "orbs",
          "ticker": "ORBS"
        },
        {
          "name": "Vetri",
          "slug": "vetri",
          "ticker": "VLD"
        },
        {
          "name": "Centauri",
          "slug": "centauri",
          "ticker": "CTX"
        },
        {
          "name": "WOLLO",
          "slug": "wollo",
          "ticker": "WLO"
        },
        {
          "name": "DAOstack",
          "slug": "daostack",
          "ticker": "GEN"
        },
        {
          "name": "Flux",
          "slug": "zel",
          "ticker": "FLUX"
        },
        {
          "name": "Tezos",
          "slug": "tezos",
          "ticker": "XTZ"
        },
        {
          "name": "Dether",
          "slug": "dether",
          "ticker": "DTH"
        },
        {
          "name": "Tornado Cash",
          "slug": "torn",
          "ticker": "TORN"
        },
        {
          "name": "Dero",
          "slug": "dero",
          "ticker": "DERO"
        },
        {
          "name": "Fire Lotto",
          "slug": "fire-lotto",
          "ticker": "FLOT"
        },
        {
          "name": "Humaniq",
          "slug": "humaniq",
          "ticker": "HMQ"
        },
        {
          "name": "ContentBox",
          "slug": "contentbox",
          "ticker": "BOX"
        },
        {
          "name": "Populous",
          "slug": "populous",
          "ticker": "PPT"
        },
        {
          "name": "BOX Token",
          "slug": "box-token",
          "ticker": "BOX"
        },
        {
          "name": "Yeet",
          "slug": "yeet-it",
          "ticker": "YEET"
        },
        {
          "name": "usdc-b",
          "slug": "usdc-b",
          "ticker": "USDC-B"
        },
        {
          "name": "Waves",
          "slug": "waves",
          "ticker": "WAVES"
        },
        {
          "name": "FaraLand",
          "slug": "faraland",
          "ticker": "FARA"
        },
        {
          "name": "BRN Metaverse",
          "slug": "brn-metaverse",
          "ticker": "BRN"
        },
        {
          "name": "Selfkey",
          "slug": "selfkey",
          "ticker": "KEY"
        },
        {
          "name": "Bilaxy Token",
          "slug": "bilaxy-token",
          "ticker": "BIA"
        },
        {
          "name": "CoinMetro",
          "slug": "coinmetro-token",
          "ticker": "XCM"
        },
        {
          "name": "AdShares",
          "slug": "adshares",
          "ticker": "ADS"
        },
        {
          "name": "Wirex",
          "slug": "wirex-token",
          "ticker": "WXT"
        },
        {
          "name": "Endor",
          "slug": "endor-protocol",
          "ticker": "EDR"
        },
        {
          "name": "Budbo",
          "slug": "budbo",
          "ticker": "BUBO"
        },
        {
          "name": "e-Money",
          "slug": "e-money-coin",
          "ticker": "NGM"
        },
        {
          "name": "DACSEE",
          "slug": "dacsee",
          "ticker": "DACS"
        },
        {
          "name": "Blank Wallet",
          "slug": "blank-wallet",
          "ticker": "BLANK"
        },
        {
          "name": "Neutrino Token",
          "slug": "neutrino-system-base-token",
          "ticker": "NSBT"
        },
        {
          "name": "Sora Validator Token",
          "slug": "sora-validator-token",
          "ticker": "VAL"
        },
        {
          "name": "KLEVA Protocol",
          "slug": "kleva-protocol",
          "ticker": "KLEVA"
        },
        {
          "name": "Ycash",
          "slug": "ycash",
          "ticker": "YEC"
        },
        {
          "name": "NuriTopia",
          "slug": "nuritopia",
          "ticker": "NBLU"
        },
        {
          "name": "MileVerse",
          "slug": "mileverse",
          "ticker": "MVC"
        },
        {
          "name": "IOST",
          "slug": "iostoken",
          "ticker": "IOST"
        },
        {
          "name": "Crowd Machine",
          "slug": "crowd-machine",
          "ticker": "CMCT"
        },
        {
          "name": "Puregold.io",
          "slug": "puregold-token",
          "ticker": "PGT"
        },
        {
          "name": "Scry.info",
          "slug": "scryinfo",
          "ticker": "DDD"
        },
        {
          "name": "Cross The Ages",
          "slug": "cross-the-ages",
          "ticker": "CTA"
        },
        {
          "name": "Lion Cat",
          "slug": "lion-cat",
          "ticker": "LCAT"
        },
        {
          "name": "BitShares",
          "slug": "bitshares",
          "ticker": "BTS"
        },
        {
          "name": "Wen",
          "slug": "wen",
          "ticker": "WEN"
        },
        {
          "name": "ZPER",
          "slug": "zper",
          "ticker": "ZPR"
        },
        {
          "name": "DUO Network",
          "slug": "duo-network-token",
          "ticker": "DUO"
        },
        {
          "name": "Digitex",
          "slug": "digitex-futures",
          "ticker": "DGTX"
        },
        {
          "name": "Red Pulse",
          "slug": "red-pulse",
          "ticker": "PHX"
        },
        {
          "name": "Silverway",
          "slug": "silverway",
          "ticker": "SLV"
        },
        {
          "name": "QuarkChain",
          "slug": "quarkchain",
          "ticker": "QKC"
        },
        {
          "name": "Like",
          "slug": "likecoin",
          "ticker": "LIKE"
        },
        {
          "name": "Ruff",
          "slug": "ruff",
          "ticker": "RUFF"
        },
        {
          "name": "Counterparty",
          "slug": "counterparty",
          "ticker": "XCP"
        },
        {
          "name": "AppCoins",
          "slug": "appcoins",
          "ticker": "APPC"
        },
        {
          "name": "Usechain Token",
          "slug": "usechain-token",
          "ticker": "USE"
        },
        {
          "name": "Ampleforth",
          "slug": "ampleforth",
          "ticker": "AMPL"
        },
        {
          "name": "Wibson",
          "slug": "wibson",
          "ticker": "WIB"
        },
        {
          "name": "Vulcan Forged PYR",
          "slug": "vulcan-forged-pyr",
          "ticker": "PYR"
        },
        {
          "name": "Ojamu",
          "slug": "ojamu",
          "ticker": "OJA"
        },
        {
          "name": "White Standard",
          "slug": "white-standard",
          "ticker": "WSD"
        },
        {
          "name": "Genaro Network",
          "slug": "genaro-network",
          "ticker": "GNX"
        },
        {
          "name": "Vega Protocol",
          "slug": "vegaprotocol",
          "ticker": "VEGA"
        },
        {
          "name": "QunQun",
          "slug": "qunqun",
          "ticker": "QUN"
        },
        {
          "name": "SolarCoin",
          "slug": "solarcoin",
          "ticker": "SLR"
        },
        {
          "name": "Bitcoin",
          "slug": "bitcoin",
          "ticker": "BTC"
        },
        {
          "name": "BHPCoin",
          "slug": "bhp-coin",
          "ticker": "BHP"
        },
        {
          "name": "Ternoa",
          "slug": "ternoa",
          "ticker": "CAPS"
        },
        {
          "name": "MASTERNET",
          "slug": "masternet",
          "ticker": "MASH"
        },
        {
          "name": "Aergo",
          "slug": "aergo",
          "ticker": "AERGO"
        },
        {
          "name": "ZKBase",
          "slug": "zkbase",
          "ticker": "ZKB"
        },
        {
          "name": "tBTC",
          "slug": "tbtc",
          "ticker": "TBTC"
        },
        {
          "name": "Swapfolio",
          "slug": "swapfolio",
          "ticker": "SWFL"
        },
        {
          "name": "KONET",
          "slug": "konet",
          "ticker": "KONET"
        },
        {
          "name": "S.C. Corinthians Fan Token",
          "slug": "sc-corinthians-fan-token",
          "ticker": "SCCP"
        },
        {
          "name": "Raiden Network Token",
          "slug": "raiden-network-token",
          "ticker": "RDN"
        },
        {
          "name": "BLOCKv",
          "slug": "blockv",
          "ticker": "VEE"
        },
        {
          "name": "Arionum",
          "slug": "arionum",
          "ticker": "ARO"
        },
        {
          "name": "USDJ",
          "slug": "usdj",
          "ticker": "USDJ"
        },
        {
          "name": "adbank",
          "slug": "adbank",
          "ticker": "ADB"
        },
        {
          "name": "Feathercoin",
          "slug": "feathercoin",
          "ticker": "FTC"
        },
        {
          "name": "Open Platform",
          "slug": "open-platform",
          "ticker": "OPEN"
        },
        {
          "name": "Viberate",
          "slug": "viberate",
          "ticker": "VIB"
        },
        {
          "name": "Devery",
          "slug": "devery",
          "ticker": "EVE"
        },
        {
          "name": "Switcheo",
          "slug": "switcheo",
          "ticker": "SWTH"
        },
        {
          "name": "Offshift",
          "slug": "offshift",
          "ticker": "XFT"
        },
        {
          "name": "BOTIFY",
          "slug": "botify",
          "ticker": "BOTIFY"
        },
        {
          "name": "Hacken Token",
          "slug": "hackenai",
          "ticker": "HAI"
        },
        {
          "name": "Cryptrust",
          "slug": "cryptrust",
          "ticker": "CTRT"
        },
        {
          "name": "GNY",
          "slug": "gny",
          "ticker": "GNY"
        },
        {
          "name": "Zerobank",
          "slug": "zerobank",
          "ticker": "ZB"
        },
        {
          "name": "3DCoin",
          "slug": "3dcoin",
          "ticker": "3DC"
        },
        {
          "name": "TouchCon",
          "slug": "touchcon",
          "ticker": "TOC"
        },
        {
          "name": "Hero Node",
          "slug": "heronode",
          "ticker": "HER"
        },
        {
          "name": "Essentia",
          "slug": "essentia",
          "ticker": "ESS"
        },
        {
          "name": "Factom",
          "slug": "factom",
          "ticker": "FCT"
        },
        {
          "name": "Frax Staked Ether",
          "slug": "frax-staked-ether",
          "ticker": "SFRXETH"
        },
        {
          "name": "NaPoleonX",
          "slug": "napoleonx",
          "ticker": "NPX"
        },
        {
          "name": "FINSCHIA",
          "slug": "finschia",
          "ticker": "FNSA"
        },
        {
          "name": "Finxflo",
          "slug": "finxflo",
          "ticker": "FXF"
        },
        {
          "name": "PolySwarm",
          "slug": "polyswarm",
          "ticker": "NCT"
        },
        {
          "name": "Egretia",
          "slug": "egretia",
          "ticker": "EGT"
        },
        {
          "name": "VINChain",
          "slug": "vinchain",
          "ticker": "VIN"
        },
        {
          "name": "DAO Maker",
          "slug": "dao-maker",
          "ticker": "DAO"
        },
        {
          "name": "TGAME",
          "slug": "tgame",
          "ticker": "TGAME"
        },
        {
          "name": "AIOZ Network",
          "slug": "aioz-network",
          "ticker": "AIOZ"
        },
        {
          "name": "Dock",
          "slug": "dock",
          "ticker": "DOCK"
        },
        {
          "name": "Lingose",
          "slug": "lingose",
          "ticker": "LING"
        },
        {
          "name": "Jibrel Network",
          "slug": "jibrel-network",
          "ticker": "JNT"
        },
        {
          "name": "ARAW",
          "slug": "araw",
          "ticker": "USDE"
        },
        {
          "name": "Deri Protocol",
          "slug": "deri-protocol",
          "ticker": "DERI"
        },
        {
          "name": "Bread",
          "slug": "bread",
          "ticker": "BRD"
        },
        {
          "name": "Amon",
          "slug": "amon",
          "ticker": "AMN"
        },
        {
          "name": "Litex",
          "slug": "litex",
          "ticker": "LXT"
        },
        {
          "name": "Neurotoken",
          "slug": "neurotoken",
          "ticker": "NTK"
        },
        {
          "name": "VVS Finance",
          "slug": "vvs-finance",
          "ticker": "VVS"
        },
        {
          "name": "Life Crypto",
          "slug": "life-crypto",
          "ticker": "LIFE"
        },
        {
          "name": "Arepacoin",
          "slug": "arepacoin",
          "ticker": "AREPA"
        },
        {
          "name": "Altlayer",
          "slug": "altlayer",
          "ticker": "ALT"
        },
        {
          "name": "XY Finance",
          "slug": "xy-finance",
          "ticker": "XY"
        },
        {
          "name": "ZIMBOCASH",
          "slug": "zimbocash",
          "ticker": "ZASH"
        },
        {
          "name": "Restart Energy MWAT",
          "slug": "restart-energy-mwat",
          "ticker": "MWAT"
        },
        {
          "name": "ApexToken",
          "slug": "apextoken",
          "ticker": "APX"
        },
        {
          "name": "High Performance Blockchain",
          "slug": "high-performance-blockchain",
          "ticker": "HPB"
        },
        {
          "name": "MEET.ONE",
          "slug": "meetone",
          "ticker": "MEETONE"
        },
        {
          "name": "CoinFi",
          "slug": "coinfi",
          "ticker": "COFI"
        },
        {
          "name": "Whiteheart",
          "slug": "whiteheart",
          "ticker": "WHITE"
        },
        {
          "name": "UFO Gaming",
          "slug": "ufo-gaming",
          "ticker": "UFO"
        },
        {
          "name": "Lithium",
          "slug": "lithium",
          "ticker": "LITH"
        },
        {
          "name": "Lingo",
          "slug": "lingo",
          "ticker": "LINGO"
        },
        {
          "name": "UPCX",
          "slug": "upcx",
          "ticker": "UPC"
        },
        {
          "name": "Sonar",
          "slug": "sonar",
          "ticker": "PING"
        },
        {
          "name": "Fabric Token",
          "slug": "fabric-token",
          "ticker": "FT"
        },
        {
          "name": "Blockchain Foundation for Innovation & Collaboration",
          "slug": "best-fintech-investment-coin",
          "ticker": "BFIC"
        },
        {
          "name": "Einsteinium",
          "slug": "einsteinium",
          "ticker": "EMC2"
        },
        {
          "name": "BitKan",
          "slug": "bitkan",
          "ticker": "KAN"
        },
        {
          "name": "EpiK Protocol",
          "slug": "epik-protocol",
          "ticker": "EPK"
        },
        {
          "name": "Engagement Token",
          "slug": "engagement-token",
          "ticker": "ENGT"
        },
        {
          "name": "Rangers Protocol",
          "slug": "rangers-protocol",
          "ticker": "RPG"
        },
        {
          "name": "BigONE Token",
          "slug": "bigone-token",
          "ticker": "ONE"
        },
        {
          "name": "OMG Network",
          "slug": "omisego",
          "ticker": "OMG"
        },
        {
          "name": "Compound Wrapped BTC",
          "slug": "compound-wrapped-btc",
          "ticker": "CWBTC"
        },
        {
          "name": "PlanetWatch",
          "slug": "planetwatch",
          "ticker": "PLANETS"
        },
        {
          "name": "Compound Basic Attention Token",
          "slug": "compound-basic-attention-token",
          "ticker": "CBAT"
        },
        {
          "name": "All In",
          "slug": "all-in",
          "ticker": "ALLIN"
        },
        {
          "name": "Naviaddress",
          "slug": "naviaddress",
          "ticker": "NAVI"
        },
        {
          "name": "Tripio",
          "slug": "tripio",
          "ticker": "TRIO"
        },
        {
          "name": "adToken",
          "slug": "adtoken",
          "ticker": "ADT"
        },
        {
          "name": "Own",
          "slug": "own",
          "ticker": "CHX"
        },
        {
          "name": "Perpetual Protocol [on Optimism]",
          "slug": "o-perpetual-protocol",
          "ticker": "PERP"
        },
        {
          "name": "ORS Group",
          "slug": "ors-group",
          "ticker": "ORS"
        },
        {
          "name": "Origin Sport",
          "slug": "origin-sport",
          "ticker": "ORS"
        },
        {
          "name": "Nuls",
          "slug": "nuls",
          "ticker": "NULS"
        },
        {
          "name": "EncrypGen",
          "slug": "encrypgen",
          "ticker": "DNA"
        },
        {
          "name": "Prosper",
          "slug": "prosper",
          "ticker": "PROS"
        },
        {
          "name": "RightMesh",
          "slug": "rightmesh",
          "ticker": "RMESH"
        },
        {
          "name": "Everex",
          "slug": "everex",
          "ticker": "EVX"
        },
        {
          "name": "Electronic Energy Coin",
          "slug": "electronic-energy-coin",
          "ticker": "E2C"
        },
        {
          "name": "Mobius",
          "slug": "mobius",
          "ticker": "MOBI"
        },
        {
          "name": "Litentry",
          "slug": "litentry",
          "ticker": "LIT"
        },
        {
          "name": "Magic Square",
          "slug": "bnb-magic-square",
          "ticker": "SQR"
        },
        {
          "name": "Bankex",
          "slug": "bankex",
          "ticker": "BKX"
        },
        {
          "name": "AstroSwap",
          "slug": "astroswap",
          "ticker": "ASTRO"
        },
        {
          "name": "Signum",
          "slug": "signum",
          "ticker": "SIGNA"
        },
        {
          "name": "Rate3 Network",
          "slug": "rate3",
          "ticker": "RTE"
        },
        {
          "name": "Matrix AI Network",
          "slug": "matrix-ai-network",
          "ticker": "MAN"
        },
        {
          "name": "Monetha",
          "slug": "monetha",
          "ticker": "MTH"
        },
        {
          "name": "SPX6900",
          "slug": "spx6900",
          "ticker": "SPX"
        },
        {
          "name": "Compound Wrapped BTC 2",
          "slug": "compound-wrapped-bitcoin-2",
          "ticker": "CWBTC"
        },
        {
          "name": "Compound 0x",
          "slug": "compound-0x",
          "ticker": "CZRX"
        },
        {
          "name": "Game7",
          "slug": "game7",
          "ticker": "G7"
        },
        {
          "name": "Wings",
          "slug": "wings",
          "ticker": "WINGS"
        },
        {
          "name": "Arbidex",
          "slug": "arbidex",
          "ticker": "ABX"
        },
        {
          "name": "SmartMesh",
          "slug": "smartmesh",
          "ticker": "SMT"
        },
        {
          "name": "Shido ETH",
          "slug": "shido-eth",
          "ticker": "SHIDO"
        },
        {
          "name": "NuBits",
          "slug": "nubits",
          "ticker": "USNBT"
        },
        {
          "name": "aelf",
          "slug": "aelf",
          "ticker": "ELF"
        },
        {
          "name": "Decentraland",
          "slug": "decentraland",
          "ticker": "MANA"
        },
        {
          "name": "DRP Utility",
          "slug": "drp-utility",
          "ticker": "DRPU"
        },
        {
          "name": "HoDooi.com",
          "slug": "hodooi",
          "ticker": "HOD"
        },
        {
          "name": "TNC Coin",
          "slug": "tnc-coin",
          "ticker": "TNC"
        },
        {
          "name": "Smart MFG",
          "slug": "smart-mfg",
          "ticker": "MFG"
        },
        {
          "name": "Tranche Finance",
          "slug": "tranche-finance",
          "ticker": "SLICE"
        },
        {
          "name": "Stratos",
          "slug": "stratos",
          "ticker": "STOS"
        },
        {
          "name": "Bitball Treasure",
          "slug": "bitball-treasure",
          "ticker": "BTRS"
        },
        {
          "name": "Klaus",
          "slug": "klaus",
          "ticker": "KLAUS"
        },
        {
          "name": "Wrapped Alvey Chain",
          "slug": "alvey-chain",
          "ticker": "WALV"
        },
        {
          "name": "Circuits of Value",
          "slug": "circuits-of-value",
          "ticker": "COVAL"
        },
        {
          "name": "GreenMed",
          "slug": "greenmed",
          "ticker": "GRMD"
        },
        {
          "name": "Status",
          "slug": "status",
          "ticker": "SNT"
        },
        {
          "name": "V Systems",
          "slug": "v-systems",
          "ticker": "VSYS"
        },
        {
          "name": "Ubique Chain Of Things",
          "slug": "ubique-chain-of-things",
          "ticker": "UCT"
        },
        {
          "name": "Peculium",
          "slug": "peculium",
          "ticker": "PCL"
        },
        {
          "name": "CapdaxToken",
          "slug": "capdaxtoken",
          "ticker": "XCD"
        },
        {
          "name": "Aragon Court",
          "slug": "aragon-court",
          "ticker": "ANJ"
        },
        {
          "name": "Dharma",
          "slug": "dharma",
          "ticker": "DHA"
        },
        {
          "name": "Empty Set Dollar",
          "slug": "empty-set-dollar",
          "ticker": "ESD"
        },
        {
          "name": "CACHE Gold",
          "slug": "cache-gold",
          "ticker": "CGT"
        },
        {
          "name": "Carlive Chain",
          "slug": "carlive-chain",
          "ticker": "IOV"
        },
        {
          "name": "Cindicator",
          "slug": "cindicator",
          "ticker": "CND"
        },
        {
          "name": "STASIS EURS",
          "slug": "stasis-euro",
          "ticker": "EURS"
        },
        {
          "name": "BOScoin",
          "slug": "boscoin",
          "ticker": "BOS"
        },
        {
          "name": "Universa",
          "slug": "universa",
          "ticker": "UTNP"
        },
        {
          "name": "Solar [on Ethereum]",
          "slug": "swipe",
          "ticker": "SXP"
        },
        {
          "name": "Sharpay",
          "slug": "sharpay",
          "ticker": "S"
        },
        {
          "name": "Rai Reflex Index",
          "slug": "rai",
          "ticker": "RAI"
        },
        {
          "name": "FONSmartChain",
          "slug": "fonsmartchain",
          "ticker": "FON"
        },
        {
          "name": "JumpToken",
          "slug": "jumptoken",
          "ticker": "JMPT"
        },
        {
          "name": "Spore",
          "slug": "spore",
          "ticker": "SPORE"
        },
        {
          "name": "Metaverse Face",
          "slug": "metaverse-face",
          "ticker": "MEFA"
        },
        {
          "name": "Meme Alliance",
          "slug": "meme-alliance",
          "ticker": "MMA"
        },
        {
          "name": "nDEX",
          "slug": "ndex",
          "ticker": "NDEX"
        },
        {
          "name": "Gamium [on Ethereum]",
          "slug": "gamium",
          "ticker": "GMM"
        },
        {
          "name": "Indorse Token",
          "slug": "indorse-token",
          "ticker": "IND"
        },
        {
          "name": "Basic Attention Token",
          "slug": "basic-attention-token",
          "ticker": "BAT"
        },
        {
          "name": "SENSO",
          "slug": "senso",
          "ticker": "SENSO"
        },
        {
          "name": "Eesee",
          "slug": "eesee",
          "ticker": "ESE"
        },
        {
          "name": "Aditus",
          "slug": "aditus",
          "ticker": "ADI"
        },
        {
          "name": "Maincoin",
          "slug": "maincoin",
          "ticker": "MNC"
        },
        {
          "name": "Bulwark",
          "slug": "bulwark",
          "ticker": "BWK"
        },
        {
          "name": "Bitpanda",
          "slug": "bitpanda-ecosystem-token",
          "ticker": "BEST"
        },
        {
          "name": "DEXGame",
          "slug": "dexgame",
          "ticker": "DXGM"
        },
        {
          "name": "Trenches AI",
          "slug": "trenches-ai",
          "ticker": "TRENCHAI"
        },
        {
          "name": "Mystery",
          "slug": "mystery-token",
          "ticker": "MYSTERY"
        },
        {
          "name": "ThingsOperatingSystem",
          "slug": "thingsoperatingsystem",
          "ticker": "TOS"
        },
        {
          "name": "TokenClub",
          "slug": "tokenclub",
          "ticker": "TCT"
        },
        {
          "name": "WePower",
          "slug": "wepower",
          "ticker": "WPR"
        },
        {
          "name": "Swarm",
          "slug": "swarm-fund",
          "ticker": "SWM"
        },
        {
          "name": "Quant",
          "slug": "quant",
          "ticker": "QNT"
        },
        {
          "name": "Stakinglab",
          "slug": "stakinglab",
          "ticker": "LABX"
        },
        {
          "name": "B20",
          "slug": "b20",
          "ticker": "B20"
        },
        {
          "name": "Balancer [on Polygon]",
          "slug": "p-balancer",
          "ticker": "BAL"
        },
        {
          "name": "TENT",
          "slug": "tent",
          "ticker": "TENT"
        },
        {
          "name": "SushiSwap: xSUSHI Token",
          "slug": "xsushi",
          "ticker": "xSUSHI"
        },
        {
          "name": "Toko Token",
          "slug": "tokocrypto",
          "ticker": "TKO"
        },
        {
          "name": "Streamity",
          "slug": "streamity",
          "ticker": "STM"
        },
        {
          "name": "Persistence",
          "slug": "persistence",
          "ticker": "XPRT"
        },
        {
          "name": "LayerK",
          "slug": "layerk",
          "ticker": "LYK"
        },
        {
          "name": "Dentacoin",
          "slug": "dentacoin",
          "ticker": "DCN"
        },
        {
          "name": "COMBO",
          "slug": "combo-network",
          "ticker": "COMBO"
        },
        {
          "name": "IoT Chain",
          "slug": "iot-chain",
          "ticker": "ITC"
        },
        {
          "name": "Evadore",
          "slug": "evadore",
          "ticker": "EVA"
        },
        {
          "name": "BORA",
          "slug": "bora",
          "ticker": "BORA"
        },
        {
          "name": "Rari Governance Token",
          "slug": "rari-governance-token",
          "ticker": "RGT"
        },
        {
          "name": "Atomic Wallet Coin",
          "slug": "atomic-wallet-coin",
          "ticker": "AWC"
        },
        {
          "name": "TokenStars",
          "slug": "tokenstars",
          "ticker": "TEAM"
        },
        {
          "name": "Neblio",
          "slug": "neblio",
          "ticker": "NEBL"
        },
        {
          "name": "Master Contract Token",
          "slug": "master-contract-token",
          "ticker": "MCT"
        },
        {
          "name": "Balancer: ETH/WBTC 50/50",
          "slug": "balancer_eth_wbtc_5050_lp",
          "ticker": "BPT-ETH/WBTC-50/50"
        },
        {
          "name": "PlatonCoin",
          "slug": "platoncoin",
          "ticker": "PLTC"
        },
        {
          "name": "H2O DAO",
          "slug": "h2o-dao",
          "ticker": "H2O"
        },
        {
          "name": "Wrapped BNB",
          "slug": "wbnb",
          "ticker": "WBNB"
        },
        {
          "name": "Incodium",
          "slug": "incodium",
          "ticker": "INCO"
        },
        {
          "name": "HelloGold",
          "slug": "hellogold",
          "ticker": "HGT"
        },
        {
          "name": "Muzika",
          "slug": "muzika",
          "ticker": "MZK"
        },
        {
          "name": "Damex Token",
          "slug": "damex-token",
          "ticker": "DAMEX"
        },
        {
          "name": "Tokenomy",
          "slug": "tokenomy",
          "ticker": "TEN"
        },
        {
          "name": "1World",
          "slug": "1world",
          "ticker": "1WO"
        },
        {
          "name": "Hathor",
          "slug": "hathor",
          "ticker": "HTR"
        },
        {
          "name": "BBS Network",
          "slug": "bbs-network",
          "ticker": "BBS"
        },
        {
          "name": "SnowGem",
          "slug": "snowgem",
          "ticker": "XSG"
        },
        {
          "name": "SafeMoon Inu",
          "slug": "safemoon-inu",
          "ticker": "SMI"
        },
        {
          "name": "TRAVA.FINANCE",
          "slug": "trava-finance",
          "ticker": "TRAVA"
        },
        {
          "name": "Light",
          "slug": "lightning",
          "ticker": "LIGHT"
        },
        {
          "name": "Avocado DAO Token",
          "slug": "avocado-dao-token",
          "ticker": "AVG"
        },
        {
          "name": "PikcioChain",
          "slug": "pikciochain",
          "ticker": "PKC"
        },
        {
          "name": "Ergo",
          "slug": "ergo",
          "ticker": "ERG"
        },
        {
          "name": "WiFi Map",
          "slug": "p-wifi-map",
          "ticker": "WIFI"
        },
        {
          "name": "SureRemit",
          "slug": "sureremit",
          "ticker": "RMT"
        },
        {
          "name": "Nestree",
          "slug": "nestree",
          "ticker": "EGG"
        },
        {
          "name": "ValueCyberToken",
          "slug": "valuecybertoken",
          "ticker": "VCT"
        },
        {
          "name": "FansTime",
          "slug": "fanstime",
          "ticker": "FTI"
        },
        {
          "name": "UniLend",
          "slug": "unilend",
          "ticker": "UFT"
        },
        {
          "name": "Traceability Chain",
          "slug": "traceability-chain",
          "ticker": "TAC"
        },
        {
          "name": "NFPrompt",
          "slug": "bnb-nfprompt",
          "ticker": "NFP"
        },
        {
          "name": "GMCoin",
          "slug": "gmcoin",
          "ticker": "GMCOIN"
        },
        {
          "name": "Goose Finance",
          "slug": "goose-finance",
          "ticker": "EGG"
        },
        {
          "name": "KYVE Network",
          "slug": "kyve-network",
          "ticker": "KYVE"
        },
        {
          "name": "NFTX",
          "slug": "nftx",
          "ticker": "NFTX"
        },
        {
          "name": "TRVL",
          "slug": "trvl",
          "ticker": "TRVL"
        },
        {
          "name": "Venus BNB",
          "slug": "venus-bnb",
          "ticker": "vBNB"
        },
        {
          "name": "Boolberry",
          "slug": "boolberry",
          "ticker": "BBR"
        },
        {
          "name": "Achain",
          "slug": "achain",
          "ticker": "ACT"
        },
        {
          "name": "Cobak Token",
          "slug": "cobak-token",
          "ticker": "CBK"
        },
        {
          "name": "DecentBet",
          "slug": "decent-bet",
          "ticker": "DBET"
        },
        {
          "name": "IQeon",
          "slug": "iqeon",
          "ticker": "IQN"
        },
        {
          "name": "Agora",
          "slug": "agora",
          "ticker": "VOTE"
        },
        {
          "name": "eCash",
          "slug": "ecash",
          "ticker": "XEC"
        },
        {
          "name": "Pawthereum",
          "slug": "pawthereum",
          "ticker": "PAWTH"
        },
        {
          "name": "Catman",
          "slug": "catman",
          "ticker": "CATMAN"
        },
        {
          "name": "Mey Network",
          "slug": "mey-network",
          "ticker": "MEY"
        },
        {
          "name": "FintruX Network",
          "slug": "fintrux-network",
          "ticker": "FTX"
        },
        {
          "name": "Banca",
          "slug": "banca",
          "ticker": "BANCA"
        },
        {
          "name": "UChain",
          "slug": "uchain",
          "ticker": "UCN"
        },
        {
          "name": "Artyfact",
          "slug": "artyfact",
          "ticker": "ARTY"
        },
        {
          "name": "The Tokenized Bitcoin",
          "slug": "the-tokenized-bitcoin",
          "ticker": "imBTC"
        },
        {
          "name": "Mithril",
          "slug": "mithril",
          "ticker": "MITH"
        },
        {
          "name": "WHY",
          "slug": "why",
          "ticker": "WHY"
        },
        {
          "name": "CumRocket",
          "slug": "cumrocket",
          "ticker": "CUMMIES"
        },
        {
          "name": "Taraxa",
          "slug": "taraxa",
          "ticker": "TARA"
        },
        {
          "name": "Unido EP",
          "slug": "unido",
          "ticker": "UDO"
        },
        {
          "name": "Conflux Network",
          "slug": "conflux-network",
          "ticker": "CFX"
        },
        {
          "name": "BEFE",
          "slug": "befe",
          "ticker": "BEFE"
        },
        {
          "name": "LayerZero [on Ethereum]",
          "slug": "layerzero",
          "ticker": "ZRO"
        },
        {
          "name": "Jupiter",
          "slug": "jupiter-ag",
          "ticker": "JUP"
        },
        {
          "name": "Staika",
          "slug": "staika",
          "ticker": "STIK"
        },
        {
          "name": "Blast",
          "slug": "blast",
          "ticker": "BLAST"
        },
        {
          "name": "YOYOW",
          "slug": "yoyow",
          "ticker": "YOYOW"
        },
        {
          "name": "Bitnation",
          "slug": "bitnation",
          "ticker": "XPAT"
        },
        {
          "name": "Thrive",
          "slug": "thrive-token",
          "ticker": "THRT"
        },
        {
          "name": "Phoenix",
          "slug": "phoenix-global-new",
          "ticker": "PHB"
        },
        {
          "name": "NewsToken",
          "slug": "newstoken",
          "ticker": "NEWOS"
        },
        {
          "name": "ZrCoin",
          "slug": "zrcoin",
          "ticker": "ZRC"
        },
        {
          "name": "IoTeX",
          "slug": "iotex",
          "ticker": "IOTX"
        },
        {
          "name": "Ultiverse",
          "slug": "ultiverse",
          "ticker": "ULTI"
        },
        {
          "name": "One Cash",
          "slug": "one-cash",
          "ticker": "ONC"
        },
        {
          "name": "Midas Dollar",
          "slug": "midas-dollar",
          "ticker": "MDO"
        },
        {
          "name": "Kattana",
          "slug": "kattana",
          "ticker": "KTN"
        },
        {
          "name": "Name Change Token",
          "slug": "name-change-token",
          "ticker": "NCT"
        },
        {
          "name": "Ren: renFIL Token",
          "slug": "renfil",
          "ticker": "RENFIL"
        },
        {
          "name": "Uniswap V2: WBTC",
          "slug": "uniswap_wbtc_eth_lp",
          "ticker": "UNI-V2 WBTC/ETH LP"
        },
        {
          "name": "Hoppy",
          "slug": "hoppy-coin",
          "ticker": "HOPPY"
        },
        {
          "name": "Tune.FM",
          "slug": "tune-fm",
          "ticker": "JAM"
        },
        {
          "name": "Blox",
          "slug": "blox",
          "ticker": "CDT"
        },
        {
          "name": "Laqira Protocol",
          "slug": "laqira-protocol",
          "ticker": "LQR"
        },
        {
          "name": "Enigma",
          "slug": "enigma",
          "ticker": "ENG"
        },
        {
          "name": "CafeSwap Token",
          "slug": "cafeswap-token",
          "ticker": "BREW"
        },
        {
          "name": "Custody Token",
          "slug": "custody-token",
          "ticker": "CUST"
        },
        {
          "name": "Structured",
          "slug": "structured",
          "ticker": "STR"
        },
        {
          "name": "Egoras",
          "slug": "egoras",
          "ticker": "EGR"
        },
        {
          "name": "FLEX",
          "slug": "flex",
          "ticker": "FLEX"
        },
        {
          "name": "Huobi BTC",
          "slug": "huobi-btc",
          "ticker": "HBTC"
        },
        {
          "name": "The Unfettered Ecosystem",
          "slug": "the-unfettered",
          "ticker": "SOULS"
        },
        {
          "name": "Bispex",
          "slug": "bispex",
          "ticker": "BPX"
        },
        {
          "name": "Golem",
          "slug": "golem-network-tokens",
          "ticker": "GLM"
        },
        {
          "name": "Moon Tropica",
          "slug": "moon-tropica",
          "ticker": "CAH"
        },
        {
          "name": "Artisse",
          "slug": "catheon-gaming",
          "ticker": "CATHEON"
        },
        {
          "name": "ArdCoin",
          "slug": "ardcoin",
          "ticker": "ARDX"
        },
        {
          "name": "hiDOODLES",
          "slug": "hidoodles",
          "ticker": "HIDOODLES"
        },
        {
          "name": "Attila",
          "slug": "attila",
          "ticker": "ATT"
        },
        {
          "name": "VNX Gold",
          "slug": "vnx-gold",
          "ticker": "VNXAU"
        },
        {
          "name": "MicroVisionChain",
          "slug": "microvisionchain",
          "ticker": "SPACE"
        },
        {
          "name": "Summit",
          "slug": "summit",
          "ticker": "SUMMIT"
        },
        {
          "name": "yOUcash",
          "slug": "youcash",
          "ticker": "YOUC"
        },
        {
          "name": "Vesper",
          "slug": "vesper",
          "ticker": "VSP"
        },
        {
          "name": "DEX.AG",
          "slug": "dex-ag",
          "ticker": "DXG"
        },
        {
          "name": "Ormeus Ecosystem",
          "slug": "ormeus-ecosystem",
          "ticker": "ECO"
        },
        {
          "name": "Mdex",
          "slug": "mdex",
          "ticker": "MDX"
        },
        {
          "name": "Matr1x Fire",
          "slug": "matr1x-fire",
          "ticker": "FIRE"
        },
        {
          "name": "Vinci",
          "slug": "vinci",
          "ticker": "VINCI"
        },
        {
          "name": "VNX",
          "slug": "vnx",
          "ticker": "VNXLU"
        },
        {
          "name": "CAPITAL X CELL",
          "slug": "capital-x-cell",
          "ticker": "CXC"
        },
        {
          "name": "suterusu",
          "slug": "suterusu",
          "ticker": "SUTER"
        },
        {
          "name": "Worldwide USD",
          "slug": "worldwide-usd",
          "ticker": "WUSD"
        },
        {
          "name": "UNIT0",
          "slug": "unit0",
          "ticker": "UNIT0"
        },
        {
          "name": "AICell",
          "slug": "aicell",
          "ticker": "AICELL"
        },
        {
          "name": "LocalCryptos ",
          "slug": "localcryptos",
          "ticker": "LCR"
        },
        {
          "name": "Binance USD [on Avalanche]",
          "slug": "a-binance-usd",
          "ticker": "BUSD"
        },
        {
          "name": "EXMR",
          "slug": "exmr",
          "ticker": "EXMR"
        },
        {
          "name": "HOPR",
          "slug": "hopr",
          "ticker": "HOPR"
        },
        {
          "name": "Liquity",
          "slug": "liquity",
          "ticker": "LQTY"
        },
        {
          "name": "DeFiChain",
          "slug": "defichain",
          "ticker": "DFI"
        },
        {
          "name": "ShareToken",
          "slug": "sharetoken",
          "ticker": "SHR"
        },
        {
          "name": "Venus LTC",
          "slug": "venus-ltc",
          "ticker": "vLTC"
        },
        {
          "name": "Safex Token",
          "slug": "safex-token",
          "ticker": "SFT"
        },
        {
          "name": "Blockium",
          "slug": "blockium",
          "ticker": "BOK"
        },
        {
          "name": "Trump Derangement Syndrome (tearsforTDS)",
          "slug": "trump-derangement-syndrome",
          "ticker": "TDS"
        },
        {
          "name": "MultiversX",
          "slug": "elrond-egld",
          "ticker": "EGLD"
        },
        {
          "name": "Chi Gastoken",
          "slug": "chi-gastoken",
          "ticker": "CHI"
        },
        {
          "name": "LIFE",
          "slug": "life",
          "ticker": "LIFE"
        },
        {
          "name": "Fortem Capital",
          "slug": "fortem-capital",
          "ticker": "FCQ"
        },
        {
          "name": "BTCPay",
          "slug": "btcpay",
          "ticker": "BCP"
        },
        {
          "name": "Nord Finance",
          "slug": "nord-finance",
          "ticker": "NORD"
        },
        {
          "name": "Aphelion",
          "slug": "aphelion",
          "ticker": "APH"
        },
        {
          "name": "Unicly Mystic Axies Collection",
          "slug": "unicly-mystic-axies-collection",
          "ticker": "UAXIE"
        },
        {
          "name": "VNX Exchange",
          "slug": "vnx-exchange",
          "ticker": "VNXLU"
        },
        {
          "name": "Fusion",
          "slug": "fusion",
          "ticker": "FSN"
        },
        {
          "name": "ZB Token",
          "slug": "zb",
          "ticker": "ZB"
        },
        {
          "name": "MVL",
          "slug": "mvl",
          "ticker": "MVL"
        },
        {
          "name": "Zebi",
          "slug": "zebi",
          "ticker": "ZCO"
        },
        {
          "name": "Dai [on BNB]",
          "slug": "bnb-multi-collateral-dai",
          "ticker": "DAI"
        },
        {
          "name": "Filenet",
          "slug": "filenet",
          "ticker": "FN"
        },
        {
          "name": "Star Atlas",
          "slug": "star-atlas",
          "ticker": "ATLAS"
        },
        {
          "name": "Medium",
          "slug": "medium",
          "ticker": "MDM"
        },
        {
          "name": "Lendf",
          "slug": "lendf",
          "ticker": "LFM"
        },
        {
          "name": "Sablier",
          "slug": "sablier",
          "ticker": "SAB"
        },
        {
          "name": "Bao Finance",
          "slug": "bao-finance",
          "ticker": "BAO"
        },
        {
          "name": "PepeFork",
          "slug": "pepefork",
          "ticker": "PORK"
        },
        {
          "name": "Rally",
          "slug": "rally",
          "ticker": "RLY"
        },
        {
          "name": "Venus SXP",
          "slug": "vsxp",
          "ticker": "vSXP"
        },
        {
          "name": "AC Milan Fan Token",
          "slug": "ac-milan-fan-token",
          "ticker": "ACM"
        },
        {
          "name": "Uniswap [on Polygon]",
          "slug": "p-uniswap",
          "ticker": "UNI"
        },
        {
          "name": "Aston Martin Cognizant Fan Token",
          "slug": "aston-martin-cognizant-fan-token",
          "ticker": "AM"
        },
        {
          "name": "CoTrader",
          "slug": "cotrader",
          "ticker": "COT"
        },
        {
          "name": "Star Atlas DAO",
          "slug": "star-atlas-polis",
          "ticker": "POLIS"
        },
        {
          "name": "Mogu",
          "slug": "mogu",
          "ticker": "MOGX"
        },
        {
          "name": "Portis",
          "slug": "portis",
          "ticker": "PRT"
        },
        {
          "name": "Zus",
          "slug": "0chain",
          "ticker": "ZCN"
        },
        {
          "name": "Alchemix",
          "slug": "alchemix",
          "ticker": "ALCX"
        },
        {
          "name": "KickToken",
          "slug": "kickico",
          "ticker": "KICK"
        },
        {
          "name": "CONUN",
          "slug": "conun",
          "ticker": "CON"
        },
        {
          "name": "Xdef Finance",
          "slug": "xdef-finance",
          "ticker": "XDEF2"
        },
        {
          "name": "Cratos",
          "slug": "cratos",
          "ticker": "CRTS"
        },
        {
          "name": "NPCoin",
          "slug": "npcoin",
          "ticker": "NPC"
        },
        {
          "name": "BetProtocol",
          "slug": "betprotocol",
          "ticker": "BEPRO"
        },
        {
          "name": "Orchid",
          "slug": "orchid",
          "ticker": "OXT"
        },
        {
          "name": "RSK",
          "slug": "rsk",
          "ticker": "RSK"
        },
        {
          "name": "Unistake",
          "slug": "unistake",
          "ticker": "UNISTAKE"
        },
        {
          "name": "Infinitecoin",
          "slug": "infinitecoin",
          "ticker": "IFC"
        },
        {
          "name": "X-CASH",
          "slug": "x-cash",
          "ticker": "XCASH"
        },
        {
          "name": "Davion",
          "slug": "davion",
          "ticker": "DAVP"
        },
        {
          "name": "Zynecoin",
          "slug": "zynecoin",
          "ticker": "ZYN"
        },
        {
          "name": "Royale Finance",
          "slug": "royale-finance",
          "ticker": "ROYA"
        },
        {
          "name": "Enzo",
          "slug": "enzo",
          "ticker": "NZO"
        },
        {
          "name": "ZBG Token",
          "slug": "zbg-token",
          "ticker": "ZT"
        },
        {
          "name": "Sparkswap",
          "slug": "sparkswap",
          "ticker": "SPR"
        },
        {
          "name": "Joystream",
          "slug": "joystream",
          "ticker": "JOY"
        },
        {
          "name": "dForce USDx",
          "slug": "usdx-stablecoin",
          "ticker": "USDX"
        },
        {
          "name": "GMX [on Arbitrum]",
          "slug": "arb-gmx",
          "ticker": "GMX"
        },
        {
          "name": "San Index",
          "slug": "sanx",
          "ticker": "SANX"
        },
        {
          "name": "Tachyon Protocol",
          "slug": "tachyon-protocol",
          "ticker": "IPX"
        },
        {
          "name": "ZetaChain",
          "slug": "zetachain",
          "ticker": "ZETA"
        },
        {
          "name": "ROAD",
          "slug": "road",
          "ticker": "ROAD"
        },
        {
          "name": "Multiplier",
          "slug": "bmultiplier",
          "ticker": "BMXX"
        },
        {
          "name": "Tokenlon Network Token",
          "slug": "tokenlon-network-token",
          "ticker": "LON"
        },
        {
          "name": "Complete New Commerce Chain",
          "slug": "complete-new-commerce-chain",
          "ticker": "CNCC"
        },
        {
          "name": "SafeBlast",
          "slug": "safeblast",
          "ticker": "BLAST"
        },
        {
          "name": "Voxel X Network",
          "slug": "voxel-x-network",
          "ticker": "VXL"
        },
        {
          "name": "MAX Exchange Token",
          "slug": "max-exchange-token",
          "ticker": "MAX"
        },
        {
          "name": "AntiMatter",
          "slug": "antimatter",
          "ticker": "MATTER"
        },
        {
          "name": "Audius",
          "slug": "audius",
          "ticker": "AUDIO"
        },
        {
          "name": "One Share",
          "slug": "one-share",
          "ticker": "ONS"
        },
        {
          "name": "Ethereum Push Notification Service",
          "slug": "epns",
          "ticker": "PUSH"
        },
        {
          "name": "The Transfer Token",
          "slug": "the-transfer-token",
          "ticker": "TTT"
        },
        {
          "name": "Mirrored ProShares VIX",
          "slug": "mirrored-proshares-vix-short-term-futures-etf",
          "ticker": "mVIXY"
        },
        {
          "name": "Mintlayer",
          "slug": "mintlayer",
          "ticker": "ML"
        },
        {
          "name": "Hiblocks",
          "slug": "hiblocks",
          "ticker": "HIBS"
        },
        {
          "name": "Playermon",
          "slug": "playermon",
          "ticker": "PYM"
        },
        {
          "name": "KANGO",
          "slug": "kango",
          "ticker": "KANGO"
        },
        {
          "name": "Lybra Finance",
          "slug": "lybra-finance",
          "ticker": "LBR"
        },
        {
          "name": "NOW Token",
          "slug": "now-token",
          "ticker": "NOW"
        },
        {
          "name": "WELL",
          "slug": "well-token",
          "ticker": "WELL"
        },
        {
          "name": "Orient Walt",
          "slug": "orient-walt",
          "ticker": "HTDF"
        },
        {
          "name": "Pizza",
          "slug": "pizza",
          "ticker": "PIZZA"
        },
        {
          "name": "Infinitar",
          "slug": "infinitar",
          "ticker": "IGT"
        },
        {
          "name": "Wrapped Evmos",
          "slug": "wrapped-evmos",
          "ticker": "WEVMOS"
        },
        {
          "name": "Blockasset",
          "slug": "blockasset",
          "ticker": "BLOCK"
        },
        {
          "name": "Pomerium",
          "slug": "pomerium-ecosystem-token",
          "ticker": "PMG"
        },
        {
          "name": "Dymension",
          "slug": "dymension",
          "ticker": "DYM"
        },
        {
          "name": "3X Short Bitcoin Token",
          "slug": "3x-short-bitcoin-token",
          "ticker": "BEAR"
        },
        {
          "name": "Bounce Token",
          "slug": "bounce",
          "ticker": "AUCTION"
        },
        {
          "name": "Ethereum Name Service",
          "slug": "ethereum-name-service",
          "ticker": "ENS"
        },
        {
          "name": "Visor.Finance",
          "slug": "visor-finance",
          "ticker": "VISR"
        },
        {
          "name": "3X Long Bitcoin Token",
          "slug": "3x-long-bitcoin-token",
          "ticker": "BULL"
        },
        {
          "name": "GoCrypto Token",
          "slug": "gocrypto-token",
          "ticker": "GOC"
        },
        {
          "name": "Venus LINK",
          "slug": "venus-link",
          "ticker": "vLINK"
        },
        {
          "name": "InsurAce",
          "slug": "insurace",
          "ticker": "INSUR"
        },
        {
          "name": "TROY",
          "slug": "troy",
          "ticker": "TROY"
        },
        {
          "name": "Centric Swap",
          "slug": "centric-swap",
          "ticker": "CNS"
        },
        {
          "name": "Tether Gold",
          "slug": "tether-gold",
          "ticker": "XAUt"
        },
        {
          "name": "Kaizen Finance",
          "slug": "kaizen-finance",
          "ticker": "KZEN"
        },
        {
          "name": "PolkaPets",
          "slug": "polkapets",
          "ticker": "PETS"
        },
        {
          "name": "DATA",
          "slug": "data",
          "ticker": "DTA"
        },
        {
          "name": "DeRace",
          "slug": "derace",
          "ticker": "ZERC"
        },
        {
          "name": "MATH",
          "slug": "math",
          "ticker": "MATH"
        },
        {
          "name": "Keystone of Opportunity & Knowledge",
          "slug": "keystone-of-opportunity-knowledge",
          "ticker": "KOK"
        },
        {
          "name": "WazirX",
          "slug": "wazirx",
          "ticker": "WRX"
        },
        {
          "name": "ARCS",
          "slug": "arcs",
          "ticker": "ARX"
        },
        {
          "name": "Big Data Protocol",
          "slug": "big-data-protocol",
          "ticker": "BDP"
        },
        {
          "name": "Idle",
          "slug": "idle",
          "ticker": "IDLE"
        },
        {
          "name": "Lock",
          "slug": "lock",
          "ticker": "LOCK"
        },
        {
          "name": "ScPrime",
          "slug": "scprime",
          "ticker": "SCP"
        },
        {
          "name": "Torque",
          "slug": "torque",
          "ticker": "TRQ"
        },
        {
          "name": "Inverse Finance",
          "slug": "inverse-finance",
          "ticker": "INV"
        },
        {
          "name": "Spartan Protocol",
          "slug": "spartan-protocol",
          "ticker": "SPARTA"
        },
        {
          "name": "NFT Index",
          "slug": "nft-index",
          "ticker": "NFTI"
        },
        {
          "name": "USDX [Kava]",
          "slug": "usdx-kava",
          "ticker": "USDX"
        },
        {
          "name": "K21",
          "slug": "k21",
          "ticker": "K21"
        },
        {
          "name": "Hydro",
          "slug": "hydro",
          "ticker": "HYDRO"
        },
        {
          "name": "Orbit Chain",
          "slug": "orbit-chain",
          "ticker": "ORC"
        },
        {
          "name": "pTokens",
          "slug": "ptokens",
          "ticker": "PTK"
        },
        {
          "name": "Hoge Finance",
          "slug": "hoge-finance",
          "ticker": "HOGE"
        },
        {
          "name": "Thorstarter",
          "slug": "thorstarter",
          "ticker": "XRUNE"
        },
        {
          "name": "HyperDAO",
          "slug": "hyperdao",
          "ticker": "HDAO"
        },
        {
          "name": "Joint Ventures",
          "slug": "joint-ventures",
          "ticker": "JOINT"
        },
        {
          "name": "apM Coin",
          "slug": "apm-coin",
          "ticker": "APM"
        },
        {
          "name": "Mars",
          "slug": "mars",
          "ticker": "Mars"
        },
        {
          "name": "Plastiks",
          "slug": "plastiks",
          "ticker": "PLASTIK"
        },
        {
          "name": "Eden",
          "slug": "eden-network",
          "ticker": "EDEN"
        },
        {
          "name": "Sakai Vault",
          "slug": "sakai-vault",
          "ticker": "SAKAI"
        },
        {
          "name": "Handshake",
          "slug": "handshake",
          "ticker": "HNS"
        },
        {
          "name": "Synthetify",
          "slug": "synthetify",
          "ticker": "SNY"
        },
        {
          "name": "Perth Mint Gold Token",
          "slug": "perth-mint-gold-token",
          "ticker": "PMGT"
        },
        {
          "name": "Validity",
          "slug": "validity",
          "ticker": "VAL"
        },
        {
          "name": "Keep3rV1",
          "slug": "keep3rv1",
          "ticker": "KP3R"
        },
        {
          "name": "Brett",
          "slug": "based-brett",
          "ticker": "BRETT"
        },
        {
          "name": "Alien Worlds",
          "slug": "alien-worlds",
          "ticker": "TLM"
        },
        {
          "name": "GravityCoin",
          "slug": "gravitycoin",
          "ticker": "GXX"
        },
        {
          "name": "Tokemak",
          "slug": "tokemak",
          "ticker": "TOKE"
        },
        {
          "name": "Incognito",
          "slug": "incognito",
          "ticker": "PRV"
        },
        {
          "name": "AstroTools",
          "slug": "astrotools",
          "ticker": "ASTRO"
        },
        {
          "name": "ChainLink [on Polygon]",
          "slug": "p-chainlink",
          "ticker": "LINK"
        },
        {
          "name": "Farcana",
          "slug": "p-farcana",
          "ticker": "FAR"
        },
        {
          "name": "Mirrored Amazon",
          "slug": "mirrored-amazon",
          "ticker": "mAMZN"
        },
        {
          "name": "Coin Artist",
          "slug": "coin-artist",
          "ticker": "COIN"
        },
        {
          "name": "Standard Tokenization Protocol",
          "slug": "stpt",
          "ticker": "STPT"
        },
        {
          "name": "Observer",
          "slug": "observer",
          "ticker": "OBSR"
        },
        {
          "name": "sXAU",
          "slug": "sxau",
          "ticker": "sXAU"
        },
        {
          "name": "Cred",
          "slug": "libra-credit",
          "ticker": "LBA"
        },
        {
          "name": "Elixxir",
          "slug": "elixxir",
          "ticker": "XX"
        },
        {
          "name": "Unizen",
          "slug": "unizen",
          "ticker": "ZCX"
        },
        {
          "name": "AXEL",
          "slug": "axel",
          "ticker": "AXEL"
        },
        {
          "name": "Oxbull.tech",
          "slug": "oxbull-tech",
          "ticker": "OXB"
        },
        {
          "name": "Rupiah Token",
          "slug": "rupiah-token",
          "ticker": "IDRT"
        },
        {
          "name": "Unifi Protocol DAO",
          "slug": "unifi-protocol-dao",
          "ticker": "UNFI"
        },
        {
          "name": "XCredit",
          "slug": "xcredit",
          "ticker": "XFYI"
        },
        {
          "name": "Aurox",
          "slug": "urus",
          "ticker": "URUS"
        },
        {
          "name": "Exeedme",
          "slug": "exeedme",
          "ticker": "XED"
        },
        {
          "name": "WeOwn",
          "slug": "we-own",
          "ticker": "CHX"
        },
        {
          "name": "Hegic [on Arbitrum]",
          "slug": "arb-hegic",
          "ticker": "HEGIC"
        },
        {
          "name": "TopBidder",
          "slug": "topbidder",
          "ticker": "BID"
        },
        {
          "name": "merkleX",
          "slug": "merklex",
          "ticker": "MKX"
        },
        {
          "name": "ParallelCoin",
          "slug": "parallelcoin",
          "ticker": "DUO"
        },
        {
          "name": "Shadows",
          "slug": "shadows",
          "ticker": "DOWS"
        },
        {
          "name": "SparkPoint",
          "slug": "sparkpoint",
          "ticker": "SRK"
        },
        {
          "name": "Rake Coin",
          "slug": "rake-coin",
          "ticker": "RAKE"
        },
        {
          "name": "Tenset",
          "slug": "tenset",
          "ticker": "10SET"
        },
        {
          "name": "Nexa",
          "slug": "nexa",
          "ticker": "NEXA"
        },
        {
          "name": "DigiByte",
          "slug": "digibyte",
          "ticker": "DGB"
        },
        {
          "name": "Ethereum Yield",
          "slug": "ethereum-yield",
          "ticker": "ETHY"
        },
        {
          "name": "Project Coin",
          "slug": "project-coin",
          "ticker": "PRJ"
        },
        {
          "name": "Bondly",
          "slug": "bondly",
          "ticker": "BONDLY"
        },
        {
          "name": "City Tycoon Games",
          "slug": "city-tycoon-games",
          "ticker": "CTG"
        },
        {
          "name": "Kylin",
          "slug": "kylin",
          "ticker": "KYL"
        },
        {
          "name": "Genesis Shards",
          "slug": "genesis-shards",
          "ticker": "GS"
        },
        {
          "name": "Bancor",
          "slug": "bancor",
          "ticker": "BNT"
        },
        {
          "name": "WOOF",
          "slug": "woof",
          "ticker": "WOOF"
        },
        {
          "name": "SYNC Network",
          "slug": "sync-network",
          "ticker": "SYNC"
        },
        {
          "name": "Falcon Project",
          "slug": "falcon-project",
          "ticker": "FNT"
        },
        {
          "name": "Global Coin Research",
          "slug": "global-coin-research",
          "ticker": "GCR"
        },
        {
          "name": "Vabble",
          "slug": "vabble",
          "ticker": "VAB"
        },
        {
          "name": "EasyFi",
          "slug": "easyfi",
          "ticker": "EASY"
        },
        {
          "name": "Ocean Protocol",
          "slug": "ocean-protocol",
          "ticker": "OCEAN"
        },
        {
          "name": "TRI SIGMA",
          "slug": "tri-sigma",
          "ticker": "TRISIG"
        },
        {
          "name": "1inch",
          "slug": "1inch",
          "ticker": "1INCH"
        },
        {
          "name": "MMOCoin",
          "slug": "mmocoin",
          "ticker": "MMO"
        },
        {
          "name": "VNX Swiss Franc",
          "slug": "vnx-swiss-franc",
          "ticker": "VCHF"
        },
        {
          "name": "PARSIQ",
          "slug": "parsiq",
          "ticker": "PRQ"
        },
        {
          "name": "Paypolitan Token",
          "slug": "paypolitan-token",
          "ticker": "EPAN"
        },
        {
          "name": "Paint",
          "slug": "paint",
          "ticker": "PAINT"
        },
        {
          "name": "BSCPAD",
          "slug": "bscpad",
          "ticker": "BSCPAD"
        },
        {
          "name": "Ethereum Stake",
          "slug": "ethereum-stake",
          "ticker": "ETHYS"
        },
        {
          "name": "Crust Shadow",
          "slug": "crust-shadow",
          "ticker": "CSM"
        },
        {
          "name": "AXPR",
          "slug": "axpr-token",
          "ticker": "AXPR"
        },
        {
          "name": "Receive Access Ecosystem",
          "slug": "receive-access-ecosystem",
          "ticker": "RAE"
        },
        {
          "name": "ExNetwork Token",
          "slug": "exnetwork-token",
          "ticker": "EXNT"
        },
        {
          "name": "Iconiq Lab",
          "slug": "iconiq-lab-token",
          "ticker": "ICNQ"
        },
        {
          "name": "Trustlines Network Token",
          "slug": "trustlines",
          "ticker": "TLN"
        },
        {
          "name": "DEOR",
          "slug": "deor",
          "ticker": "DEOR"
        },
        {
          "name": "Acute Angle Cloud",
          "slug": "acute-angle-cloud",
          "ticker": "AAC"
        },
        {
          "name": "Odin Protocol",
          "slug": "odin-protocol",
          "ticker": "ODIN"
        },
        {
          "name": "Kwenta",
          "slug": "o-kwenta",
          "ticker": "KWENTA"
        },
        {
          "name": "Polkalokr",
          "slug": "polkalokr",
          "ticker": "LKR"
        },
        {
          "name": "Huobi Pool Token",
          "slug": "huobi-pool-token",
          "ticker": "HPT"
        },
        {
          "name": "CasperLabs",
          "slug": "casperlabs",
          "ticker": "CLX"
        },
        {
          "name": "Yield",
          "slug": "yield",
          "ticker": "YLD"
        },
        {
          "name": "StakedZEN",
          "slug": "stakedzen",
          "ticker": "STZEN"
        },
        {
          "name": "Uquid Coin",
          "slug": "uquid_coin",
          "ticker": "UQC"
        },
        {
          "name": "DEAPcoin",
          "slug": "deapcoin",
          "ticker": "DEP"
        },
        {
          "name": "JUST",
          "slug": "just",
          "ticker": "JST"
        },
        {
          "name": "KardiaChain",
          "slug": "kardiachain",
          "ticker": "KAI"
        },
        {
          "name": "Tribal Finance",
          "slug": "tribal-token",
          "ticker": "TRIBL"
        },
        {
          "name": "Pino",
          "slug": "pino",
          "ticker": "PINO"
        },
        {
          "name": "PCHAIN",
          "slug": "pchain",
          "ticker": "PI"
        },
        {
          "name": "ECOMI",
          "slug": "ecomi-new",
          "ticker": "OMI"
        },
        {
          "name": "Seele-N",
          "slug": "seele",
          "ticker": "SEELE"
        },
        {
          "name": "Revain",
          "slug": "revain",
          "ticker": "REV"
        },
        {
          "name": "Pundi X[new]",
          "slug": "pundix-new",
          "ticker": "PUNDIX"
        },
        {
          "name": "Minty Art",
          "slug": "minty-art",
          "ticker": "MINTY"
        },
        {
          "name": "Dog (Bitcoin)",
          "slug": "dog-go-to-the-moon-rune",
          "ticker": "DOG"
        },
        {
          "name": "SURF Finance",
          "slug": "surf",
          "ticker": "SURF"
        },
        {
          "name": "Paris Saint-Germain Fan Token",
          "slug": "paris-saint-germain-fan-token",
          "ticker": "PSG"
        },
        {
          "name": "I/O Coin",
          "slug": "iocoin",
          "ticker": "IOC"
        },
        {
          "name": "Kishu Inu",
          "slug": "kishu-inu",
          "ticker": "KISHU"
        },
        {
          "name": "Juventus Fan Toke",
          "slug": "juventus-fan-token",
          "ticker": "JUV"
        },
        {
          "name": "Switch",
          "slug": "switch",
          "ticker": "ESH"
        },
        {
          "name": "Heroes of Mavia",
          "slug": "heroes-of-mavia",
          "ticker": "MAVIA"
        },
        {
          "name": "Flexacoin",
          "slug": "flexacoin",
          "ticker": "FXC"
        },
        {
          "name": "Compound Dai",
          "slug": "compound-dai",
          "ticker": "CDAI"
        },
        {
          "name": "Compound Ether",
          "slug": "compound-ether",
          "ticker": "CETH"
        },
        {
          "name": "THORChain",
          "slug": "thorchain",
          "ticker": "RUNE"
        },
        {
          "name": "Bifrost",
          "slug": "bifrost",
          "ticker": "BFC"
        },
        {
          "name": "Gold Poker",
          "slug": "gold-poker",
          "ticker": "GPKR"
        },
        {
          "name": "Ruler Protocol",
          "slug": "ruler-protocol",
          "ticker": "RULER"
        },
        {
          "name": "Totem",
          "slug": "totem",
          "ticker": "TOTEM"
        },
        {
          "name": "YVS.Finance",
          "slug": "yvs-finance",
          "ticker": "YVS"
        },
        {
          "name": "Non-Playable Coin",
          "slug": "non-playable-coin",
          "ticker": "NPC"
        },
        {
          "name": "ANDY",
          "slug": "boysclubandy",
          "ticker": "ANDY"
        },
        {
          "name": "hiENS3",
          "slug": "hiens3",
          "ticker": "HIENS3"
        },
        {
          "name": "Constellation",
          "slug": "constellation",
          "ticker": "DAG"
        },
        {
          "name": "Ethereum Classic",
          "slug": "ethereum-classic",
          "ticker": "ETC"
        },
        {
          "name": "Aeron",
          "slug": "aeron",
          "ticker": "ARNX"
        },
        {
          "name": "FinNexus",
          "slug": "finnexus",
          "ticker": "FNX"
        },
        {
          "name": "Aave [on Optimism]",
          "slug": "o-aave",
          "ticker": "AAVE"
        },
        {
          "name": "Yield Guild Games",
          "slug": "yield-guild-games",
          "ticker": "YGG"
        },
        {
          "name": "Verse",
          "slug": "verse-token",
          "ticker": "VERSE"
        },
        {
          "name": "Peanut",
          "slug": "peanut",
          "ticker": "NUX"
        },
        {
          "name": "renBTC",
          "slug": "renbtc",
          "ticker": "RENBTC"
        },
        {
          "name": "Ampleforth Governance Token",
          "slug": "ampleforth-governance-token",
          "ticker": "FORTH"
        },
        {
          "name": "Olympus",
          "slug": "olympus",
          "ticker": "OHM"
        },
        {
          "name": "StormX",
          "slug": "stormx",
          "ticker": "STMX"
        },
        {
          "name": "LOCGame",
          "slug": "locgame",
          "ticker": "LOCG"
        },
        {
          "name": "Boson Protocol",
          "slug": "boson-protocol",
          "ticker": "BOSON"
        },
        {
          "name": "RING X PLATFORM",
          "slug": "ring-x-platform",
          "ticker": "RINGX"
        },
        {
          "name": "HEX",
          "slug": "hex",
          "ticker": "HEX"
        },
        {
          "name": "KINE",
          "slug": "kine",
          "ticker": "KINE"
        },
        {
          "name": "WaykiChain Governance Coin",
          "slug": "waykichain-governance-coin",
          "ticker": "WGRT"
        },
        {
          "name": "All Sports",
          "slug": "all-sports",
          "ticker": "SOC"
        },
        {
          "name": "Kyber Network Crystal",
          "slug": "kyber-network",
          "ticker": "KNC"
        },
        {
          "name": "CEREAL",
          "slug": "dodreamchain",
          "ticker": "CEP"
        },
        {
          "name": "Morpheus Network",
          "slug": "morpheus-network",
          "ticker": "MNW"
        },
        {
          "name": "Whole Network",
          "slug": "whole-network",
          "ticker": "NODE"
        },
        {
          "name": "LCX",
          "slug": "lcx",
          "ticker": "LCX"
        },
        {
          "name": "Treecle",
          "slug": "treecle",
          "ticker": "TRCL"
        },
        {
          "name": "Robonomics Web Services",
          "slug": "robonomics-web-services",
          "ticker": "RWS"
        },
        {
          "name": "DEUS Finance",
          "slug": "deus-finance-2",
          "ticker": "DEUS"
        },
        {
          "name": "Baby Bonk",
          "slug": "bnb-baby-bonk-coin",
          "ticker": "BABYBONK"
        },
        {
          "name": "BTSE",
          "slug": "btse",
          "ticker": "BTSE"
        },
        {
          "name": "Crowns",
          "slug": "crowns",
          "ticker": "CWS"
        },
        {
          "name": "Renzo",
          "slug": "renzo",
          "ticker": "REZ"
        },
        {
          "name": "Smoothy",
          "slug": "smoothy",
          "ticker": "SMTY"
        },
        {
          "name": "Xend Finance",
          "slug": "xend-finance",
          "ticker": "XEND"
        },
        {
          "name": "Popsicle Finance",
          "slug": "popsicle-finance",
          "ticker": "ICE"
        },
        {
          "name": "Unicly Hashmasks Collection",
          "slug": "unicly-hashmasks-collection",
          "ticker": "UMASK"
        },
        {
          "name": "BakeryToken",
          "slug": "bakerytoken",
          "ticker": "BAKE"
        },
        {
          "name": "CHADS VC",
          "slug": "chads-vc",
          "ticker": "CHADS"
        },
        {
          "name": "Public Masterpiece Token",
          "slug": "public-masterpiece-token",
          "ticker": "PMT"
        },
        {
          "name": "Cat Token",
          "slug": "cat-token",
          "ticker": "CAT"
        },
        {
          "name": "Hoo Token",
          "slug": "hoo-token",
          "ticker": "HOO"
        },
        {
          "name": "NEST Protocol",
          "slug": "nest-protocol",
          "ticker": "NEST"
        },
        {
          "name": "Portion",
          "slug": "portion",
          "ticker": "PRT"
        },
        {
          "name": "Orion Protocol",
          "slug": "orion-protocol",
          "ticker": "ORN"
        },
        {
          "name": "1-UP Platform",
          "slug": "1-up",
          "ticker": "1-UP"
        },
        {
          "name": "Dogelon Mars",
          "slug": "dogelon",
          "ticker": "ELON"
        },
        {
          "name": "Multiplier",
          "slug": "multiplier",
          "ticker": "MXX"
        },
        {
          "name": "Decentr",
          "slug": "decentr",
          "ticker": "DEC"
        },
        {
          "name": "Houdini Swap",
          "slug": "lock-token",
          "ticker": "LOCK"
        },
        {
          "name": "MCDex",
          "slug": "mcdex",
          "ticker": "MCB"
        },
        {
          "name": "Alchemist",
          "slug": "alchemist",
          "ticker": "MIST"
        },
        {
          "name": "Dfyn Network",
          "slug": "dfyn-network",
          "ticker": "DFYN"
        },
        {
          "name": "yearn.finance",
          "slug": "yearn-finance",
          "ticker": "YFI"
        },
        {
          "name": "Swell Network",
          "slug": "swell-network",
          "ticker": "SWELL"
        },
        {
          "name": "VIDT DAO",
          "slug": "vidt-dao",
          "ticker": "VIDT"
        },
        {
          "name": "LunchMoney",
          "slug": "lunchmoney",
          "ticker": "LMY"
        },
        {
          "name": "Legia Warsaw Fan Token",
          "slug": "legia-warsaw-fan-token",
          "ticker": "LEG"
        },
        {
          "name": "Wrapped NXM",
          "slug": "wrapped-nxm",
          "ticker": "WNXM"
        },
        {
          "name": "Liquidity Dividends Protocol",
          "slug": "liquidity-dividends-protocol",
          "ticker": "LID"
        },
        {
          "name": "SHIBA INU",
          "slug": "shiba-inu",
          "ticker": "SHIB"
        },
        {
          "name": "IG Gold",
          "slug": "ig-gold",
          "ticker": "IGG"
        },
        {
          "name": "Republic Note",
          "slug": "republicnote",
          "ticker": "RPN"
        },
        {
          "name": "Method Finance",
          "slug": "method-finance",
          "ticker": "MTHD"
        },
        {
          "name": "Bitbns",
          "slug": "bitbns",
          "ticker": "BNS"
        },
        {
          "name": "TrueUSD [on Avalanche]",
          "slug": "a-trueusd",
          "ticker": "TUSD"
        },
        {
          "name": "WOM Protocol",
          "slug": "wom-protocol",
          "ticker": "WOM"
        },
        {
          "name": "Tendies",
          "slug": "tendies",
          "ticker": "TEND"
        },
        {
          "name": "Antiample",
          "slug": "antiample",
          "ticker": "XAMP"
        },
        {
          "name": "AGA Token",
          "slug": "aga",
          "ticker": "AGA"
        },
        {
          "name": "Fear NFTs",
          "slug": "fear-nfts",
          "ticker": "FEAR"
        },
        {
          "name": "Libertas Token",
          "slug": "libertas-token",
          "ticker": "LIBERTAS"
        },
        {
          "name": "DRIFE",
          "slug": "drife",
          "ticker": "$DRF"
        },
        {
          "name": "Sport and Leisure",
          "slug": "sport-and-leisure",
          "ticker": "SNL"
        },
        {
          "name": "Mask Network",
          "slug": "mask-network",
          "ticker": "MASK"
        },
        {
          "name": "SUKU",
          "slug": "suku",
          "ticker": "SUKU"
        },
        {
          "name": "AAX Token",
          "slug": "aax-token",
          "ticker": "AAB"
        },
        {
          "name": "Sora",
          "slug": "sora",
          "ticker": "XOR"
        },
        {
          "name": "MobileCoin",
          "slug": "mobilecoin",
          "ticker": "MOB"
        },
        {
          "name": "Spore Finance",
          "slug": "spore-finance",
          "ticker": "SPORE"
        },
        {
          "name": "Vanilla Network",
          "slug": "vanilla-network",
          "ticker": "VNLA"
        },
        {
          "name": "APENFT",
          "slug": "apenft",
          "ticker": "NFT"
        },
        {
          "name": "Dos Network",
          "slug": "dos-network",
          "ticker": "DOS"
        },
        {
          "name": "bAlpha",
          "slug": "balpha",
          "ticker": "BALPHA"
        },
        {
          "name": "Augur",
          "slug": "augur",
          "ticker": "REP"
        },
        {
          "name": "Origin Dollar",
          "slug": "origin-dollar",
          "ticker": "OUSD"
        },
        {
          "name": "Zap",
          "slug": "zap",
          "ticker": "ZAP"
        },
        {
          "name": "Truebit",
          "slug": "truebit",
          "ticker": "TRU"
        },
        {
          "name": "Goons of Balatroon",
          "slug": "goons-of-balatroon",
          "ticker": "GOB"
        },
        {
          "name": "KISSAN",
          "slug": "kissan",
          "ticker": "KSN"
        },
        {
          "name": "FXDX",
          "slug": "fxdx-exchange",
          "ticker": "FXDX"
        },
        {
          "name": "MetaQ",
          "slug": "metaq",
          "ticker": "METAQ"
        },
        {
          "name": "SOMESING",
          "slug": "somesing",
          "ticker": "SSX"
        },
        {
          "name": "Movement",
          "slug": "movement",
          "ticker": "MOVE"
        },
        {
          "name": "Suilend",
          "slug": "suilend",
          "ticker": "SEND"
        },
        {
          "name": "Fabric",
          "slug": "fabric",
          "ticker": "FAB"
        },
        {
          "name": "İstanbul Başakşehir Fan Token",
          "slug": "istanbul-basaksehir-fan-token",
          "ticker": "IBFK"
        },
        {
          "name": "Tacos",
          "slug": "tacos",
          "ticker": "TACO"
        },
        {
          "name": "Nerd Bot",
          "slug": "nerd-bot",
          "ticker": "NERD"
        },
        {
          "name": "Ethverse",
          "slug": "ethverse",
          "ticker": "ETHV"
        },
        {
          "name": "SafeMoon",
          "slug": "safemoon",
          "ticker": "SAFEMOON"
        },
        {
          "name": "hiBAYC",
          "slug": "hibayc",
          "ticker": "HIBAYC"
        },
        {
          "name": "Coin98",
          "slug": "coin98",
          "ticker": "C98"
        },
        {
          "name": "PieDAO DEFI++",
          "slug": "piedao-defi",
          "ticker": "DEFI++"
        },
        {
          "name": "Kiba Inu",
          "slug": "kiba-inu",
          "ticker": "KIBA"
        },
        {
          "name": "Chromia [on Ethereum]",
          "slug": "chromia",
          "ticker": "CHR"
        },
        {
          "name": "PERI Finance",
          "slug": "peri-finance",
          "ticker": "PERI"
        },
        {
          "name": "Göztepe S.K. Fan Token",
          "slug": "goztepe-sk-fantoken",
          "ticker": "GOZ"
        },
        {
          "name": "SunContract",
          "slug": "sun-contract",
          "ticker": "SNC"
        },
        {
          "name": "Strong",
          "slug": "strong",
          "ticker": "STRONG"
        },
        {
          "name": "DXdao",
          "slug": "dxdao",
          "ticker": "DXD"
        },
        {
          "name": "Tower",
          "slug": "tower-token",
          "ticker": "TOWER"
        },
        {
          "name": "NEAR Protocol",
          "slug": "near-protocol",
          "ticker": "NEAR"
        },
        {
          "name": "VIDT Datalink",
          "slug": "v-id",
          "ticker": "VIDT"
        },
        {
          "name": "YF Link",
          "slug": "yflink",
          "ticker": "YFL"
        },
        {
          "name": "STO Cash",
          "slug": "bnb-sto-cash",
          "ticker": "STOC"
        },
        {
          "name": "DOGEFI",
          "slug": "dogefi",
          "ticker": "DOGEFI"
        },
        {
          "name": "Loser Coin",
          "slug": "loser-coin",
          "ticker": "LOWB"
        },
        {
          "name": "Aave SNX",
          "slug": "aave-snx",
          "ticker": "aSNX"
        },
        {
          "name": "Arcona",
          "slug": "arcona",
          "ticker": "ARCONA"
        },
        {
          "name": "Revolt 2 Earn",
          "slug": "revolt-2-earn",
          "ticker": "RVLT"
        },
        {
          "name": "DSLA Protocol",
          "slug": "dsla-protocol",
          "ticker": "DSLA"
        },
        {
          "name": "Waves Enterprise",
          "slug": "waves-enterprise",
          "ticker": "WEST"
        },
        {
          "name": "Casper",
          "slug": "casper",
          "ticker": "CSPR"
        },
        {
          "name": "PowerTrade",
          "slug": "powertrade-fuel",
          "ticker": "PTF"
        },
        {
          "name": "LayerZero [on Arbitrum]",
          "slug": "arb-layerzero",
          "ticker": "ZRO"
        },
        {
          "name": "Chia Network",
          "slug": "chia-network",
          "ticker": "XCH"
        },
        {
          "name": "Nucleus Vision",
          "slug": "nucleus-vision",
          "ticker": "NCash"
        },
        {
          "name": "WingShop",
          "slug": "wingshop",
          "ticker": "WING"
        },
        {
          "name": "MixTrust",
          "slug": "mixtrust",
          "ticker": "MXT"
        },
        {
          "name": "Keysians Network",
          "slug": "keysians-network",
          "ticker": "KEN"
        },
        {
          "name": "Unisocks",
          "slug": "unisocks",
          "ticker": "SOCKS"
        },
        {
          "name": "Iconic Token",
          "slug": "iconic-token",
          "ticker": "ICNQ"
        },
        {
          "name": "Bird.Money",
          "slug": "bird-money",
          "ticker": "BIRD"
        },
        {
          "name": "Pepemon Pepeballs",
          "slug": "pepemon-pepeballs",
          "ticker": "PPBLZ"
        },
        {
          "name": "LooksRare",
          "slug": "looksrare",
          "ticker": "LOOKS"
        },
        {
          "name": "Force For Fast",
          "slug": "force-for-fast",
          "ticker": "FFF"
        },
        {
          "name": "Zelwin",
          "slug": "zelwin",
          "ticker": "ZLW"
        },
        {
          "name": "DefiDollar DAO",
          "slug": "defidollar-dao",
          "ticker": "DFD"
        },
        {
          "name": "Jasmy",
          "slug": "jasmy",
          "ticker": "JASMY"
        },
        {
          "name": "Alephium",
          "slug": "alephium",
          "ticker": "ALPH"
        },
        {
          "name": "Ceres",
          "slug": "ceres",
          "ticker": "CERES"
        },
        {
          "name": "Bridge Oracle",
          "slug": "bridge-oracle",
          "ticker": "BRG"
        },
        {
          "name": "Strike",
          "slug": "strike",
          "ticker": "STRK"
        },
        {
          "name": "Polkadot",
          "slug": "polkadot-new",
          "ticker": "DOT"
        },
        {
          "name": "SingularityNET",
          "slug": "singularitynet",
          "ticker": "AGIX"
        },
        {
          "name": "YUNo.finance",
          "slug": "yuno-finance",
          "ticker": "YUNO"
        },
        {
          "name": "Oxygen",
          "slug": "oxygen",
          "ticker": "OXY"
        },
        {
          "name": "Fox Trading",
          "slug": "fox-trading",
          "ticker": "FOXT"
        },
        {
          "name": "Raydium",
          "slug": "raydium",
          "ticker": "RAY"
        },
        {
          "name": "Liquity USD",
          "slug": "liquity-usd",
          "ticker": "LUSD"
        },
        {
          "name": "Quantfury Token",
          "slug": "quantfury-token",
          "ticker": "QTF"
        },
        {
          "name": "Axis DeFi",
          "slug": "axis-defi",
          "ticker": "AXIS"
        },
        {
          "name": "Pmeer",
          "slug": "pmeer",
          "ticker": "PMEER"
        },
        {
          "name": "HARD Protocol",
          "slug": "hard-protocol",
          "ticker": "HARD"
        },
        {
          "name": "WebDollar",
          "slug": "webdollar",
          "ticker": "WEBD"
        },
        {
          "name": "AceD",
          "slug": "aced",
          "ticker": "ACED"
        },
        {
          "name": "DegenVC",
          "slug": "degenvc",
          "ticker": "DGVC"
        },
        {
          "name": "Konomi Network",
          "slug": "konomi-network",
          "ticker": "KONO"
        },
        {
          "name": "Growth DeFi",
          "slug": "growthdefi",
          "ticker": "GRO"
        },
        {
          "name": "Everest",
          "slug": "everest",
          "ticker": "ID"
        },
        {
          "name": "Lien",
          "slug": "lien",
          "ticker": "LIEN"
        },
        {
          "name": "Trias Token",
          "slug": "trias-token",
          "ticker": "TRIAS"
        },
        {
          "name": "UniCrypt",
          "slug": "uncx",
          "ticker": "UNCX"
        },
        {
          "name": "Darwinia Network",
          "slug": "darwinia-network",
          "ticker": "RING"
        },
        {
          "name": "Darwinia Commitment Token",
          "slug": "darwinia-commitment-token",
          "ticker": "KTON"
        },
        {
          "name": "ORDI",
          "slug": "ordi",
          "ticker": "ORDI"
        },
        {
          "name": "Robonomics.network",
          "slug": "robonomics-network",
          "ticker": "XRT"
        },
        {
          "name": "PolkaFoundry",
          "slug": "polkafoundry",
          "ticker": "PKF"
        },
        {
          "name": "Rio DeFi",
          "slug": "rio-defi",
          "ticker": "RFUEL"
        },
        {
          "name": "Automata Network",
          "slug": "automata-network",
          "ticker": "ATA"
        },
        {
          "name": "Convex Finance",
          "slug": "convex-finance",
          "ticker": "CVX"
        },
        {
          "name": "Indexed Finance",
          "slug": "indexed-finance",
          "ticker": "NDX"
        },
        {
          "name": "Team Vitality Fan Token",
          "slug": "team-vitality-fan-token",
          "ticker": "VIT"
        },
        {
          "name": "Venus",
          "slug": "venus",
          "ticker": "XVS"
        },
        {
          "name": "TRUMP MAGA (trumpmaga.me)",
          "slug": "trump-maga-trumpmaga-me",
          "ticker": "MAGA"
        },
        {
          "name": "Dextrust",
          "slug": "dextrust",
          "ticker": "DETS"
        },
        {
          "name": "Syscoin",
          "slug": "syscoin",
          "ticker": "SYS"
        },
        {
          "name": "HUNT",
          "slug": "hunt",
          "ticker": "HUNT"
        },
        {
          "name": "Trade Token X",
          "slug": "trade-token-x",
          "ticker": "TIOX"
        },
        {
          "name": "Pacoca",
          "slug": "pacoca",
          "ticker": "PACOCA"
        },
        {
          "name": "Zenfuse",
          "slug": "zenfuse",
          "ticker": "ZEFU"
        },
        {
          "name": "Minter Network",
          "slug": "minter-network",
          "ticker": "BIP"
        },
        {
          "name": "HAPI",
          "slug": "hapi-one",
          "ticker": "HAPI"
        },
        {
          "name": "IOI Token",
          "slug": "trade-race-manager",
          "ticker": "IOI"
        },
        {
          "name": "Cortex",
          "slug": "cortex",
          "ticker": "CTXC"
        },
        {
          "name": "SmartCredit Token",
          "slug": "smartcredit-token",
          "ticker": "SMARTCREDIT"
        },
        {
          "name": "Sao Paulo FC Fan Token",
          "slug": "sao-paulo-fc-fan-token",
          "ticker": "SPFC"
        },
        {
          "name": "wIOTA",
          "slug": "wiota",
          "ticker": "wIOTA"
        },
        {
          "name": "KIMCHI.finance",
          "slug": "kimchi-finance",
          "ticker": "KIMCHI"
        },
        {
          "name": "DUST Protocol",
          "slug": "p-dust-protocol",
          "ticker": "DUST"
        },
        {
          "name": "BambooDeFi",
          "slug": "bamboo-defi",
          "ticker": "BAMBOO"
        },
        {
          "name": "Gitcoin",
          "slug": "gitcoin",
          "ticker": "GTC"
        },
        {
          "name": "CateCoin",
          "slug": "catecoin",
          "ticker": "CATE"
        },
        {
          "name": "CVCoin",
          "slug": "cvcoin",
          "ticker": "CVN"
        },
        {
          "name": "Raze Network",
          "slug": "raze-network",
          "ticker": "RAZE"
        },
        {
          "name": "DistX",
          "slug": "distx",
          "ticker": "DISTX"
        },
        {
          "name": "Ethereum Gold Project",
          "slug": "ethereum-gold-project",
          "ticker": "ETGP"
        },
        {
          "name": "Node Runners",
          "slug": "node-runners",
          "ticker": "NDR"
        },
        {
          "name": "Baby BNB",
          "slug": "bnb-baby-bnb",
          "ticker": "BABYBNB"
        },
        {
          "name": "UBXS Token",
          "slug": "ubxs",
          "ticker": "UBXS"
        },
        {
          "name": "FOMO (fomo.fund)",
          "slug": "fomo-fund",
          "ticker": "FOMO"
        },
        {
          "name": "Mochi (New)",
          "slug": "mochi-the-catcoin-new",
          "ticker": "MOCHI"
        },
        {
          "name": "Wrapped Gen-0 CryptoKitties",
          "slug": "wrapped-gen-0-cryptokitties",
          "ticker": "WG0"
        },
        {
          "name": "Harvest Finance",
          "slug": "harvest-finance",
          "ticker": "FARM"
        },
        {
          "name": "Props",
          "slug": "props",
          "ticker": "PROPS"
        },
        {
          "name": "OIN Finance",
          "slug": "oin-finance",
          "ticker": "OIN"
        },
        {
          "name": "Frax [on Avalanche]",
          "slug": "a-frax",
          "ticker": "FRAX"
        },
        {
          "name": "Earneo",
          "slug": "earneo",
          "ticker": "RNO"
        },
        {
          "name": "Carrot",
          "slug": "carrot",
          "ticker": "CRT"
        },
        {
          "name": "Portuma",
          "slug": "portuma",
          "ticker": "POR"
        },
        {
          "name": "YFDAI.FINANCE",
          "slug": "yfdai-finance",
          "ticker": "YF-DAI"
        },
        {
          "name": "XSGD",
          "slug": "xsgd",
          "ticker": "XSGD"
        },
        {
          "name": "RocketX",
          "slug": "rocket-vault-rocketx",
          "ticker": "RVF"
        },
        {
          "name": "Snek",
          "slug": "snek",
          "ticker": "SNEK"
        },
        {
          "name": "TONToken",
          "slug": "tontoken",
          "ticker": "TON"
        },
        {
          "name": "Digital Reserve Currency",
          "slug": "digital-reserve-currency",
          "ticker": "DRC"
        },
        {
          "name": "xDai",
          "slug": "xdai",
          "ticker": "STAKE"
        },
        {
          "name": "Celsius ",
          "slug": "celsius",
          "ticker": "CEL"
        },
        {
          "name": "Swerve",
          "slug": "swerve",
          "ticker": "SWRV"
        },
        {
          "name": "Lepricon",
          "slug": "lepricon",
          "ticker": "L3P"
        },
        {
          "name": "YFFS Finance",
          "slug": "yffs",
          "ticker": "YFFS"
        },
        {
          "name": "OpenDAO",
          "slug": "opendao",
          "ticker": "SOS"
        },
        {
          "name": "Chainflip",
          "slug": "chainflip",
          "ticker": "FLIP"
        },
        {
          "name": "Numogram",
          "slug": "numogram",
          "ticker": "GNON"
        },
        {
          "name": "Zerebro",
          "slug": "zerebro",
          "ticker": "ZEREBRO"
        },
        {
          "name": "DAFI Protocol",
          "slug": "dafi-protocol",
          "ticker": "DAFI"
        },
        {
          "name": "Anchor Protocol",
          "slug": "anchor-protocol",
          "ticker": "ANC"
        },
        {
          "name": "JackPool.finance",
          "slug": "jackpool-finance",
          "ticker": "JFI"
        },
        {
          "name": "BNSD Finance",
          "slug": "bnsd-finance",
          "ticker": "BNSD"
        },
        {
          "name": "Aleph.im",
          "slug": "aleph-im",
          "ticker": "ALEPH"
        },
        {
          "name": "JustLiquidity",
          "slug": "justliquidity",
          "ticker": "JUL"
        },
        {
          "name": "Dracula Token",
          "slug": "dracula-token",
          "ticker": "DRC"
        },
        {
          "name": "Crabada",
          "slug": "crabada",
          "ticker": "CRA"
        },
        {
          "name": "DefiBox",
          "slug": "defibox",
          "ticker": "BOX"
        },
        {
          "name": "Alchemy Pay",
          "slug": "alchemy-pay",
          "ticker": "ACH"
        },
        {
          "name": "Magic Eden",
          "slug": "magiceden",
          "ticker": "ME"
        },
        {
          "name": "BitDCA",
          "slug": "bitdca",
          "ticker": "BDCA"
        },
        {
          "name": "tBTC",
          "slug": "tbtc-token",
          "ticker": "TBTC"
        },
        {
          "name": "catwifhat",
          "slug": "catwifhatsolana",
          "ticker": "CWIF"
        },
        {
          "name": "Geeq",
          "slug": "geeq",
          "ticker": "GEEQ"
        },
        {
          "name": "swETH",
          "slug": "swell-sweth",
          "ticker": "SWETH"
        },
        {
          "name": "Swingby",
          "slug": "swingby",
          "ticker": "SWINGBY"
        },
        {
          "name": "Ferrum Network",
          "slug": "ferrum-network",
          "ticker": "FRM"
        },
        {
          "name": "MiL.k",
          "slug": "milk-alliance",
          "ticker": "MLK"
        },
        {
          "name": "Hedget",
          "slug": "hedget",
          "ticker": "HGET"
        },
        {
          "name": "BOB",
          "slug": "bob1",
          "ticker": "BOB"
        },
        {
          "name": "tao.bot",
          "slug": "tao-bot",
          "ticker": "TAOBOT"
        },
        {
          "name": "MAX",
          "slug": "max-2",
          "ticker": "MAX"
        },
        {
          "name": "GoPlus Security",
          "slug": "goplus-security",
          "ticker": "GPS"
        },
        {
          "name": "RZcoin",
          "slug": "rzcoin",
          "ticker": "RZ"
        },
        {
          "name": "Nugget Trap Gold Token",
          "slug": "nugget-trap-gold-token",
          "ticker": "NGTG$$"
        },
        {
          "name": "Jewelry Token",
          "slug": "jewelry-token",
          "ticker": "JEWELRY"
        },
        {
          "name": "Hippocrat",
          "slug": "hippocrat",
          "ticker": "HPO"
        },
        {
          "name": "Krypton Galaxy",
          "slug": "krypton-galaxy-coin",
          "ticker": "KGC"
        },
        {
          "name": "Base Protocol",
          "slug": "base-protocol",
          "ticker": "BASE"
        },
        {
          "name": "Phala.Network",
          "slug": "phala-network",
          "ticker": "PHA"
        },
        {
          "name": "Digital Fantasy Sports",
          "slug": "digital-fantasy-sports",
          "ticker": "DFS"
        },
        {
          "name": "TotemFi",
          "slug": "totemfi",
          "ticker": "TOTM"
        },
        {
          "name": "Tranchess",
          "slug": "tranchess",
          "ticker": "CHESS"
        },
        {
          "name": "Tornado",
          "slug": "tornado",
          "ticker": "TCORE"
        },
        {
          "name": "FC Barcelona Fan Token",
          "slug": "fc-barcelona-fan-token",
          "ticker": "BAR"
        },
        {
          "name": "MagicCraft",
          "slug": "magiccraft",
          "ticker": "MCRT"
        },
        {
          "name": "MEMETOON",
          "slug": "memetoon",
          "ticker": "MEME"
        },
        {
          "name": "GameSwift",
          "slug": "gameswift",
          "ticker": "GSWIFT"
        },
        {
          "name": "CropBytes",
          "slug": "cropbytes",
          "ticker": "CBX"
        },
        {
          "name": "ParagonsDAO",
          "slug": "paragonsdao",
          "ticker": "PDT"
        },
        {
          "name": "Waltonchain",
          "slug": "waltonchain",
          "ticker": "WTC"
        },
        {
          "name": "Symbol",
          "slug": "symbol",
          "ticker": "XYM"
        },
        {
          "name": "Peony",
          "slug": "peony",
          "ticker": "PNY"
        },
        {
          "name": "Coinsbit Token",
          "slug": "coinsbit-token",
          "ticker": "CNB"
        },
        {
          "name": "DeXe",
          "slug": "dexe",
          "ticker": "DEXE"
        },
        {
          "name": "Wing",
          "slug": "wing",
          "ticker": "WING"
        },
        {
          "name": "Power Index Pool Token",
          "slug": "power-index-pool-token",
          "ticker": "PIPT"
        },
        {
          "name": "ApeSwap Finance",
          "slug": "apeswap-finance",
          "ticker": "BANANA"
        },
        {
          "name": "Frontier",
          "slug": "frontier",
          "ticker": "FRONT"
        },
        {
          "name": "DeFi Pulse Index",
          "slug": "defi-pulse-index",
          "ticker": "DPI"
        },
        {
          "name": "Hamster",
          "slug": "hamster",
          "ticker": "HAM"
        },
        {
          "name": "KLAYswap Protocol",
          "slug": "klayswap-protocol",
          "ticker": "KSP"
        },
        {
          "name": "New BitShares",
          "slug": "new-bitshares",
          "ticker": "NBS"
        },
        {
          "name": "GALAXIA",
          "slug": "galaxia",
          "ticker": "GXA"
        },
        {
          "name": "Linear",
          "slug": "linear",
          "ticker": "LINA"
        },
        {
          "name": "Samurai",
          "slug": "samurai",
          "ticker": "SAM"
        },
        {
          "name": "Wolves of Wall Street",
          "slug": "wolves-of-wall-street",
          "ticker": "WOWS"
        },
        {
          "name": "Gold",
          "slug": "the-gold-token",
          "ticker": "GOLD"
        },
        {
          "name": "Float Protocol",
          "slug": "float-protocol",
          "ticker": "BANK"
        },
        {
          "name": "Beowulf",
          "slug": "beowulf",
          "ticker": "BWF"
        },
        {
          "name": "RIZON",
          "slug": "rizon-blockchain",
          "ticker": "ATOLO"
        },
        {
          "name": "OneSwap DAO Token",
          "slug": "oneswap-dao-token",
          "ticker": "ONES"
        },
        {
          "name": "Gifto",
          "slug": "gifto",
          "ticker": "GFT"
        },
        {
          "name": "ATOR Protocol",
          "slug": "airtor-protocol",
          "ticker": "ATOR"
        },
        {
          "name": "Leverj",
          "slug": "leverj",
          "ticker": "LEV"
        },
        {
          "name": "KCCPAD",
          "slug": "kccpad",
          "ticker": "KCCPAD"
        },
        {
          "name": "pNetwork",
          "slug": "pnetwork",
          "ticker": "PNT"
        },
        {
          "name": "Illuvium",
          "slug": "illuvium",
          "ticker": "ILV"
        },
        {
          "name": "Burger Swap",
          "slug": "burger-swap",
          "ticker": "BURGER"
        },
        {
          "name": "NFTY Token",
          "slug": "nfty-network",
          "ticker": "NFTY"
        },
        {
          "name": "Chainge",
          "slug": "chainge",
          "ticker": "XCHNG"
        },
        {
          "name": "Covalent",
          "slug": "covalent",
          "ticker": "CQT"
        },
        {
          "name": "Newscrypto",
          "slug": "newscrypto",
          "ticker": "NWC"
        },
        {
          "name": "Chainbing",
          "slug": "chainbing",
          "ticker": "CBG"
        },
        {
          "name": "PKT",
          "slug": "pkt",
          "ticker": "PKT"
        },
        {
          "name": "cVault.finance",
          "slug": "cvault-finance",
          "ticker": "CORE"
        },
        {
          "name": "Moutai",
          "slug": "moutai",
          "ticker": "MOUTAI"
        },
        {
          "name": "Bonded Finance",
          "slug": "bonded-finance",
          "ticker": "BOND"
        },
        {
          "name": "Stater",
          "slug": "stater",
          "ticker": "STR"
        },
        {
          "name": "Eight Hours",
          "slug": "eight-hours",
          "ticker": "EHRT"
        },
        {
          "name": "DappRadar",
          "slug": "dappradar",
          "ticker": "RADAR"
        },
        {
          "name": "Burnedfi",
          "slug": "burnedfi-app",
          "ticker": "BURN"
        },
        {
          "name": "UnlimitedIP",
          "slug": "unlimitedip",
          "ticker": "BTCU"
        },
        {
          "name": "FOX Token",
          "slug": "fox-token",
          "ticker": "FOX"
        },
        {
          "name": "Bidao",
          "slug": "bidao",
          "ticker": "BID"
        },
        {
          "name": "Netbox Coin",
          "slug": "netbox-coin",
          "ticker": "NBX"
        },
        {
          "name": "Mina",
          "slug": "mina",
          "ticker": "MINA"
        },
        {
          "name": "EnterCoin",
          "slug": "entercoin",
          "ticker": "ENTRC"
        },
        {
          "name": "Deflect",
          "slug": "deflect",
          "ticker": "DEFLCT"
        },
        {
          "name": "Metaverse Index",
          "slug": "metaverse-index",
          "ticker": "MVI"
        },
        {
          "name": "Bitcoin Standard Hashrate Token",
          "slug": "bitcoin-standard-hashrate-token",
          "ticker": "BTCST"
        },
        {
          "name": "Venus ETH",
          "slug": "venus-eth",
          "ticker": "vETH"
        },
        {
          "name": "Lyra [on Ethereum]",
          "slug": "lyra-finance",
          "ticker": "LYRA"
        },
        {
          "name": "Step Finance",
          "slug": "step-finance",
          "ticker": "STEP"
        },
        {
          "name": "FM Gallery",
          "slug": "fm-gallery",
          "ticker": "FMG"
        },
        {
          "name": "Ellipsis",
          "slug": "ellipsis",
          "ticker": "EPS"
        },
        {
          "name": "KuCoin Token",
          "slug": "kucoin-shares",
          "ticker": "KCS"
        },
        {
          "name": "Hyve",
          "slug": "hyve",
          "ticker": "HYVE"
        },
        {
          "name": "KARRAT",
          "slug": "karrat",
          "ticker": "KARRAT"
        },
        {
          "name": "0x",
          "slug": "0x",
          "ticker": "ZRX"
        },
        {
          "name": "Dawn Protocol",
          "slug": "dawn-protocol",
          "ticker": "DAWN"
        },
        {
          "name": "BullPerks",
          "slug": "bullperks",
          "ticker": "BLP"
        },
        {
          "name": "Moonbeam",
          "slug": "moonbeam",
          "ticker": "GLMR"
        },
        {
          "name": "A2DAO",
          "slug": "a2dao",
          "ticker": "ATD"
        },
        {
          "name": "Impossible Finance",
          "slug": "impossible-finance",
          "ticker": "IF"
        },
        {
          "name": "CryptoBlades",
          "slug": "cryptoblades",
          "ticker": "SKILL"
        },
        {
          "name": "OctoFi",
          "slug": "octofi",
          "ticker": "OCTO"
        },
        {
          "name": "Jade Protocol",
          "slug": "jade-protocol",
          "ticker": "JADE"
        },
        {
          "name": "BitGuild PLAT",
          "slug": "bitguild-plat",
          "ticker": "PLAT"
        },
        {
          "name": "Tellor",
          "slug": "tellor",
          "ticker": "TRB"
        },
        {
          "name": "Professional Fighters League Fan Token",
          "slug": "professional-fighters-league-fan-token",
          "ticker": "PFL"
        },
        {
          "name": "Plant Vs Undead",
          "slug": "plantvsundead",
          "ticker": "PVU"
        },
        {
          "name": "Livepeer",
          "slug": "livepeer",
          "ticker": "LPT"
        },
        {
          "name": "Celer",
          "slug": "celer-network",
          "ticker": "CELR"
        },
        {
          "name": "Wanchain",
          "slug": "wanchain",
          "ticker": "WAN"
        },
        {
          "name": "WhiteRock",
          "slug": "whiterock",
          "ticker": "WHITE"
        },
        {
          "name": "Anyswap",
          "slug": "anyswap",
          "ticker": "ANY"
        },
        {
          "name": "Polyient Games Governance Token",
          "slug": "polyient-games-governance-token",
          "ticker": "PGT"
        },
        {
          "name": "Azuki",
          "slug": "azuki",
          "ticker": "AZUKI"
        },
        {
          "name": "Soverain",
          "slug": "soverain",
          "ticker": "SOVE"
        },
        {
          "name": "MX Token",
          "slug": "mx-token",
          "ticker": "MX"
        },
        {
          "name": "DDEX",
          "slug": "ddex",
          "ticker": "DDEX"
        },
        {
          "name": "Keydonix",
          "slug": "keydonix",
          "ticker": "KDN"
        },
        {
          "name": "Wrapped Basic CryptoKitties",
          "slug": "wrapped-cryptokitties",
          "ticker": "WCK"
        },
        {
          "name": "Hedera",
          "slug": "hedera-hashgraph",
          "ticker": "HBAR"
        },
        {
          "name": "Image Generation AI",
          "slug": "image-generation-ai",
          "ticker": "IMGNAI"
        },
        {
          "name": "Lido Staked SOL",
          "slug": "lido-for-solana",
          "ticker": "stSOL"
        },
        {
          "name": "Sparkle",
          "slug": "sparkle",
          "ticker": "SPRKL"
        },
        {
          "name": "Haven Protocol",
          "slug": "haven-protocol",
          "ticker": "XHV"
        },
        {
          "name": "ECC",
          "slug": "eccoin",
          "ticker": "ECC"
        },
        {
          "name": "Zenon",
          "slug": "zenon",
          "ticker": "ZNN"
        },
        {
          "name": "Klaytn",
          "slug": "klaytn",
          "ticker": "KLAY"
        },
        {
          "name": "Phantasma Protocol",
          "slug": "phantasma",
          "ticker": "SOUL"
        },
        {
          "name": "Commercium",
          "slug": "commercium",
          "ticker": "CMM"
        },
        {
          "name": "0xBitcoin",
          "slug": "0xbtc",
          "ticker": "0xBTC"
        },
        {
          "name": "DigixDAO",
          "slug": "digixdao",
          "ticker": "DGD"
        },
        {
          "name": "ZelCash",
          "slug": "zelcash",
          "ticker": "ZEL"
        },
        {
          "name": "Throne",
          "slug": "throne",
          "ticker": "THN"
        },
        {
          "name": "KiboShib",
          "slug": "kiboshib",
          "ticker": "KIBSHI"
        },
        {
          "name": "Nexus",
          "slug": "nexus",
          "ticker": "NXS"
        },
        {
          "name": "Vertcoin",
          "slug": "vertcoin",
          "ticker": "VTC"
        },
        {
          "name": "Stellite",
          "slug": "stellite",
          "ticker": "XTL"
        },
        {
          "name": "ESBC",
          "slug": "esbc",
          "ticker": "ESBC"
        },
        {
          "name": "Ravencoin",
          "slug": "ravencoin",
          "ticker": "RVN"
        },
        {
          "name": "TokenPocket",
          "slug": "tokenpocket",
          "ticker": "TPT"
        },
        {
          "name": "Dogecoin",
          "slug": "dogecoin",
          "ticker": "DOGE"
        },
        {
          "name": "Rocket Pool",
          "slug": "rocket-pool",
          "ticker": "RPL"
        },
        {
          "name": "Pigeoncoin",
          "slug": "pigeoncoin",
          "ticker": "PGN"
        },
        {
          "name": "SUQA",
          "slug": "suqa",
          "ticker": "SUQA"
        },
        {
          "name": "Internet of Energy Network",
          "slug": "internet-of-energy-network",
          "ticker": "IOEN"
        },
        {
          "name": "LitLab Games",
          "slug": "litlab-games",
          "ticker": "LITT"
        },
        {
          "name": "NFTrade",
          "slug": "nftrade",
          "ticker": "NFTD"
        },
        {
          "name": "Signata",
          "slug": "signata",
          "ticker": "SATA"
        },
        {
          "name": "Brat",
          "slug": "brat-on-base",
          "ticker": "BRAT"
        },
        {
          "name": "Pirate Chain",
          "slug": "pirate-chain",
          "ticker": "ARRR"
        },
        {
          "name": "Semux",
          "slug": "semux",
          "ticker": "SEM"
        },
        {
          "name": "Grin",
          "slug": "grin",
          "ticker": "GRIN"
        },
        {
          "name": "Vipstar Coin",
          "slug": "vipstar-coin",
          "ticker": "VIPS"
        },
        {
          "name": "Bertram The Pomeranian",
          "slug": "bertram-the-pomeranian",
          "ticker": "BERT"
        },
        {
          "name": "Siacoin",
          "slug": "siacoin",
          "ticker": "SC"
        },
        {
          "name": "StarLink",
          "slug": "star-link",
          "ticker": "STARL"
        },
        {
          "name": "ABBC Coin",
          "slug": "abbc-coin",
          "ticker": "ABBC"
        },
        {
          "name": "LBRY Credits",
          "slug": "library-credit",
          "ticker": "LBC"
        },
        {
          "name": "Stakeborg DAO",
          "slug": "stakeborg-dao",
          "ticker": "STANDARD"
        },
        {
          "name": "NFTify",
          "slug": "nftify",
          "ticker": "N1"
        },
        {
          "name": "SaTT",
          "slug": "satt",
          "ticker": "SATT"
        },
        {
          "name": "tomiNet",
          "slug": "tominet",
          "ticker": "TOMI"
        },
        {
          "name": "Absolute",
          "slug": "absolute",
          "ticker": "ABS"
        },
        {
          "name": "Holyheld",
          "slug": "holyheld",
          "ticker": "HOLY"
        },
        {
          "name": "Value Liquidity",
          "slug": "value-defi-protocol",
          "ticker": "VALUE"
        },
        {
          "name": "Jarvis Network",
          "slug": "jarvis-network",
          "ticker": "JRT"
        },
        {
          "name": "ACRE",
          "slug": "acre",
          "ticker": "ACRE"
        },
        {
          "name": "Thore Cash",
          "slug": "thore-cash",
          "ticker": "TCH"
        },
        {
          "name": "InflationCoin",
          "slug": "inflationcoin",
          "ticker": "IFLT"
        },
        {
          "name": "Giant",
          "slug": "giant-coin",
          "ticker": "GIC"
        },
        {
          "name": "smARTOFGIVING",
          "slug": "smartofgiving",
          "ticker": "AOG"
        },
        {
          "name": "Doki Doki Finance",
          "slug": "doki-doki-finance",
          "ticker": "DOKI"
        },
        {
          "name": "HanChain",
          "slug": "hanchain",
          "ticker": "HAN"
        },
        {
          "name": "My DeFi Pet",
          "slug": "my-defi-pet",
          "ticker": "DPET"
        },
        {
          "name": "Green Ben",
          "slug": "green-ben",
          "ticker": "EBEN"
        },
        {
          "name": "Cyclone Protocol",
          "slug": "cyclone-protocol",
          "ticker": "CYC"
        },
        {
          "name": "Wrapped ApeCoin",
          "slug": "wrapped-apecoin-ape",
          "ticker": "WAPE"
        },
        {
          "name": "Fartcoin",
          "slug": "fartcoin",
          "ticker": "FARTCOIN"
        },
        {
          "name": "MOO DENG (moodeng.vip)",
          "slug": "moo-deng-token",
          "ticker": "MOODENG"
        },
        {
          "name": "Omnitude",
          "slug": "omnitude",
          "ticker": "ECOM"
        },
        {
          "name": "VelasPad",
          "slug": "velaspad",
          "ticker": "VLXPAD"
        },
        {
          "name": "BitCoin One",
          "slug": "bitcoin-one",
          "ticker": "BTCONE"
        },
        {
          "name": "GridCoin",
          "slug": "gridcoin",
          "ticker": "GRC"
        },
        {
          "name": "EVOS",
          "slug": "evos",
          "ticker": "EVOS"
        },
        {
          "name": "Basis Gold Share",
          "slug": "basis-gold-share",
          "ticker": "BSGS"
        },
        {
          "name": "sUSD",
          "slug": "susd",
          "ticker": "SUSD"
        },
        {
          "name": "Dimecoin",
          "slug": "dimecoin",
          "ticker": "DIME"
        },
        {
          "name": "Neutrino USD",
          "slug": "neutrino-dollar",
          "ticker": "USDN"
        },
        {
          "name": "Gearbox Protocol",
          "slug": "gearbox-protocol",
          "ticker": "GEAR"
        },
        {
          "name": "REGENT COIN",
          "slug": "regent-coin",
          "ticker": "REGENT"
        },
        {
          "name": "PearDAO",
          "slug": "peardao",
          "ticker": "PEX"
        },
        {
          "name": "Naka Bodhi Token",
          "slug": "naka-bodhi-token",
          "ticker": "NBOT"
        },
        {
          "name": "NavCoin",
          "slug": "nav-coin",
          "ticker": "NAV"
        },
        {
          "name": "VerusCoin",
          "slug": "veruscoin",
          "ticker": "VRSC"
        },
        {
          "name": "MidasProtocol",
          "slug": "midasprotocol",
          "ticker": "MAS"
        },
        {
          "name": "Labh Coin",
          "slug": "labh-coin",
          "ticker": "LABH"
        },
        {
          "name": "SafeInsure",
          "slug": "safeinsure",
          "ticker": "SINS"
        },
        {
          "name": "SmartCash",
          "slug": "smartcash",
          "ticker": "SMART"
        },
        {
          "name": "Bridge Protocol",
          "slug": "bridge-protocol",
          "ticker": "IAM"
        },
        {
          "name": "Mist",
          "slug": "mist",
          "ticker": "MIST"
        },
        {
          "name": "Kommunitas",
          "slug": "kommunitas",
          "ticker": "KOM"
        },
        {
          "name": "MASQ",
          "slug": "masq",
          "ticker": "MASQ"
        },
        {
          "name": "dForce",
          "slug": "dforce",
          "ticker": "DF"
        },
        {
          "name": "Ren",
          "slug": "ren",
          "ticker": "REN"
        },
        {
          "name": "Ontology Gas",
          "slug": "ontology-gas",
          "ticker": "ONG"
        },
        {
          "name": "IguVerse",
          "slug": "iguverse",
          "ticker": "IGU"
        },
        {
          "name": "Cazcoin",
          "slug": "cazcoin",
          "ticker": "CAZ"
        },
        {
          "name": "Cashaa",
          "slug": "cashaa",
          "ticker": "CAS"
        },
        {
          "name": "BridgeCoin",
          "slug": "bridgecoin",
          "ticker": "BCO"
        },
        {
          "name": "CMITCOIN",
          "slug": "cmitcoin",
          "ticker": "CMIT"
        },
        {
          "name": "eXPerience Chain",
          "slug": "experience-chain",
          "ticker": "XPC"
        },
        {
          "name": "CyberFi Token",
          "slug": "cyberfi",
          "ticker": "CFi"
        },
        {
          "name": "Grok",
          "slug": "grok-erc",
          "ticker": "GROK"
        },
        {
          "name": "Vyvo Coin",
          "slug": "vyvo-smart-chain",
          "ticker": "VSC"
        },
        {
          "name": "KubeCoin",
          "slug": "kubecoin",
          "ticker": "KUBE"
        },
        {
          "name": "Wicrypt",
          "slug": "wicrypt",
          "ticker": "WNT"
        },
        {
          "name": "Edgeless",
          "slug": "edgeless",
          "ticker": "EDG"
        },
        {
          "name": "ProxyNode",
          "slug": "proxynode",
          "ticker": "PRX"
        },
        {
          "name": "Verge",
          "slug": "verge",
          "ticker": "XVG"
        },
        {
          "name": "Shift",
          "slug": "shift",
          "ticker": "SHIFT"
        },
        {
          "name": "DODO",
          "slug": "dodo",
          "ticker": "DODO"
        },
        {
          "name": "The Abyss",
          "slug": "the-abyss",
          "ticker": "ABYSS"
        },
        {
          "name": "Hydrogen",
          "slug": "hydrogen",
          "ticker": "HYDRO"
        },
        {
          "name": "Ribbon Finance",
          "slug": "ribbon-finance",
          "ticker": "RBN"
        },
        {
          "name": "SingularDTV",
          "slug": "singulardtv",
          "ticker": "SNGLS"
        },
        {
          "name": "Handy",
          "slug": "handy",
          "ticker": "HANDY"
        },
        {
          "name": "Kinic",
          "slug": "kinic",
          "ticker": "KINIC"
        },
        {
          "name": "REI Network",
          "slug": "rei-network",
          "ticker": "REI"
        },
        {
          "name": "Opal",
          "slug": "opal",
          "ticker": "OPAL"
        },
        {
          "name": "Skycoin",
          "slug": "skycoin",
          "ticker": "SKY"
        },
        {
          "name": "IDEX",
          "slug": "idex",
          "ticker": "IDEX"
        },
        {
          "name": "bitcoin2network",
          "slug": "bitcoin2network",
          "ticker": "B2N"
        },
        {
          "name": "Metaverse ETP",
          "slug": "metaverse",
          "ticker": "ETP"
        },
        {
          "name": "MarcoPolo Protocol",
          "slug": "marcopolo-protocol",
          "ticker": "MAP"
        },
        {
          "name": "Aion",
          "slug": "aion",
          "ticker": "AION"
        },
        {
          "name": "ImageCoin",
          "slug": "imagecoin",
          "ticker": "IMG"
        },
        {
          "name": "Aragon",
          "slug": "aragon",
          "ticker": "ANT"
        },
        {
          "name": "NEO",
          "slug": "neo",
          "ticker": "NEO"
        },
        {
          "name": "SENATE DAO",
          "slug": "sidus-heroes",
          "ticker": "SENATE"
        },
        {
          "name": "Tether [on Avalanche]",
          "slug": "a-tether",
          "ticker": "USDT"
        },
        {
          "name": "Mercury",
          "slug": "mercury",
          "ticker": "MER"
        },
        {
          "name": "YIELD App",
          "slug": "yield-app",
          "ticker": "YLD"
        },
        {
          "name": "AdultChain",
          "slug": "adultchain",
          "ticker": "XXX"
        },
        {
          "name": "GINcoin",
          "slug": "gincoin",
          "ticker": "GIN"
        },
        {
          "name": "Edge Matrix Computing",
          "slug": "edge-matrix-computing",
          "ticker": "EMC"
        },
        {
          "name": "Project Pai",
          "slug": "project-pai",
          "ticker": "PAI"
        },
        {
          "name": "ChainLink [on Ethereum]",
          "slug": "chainlink",
          "ticker": "LINK"
        },
        {
          "name": "Kira Network",
          "slug": "kira-network",
          "ticker": "KEX"
        },
        {
          "name": "Dark Frontiers",
          "slug": "dark-frontiers",
          "ticker": "DARK"
        },
        {
          "name": "DeepBook Protocol",
          "slug": "deepbook-protocol",
          "ticker": "DEEP"
        },
        {
          "name": "Hivemapper",
          "slug": "hivemapper",
          "ticker": "HONEY"
        },
        {
          "name": "Harmony",
          "slug": "harmony",
          "ticker": "ONE"
        },
        {
          "name": "Nakamoto Games",
          "slug": "p-nakamoto-games",
          "ticker": "NAKA"
        },
        {
          "name": "GameCredits",
          "slug": "gamecredits",
          "ticker": "GAME"
        },
        {
          "name": "Bonfida",
          "slug": "bonfida",
          "ticker": "FIDA"
        },
        {
          "name": "Delphy",
          "slug": "delphy",
          "ticker": "DPY"
        },
        {
          "name": "Ubiq",
          "slug": "ubiq",
          "ticker": "UBQ"
        },
        {
          "name": "PIVX",
          "slug": "pivx",
          "ticker": "PIVX"
        },
        {
          "name": "win.win",
          "slug": "win-win",
          "ticker": "TWINS"
        },
        {
          "name": "Edwin",
          "slug": "edwin",
          "ticker": "EDWIN"
        },
        {
          "name": "Polychain Monsters",
          "slug": "polkamon",
          "ticker": "PMON"
        },
        {
          "name": "Venus BUSD",
          "slug": "venus-busd",
          "ticker": "vBUSD"
        },
        {
          "name": "O3 Swap",
          "slug": "o3-swap",
          "ticker": "O3"
        },
        {
          "name": "Mirrored Tesla",
          "slug": "mirrored-tesla",
          "ticker": "mTSLA"
        },
        {
          "name": "Sentivate",
          "slug": "sentivate",
          "ticker": "SNTVT"
        },
        {
          "name": "OPCoinX",
          "slug": "opcoinx",
          "ticker": "OPCX"
        },
        {
          "name": "Blocknode",
          "slug": "blocknode",
          "ticker": "BND"
        },
        {
          "name": "Metacade",
          "slug": "metacade",
          "ticker": "MCADE"
        },
        {
          "name": "Yeld Finance",
          "slug": "yeld-finance",
          "ticker": "YELD"
        },
        {
          "name": "Energi",
          "slug": "energi",
          "ticker": "NRG"
        },
        {
          "name": "Woonkly Power",
          "slug": "woonkly-power",
          "ticker": "WOOP"
        },
        {
          "name": "VeThor Token",
          "slug": "vethor-token",
          "ticker": "VTHO"
        },
        {
          "name": "Edgeware",
          "slug": "edgeware",
          "ticker": "EDG"
        },
        {
          "name": "Neutron",
          "slug": "neutron-ntrn",
          "ticker": "NTRN"
        },
        {
          "name": "L",
          "slug": "l",
          "ticker": "L"
        },
        {
          "name": "Stella [on Ethereum]",
          "slug": "alpha-finance-lab",
          "ticker": "ALPHA"
        },
        {
          "name": "Hemule",
          "slug": "hemule",
          "ticker": "HEMULE"
        },
        {
          "name": "Monolith",
          "slug": "monolith",
          "ticker": "TKN"
        },
        {
          "name": "Bitblocks",
          "slug": "bitblocks",
          "ticker": "BBK"
        },
        {
          "name": "TOP",
          "slug": "top",
          "ticker": "TOP"
        },
        {
          "name": "UNKJD",
          "slug": "unkjd",
          "ticker": "MBS"
        },
        {
          "name": "STK",
          "slug": "stk",
          "ticker": "STK"
        },
        {
          "name": "ROIyal Coin",
          "slug": "roiyal-coin",
          "ticker": "ROCO"
        },
        {
          "name": "Stably USD",
          "slug": "stableusd",
          "ticker": "USDS"
        },
        {
          "name": "Natus Vincere Fan Token",
          "slug": "natus-vincere-fan-token",
          "ticker": "NAVI"
        },
        {
          "name": "Millonarios FC Fan Token",
          "slug": "millonarios-fc-fan-token",
          "ticker": "MFC"
        },
        {
          "name": "Nabox",
          "slug": "nabox",
          "ticker": "NABOX"
        },
        {
          "name": "BiFi",
          "slug": "bifi",
          "ticker": "BIFI"
        },
        {
          "name": "ZooKeeper",
          "slug": "zookeeper",
          "ticker": "ZOO"
        },
        {
          "name": "Gem Exchange And Trading",
          "slug": "gem-exchange-and-trading",
          "ticker": "GXT"
        },
        {
          "name": "Aegeus",
          "slug": "aegeus",
          "ticker": "AEG"
        },
        {
          "name": "TABOO TOKEN",
          "slug": "taboo-token",
          "ticker": "TABOO"
        },
        {
          "name": "Lightpaycoin",
          "slug": "lightpaycoin",
          "ticker": "LPC"
        },
        {
          "name": "Digix Gold Token",
          "slug": "digix-gold-token",
          "ticker": "DGX"
        },
        {
          "name": "Electra",
          "slug": "electra",
          "ticker": "ECA"
        },
        {
          "name": "Aencoin",
          "slug": "aencoin",
          "ticker": "AEN"
        },
        {
          "name": "EXMO Coin",
          "slug": "exmo-coin",
          "ticker": "EXM"
        },
        {
          "name": "LinqAI",
          "slug": "linqai",
          "ticker": "LNQ"
        },
        {
          "name": "Drift",
          "slug": "drift",
          "ticker": "DRIFT"
        },
        {
          "name": "HyperCycle",
          "slug": "hypercycle",
          "ticker": "HYPC"
        },
        {
          "name": "Commune AI",
          "slug": "commune-ai",
          "ticker": "COMAI"
        },
        {
          "name": "Kadena",
          "slug": "kadena",
          "ticker": "KDA"
        },
        {
          "name": "Horizen",
          "slug": "zencash",
          "ticker": "ZEN"
        },
        {
          "name": "Sentinel Protocol",
          "slug": "sentinel-protocol",
          "ticker": "BOUNTY"
        },
        {
          "name": "Zcash",
          "slug": "zcash",
          "ticker": "ZEC"
        },
        {
          "name": "MetaBeat",
          "slug": "p-metabeat",
          "ticker": "BEAT"
        },
        {
          "name": "Eldarune",
          "slug": "eldarune",
          "ticker": "ELDA"
        },
        {
          "name": "Dash",
          "slug": "dash",
          "ticker": "DASH"
        },
        {
          "name": "Origin Protocol",
          "slug": "origin-protocol",
          "ticker": "OGN"
        },
        {
          "name": "Leverj Gluon",
          "slug": "leverj-gluon",
          "ticker": "L2"
        },
        {
          "name": "Liquid Staked ETH",
          "slug": "liquid-staked-eth",
          "ticker": "LSETH"
        },
        {
          "name": "Kima Network",
          "slug": "kima-network",
          "ticker": "KIMA"
        },
        {
          "name": "Veil",
          "slug": "veil",
          "ticker": "VEIL"
        },
        {
          "name": "CREA",
          "slug": "crea",
          "ticker": "CREA"
        },
        {
          "name": "Qwertycoin",
          "slug": "qwertycoin",
          "ticker": "QWC"
        },
        {
          "name": "NIX Bridge Token",
          "slug": "nix-bridge-token",
          "ticker": "NBT"
        },
        {
          "name": "Neopin",
          "slug": "neopin",
          "ticker": "NPT"
        },
        {
          "name": "Nyzo",
          "slug": "nyzo",
          "ticker": "NYZO"
        },
        {
          "name": "Spores Network",
          "slug": "spores-network",
          "ticker": "SPO"
        },
        {
          "name": "NewYorkCoin",
          "slug": "newyorkcoin",
          "ticker": "NYC"
        },
        {
          "name": "DeFiZap",
          "slug": "defizap",
          "ticker": "DFZ"
        },
        {
          "name": "88mph",
          "slug": "88mph",
          "ticker": "MPH"
        },
        {
          "name": "Fringe Finance",
          "slug": "fringe-finance",
          "ticker": "FRIN"
        },
        {
          "name": "RSS3",
          "slug": "rss3",
          "ticker": "RSS3"
        },
        {
          "name": "Ninneko",
          "slug": "ninneko",
          "ticker": "NINO"
        },
        {
          "name": "ether.fi Staked ETH",
          "slug": "ether-fi",
          "ticker": "EETH"
        },
        {
          "name": "Junkcoin",
          "slug": "junkcoin",
          "ticker": "JKC"
        },
        {
          "name": "tokenbot",
          "slug": "tokenbot-2",
          "ticker": "CLANKER"
        },
        {
          "name": "Andy BSC",
          "slug": "andytoken-bsc",
          "ticker": "ANDY"
        },
        {
          "name": "Blocknet",
          "slug": "blocknet",
          "ticker": "BLOCK"
        },
        {
          "name": "Shopping",
          "slug": "shopping",
          "ticker": "SPI"
        },
        {
          "name": "Algorand",
          "slug": "algorand",
          "ticker": "ALGO"
        },
        {
          "name": "pBTC35A",
          "slug": "pbtc35a",
          "ticker": "pBTC35A"
        },
        {
          "name": "PoolTogether",
          "slug": "pooltogether",
          "ticker": "POOL"
        },
        {
          "name": "Forest Knight",
          "slug": "forest-knight",
          "ticker": "KNIGHT"
        },
        {
          "name": "Stader ETHx",
          "slug": "stader-ethx",
          "ticker": "ETHX"
        },
        {
          "name": "Ondori",
          "slug": "ondori",
          "ticker": "RSTR"
        },
        {
          "name": "Wrapped Pulse",
          "slug": "wrapped-pulse",
          "ticker": "WPLS"
        },
        {
          "name": "UNION Protocol Governance Token",
          "slug": "union-protocol-governance-token",
          "ticker": "UNN"
        },
        {
          "name": "Tiger King",
          "slug": "tiger-king",
          "ticker": "TKING"
        },
        {
          "name": "Keep Network",
          "slug": "keep-network",
          "ticker": "KEEP"
        },
        {
          "name": "Cartesi",
          "slug": "cartesi",
          "ticker": "CTSI"
        },
        {
          "name": "Badger DAO",
          "slug": "badger-dao",
          "ticker": "BADGER"
        },
        {
          "name": "Akita Inu",
          "slug": "akita-inu",
          "ticker": "AKITA"
        },
        {
          "name": "YFFII Finance",
          "slug": "yffii-finance",
          "ticker": "YFFII"
        },
        {
          "name": "Verasity",
          "slug": "verasity",
          "ticker": "VRA"
        },
        {
          "name": "Mirror Protocol",
          "slug": "mirror-protocol",
          "ticker": "MIR"
        },
        {
          "name": "Arweave",
          "slug": "arweave",
          "ticker": "AR"
        },
        {
          "name": "Compound",
          "slug": "compound",
          "ticker": "COMP"
        },
        {
          "name": "NuCypher",
          "slug": "nucypher",
          "ticker": "NU"
        },
        {
          "name": "Monavale",
          "slug": "monavale",
          "ticker": "MONA"
        },
        {
          "name": "Celo",
          "slug": "celo",
          "ticker": "CELO"
        },
        {
          "name": "Dimitra",
          "slug": "dimitra",
          "ticker": "DMTR"
        },
        {
          "name": "Ambrosus",
          "slug": "amber",
          "ticker": "AMB"
        },
        {
          "name": "Vectorspace",
          "slug": "vectorspace-ai",
          "ticker": "VXV"
        },
        {
          "name": "Morphware",
          "slug": "morphware",
          "ticker": "XMW"
        },
        {
          "name": "TemDAO",
          "slug": "temdao",
          "ticker": "TEM"
        },
        {
          "name": "Rainmaker Games",
          "slug": "rainmaker-games",
          "ticker": "RAIN"
        },
        {
          "name": "Sanctum Infinity",
          "slug": "sanctum-infinity",
          "ticker": "INF"
        },
        {
          "name": "Jito Staked SOL",
          "slug": "jito-staked-sol",
          "ticker": "JITOSOL"
        },
        {
          "name": "Palette",
          "slug": "palette",
          "ticker": "PLT"
        },
        {
          "name": "404Aliens",
          "slug": "404aliens",
          "ticker": "404A"
        },
        {
          "name": "Serum",
          "slug": "serum",
          "ticker": "SRM"
        },
        {
          "name": "TrustSwap",
          "slug": "trustswap",
          "ticker": "SWAP"
        },
        {
          "name": "DFI.Money",
          "slug": "yearn-finance-ii",
          "ticker": "YFII"
        },
        {
          "name": "Meta",
          "slug": "meta",
          "ticker": "MTA"
        },
        {
          "name": "Custodiy",
          "slug": "bnb-custodiy",
          "ticker": "CTY"
        },
        {
          "name": "NerveNetwork",
          "slug": "nervenetwork",
          "ticker": "NVT"
        },
        {
          "name": "Meter",
          "slug": "meter",
          "ticker": "MTRG"
        },
        {
          "name": "Ontology",
          "slug": "ontology",
          "ticker": "ONT"
        },
        {
          "name": "Young Boys Fan Token",
          "slug": "young-boys-fan-token",
          "ticker": "YBO"
        },
        {
          "name": "Wilder World",
          "slug": "wilder-world",
          "ticker": "WILD"
        },
        {
          "name": "Dogechain",
          "slug": "dogechain",
          "ticker": "DC"
        },
        {
          "name": "Gameswap",
          "slug": "gameswap",
          "ticker": "GSWAP"
        },
        {
          "name": "FIO Protocol",
          "slug": "fio-protocol",
          "ticker": "FIO"
        },
        {
          "name": "Datamine",
          "slug": "datamine",
          "ticker": "DAM"
        },
        {
          "name": "Meridian Network",
          "slug": "meridian-network",
          "ticker": "LOCK"
        },
        {
          "name": "Proton",
          "slug": "proton",
          "ticker": "XPR"
        },
        {
          "name": "Cream Finance",
          "slug": "cream-finance",
          "ticker": "CREAM"
        },
        {
          "name": "The Sandbox",
          "slug": "the-sandbox",
          "ticker": "SAND"
        },
        {
          "name": "MetamonkeyAi",
          "slug": "metamonkeyai",
          "ticker": "MMAI"
        },
        {
          "name": "Dynamic Set Dollar",
          "slug": "dynamic-set-dollar",
          "ticker": "DSD"
        },
        {
          "name": "Bitgear",
          "slug": "bitgear",
          "ticker": "GEAR"
        },
        {
          "name": "Hakka.Finance",
          "slug": "hakka-finance",
          "ticker": "HAKKA"
        },
        {
          "name": "GET Protocol",
          "slug": "get-protocol",
          "ticker": "GET"
        },
        {
          "name": "PROXI",
          "slug": "proxi",
          "ticker": "CREDIT"
        },
        {
          "name": "Kusama",
          "slug": "kusama",
          "ticker": "KSM"
        },
        {
          "name": "Open Predict Token",
          "slug": "open-predict-token",
          "ticker": "OPT"
        },
        {
          "name": "UniLayer",
          "slug": "unilayer",
          "ticker": "LAYER"
        },
        {
          "name": "Mancium",
          "slug": "mancium",
          "ticker": "MANC"
        },
        {
          "name": "Rubic",
          "slug": "rubic",
          "ticker": "RBC"
        },
        {
          "name": "Gravity Finance",
          "slug": "gravity-finance",
          "ticker": "GFI"
        },
        {
          "name": "BeamSwap",
          "slug": "beamswap",
          "ticker": "GLINT"
        },
        {
          "name": "Oddz",
          "slug": "oddz",
          "ticker": "ODDZ"
        },
        {
          "name": "Secret",
          "slug": "secret",
          "ticker": "SCRT"
        },
        {
          "name": "Vidya",
          "slug": "vidya",
          "ticker": "VIDYA"
        },
        {
          "name": "FRAKT Token",
          "slug": "frakt-token",
          "ticker": "FRKT"
        },
        {
          "name": "yffi finance",
          "slug": "yffi-finance",
          "ticker": "YFFI"
        },
        {
          "name": "GT Protocol",
          "slug": "bnb-gt-protocol",
          "ticker": "GTAI"
        },
        {
          "name": "Cirus Foundation",
          "slug": "cirus-foundation",
          "ticker": "CIRUS"
        },
        {
          "name": "Chain Games",
          "slug": "chain-games",
          "ticker": "CHAIN"
        },
        {
          "name": "WETH [on Optimism]",
          "slug": "o-weth",
          "ticker": "WETH"
        },
        {
          "name": "Fuse Network",
          "slug": "fuse-network",
          "ticker": "FUSE"
        },
        {
          "name": "SushiSwap [on Ethereum]",
          "slug": "sushi",
          "ticker": "SUSHI"
        },
        {
          "name": "Shib Original Vision",
          "slug": "shib-original-vision",
          "ticker": "SOV"
        },
        {
          "name": "Wrapped Bitcoin [on Arbitrum]",
          "slug": "arb-wrapped-bitcoin",
          "ticker": "WBTC"
        },
        {
          "name": "VAIOT",
          "slug": "vaiot",
          "ticker": "VAI"
        },
        {
          "name": "Musk It",
          "slug": "musk-it",
          "ticker": "MUSKIT"
        },
        {
          "name": "StarSlax",
          "slug": "starslax",
          "ticker": "SSLX"
        },
        {
          "name": "Galaxy Heroes Coin",
          "slug": "galaxy-heroes-coin",
          "ticker": "GHC"
        },
        {
          "name": "Pearl",
          "slug": "pearl",
          "ticker": "PEARL"
        },
        {
          "name": "Boosted Finance",
          "slug": "boosted-finance",
          "ticker": "BOOST"
        },
        {
          "name": "SalmonSwap",
          "slug": "salmonswap",
          "ticker": "SAL"
        },
        {
          "name": "RMRK",
          "slug": "rmrk",
          "ticker": "RMRK"
        },
        {
          "name": "SOHOTRN",
          "slug": "sohotrn",
          "ticker": "SOHOT"
        },
        {
          "name": "Analog",
          "slug": "analog",
          "ticker": "ANLOG"
        },
        {
          "name": "WELF",
          "slug": "welf",
          "ticker": "WELF"
        },
        {
          "name": "Forward Protocol",
          "slug": "forward-protocol",
          "ticker": "FORWARD"
        },
        {
          "name": "Wrapped ONUS",
          "slug": "wrapped-onus",
          "ticker": "WONUS"
        },
        {
          "name": "Broccoli (firstbroccoli.com)",
          "slug": "broccoli",
          "ticker": "BROCCOLI"
        },
        {
          "name": "Unitrade",
          "slug": "unitrade",
          "ticker": "TRADE"
        },
        {
          "name": "NuNet",
          "slug": "nunet",
          "ticker": "NTX"
        },
        {
          "name": "Acquire.Fi",
          "slug": "acquire-fi",
          "ticker": "ACQ"
        },
        {
          "name": "Golff",
          "slug": "golff",
          "ticker": "GOF"
        },
        {
          "name": "Pickle Finance",
          "slug": "pickle-finance",
          "ticker": "PICKLE"
        },
        {
          "name": "PowerPool",
          "slug": "powerpool",
          "ticker": "CVP"
        },
        {
          "name": "Spheroid Universe",
          "slug": "spheroid-universe",
          "ticker": "SPH"
        },
        {
          "name": "Earn Network",
          "slug": "earn-network",
          "ticker": "EARN"
        },
        {
          "name": "The Graph",
          "slug": "the-graph",
          "ticker": "GRT"
        },
        {
          "name": "Zoracles",
          "slug": "zoracles",
          "ticker": "ZORA"
        },
        {
          "name": "Bella Protocol",
          "slug": "bella-protocol",
          "ticker": "BEL"
        },
        {
          "name": "Flamingo",
          "slug": "flamingo",
          "ticker": "FLM"
        },
        {
          "name": "dHedge DAO",
          "slug": "dhedge-dao",
          "ticker": "DHT"
        },
        {
          "name": "Energy Web Token",
          "slug": "energy-web-token",
          "ticker": "EWT"
        },
        {
          "name": "yieldfarming.insure",
          "slug": "yieldfarming-insure",
          "ticker": "SAFE"
        },
        {
          "name": "Dego Finance",
          "slug": "dego-finance",
          "ticker": "DEGO"
        },
        {
          "name": "SafeCoin",
          "slug": "safecoin",
          "ticker": "SAFE"
        },
        {
          "name": "Openfabric AI",
          "slug": "bnb-openfabric-ai",
          "ticker": "OFN"
        },
        {
          "name": "Rarible",
          "slug": "rarible",
          "ticker": "RARI"
        },
        {
          "name": "AcknoLedger",
          "slug": "acknoledger",
          "ticker": "ACK"
        },
        {
          "name": "Belt Finance",
          "slug": "belt",
          "ticker": "BELT"
        },
        {
          "name": "Sapphire",
          "slug": "sapphire",
          "ticker": "SAPP"
        },
        {
          "name": "Nyan Heroes",
          "slug": "nyan-heroes",
          "ticker": "NYAN"
        },
        {
          "name": "USD Coin Bridged ZED20",
          "slug": "usd-coin-bridged-zed20",
          "ticker": "USDC.z"
        },
        {
          "name": "Mirrored Alibaba",
          "slug": "mirrored-alibaba",
          "ticker": "mBABA"
        },
        {
          "name": "Dai [on Polygon]",
          "slug": "p-multi-collateral-dai",
          "ticker": "DAI"
        },
        {
          "name": "Fwog Takes",
          "slug": "fwog-takes",
          "ticker": "FWOG"
        },
        {
          "name": "Hey Anon",
          "slug": "hey-anon",
          "ticker": "ANON"
        },
        {
          "name": "PieDAO DOUGH v2",
          "slug": "piedao-dough-v2",
          "ticker": "DOUGH"
        },
        {
          "name": "SmartKey",
          "slug": "smartkey",
          "ticker": "SKEY"
        },
        {
          "name": "Agatech",
          "slug": "agatech",
          "ticker": "AGATA"
        },
        {
          "name": "Perry",
          "slug": "perry-bnb",
          "ticker": "PERRY"
        },
        {
          "name": "Lossless",
          "slug": "lossless",
          "ticker": "LSS"
        },
        {
          "name": "Wall Street Games",
          "slug": "wall-street-games-new",
          "ticker": "WSG"
        },
        {
          "name": "Orange",
          "slug": "orange-crypto",
          "ticker": "ORNJ"
        },
        {
          "name": "Cheems",
          "slug": "cheems",
          "ticker": "CHEEMS"
        },
        {
          "name": "Multiverse",
          "slug": "multiverse",
          "ticker": "AI"
        },
        {
          "name": "Shiden Network",
          "slug": "shiden-network",
          "ticker": "SDN"
        },
        {
          "name": "Songbird",
          "slug": "songbird",
          "ticker": "SGB"
        },
        {
          "name": "Beyond Protocol",
          "slug": "beyond-protocol",
          "ticker": "BP"
        },
        {
          "name": "BabySwap",
          "slug": "babyswap",
          "ticker": "BABY"
        },
        {
          "name": "SORA GROK",
          "slug": "sora-grok",
          "ticker": "GROK"
        },
        {
          "name": "Wibegram",
          "slug": "wibegram",
          "ticker": "WIBE"
        },
        {
          "name": "EPIK Prime",
          "slug": "epik-prime",
          "ticker": "EPIK"
        },
        {
          "name": "SKI MASK BRETT",
          "slug": "ski-mask-brett",
          "ticker": "SKIB"
        },
        {
          "name": "AKITA-BSC",
          "slug": "akita-bsc",
          "ticker": "AKITA"
        },
        {
          "name": "E4C",
          "slug": "e4c-final-salvation",
          "ticker": "E4C"
        },
        {
          "name": "Rimbit",
          "slug": "rimbit",
          "ticker": "RBT"
        },
        {
          "name": "neur.sh",
          "slug": "neur-sh",
          "ticker": "NEUR"
        },
        {
          "name": "Arianee",
          "slug": "arianee-protocol",
          "ticker": "ARIA20"
        },
        {
          "name": "MintMe.com Coin",
          "slug": "mintme-com-coin",
          "ticker": "MINTME"
        },
        {
          "name": "OpenOcean",
          "slug": "openocean",
          "ticker": "OOE"
        },
        {
          "name": "Merit Circle",
          "slug": "merit-circle",
          "ticker": "MC"
        },
        {
          "name": "Alitas",
          "slug": "alitas",
          "ticker": "ALT"
        },
        {
          "name": "Neos Credits",
          "slug": "neos-credits",
          "ticker": "NCR"
        },
        {
          "name": "Unifty",
          "slug": "unifty",
          "ticker": "NIF"
        },
        {
          "name": "Impossible Decentralized Incubator Access",
          "slug": "impossible-decentralized-incubator-access",
          "ticker": "IDIA"
        },
        {
          "name": "RChain",
          "slug": "rchain",
          "ticker": "REV"
        },
        {
          "name": "Sovryn",
          "slug": "sovryn",
          "ticker": "SOV"
        },
        {
          "name": "Dvision Network",
          "slug": "dvision-network",
          "ticker": "DVI"
        },
        {
          "name": "Swarm",
          "slug": "ethereum-swarm",
          "ticker": "BZZ"
        },
        {
          "name": "SifChain",
          "slug": "sifchain",
          "ticker": "erowan"
        },
        {
          "name": "XeniosCoin",
          "slug": "xenioscoin",
          "ticker": "XNC"
        },
        {
          "name": "Human",
          "slug": "human",
          "ticker": "HMT"
        },
        {
          "name": "Radix",
          "slug": "radix-protocol",
          "ticker": "XRD"
        },
        {
          "name": "Equalizer",
          "slug": "equalizer",
          "ticker": "EQZ"
        },
        {
          "name": "GamerCoin",
          "slug": "gamercoin",
          "ticker": "GHX"
        },
        {
          "name": "Darma Cash",
          "slug": "darma-cash",
          "ticker": "DMCH"
        },
        {
          "name": "PlatON",
          "slug": "platon",
          "ticker": "LAT"
        },
        {
          "name": "Reploy",
          "slug": "reploy",
          "ticker": "RAI"
        },
        {
          "name": "Samoyedcoin",
          "slug": "samoyedcoin",
          "ticker": "SAMO"
        },
        {
          "name": "Spell Token",
          "slug": "spell-token",
          "ticker": "SPELL"
        },
        {
          "name": "RichQUACK.com",
          "slug": "richquack-com",
          "ticker": "QUACK"
        },
        {
          "name": "DeversiFi",
          "slug": "deversifi",
          "ticker": "DVF"
        },
        {
          "name": "Saito",
          "slug": "saito",
          "ticker": "SAITO"
        },
        {
          "name": "BitBall",
          "slug": "bitball",
          "ticker": "BTB"
        },
        {
          "name": "Lotto",
          "slug": "lotto",
          "ticker": "LOTTO"
        },
        {
          "name": "CryptoPlanes",
          "slug": "cryptoplanes",
          "ticker": "CPAN"
        },
        {
          "name": "Defina Finance",
          "slug": "defina-finance",
          "ticker": "FINA"
        },
        {
          "name": "Gemma Extending Tech",
          "slug": "gemma-extending-tech",
          "ticker": "GXT"
        },
        {
          "name": "Deeper Network",
          "slug": "deeper-network",
          "ticker": "DPR"
        },
        {
          "name": "Bloomzed Loyalty Club Ticket",
          "slug": "bloomzed-token",
          "ticker": "BLCT"
        },
        {
          "name": "StarTerra",
          "slug": "starterra",
          "ticker": "STT"
        },
        {
          "name": "Pitbull",
          "slug": "pitbull",
          "ticker": "PIT"
        },
        {
          "name": "Numbers Protocol",
          "slug": "numbers-protocol",
          "ticker": "NUM"
        },
        {
          "name": "Alpha Quark Token",
          "slug": "alpha-quark-token",
          "ticker": "AQT"
        },
        {
          "name": "DeFi Land",
          "slug": "defi-land",
          "ticker": "DFL"
        },
        {
          "name": "BEPRO Network",
          "slug": "bepro-network",
          "ticker": "BEPRO"
        },
        {
          "name": "DOGGY",
          "slug": "doggy",
          "ticker": "DOGGY"
        },
        {
          "name": "Epic Cash",
          "slug": "epic-cash",
          "ticker": "EPIC"
        },
        {
          "name": "Ooki Protocol",
          "slug": "ooki-protocol",
          "ticker": "OOKI"
        },
        {
          "name": "SpookySwap",
          "slug": "spookyswap",
          "ticker": "BOO"
        },
        {
          "name": "Velo",
          "slug": "velo",
          "ticker": "VELO"
        },
        {
          "name": "Acala Token",
          "slug": "acala",
          "ticker": "ACA"
        },
        {
          "name": "Osmosis",
          "slug": "osmosis",
          "ticker": "OSMO"
        },
        {
          "name": "BitTorrent",
          "slug": "bittorrent",
          "ticker": "BTTOLD"
        },
        {
          "name": "Moonray",
          "slug": "moonray",
          "ticker": "MNRY"
        },
        {
          "name": "World Mobile Token",
          "slug": "world-mobile-token",
          "ticker": "WMTX"
        },
        {
          "name": "STARSHIP",
          "slug": "starship",
          "ticker": "STARSHIP"
        },
        {
          "name": "iMe Lab",
          "slug": "ime-lab",
          "ticker": "LIME"
        },
        {
          "name": "Polygon [on Ethereum]",
          "slug": "matic-network",
          "ticker": "MATIC"
        },
        {
          "name": "pSTAKE Finance",
          "slug": "pstake-finance",
          "ticker": "PSTAKE"
        },
        {
          "name": "Chitan",
          "slug": "chitan",
          "ticker": "CHITAN"
        },
        {
          "name": "TON Station",
          "slug": "tonstation",
          "ticker": "SOON"
        },
        {
          "name": "Micro GPT",
          "slug": "micro-gpt",
          "ticker": "$MICRO"
        },
        {
          "name": "sETH2",
          "slug": "seth2",
          "ticker": "SETH2"
        },
        {
          "name": "JPool Staked SOL (JSOL)",
          "slug": "jpool",
          "ticker": "JSOL"
        },
        {
          "name": "UnFederalReserve",
          "slug": "unfederalreserve",
          "ticker": "eRSDL"
        },
        {
          "name": "Moby",
          "slug": "moby",
          "ticker": "MOBY"
        },
        {
          "name": "Somnium Space Cubes",
          "slug": "somnium-space-cubes",
          "ticker": "CUBE"
        },
        {
          "name": "RYO Coin",
          "slug": "ryo-coin",
          "ticker": "RYO"
        },
        {
          "name": "Highstreet",
          "slug": "highstreet",
          "ticker": "HIGH"
        },
        {
          "name": "YAM V2",
          "slug": "yam-v2",
          "ticker": "YAMV2"
        },
        {
          "name": "Findora",
          "slug": "findora",
          "ticker": "FRA"
        },
        {
          "name": "mETH Protocol",
          "slug": "meth-protocol",
          "ticker": "COOK"
        },
        {
          "name": "Bone ShibaSwap",
          "slug": "bone-shibaswap",
          "ticker": "BONE"
        },
        {
          "name": "Astar",
          "slug": "astar",
          "ticker": "ASTR"
        },
        {
          "name": "Finanx AI",
          "slug": "finanx-ai",
          "ticker": "FNXAI"
        },
        {
          "name": "XCAD Network",
          "slug": "xcad-network",
          "ticker": "XCAD"
        },
        {
          "name": "TRUMP DOGS",
          "slug": "trump-dogs",
          "ticker": "DOGS"
        },
        {
          "name": "Venus DOT",
          "slug": "venus-dot",
          "ticker": "vDOT"
        },
        {
          "name": "Zano",
          "slug": "zano",
          "ticker": "ZANO"
        },
        {
          "name": "Aston Villa Fan Token",
          "slug": "aston-villa-fan-token",
          "ticker": "AVL"
        },
        {
          "name": "Mirrored iShares Silver Trust",
          "slug": "mirrored-ishares-silver-trust",
          "ticker": "mSLV"
        },
        {
          "name": "Mirrored Twitter",
          "slug": "mirrored-twitter",
          "ticker": "mTWTR"
        },
        {
          "name": "Mirrored United States Oil Fund",
          "slug": "mirrored-united-states-oil-fund",
          "ticker": "mUSO"
        },
        {
          "name": "Wrapped Velas",
          "slug": "wrapped-velas",
          "ticker": "WVLX"
        },
        {
          "name": "Victoria VR",
          "slug": "victoria-vr",
          "ticker": "VR"
        },
        {
          "name": "DinoSwap",
          "slug": "dinoswap",
          "ticker": "DINO"
        },
        {
          "name": "Baby Pengu",
          "slug": "baby-pengu",
          "ticker": "BABYPENGU"
        },
        {
          "name": "Maverick Protocol",
          "slug": "maverick-protocol",
          "ticker": "MAV"
        },
        {
          "name": "Conceal",
          "slug": "conceal",
          "ticker": "CCX"
        },
        {
          "name": "Retreeb",
          "slug": "retreeb",
          "ticker": "TREEB"
        },
        {
          "name": "BOOK OF MEME",
          "slug": "book-of-meme",
          "ticker": "BOME"
        },
        {
          "name": "Threshold",
          "slug": "threshold",
          "ticker": "T"
        },
        {
          "name": "Crabada",
          "slug": "a-crabada",
          "ticker": "CRA"
        },
        {
          "name": "DOLA",
          "slug": "inverse-finance-dola-stablecoin",
          "ticker": "DOLA"
        },
        {
          "name": "MEVerse",
          "slug": "meverse",
          "ticker": "MEV"
        },
        {
          "name": "Benqi",
          "slug": "a-benqi",
          "ticker": "QI"
        },
        {
          "name": "ADreward",
          "slug": "adreward",
          "ticker": "AD"
        },
        {
          "name": "ALEX Lab",
          "slug": "alex-lab",
          "ticker": "ALEX"
        },
        {
          "name": "PEPE MAGA",
          "slug": "pepemaga-me",
          "ticker": "MAGA"
        },
        {
          "name": "Turbo On Base",
          "slug": "turbo-on-base",
          "ticker": "TURBO"
        },
        {
          "name": "DeFi Yield Protocol",
          "slug": "defi-yield-protocol",
          "ticker": "DYP"
        },
        {
          "name": "Quartz",
          "slug": "sandclock",
          "ticker": "QUARTZ"
        },
        {
          "name": "Position Exchange",
          "slug": "position-exchange",
          "ticker": "POSI"
        },
        {
          "name": "Ankr [on Ethereum]",
          "slug": "ankr",
          "ticker": "ANKR"
        },
        {
          "name": "Cronos",
          "slug": "crypto-com-coin",
          "ticker": "CRO"
        },
        {
          "name": "Patriot on Base",
          "slug": "patriot-on-base",
          "ticker": "PATRIOT"
        },
        {
          "name": "Alpine F1 Team Fan Token",
          "slug": "alpine-f1-team-fan-token",
          "ticker": "ALPINE"
        },
        {
          "name": "Florin",
          "slug": "florin",
          "ticker": "XFL"
        },
        {
          "name": "Atlas Navi",
          "slug": "atlas-navi",
          "ticker": "NAVI"
        },
        {
          "name": "Counos X",
          "slug": "counos-x",
          "ticker": "CCXX"
        },
        {
          "name": "Thetan Arena",
          "slug": "thetan-arena",
          "ticker": "THG"
        },
        {
          "name": "Adappter Token",
          "slug": "adappter-token",
          "ticker": "ADP"
        },
        {
          "name": "SHILL Token",
          "slug": "project-seed",
          "ticker": "SHILL"
        },
        {
          "name": "EscoinToken",
          "slug": "escointoken",
          "ticker": "ELG"
        },
        {
          "name": "Wrapped TRON",
          "slug": "wrapped-tron",
          "ticker": "WTRX"
        },
        {
          "name": "BitTorrent (new)",
          "slug": "bittorrent-new",
          "ticker": "BTT"
        },
        {
          "name": "MetaPets",
          "slug": "metapets",
          "ticker": "METAPETS"
        },
        {
          "name": "S.S. Lazio Fan Token",
          "slug": "lazio-fan-token",
          "ticker": "LAZIO"
        },
        {
          "name": "MContent",
          "slug": "mcontent",
          "ticker": "MCONTENT"
        },
        {
          "name": "Etherisc",
          "slug": "etherisc",
          "ticker": "DIP"
        },
        {
          "name": "SUI Desci Agents",
          "slug": "sui-desci-agents",
          "ticker": "DESCI"
        },
        {
          "name": "EYWA",
          "slug": "eywa",
          "ticker": "EYWA"
        },
        {
          "name": "ALYATTES",
          "slug": "alyattes",
          "ticker": "ALYA"
        },
        {
          "name": "DeGate",
          "slug": "degate",
          "ticker": "DG"
        },
        {
          "name": "Biconomy Exchange Token",
          "slug": "biconomy-token",
          "ticker": "BIT"
        },
        {
          "name": "PAWSWAP",
          "slug": "pawswap",
          "ticker": "PAW"
        },
        {
          "name": "Goldfinch",
          "slug": "goldfinch-protocol",
          "ticker": "GFI"
        },
        {
          "name": "ETHA Lend",
          "slug": "etha-lend",
          "ticker": "ETHA"
        },
        {
          "name": "NYM",
          "slug": "nym",
          "ticker": "NYM"
        },
        {
          "name": "Revolve Games",
          "slug": "revolve-games",
          "ticker": "RPG"
        },
        {
          "name": "Satoshi AI agent by Virtuals",
          "slug": "satoshi-ai-agent-by-virtuals",
          "ticker": "SAINT"
        },
        {
          "name": "1eco",
          "slug": "1eco",
          "ticker": "1ECO"
        },
        {
          "name": "Wrapped Cardano",
          "slug": "wrapped-cardano",
          "ticker": "WADA"
        },
        {
          "name": "HI",
          "slug": "hi-dollar",
          "ticker": "HI"
        },
        {
          "name": "Cardano",
          "slug": "cardano",
          "ticker": "ADA"
        },
        {
          "name": "GuildFi",
          "slug": "guildfi",
          "ticker": "GF"
        },
        {
          "name": "EverRise",
          "slug": "everrise",
          "ticker": "RISE"
        },
        {
          "name": "Ispolink",
          "slug": "ispolink",
          "ticker": "ISP"
        },
        {
          "name": "Hydra",
          "slug": "hydra",
          "ticker": "HYDRA"
        },
        {
          "name": "Aurora",
          "slug": "aurora-near",
          "ticker": "AURORA"
        },
        {
          "name": "Toncoin",
          "slug": "toncoin",
          "ticker": "TON"
        },
        {
          "name": "Polkacity",
          "slug": "polkacity",
          "ticker": "POLC"
        },
        {
          "name": "Hillstone Finance",
          "slug": "hillstone",
          "ticker": "HSF"
        },
        {
          "name": "GensoKishi Metaverse",
          "slug": "gensokishis-metaverse",
          "ticker": "MV"
        },
        {
          "name": "Meson Network",
          "slug": "meson-network",
          "ticker": "MSN"
        },
        {
          "name": "Trabzonspor Fan Token",
          "slug": "trabzonspor-fan-token",
          "ticker": "TRA"
        },
        {
          "name": "Haedal Staked SUI",
          "slug": "haedal-staked-sui",
          "ticker": "HASUI"
        },
        {
          "name": "TRIO (OrdinalsBot)",
          "slug": "ordinalsbot",
          "ticker": "TRIO"
        },
        {
          "name": "Lista DAO",
          "slug": "lista-dao",
          "ticker": "LISTA"
        },
        {
          "name": "Black Phoenix",
          "slug": "black-phoenix",
          "ticker": "BPX"
        },
        {
          "name": "DOGE on Solana",
          "slug": "doge-on-solana",
          "ticker": "SDOGE"
        },
        {
          "name": "Formation Fi",
          "slug": "formation-fi",
          "ticker": "FORM"
        },
        {
          "name": "Vader Protocol",
          "slug": "vader-protocol",
          "ticker": "VADER"
        },
        {
          "name": "League of Kingdoms Arena",
          "slug": "league-of-kingdoms",
          "ticker": "LOKA"
        },
        {
          "name": "ApeCoin",
          "slug": "apecoin-ape",
          "ticker": "APE"
        },
        {
          "name": "ThunderCore",
          "slug": "thundercore",
          "ticker": "TT"
        },
        {
          "name": "Mystery On Base",
          "slug": "mystery-on-base",
          "ticker": "MYSTERY"
        },
        {
          "name": "Ozone Metaverse",
          "slug": "ozone-metaverse",
          "ticker": "OZONE"
        },
        {
          "name": "Fruits",
          "slug": "fruits-eco",
          "ticker": "FRTS"
        },
        {
          "name": "TRON",
          "slug": "tron",
          "ticker": "TRX"
        },
        {
          "name": "Aavegotchi",
          "slug": "aavegotchi-ghst-token",
          "ticker": "GHST"
        },
        {
          "name": "Gains Network [on Arbitrum]",
          "slug": "arb-gains-network",
          "ticker": "GNS"
        },
        {
          "name": "Genopets",
          "slug": "genopets",
          "ticker": "GENE"
        },
        {
          "name": "ApolloX",
          "slug": "apollox",
          "ticker": "APX"
        },
        {
          "name": "Stargate Finance [on Ethereum]",
          "slug": "stargate-finance",
          "ticker": "STG"
        },
        {
          "name": "Theta Network",
          "slug": "theta",
          "ticker": "THETA"
        },
        {
          "name": "Freya by Virtuals",
          "slug": "freya-by-virtuals",
          "ticker": "FREYA"
        },
        {
          "name": "ZENZO",
          "slug": "zenzo",
          "ticker": "ZNZ"
        },
        {
          "name": "BNB",
          "slug": "binance-coin",
          "ticker": "BNB"
        },
        {
          "name": "SLERF",
          "slug": "slerf",
          "ticker": "SLERF"
        },
        {
          "name": "Pikaboss",
          "slug": "pikachu",
          "ticker": "PIKA"
        },
        {
          "name": "Ronin",
          "slug": "ronin",
          "ticker": "RON"
        },
        {
          "name": "Aventus",
          "slug": "aventus",
          "ticker": "AVT"
        },
        {
          "name": "Wrapped Fantom",
          "slug": "wrapped-fantom",
          "ticker": "WFTM"
        },
        {
          "name": "USDD [on Ethereum]",
          "slug": "usdd",
          "ticker": "USDD"
        },
        {
          "name": "MARBLEX",
          "slug": "marblex",
          "ticker": "MBX"
        },
        {
          "name": "WETH [ERC20]",
          "slug": "weth",
          "ticker": "WETH"
        },
        {
          "name": "FIGHT TO MAGA",
          "slug": "fight-to-maga",
          "ticker": "FIGHT"
        },
        {
          "name": "Eigenpie mstETH",
          "slug": "eigenpie-msteth",
          "ticker": "MSTETH"
        },
        {
          "name": "MANTRA",
          "slug": "mantra-dao",
          "ticker": "OM"
        },
        {
          "name": "PAC Protocol",
          "slug": "paccoin",
          "ticker": "PAC"
        },
        {
          "name": "MAGA",
          "slug": "maga",
          "ticker": "TRUMP"
        },
        {
          "name": "OctaSpace",
          "slug": "octaspace",
          "ticker": "OCTA"
        },
        {
          "name": "IAGON",
          "slug": "iagon",
          "ticker": "IAG"
        },
        {
          "name": "Perpetual Protocol [on Ethereum]",
          "slug": "perpetual-protocol",
          "ticker": "PERP"
        },
        {
          "name": "RAI Finance",
          "slug": "rai-finance-sofi",
          "ticker": "SOFI"
        },
        {
          "name": "mStable USD",
          "slug": "mstable-usd",
          "ticker": "MUSD"
        },
        {
          "name": "Shadow Token",
          "slug": "genesysgo-shadow",
          "ticker": "SHDW"
        },
        {
          "name": "Green Satoshi Token [on Solana]",
          "slug": "green-satoshi-token",
          "ticker": "GST"
        },
        {
          "name": "TerraUSD",
          "slug": "terrausd",
          "ticker": "USTC"
        },
        {
          "name": "NXM",
          "slug": "nexus-mutual",
          "ticker": "NXM"
        },
        {
          "name": "Veno Finance",
          "slug": "veno-finance-vno",
          "ticker": "VNO"
        },
        {
          "name": "Terra Classic",
          "slug": "luna",
          "ticker": "LUNC"
        },
        {
          "name": "Bitcoin Wizards",
          "slug": "bitcoin-wizards",
          "ticker": "WZRD"
        },
        {
          "name": "Polymesh",
          "slug": "polymesh",
          "ticker": "POLYX"
        },
        {
          "name": "Rug World Assets",
          "slug": "rug-world-assets",
          "ticker": "RWA"
        },
        {
          "name": "Terra",
          "slug": "terra-luna-v2",
          "ticker": "LUNA"
        },
        {
          "name": "Streamflow",
          "slug": "streamflow",
          "ticker": "STREAM"
        },
        {
          "name": "WETH [on Polygon]",
          "slug": "p-weth",
          "ticker": "WETH"
        },
        {
          "name": "Lido DAO Token",
          "slug": "lido-dao",
          "ticker": "LDO"
        },
        {
          "name": "QuickSwap [on Polygon]",
          "slug": "p-quickswap",
          "ticker": "QUICK"
        },
        {
          "name": "Propchain",
          "slug": "propchain",
          "ticker": "PROPC"
        },
        {
          "name": "Pangolin",
          "slug": "a-pangolin",
          "ticker": "PNG"
        },
        {
          "name": "Tether [on Arbitrum]",
          "slug": "arb-tether",
          "ticker": "USDT"
        },
        {
          "name": "ZKFair",
          "slug": "zkfair",
          "ticker": "ZKF"
        },
        {
          "name": "WETH [on Arbitrum]",
          "slug": "arb-weth",
          "ticker": "WETH"
        },
        {
          "name": "Synapse [on Ethereum]",
          "slug": "synapse-2",
          "ticker": "SYN"
        },
        {
          "name": "USD Coin [on Arbitrum]",
          "slug": "arb-usd-coin",
          "ticker": "USDC"
        },
        {
          "name": "Bitcoin EDenRich",
          "slug": "bitcoin-edenrich",
          "ticker": "BITBEDR"
        },
        {
          "name": "pippin",
          "slug": "pippin",
          "ticker": "PIPPIN"
        },
        {
          "name": "Bio Protocol",
          "slug": "bio",
          "ticker": "BIO"
        },
        {
          "name": "Cryptex Finance",
          "slug": "cryptex-finance",
          "ticker": "CTX"
        },
        {
          "name": "Cult DAO",
          "slug": "cult-dao",
          "ticker": "CULT"
        },
        {
          "name": "TiFi Token",
          "slug": "tifi-token",
          "ticker": "TIFI"
        },
        {
          "name": "Stader",
          "slug": "stader",
          "ticker": "SD"
        },
        {
          "name": "Wrapped Everscale",
          "slug": "wrapped-everscale",
          "ticker": "WEVER"
        },
        {
          "name": "Shentu",
          "slug": "shentu",
          "ticker": "CTK"
        },
        {
          "name": "Pip",
          "slug": "pip",
          "ticker": "PIP"
        },
        {
          "name": "Forta",
          "slug": "forta",
          "ticker": "FORT"
        },
        {
          "name": "Celo Euro",
          "slug": "celo-euro",
          "ticker": "CEUR"
        },
        {
          "name": "BurgerCities",
          "slug": "burger-cities",
          "ticker": "BURGER"
        },
        {
          "name": "AnimalGo",
          "slug": "animalgo",
          "ticker": "GOM2"
        },
        {
          "name": "Pendle [on Ethereum]",
          "slug": "pendle",
          "ticker": "PENDLE"
        },
        {
          "name": "bemo staked TON",
          "slug": "bemo-staked-ton",
          "ticker": "stTON"
        },
        {
          "name": "Machine Xchange Coin",
          "slug": "machine-xchange-coin",
          "ticker": "MXC"
        },
        {
          "name": "Moonwell",
          "slug": "moonwell-artemis",
          "ticker": "WELL"
        },
        {
          "name": "Curve [on Ethereum]",
          "slug": "curve",
          "ticker": "CRV"
        },
        {
          "name": "Galxe",
          "slug": "project-galaxy",
          "ticker": "GAL"
        },
        {
          "name": "ParallelAI",
          "slug": "parallelai",
          "ticker": "PAI"
        },
        {
          "name": "Node AI",
          "slug": "node-ai",
          "ticker": "GPU"
        },
        {
          "name": "Smart Layer Network",
          "slug": "smart-layer-network",
          "ticker": "SLN"
        },
        {
          "name": "Lumerin",
          "slug": "lumerin",
          "ticker": "LMR"
        },
        {
          "name": "Defigram",
          "slug": "defigram",
          "ticker": "DFG"
        },
        {
          "name": "Firo",
          "slug": "zcoin",
          "ticker": "XZC"
        },
        {
          "name": "HarryPotterObamaSonic10Inu 2.0",
          "slug": "harrypotterobamasonic10inu-2",
          "ticker": "BITCOIN"
        },
        {
          "name": "Immutable X",
          "slug": "immutable-x",
          "ticker": "IMX"
        },
        {
          "name": "ROGin AI",
          "slug": "rogin-ai",
          "ticker": "ROG"
        },
        {
          "name": "Radiant Capital",
          "slug": "arb-radiant-capital",
          "ticker": "RDNT"
        },
        {
          "name": "Dopex",
          "slug": "arb-dopex",
          "ticker": "DPX"
        },
        {
          "name": "Tether [on Optimism]",
          "slug": "o-tether",
          "ticker": "USDT"
        },
        {
          "name": "Dai [on Arbitrum]",
          "slug": "arb-multi-collateral-dai",
          "ticker": "DAI"
        },
        {
          "name": "Top Hat",
          "slug": "top-hat",
          "ticker": "HAT"
        },
        {
          "name": "DOGEai",
          "slug": "dogeai",
          "ticker": "DOGEAI"
        },
        {
          "name": "API3",
          "slug": "api3",
          "ticker": "API3"
        },
        {
          "name": "Dacxi",
          "slug": "dacxi",
          "ticker": "DXI"
        },
        {
          "name": "DOGMI",
          "slug": "dogmi",
          "ticker": "DOGMI"
        },
        {
          "name": "Uno Re",
          "slug": "unore",
          "ticker": "UNO"
        },
        {
          "name": "Digital Financial Exchange",
          "slug": "digital-financial-exchange",
          "ticker": "DIFX"
        },
        {
          "name": "Uniswap [on Ethereum]",
          "slug": "uniswap",
          "ticker": "UNI"
        },
        {
          "name": "LETSTOP",
          "slug": "letstop",
          "ticker": "STOP"
        },
        {
          "name": "Synthetix [on Optimism]",
          "slug": "o-synthetix-network-token",
          "ticker": "SNX"
        },
        {
          "name": "Curve [on Arbitrum]",
          "slug": "arb-curve",
          "ticker": "CRV"
        },
        {
          "name": "Rain Coin",
          "slug": "rain-coin",
          "ticker": "RAIN"
        },
        {
          "name": "Sipher",
          "slug": "sipher",
          "ticker": "SIPHER"
        },
        {
          "name": "Streamit Coin",
          "slug": "streamit-coin",
          "ticker": "STREAM"
        },
        {
          "name": "YAM V1",
          "slug": "yamv3",
          "ticker": "YAM"
        },
        {
          "name": "Staked TRX",
          "slug": "staked-trx",
          "ticker": "STRX"
        },
        {
          "name": "Redbelly Network",
          "slug": "redbelly-network",
          "ticker": "RBNT"
        },
        {
          "name": "deBridge",
          "slug": "debridge",
          "ticker": "DBR"
        },
        {
          "name": "Beam",
          "slug": "onbeam",
          "ticker": "BEAM"
        },
        {
          "name": "Mint Marble",
          "slug": "mint-marble",
          "ticker": "MIM"
        },
        {
          "name": "Diamond Launch",
          "slug": "diamond-launch",
          "ticker": "DLC"
        },
        {
          "name": "PegNet",
          "slug": "pegnet",
          "ticker": "CHF"
        },
        {
          "name": "Kaspa",
          "slug": "kaspa",
          "ticker": "KAS"
        },
        {
          "name": "LayerZero [on Optimism]",
          "slug": "o-layerzero",
          "ticker": "ZRO"
        },
        {
          "name": "SpiritSwap",
          "slug": "spiritswap",
          "ticker": "SPIRIT"
        },
        {
          "name": "Tether [on Polygon]",
          "slug": "p-tether",
          "ticker": "USDT"
        },
        {
          "name": "EthereumPoW",
          "slug": "ethereum-pow",
          "ticker": "ETHW"
        },
        {
          "name": "StablR USD",
          "slug": "stablr-usd",
          "ticker": "USDR"
        },
        {
          "name": "BizAuto",
          "slug": "bizauto",
          "ticker": "BIZA"
        },
        {
          "name": "Yfi.mobi",
          "slug": "yfi-mobi",
          "ticker": "YFIM"
        },
        {
          "name": "Terran Coin",
          "slug": "terran-coin",
          "ticker": "TRR"
        },
        {
          "name": "Hashflow",
          "slug": "hashflow",
          "ticker": "HFT"
        },
        {
          "name": "Sweat Economy",
          "slug": "sweat-economy",
          "ticker": "SWEAT"
        },
        {
          "name": "Bitcicoin",
          "slug": "bitcicoin",
          "ticker": "BITCI"
        },
        {
          "name": "SushiSwap [on Arbitrum]",
          "slug": "arb-sushi",
          "ticker": "SUSHI"
        },
        {
          "name": "Vai",
          "slug": "vai",
          "ticker": "VAI"
        },
        {
          "name": "Everscale",
          "slug": "ton-crystal",
          "ticker": "EVER"
        },
        {
          "name": "Solidus Ai Tech",
          "slug": "bnb-solidus-ai-tech",
          "ticker": "AITECH"
        },
        {
          "name": "Evmos",
          "slug": "evmos",
          "ticker": "EVMOS"
        },
        {
          "name": "Shiro Neko",
          "slug": "shiro-neko",
          "ticker": "SHIRO"
        },
        {
          "name": "Squid Game (squidgame.top)",
          "slug": "squid-game-top",
          "ticker": "SQUID"
        },
        {
          "name": "Guild of Guardians",
          "slug": "guild-of-guardians",
          "ticker": "GOG"
        },
        {
          "name": "STFX",
          "slug": "stfx",
          "ticker": "STFX"
        },
        {
          "name": "Argentine Football Association Fan Token",
          "slug": "argentinefootballassociationfantoken",
          "ticker": "ARG"
        },
        {
          "name": "Captain Tsubasa",
          "slug": "captain-tsubasa",
          "ticker": "TSUGT"
        },
        {
          "name": "Marinade",
          "slug": "mnde",
          "ticker": "MNDE"
        },
        {
          "name": "Neurai",
          "slug": "neurai",
          "ticker": "XNA"
        },
        {
          "name": "Rainicorn",
          "slug": "rainicorn",
          "ticker": "RAINI"
        },
        {
          "name": "EFFORCE",
          "slug": "efforce",
          "ticker": "WOZX"
        },
        {
          "name": "Axelar",
          "slug": "axelar",
          "ticker": "AXL"
        },
        {
          "name": "GateToken",
          "slug": "gatechain-token",
          "ticker": "GT"
        },
        {
          "name": "Braintrust",
          "slug": "BTRST",
          "ticker": "BTRST"
        },
        {
          "name": "Redd",
          "slug": "reddcoin",
          "ticker": "RDD"
        },
        {
          "name": "Chrono.tech",
          "slug": "chronotech",
          "ticker": "TIME"
        },
        {
          "name": "BOB",
          "slug": "bob-fun",
          "ticker": "BOB"
        },
        {
          "name": "ReflectionAI",
          "slug": "reflectionai",
          "ticker": "RECT"
        },
        {
          "name": "Orbcity",
          "slug": "orbcity",
          "ticker": "ORB"
        },
        {
          "name": "Octopus Network",
          "slug": "octopus-network",
          "ticker": "OCT"
        },
        {
          "name": "ONBUFF",
          "slug": "onbuff",
          "ticker": "ONIT"
        },
        {
          "name": "ReapChain",
          "slug": "reapchain",
          "ticker": "REAP"
        },
        {
          "name": "SafePal",
          "slug": "bnb-safepal",
          "ticker": "SFP"
        },
        {
          "name": "Artyfact",
          "slug": "bnb-artyfact",
          "ticker": "ARTY"
        },
        {
          "name": "Nsure.Network",
          "slug": "nsure-network",
          "ticker": "NSURE"
        },
        {
          "name": "Apeiron",
          "slug": "apeiron",
          "ticker": "APRS"
        },
        {
          "name": "Evan",
          "slug": "evanthehobo-com",
          "ticker": "EVAN"
        },
        {
          "name": "Komet",
          "slug": "komet",
          "ticker": "KOMET"
        },
        {
          "name": "LeverFi",
          "slug": "lever",
          "ticker": "LEVER"
        },
        {
          "name": "Prom",
          "slug": "prom",
          "ticker": "PROM"
        },
        {
          "name": "Gains Network [on Polygon]",
          "slug": "p-gains-network",
          "ticker": "GNS"
        },
        {
          "name": "Emirex Token",
          "slug": "emirex",
          "ticker": "EMRX"
        },
        {
          "name": "USD Coin [on BNB]",
          "slug": "bnb-usd-coin",
          "ticker": "USDC"
        },
        {
          "name": "The Force Protocol",
          "slug": "forceprotocol",
          "ticker": "FOR"
        },
        {
          "name": "COCOCOIN",
          "slug": "cococoin-bsc",
          "ticker": "COCO"
        },
        {
          "name": "Bluefin",
          "slug": "bluefin",
          "ticker": "BLUE"
        },
        {
          "name": "AstraAI",
          "slug": "astraai",
          "ticker": "ASTRA"
        },
        {
          "name": "Solar [on BNB]",
          "slug": "bnb-swipe",
          "ticker": "SXP"
        },
        {
          "name": "SushiSwap [on BNB]",
          "slug": "bnb-sushi",
          "ticker": "SUSHI"
        },
        {
          "name": "USD Coin [on Polygon]",
          "slug": "p-usd-coin",
          "ticker": "USDC"
        },
        {
          "name": "Trust Wallet Token",
          "slug": "bnb-trust-wallet-token",
          "ticker": "TWT"
        },
        {
          "name": "Flamengo Fan Token",
          "slug": "flamengo-fan-token",
          "ticker": "MENGO"
        },
        {
          "name": "Chromia [on BNB]",
          "slug": "bnb-chromia",
          "ticker": "CHR"
        },
        {
          "name": "Shido Network",
          "slug": "shido-network",
          "ticker": "SHIDO"
        },
        {
          "name": "WeBuy",
          "slug": "webuy",
          "ticker": "WE"
        },
        {
          "name": "Sui Name Service",
          "slug": "sui-name-service",
          "ticker": "NS"
        },
        {
          "name": "xCrypt Token",
          "slug": "xcrypt",
          "ticker": "XCT"
        },
        {
          "name": "Bad Idea AI",
          "slug": "bad-idea-ai",
          "ticker": "BAD"
        },
        {
          "name": "ProBit Token",
          "slug": "probit",
          "ticker": "PROB"
        },
        {
          "name": "INT Chain",
          "slug": "internet-node-token",
          "ticker": "INT"
        },
        {
          "name": "RabbitX",
          "slug": "rabbitx",
          "ticker": "RBX"
        },
        {
          "name": "PLAYA3ULL GAMES",
          "slug": "playa3ull-games-new",
          "ticker": "3ULL"
        },
        {
          "name": "Bazaars",
          "slug": "bazaars",
          "ticker": "BZR"
        },
        {
          "name": "Saitama",
          "slug": "saitama-inu-new",
          "ticker": "STC"
        },
        {
          "name": "Anchored Coins AEUR",
          "slug": "anchored-coins-aeur",
          "ticker": "AEUR"
        },
        {
          "name": "Coreum",
          "slug": "coreum",
          "ticker": "COREUM"
        },
        {
          "name": "Enzyme",
          "slug": "melon",
          "ticker": "MLN"
        },
        {
          "name": "FUNToken",
          "slug": "funfair",
          "ticker": "FUN"
        },
        {
          "name": "Uniswap V2: LINK",
          "slug": "uniswap_link_eth_lp",
          "ticker": "UNI-V2 LINK/ETH LP"
        },
        {
          "name": "Uniswap V2: REN",
          "slug": "uniswap_ren_eth_lp",
          "ticker": "UNI-V2 REN/ETH LP"
        },
        {
          "name": "Gelato Uniswap USDC/USDT LP",
          "slug": "gelato_uniswap_usdc_usdt_lp",
          "ticker": "G-UNI USDC/USDT LP"
        },
        {
          "name": "Blue Whale EXchange",
          "slug": "blue-whale-token",
          "ticker": "BWX"
        },
        {
          "name": "Filecoin",
          "slug": "file-coin",
          "ticker": "FIL"
        },
        {
          "name": "Muse",
          "slug": "muse",
          "ticker": "MUSE"
        },
        {
          "name": "Wrapped TAO",
          "slug": "wrapped-tao",
          "ticker": "WTAO"
        },
        {
          "name": "Super Zero Protocol",
          "slug": "super-zero",
          "ticker": "SERO"
        },
        {
          "name": "Blur",
          "slug": "blur-token",
          "ticker": "BLUR"
        },
        {
          "name": "BOBO",
          "slug": "bobo-coin",
          "ticker": "BOBO"
        },
        {
          "name": "Gelato Uniswap DAI/USDC LP",
          "slug": "gelato_uniswap_dai_usdc_lp",
          "ticker": "G-UNI DAI/USDC LP"
        },
        {
          "name": "Uniswap V2: CRV",
          "slug": "uniswap_crv_eth_lp",
          "ticker": "UNI-V2 CRV/ETH LP"
        },
        {
          "name": "Uniswap V2: MKR",
          "slug": "uniswap_mkr_eth_lp",
          "ticker": "UNI-V2 MKR/ETH LP"
        },
        {
          "name": "Hive",
          "slug": "hive",
          "ticker": "HIVE"
        },
        {
          "name": "Grid+",
          "slug": "gridplus",
          "ticker": "GRID"
        },
        {
          "name": "Syntropy",
          "slug": "noia-network",
          "ticker": "NOIA"
        },
        {
          "name": "Stone DeFi",
          "slug": "stone",
          "ticker": "STN"
        },
        {
          "name": "Optimism",
          "slug": "o-optimism",
          "ticker": "OP"
        },
        {
          "name": "DIA",
          "slug": "dia-data",
          "ticker": "DIA"
        },
        {
          "name": "RSK Smart Bitcoin",
          "slug": "smart-bitcoin",
          "ticker": "RBTC"
        },
        {
          "name": "Tron Bull",
          "slug": "tron-bull-tbull",
          "ticker": "TBULL"
        },
        {
          "name": "Shoggoth (shoggoth.monster)",
          "slug": "shoggoth-monster",
          "ticker": "SHOGGOTH"
        },
        {
          "name": "Welshcorgicoin",
          "slug": "welshcorgicoin",
          "ticker": "WELSH"
        },
        {
          "name": "RWAX",
          "slug": "moonapp",
          "ticker": "APP"
        },
        {
          "name": "Wet Ass Pussy",
          "slug": "wet-ass-pussy",
          "ticker": "WAP"
        },
        {
          "name": "Euro Coin",
          "slug": "euro-coin",
          "ticker": "EURC"
        },
        {
          "name": "USD Coin [on Optimism]",
          "slug": "o-usd-coin",
          "ticker": "USDC"
        },
        {
          "name": "Aleph Zero",
          "slug": "aleph-zero",
          "ticker": "AZERO"
        },
        {
          "name": "RSK Infrastructure Framework",
          "slug": "rif-token",
          "ticker": "RIF"
        },
        {
          "name": "Dai [on Ethereum]",
          "slug": "multi-collateral-dai",
          "ticker": "DAI"
        },
        {
          "name": "ChainLink [on Optimism]",
          "slug": "o-chainlink",
          "ticker": "LINK"
        },
        {
          "name": "Elon Trump Fart",
          "slug": "elon-trump-fart",
          "ticker": "ETF500"
        },
        {
          "name": "Stride",
          "slug": "stride",
          "ticker": "STRD"
        },
        {
          "name": "Gold DAO",
          "slug": "gold-dao",
          "ticker": "GOLDAO"
        },
        {
          "name": "Glacier Network",
          "slug": "glacier-network",
          "ticker": "GLS"
        },
        {
          "name": "Major Frog",
          "slug": "major-frog",
          "ticker": "MAJOR"
        },
        {
          "name": "Akuma Inu",
          "slug": "akuma-inu",
          "ticker": "$AKUMA"
        },
        {
          "name": "Frax Share",
          "slug": "frax-share",
          "ticker": "FXS"
        },
        {
          "name": "Onyxcoin",
          "slug": "chain",
          "ticker": "XCN"
        },
        {
          "name": "Hifi Finance (old)",
          "slug": "mainframe",
          "ticker": "MFT"
        },
        {
          "name": "Axie Infinity [on Ethereum]",
          "slug": "axie-infinity",
          "ticker": "AXS"
        },
        {
          "name": "Purple Pepe",
          "slug": "purple-pepe",
          "ticker": "$PURPE"
        },
        {
          "name": "Windoge98",
          "slug": "windoge98",
          "ticker": "EXE"
        },
        {
          "name": "BOOM DAO",
          "slug": "boom-dao",
          "ticker": "BOOM"
        },
        {
          "name": "Free Palestine",
          "slug": "free-palestine",
          "ticker": "YAFA"
        },
        {
          "name": "Eliza (elizawakesup)",
          "slug": "eliza-wakesup-ai",
          "ticker": "ELIZA"
        },
        {
          "name": "SynFutures",
          "slug": "synfutures",
          "ticker": "F"
        },
        {
          "name": "WUFFI",
          "slug": "wuffi",
          "ticker": "WUF"
        },
        {
          "name": "Bitget Token",
          "slug": "bitget-token-new",
          "ticker": "BGB"
        },
        {
          "name": "Concordium",
          "slug": "concordium",
          "ticker": "CCD"
        },
        {
          "name": "Dejitaru Tsuka",
          "slug": "dejitaru-tsuka",
          "ticker": "TSUKA"
        },
        {
          "name": "Hifi Finance",
          "slug": "hifi-finance-new",
          "ticker": "HIFI"
        },
        {
          "name": "Revolution Populi",
          "slug": "revolution-populi",
          "ticker": "RVP"
        },
        {
          "name": "Blockburn",
          "slug": "blockburn",
          "ticker": "BURN"
        },
        {
          "name": "Grove Coin",
          "slug": "grove3",
          "ticker": "GRV"
        },
        {
          "name": "Core",
          "slug": "core-dao",
          "ticker": "CORE"
        },
        {
          "name": "Hot or Not",
          "slug": "hotornot",
          "ticker": "HOT"
        },
        {
          "name": "Flare",
          "slug": "flare",
          "ticker": "FLR"
        },
        {
          "name": "Rekt (rektcoin.com)",
          "slug": "rekt-eth",
          "ticker": "REKT"
        },
        {
          "name": "IPVERSE",
          "slug": "ipverse-eth",
          "ticker": "IPV"
        },
        {
          "name": "MyNeighborAlice [on Ethereum]",
          "slug": "myneighboralice",
          "ticker": "ALICE"
        },
        {
          "name": "Magic Internet Money [on Avalanche]",
          "slug": "a-magic-internet-money",
          "ticker": "MIM"
        },
        {
          "name": "r/CryptoCurrency Moons",
          "slug": "moon",
          "ticker": "MOON"
        },
        {
          "name": "WAGMI Games",
          "slug": "wagmi-game-2",
          "ticker": "WAGMIGAMES"
        },
        {
          "name": "Wagie Bot",
          "slug": "wagie-bot",
          "ticker": "WAGIEBOT"
        },
        {
          "name": "Roobee",
          "slug": "roobee",
          "ticker": "ROOBEE"
        },
        {
          "name": "SwissBorg",
          "slug": "swissborg",
          "ticker": "BORG"
        },
        {
          "name": "Bald",
          "slug": "bald",
          "ticker": "BALD"
        },
        {
          "name": "MemeFi",
          "slug": "meme-fi",
          "ticker": "MEMEFI"
        },
        {
          "name": "Rewardable",
          "slug": "rewardable",
          "ticker": "REWARD"
        },
        {
          "name": "BugsCoin",
          "slug": "bugscoin",
          "ticker": "BGSC"
        },
        {
          "name": "Nervos Network",
          "slug": "nervos-network",
          "ticker": "CKB"
        },
        {
          "name": "Sonic",
          "slug": "sonic-app",
          "ticker": "SONIC"
        },
        {
          "name": "Major",
          "slug": "major",
          "ticker": "MAJOR"
        },
        {
          "name": "UniBot",
          "slug": "unibot-eth",
          "ticker": "UNIBOT"
        },
        {
          "name": "Magic Internet Money [on Ethereum]",
          "slug": "magic-internet-money",
          "ticker": "MIM"
        },
        {
          "name": "Colony",
          "slug": "a-colony",
          "ticker": "CLY"
        },
        {
          "name": "OpenChat",
          "slug": "openchat",
          "ticker": "CHAT"
        },
        {
          "name": "Spike",
          "slug": "spike-on-sol",
          "ticker": "SPIKE"
        },
        {
          "name": "Stratis [New]",
          "slug": "stratis-new",
          "ticker": "STRAX"
        },
        {
          "name": "MetFi",
          "slug": "metfi2",
          "ticker": "METFI"
        },
        {
          "name": "HashAI",
          "slug": "hashai",
          "ticker": "HASHAI"
        },
        {
          "name": "SwarmNode.ai",
          "slug": "swarmnode-ai",
          "ticker": "SNAI"
        },
        {
          "name": "BFG Token",
          "slug": "betfury",
          "ticker": "BFG"
        },
        {
          "name": "Realio Network",
          "slug": "realio-network",
          "ticker": "RIO"
        },
        {
          "name": "AI Analysis Token",
          "slug": "ai-analysis-token",
          "ticker": "AIAT"
        },
        {
          "name": "Cere Network",
          "slug": "cere-network",
          "ticker": "CERE"
        },
        {
          "name": "PayPal USD",
          "slug": "paypal-usd",
          "ticker": "PYUSD"
        },
        {
          "name": "MoonBot",
          "slug": "moonbot",
          "ticker": "MBOT"
        },
        {
          "name": "COTI",
          "slug": "coti",
          "ticker": "COTI"
        },
        {
          "name": "Wrapped Bitcoin [on Ethereum]",
          "slug": "wrapped-bitcoin",
          "ticker": "WBTC"
        },
        {
          "name": "BIAO",
          "slug": "biaotoken",
          "ticker": "BIAO"
        },
        {
          "name": "Gaimin",
          "slug": "bnb-gaimin",
          "ticker": "GMRX"
        },
        {
          "name": "Gulden",
          "slug": "gulden",
          "ticker": "MUNT"
        },
        {
          "name": "LootBot",
          "slug": "lootbot",
          "ticker": "LOOT"
        },
        {
          "name": "Alon",
          "slug": "alon",
          "ticker": "ALON"
        },
        {
          "name": "cat in a dogs world",
          "slug": "mew",
          "ticker": "MEW"
        },
        {
          "name": "Orbler",
          "slug": "orbler",
          "ticker": "ORBR"
        },
        {
          "name": "Swarm Markets",
          "slug": "swarm-markets",
          "ticker": "SMT"
        },
        {
          "name": "JITO",
          "slug": "jito",
          "ticker": "JTO"
        },
        {
          "name": "Sei",
          "slug": "sei",
          "ticker": "SEI"
        },
        {
          "name": "crvUSD",
          "slug": "crvusd",
          "ticker": "CRVUSD"
        },
        {
          "name": "LUKSO",
          "slug": "lukso-network",
          "ticker": "LYX"
        },
        {
          "name": "Toshi",
          "slug": "toshithecat",
          "ticker": "TOSHI"
        },
        {
          "name": "Shrapnel",
          "slug": "shrapnel-com",
          "ticker": "SHRAP"
        },
        {
          "name": "Timeless",
          "slug": "timeless",
          "ticker": "LIT"
        },
        {
          "name": "Degen",
          "slug": "degen-base",
          "ticker": "DEGEN"
        },
        {
          "name": "OKC Token",
          "slug": "okt",
          "ticker": "OKT"
        },
        {
          "name": "Pendle [on Arbitrum]",
          "slug": "arb-pendle",
          "ticker": "PENDLE"
        },
        {
          "name": "GHO",
          "slug": "gho",
          "ticker": "GHO"
        },
        {
          "name": "HarryPotterObamaPacMan8Inu",
          "slug": "harrypotterobamapacman8inu",
          "ticker": "XRP"
        },
        {
          "name": "Landshare",
          "slug": "bnb-landshare",
          "ticker": "LAND"
        },
        {
          "name": "Strawberry AI",
          "slug": "strawberry-ai",
          "ticker": "BERRY"
        },
        {
          "name": "Echelon Prime",
          "slug": "echelon-prime",
          "ticker": "PRIME"
        },
        {
          "name": "Degen Spartan AI",
          "slug": "degen-spartan-ai",
          "ticker": "DEGENAI"
        },
        {
          "name": "Unification",
          "slug": "unification",
          "ticker": "FUND"
        },
        {
          "name": "Limitus",
          "slug": "limitus",
          "ticker": "LMT"
        },
        {
          "name": "Galatasaray Fan Token",
          "slug": "galatasaray-fan-token",
          "ticker": "GAL"
        },
        {
          "name": "ELYSIA",
          "slug": "elysia",
          "ticker": "EL"
        },
        {
          "name": "WhiteBIT Token",
          "slug": "whitebit-token",
          "ticker": "WBT"
        },
        {
          "name": "PointPay",
          "slug": "pointpay",
          "ticker": "PXP"
        },
        {
          "name": "Gleec Coin",
          "slug": "gleec",
          "ticker": "GLEEC"
        },
        {
          "name": "Radworks",
          "slug": "radworks",
          "ticker": "RAD"
        },
        {
          "name": "MON Protocol",
          "slug": "mon",
          "ticker": "MON"
        },
        {
          "name": "BreederDAO",
          "slug": "breederdao",
          "ticker": "SOVRN"
        },
        {
          "name": "Frax [on Arbitrum]",
          "slug": "arb-frax",
          "ticker": "FRAX"
        },
        {
          "name": "Merlin Chain",
          "slug": "merlin-chain",
          "ticker": "MERL"
        },
        {
          "name": "LABS Group",
          "slug": "bnb-labs-group",
          "ticker": "LABS"
        },
        {
          "name": "Pepe Unchained",
          "slug": "pepe-unchained",
          "ticker": "PEPU"
        },
        {
          "name": "48 Club Token",
          "slug": "bnb48-club-token",
          "ticker": "KOGE"
        },
        {
          "name": "Atletico De Madrid Fan Token",
          "slug": "atletico-de-madrid-fan-token",
          "ticker": "ATM"
        },
        {
          "name": "Ethena",
          "slug": "ethena",
          "ticker": "ENA"
        },
        {
          "name": "Dynex",
          "slug": "dynex",
          "ticker": "DNX"
        },
        {
          "name": "Arbitrum",
          "slug": "arb-arbitrum",
          "ticker": "ARB"
        },
        {
          "name": "Mantle",
          "slug": "mantle",
          "ticker": "MNT"
        },
        {
          "name": "Multibit",
          "slug": "multibit",
          "ticker": "MUBI"
        },
        {
          "name": "Statter Network",
          "slug": "statter-network",
          "ticker": "STT"
        },
        {
          "name": "Arkham",
          "slug": "arkham",
          "ticker": "ARKM"
        },
        {
          "name": "Clore.ai",
          "slug": "clore-ai",
          "ticker": "CLORE"
        },
        {
          "name": "Quiztok",
          "slug": "quiztok",
          "ticker": "QTCON"
        },
        {
          "name": "Neon",
          "slug": "neon",
          "ticker": "NEON"
        },
        {
          "name": "Rollbit Coin",
          "slug": "rollbit-coin",
          "ticker": "RLB"
        },
        {
          "name": "Helium Mobile",
          "slug": "helium-mobile",
          "ticker": "MOBILE"
        },
        {
          "name": "Worldcoin [on Ethereum]",
          "slug": "worldcoin-org",
          "ticker": "WLD"
        },
        {
          "name": "KlimaDAO",
          "slug": "p-klima-dao",
          "ticker": "KLIMA"
        },
        {
          "name": "Synternet",
          "slug": "synternet",
          "ticker": "SYNT"
        },
        {
          "name": "CyberConnect [on Optimism]",
          "slug": "o-cyberconnect",
          "ticker": "CYBER"
        },
        {
          "name": "Access Protocol",
          "slug": "access-protocol",
          "ticker": "ACS"
        },
        {
          "name": "HarryPotterObamaSonic10Inu (ERC-20)",
          "slug": "harrypotterobamasonic10inu-eth",
          "ticker": "BITCOIN"
        },
        {
          "name": "ChainLink [on Arbitrum]",
          "slug": "arb-chainlink",
          "ticker": "LINK"
        },
        {
          "name": "XRP Ledger",
          "slug": "xrp",
          "ticker": "XRP"
        },
        {
          "name": "AS Roma Fan Token",
          "slug": "as-roma-fan-token",
          "ticker": "ASR"
        },
        {
          "name": "CyberConnect [on Ethereum]",
          "slug": "cyberconnect",
          "ticker": "CYBER"
        },
        {
          "name": "Worldcoin [on Optimism]",
          "slug": "o-worldcoin-org",
          "ticker": "WLD"
        },
        {
          "name": "Port3 Network",
          "slug": "port3-network",
          "ticker": "PORT3"
        },
        {
          "name": "yesnoerror",
          "slug": "yesnoerror",
          "ticker": "YNE"
        },
        {
          "name": "Banana For Scale",
          "slug": "banana-for-scale",
          "ticker": "BANANAS31"
        },
        {
          "name": "Odos",
          "slug": "odos",
          "ticker": "ODOS"
        },
        {
          "name": "Axol",
          "slug": "axol",
          "ticker": "AXOL"
        },
        {
          "name": "Cakepie",
          "slug": "cakepie-xyz",
          "ticker": "CKP"
        },
        {
          "name": "XPLA",
          "slug": "xpla",
          "ticker": "XPLA"
        },
        {
          "name": "Cypherium",
          "slug": "cypherium",
          "ticker": "CPH"
        },
        {
          "name": "Smooth Love Potion",
          "slug": "small-love-potion",
          "ticker": "SLP"
        },
        {
          "name": "Floki [on BNB Chain]",
          "slug": "bnb-floki-inu",
          "ticker": "FLOKI"
        },
        {
          "name": "QuickSwap [on Ethereum]",
          "slug": "quickswap",
          "ticker": "QUICK"
        },
        {
          "name": "Pocket Network",
          "slug": "pocket-network",
          "ticker": "POKT"
        },
        {
          "name": "HODL",
          "slug": "hodl",
          "ticker": "HODL"
        },
        {
          "name": "My Lovely Planet",
          "slug": "p-my-lovely-planet",
          "ticker": "MLC"
        },
        {
          "name": "Patriot",
          "slug": "patriot",
          "ticker": "PATRIOT"
        },
        {
          "name": "Pepecoin",
          "slug": "pepecoin-org",
          "ticker": "PEP"
        },
        {
          "name": "Sharp",
          "slug": "sharp-token",
          "ticker": "SHARP"
        },
        {
          "name": "Falcon USD",
          "slug": "falcon-finance",
          "ticker": "USDf"
        },
        {
          "name": "OG Fan Token",
          "slug": "og-fan-token",
          "ticker": "OG"
        },
        {
          "name": "Oasys",
          "slug": "oasys",
          "ticker": "OAS"
        },
        {
          "name": "Optimus AI",
          "slug": "optimus-ai",
          "ticker": "OPTI"
        },
        {
          "name": "USD Coin [on Ethereum]",
          "slug": "usd-coin",
          "ticker": "USDC"
        },
        {
          "name": "Tether [on Ethereum]",
          "slug": "tether",
          "ticker": "USDT"
        },
        {
          "name": "PLANET [on Ethereum]",
          "slug": "planettoken",
          "ticker": "PLANET"
        },
        {
          "name": "Fenerbahçe Token",
          "slug": "fenerbahce-token",
          "ticker": "FB"
        },
        {
          "name": "Crust Network",
          "slug": "crustnetwork",
          "ticker": "CRU"
        },
        {
          "name": "Dora Factory (old)",
          "slug": "dora-factory",
          "ticker": "DORA"
        },
        {
          "name": "Wormhole",
          "slug": "wormhole",
          "ticker": "W"
        },
        {
          "name": "DMAIL Network",
          "slug": "dmail-network",
          "ticker": "DMAIL"
        },
        {
          "name": "Vana",
          "slug": "vana",
          "ticker": "VANA"
        },
        {
          "name": "Treasure [on Arbitrum]",
          "slug": "arb-magic-token",
          "ticker": "MAGIC"
        },
        {
          "name": "GMT [on BNB]",
          "slug": "bnb-green-metaverse-token",
          "ticker": "GMT"
        },
        {
          "name": "NFT Worlds [on Ethereum]",
          "slug": "nft-worlds",
          "ticker": "WRLD"
        },
        {
          "name": "Tamadoge",
          "slug": "tamadoge",
          "ticker": "TAMA"
        },
        {
          "name": "Qubic",
          "slug": "qubic",
          "ticker": "QUBIC"
        },
        {
          "name": "Pepe",
          "slug": "pepe",
          "ticker": "PEPE"
        },
        {
          "name": "Sui",
          "slug": "sui",
          "ticker": "SUI"
        },
        {
          "name": "Volo Staked SUI",
          "slug": "volo-staked-sui",
          "ticker": "VSUI"
        },
        {
          "name": "Aethir",
          "slug": "aethir",
          "ticker": "ATH"
        },
        {
          "name": "Coinweb",
          "slug": "coinweb",
          "ticker": "CWEB"
        },
        {
          "name": "Pudgy Penguins",
          "slug": "pudgy-penguins",
          "ticker": "PENGU"
        },
        {
          "name": "WOO",
          "slug": "wootrade",
          "ticker": "WOO"
        },
        {
          "name": "Radiant",
          "slug": "radiant",
          "ticker": "RXD"
        },
        {
          "name": "SPACE ID [on Ethereum]",
          "slug": "space-id",
          "ticker": "ID"
        },
        {
          "name": "Wrapped Mantle",
          "slug": "wrapped-mantle",
          "ticker": "WMNT"
        },
        {
          "name": "Rocket Pool ETH",
          "slug": "rocket-pool-eth",
          "ticker": "RETH"
        },
        {
          "name": "Bitfinity Network",
          "slug": "bitfinity-network",
          "ticker": "BTF"
        },
        {
          "name": "Onomy Protocol",
          "slug": "onomy-protocol",
          "ticker": "NOM"
        },
        {
          "name": "Dione Protocol",
          "slug": "dione-protocol",
          "ticker": "DIONE"
        },
        {
          "name": "IX Swap",
          "slug": "ix-swap",
          "ticker": "IXS"
        },
        {
          "name": "LandX Finance",
          "slug": "landx-finance",
          "ticker": "LNDX"
        },
        {
          "name": "peaq",
          "slug": "peaq",
          "ticker": "PEAQ"
        },
        {
          "name": "TokenFi",
          "slug": "tokenfi",
          "ticker": "TOKEN"
        },
        {
          "name": "Polygon Ecosystem Token",
          "slug": "polygon-ecosystem-token",
          "ticker": "POL"
        },
        {
          "name": "Venom",
          "slug": "venom",
          "ticker": "VENOM"
        },
        {
          "name": "Staked ENA",
          "slug": "staked-ethena",
          "ticker": "sENA"
        },
        {
          "name": "Unicorn Fart Dust",
          "slug": "unicorn-fart-dust",
          "ticker": "UFD"
        },
        {
          "name": "Solana Swap",
          "slug": "solana-swap",
          "ticker": "SOS"
        },
        {
          "name": "Wrapped Chiliz",
          "slug": "wrapped-chiliz",
          "ticker": "WCHZ"
        },
        {
          "name": "Frax [on Polygon]",
          "slug": "p-frax",
          "ticker": "FRAX"
        },
        {
          "name": "Dai [on Avalanche]",
          "slug": "a-multi-collateral-dai",
          "ticker": "DAI"
        },
        {
          "name": "Frax [on Ethereum]",
          "slug": "frax",
          "ticker": "FRAX"
        },
        {
          "name": "Binance USD [on Polygon]",
          "slug": "p-binance-usd",
          "ticker": "BUSD"
        },
        {
          "name": "TrueUSD [on BNB Chain]",
          "slug": "bnb-trueusd",
          "ticker": "TUSD"
        },
        {
          "name": "Butthole Coin",
          "slug": "butthole-coin",
          "ticker": "BHC"
        },
        {
          "name": "LOFI",
          "slug": "lofitheyeti",
          "ticker": "LOFI"
        },
        {
          "name": "Dora Factory (new)",
          "slug": "dora-factory-new",
          "ticker": "DORA"
        },
        {
          "name": "Alva",
          "slug": "alva",
          "ticker": "AA"
        },
        {
          "name": "IOTA",
          "slug": "iota",
          "ticker": "IOTA"
        },
        {
          "name": "Fasttoken",
          "slug": "fasttoken",
          "ticker": "FTN"
        },
        {
          "name": "Wall Street Memes",
          "slug": "wall-street-memes",
          "ticker": "WSM"
        },
        {
          "name": "Sperax",
          "slug": "arb-sperax",
          "ticker": "SPA"
        },
        {
          "name": "Koma Inu",
          "slug": "bnb-koma-inu",
          "ticker": "KOMA"
        },
        {
          "name": "Brickken",
          "slug": "brickken",
          "ticker": "BKN"
        },
        {
          "name": "Chintai",
          "slug": "chex-token",
          "ticker": "CHEX"
        },
        {
          "name": "Slothana",
          "slug": "slothana",
          "ticker": "SLOTH"
        },
        {
          "name": "TrumpCoin",
          "slug": "trumpcoin-solana",
          "ticker": "DJT"
        },
        {
          "name": "PUPS (Ordinals)",
          "slug": "pups-ordinals",
          "ticker": "PUPS"
        },
        {
          "name": "Saga",
          "slug": "saga",
          "ticker": "SAGA"
        },
        {
          "name": "Fartboy",
          "slug": "fartboy",
          "ticker": "FARTBOY"
        },
        {
          "name": "Big Time",
          "slug": "big-time",
          "ticker": "BIGTIME"
        },
        {
          "name": "Poolz Finance",
          "slug": "poolz-finance",
          "ticker": "POOLX"
        },
        {
          "name": "Gomining",
          "slug": "gomining-token",
          "ticker": "GOMINING"
        },
        {
          "name": "Starknet",
          "slug": "starknet-token",
          "ticker": "STRK"
        },
        {
          "name": "Treasure [on Ethereum]",
          "slug": "magic-token",
          "ticker": "MAGIC"
        },
        {
          "name": "Tensor",
          "slug": "tensor",
          "ticker": "TNSR"
        },
        {
          "name": "Wrapped eETH",
          "slug": "wrapped-eeth",
          "ticker": "weETH"
        },
        {
          "name": "LayerAI [on Ethereum]",
          "slug": "cryptogpt",
          "ticker": "LAI"
        },
        {
          "name": "Masa",
          "slug": "masa-network",
          "ticker": "MASA"
        },
        {
          "name": "ViciCoin",
          "slug": "vicicoin",
          "ticker": "VCNT"
        },
        {
          "name": "ResearchCoin",
          "slug": "researchcoin",
          "ticker": "RSC"
        },
        {
          "name": "XDB CHAIN",
          "slug": "xdbchain",
          "ticker": "XDB"
        },
        {
          "name": "SyncGPT",
          "slug": "sync-gpt",
          "ticker": "SYNC"
        },
        {
          "name": "Upland (SPARKLET)",
          "slug": "upland-sparklet",
          "ticker": "SPARKLET"
        },
        {
          "name": "ORA",
          "slug": "ora",
          "ticker": "ORA"
        },
        {
          "name": "DeHub",
          "slug": "dehub",
          "ticker": "DHB"
        },
        {
          "name": "Carrieverse",
          "slug": "p-carrieverse",
          "ticker": "CVTX"
        },
        {
          "name": "Oobit",
          "slug": "oobit",
          "ticker": "OBT"
        },
        {
          "name": "SuperWalk",
          "slug": "superwalk",
          "ticker": "GRND"
        },
        {
          "name": "Izumi Finance",
          "slug": "izumi-finance",
          "ticker": "IZI"
        },
        {
          "name": "Turbo",
          "slug": "turbo",
          "ticker": "TURBO"
        },
        {
          "name": "Napoli Fan Token",
          "slug": "napoli-fan-token",
          "ticker": "NAP"
        },
        {
          "name": "LeisureMeta",
          "slug": "leisuremeta",
          "ticker": "LM"
        },
        {
          "name": "Pepe 2.0",
          "slug": "pepe-2-0",
          "ticker": "PEPE2.0"
        },
        {
          "name": "Botto",
          "slug": "botto",
          "ticker": "BOTTO"
        },
        {
          "name": "Celestia",
          "slug": "celestia",
          "ticker": "TIA"
        },
        {
          "name": "ROA CORE",
          "slug": "roa-core",
          "ticker": "ROA"
        },
        {
          "name": "Synth sOIL",
          "slug": "soil",
          "ticker": "SOIL"
        },
        {
          "name": "Myria",
          "slug": "myria",
          "ticker": "MYRIA"
        },
        {
          "name": "The Root Network",
          "slug": "the-root-network",
          "ticker": "ROOT"
        },
        {
          "name": "Minswap",
          "slug": "minswap",
          "ticker": "MIN"
        },
        {
          "name": "Bonk",
          "slug": "bonk1",
          "ticker": "BONK"
        },
        {
          "name": "ALTAVA",
          "slug": "altava",
          "ticker": "TAVA"
        },
        {
          "name": "THE BALKAN DWARF",
          "slug": "the-balkan-dwarf",
          "ticker": "KEKEC"
        },
        {
          "name": "McDull",
          "slug": "mcdull",
          "ticker": "MCDULL"
        },
        {
          "name": "Smog",
          "slug": "smog",
          "ticker": "SMOG"
        },
        {
          "name": "LORDS",
          "slug": "lords",
          "ticker": "LORDS"
        },
        {
          "name": "HAVAH",
          "slug": "havah",
          "ticker": "HVH"
        },
        {
          "name": "GameStop",
          "slug": "gme",
          "ticker": "GME"
        },
        {
          "name": "SKALE",
          "slug": "skale-network",
          "ticker": "SKL"
        },
        {
          "name": "Ponke",
          "slug": "ponke",
          "ticker": "PONKE"
        },
        {
          "name": "RACA",
          "slug": "radio-caca",
          "ticker": "RACA"
        },
        {
          "name": "MeconCash",
          "slug": "meconcash",
          "ticker": "MCH"
        },
        {
          "name": "Unbound",
          "slug": "unbound",
          "ticker": "UNB"
        },
        {
          "name": "Apu Apustaja",
          "slug": "apu-apustaja",
          "ticker": "APU"
        },
        {
          "name": "Super Trump",
          "slug": "super-trump-io",
          "ticker": "STRUMP"
        },
        {
          "name": "Voyager Token",
          "slug": "ethos",
          "ticker": "VGX"
        },
        {
          "name": "The Emerald Company",
          "slug": "the-emerald-company",
          "ticker": "EMRLD"
        },
        {
          "name": "Metis",
          "slug": "metisdao",
          "ticker": "METIS"
        },
        {
          "name": "Aerodrome Finance",
          "slug": "aerodrome-finance",
          "ticker": "AERO"
        },
        {
          "name": "Chain-key Bitcoin",
          "slug": "chain-key-bitcoin",
          "ticker": "CKBTC"
        },
        {
          "name": "Zypto",
          "slug": "zypto",
          "ticker": "ZYPTO"
        },
        {
          "name": "Inter Milan Fan Token",
          "slug": "inter-milan-fan-token",
          "ticker": "INTER"
        },
        {
          "name": "Giant Mammoth",
          "slug": "giant-mammoth",
          "ticker": "GMMT"
        },
        {
          "name": "Katana Inu [on Ethereum]",
          "slug": "katana-inu",
          "ticker": "KATA"
        },
        {
          "name": "DeFi Kingdoms",
          "slug": "defi-kingdoms",
          "ticker": "JEWEL"
        },
        {
          "name": "RMPL",
          "slug": "rmpl",
          "ticker": "RMPL"
        },
        {
          "name": "Pixels",
          "slug": "pixels",
          "ticker": "PIXEL"
        },
        {
          "name": "ICB Network",
          "slug": "icb-network",
          "ticker": "ICBX"
        },
        {
          "name": "Nya",
          "slug": "nya",
          "ticker": "NYA"
        },
        {
          "name": "World of Dypians",
          "slug": "world-of-dypians",
          "ticker": "WOD"
        },
        {
          "name": "Ski Mask Dog",
          "slug": "ski-mask-dog",
          "ticker": "SKI"
        },
        {
          "name": "MNEE",
          "slug": "mnee",
          "ticker": "MNEE"
        },
        {
          "name": "Ethena USDe",
          "slug": "ethena-usde",
          "ticker": "USDe"
        },
        {
          "name": "Across Protocol",
          "slug": "across-protocol",
          "ticker": "ACX"
        },
        {
          "name": "Symbiosis",
          "slug": "symbiosis-finance",
          "ticker": "SIS"
        },
        {
          "name": "SIDUS",
          "slug": "sidus-heroes-sidus-token",
          "ticker": "SIDUS"
        },
        {
          "name": "Gari Network",
          "slug": "gari",
          "ticker": "GARI"
        },
        {
          "name": "Notcoin",
          "slug": "notcoin",
          "ticker": "NOT"
        },
        {
          "name": "Lista Staked BNB",
          "slug": "slisbnb",
          "ticker": "slisBNB"
        },
        {
          "name": "ApeX Protocol [on Ethereum]",
          "slug": "apex-token",
          "ticker": "APEX"
        },
        {
          "name": "PlayDapp",
          "slug": "playdapp",
          "ticker": "PDA"
        },
        {
          "name": "Crown by Third Time Games",
          "slug": "crown-by-third-time-games",
          "ticker": "CROWN"
        },
        {
          "name": "USDB",
          "slug": "usdb",
          "ticker": "USDB"
        },
        {
          "name": "Grand Base",
          "slug": "grand-base",
          "ticker": "GB"
        },
        {
          "name": "Urolithin A",
          "slug": "urolithin-a",
          "ticker": "URO"
        },
        {
          "name": "Rifampicin",
          "slug": "rifampicin",
          "ticker": "RIF"
        },
        {
          "name": "Cheelee",
          "slug": "bnb-cheelee",
          "ticker": "CHEEL"
        },
        {
          "name": "Volt Inu",
          "slug": "volt-inu-v2",
          "ticker": "VOLT"
        },
        {
          "name": "KRYZA Exchange",
          "slug": "kryza-exchange",
          "ticker": "KRX"
        },
        {
          "name": "Credefi",
          "slug": "credefi",
          "ticker": "CREDI"
        },
        {
          "name": "ERC20",
          "slug": "erc20",
          "ticker": "ERC20"
        },
        {
          "name": "DexCheck",
          "slug": "dexcheck",
          "ticker": "DCK"
        },
        {
          "name": "Beacon ETH",
          "slug": "beacon-eth",
          "ticker": "BETH"
        },
        {
          "name": "Lumoz",
          "slug": "lumoz",
          "ticker": "MOZ"
        },
        {
          "name": "METAVERSE",
          "slug": "aipool",
          "ticker": "METAV"
        },
        {
          "name": "ORIGYN",
          "slug": "origyn-foundation",
          "ticker": "OGY"
        },
        {
          "name": "MAP Protocol",
          "slug": "map-protocol",
          "ticker": "MAPO"
        },
        {
          "name": "dYdX",
          "slug": "dydx",
          "ticker": "DYDX"
        },
        {
          "name": "Pax Dollar [on Ethereum]",
          "slug": "paxos-standard",
          "ticker": "USDP"
        },
        {
          "name": "Autonolas",
          "slug": "autonolas",
          "ticker": "OLAS"
        },
        {
          "name": "Aimonica Brands",
          "slug": "aimonica-brands",
          "ticker": "AIMONICA"
        },
        {
          "name": "5ire",
          "slug": "5ire",
          "ticker": "5IRE"
        },
        {
          "name": "PaLM AI",
          "slug": "palm-ai",
          "ticker": "PALM"
        },
        {
          "name": "zkSwap Finance",
          "slug": "zkswap-finance",
          "ticker": "ZF"
        },
        {
          "name": "Solchat",
          "slug": "solchat",
          "ticker": "CHAT"
        },
        {
          "name": "Asterix Labs",
          "slug": "asterix-labs",
          "ticker": "ASTX"
        },
        {
          "name": "dYdX (Native)",
          "slug": "dydx-chain",
          "ticker": "DYDX"
        },
        {
          "name": "Shido (New)",
          "slug": "shido-inu-new",
          "ticker": "SHIDO"
        },
        {
          "name": "Nosana",
          "slug": "nosana",
          "ticker": "NOS"
        },
        {
          "name": "XDEFI Wallet",
          "slug": "xdefi-wallet",
          "ticker": "CTRL"
        },
        {
          "name": "Arsenal Fan Token",
          "slug": "arsenal-fan-token",
          "ticker": "AFC"
        },
        {
          "name": "IMPT",
          "slug": "impt",
          "ticker": "IMPT"
        },
        {
          "name": "Altered State Token",
          "slug": "altered-state-token",
          "ticker": "ASTO"
        },
        {
          "name": "Virtua",
          "slug": "terra-virtua-kolekt",
          "ticker": "TVK"
        },
        {
          "name": "Cipher",
          "slug": "p-cipher-v2",
          "ticker": "CPR"
        },
        {
          "name": "Prometheum Prodigy",
          "slug": "prometheum-prodigy",
          "ticker": "PMPY"
        },
        {
          "name": "Novasama Technologies",
          "slug": "novasama",
          "ticker": "NVST"
        },
        {
          "name": "OORT",
          "slug": "oortech",
          "ticker": "OORT"
        },
        {
          "name": "Areon Network",
          "slug": "areon-network",
          "ticker": "AREA"
        },
        {
          "name": "Zeebu",
          "slug": "zeebu",
          "ticker": "ZBU"
        },
        {
          "name": "Kimbo",
          "slug": "kimbo",
          "ticker": "KIMBO"
        },
        {
          "name": "Solama",
          "slug": "solama",
          "ticker": "SOLAMA"
        },
        {
          "name": "Zebec Network",
          "slug": "zebec-network",
          "ticker": "ZBCN"
        },
        {
          "name": "Popcat (SOL)",
          "slug": "popcat-sol",
          "ticker": "POPCAT"
        },
        {
          "name": "Mantle Staked Ether",
          "slug": "mantle-staked-ether",
          "ticker": "METH"
        },
        {
          "name": "BEERCOIN",
          "slug": "beercoin",
          "ticker": "BEER"
        },
        {
          "name": "Hot Doge",
          "slug": "hot-doge",
          "ticker": "HOTDOGE"
        },
        {
          "name": "Fric",
          "slug": "fric",
          "ticker": "FRIC"
        },
        {
          "name": "Plankton in Pain",
          "slug": "plankton-in-pain",
          "ticker": "AAAHHM"
        },
        {
          "name": "MetaMUI",
          "slug": "metamui",
          "ticker": "MMUI"
        },
        {
          "name": "Berachain",
          "slug": "berachain",
          "ticker": "BERA"
        },
        {
          "name": "CreatorBid",
          "slug": "bnb-creatorbid",
          "ticker": "BID"
        },
        {
          "name": "MSTR2100",
          "slug": "mstr2100",
          "ticker": "MSTR"
        },
        {
          "name": "Non-Playable Coin Solana",
          "slug": "non-playable-coin-solana",
          "ticker": "NPCS"
        },
        {
          "name": "Department Of Government Efficiency (dogegov.com)",
          "slug": "department-of-government-efficiency-token",
          "ticker": "DOGE"
        },
        {
          "name": "PlotX",
          "slug": "plotx",
          "ticker": "PLOT"
        },
        {
          "name": "DeFinder Capital",
          "slug": "definder-capital",
          "ticker": "DFC"
        },
        {
          "name": "Volumint",
          "slug": "volumint",
          "ticker": "VMINT"
        },
        {
          "name": "beoble",
          "slug": "beoble",
          "ticker": "BBL"
        },
        {
          "name": "Shina Inu",
          "slug": "shina-inu",
          "ticker": "SHI"
        },
        {
          "name": "Synesis One",
          "slug": "synesis-one",
          "ticker": "SNS"
        },
        {
          "name": "Affyn",
          "slug": "affyn",
          "ticker": "FYN"
        },
        {
          "name": "Stacks",
          "slug": "blockstack",
          "ticker": "STX"
        },
        {
          "name": "Real Smurf Cat (ETH)",
          "slug": "real-smurf-cat-eth",
          "ticker": "SMURFCAT"
        },
        {
          "name": "Aevo",
          "slug": "aevo",
          "ticker": "AEVO"
        },
        {
          "name": "Ben the Dog",
          "slug": "ben-the-dog",
          "ticker": "BENDOG"
        },
        {
          "name": "AiMalls",
          "slug": "aimalls",
          "ticker": "AIT"
        },
        {
          "name": "Chain of Legends",
          "slug": "chain-of-legends",
          "ticker": "CLEG"
        },
        {
          "name": "MurAll",
          "slug": "murall",
          "ticker": "PAINT"
        },
        {
          "name": "Shirtum",
          "slug": "shirtum",
          "ticker": "SHI"
        },
        {
          "name": "Alanyaspor Fan Token",
          "slug": "alanyaspor-fan-token",
          "ticker": "ALA"
        },
        {
          "name": "ZAIBOT",
          "slug": "zaibot",
          "ticker": "ZAI"
        },
        {
          "name": "XCarnival",
          "slug": "xcarnival",
          "ticker": "XCV"
        },
        {
          "name": "ElvishMagic",
          "slug": "elvishmagic",
          "ticker": "EMAGIC"
        },
        {
          "name": "Ideaology",
          "slug": "ideaology",
          "ticker": "IDEA"
        },
        {
          "name": "PUMLx",
          "slug": "pumlx",
          "ticker": "PUMLX"
        },
        {
          "name": "Oggy Inu (ETH)",
          "slug": "oggy-inu-eth",
          "ticker": "OGGY"
        },
        {
          "name": "Valencia CF Fan Token",
          "slug": "valencia-cf-fan-token",
          "ticker": "VCF"
        },
        {
          "name": "King Shiba",
          "slug": "king-shiba",
          "ticker": "KINGSHIB"
        },
        {
          "name": "ALLY",
          "slug": "ally",
          "ticker": "ALY"
        },
        {
          "name": "Hot Cross",
          "slug": "hot-cross",
          "ticker": "HOTCROSS"
        },
        {
          "name": "Etherland",
          "slug": "etherland",
          "ticker": "ELAND"
        },
        {
          "name": "Peng",
          "slug": "peng-sol",
          "ticker": "PENG"
        },
        {
          "name": "Andy on SOL",
          "slug": "andy-on-sol",
          "ticker": "ANDY"
        },
        {
          "name": "LOAF CAT",
          "slug": "loaf-cat",
          "ticker": "LOAF"
        },
        {
          "name": "Experimental Finance",
          "slug": "flare-finance",
          "ticker": "EXFI"
        },
        {
          "name": "OVR",
          "slug": "ovr",
          "ticker": "OVR"
        },
        {
          "name": "Omax Coin",
          "slug": "omax-token",
          "ticker": "OMAX"
        },
        {
          "name": "Niobium Coin",
          "slug": "niobium-coin",
          "ticker": "SHC"
        },
        {
          "name": "Tectum",
          "slug": "tectum",
          "ticker": "TET"
        },
        {
          "name": "blockbank",
          "slug": "blockbank",
          "ticker": "BBANK"
        },
        {
          "name": "Konnect",
          "slug": "konnect",
          "ticker": "KCT"
        },
        {
          "name": "Restaked Swell Ethereum",
          "slug": "restaked-swell-ethereum",
          "ticker": "RSWETH"
        },
        {
          "name": "DUKO",
          "slug": "duko",
          "ticker": "DUKO"
        },
        {
          "name": "Safe",
          "slug": "safe1",
          "ticker": "SAFE"
        },
        {
          "name": "RealFevr",
          "slug": "bnb-realfevr",
          "ticker": "FEVR"
        },
        {
          "name": "Landshare",
          "slug": "landshare",
          "ticker": "LAND"
        },
        {
          "name": "Plearn",
          "slug": "plearn",
          "ticker": "PLN"
        },
        {
          "name": "Colana",
          "slug": "dogecola",
          "ticker": "COL"
        },
        {
          "name": "Powsche",
          "slug": "powsche",
          "ticker": "POWSCHE"
        },
        {
          "name": "Baanx",
          "slug": "baanx",
          "ticker": "BXX"
        },
        {
          "name": "EverValue Coin",
          "slug": "evervalue-coin",
          "ticker": "EVA"
        },
        {
          "name": "Whales Market",
          "slug": "whales-market",
          "ticker": "WHALES"
        },
        {
          "name": "Stonks",
          "slug": "stonks",
          "ticker": "STNK"
        },
        {
          "name": "Dusk",
          "slug": "dusk-network",
          "ticker": "DUSK"
        },
        {
          "name": "Origin DeFi Governance",
          "slug": "origin-dollar-governance",
          "ticker": "OGV"
        },
        {
          "name": "LimeWire",
          "slug": "limewire",
          "ticker": "LMWR"
        },
        {
          "name": "PAAL AI",
          "slug": "paal-ai",
          "ticker": "PAAL"
        },
        {
          "name": "Viction",
          "slug": "viction",
          "ticker": "VIC"
        },
        {
          "name": "Ethena Staked USDe",
          "slug": "ethena-staked-usde",
          "ticker": "sUSDe"
        },
        {
          "name": "Lush AI",
          "slug": "lush-ai",
          "ticker": "LUSH"
        },
        {
          "name": "ChainSwap",
          "slug": "chain-swap",
          "ticker": "CSWAP"
        },
        {
          "name": "Minati Coin",
          "slug": "minati-coin",
          "ticker": "MNTC"
        },
        {
          "name": "Prisma mkUSD",
          "slug": "prisma-mkusd",
          "ticker": "MKUSD"
        },
        {
          "name": "MAGA VP",
          "slug": "maga-vp",
          "ticker": "MVP"
        },
        {
          "name": "Baby Bonk",
          "slug": "baby-bonk-coin",
          "ticker": "BABYBONK"
        },
        {
          "name": "Honk",
          "slug": "honk",
          "ticker": "HONK"
        },
        {
          "name": "Gala",
          "slug": "gala-v2",
          "ticker": "GALA"
        },
        {
          "name": "Parcl",
          "slug": "parcl",
          "ticker": "PRCL"
        },
        {
          "name": "GameBuild",
          "slug": "gamebuild",
          "ticker": "GAME"
        },
        {
          "name": "HahaYes",
          "slug": "hahayes",
          "ticker": "RIZO"
        },
        {
          "name": "B3 (Base)",
          "slug": "b3",
          "ticker": "B3"
        },
        {
          "name": "neversol",
          "slug": "neversol",
          "ticker": "NEVER"
        },
        {
          "name": "Sealwifhat",
          "slug": "sealwifhat",
          "ticker": "SI"
        },
        {
          "name": "Orca",
          "slug": "orca",
          "ticker": "ORCA"
        },
        {
          "name": "PERL.eco",
          "slug": "perl-eco",
          "ticker": "PERL"
        },
        {
          "name": "VNX Euro",
          "slug": "vnx-euro",
          "ticker": "VEUR"
        },
        {
          "name": "Portal",
          "slug": "portal-gaming",
          "ticker": "PORTAL"
        },
        {
          "name": "Pyth Network",
          "slug": "pyth-network",
          "ticker": "PYTH"
        },
        {
          "name": "ApeX Protocol [on Arbitrum]",
          "slug": "arb-apex-token",
          "ticker": "APEX"
        },
        {
          "name": "aiRight",
          "slug": "airight",
          "ticker": "AIRI"
        },
        {
          "name": "GamesPad",
          "slug": "gamespad",
          "ticker": "GMPD"
        },
        {
          "name": "CryptoZoon",
          "slug": "cryptozoon",
          "ticker": "ZOON"
        },
        {
          "name": "Wizardia",
          "slug": "wizardia",
          "ticker": "WZRD"
        },
        {
          "name": "LogX Network",
          "slug": "logx",
          "ticker": "LOGX"
        },
        {
          "name": "Gemie",
          "slug": "gemie",
          "ticker": "GEM"
        },
        {
          "name": "Neurashi",
          "slug": "neurashi",
          "ticker": "NEI"
        },
        {
          "name": "Kaby Arena",
          "slug": "kaby-arena",
          "ticker": "KABY"
        },
        {
          "name": "Dexlab",
          "slug": "dexlab",
          "ticker": "DXL"
        },
        {
          "name": "AVA",
          "slug": "ava-sol",
          "ticker": "AVA"
        },
        {
          "name": "Wrapped IoTeX",
          "slug": "wrapped-iotex",
          "ticker": "WIOTX"
        },
        {
          "name": "Ledger AI",
          "slug": "ledger-ai",
          "ticker": "LEDGER"
        },
        {
          "name": "BENQI Liquid Staked AVAX",
          "slug": "benqi-liquid-staked-avax",
          "ticker": "sAVAX"
        },
        {
          "name": "Wrapped Centrifuge",
          "slug": "wrapped-centrifuge",
          "ticker": "WCFG"
        },
        {
          "name": "Team Heretics Fan Token",
          "slug": "team-heretics-fan-token",
          "ticker": "TH"
        },
        {
          "name": "Salad",
          "slug": "salad",
          "ticker": "SALD"
        },
        {
          "name": "Virtual Versions",
          "slug": "vv-token",
          "ticker": "VV"
        },
        {
          "name": "Karat",
          "slug": "karat",
          "ticker": "KAT"
        },
        {
          "name": "Dreams Quest",
          "slug": "dreams-quest",
          "ticker": "DREAMS"
        },
        {
          "name": "Mars Token",
          "slug": "mars-token",
          "ticker": "MRST"
        },
        {
          "name": "MetaFighter",
          "slug": "metafighter",
          "ticker": "MF"
        },
        {
          "name": "Clube Atlético Mineiro Fan Token",
          "slug": "clube-atletico-mineiro-fan-token",
          "ticker": "GALO"
        },
        {
          "name": "POLKER",
          "slug": "polker",
          "ticker": "PKR"
        },
        {
          "name": "BlackCardCoin",
          "slug": "blackcardcoin",
          "ticker": "BCCOIN"
        },
        {
          "name": "Kujira [on Ethereum]",
          "slug": "kujira",
          "ticker": "KUJI"
        },
        {
          "name": "Aave [on Ethereum]",
          "slug": "aave",
          "ticker": "AAVE"
        },
        {
          "name": "Mines of Dalarnia [on Ethereum]",
          "slug": "mines-of-dalarnia",
          "ticker": "D"
        },
        {
          "name": "Palmswap",
          "slug": "bnb-palmswap",
          "ticker": "PALM"
        },
        {
          "name": "Coq Inu",
          "slug": "a-coq-inu",
          "ticker": "COQ"
        },
        {
          "name": "ZeroLend",
          "slug": "zerolend",
          "ticker": "ZERO"
        },
        {
          "name": "Elixir Games",
          "slug": "elixir-games",
          "ticker": "ELIX"
        },
        {
          "name": "Ethena Labs (USDTb)",
          "slug": "ethena-labs-usdtb",
          "ticker": "USDTb"
        },
        {
          "name": "Grumpy (Ordinals)",
          "slug": "grumpy",
          "ticker": "GRUM"
        },
        {
          "name": "Myro",
          "slug": "myro",
          "ticker": "MYRO"
        },
        {
          "name": "OmniFlix Network",
          "slug": "omniflix-network",
          "ticker": "FLIX"
        },
        {
          "name": "Super Champs",
          "slug": "super-champs",
          "ticker": "CHAMP"
        },
        {
          "name": "Walken",
          "slug": "walken",
          "ticker": "WLKN"
        },
        {
          "name": "Propbase",
          "slug": "propbase",
          "ticker": "PROPS"
        },
        {
          "name": "Helium IOT",
          "slug": "helium-iot",
          "ticker": "IOT"
        },
        {
          "name": "Aptos",
          "slug": "aptos",
          "ticker": "APT"
        },
        {
          "name": "MetaFighter",
          "slug": "bnb-metafighter",
          "ticker": "MF"
        },
        {
          "name": "Banana Gun",
          "slug": "banana-gun",
          "ticker": "BANANA"
        },
        {
          "name": "Marlin",
          "slug": "marlin",
          "ticker": "POND"
        },
        {
          "name": "Decimal",
          "slug": "decimal",
          "ticker": "DEL"
        },
        {
          "name": "dogwifhat",
          "slug": "dogwifhat",
          "ticker": "WIF"
        },
        {
          "name": "Elemon",
          "slug": "elemon",
          "ticker": "ELMON"
        },
        {
          "name": "Chronicle",
          "slug": "chronicle",
          "ticker": "XNL"
        },
        {
          "name": "Football World Community",
          "slug": "qatar-2022-token",
          "ticker": "FWC"
        },
        {
          "name": "Oggy Inu (BSC)",
          "slug": "oggy-inu",
          "ticker": "OGGY"
        },
        {
          "name": "ISLAMICOIN",
          "slug": "islamicoin",
          "ticker": "ISLAMI"
        },
        {
          "name": "Kripto koin",
          "slug": "kripto-koin",
          "ticker": "KRIPTO"
        },
        {
          "name": "Standard",
          "slug": "standard-protocol",
          "ticker": "STND"
        },
        {
          "name": "Little Rabbit v2",
          "slug": "little-rabbit-v2",
          "ticker": "LTRBT"
        },
        {
          "name": "ROCKI",
          "slug": "rocki",
          "ticker": "ROCKI"
        },
        {
          "name": "Shopping.io",
          "slug": "shopping-io-token",
          "ticker": "SHOP"
        },
        {
          "name": "hiPUNKS",
          "slug": "hipunks",
          "ticker": "HIPUNKS"
        },
        {
          "name": "Miracle Play",
          "slug": "miracle-play",
          "ticker": "MPT"
        },
        {
          "name": "Caduceus",
          "slug": "caduceus-foundation",
          "ticker": "CMP"
        },
        {
          "name": "Hord",
          "slug": "hord",
          "ticker": "HORD"
        },
        {
          "name": "hiPENGUINS",
          "slug": "hipenguins",
          "ticker": "HIPENGUINS"
        },
        {
          "name": "AgeOfGods",
          "slug": "ageofgods",
          "ticker": "AOG"
        },
        {
          "name": "KING",
          "slug": "king",
          "ticker": "KING"
        },
        {
          "name": "XELIS",
          "slug": "xelis",
          "ticker": "XEL"
        },
        {
          "name": "Coupon Assets",
          "slug": "coupon-assets",
          "ticker": "CA"
        },
        {
          "name": "Alvara Protocol",
          "slug": "alvara-protocol",
          "ticker": "ALVA"
        },
        {
          "name": "QnA3.AI",
          "slug": "qna3ai",
          "ticker": "GPT"
        },
        {
          "name": "Neos.ai",
          "slug": "neos-ai",
          "ticker": "NEOS"
        },
        {
          "name": "PussFi",
          "slug": "pussfi",
          "ticker": "PUSS"
        },
        {
          "name": "Rainbow Token",
          "slug": "rainbowtoken",
          "ticker": "RBW"
        },
        {
          "name": "Wownero",
          "slug": "wownero",
          "ticker": "WOW"
        },
        {
          "name": "Lyra [on Optimism]",
          "slug": "o-lyra-finance",
          "ticker": "LYRA"
        },
        {
          "name": "IX Token",
          "slug": "p-ixt-token",
          "ticker": "IXT"
        },
        {
          "name": "Frax Price Index",
          "slug": "frax-price-index",
          "ticker": "FPI"
        },
        {
          "name": "Quickswap[New]",
          "slug": "p-quickswap-new",
          "ticker": "QUICK"
        },
        {
          "name": "Soil",
          "slug": "p-soil",
          "ticker": "SOIL"
        },
        {
          "name": "Orbcity",
          "slug": "p-orbcity",
          "ticker": "ORB"
        },
        {
          "name": "Heroes Chained",
          "slug": "a-heroes-chained",
          "ticker": "HEC"
        },
        {
          "name": "Internet Computer",
          "slug": "internet-computer",
          "ticker": "ICP"
        },
        {
          "name": "SolCex",
          "slug": "solcex",
          "ticker": "SOLCEX"
        },
        {
          "name": "Ethervista",
          "slug": "ethervista",
          "ticker": "VISTA"
        },
        {
          "name": "Realis Worlds",
          "slug": "realis-worlds",
          "ticker": "REALIS"
        },
        {
          "name": "Bellscoin",
          "slug": "bellscoin",
          "ticker": "BELLS"
        },
        {
          "name": "BLOCKLORDS",
          "slug": "blocklords",
          "ticker": "LRDS"
        },
        {
          "name": "Spectral",
          "slug": "spectral",
          "ticker": "SPEC"
        },
        {
          "name": "Usual",
          "slug": "usual",
          "ticker": "USUAL"
        },
        {
          "name": "Thala",
          "slug": "thala",
          "ticker": "THL"
        },
        {
          "name": "Nuco.cloud",
          "slug": "nuco-cloud",
          "ticker": "NCDT"
        },
        {
          "name": "Magaverse",
          "slug": "magaverse",
          "ticker": "MVRS"
        },
        {
          "name": "Floki [on Ethereum]",
          "slug": "floki-inu-v2",
          "ticker": "FLOKI"
        },
        {
          "name": "MCOIN",
          "slug": "mcoin1",
          "ticker": "MCOIN"
        },
        {
          "name": "PlayZap",
          "slug": "bnb-playzap",
          "ticker": "PZP"
        },
        {
          "name": "School Hack Coin",
          "slug": "school-hack-coin",
          "ticker": "SHC"
        },
        {
          "name": "SaucerSwap",
          "slug": "saucerswap",
          "ticker": "SAUCE"
        },
        {
          "name": "GEODNET",
          "slug": "geodnet",
          "ticker": "GEOD"
        },
        {
          "name": "Oho",
          "slug": "oho",
          "ticker": "OHO"
        },
        {
          "name": "VGX Token",
          "slug": "vgx-token",
          "ticker": "VGX"
        },
        {
          "name": "Ambire Wallet",
          "slug": "ambire-wallet",
          "ticker": "WALLET"
        },
        {
          "name": "Scallop",
          "slug": "scallop-protocol",
          "ticker": "SCA"
        },
        {
          "name": "StorX Network",
          "slug": "storx-network",
          "ticker": "SRX"
        },
        {
          "name": "ArbDoge AI",
          "slug": "arb-arbdoge-ai",
          "ticker": "AIDOGE"
        },
        {
          "name": "Defactor",
          "slug": "p-defactor",
          "ticker": "FACTR"
        },
        {
          "name": "Permission Coin",
          "slug": "p-permission-coin",
          "ticker": "ASK"
        },
        {
          "name": "GetKicks",
          "slug": "bnb-getkicks",
          "ticker": "KICKS"
        },
        {
          "name": "AIgentX",
          "slug": "aigentx",
          "ticker": "AGX"
        },
        {
          "name": "Mode",
          "slug": "mode",
          "ticker": "MODE"
        },
        {
          "name": "Memes AI",
          "slug": "memesai",
          "ticker": "MemesAI"
        },
        {
          "name": "Kaia",
          "slug": "kaia",
          "ticker": "KAIA"
        },
        {
          "name": "Happy Cat",
          "slug": "happy-cat-on-sol",
          "ticker": "HAPPY"
        },
        {
          "name": "Azit",
          "slug": "azit",
          "ticker": "AZIT"
        },
        {
          "name": "Grass",
          "slug": "grass",
          "ticker": "GRASS"
        },
        {
          "name": "Baby Doge Coin",
          "slug": "bnb-1m-baby-doge-coin",
          "ticker": "1MBABYDOGE"
        },
        {
          "name": "hehe",
          "slug": "hehe-sol",
          "ticker": "HEHE"
        },
        {
          "name": "Lollybomb Meme Coin",
          "slug": "lollybomb-meme-coin",
          "ticker": "BOMB"
        },
        {
          "name": "JOE",
          "slug": "a-joe",
          "ticker": "JOE"
        },
        {
          "name": "CyberConnect [on BNB Chain]",
          "slug": "bnb-cyberconnect",
          "ticker": "CYBER"
        },
        {
          "name": "Chain of Legends",
          "slug": "bnb-chain-of-legends",
          "ticker": "CLEG"
        },
        {
          "name": "Trillioner",
          "slug": "bnb-trillioner",
          "ticker": "TLC"
        },
        {
          "name": "Dolan Duck",
          "slug": "dolan-duck",
          "ticker": "DOLAN"
        },
        {
          "name": "Luckycoin",
          "slug": "luckycoin",
          "ticker": "LKY"
        },
        {
          "name": "Lumia",
          "slug": "lumia",
          "ticker": "LUMIA"
        },
        {
          "name": "BlueMove",
          "slug": "bluemove",
          "ticker": "MOVE"
        },
        {
          "name": "Luna by Virtuals",
          "slug": "luna-by-virtuals",
          "ticker": "LUNA"
        },
        {
          "name": "ether.fi",
          "slug": "ether-fi-ethfi",
          "ticker": "ETHFI"
        },
        {
          "name": "AI Companions",
          "slug": "bnb-ai-companions",
          "ticker": "AIC"
        },
        {
          "name": "PepeCoin",
          "slug": "pepecoin",
          "ticker": "PEPECOIN"
        },
        {
          "name": "Elixir deUSD",
          "slug": "elixir-deusd",
          "ticker": "DEUSD"
        },
        {
          "name": "ARC",
          "slug": "arc",
          "ticker": "ARC"
        },
        {
          "name": "Rejuve.AI",
          "slug": "rejuve-ai",
          "ticker": "RJV"
        },
        {
          "name": "Zenrock",
          "slug": "zenrock",
          "ticker": "ROCK"
        },
        {
          "name": "The White Lion",
          "slug": "the-white-lion",
          "ticker": "KIMBA"
        },
        {
          "name": "KOMPETE",
          "slug": "kompete",
          "ticker": "KOMPETE"
        },
        {
          "name": "GOGGLES",
          "slug": "goggles",
          "ticker": "GOGLZ"
        },
        {
          "name": "PoSciDonDAO",
          "slug": "poscidondao",
          "ticker": "SCI"
        },
        {
          "name": "Hosky Token",
          "slug": "hosky-token",
          "ticker": "HOSKY"
        },
        {
          "name": "Wrapped Islamic Coin",
          "slug": "wrapped-islamic-coin",
          "ticker": "WISLM"
        },
        {
          "name": "Azuro Protocol",
          "slug": "azuro-protocol",
          "ticker": "AZUR"
        },
        {
          "name": "PERI Finance",
          "slug": "p-peri-finance",
          "ticker": "PERI"
        },
        {
          "name": "Undeads Games",
          "slug": "undeads-games",
          "ticker": "UDS"
        },
        {
          "name": "DIMO",
          "slug": "p-dimo",
          "ticker": "DIMO"
        },
        {
          "name": "Eldarune",
          "slug": "bnb-eldarune",
          "ticker": "ELDA"
        },
        {
          "name": "Daddy Tate",
          "slug": "daddy-tate",
          "ticker": "DADDY"
        },
        {
          "name": "Usual USD",
          "slug": "usual-usd",
          "ticker": "USD0"
        },
        {
          "name": "SuperVerse",
          "slug": "superfarm",
          "ticker": "SUPER"
        },
        {
          "name": "X Empire",
          "slug": "x-empire",
          "ticker": "X"
        },
        {
          "name": "THENA",
          "slug": "bnb-thena",
          "ticker": "THE"
        },
        {
          "name": "TenUp",
          "slug": "tenup",
          "ticker": "TUP"
        },
        {
          "name": "Hyperliquid",
          "slug": "hyperliquid",
          "ticker": "HYPE"
        },
        {
          "name": "Gui Inu",
          "slug": "gui-inu",
          "ticker": "GUI"
        },
        {
          "name": "Global Dollar",
          "slug": "global-dollar-usdg",
          "ticker": "USDG"
        },
        {
          "name": "ClinTex CTi",
          "slug": "clintex-cti",
          "ticker": "CTI"
        },
        {
          "name": "Morpho",
          "slug": "morpho",
          "ticker": "MORPHO"
        },
        {
          "name": "Ta-da",
          "slug": "ta-da",
          "ticker": "TADA"
        },
        {
          "name": "Chihuahua",
          "slug": "chihuahua-wtf",
          "ticker": "HUAHUA"
        },
        {
          "name": "CatSlap",
          "slug": "catslap",
          "ticker": "SLAP"
        },
        {
          "name": "Boop",
          "slug": "boop-the-coin",
          "ticker": "BOOP"
        },
        {
          "name": "Sekuya Multiverse",
          "slug": "sekuya-multiverse",
          "ticker": "SKYA"
        },
        {
          "name": "Lovely Finance [New]",
          "slug": "lovely-inu-new",
          "ticker": "LOVELY"
        },
        {
          "name": "Decentral Games ICE",
          "slug": "p-decentral-games-ice",
          "ticker": "ICE"
        },
        {
          "name": "Frax [on BNB]",
          "slug": "bnb-frax",
          "ticker": "FRAX"
        },
        {
          "name": "StrikeX",
          "slug": "bnb-strikecoin",
          "ticker": "STRX"
        },
        {
          "name": "Meta Monopoly",
          "slug": "meta-monopoly",
          "ticker": "MONOPOLY"
        },
        {
          "name": "DOGS",
          "slug": "dogs",
          "ticker": "DOGS"
        },
        {
          "name": "Smoking Chicken Fish",
          "slug": "smoking-chicken-fish",
          "ticker": "SCF"
        },
        {
          "name": "Simon's Cat",
          "slug": "bnb-simonscat",
          "ticker": "CAT"
        },
        {
          "name": "ETHEREUM IS GOOD",
          "slug": "ethereum-is-good",
          "ticker": "EBULL"
        },
        {
          "name": "Osaka Protocol",
          "slug": "osaka-protocol",
          "ticker": "OSAK"
        },
        {
          "name": "RETARDIO",
          "slug": "retardio",
          "ticker": "RETARDIO"
        },
        {
          "name": "CatDog",
          "slug": "catdog-io",
          "ticker": "CATDOG"
        },
        {
          "name": "Alkimi",
          "slug": "alkimi",
          "ticker": "ADS"
        },
        {
          "name": "Fluence",
          "slug": "fluence-network",
          "ticker": "FLT"
        },
        {
          "name": "Invest Zone",
          "slug": "invest-zone",
          "ticker": "IVfun"
        },
        {
          "name": "Self Chain",
          "slug": "self-chain",
          "ticker": "SLF"
        },
        {
          "name": "FEED EVERY GORILLA",
          "slug": "feed-every-gorilla",
          "ticker": "FEG"
        },
        {
          "name": "Q Protocol",
          "slug": "arb-q-protocol",
          "ticker": "QGOV"
        },
        {
          "name": "LUCE",
          "slug": "luce",
          "ticker": "LUCE"
        },
        {
          "name": "Velodrome Finance",
          "slug": "o-velodrome-finance",
          "ticker": "VELO"
        },
        {
          "name": "Baby Doge Coin",
          "slug": "bnb-baby-doge-coin",
          "ticker": "BabyDoge"
        },
        {
          "name": "Synapse [on Arbitrum]",
          "slug": "arb-synapse-2",
          "ticker": "SYN"
        },
        {
          "name": "Affyn",
          "slug": "p-affyn",
          "ticker": "FYN"
        },
        {
          "name": "NATIX Network",
          "slug": "natix-network",
          "ticker": "NATIX"
        },
        {
          "name": "Crypto-AI-Robo.com",
          "slug": "crypto-ai-robo",
          "ticker": "CAIR"
        },
        {
          "name": "Matr1x",
          "slug": "matr1x",
          "ticker": "MAX"
        },
        {
          "name": "Kendu Inu",
          "slug": "kendu-inu",
          "ticker": "KENDU"
        },
        {
          "name": "UXLINK",
          "slug": "arb-uxlink",
          "ticker": "UXLINK"
        },
        {
          "name": "Brainlet",
          "slug": "brainlet",
          "ticker": "BRAINLET"
        },
        {
          "name": "MetaMAFIA",
          "slug": "metamafia",
          "ticker": "MAF"
        },
        {
          "name": "Koinos",
          "slug": "koinos",
          "ticker": "KOIN"
        },
        {
          "name": "Ginnan The Cat",
          "slug": "ginnan-the-cat",
          "ticker": "GINNAN"
        },
        {
          "name": "RyuJin",
          "slug": "ryujin",
          "ticker": "RYU"
        },
        {
          "name": "Just a chill guy",
          "slug": "just-a-chill-guy",
          "ticker": "CHILLGUY"
        },
        {
          "name": "Shieldeum",
          "slug": "shieldeum",
          "ticker": "SDM"
        },
        {
          "name": "Alphakek AI",
          "slug": "alphakek-ai",
          "ticker": "AIKEK"
        },
        {
          "name": "Winnerz",
          "slug": "winnerz",
          "ticker": "WNZ"
        },
        {
          "name": "Quidax Token",
          "slug": "quidax",
          "ticker": "QDX"
        },
        {
          "name": "Bitcoin Virtual Machine",
          "slug": "bvm",
          "ticker": "BVM"
        },
        {
          "name": "Common Wealth",
          "slug": "common-wealth",
          "ticker": "WLTH"
        },
        {
          "name": "Binance USD [on BNB]",
          "slug": "bnb-binance-usd",
          "ticker": "BUSD"
        },
        {
          "name": "USD Coin [on Avalanche]",
          "slug": "a-usd-coin",
          "ticker": "USDC"
        },
        {
          "name": "Wat",
          "slug": "wat",
          "ticker": "WAT"
        },
        {
          "name": "Koala AI",
          "slug": "koala-ai",
          "ticker": "KOKO"
        },
        {
          "name": "PancakeSwap",
          "slug": "bnb-pancakeswap",
          "ticker": "CAKE"
        },
        {
          "name": "BasedAI",
          "slug": "basedai",
          "ticker": "BASEDAI"
        },
        {
          "name": "Eurite",
          "slug": "eurite",
          "ticker": "EURI"
        },
        {
          "name": "Act I : The AI Prophecy",
          "slug": "act-i-the-ai-prophecy",
          "ticker": "ACT"
        },
        {
          "name": "Hege",
          "slug": "hege",
          "ticker": "HEGE"
        },
        {
          "name": "Omni Network",
          "slug": "omni-network",
          "ticker": "OMNI"
        },
        {
          "name": "Mysterium",
          "slug": "mysterium",
          "ticker": "MYST"
        },
        {
          "name": "First Convicted Raccon Fred",
          "slug": "first-convicted-raccon-fred",
          "ticker": "FRED"
        },
        {
          "name": "Troll",
          "slug": "troll-new",
          "ticker": "TROLL"
        },
        {
          "name": "Hamster Kombat",
          "slug": "hamster-kombat",
          "ticker": "HMSTR"
        },
        {
          "name": "SAD HAMSTER",
          "slug": "sad-hamster",
          "ticker": "HAMMY"
        },
        {
          "name": "insurance",
          "slug": "bnb-insurance",
          "ticker": "INSURANCE"
        },
        {
          "name": "OpenGPU",
          "slug": "open-gpu",
          "ticker": "oGPU"
        },
        {
          "name": "Mythos",
          "slug": "mythos",
          "ticker": "MYTH"
        },
        {
          "name": "Sabai Protocol",
          "slug": "sabai-ecoverse",
          "ticker": "SABAI"
        },
        {
          "name": "Voxies",
          "slug": "p-voxies",
          "ticker": "VOXEL"
        },
        {
          "name": "Stargate Finance [on Arbitrum]",
          "slug": "arb-stargate-finance",
          "ticker": "STG"
        },
        {
          "name": "NFT Worlds [on Polygon]",
          "slug": "p-nft-worlds",
          "ticker": "WRLD"
        },
        {
          "name": "Sky Dollar",
          "slug": "usds",
          "ticker": "USDS"
        },
        {
          "name": "8-Bit Coin",
          "slug": "mario-coin",
          "ticker": "COIN"
        },
        {
          "name": "DecideAI",
          "slug": "decideai",
          "ticker": "DCD"
        },
        {
          "name": "Tethereum",
          "slug": "bnb-tethereum",
          "ticker": "T99"
        },
        {
          "name": "SwissCheese",
          "slug": "p-swisscheese",
          "ticker": "SWCH"
        },
        {
          "name": "Decentral Games",
          "slug": "decentral-games",
          "ticker": "DG"
        },
        {
          "name": "Fractal Bitcoin",
          "slug": "fractal-bitcoin",
          "ticker": "FB"
        },
        {
          "name": "Aviator",
          "slug": "aviator",
          "ticker": "AVI"
        },
        {
          "name": "Tron Bull",
          "slug": "tron-bull",
          "ticker": "BULL"
        },
        {
          "name": "Aleo",
          "slug": "aleo",
          "ticker": "ALEO"
        },
        {
          "name": "Sky",
          "slug": "sky",
          "ticker": "SKY"
        },
        {
          "name": "Paxe",
          "slug": "bnb-paxe",
          "ticker": "PAXE"
        },
        {
          "name": "Moo Deng (moodengsol.com)",
          "slug": "moo-deng-solana",
          "ticker": "MOODENG"
        },
        {
          "name": "MESSIER",
          "slug": "messier",
          "ticker": "M87"
        },
        {
          "name": "Orderly Network",
          "slug": "orderly-network",
          "ticker": "ORDER"
        },
        {
          "name": "Bitget Wallet Token",
          "slug": "bitget-wallet-token",
          "ticker": "BWB"
        },
        {
          "name": "Wonderland",
          "slug": "a-wonderland",
          "ticker": "TIME"
        },
        {
          "name": "USDD [on BNB Chain]",
          "slug": "bnb-usdd",
          "ticker": "USDD"
        },
        {
          "name": "Peanut the Squirrel",
          "slug": "peanut-the-squirrel",
          "ticker": "PNUT"
        },
        {
          "name": "QnA3.AI",
          "slug": "bnb-qna3ai",
          "ticker": "GPT"
        },
        {
          "name": "Puffy",
          "slug": "puffy",
          "ticker": "PUFFY"
        },
        {
          "name": "Router Protocol (New)",
          "slug": "router-protocol-2",
          "ticker": "ROUTE"
        },
        {
          "name": "Creta World",
          "slug": "p-creta-world",
          "ticker": "CRETA"
        },
        {
          "name": "DOJO Protocol",
          "slug": "dojo-protocol",
          "ticker": "DOAI"
        },
        {
          "name": "Magpie",
          "slug": "magpie",
          "ticker": "MGP"
        },
        {
          "name": "Pups (Bitcoin)",
          "slug": "pups-runes",
          "ticker": "PUPS"
        },
        {
          "name": "Book of Ethereum",
          "slug": "book-of-ethereum",
          "ticker": "BOOE"
        },
        {
          "name": "GameStop",
          "slug": "gamestop",
          "ticker": "GME"
        },
        {
          "name": "Alchemist AI",
          "slug": "alchemist-ai",
          "ticker": "ALCH"
        },
        {
          "name": "Cate",
          "slug": "cate",
          "ticker": "CATE"
        },
        {
          "name": "MUNCAT",
          "slug": "muncat",
          "ticker": "MUNCAT"
        },
        {
          "name": "TG Casino",
          "slug": "tg-casino",
          "ticker": "TGC"
        },
        {
          "name": "Wrapped Dog",
          "slug": "wrapped-dog",
          "ticker": "WDOG"
        },
        {
          "name": "MAD",
          "slug": "mad-token",
          "ticker": "MAD"
        },
        {
          "name": "First Neiro On Ethereum",
          "slug": "first-neiro-on-ethereum",
          "ticker": "NEIRO"
        },
        {
          "name": "Cheems (cheems.pet)",
          "slug": "bnb-cheems-pet",
          "ticker": "CHEEMS"
        },
        {
          "name": "NAVI Protocol",
          "slug": "navi-protocol",
          "ticker": "NAVX"
        },
        {
          "name": "Vector Smart Gas",
          "slug": "vitalik-smart-gas",
          "ticker": "VSG"
        },
        {
          "name": "Large Language Model",
          "slug": "large-language-model",
          "ticker": "LLM"
        },
        {
          "name": "REVOX",
          "slug": "revox",
          "ticker": "REX"
        },
        {
          "name": "Sentio Protocol",
          "slug": "sentio-protocol",
          "ticker": "SEN"
        },
        {
          "name": "Genesis Worlds",
          "slug": "p-genesis-worlds",
          "ticker": "GENESIS"
        },
        {
          "name": "Crash On Base",
          "slug": "crash-on-base",
          "ticker": "CRASH"
        },
        {
          "name": "Data Ownership Protocol",
          "slug": "data-ownership-protocol",
          "ticker": "DOP"
        },
        {
          "name": "Step App",
          "slug": "a-step-app",
          "ticker": "FITFI"
        },
        {
          "name": "zKML",
          "slug": "zkml",
          "ticker": "ZKML"
        },
        {
          "name": "Gravity",
          "slug": "gravity-token",
          "ticker": "G"
        },
        {
          "name": "Mother Iggy",
          "slug": "mother-iggy",
          "ticker": "MOTHER"
        },
        {
          "name": "Rebel Bots",
          "slug": "p-rebel-bots",
          "ticker": "RBLS"
        },
        {
          "name": "Philtoken",
          "slug": "philtoken",
          "ticker": "PHIL"
        },
        {
          "name": "Witch Token",
          "slug": "witch-token",
          "ticker": "WITCH"
        },
        {
          "name": "Giko Cat",
          "slug": "giko-cat",
          "ticker": "GIKO"
        },
        {
          "name": "BILLION•DOLLAR•CAT",
          "slug": "billion-dollar-cat",
          "ticker": "BDC"
        },
        {
          "name": "EigenLayer",
          "slug": "eigenlayer",
          "ticker": "EIGEN"
        },
        {
          "name": "Catizen",
          "slug": "catizen",
          "ticker": "CATI"
        },
        {
          "name": "Fwog",
          "slug": "fwog-solana",
          "ticker": "FWOG"
        },
        {
          "name": "Monkey Pox",
          "slug": "monkey-pox",
          "ticker": "POX"
        },
        {
          "name": "michi",
          "slug": "michi",
          "ticker": "$MICHI"
        },
        {
          "name": "NeuralAI",
          "slug": "neuralai",
          "ticker": "NEURAL"
        },
        {
          "name": "RabBitcoin",
          "slug": "rockyrabbit",
          "ticker": "RBTC"
        },
        {
          "name": "Humans.ai",
          "slug": "humans-ai",
          "ticker": "HEART"
        },
        {
          "name": "Devve",
          "slug": "devve",
          "ticker": "DEVVE"
        },
        {
          "name": "MANEKI",
          "slug": "maneki-coin",
          "ticker": "MANEKI"
        },
        {
          "name": "Doge Eat Doge",
          "slug": "doge-eat-doge",
          "ticker": "OMNOM"
        },
        {
          "name": "GMX [on Avalanche]",
          "slug": "a-gmx",
          "ticker": "GMX"
        },
        {
          "name": "Savings Dai",
          "slug": "savings-dai",
          "ticker": "SDAI"
        },
        {
          "name": "Magic Internet Money [on Arbitrum]",
          "slug": "arb-magic-internet-money",
          "ticker": "MIM"
        },
        {
          "name": "Render",
          "slug": "render",
          "ticker": "RENDER"
        },
        {
          "name": "SPACE ID [on BNB Chain]",
          "slug": "bnb-space-id",
          "ticker": "ID"
        },
        {
          "name": "Kasta",
          "slug": "p-kasta",
          "ticker": "KASTA"
        },
        {
          "name": "UTYABSWAP",
          "slug": "utya-black",
          "ticker": "UTYAB"
        },
        {
          "name": "Mystiko Network",
          "slug": "mystiko-network",
          "ticker": "XZK"
        },
        {
          "name": "Aura",
          "slug": "aura",
          "ticker": "AURA"
        },
        {
          "name": "Mr Miggles",
          "slug": "mr-miggles",
          "ticker": "MIGGLES"
        },
        {
          "name": "OX Coin [on Ethereum]",
          "slug": "ox-coin",
          "ticker": "OX"
        },
        {
          "name": "FractonX",
          "slug": "fracton-protocol",
          "ticker": "FT"
        },
        {
          "name": "Landwolf 0x67",
          "slug": "landwolf-coin",
          "ticker": "WOLF"
        },
        {
          "name": "BUBCAT",
          "slug": "bnb-bubcat",
          "ticker": "BUB"
        },
        {
          "name": "mini",
          "slug": "mini",
          "ticker": "MINI"
        },
        {
          "name": "Cook Finance",
          "slug": "cook-protocol",
          "ticker": "COOK"
        },
        {
          "name": "Worbli",
          "slug": "worbli",
          "ticker": "WBL"
        },
        {
          "name": "GXChain",
          "slug": "gxchain",
          "ticker": "GXC"
        },
        {
          "name": "Xaya",
          "slug": "xaya",
          "ticker": "CHI"
        },
        {
          "name": "SIRIN LABS Token",
          "slug": "sirin-labs-token",
          "ticker": "SRN"
        },
        {
          "name": "Fatcoin",
          "slug": "fatcoin",
          "ticker": "FAT"
        },
        {
          "name": "Decimated",
          "slug": "decimated",
          "ticker": "DIO"
        },
        {
          "name": "Krios",
          "slug": "krios",
          "ticker": "KRI"
        },
        {
          "name": "TrueChain",
          "slug": "truechain",
          "ticker": "TRUE"
        },
        {
          "name": "VisionX",
          "slug": "visionx",
          "ticker": "VNX"
        },
        {
          "name": "Rotharium",
          "slug": "rotharium",
          "ticker": "RTH"
        },
        {
          "name": "SyncFab",
          "slug": "syncfab",
          "ticker": "MFG"
        },
        {
          "name": "TTC",
          "slug": "ttc-protocol",
          "ticker": "TTC"
        },
        {
          "name": "Niftyx Protocol",
          "slug": "shroom-finance",
          "ticker": "SHROOM"
        },
        {
          "name": "Zeedex",
          "slug": "zeedex",
          "ticker": "ZDEX"
        },
        {
          "name": "Bitcoin Token",
          "slug": "bitcoin-token",
          "ticker": "BTK"
        },
        {
          "name": "Lukki",
          "slug": "lukki",
          "ticker": "LOT"
        },
        {
          "name": "Tether [on BNB]",
          "slug": "bnb-tether",
          "ticker": "USDT"
        },
        {
          "name": "Wrapped Bitcoin [on Optimism]",
          "slug": "o-wrapped-bitcoin",
          "ticker": "WBTC"
        },
        {
          "name": "Kujira [on Arbitrum]",
          "slug": "arb-kujira",
          "ticker": "KUJI"
        },
        {
          "name": "Kelp DAO Restaked ETH",
          "slug": "kelp-dao-restaked-eth",
          "ticker": "RSETH"
        },
        {
          "name": "Renzo Restaked ETH",
          "slug": "renzo-restaked-eth",
          "ticker": "EZETH"
        },
        {
          "name": "fanC",
          "slug": "fanc",
          "ticker": "FANC"
        },
        {
          "name": "pufETH",
          "slug": "pufeth",
          "ticker": "PUFETH"
        },
        {
          "name": "BADMAD ROBOTS",
          "slug": "p-drunk-robots",
          "ticker": "METAL"
        },
        {
          "name": "ZTX",
          "slug": "arb-ztx",
          "ticker": "ZTX"
        },
        {
          "name": "Creditcoin",
          "slug": "creditcoin",
          "ticker": "CTC"
        },
        {
          "name": "AllianceBlock Nexera",
          "slug": "allianceblock-nexera",
          "ticker": "NXRA"
        },
        {
          "name": "Polyhedra Network",
          "slug": "polyhedra-network",
          "ticker": "ZKJ"
        },
        {
          "name": "QORPO WORLD",
          "slug": "qorpo",
          "ticker": "QORPO"
        },
        {
          "name": "Avail",
          "slug": "avail",
          "ticker": "AVAIL"
        },
        {
          "name": "Blocksquare Token",
          "slug": "blocksquare-token",
          "ticker": "BST"
        },
        {
          "name": "Pikamoon",
          "slug": "pikamoon-pika",
          "ticker": "PIKA"
        },
        {
          "name": "Shido [New]",
          "slug": "shido-new",
          "ticker": "SHIDO"
        },
        {
          "name": "Torum",
          "slug": "torum",
          "ticker": "XTM"
        },
        {
          "name": "GameGPT",
          "slug": "duel",
          "ticker": "DUEL"
        },
        {
          "name": "Corite",
          "slug": "corite",
          "ticker": "CO"
        },
        {
          "name": "Neiro Ethereum",
          "slug": "neiro-eth",
          "ticker": "NEIRO"
        },
        {
          "name": "Bitkub Coin",
          "slug": "bitkub-coin",
          "ticker": "KUB"
        },
        {
          "name": "Mumu the Bull",
          "slug": "mumu-ing",
          "ticker": "MUMU"
        },
        {
          "name": "Artificial Superintelligence Alliance",
          "slug": "artificial-superintelligence-alliance",
          "ticker": "FET"
        },
        {
          "name": "Billy",
          "slug": "billy",
          "ticker": "BILLY"
        },
        {
          "name": "Covalent X Token",
          "slug": "covalent-x",
          "ticker": "CXT"
        },
        {
          "name": "Gigachad",
          "slug": "gigachad-meme",
          "ticker": "GIGA"
        },
        {
          "name": "Cornucopias",
          "slug": "cornucopias",
          "ticker": "COPI"
        },
        {
          "name": "Contango",
          "slug": "contango",
          "ticker": "TANGO"
        },
        {
          "name": "WeSendit",
          "slug": "bnb-wesendit",
          "ticker": "WSI"
        },
        {
          "name": "PirateCash",
          "slug": "bnb-piratecash",
          "ticker": "PIRATE"
        },
        {
          "name": "Bloktopia",
          "slug": "p-bloktopia",
          "ticker": "BLOK"
        },
        {
          "name": "Pax Dollar [on BNB]",
          "slug": "bnb-paxos-standard",
          "ticker": "USDP"
        },
        {
          "name": "Aave [on Polygon]",
          "slug": "p-aave",
          "ticker": "AAVE"
        },
        {
          "name": "SATS (Ordinals)",
          "slug": "sats-ordinals",
          "ticker": "SATS"
        },
        {
          "name": "Partisia Blockchain",
          "slug": "partisia-blockchain",
          "ticker": "MPC"
        },
        {
          "name": "Brett (ETH)",
          "slug": "brett-coin",
          "ticker": "BRETT"
        },
        {
          "name": "BASE",
          "slug": "swapbased-base",
          "ticker": "BASE"
        },
        {
          "name": "FU Coin",
          "slug": "bnb-fu-coin",
          "ticker": "FU"
        },
        {
          "name": "SIGMA",
          "slug": "sigma-sol",
          "ticker": "SIGMA"
        },
        {
          "name": "Harambe",
          "slug": "harambe-on-solana",
          "ticker": "HARAMBE"
        },
        {
          "name": "Doland Tremp",
          "slug": "doland-tremp",
          "ticker": "TREMP"
        },
        {
          "name": "Bobaoppa",
          "slug": "bobaoppa",
          "ticker": "BOBAOPPA"
        },
        {
          "name": "Law Blocks (AI)",
          "slug": "law-blocks",
          "ticker": "LBT"
        },
        {
          "name": "Goatseus Maximus",
          "slug": "goatseus-maximus",
          "ticker": "GOAT"
        },
        {
          "name": "Mr Mint",
          "slug": "bnb-mr-mint",
          "ticker": "MNT"
        },
        {
          "name": "Skibidi Toilet",
          "slug": "skibidi-toilet-memecoin",
          "ticker": "SKBDI"
        },
        {
          "name": "io.net",
          "slug": "io-net",
          "ticker": "IO"
        },
        {
          "name": "Shrub",
          "slug": "shrub",
          "ticker": "SHRUB"
        },
        {
          "name": "ai16z",
          "slug": "ai16z",
          "ticker": "AI16Z"
        },
        {
          "name": "Comedian",
          "slug": "comedian",
          "ticker": "BAN"
        },
        {
          "name": "Virtuals Protocol",
          "slug": "virtual-protocol",
          "ticker": "VIRTUAL"
        },
        {
          "name": "Destra Network",
          "slug": "destra-network",
          "ticker": "DSYNC"
        },
        {
          "name": "TARS Protocol",
          "slug": "tars-protocol",
          "ticker": "TAI"
        },
        {
          "name": "Polytrade [on Polygon]",
          "slug": "p-polytrade",
          "ticker": "TRADE"
        },
        {
          "name": "Horizon Protocol",
          "slug": "bnb-horizon-protocol",
          "ticker": "HZN"
        },
        {
          "name": "aixbt by Virtuals",
          "slug": "aixbt",
          "ticker": "AIXBT"
        },
        {
          "name": "Open Loot",
          "slug": "open-loot",
          "ticker": "OL"
        },
        {
          "name": "Laika AI",
          "slug": "laika-ai",
          "ticker": "LKI"
        },
        {
          "name": "Jen-Hsun Huang",
          "slug": "jen-hsun-huang",
          "ticker": "JHH"
        },
        {
          "name": "Puffer",
          "slug": "puffer",
          "ticker": "PUFFER"
        },
        {
          "name": "Pixer Eternity",
          "slug": "bnb-pixer-eternity",
          "ticker": "PXT"
        },
        {
          "name": "Satoshi Airline",
          "slug": "p-satoshi-airline",
          "ticker": "JET"
        },
        {
          "name": "Gems",
          "slug": "gems-vip",
          "ticker": "GEMS"
        },
        {
          "name": "SelfieDogCoin",
          "slug": "selfiedogcoin",
          "ticker": "SELFIE"
        },
        {
          "name": "nubcat",
          "slug": "nubcat",
          "ticker": "NUB"
        },
        {
          "name": "UNI",
          "slug": "uni-sui",
          "ticker": "UNI"
        },
        {
          "name": "CoW Protocol",
          "slug": "cow-protocol",
          "ticker": "COW"
        },
        {
          "name": "CARV",
          "slug": "arb-carv",
          "ticker": "CARV"
        },
        {
          "name": "Hasbulla's Cat",
          "slug": "hasbullas-cat",
          "ticker": "BARSIK"
        },
        {
          "name": "littlemanyu",
          "slug": "littlemanyu",
          "ticker": "MANYU"
        },
        {
          "name": "Dasha",
          "slug": "dasha",
          "ticker": "VVAIFU"
        },
        {
          "name": "RealGOAT",
          "slug": "real-goat",
          "ticker": "RGOAT"
        },
        {
          "name": "LumiWave",
          "slug": "lumiwave",
          "ticker": "LWA"
        },
        {
          "name": "Tokamak Network",
          "slug": "tokamak-network",
          "ticker": "TOKAMAK"
        },
        {
          "name": "Ethernity Chain",
          "slug": "ethernity-chain",
          "ticker": "EPIC"
        },
        {
          "name": "TAOCat by Virtuals & Masa",
          "slug": "taocat-by-virtuals-masa",
          "ticker": "TAOCAT"
        },
        {
          "name": "XION",
          "slug": "xion",
          "ticker": "XION"
        },
        {
          "name": "Swarms",
          "slug": "swarms",
          "ticker": "SWARMS"
        },
        {
          "name": "sekoia by Virtuals",
          "slug": "sekoia-by-virtuals",
          "ticker": "SEKOIA"
        },
        {
          "name": "Talent Protocol",
          "slug": "talent-protocol",
          "ticker": "TALENT"
        },
        {
          "name": "OFFICIAL TRUMP",
          "slug": "official-trump",
          "ticker": "TRUMP"
        },
        {
          "name": "Planet IX(formerly IX token)",
          "slug": "ix-token",
          "ticker": "IXT"
        },
        {
          "name": "Pwease",
          "slug": "pwease",
          "ticker": "PWEASE"
        },
        {
          "name": "M3M3",
          "slug": "m3m3",
          "ticker": "M3M3"
        },
        {
          "name": "Foxy",
          "slug": "foxy",
          "ticker": "FOXY"
        },
        {
          "name": "USDP Stablecoin",
          "slug": "usdp",
          "ticker": "USDP"
        },
        {
          "name": "SPX69000",
          "slug": "spx-net",
          "ticker": "SPX"
        },
        {
          "name": "Project89",
          "slug": "project89",
          "ticker": "PROJECT89"
        },
        {
          "name": "GRIFFAIN",
          "slug": "griffain",
          "ticker": "GRIFFAIN"
        },
        {
          "name": "Eliza (ai16zeliza)",
          "slug": "ai16zeliza",
          "ticker": "ELIZA"
        },
        {
          "name": "XUSD",
          "slug": "straitsx-xusd",
          "ticker": "XUSD"
        },
        {
          "name": "doginme",
          "slug": "doginme",
          "ticker": "DOGINME"
        },
        {
          "name": "VaderAI by Virtuals",
          "slug": "vaderai-by-virtuals",
          "ticker": "VADER"
        },
        {
          "name": "ORBIT",
          "slug": "orbit-ai",
          "ticker": "GRIFT"
        },
        {
          "name": "Bubblemaps",
          "slug": "bubblemaps",
          "ticker": "BMT"
        },
        {
          "name": "GoldPro Token",
          "slug": "p-ipmb",
          "ticker": "GPRO"
        },
        {
          "name": "Hive AI",
          "slug": "hive-ai",
          "ticker": "BUZZ"
        },
        {
          "name": "Solv Protocol",
          "slug": "bnb-solv-protocol",
          "ticker": "SOLV"
        },
        {
          "name": "Melania Meme",
          "slug": "melania-meme",
          "ticker": "MELANIA"
        },
        {
          "name": "Legacy Token",
          "slug": "bnb-legacy-network",
          "ticker": "LGCT"
        },
        {
          "name": "World Liberty Financial USD",
          "slug": "usd1",
          "ticker": "USD1"
        },
        {
          "name": "Saros",
          "slug": "saros",
          "ticker": "SAROS"
        },
        {
          "name": "Mubarak",
          "slug": "bnb-mubarak",
          "ticker": "MUBARAK"
        },
        {
          "name": "WalletConnect Token",
          "slug": "walletconnect-token",
          "ticker": "WCT"
        },
        {
          "name": "Plume",
          "slug": "plume",
          "ticker": "PLUME"
        },
        {
          "name": "Nillion",
          "slug": "nillion",
          "ticker": "NIL"
        },
        {
          "name": "Bedrock",
          "slug": "bnb-bedrock-dao",
          "ticker": "BR"
        },
        {
          "name": "Epic Chain",
          "slug": "epic-chain",
          "ticker": "EPIC"
        },
        {
          "name": "Sonic (prev. FTM)",
          "slug": "sonic",
          "ticker": "S"
        },
        {
          "name": "KernelDAO",
          "slug": "kerneldao",
          "ticker": "KERNEL"
        },
        {
          "name": "GUNZ",
          "slug": "gunz",
          "ticker": "GUN"
        },
        {
          "name": "Jambo",
          "slug": "jambo",
          "ticker": "J"
        },
        {
          "name": "Avalon Labs (AVL)",
          "slug": "avalon-labs-avl",
          "ticker": "AVL"
        },
        {
          "name": "HashKey Platform Token",
          "slug": "hashkey-platform-token",
          "ticker": "HSK"
        },
        {
          "name": "Orbiter Finance",
          "slug": "orbiter-finance",
          "ticker": "OBT"
        },
        {
          "name": "Kekius Maximus (kekiusmaximus.vip)",
          "slug": "kekius-maximus-vip",
          "ticker": "KEKIUS"
        },
        {
          "name": "Vine Coin",
          "slug": "vine-coin",
          "ticker": "VINE"
        },
        {
          "name": "Fuel Network",
          "slug": "fuel-network",
          "ticker": "FUEL"
        },
        {
          "name": "Tajir Tech Hub",
          "slug": "tajir-tech-hub",
          "ticker": "TJRM"
        },
        {
          "name": "Xterio",
          "slug": "bnb-xterio",
          "ticker": "XTER"
        },
        {
          "name": "StakeStone",
          "slug": "bnb-stakestone",
          "ticker": "STO"
        },
        {
          "name": "Retard Finder Coin",
          "slug": "retard-finder-coin",
          "ticker": "RFC"
        },
        {
          "name": "StablR Euro",
          "slug": "stablr-euro",
          "ticker": "EURR"
        },
        {
          "name": "CrossFi",
          "slug": "crossfinance",
          "ticker": "XFI"
        },
        {
          "name": "Humanode",
          "slug": "humanode",
          "ticker": "HMND"
        },
        {
          "name": "Fluid",
          "slug": "instadapp",
          "ticker": "FLUID"
        },
        {
          "name": "Bittensor",
          "slug": "bittensor",
          "ticker": "TAO"
        },
        {
          "name": "Travala",
          "slug": "ava",
          "ticker": "AVA"
        },
        {
          "name": "Pythia",
          "slug": "pythia",
          "ticker": "PYTHIA"
        },
        {
          "name": "GATSBY (gatsby.fi)",
          "slug": "gatsby-fi",
          "ticker": "GATSBY"
        },
        {
          "name": "Digimon",
          "slug": "digimon",
          "ticker": "DIGIMON"
        },
        {
          "name": "UNIPOLY",
          "slug": "unipoly",
          "ticker": "UNP"
        },
        {
          "name": "Venice Token",
          "slug": "venice-token",
          "ticker": "VVV"
        },
        {
          "name": "Coinbase Wrapped BTC",
          "slug": "coinbase-wrapped-btc",
          "ticker": "CBBTC"
        },
        {
          "name": "Jelly-My-Jelly",
          "slug": "jelly-my-jelly",
          "ticker": "JELLYJELLY"
        },
        {
          "name": "Cointel",
          "slug": "a-cointel",
          "ticker": "COLS"
        },
        {
          "name": "Alpha",
          "slug": "realalphacoin",
          "ticker": "ALPHA"
        },
        {
          "name": "Clustr Labs",
          "slug": "clustr-labs",
          "ticker": "CLUSTR"
        },
        {
          "name": "GOHOME",
          "slug": "gohome",
          "ticker": "GOHOME"
        },
        {
          "name": "Maple Finance",
          "slug": "maple-finance",
          "ticker": "SYRUP"
        },
        {
          "name": "Particle Network",
          "slug": "bnb-particle-network",
          "ticker": "PARTI"
        },
        {
          "name": "Babylon",
          "slug": "babylon",
          "ticker": "BABY"
        },
        {
          "name": "CryptoTradingFund",
          "slug": "cryptotradingfund",
          "ticker": "CTF"
        },
        {
          "name": "Injective",
          "slug": "injective-protocol",
          "ticker": "INJ"
        },
        {
          "name": "Synternet",
          "slug": "synternet-v2",
          "ticker": "SYNT"
        },
        {
          "name": "Tutorial",
          "slug": "tutorial",
          "ticker": "TUT"
        },
        {
          "name": "Test",
          "slug": "bnb-test-token-bsc",
          "ticker": "TST"
        },
        {
          "name": "Mansory",
          "slug": "mansory",
          "ticker": "MNSRY"
        },
        {
          "name": "Jupiter Perps LP",
          "slug": "jupiter-perps-lp",
          "ticker": "JLP"
        },
        {
          "name": "Chain-key Ethereum",
          "slug": "chain-key-ethereum",
          "ticker": "CKETH"
        },
        {
          "name": "SUPRA",
          "slug": "supra",
          "ticker": "SUPRA"
        },
        {
          "name": "VICE Token",
          "slug": "vice-token",
          "ticker": "VICE"
        },
        {
          "name": " Chain-key USDT",
          "slug": "chain-key-usdt",
          "ticker": "CKUSDT"
        },
        {
          "name": "  Chain-key USDC",
          "slug": "chain-key-usdc",
          "ticker": "CKUSDC"
        },
        {
          "name": "The Arena",
          "slug": "a-the-arena",
          "ticker": "ARENA"
        },
        {
          "name": "GAMA Coin",
          "slug": "gama-coin",
          "ticker": "GAMA"
        },
        {
          "name": "Vertical AI",
          "slug": "vertical-ai",
          "ticker": "VERTAI"
        },
        {
          "name": "Seraph",
          "slug": "seraph",
          "ticker": "SERAPH"
        },
        {
          "name": "Zigchain",
          "slug": "zigcoin",
          "ticker": "ZIG"
        },
        {
          "name": "Keyboard Cat",
          "slug": "keyboard-cat",
          "ticker": "KEYCAT"
        },
        {
          "name": "SideShift Token",
          "slug": "sideshift-token",
          "ticker": "XAI"
        },
        {
          "name": "EarthMeta",
          "slug": "earthmeta",
          "ticker": "EMT"
        },
        {
          "name": "Camino Network",
          "slug": "camino-network",
          "ticker": "CAM"
        },
        {
          "name": "Kappa",
          "slug": "kappa",
          "ticker": "KAPPA"
        },
        {
          "name": "Central African Republic Meme",
          "slug": "central-african-republic-meme",
          "ticker": "CAR"
        },
        {
          "name": "MyShell",
          "slug": "myshell",
          "ticker": "SHELL"
        },
        {
          "name": "Stool Prisondente",
          "slug": "stool-prisondente",
          "ticker": "JAILSTOOL"
        },
        {
          "name": "PinLink",
          "slug": "pinlink",
          "ticker": "PIN"
        },
        {
          "name": "20wstETH-80AAVE",
          "slug": "20wstETH80AAVE",
          "ticker": "20wstETH80AAVE"
        },
        {
          "name": "Story",
          "slug": "story-protocol",
          "ticker": "IP"
        },
        {
          "name": "Solayer",
          "slug": "solayer",
          "ticker": "LAYER"
        },
        {
          "name": "Kamino Finance",
          "slug": "kamino-finance",
          "ticker": "KMNO"
        },
        {
          "name": "Ripple USD",
          "slug": "ripple-usd",
          "ticker": "RLUSD"
        },
        {
          "name": "CZ'S Dog (broccoli.gg)",
          "slug": "czsdog-broccoli",
          "ticker": "BROCCOLI"
        },
        {
          "name": "Global Commercial Business",
          "slug": "global-commercial-business",
          "ticker": "GCB"
        },
        {
          "name": "Zircuit",
          "slug": "zircuit",
          "ticker": "ZRC"
        },
        {
          "name": "Broccoli (broccolibnb.org)",
          "slug": "broccoli-czdog",
          "ticker": "BROCCOLI"
        },
        {
          "name": "Newton",
          "slug": "newton",
          "ticker": "AB"
        },
        {
          "name": "Astherus USDF",
          "slug": "usdf",
          "ticker": "USDF"
        },
        {
          "name": "KAITO",
          "slug": "kaito",
          "ticker": "KAITO"
        },
        {
          "name": "LIBRA",
          "slug": "libra-viva-la-libertad-project",
          "ticker": "LIBRA"
        },
        {
          "name": "Vameon",
          "slug": "vameon",
          "ticker": "VON"
        },
        {
          "name": "Dohrnii",
          "slug": "dohrnii",
          "ticker": "DHN"
        },
        {
          "name": "Pi",
          "slug": "pi",
          "ticker": "PI"
        },
        {
          "name": "Walrus",
          "slug": "walrus-xyz",
          "ticker": "WAL"
        },
        {
          "name": "Acet",
          "slug": "acet",
          "ticker": "ACT"
        },
        {
          "name": "would",
          "slug": "wouldmeme",
          "ticker": "WOULD"
        },
        {
          "name": "KiloEx",
          "slug": "bnb-kiloex",
          "ticker": "KILO"
        },
        {
          "name": "Car",
          "slug": "car",
          "ticker": "CAR"
        },
        {
          "name": "Gochujangcoin",
          "slug": "gochujangcoin",
          "ticker": "GOCHU"
        },
        {
          "name": "Derive",
          "slug": "derive",
          "ticker": "DRV"
        },
        {
          "name": "siren",
          "slug": "siren-bsc",
          "ticker": "SIREN"
        },
        {
          "name": "ShibaBitcoin",
          "slug": "shibabitcoin",
          "ticker": "SBBTC"
        },
        {
          "name": "Pain (paintoken.com)",
          "slug": "pain",
          "ticker": "PAIN"
        },
        {
          "name": "Entangle",
          "slug": "entangle",
          "ticker": "NTGL"
        },
        {
          "name": "Roam",
          "slug": "roam",
          "ticker": "ROAM"
        },
        {
          "name": "DIAM",
          "slug": "diam",
          "ticker": "DIAM"
        },
        {
          "name": "Heima",
          "slug": "heima",
          "ticker": "HEI"
        }
      ]
    }
  }
