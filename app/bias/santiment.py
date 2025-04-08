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
    def __init__(self):
        self.api_key = os.getenv("SANTIMENT_API_KEY", "")
        self.metric = "sentiment_weighted_total_1d_v2"

    def iso_date(self, days_ago: int = 0) -> str:
        dt = datetime.utcnow() - timedelta(days=days_ago)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    def fetch_sentiment_data(self, symbol: str) -> Optional[list[float]]:
        headers = {
            "Content-Type": "application/graphql",
            "Authorization": f"Apikey {self.api_key}",
        }

        SantimentFromTimeDaysAgo = int(get_config("SantimentFromTimeDaysAgo", 36))
        SantimentToTimeDaysAgo = int(get_config("SantimentToTimeDaysAgo", 30))
        from_time = self.iso_date(SantimentFromTimeDaysAgo)
        to_time = self.iso_date(SantimentToTimeDaysAgo)

        query = f"""
        {{
            getMetric(metric: "{self.metric}") {{
                timeseriesData(
                    slug: "bitcoin",
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
            logger.info(data)

            points = data["data"]["getMetric"]["timeseriesData"]
            values = [point["value"] for point in points if point["value"] is not None]

            logger.info(f"Fetched {len(values)} sentiment values")
            return values[-6:]  # 5 for average, 1 for latest
        except Exception as e:
            logger.exception(f"Error fetching data from Santiment: {e}")
            return None

    def bias(self, biasRequest: BiasRequest) -> BiasResponse:
        threshold = float(get_config("SantimentThreshold", 1.0))
        values = self.fetch_sentiment_data(biasRequest.symbol)

        if values is None or len(values) < 6:
            return BiasResponse(
                bias=BiasType.NEUTRAL,
                error="Not enough data to determine signal.",
                usedSymbol=True,
                reason="Santiment API returned insufficient or invalid data."
            )

        prev5 = values[:-1]
        latest = values[-1]
        avg = sum(prev5) / len(prev5)
        diff = latest - avg

        logger.info(f"Latest sentiment: {latest:.2f}")
        logger.info(f"5-day average sentiment: {avg:.2f}")
        logger.info(f"Difference: {diff:.2f}, Threshold: {threshold}")

        if diff > threshold:
            signal = BiasType.LONG
        elif diff < -threshold:
            signal = BiasType.SHORT
        else:
            signal = BiasType.NEUTRAL

        reason = f"Latest: {latest:.2f}, Avg: {avg:.2f}, Diff: {diff:.2f}, Threshold: {threshold:.2f}"

        return BiasResponse(
            bias=signal,
            usedSymbol=True,
            reason=reason
        )

