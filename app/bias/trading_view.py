import json
import requests
from bias.interface import BiasInterface, BiasRequest, BiasResponse, BiasType
from utils import get_logger

logger = get_logger()

class TradingView(BiasInterface):
    ignore = True

    def moving_averages(self, signals):
        buy_count = 0
        sell_count = 0
        neutral_count = 0

        ma_signals = [
            'EMA10', 'SMA10', 'EMA20', 'SMA20', 'EMA30', 'SMA30',
            'EMA50', 'SMA50', 'EMA100', 'SMA100', 'EMA200', 'SMA200',
            'Ichimoku.BLine', 'VWMA', 'HullMA9'
        ]

        for ma in ma_signals:
            value = signals.get(f'{ma}|5', 0)
            close_price = signals.get('close|5', 0)
            if value > close_price:
                sell_count += 1
            elif value < close_price:
                buy_count += 1
            else:
                neutral_count += 1            
        logger.info("Moving Averages:")
        logger.info(f"Sell: {sell_count}, Neutral: {neutral_count}, Buy: {buy_count}")

        return buy_count, sell_count, neutral_count

# The thresholds and logic in the function are based on common technical analysis principles used by traders. Here's where those values generally come from:
#
# 1. **RSI (Relative Strength Index)**:  
#    - Over 70: Overbought → **Sell**
#    - Below 30: Oversold → **Buy**
#    - In between: **Neutral**  
#    
# 2. **Stochastic %K**:  
#    - Over 80: Overbought → **Sell**  
#    - Below 20: Oversold → **Buy**  
#    - In between: **Neutral**  
#
# 3. **CCI (Commodity Channel Index)**:  
#    - Over 100: Strong Uptrend → **Buy**  
#    - Below -100: Strong Downtrend → **Sell**  
#    - In between: **Neutral**  
#
# 4. **ADX (Average Directional Index)**:  
#    - ADX above 25: Strong trend → Determine using +DI and -DI  
#    - +DI > -DI → **Buy**  
#    - -DI > +DI → **Sell**  
#    - ADX below 25: Weak trend → **Neutral**  
#
# 5. **AO (Awesome Oscillator)**:  
#    - Positive → **Buy**  
#    - Negative → **Sell**  
#
# 6. **Momentum**:  
#    - Positive → **Buy**  
#    - Negative → **Sell**  
#
# 7. **MACD**:  
#    - MACD > Signal → **Buy**  
#    - MACD < Signal → **Sell**  
#    - Equal → **Neutral**  
#
# 8. **Stochastic RSI**:  
#    - Over 80: Overbought → **Sell**  
#    - Below 20: Oversold → **Buy**  
#    - In between: **Neutral**  
#
# 9. **Williams %R**:  
#    - Below -80: Oversold → **Buy**  
#    - Above -20: Overbought → **Sell**  
#    - In between: **Neutral**  
#
# 10. **Bull Bear Power**:  
#     - Positive → **Buy**  
#     - Negative → **Sell**  
#
# 11. **Ultimate Oscillator (UO)**:  
#     - Over 70: Overbought → **Sell**  
#     - Below 30: Oversold → **Buy**  
#     - In between: **Neutral**  

    def analyse(self, signals):
        buy_count = 0
        sell_count = 0
        neutral_count = 0

        # Relative Strength Index (RSI)
        rsi = signals.get('RSI|5', 50)
        logger.info(f"RSI: {rsi}")
        if rsi > 70:
            sell_count += 1
            logger.info("Sell")
        elif rsi < 30:
            buy_count += 1
            logger.info("Buy")
        else:
            neutral_count += 1
            logger.info("Neutral")

        # Stochastic %K
        stochastic_k = signals.get('Stoch.K|5', 50)
        logger.info(f"Stochastic %K: {stochastic_k}")
        if stochastic_k > 80:
            sell_count += 1
            logger.info("Sell")
        elif stochastic_k < 20:
            buy_count += 1
            logger.info("Buy")
        else:
            neutral_count += 1
            logger.info("Neutral")

        # Commodity Channel Index (CCI)
        cci = signals.get('CCI20|5', 0)
        logger.info(f"CCI: {cci}")
        if cci > 100:
            buy_count += 1
            logger.info("Buy")
        elif cci < -100:
            sell_count += 1
            logger.info("Sell")
        else:
            neutral_count += 1
            logger.info("Neutral")

        # Average Directional Index (ADX)
        adx = signals.get('ADX|5', 0)
        logger.info(f"ADX: {adx}")
        adx_pos_di = signals.get('ADX+DI|5', 0)
        adx_neg_di = signals.get('ADX-DI|5', 0)
        if adx > 25:
            if adx_pos_di > adx_neg_di:
                buy_count += 1
                logger.info("Buy")
            else:
                sell_count += 1
                logger.info("Sell")
        else:
            neutral_count += 1
            logger.info("Neutral")

        # Awesome Oscillator (AO)
        ao = signals.get('AO|5', 0)
        logger.info(f"AO: {ao}")
        if ao > 0:
            buy_count += 1
            logger.info("Buy")
        else:
            sell_count += 1
            logger.info("Sell")

        # Momentum (Mom)
        momentum = signals.get('Mom|5', 0)
        logger.info(f"Momentum: {momentum}")
        if momentum > 0:
            buy_count += 1
            logger.info("Buy")
        else:
            sell_count += 1
            logger.info("Sell")

        # MACD (Moving Average Convergence Divergence)
        macd = signals.get('MACD.macd|5', 0)
        logger.info(f"MACD: {macd}")
        macd_signal = signals.get('MACD.signal|5', 0)
        logger.info(f"MACD Signal: {macd_signal}")
        if macd > macd_signal:
            buy_count += 1
            logger.info("Buy")
        elif macd < macd_signal:
            sell_count += 1
            logger.info("Sell")
        else:
            neutral_count += 1
            logger.info("Neutral")

        # Stochastic RSI
        stoch_rsi_k = signals.get('Stoch.RSI.K|5', 50)
        logger.info(f"Stochastic RSI %K: {stoch_rsi_k}")
        if stoch_rsi_k > 80:
            sell_count += 1
            logger.info("Sell")
        elif stoch_rsi_k < 20:
            buy_count += 1
            logger.info("Buy")
        else:
            neutral_count += 1
            logger.info("Neutral")

        # Williams Percent Range (W.R)
        w_r = signals.get('W.R|5', 0)
        logger.info(f"Williams %R: {w_r}")
        if w_r < -80:
            buy_count += 1
            logger.info("Buy")
        elif w_r > -20:
            sell_count += 1
            logger.info("Sell")
        else:
            neutral_count += 1
            logger.info("Neutral")

        # Bull Bear Power (BBPower)
        bb_power = signals.get('BBPower|5', 0)
        logger.info(f"Bull Bear Power: {bb_power}")
        if bb_power > 0:
            buy_count += 1
            logger.info("Buy")
        else:
            sell_count += 1
            logger.info("Sell")

        # Ultimate Oscillator (UO)
        uo = signals.get('UO|5', 50)
        logger.info(f"Ultimate Oscillator: {uo}")
        if uo > 70:
            sell_count += 1
            logger.info("Sell")
        elif uo < 30:
            buy_count += 1
            logger.info("Buy")
        else:
            neutral_count += 1
            logger.info("Neutral")

        logger.info("Oscillators:")
        logger.info(f"Sell: {sell_count}, Neutral: {neutral_count}, Buy: {buy_count}")

        buy_count_ma, sell_count_ma, neutral_count_ma = self.moving_averages(signals)
        buy_count += buy_count_ma
        sell_count += sell_count_ma
        neutral_count += neutral_count_ma

        logger.info("totals")
        logger.info(f"Sell: {sell_count}, Neutral: {neutral_count}, Buy: {buy_count}")
        total = buy_count + sell_count + neutral_count
        if buy_count / total >= 0.8:
            return BiasType.LONG
        elif sell_count / total >= 0.8:
            return BiasType.SHORT
        else:
            return BiasType.NEUTRAL

    def bias(self, biasRequest: BiasRequest) -> BiasResponse:
        endpoint = f"https://scanner.tradingview.com/symbol?symbol=BITSTAMP%3A{biasRequest.symbol}USD&fields=Recommend.Other%7C5%2CRecommend.All%7C5%2CRecommend.MA%7C5%2CRSI%7C5%2CRSI%5B1%5D%7C5%2CStoch.K%7C5%2CStoch.D%7C5%2CStoch.K%5B1%5D%7C5%2CStoch.D%5B1%5D%7C5%2CCCI20%7C5%2CCCI20%5B1%5D%7C5%2CADX%7C5%2CADX%2BDI%7C5%2CADX-DI%7C5%2CADX%2BDI%5B1%5D%7C5%2CADX-DI%5B1%5D%7C5%2CAO%7C5%2CAO%5B1%5D%7C5%2CAO%5B2%5D%7C5%2CMom%7C5%2CMom%5B1%5D%7C5%2CMACD.macd%7C5%2CMACD.signal%7C5%2CRec.Stoch.RSI%7C5%2CStoch.RSI.K%7C5%2CRec.WR%7C5%2CW.R%7C5%2CRec.BBPower%7C5%2CBBPower%7C5%2CRec.UO%7C5%2CUO%7C5%2CEMA10%7C5%2Cclose%7C5%2CSMA10%7C5%2CEMA20%7C5%2CSMA20%7C5%2CEMA30%7C5%2CSMA30%7C5%2CEMA50%7C5%2CSMA50%7C5%2CEMA100%7C5%2CSMA100%7C5%2CEMA200%7C5%2CSMA200%7C5%2CRec.Ichimoku%7C5%2CIchimoku.BLine%7C5%2CRec.VWMA%7C5%2CVWMA%7C5%2CRec.HullMA9%7C5%2CHullMA9%7C5%2CPivot.M.Classic.R3%7C5%2CPivot.M.Classic.R2%7C5%2CPivot.M.Classic.R1%7C5%2CPivot.M.Classic.Middle%7C5%2CPivot.M.Classic.S1%7C5%2CPivot.M.Classic.S2%7C5%2CPivot.M.Classic.S3%7C5%2CPivot.M.Fibonacci.R3%7C5%2CPivot.M.Fibonacci.R2%7C5%2CPivot.M.Fibonacci.R1%7C5%2CPivot.M.Fibonacci.Middle%7C5%2CPivot.M.Fibonacci.S1%7C5%2CPivot.M.Fibonacci.S2%7C5%2CPivot.M.Fibonacci.S3%7C5%2CPivot.M.Camarilla.R3%7C5%2CPivot.M.Camarilla.R2%7C5%2CPivot.M.Camarilla.R1%7C5%2CPivot.M.Camarilla.Middle%7C5%2CPivot.M.Camarilla.S1%7C5%2CPivot.M.Camarilla.S2%7C5%2CPivot.M.Camarilla.S3%7C5%2CPivot.M.Woodie.R3%7C5%2CPivot.M.Woodie.R2%7C5%2CPivot.M.Woodie.R1%7C5%2CPivot.M.Woodie.Middle%7C5%2CPivot.M.Woodie.S1%7C5%2CPivot.M.Woodie.S2%7C5%2CPivot.M.Woodie.S3%7C5%2CPivot.M.Demark.R1%7C5%2CPivot.M.Demark.Middle%7C5%2CPivot.M.Demark.S1%7C5&no_404=true&label-product=popup-technicals"
        response = requests.get(endpoint)
        data = response.json()
        bias = self.analyse(data)
        return BiasResponse(bias=bias)

# logger.info(TradingView().bias(BiasRequest(symbol="BTC")))
