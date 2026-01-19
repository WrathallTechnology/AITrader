"""
Crypto Scanner - Automatically discovers and ranks crypto trading opportunities.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np

from src.client import AlpacaClient
from src.data.processor import DataProcessor

logger = logging.getLogger(__name__)


@dataclass
class CryptoOpportunity:
    """Represents a crypto trading opportunity."""
    symbol: str
    price: float
    change_24h: float  # Percentage
    volume_24h: float
    volatility: float  # Annualized
    rsi: float
    trend_strength: float  # -1 to 1
    score: float  # Overall opportunity score
    reason: str


class CryptoScanner:
    """
    Scans and ranks crypto pairs to find the best trading opportunities.
    """

    # All crypto pairs available on Alpaca
    ALL_CRYPTO_PAIRS = [
        "BTC/USD", "ETH/USD", "LTC/USD", "BCH/USD",
        "LINK/USD", "UNI/USD", "AAVE/USD", "SUSHI/USD",
        "YFI/USD", "MKR/USD", "COMP/USD", "SNX/USD",
        "CRV/USD", "BAL/USD", "GRT/USD", "BAT/USD",
        "DOT/USD", "SOL/USD", "AVAX/USD", "MATIC/USD",
        "ATOM/USD", "ALGO/USD", "XLM/USD", "XTZ/USD",
        "DOGE/USD", "SHIB/USD", "ADA/USD", "TRX/USD",
    ]

    def __init__(
        self,
        client: AlpacaClient,
        min_volume_usd: float = 100_000,  # Minimum 24h volume
        min_price: float = 0.0001,  # Minimum price
        max_pairs: int = 10,  # Maximum pairs to return
    ):
        self.client = client
        self.min_volume_usd = min_volume_usd
        self.min_price = min_price
        self.max_pairs = max_pairs

    def scan_all(self) -> list[CryptoOpportunity]:
        """
        Scan all crypto pairs and return ranked opportunities.
        """
        logger.info(f"Scanning {len(self.ALL_CRYPTO_PAIRS)} crypto pairs...")
        opportunities = []

        for symbol in self.ALL_CRYPTO_PAIRS:
            try:
                opp = self._analyze_pair(symbol)
                if opp:
                    opportunities.append(opp)
            except Exception as e:
                logger.debug(f"Error analyzing {symbol}: {e}")

        # Sort by score (highest first)
        opportunities.sort(key=lambda x: x.score, reverse=True)

        logger.info(f"Found {len(opportunities)} tradeable pairs")
        return opportunities[:self.max_pairs]

    def _analyze_pair(self, symbol: str) -> Optional[CryptoOpportunity]:
        """Analyze a single crypto pair."""
        try:
            # Fetch recent data
            bars = self.client.get_crypto_bars(
                symbols=[symbol],
                timeframe="1Hour",
                start=datetime.now() - timedelta(days=7),
                end=datetime.now(),
            )

            if not bars or symbol not in bars:
                logger.debug(f"{symbol}: No data returned")
                return None

            df = bars[symbol]
            if len(df) < 24:  # Need at least 24 hours of data
                logger.debug(f"{symbol}: Only {len(df)} bars (need 24)")
                return None

            # Add indicators
            df = DataProcessor.add_technical_indicators(df)

            # Get latest values
            latest = df.iloc[-1]
            price = float(latest["close"])

            # Filter by minimum price
            if price < self.min_price:
                logger.debug(f"{symbol}: Price ${price} below min ${self.min_price}")
                return None

            # Calculate 24h metrics
            df_24h = df.tail(24)
            volume_24h = float(df_24h["volume"].sum()) * price
            change_24h = (price - float(df_24h["close"].iloc[0])) / float(df_24h["close"].iloc[0]) * 100

            # Filter by minimum volume
            if volume_24h < self.min_volume_usd:
                logger.debug(f"{symbol}: Volume ${volume_24h:,.0f} below min ${self.min_volume_usd:,.0f}")
                return None

            logger.debug(f"{symbol}: PASSED filters - price=${price:.2f}, vol=${volume_24h:,.0f}")

            # Get indicators
            rsi = float(latest["rsi"]) if "rsi" in df.columns else 50
            volatility = float(latest["volatility"]) if "volatility" in df.columns else 0
            macd = float(latest["macd"]) if "macd" in df.columns else 0
            macd_signal = float(latest["macd_signal"]) if "macd_signal" in df.columns else 0

            # Calculate trend strength (-1 to 1)
            sma_20 = float(latest["sma_20"]) if "sma_20" in df.columns else price
            sma_50 = float(latest["sma_50"]) if "sma_50" in df.columns else price

            if sma_50 > 0:
                trend_strength = (sma_20 - sma_50) / sma_50
                trend_strength = max(-1, min(1, trend_strength * 10))  # Normalize
            else:
                trend_strength = 0

            # Calculate opportunity score
            score, reason = self._calculate_score(
                price=price,
                change_24h=change_24h,
                volume_24h=volume_24h,
                volatility=volatility,
                rsi=rsi,
                macd=macd,
                macd_signal=macd_signal,
                trend_strength=trend_strength,
            )

            return CryptoOpportunity(
                symbol=symbol,
                price=price,
                change_24h=change_24h,
                volume_24h=volume_24h,
                volatility=volatility * 100,  # Convert to percentage
                rsi=rsi,
                trend_strength=trend_strength,
                score=score,
                reason=reason,
            )

        except Exception as e:
            logger.debug(f"Error analyzing {symbol}: {e}")
            return None

    def _calculate_score(
        self,
        price: float,
        change_24h: float,
        volume_24h: float,
        volatility: float,
        rsi: float,
        macd: float,
        macd_signal: float,
        trend_strength: float,
    ) -> tuple[float, str]:
        """
        Calculate opportunity score (0-100) based on multiple factors.

        Higher score = better opportunity
        """
        score = 50  # Start neutral
        reasons = []

        # RSI extremes (oversold = buy opportunity, overbought = sell opportunity)
        if rsi < 30:
            score += 20
            reasons.append(f"RSI oversold ({rsi:.0f})")
        elif rsi < 40:
            score += 10
            reasons.append(f"RSI low ({rsi:.0f})")
        elif rsi > 70:
            score += 15  # Momentum opportunity
            reasons.append(f"RSI high momentum ({rsi:.0f})")

        # MACD crossover
        if macd > macd_signal and macd > 0:
            score += 15
            reasons.append("MACD bullish")
        elif macd < macd_signal and macd < 0:
            score += 10
            reasons.append("MACD bearish (short opp)")

        # Volatility (moderate volatility is good for trading)
        if 0.3 < volatility < 1.0:
            score += 10
            reasons.append("Good volatility")
        elif volatility > 1.0:
            score += 5
            reasons.append("High volatility")

        # Volume (higher = more liquid)
        if volume_24h > 1_000_000:
            score += 10
            reasons.append("High volume")
        elif volume_24h > 500_000:
            score += 5
            reasons.append("Good volume")

        # Trend strength
        if abs(trend_strength) > 0.3:
            score += 10
            reasons.append(f"Strong trend ({'+' if trend_strength > 0 else ''}{trend_strength:.2f})")

        # Recent price action
        if abs(change_24h) > 5:
            score += 5
            reasons.append(f"Active ({change_24h:+.1f}% 24h)")

        # Cap score at 100
        score = min(100, max(0, score))

        return score, "; ".join(reasons) if reasons else "Neutral"

    def get_top_opportunities(
        self,
        count: int = 5,
        min_score: float = 50,
    ) -> list[CryptoOpportunity]:
        """
        Get top trading opportunities above minimum score.
        """
        all_opps = self.scan_all()
        filtered = [o for o in all_opps if o.score >= min_score]
        return filtered[:count]

    def get_watchlist(self, count: int = 5) -> list[str]:
        """
        Get dynamic watchlist of best crypto symbols to trade.
        """
        opportunities = self.get_top_opportunities(count=count, min_score=40)
        symbols = [o.symbol for o in opportunities]

        # Always include BTC and ETH if not already
        for base in ["BTC/USD", "ETH/USD"]:
            if base not in symbols:
                symbols.append(base)

        logger.info(f"Dynamic watchlist: {symbols}")
        return symbols[:count]
