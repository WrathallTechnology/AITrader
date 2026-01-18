"""
Data processing and feature engineering module.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Processes raw market data and generates features for trading strategies.
    """

    @staticmethod
    def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add common technical indicators to the dataframe.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added indicator columns
        """
        df = df.copy()

        # Simple Moving Averages
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["sma_50"] = df["close"].rolling(window=50).mean()
        df["sma_200"] = df["close"].rolling(window=200).mean()

        # Exponential Moving Averages
        df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
        df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()

        # MACD
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        # RSI
        df["rsi"] = DataProcessor.calculate_rsi(df["close"], period=14)

        # Bollinger Bands
        df["bb_middle"] = df["close"].rolling(window=20).mean()
        bb_std = df["close"].rolling(window=20).std()
        df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
        df["bb_lower"] = df["bb_middle"] - (bb_std * 2)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

        # ATR (Average True Range)
        df["atr"] = DataProcessor.calculate_atr(df, period=14)

        # Volume indicators
        df["volume_sma"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma"]

        # Price changes
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        # Volatility
        df["volatility"] = df["returns"].rolling(window=20).std() * np.sqrt(252)

        return df

    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()

        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = df["high"]
        low = df["low"]
        close = df["close"].shift(1)

        tr1 = high - low
        tr2 = (high - close).abs()
        tr3 = (low - close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()

        return atr

    @staticmethod
    def prepare_ml_features(
        df: pd.DataFrame,
        lookback: int = 60,
        prediction_horizon: int = 5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and labels for ML model training.

        Args:
            df: DataFrame with technical indicators
            lookback: Number of historical periods to use as features
            prediction_horizon: Number of periods ahead to predict

        Returns:
            Tuple of (features, labels) as numpy arrays
        """
        # Ensure we have indicators
        if "rsi" not in df.columns:
            df = DataProcessor.add_technical_indicators(df)

        # Select feature columns
        feature_cols = [
            "returns", "log_returns", "volatility",
            "rsi", "macd", "macd_histogram",
            "bb_width", "volume_ratio", "atr"
        ]

        # Filter to available columns
        feature_cols = [c for c in feature_cols if c in df.columns]

        # Drop NaN values
        df_clean = df[feature_cols].dropna()

        if len(df_clean) < lookback + prediction_horizon:
            raise ValueError(f"Not enough data. Need at least {lookback + prediction_horizon} rows.")

        # Create sequences
        X, y = [], []

        for i in range(lookback, len(df_clean) - prediction_horizon):
            # Features: lookback periods of all indicators
            X.append(df_clean.iloc[i - lookback:i].values)

            # Label: price direction over prediction horizon
            future_return = df_clean["returns"].iloc[i:i + prediction_horizon].sum()
            y.append(1 if future_return > 0 else 0)  # Binary classification

        return np.array(X), np.array(y)

    @staticmethod
    def normalize_features(
        X: np.ndarray,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalize features using z-score normalization.

        Returns:
            Tuple of (normalized_X, mean, std)
        """
        # Reshape for normalization
        original_shape = X.shape
        X_flat = X.reshape(-1, X.shape[-1])

        if mean is None:
            mean = np.mean(X_flat, axis=0)
        if std is None:
            std = np.std(X_flat, axis=0)
            std[std == 0] = 1  # Avoid division by zero

        X_normalized = (X_flat - mean) / std
        X_normalized = X_normalized.reshape(original_shape)

        return X_normalized, mean, std

    @staticmethod
    def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate basic trading signals from technical indicators.
        """
        df = df.copy()

        # RSI signals
        df["rsi_signal"] = 0
        df.loc[df["rsi"] < 30, "rsi_signal"] = 1  # Oversold - buy
        df.loc[df["rsi"] > 70, "rsi_signal"] = -1  # Overbought - sell

        # MACD signals
        df["macd_signal_line"] = 0
        df.loc[df["macd"] > df["macd_signal"], "macd_signal_line"] = 1
        df.loc[df["macd"] < df["macd_signal"], "macd_signal_line"] = -1

        # Moving average crossover
        df["ma_signal"] = 0
        df.loc[df["sma_20"] > df["sma_50"], "ma_signal"] = 1
        df.loc[df["sma_20"] < df["sma_50"], "ma_signal"] = -1

        # Bollinger Band signals
        df["bb_signal"] = 0
        df.loc[df["close"] < df["bb_lower"], "bb_signal"] = 1  # Price below lower band
        df.loc[df["close"] > df["bb_upper"], "bb_signal"] = -1  # Price above upper band

        return df
