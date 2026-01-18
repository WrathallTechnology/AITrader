"""
Machine learning based trading strategy.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from .base import BaseStrategy, Signal, SignalType
from src.models.predictor import PricePredictor
from src.data.processor import DataProcessor
from config import StrategyConfig

logger = logging.getLogger(__name__)


class MLStrategy(BaseStrategy):
    """
    Trading strategy based on ML price predictions.

    Uses a trained model to predict whether price will go up or down,
    and generates signals based on the prediction and confidence level.
    """

    def __init__(
        self,
        name: str = "ml",
        weight: float = 1.0,
        config: Optional[StrategyConfig] = None,
        model_path: Optional[Path] = None,
        min_confidence: float = 0.55,
    ):
        super().__init__(name, weight)
        self.config = config or StrategyConfig()
        self.min_confidence = min_confidence

        self.predictor = PricePredictor(
            lookback=self.config.lookback_period,
            prediction_horizon=self.config.prediction_horizon,
            model_path=model_path,
        )

    def train(self, data: pd.DataFrame) -> None:
        """Train the ML model."""
        logger.info("Training ML strategy...")

        # Add indicators
        data = DataProcessor.add_technical_indicators(data)

        # Train predictor
        metrics = self.predictor.train(data)
        logger.info(f"Training complete. Test accuracy: {metrics['test_accuracy']:.4f}")

        # Save model
        self.predictor.save()

        self._is_trained = True

    def load_model(self, path: Optional[Path] = None) -> None:
        """Load a pre-trained model."""
        self.predictor.load(path)
        self._is_trained = True

    def generate_signal(self, symbol: str, data: pd.DataFrame) -> Signal:
        """Generate trading signal based on ML prediction."""
        if not self._is_trained:
            logger.warning("ML model not trained!")
            return Signal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                confidence=0.0,
                reason="Model not trained",
            )

        # Ensure we have indicators
        if "rsi" not in data.columns:
            data = DataProcessor.add_technical_indicators(data)

        try:
            prediction, confidence = self.predictor.predict(data)

            # Get current price
            price = float(data["close"].iloc[-1])
            atr = float(data["atr"].iloc[-1]) if "atr" in data.columns else price * 0.02

            # Determine signal based on prediction and confidence
            if confidence < self.min_confidence:
                return Signal(
                    symbol=symbol,
                    signal_type=SignalType.HOLD,
                    confidence=confidence,
                    price=price,
                    reason=f"ML confidence {confidence:.2%} below threshold",
                )

            if prediction == 1:  # Predict up
                signal_type = SignalType.BUY
                stop_loss = price - (2 * atr)
                take_profit = price + (3 * atr)
            else:  # Predict down
                signal_type = SignalType.SELL
                stop_loss = price + (2 * atr)
                take_profit = price - (3 * atr)

            return Signal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=f"ML prediction: {'UP' if prediction == 1 else 'DOWN'} ({confidence:.2%})",
            )

        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return Signal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                confidence=0.0,
                reason=f"Prediction error: {e}",
            )

    @property
    def is_trained(self) -> bool:
        return self.predictor.is_trained
