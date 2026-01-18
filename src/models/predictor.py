"""
Machine learning price prediction model.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from src.data.processor import DataProcessor

logger = logging.getLogger(__name__)


class PricePredictor:
    """
    ML model for predicting price direction.

    Uses an ensemble of classifiers to predict whether price
    will go up or down over a given horizon.
    """

    def __init__(
        self,
        lookback: int = 60,
        prediction_horizon: int = 5,
        model_path: Optional[Path] = None,
    ):
        self.lookback = lookback
        self.prediction_horizon = prediction_horizon
        self.model_path = model_path or Path("models/price_predictor.pkl")

        self._model: Optional[GradientBoostingClassifier] = None
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_std: Optional[np.ndarray] = None
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def train(
        self,
        data: pd.DataFrame,
        test_size: float = 0.2,
        verbose: bool = True,
    ) -> dict:
        """
        Train the prediction model.

        Args:
            data: DataFrame with technical indicators
            test_size: Fraction of data to use for testing
            verbose: Print training metrics

        Returns:
            Dictionary with training metrics
        """
        logger.info("Preparing training data...")

        # Add indicators if not present
        if "rsi" not in data.columns:
            data = DataProcessor.add_technical_indicators(data)

        # Prepare features and labels
        X, y = DataProcessor.prepare_ml_features(
            data,
            lookback=self.lookback,
            prediction_horizon=self.prediction_horizon,
        )

        # Normalize features
        X_norm, self._feature_mean, self._feature_std = DataProcessor.normalize_features(X)

        # Flatten for sklearn
        X_flat = X_norm.reshape(X_norm.shape[0], -1)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_flat, y, test_size=test_size, shuffle=False
        )

        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")

        # Train model
        self._model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        )

        logger.info("Training model...")
        self._model.fit(X_train, y_train)

        # Evaluate
        train_pred = self._model.predict(X_train)
        test_pred = self._model.predict(X_test)

        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)

        metrics = {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

        if verbose:
            logger.info(f"Train accuracy: {train_accuracy:.4f}")
            logger.info(f"Test accuracy: {test_accuracy:.4f}")
            logger.info("\nClassification Report (Test Set):")
            logger.info(classification_report(y_test, test_pred))

        self._is_trained = True
        return metrics

    def predict(self, data: pd.DataFrame) -> tuple[int, float]:
        """
        Predict price direction.

        Args:
            data: DataFrame with recent price data

        Returns:
            Tuple of (prediction, confidence)
            prediction: 1 for up, 0 for down
            confidence: probability of the prediction
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        # Add indicators if needed
        if "rsi" not in data.columns:
            data = DataProcessor.add_technical_indicators(data)

        # Prepare features
        X, _ = DataProcessor.prepare_ml_features(
            data,
            lookback=self.lookback,
            prediction_horizon=self.prediction_horizon,
        )

        if len(X) == 0:
            raise ValueError("Insufficient data for prediction")

        # Use most recent window
        X_latest = X[-1:].reshape(1, -1)

        # Normalize
        X_norm = (X_latest - self._feature_mean.flatten()) / self._feature_std.flatten()

        # Flatten
        X_flat = X_norm.reshape(1, -1)

        # Predict
        prediction = self._model.predict(X_flat)[0]
        probabilities = self._model.predict_proba(X_flat)[0]
        confidence = float(max(probabilities))

        return int(prediction), confidence

    def save(self, path: Optional[Path] = None) -> None:
        """Save model to disk."""
        save_path = path or self.model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self._model,
            "feature_mean": self._feature_mean,
            "feature_std": self._feature_std,
            "lookback": self.lookback,
            "prediction_horizon": self.prediction_horizon,
        }

        with open(save_path, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {save_path}")

    def load(self, path: Optional[Path] = None) -> None:
        """Load model from disk."""
        load_path = path or self.model_path

        if not load_path.exists():
            raise FileNotFoundError(f"Model not found: {load_path}")

        with open(load_path, "rb") as f:
            model_data = pickle.load(f)

        self._model = model_data["model"]
        self._feature_mean = model_data["feature_mean"]
        self._feature_std = model_data["feature_std"]
        self.lookback = model_data["lookback"]
        self.prediction_horizon = model_data["prediction_horizon"]
        self._is_trained = True

        logger.info(f"Model loaded from {load_path}")
