"""
Feature reconstruction + score generation for the smart-meter anomaly
inference pipeline.

Input:  raw stats forwarded by InferenceClientService.java (Spring Boot)
Output: {anomaly_detected, anomaly_score, confidence_score} matching the
        Spring Boot <-> Python Inference Contract in README.md.
"""

import threading
from typing import Deque, Optional
from collections import deque, defaultdict

import joblib
import numpy as np
from datetime import datetime, timezone

from ..consts import (
    CONFIDENCE_THRESHOLD, 
    MODELS_DIR, 
    ROLLING_WINDOW, 
    PredictionResult
)


class AnomalyPredictor:
    """Wraps the three pretrained artifacts (scaler, isolation forest,
    LightGBM classifier) behind a single `.predict()` call, and maintains a
    small per-household rolling buffer so `rolling_mean_3h` can be derived
    server-side when the caller only supplies the instantaneous reading."""

    def __init__(self, models_dir = MODELS_DIR):
        self.scaler = joblib.load(models_dir / "nn_scaler.pkl")
        self.isolation_forest = joblib.load(models_dir / "isolation_forest_model.pkl")
        self.lightgbm = joblib.load(models_dir / "lightgbm_model.pkl")
        self._lock = threading.Lock()
        self._history: dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))
    
    def _rolling_mean_3h(self, household_id: str, kw_consumed: float) -> float:
        with self._lock:
            buf = self._history[household_id]
            buf.append(kw_consumed)
            return float(np.mean(buf))

    def _resolve_hour_and_dow(self, timestamp: str | None) -> tuple[int, int]:
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError:
                dt = datetime.now(timezone.utc)
        else:
            dt = datetime.now(timezone.utc)
        return dt.hour, dt.weekday()
    
    def predict(
        self,
        household_id: str,
        kw_consumed: float,
        timestamp: Optional[str],
        rolling_mean_3h: Optional[float],
    ) -> PredictionResult:
        hour, day_of_week = self._resolve_hour_and_dow(timestamp)
        r_mean = (
            rolling_mean_3h
            if rolling_mean_3h is not None
            else self._rolling_mean_3h(household_id, kw_consumed)
        )

        feature_vector = np.array([[hour, day_of_week, r_mean, kw_consumed]])
        scaled = self.scaler.transform(feature_vector)

        # Isolation Forest: negative decision_function values are more anomalous
        raw_iso_score = float(self.isolation_forest.decision_function(scaled)[0])
        iso_flags_outlier = self.isolation_forest.predict(scaled)[0] == -1

        # LightGBM: probability of the positive ("anomaly") class.
        confidence = float(self.lightgbm.predict_proba(scaled)[0][1])
        anomaly_detected = iso_flags_outlier or (confidence >= CONFIDENCE_THRESHOLD)

        return PredictionResult(
            anomaly_detected=anomaly_detected,
            anomaly_score=round(raw_iso_score, 6),
            confidence_score=round(confidence, 6),
        )


predictor = AnomalyPredictor()