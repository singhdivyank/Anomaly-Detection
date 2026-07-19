from dataclasses import dataclass
from pathlib import Path
from pydantic import BaseModel, Field

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
CONFIDENCE_THRESHOLD = 0.5
ROLLING_WINDOW = 6


class PredictRequest(BaseModel):
    household_id: str = Field(..., examples=["MAC000123"])
    timestamp: str | None = Field(None, examples=["2026-06-18T02:35:10Z"])
    kw_consumed: float = Field(None, examples=[0.432])
    pricing_tier: str | None = Field(None, examples=["dToU_High"])
    is_weekend: bool | None = Field(None, examples=[False])
    rolling_mean_3h: float | None = Field(
        None, description="Optional precomputed rolling mean; server derives it if omitted."
    )


class PredictResponse(BaseModel):
    anomaly_detected: bool
    anomaly_score: float
    confidence_score: float


@dataclass
class PredictionResult:
    anomaly_detected: bool
    anomaly_score: float
    confidence_score: float
