"""
FastAPI microservice exposing the HTTP validation/inference endpoint called
synchronously by core-backend's InferenceClientService (Spring WebClient).
"""

from fastapi import FastAPI, HTTPException

from consts import PredictRequest, PredictResponse
from services.predictor import predictor

app = FastAPI(
    title="Grid Analytics ML Inference Sidecar",
    description="High-throughput encapsulation of pretrained Isolation Forest, "
    "LightGBM, and Neural Network scaler models for smart-meter anomaly detection.",
    version="1.0.0",
)

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "models_loaded": True}

@app.post("/api/v1/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        result = predictor.predict(
            household_id=payload.household_id,
            kw_consumed=payload.kw_consumed,
            timestamp=payload.timestamp,
            rolling_mean_3h=payload.rolling_mean_3h,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"inference failed: {exc}")
    
    
    return PredictResponse(
        anomaly_detected=result.anomaly_detected,
        anomaly_score=result.anomaly_score,
        confidence_score=result.confidence_score,
    )
