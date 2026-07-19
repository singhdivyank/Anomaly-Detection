package com.grid.analytics.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Response payload from POST /api/v1/predict, matching the Inference
 * Response Payload contract in README.md exactly:
 * {anomaly_detected, anomaly_score, confidence_score}
 */
public record InferenceResponse(
                @JsonProperty("anomaly_detected") boolean anomalyDetected,
                @JsonProperty("anomaly_score") double anomalyScore,
                @JsonProperty("confidence_score") double confidenceScore) {
}
