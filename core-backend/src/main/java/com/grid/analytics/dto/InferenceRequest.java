package com.grid.analytics.dto;

import java.time.Instant;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Request payload for POST /api/v1/predict on the FastAPI sidecar, matching
 * the Spring Boot -> Python Inference Contract in README.md.
 */
public record InferenceRequest(
        @JsonProperty("household_id") String householdId,
        @JsonProperty("timestamp") Instant timestamp,
        @JsonProperty("kw_consumed") double kwConsumed,
        @JsonProperty("pricing_tier") String pricingTier,
        @JsonProperty("is_weekend") boolean isWeekend) {
    public static InferenceRequest from(TelemetryMessage msg) {
        return new InferenceRequest(
                msg.householdId(), msg.timestamp(), msg.kwConsumed(), msg.pricingTier(), msg.isWeekend());
    }
}