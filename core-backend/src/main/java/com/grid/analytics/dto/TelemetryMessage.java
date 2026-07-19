package com.grid.analytics.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.time.Instant;

/**
 * Mirrors the Kafka Ingestion Contract in README.md exactly:
 * {household_id, timestamp, kw_consumed, pricing_tier, is_weekend}
 */
public record TelemetryMessage(
                @JsonProperty("household_id") String householdId,
                @JsonProperty("timestamp") Instant timestamp,
                @JsonProperty("kw_consumed") double kwConsumed,
                @JsonProperty("pricing_tier") String pricingTier,
                @JsonProperty("is_weekend") boolean isWeekend) {
}
