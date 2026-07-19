package com.grid.analytics.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.time.Instant;

/**
 * Payload pushed to STOMP destination /topic/live-alerts, shaped to feed
 * dashboard-ui-spec.md's Live Event Feed columns directly:
 * Severity, Alert Type, Meter ID, Timestamp. Also carries the household's
 * resolved geospatial location so the Geospatial Grid Health map can plot
 * the anomaly at its real position without a second round-trip.
 */
public record AlertBroadcast(
                @JsonProperty("household_id") String householdId,
                @JsonProperty("severity") String severity,
                @JsonProperty("alert_type") String alertType,
                @JsonProperty("kw_consumed") double kwConsumed,
                @JsonProperty("anomaly_score") double anomalyScore,
                @JsonProperty("confidence_score") double confidenceScore,
                @JsonProperty("timestamp") Instant timestamp,
                @JsonProperty("geospatial") GeoSpatialInfo geospatial) {
}
