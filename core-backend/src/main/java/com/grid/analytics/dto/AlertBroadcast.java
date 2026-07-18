package main.java.com.grid.analytics.dto;

import java.time.Instant;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Payload pushed to STOMP destination /topic/live-alerts, shaped to feed
 * dashboard-ui-spec.md's Live Event Feed columns directly:
 * Severity, Alert Type, Meter ID, Timestamp.
 */
public record AlertBroadcast(
                @JsonProperty("household_id") String householdId,
                @JsonProperty("severity") String severity,
                @JsonProperty("alert_type") String alertType,
                @JsonProperty("kw_consumed") double kwConsumed,
                @JsonProperty("anomaly_score") double anomalyScore,
                @JsonProperty("confidence_score") double confidenceScore,
                @JsonProperty("timestamp") Instant timestamp) {
}
