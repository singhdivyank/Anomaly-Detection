package main.java.com.grid.analytics.dto;

import java.time.Instant;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Payload pushed to STOMP destination /topic/telemetry-ticks, feeding
 * Component A (High-Throughput Stream Chart) in dashboard-ui-spec.md: the
 * live meter reading line plus the fields needed for the custom tooltip
 * (household_id, kW, anomaly_score) and the anomaly intercept node filter
 * (anomaly_detected == true).
 */
public record TelemetryTick(
        @JsonProperty("household_id") String householdId,
        @JsonProperty("kw_consumed") double kwConsumed,
        @JsonProperty("expected_baseline") double expectedBaseline,
        @JsonProperty("anomaly_score") double anomalyScore,
        @JsonProperty("anomaly_detected") boolean anomalyDetected,
        @JsonProperty("timestamp") Instant timestamp) {
}
