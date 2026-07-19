package com.grid.analytics.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.time.Instant;

/**
 * Payload pushed to STOMP destination /topic/telemetry-ticks, feeding
 * Component A (High-Throughput Stream Chart): the live meter reading
 * line plus the fields needed for the custom tooltip (household_id,
 * kW, anomaly_score) and the anomaly intercept node filter
 * (anomaly_detected == true). Also carries the household's resolved
 * geospatial location, hydrated from the grid_assets Asset Registry
 * via GeospatialLookupService, for the Geospatial Grid Health map.
 */
public record TelemetryTick(
        @JsonProperty("household_id") String householdId,
        @JsonProperty("kw_consumed") double kwConsumed,
        @JsonProperty("expected_baseline") double expectedBaseline,
        @JsonProperty("anomaly_score") double anomalyScore,
        @JsonProperty("anomaly_detected") boolean anomalyDetected,
        @JsonProperty("timestamp") Instant timestamp,
        @JsonProperty("geospatial") GeoSpatialInfo geospatial) {
}
