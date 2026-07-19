package com.grid.analytics.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Nested payload appended to telemetry ticks and alert broadcasts once the
 * household_id has been resolved against the grid_assets registry table.
 * Matches the "Hydrated WebSocket Outbound" shape in the geospatial spec:
 * { "borough": ..., "latitude": ..., "longitude": ... }
 */
public record GeoSpatialInfo(
        @JsonProperty("borough") String borough,
        @JsonProperty("latitude") double latitude,
        @JsonProperty("longitude") double longitude) {
}
