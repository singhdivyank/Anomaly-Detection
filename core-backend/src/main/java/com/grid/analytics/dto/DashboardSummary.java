package com.grid.analytics.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Backing payload for the four top-level metric cards in the Operations
 * Overview: Active Meters, Grid Overloads, Demand Non-Adherence, System
 * Efficiency.
 */
public record DashboardSummary(
                @JsonProperty("active_meters") long activeMeters,
                @JsonProperty("grid_overloads") long gridOverloads,
                @JsonProperty("demand_non_adherence") long demandNonAdherence,
                @JsonProperty("system_efficiency_pct") double systemEfficiencyPct) {
}
