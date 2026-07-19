package com.grid.analytics.model;

import lombok.AllArgsConstructor;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;

import java.io.Serializable;
import java.time.Instant;
import java.util.UUID;

/**
 * Composite key (alert_id, timestamp) for SpatialAnomalyAlert, matching the
 * hypertable requirement that the partitioning column be part of the
 * primary key.
 */
@NoArgsConstructor
@AllArgsConstructor
@EqualsAndHashCode
public class SpatialAnomalyAlertId implements Serializable {
    private UUID alertId;
    private Instant timestamp;
}
