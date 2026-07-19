package com.grid.analytics.model;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.IdClass;
import jakarta.persistence.Table;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.time.Instant;
import java.util.UUID;

/**
 * Maps to the `spatial_anomaly_alerts` Timescale hypertable -- raw
 * historical anomaly points kept separately from the richer relational
 * `anomaly_alert` table, purely for geospatial time-series queries (e.g.
 * "anomaly density by borough over the last 24h"). Written alongside
 * `anomaly_alert` whenever SmartMeterStreamConsumer detects an anomaly.
 */
@Entity
@Table(name = "spatial_anomaly_alerts")
@IdClass(SpatialAnomalyAlertId.class)
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class SpatialAnomalyAlert {

    @Id
    @Column(name = "alert_id", nullable = false)
    private UUID alertId;

    @Id
    @Column(name = "timestamp", nullable = false)
    private Instant timestamp;

    @Column(name = "household_id", nullable = false, length = 50)
    private String householdId;

    @Column(name = "anomaly_score", nullable = false)
    private double anomalyScore;
}
