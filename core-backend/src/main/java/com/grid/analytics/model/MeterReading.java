package com.grid.analytics.model;

import java.time.Instant;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

/**
 * Maps to the `meter_reading` Timescale hypertable. Strictly raw IoT tick
 * data, this must not be mixed with relational entity/tenant tables.
 */
@Entity
@Table(name = "meter_reading")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class MeterReading {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "household_id", nullable = false, length = 64)
    private String householdId;

    @Column(name = "reading_time", nullable = false)
    private Instant readingTime;

    @Column(name = "kw_consumed", nullable = false)
    private double kwConsumed;

    @Column(name = "pricing_tier", length = 32)
    private String pricingTier;

    @Column(name = "is_weekend", nullable = false)
    private boolean isWeekend;

    @Column(name = "anomaly_score")
    private Double anomalyScore;

    @Column(name = "confidence_score")
    private Double confidenceScore;

    @Column(name = "anomaly_detected", nullable = false)
    private boolean anomalyDetected;

    @Column(name = "ingested_at")
    private Instant ingestedAt;
}
