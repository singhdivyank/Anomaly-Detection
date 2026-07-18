package main.java.com.grid.analytics.model;

import java.time.Instant;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

/**
 * Maps to the relational `anomaly_alert` table,
 * kept separate from the raw tick hypertable.
 */
@Entity
@Table(name = "anomaly_alert")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class AnomalyAlert {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "household_id", nullable = false, length = 64)
    private String householdId;

    /**
     * One of: Severe, Moderate, Minor
     */
    @Column(name = "severity", nullable = false, length = 16)
    private String severity;

    /** e.g. "Technical Anomaly", "Non-Adherence (dToU_High)", "Grid Overload". */
    @Column(name = "alert_type", nullable = false, length = 64)
    private String alertType;

    @Column(name = "kw_consumed", nullable = false)
    private double kwConsumed;

    @Column(name = "anomaly_score", nullable = false)
    private double anomalyScore;

    @Column(name = "confidence_score", nullable = false)
    private double confidenceScore;

    @Column(name = "occurred_at", nullable = false)
    private Instant occurredAt;

    @Column(name = "created_at")
    private Instant createdAt;

    @Column(name = "acknowledged", nullable = false)
    private boolean acknowledged;
}
