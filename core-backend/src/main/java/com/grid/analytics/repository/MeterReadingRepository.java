package com.grid.analytics.repository;

import java.time.Instant;
import java.util.List;

import com.grid.analytics.model.MeterReading;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

public interface MeterReadingRepository extends JpaRepository<MeterReading, Long> {

    @Query("SELECT COUNT(DISTINCT m.householdId) FROM MeterReading m WHERE m.readingTime >= :since")
    long countDistinctActiveHouseholdsSince(@Param("since") Instant since);

    @Query("SELECT COUNT(m) FROM MeterReading m WHERE m.anomalyDetected = true AND m.readingTime >= :since")
    long countAnomalousReadingsSince(@Param("since") Instant since);

    @Query("SELECT m FROM MeterReading m WHERE m.readingTime >= :since ORDER BY m.readingTime DESC")
    List<MeterReading> findRecentSince(@Param("since") Instant since);
}
