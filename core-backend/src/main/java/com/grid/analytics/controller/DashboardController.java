package com.grid.analytics.controller;

import com.grid.analytics.dto.DashboardSummary;
import com.grid.analytics.repository.AnomalyAlertRepository;
import com.grid.analytics.repository.MeterReadingRepository;
import lombok.RequiredArgsConstructor;

import java.time.Instant;
import java.time.Duration;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * Backs the four Operations Overview metric cards: Active Meters, Grid
 * Overloads, Demand Non-Adherence, System Efficiency.
 */
@RestController
@RequestMapping("/api/dashboard")
@RequiredArgsConstructor
public class DashboardController {

    private final MeterReadingRepository meterReadingRepository;
    private final AnomalyAlertRepository anomalyAlertRepository;

    private static final Duration ACTIVE_WINDOW = Duration.ofHours(1);

    @GetMapping("/summary")
    public DashboardSummary summary() {
        Instant since = Instant.now().minus(ACTIVE_WINDOW);

        long activeMeters = meterReadingRepository.countDistinctActiveHouseholdsSince(since);
        long gridOverloads = anomalyAlertRepository.countByAlertType("Grid Overload");
        long demandNonAdherence = anomalyAlertRepository.countByAlertType("Non-Adherence (dToU_High)");
        long totalReadings = meterReadingRepository.findRecentSince(since).size();
        long anomalousReadings = meterReadingRepository.countAnomalousReadingsSince(since);
        double systemEfficiencyPct = totalReadings == 0
                ? 100.0
                : 100.0 * (1.0 - ((double) anomalousReadings / totalReadings));

        return new DashboardSummary(
                activeMeters,
                gridOverloads,
                demandNonAdherence,
                Math.round(systemEfficiencyPct * 10.0) / 10.0);
    }
}
