package com.grid.analytics.controller;

import lombok.RequiredArgsConstructor;

import com.grid.analytics.model.AnomalyAlert;
import com.grid.analytics.repository.AnomalyAlertRepository;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.web.bind.annotation.*;

/**
 * Backs the "Live Alerts Feed" data grid in dashboard-ui-spec.md. Initial
 * page load / historical browsing goes through this REST endpoint; new
 * alerts after that arrive live over /topic/live-alerts.
 */
@RestController
@RequestMapping("/api/alerts")
@RequiredArgsConstructor
public class AlertController {

    private final AnomalyAlertRepository anomalyAlertRepository;

    @GetMapping
    public Page<AnomalyAlert> listAlerts(
            @RequestParam(required = false) String severity,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size) {

        Pageable pageable = PageRequest.of(page, size, Sort.by(Sort.Direction.DESC, "occurredAt"));

        if (severity != null && !severity.isBlank()) {
            return anomalyAlertRepository.findBySeverityOrderByOccurredAtDesc(severity, pageable);
        }
        return anomalyAlertRepository.findAllByOrderByOccurredAtDesc(pageable);
    }

    @GetMapping("/{id}")
    public AnomalyAlert getAlert(@PathVariable Long id) {
        return anomalyAlertRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Alert not found: " + id));
    }

}
