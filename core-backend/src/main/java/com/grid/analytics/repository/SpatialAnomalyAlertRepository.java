package com.grid.analytics.repository;

import com.grid.analytics.model.SpatialAnomalyAlert;
import com.grid.analytics.model.SpatialAnomalyAlertId;
import org.springframework.data.jpa.repository.JpaRepository;

public interface SpatialAnomalyAlertRepository
        extends JpaRepository<SpatialAnomalyAlert, SpatialAnomalyAlertId> {
}
