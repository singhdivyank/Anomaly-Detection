package main.java.com.grid.analytics.repository;

import main.java.com.grid.analytics.model.AnomalyAlert;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

public interface AnomalyAlertRepository extends JpaRepository<AnomalyAlert, Long> {

    Page<AnomalyAlert> findAllByOrderByOccurredAtDesc(Pageable pageable);

    Page<AnomalyAlert> findBySeverityOrderByOccurredAtDesc(String severity, Pageable pageable);

    long countBySeverity(String severity);

    long countByAlertType(String alertType);
}
