package com.grid.analytics.consumer;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

import java.time.Instant;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.grid.analytics.dto.AlertBroadcast;
import com.grid.analytics.dto.GeoSpatialInfo;
import com.grid.analytics.dto.InferenceResponse;
import com.grid.analytics.dto.TelemetryMessage;
import com.grid.analytics.dto.TelemetryTick;
import com.grid.analytics.model.AnomalyAlert;
import com.grid.analytics.model.MeterReading;
import com.grid.analytics.model.SpatialAnomalyAlert;
import com.grid.analytics.repository.AnomalyAlertRepository;
import com.grid.analytics.repository.MeterReadingRepository;
import com.grid.analytics.repository.SpatialAnomalyAlertRepository;
import com.grid.analytics.service.GeoSpatialLookupService;
import com.grid.analytics.service.InferenceClientService;
import com.grid.analytics.websocket.WebSocketAlertBroker;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.kafka.support.Acknowledgment;
import org.springframework.stereotype.Component;

/**
 * Listens to the `smart-meter-telemetry` Kafka topic (partitioned by
 * household_id), synchronously persists every reading, asynchronously
 * invokes the ML inference sidecar, and pushes results straight to the
 * dashboard over WebSocket without ever polling the database.
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class SmartMeterStreamConsumer {

    private final ObjectMapper objectMapper;
    private final InferenceClientService inferenceClientService;
    private final MeterReadingRepository meterReadingRepository;
    private final AnomalyAlertRepository anomalyAlertRepository;
    private final SpatialAnomalyAlertRepository spatialAnomalyAlertRepository;
    private final GeoSpatialLookupService geospatialLookupService;
    private final WebSocketAlertBroker webSocketAlertBroker;

    private final Map<String, Double> expectedBaselineByHousehold = new ConcurrentHashMap<>();
    private static final double EMA_ALPHA = 0.2;

    @KafkaListener(topics = "${grid.kafka.topic:smart-meter-telemetry}", containerFactory = "kafkaListenerContainerFactory")
    public void onMessage(String rawValue, Acknowledgment acknowledgment) {
        try {
            TelemetryMessage message = objectMapper.readValue(rawValue, TelemetryMessage.class);
            process(message);
        } catch (Exception ex) {
            log.error("Failed to process telemetry record, delegating to error handler: {}", ex.getMessage());
            throw new IllegalStateException("telemetry processing failed", ex);
        } finally {
            acknowledgment.acknowledge();
        }
    }

    private void process(TelemetryMessage message) {
        double baseline = updateExpectedBaseline(message.householdId(), message.kwConsumed());

        inferenceClientService.predict(message)
                .doOnNext(inference -> handleInferenceResult(message, inference, baseline))
                .doOnError(ex -> log.error("Unhandled inference error for household={}: {}",
                        message.householdId(), ex.getMessage()))
                .subscribe();
    }

    private void handleInferenceResult(TelemetryMessage message, InferenceResponse inference, double baseline) {
        Instant now = Instant.now();
        GeoSpatialInfo geo = geospatialLookupService.resolve(message.householdId());

        MeterReading reading = MeterReading.builder()
                .householdId(message.householdId())
                .readingTime(message.timestamp())
                .kwConsumed(message.kwConsumed())
                .pricingTier(message.pricingTier())
                .isWeekend(message.isWeekend())
                .anomalyScore(inference.anomalyScore())
                .confidenceScore(inference.confidenceScore())
                .anomalyDetected(inference.anomalyDetected())
                .ingestedAt(now)
                .build();
        meterReadingRepository.save(reading);

        webSocketAlertBroker.broadcastTelemetryTick(new TelemetryTick(
                message.householdId(),
                message.kwConsumed(),
                baseline,
                inference.anomalyScore(),
                inference.anomalyDetected(),
                message.timestamp(),
                geo));

        if (inference.anomalyDetected()) {
            persistAndBroadcastAlert(message, inference, geo);
        }
    }

    private void persistAndBroadcastAlert(TelemetryMessage message, InferenceResponse inference, GeoSpatialInfo geo) {
        String alertType = classifyAlertType(message, inference);
        String severity = classifySeverity(inference);

        AnomalyAlert alert = AnomalyAlert.builder()
                .householdId(message.householdId())
                .severity(severity)
                .alertType(alertType)
                .kwConsumed(message.kwConsumed())
                .anomalyScore(inference.anomalyScore())
                .confidenceScore(inference.confidenceScore())
                .occurredAt(message.timestamp())
                .createdAt(Instant.now())
                .acknowledged(false)
                .build();
        anomalyAlertRepository.save(alert);

        spatialAnomalyAlertRepository.save(SpatialAnomalyAlert.builder()
                .alertId(UUID.randomUUID())
                .householdId(message.householdId())
                .timestamp(message.timestamp())
                .anomalyScore(inference.anomalyScore())
                .build());

        webSocketAlertBroker.broadcastAlert(new AlertBroadcast(
                message.householdId(),
                severity,
                alertType,
                message.kwConsumed(),
                inference.anomalyScore(),
                inference.confidenceScore(),
                message.timestamp(),
                geo));
    }

    private String classifyAlertType(TelemetryMessage message, InferenceResponse inference) {
        if ("dToU_High".equalsIgnoreCase(message.pricingTier())) {
            return "Non-Adherence (dToU_High)";
        }
        if (message.kwConsumed() > 5.0) {
            return "Grid Overload";
        }
        return "Technical Anomaly";
    }

    private String classifySeverity(InferenceResponse inference) {
        final double confidence = inference.confidenceScore();
        if (confidence >= 0.85) {
            return "Severe";
        } else if (confidence >= 0.5) {
            return "Moderate";
        }
        return "Minor";
    }

    private double updateExpectedBaseline(String householdId, double latestReading) {
        return expectedBaselineByHousehold.merge(householdId, latestReading,
                (previous, current) -> previous + EMA_ALPHA * (current - previous));
    }
}
