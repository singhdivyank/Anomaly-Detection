package com.grid.analytics.websocket;

import com.grid.analytics.dto.AlertBroadcast;
import com.grid.analytics.dto.TelemetryTick;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.stereotype.Component;

/**
 * Pushes payloads directly to the dashboard client over the two channels
 * defined in dashboard-ui-spec.md's useGridWebSocket hook:
 * - /topic/live-alerts (high priority flagged alerts)
 * - /topic/telemetry-ticks (global grid monitoring stream ticks)
 * No database polling is involved -- the broadcast happens inline as part
 * of Kafka message processing.
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class WebSocketAlertBroker {

    private static final String ALERTS_TOPIC = "/topic/live-alerts";
    private static final String TELEMETRY_TOPIC = "/topic/telemetry-ticks";

    private final SimpMessagingTemplate messagingTemplate;

    public void broadcastAlert(AlertBroadcast alert) {
        try {
            messagingTemplate.convertAndSend(ALERTS_TOPIC, alert);
        } catch (Exception ex) {
            log.warn("Failed to broadcast alert for household={}: {}", alert.householdId(), ex.getMessage());
        }
    }

    public void broadcastTelemetryTick(TelemetryTick tick) {
        try {
            messagingTemplate.convertAndSend(TELEMETRY_TOPIC, tick);
        } catch (Exception ex) {
            log.warn("Failed to broadcast telemetry tick for household={}: {}", tick.householdId(), ex.getMessage());
        }
    }
}
