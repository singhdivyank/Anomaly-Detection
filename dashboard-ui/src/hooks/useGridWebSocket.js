import { useEffect, useRef, useState } from "react";
import { Client } from "@stomp/stompjs";
import { generateTelemetryTick, tickToAlert } from "../lib/mockData";

const BROKER_URL = import.meta.env.VITE_WS_BROKER_URL || "ws://localhost:8080/grid-ws-broker";
const CONNECT_TIMEOUT_MS = 4000;
const DEMO_TICK_INTERVAL_MS = 900;

/**
 * Reactive state hook handling live data stream updates.
 *
 * Matches the STOMP Hook Pattern from dashboard-ui-spec.md exactly:
 *   - Channel 1 (/topic/live-alerts): high priority flagged alerts
 *   - Channel 2 (/topic/telemetry-ticks): global grid monitoring stream ticks
 *
 * If the Spring Boot broker can't be reached within CONNECT_TIMEOUT_MS, this
 * falls back to a local simulator so the console still reads as "live" for
 * demo/dev purposes -- `connectionStatus` tells the UI which mode it's in.
 */

export const useGridWebSocket = (onAlertTriggered, onTelemetryTick) => {
    const [connectionStatus, setConnectionStatus] = useState("connecting");
    const onAlertRef = useRef(onAlertTriggered);
    const onTickRef = useRef(onTelemetryTick);
    onAlertRef.current = onAlertTriggered;
    onTickRef.current = onTelemetryTick;

    useEffect(() => {
        let demoInterval = null;
        let fallbackTimer = null;
        let connected = false;

        const startDemoMode = () => {
            if (demoInterval) return;
            setConnectionStatus("demo");
            demoInterval = setInterval(() => {
                const tickPayload = generateTelemetryTick();
                onTickRef.current?.(tickPayload);
                if (tickPayload.anomaly_detected) {
                onAlertRef.current?.(tickToAlert(tickPayload));
                }
            }, DEMO_TICK_INTERVAL_MS);
        };

        const stompClient = new Client({
            brokerURL: BROKER_URL,
            reconnectDelay: 5000,
            heartbeatIncoming: 4000,
            heartbeatOutgoing: 4000,
        });

        stompClient.onConnect = () => {
            connected = true;
            clearTimeout(fallbackTimer);
            if (demoInterval) {
                clearInterval(demoInterval);
                demoInterval = null;
            }
            setConnectionStatus("live");
            stompClient.subscribe("/topic/live-alerts", (msg) => {
                if (msg.body) onAlertRef.current?.(JSON.parse(msg.body));
            });
            stompClient.subscribe("/topic/telemetry-ticks", (msg) => {
                if (msg.body) onTickRef.current?.(JSON.parse(msg.body));
            });
        };

        stompClient.onWebSocketClose = () => {
            if (connected) {
                connected = false;
                startDemoMode();
            }
        };

        stompClient.onStompError = () => {
            if (!connected) startDemoMode();
        };

        stompClient.activate();

        fallbackTimer = setTimeout(() => {
            if (!connected) startDemoMode();
        }, CONNECT_TIMEOUT_MS);

        return () => {
            clearTimeout(fallbackTimer);
            if (demoInterval) clearInterval(demoInterval);
            stompClient.deactivate();
        };
    }, []);

    return { connectionStatus };
};