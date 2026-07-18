import { useCallback, useMemo, useRef, useState } from "react";
import { Radio, ZapOff, ShieldAlert, Gauge } from "lucide-react";
import Sidebar from "./components/Sidebar";
import TopBar from "./components/TopBar";
import MetricCard from "./components/MetricCard";
import RealTimeChart from "./components/RealTimeChart";
import AlertBanner from "./components/AlertBanner";
import GridHealthMap from "./components/GridHealthMap";
import { useGridWebSocket } from "./hooks/useGridWebSocket";

const CHART_WINDOW = 50;
const ALERT_FEED_LIMIT = 60;
const ACTIVE_METER_WINDOW_MS = 5 * 60 * 1000;

export default function App() {
    const [ticks, setTicks] = useState([]);
    const [alerts, setAlerts] = useState([]);
    const seenHouseholds = useRef(new Map());

    const handleTelemetryTick = useCallback((tick) => {
        seenHouseholds.current.set(tick.household_id, Date.now());
        setTicks((prev) => {
            const next = [...prev, tick];
            return next.length > CHART_WINDOW ? next.slice(next.length - CHART_WINDOW) : next;
        });
    }, []);

    const handleAlertTriggered = useCallback((alert) => {
        setAlerts((prev) => [alert, ...prev].slice(0, ALERT_FEED_LIMIT));
    }, []);

    const { connectionStatus } = useGridWebSocket(handleAlertTriggered, handleTelemetryTick);

    const metrics = useMemo(() => {
        const now = Date.now();
        let activeMeters = 0;
        for (const lastSeen of seenHouseholds.current.values()) {
            if (now - lastSeen <= ACTIVE_METER_WINDOW_MS) activeMeters += 1;
        }

        const gridOverloads = alerts.filter((a) => a.alert_type === "Grid Overload").length;
        const demandNonAdherence = alerts.filter((a) =>
            a.alert_type?.startsWith("Non-Adherence")
        ).length;

        const anomalousTicks = ticks.filter((t) => t.anomaly_detected).length;
        const systemEfficiency =
        ticks.length === 0 ? 100 : 100 * (1 - anomalousTicks / ticks.length);

        return {
            activeMeters,
            gridOverloads,
            demandNonAdherence,
            systemEfficiency: systemEfficiency.toFixed(1),
        };
    }, [ticks, alerts]);

    return (
        <div className="grid-texture flex h-screen bg-canvas text-gray-100">
            <Sidebar />

            <div className="flex min-w-0 flex-1 flex-col">
                <TopBar connectionStatus={connectionStatus} />

                <main className="flex-1 overflow-y-auto p-6">
                <h1 className="font-display text-xl font-semibold text-gray-100">
                    Live Grid Reliability Center <span className="text-gray-500">— North Region</span>
                </h1>

                <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
                    <MetricCard
                    label="Active Meters"
                    value={metrics.activeMeters.toLocaleString()}
                    icon={Radio}
                    tone="emerald"
                    />
                    <MetricCard
                    label="Grid Overloads"
                    value={metrics.gridOverloads}
                    icon={ZapOff}
                    tone="threat"
                    />
                    <MetricCard
                    label="Demand Non-Adherence"
                    value={metrics.demandNonAdherence}
                    icon={ShieldAlert}
                    tone="threat"
                    />
                    <MetricCard
                    label="System Efficiency"
                    value={metrics.systemEfficiency}
                    suffix="%"
                    icon={Gauge}
                    tone="emerald"
                    />
                </div>

                <div className="mt-4 grid grid-cols-1 gap-4 xl:grid-cols-3">
                    <div className="space-y-4 xl:col-span-2">
                    <RealTimeChart ticks={ticks} />
                    <AlertBanner alerts={alerts} />
                    </div>
                    <GridHealthMap recentAlerts={alerts} />
                </div>
                </main>
            </div>
        </div>
    );
}