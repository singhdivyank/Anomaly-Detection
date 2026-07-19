import { useMemo } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";
import { MoreHorizontal } from "lucide-react";
import { WINDOW_SIZE } from "./consts";

function formatClock(iso) {
  const d = new Date(iso);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function AnomalyDot(props) {
  const { cx, cy, payload } = props;
  if (!payload?.anomaly_detected || cx == null || cy == null) return null;

  return (
    <g>
      <circle cx={cx} cy={cy} r={8} fill="#f87171" opacity={0.35} className="animate-pulse-ring" />
      <circle cx={cx} cy={cy} r={4} fill="#ef4444" stroke="#ffffff" strokeWidth={1} />
    </g>
  );
}

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const point = payload[0]?.payload;
  if (!point) return null;

  return (
    <div className="rounded-lg border border-gray-800/60 bg-[#0d1320]/95 px-3 py-2 text-[11px] shadow-console backdrop-blur-md">
      <div className="mb-1 font-mono text-activity-400">{point.household_id}</div>
      <div className="flex justify-between gap-6 text-gray-400">
        <span>Load</span>
        <span className="font-mono text-gray-100">{point.kw_consumed.toFixed(3)} kW</span>
      </div>
      <div className="flex justify-between gap-6 text-gray-400">
        <span>Anomaly score</span>
        <span
          className={`font-mono ${point.anomaly_detected ? "text-threat-400" : "text-gray-100"}`}
        >
          {point.anomaly_score.toFixed(4)}
        </span>
      </div>
      <div className="mt-1 text-gray-500">{label}</div>
    </div>
  );
}

export default function RealTimeChart({ ticks }) {
  const data = useMemo(() => {
    return ticks.slice(-WINDOW_SIZE).map((t) => ({
      ...t,
      clock: formatClock(t.timestamp),
    }));
  }, [ticks]);

  const latestAnomaly = [...data].reverse().find((d) => d.anomaly_detected);

  return (
    <div className="rounded-xl border border-gray-800/60 bg-surface/80 p-4 backdrop-blur-md">
      <div className="mb-1 flex items-start justify-between">
        <div>
          <h2 className="font-display text-sm font-semibold text-gray-100">
            Real-Time Smart Meter Telemetry
          </h2>
          <span className="text-xs text-gray-500">Global Aggregate</span>
        </div>
        <button
          type="button"
          aria-label="Chart options"
          className="rounded-md p-1 text-gray-500 hover:bg-white/5 hover:text-gray-300"
        >
          <MoreHorizontal size={16} />
        </button>
      </div>

      <div className="mb-2 flex items-center gap-4 text-xs text-gray-400">
        <span className="flex items-center gap-1.5">
          <span className="h-0.5 w-3.5 rounded-full bg-emerald-400" /> Expected Baseline
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-0.5 w-3.5 rounded-full bg-white" /> Current Usage (kW)
        </span>
      </div>

      <div className="h-[220px]">
        {data.length === 0 ? (
          <div className="flex h-full items-center justify-center text-xs text-gray-500">
            Waiting for telemetry…
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 10, right: 8, bottom: 0, left: -18 }}>
              <CartesianGrid stroke="#1f2937" strokeDasharray="3 5" vertical={false} />
              <XAxis
                dataKey="clock"
                tick={{ fill: "#64748b", fontSize: 10 }}
                axisLine={{ stroke: "#1f2937" }}
                tickLine={false}
                minTickGap={40}
              />
              <YAxis
                tick={{ fill: "#64748b", fontSize: 10 }}
                axisLine={false}
                tickLine={false}
                width={36}
              />
              <Tooltip content={<ChartTooltip />} cursor={{ stroke: "#334155", strokeWidth: 1 }} />
              <Line
                type="monotone"
                dataKey="expected_baseline"
                stroke="#34d399"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
              <Line
                type="linear"
                dataKey="kw_consumed"
                stroke="#ffffff"
                strokeWidth={2.5}
                dot={<AnomalyDot />}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      {latestAnomaly && (
        <div className="mt-2 flex items-center gap-2 rounded-md border border-threat-500/30 bg-threat-950/40 px-3 py-1.5 text-[11px] text-threat-400">
          <span className="h-1.5 w-1.5 rounded-full bg-threat-400" />
          Anomaly detected — <span className="font-mono">{latestAnomaly.household_id}</span> at{" "}
          {formatClock(latestAnomaly.timestamp)}
        </div>
      )}
    </div>
  );
}