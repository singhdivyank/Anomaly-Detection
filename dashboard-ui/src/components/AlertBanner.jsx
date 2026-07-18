import { AlertOctagon, MoreHorizontal } from "lucide-react";
import { SEVERITY_STYLES } from "./consts";
import clsx from "clsx";

function formatUtc(iso) {
  const d = new Date(iso);
  return d.toISOString().slice(11, 19) + "Z";
}

export default function AlertBanner({ alerts }) {
  return (
    <div className="rounded-xl border border-gray-800/60 bg-surface/80 p-4 backdrop-blur-md">
      <div className="mb-3 flex items-start justify-between">
        <div className="flex items-start gap-2">
          <span className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-threat-500/15 text-threat-400">
            <AlertOctagon size={13} />
          </span>
          <div>
            <h2 className="font-display text-sm font-semibold text-gray-100">Live Alerts Feed</h2>
            <span className="text-xs text-gray-500">Recent critical alerts in real-time</span>
          </div>
        </div>
        <button
          type="button"
          aria-label="Feed options"
          className="rounded-md p-1 text-gray-500 hover:bg-white/5 hover:text-gray-300"
        >
          <MoreHorizontal size={16} />
        </button>
      </div>

      <div className="grid grid-cols-[92px_1fr_120px_84px] gap-2 border-b border-gray-800/60 pb-2 text-[11px] uppercase tracking-wide text-gray-500">
        <span>Severity</span>
        <span>Alert Type</span>
        <span>Meter ID</span>
        <span className="text-right">Timestamp</span>
      </div>

      <div className="scrollbar-thin h-[280px] overflow-y-auto">
        {alerts.length === 0 ? (
          <div className="flex h-full items-center justify-center text-xs text-gray-500">
            No alerts yet — the grid is stable.
          </div>
        ) : (
          alerts.map((alert, i) => (
            <div
              key={`${alert.household_id}-${alert.timestamp}-${i}`}
              className="grid grid-cols-[92px_1fr_120px_84px] items-center gap-2 border-b border-gray-800/30 py-2.5 text-[12px] last:border-b-0"
            >
              <span>
                <span
                  className={clsx(
                    "inline-block rounded px-1.5 py-0.5 text-[10px] font-medium",
                    SEVERITY_STYLES[alert.severity] ?? SEVERITY_STYLES.Minor
                  )}
                >
                  [{alert.severity}]
                </span>
              </span>
              <span className="truncate text-gray-300">{alert.alert_type}</span>
              <span className="truncate font-mono tracking-wider text-activity-400">
                {alert.household_id}
              </span>
              <span className="text-right font-mono text-gray-500">
                {formatUtc(alert.timestamp)}
              </span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}