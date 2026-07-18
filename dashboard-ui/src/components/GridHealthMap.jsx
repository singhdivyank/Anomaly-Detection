import { useMemo } from "react";
import { REGION_LABEL } from "./consts";

function seededPoints(count, seed) {
  const points = [];
  let s = seed;
  const rand = () => {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280;
  };
  for (let i = 0; i < count; i++) {
    points.push({ x: 8 + rand() * 84, y: 10 + rand() * 80 });
  }
  return points;
}

export default function GridHealthMap({ recentAlerts }) {
    const nodes = useMemo(() => seededPoints(70, 42), []);
    
    const anomalyHouseholds = useMemo(() => {
        const seen = new Map();
        for (const a of recentAlerts.slice(0, 6)) {
            if (!seen.has(a.household_id)) seen.set(a.household_id, a);
        }
        return Array.from(seen.values()).slice(0, 3);
    }, [recentAlerts]);

    return (
        <div className="flex h-full flex-col rounded-xl border border-gray-800/60 bg-surface/80 p-4 backdrop-blur-md">
        <div className="mb-1">
            <h2 className="font-display text-sm font-semibold text-gray-100">Geospatial Grid Health</h2>
        </div>
        <div className="mb-3 flex items-center gap-4 text-xs text-gray-400">
            <span className="flex items-center gap-1.5">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" /> Household nodes
            </span>
            <span className="flex items-center gap-1.5">
            <span className="h-1.5 w-1.5 rounded-full bg-threat-400" /> Anomaly node
            </span>
        </div>

        <div className="relative flex-1 overflow-hidden rounded-lg border border-gray-800/40 bg-[#0d1420]">
            <svg viewBox="0 0 100 100" className="h-full w-full" preserveAspectRatio="none">
            <defs>
                <radialGradient id="landmass" cx="45%" cy="55%" r="70%">
                <stop offset="0%" stopColor="#16202f" />
                <stop offset="100%" stopColor="#0d1420" />
                </radialGradient>
            </defs>
            <rect width="100" height="100" fill="url(#landmass)" />
            <path
                d="M0,70 Q20,55 15,35 Q10,15 30,10 L100,5 L100,100 L0,100 Z"
                fill="#111a27"
                opacity="0.6"
            />

            {nodes.map((p, i) => (
                <circle key={i} cx={p.x} cy={p.y} r={0.8} fill="#34d399" opacity={0.75} />
            ))}

            {anomalyHouseholds.map((a, i) => {
                const p = nodes[(i * 13) % nodes.length];
                return (
                <g key={a.household_id}>
                    <circle cx={p.x} cy={p.y} r={2.4} fill="#f87171" opacity={0.35}>
                    <animate attributeName="r" values="1.6;3.4;1.6" dur="1.8s" repeatCount="indefinite" />
                    <animate attributeName="opacity" values="0.5;0;0.5" dur="1.8s" repeatCount="indefinite" />
                    </circle>
                    <circle cx={p.x} cy={p.y} r={1.2} fill="#ef4444" stroke="#fff" strokeWidth={0.3} />
                </g>
                );
            })}
            </svg>

            <div className="pointer-events-none absolute inset-0">
            {anomalyHouseholds.map((a, i) => {
                const p = nodes[(i * 13) % nodes.length];
                return (
                <div
                    key={a.household_id}
                    className="absolute -translate-x-1/2 -translate-y-full rounded-md border border-threat-500/40 bg-[#0d1320]/95 px-2 py-1 text-[10px] leading-tight shadow-console"
                    style={{ left: `${p.x}%`, top: `${p.y}%` }}
                >
                    <div className="text-gray-500">Household ID:</div>
                    <div className="font-mono text-threat-400">{a.household_id}</div>
                </div>
                );
            })}
            </div>

            <span className="absolute bottom-2 right-2 rounded-full border border-gray-800/60 bg-[#0d1320]/80 px-2 py-0.5 text-[10px] text-gray-500">
            {REGION_LABEL}
            </span>
        </div>
        </div>
    );
}