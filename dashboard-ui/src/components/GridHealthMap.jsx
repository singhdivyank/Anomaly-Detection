import { useMemo } from "react";
import { MapContainer, TileLayer, CircleMarker, Tooltip as LeafletTooltip } from "react-leaflet";
import { REGION_LABEL, LONDON_CENTER, DEFAULT_ZOOM } from "./consts";
import "leaflet/dist/leaflet.css";

const DARK_TILE_URL = "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png";
const DARK_TILE_ATTRIBUTION =
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>';

function formatUtc(iso) {
  return new Date(iso).toISOString().slice(11, 19) + "Z";
}

export default function GridHealthMap({ ticks, alerts }) {
  const householdNodes = useMemo(() => {
    const byHousehold = new Map();
    for (const t of ticks) {
      if (t.geospatial) byHousehold.set(t.household_id, t);
    }
    return Array.from(byHousehold.values());
  }, [ticks]);

  const anomalyNodes = useMemo(() => {
    const seen = new Map();
    for (const a of alerts) {
      if (a.geospatial && !seen.has(a.household_id)) seen.set(a.household_id, a);
    }
    return Array.from(seen.values()).slice(0, 8);
  }, [alerts]);

  return (
    <div className="flex h-full min-h-[420px] flex-col rounded-xl border border-gray-800/60 bg-surface/80 p-4 backdrop-blur-md">
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

      <div className="relative min-h-[320px] flex-1 overflow-hidden rounded-lg border border-gray-800/40">
        <MapContainer
          center={LONDON_CENTER}
          zoom={DEFAULT_ZOOM}
          scrollWheelZoom={false}
          className="h-full w-full"
          style={{ background: "#0d1420" }}
        >
          <TileLayer url={DARK_TILE_URL} attribution={DARK_TILE_ATTRIBUTION} />

          {householdNodes.map((t) => (
            <CircleMarker
              key={t.household_id}
              center={[t.geospatial.latitude, t.geospatial.longitude]}
              radius={5}
              pathOptions={{ color: "#34d399", fillColor: "#34d399", fillOpacity: 0.85, weight: 1 }}
            >
              <LeafletTooltip direction="top" offset={[0, -4]} opacity={1}>
                <div className="font-mono text-[11px]">
                  <div className="text-activity-400">{t.household_id}</div>
                  <div className="text-gray-500">{t.geospatial.borough}</div>
                  <div>{t.kw_consumed.toFixed(3)} kW</div>
                </div>
              </LeafletTooltip>
            </CircleMarker>
          ))}

          {anomalyNodes.map((a) => (
            <CircleMarker
              key={`anomaly-${a.household_id}`}
              center={[a.geospatial.latitude, a.geospatial.longitude]}
              radius={9}
              className="animate-pulse-ring"
              pathOptions={{ color: "#ef4444", fillColor: "#f87171", fillOpacity: 0.9, weight: 1.5 }}
            >
              <LeafletTooltip direction="top" offset={[0, -6]} opacity={1} permanent={false}>
                <div className="font-mono text-[11px]">
                  <div className="text-threat-400">
                    [{a.severity}] {a.household_id}
                  </div>
                  <div className="text-gray-500">{a.geospatial.borough}</div>
                  <div>{formatUtc(a.timestamp)}</div>
                </div>
              </LeafletTooltip>
            </CircleMarker>
          ))}
        </MapContainer>

        <span className="pointer-events-none absolute bottom-2 right-2 z-[1000] rounded-full border border-gray-800/60 bg-[#0d1320]/80 px-2 py-0.5 text-[10px] text-gray-500">
          {REGION_LABEL}
        </span>
      </div>
    </div>
  );
}