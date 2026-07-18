export const STATUS_CONFIG = {
  live: { label: "Live", dot: "bg-emerald-400", text: "text-emerald-400" },
  demo: { label: "Demo Mode", dot: "bg-activity-400", text: "text-activity-400" },
  connecting: { label: "Connecting…", dot: "bg-gray-500", text: "text-gray-400" },
};

export const SEVERITY_STYLES = {
  Severe: "text-threat-400 bg-threat-950/40 border border-threat-500/40",
  Moderate: "text-amber-300 bg-amber-950/30 border border-amber-700/40",
  Minor: "text-gray-300 bg-white/5 border border-gray-700/40",
};

export const TONE_STYLES = {
  neutral: { icon: "bg-white/5 text-gray-300", value: "text-gray-100" },
  emerald: { icon: "bg-emerald-400/10 text-emerald-400", value: "text-emerald-400" },
  threat: { icon: "bg-threat-500/10 text-threat-400", value: "text-threat-400" },
};

export const NAV_ITEMS = [
  { label: "Operations Overview", icon: LayoutGrid, active: true },
  { label: "Meter Streams", icon: Radio },
  { label: "Live Alerts", icon: AlertTriangle },
  { label: "Demand Response", icon: Repeat },
];

export const WINDOW_SIZE = 50;
export const REGION_LABEL = "North Region";