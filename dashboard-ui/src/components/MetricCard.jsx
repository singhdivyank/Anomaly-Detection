import { TONE_STYLES } from "./consts";
import clsx from "clsx";

export default function MetricCard({ label, value, icon: Icon, tone = "neutral", suffix }) {
    const styles = TONE_STYLES[tone] ?? TONE_STYLES.neutral;

    return (
        <div className="rounded-xl border border-gray-800/60 bg-surface/80 p-4 backdrop-blur-md">
        <div className="flex items-start justify-between">
            <span className="text-xs text-gray-400">{label}</span>
            <span className={clsx("flex h-7 w-7 items-center justify-center rounded-md", styles.icon)}>
            <Icon size={14} strokeWidth={2.25} />
            </span>
        </div>
        <div className={clsx("mt-2 font-display text-2xl font-semibold tabular-nums", styles.value)}>
            {value}
            {suffix && <span className="ml-0.5 text-base">{suffix}</span>}
        </div>
        </div>
    );
}
