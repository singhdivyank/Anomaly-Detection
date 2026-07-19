import { Atom, Layers, Circle } from "lucide-react";
import { STATUS_CONFIG} from "./consts";
import clsx from "clsx";

export default function TopBar({ connectionStatus }) {
    const status = STATUS_CONFIG[connectionStatus] ?? STATUS_CONFIG.connecting;

    return (
    <header className="flex h-14 items-center justify-between border-b border-gray-800/60 bg-surface/80 px-6 backdrop-blur-md">
      <div className="flex items-center gap-2 text-sm text-gray-400">
        <span>Home</span>
      </div>

      <div className="flex items-center gap-3">
        <span
          className={clsx(
            "flex items-center gap-1.5 rounded-full border border-gray-800/60 px-2.5 py-1 font-mono text-[11px]",
            status.text
          )}
        >
          <Circle size={7} className={clsx("fill-current", status.dot)} strokeWidth={0} />
          {status.label}
        </span>

        <span className="flex items-center gap-1.5 rounded-full border border-gray-800/60 px-2.5 py-1 text-[11px] text-gray-400">
          <Atom size={13} className="text-activity-400" />
          React.js 19
        </span>

        <span className="flex items-center gap-1.5 rounded-full border border-gray-800/60 px-2.5 py-1 text-[11px] text-gray-400">
          <Layers size={13} className="text-activity-400" />
          Tailwind
        </span>

        <div className="h-7 w-7 rounded-full bg-gradient-to-br from-activity-400 to-emerald-400" />
      </div>
    </header>
  );
}