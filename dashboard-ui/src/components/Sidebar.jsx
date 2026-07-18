import { Settings, ChevronsLeft, Waves } from "lucide-react";
import { useState } from "react";
import { NAV_ITEMS } from "./consts";
import clsx from "clsx";

export default function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <aside
      className={clsx(
        "flex h-full flex-col border-r border-gray-800/60 bg-surface/80 backdrop-blur-md transition-all duration-200",
        collapsed ? "w-[68px]" : "w-[240px]"
      )}
    >
      <div className="flex items-center justify-between px-4 py-5">
        <div className="flex items-center gap-2 overflow-hidden">
          <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-emerald-400/10 text-emerald-400">
            <Waves size={18} strokeWidth={2.25} />
          </span>
          {!collapsed && (
            <span className="font-display text-sm font-semibold tracking-wide text-gray-100">
              GridPulse
            </span>
          )}
        </div>
        <button
          type="button"
          onClick={() => setCollapsed((c) => !c)}
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          className="rounded-md p-1 text-gray-500 hover:bg-white/5 hover:text-gray-300"
        >
          <ChevronsLeft
            size={16}
            className={clsx("transition-transform duration-200", collapsed && "rotate-180")}
          />
        </button>
      </div>

      <nav className="flex-1 space-y-1 px-3">
        {NAV_ITEMS.map(({ label, icon: Icon, active }) => (
          <button
            key={label}
            type="button"
            className={clsx(
              "flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-sm transition-colors",
              active
                ? "bg-white/5 text-gray-100"
                : "text-gray-400 hover:bg-white/5 hover:text-gray-200"
            )}
          >
            <Icon size={17} strokeWidth={2} className={clsx(active && "text-emerald-400")} />
            {!collapsed && <span className="truncate">{label}</span>}
          </button>
        ))}
      </nav>

      <div className="px-3 pb-5">
        <button
          type="button"
          className="flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-sm text-gray-400 hover:bg-white/5 hover:text-gray-200"
        >
          <Settings size={17} strokeWidth={2} />
          {!collapsed && <span>Settings</span>}
        </button>
      </div>
    </aside>
  );
}