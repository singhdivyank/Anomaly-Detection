export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        canvas: "#0b0f19",
        surface: "#131a26",
        line: "#1f2937",
        emerald: {
          400: "#34d399",
        },
        threat: {
          400: "#f87171",
          500: "#ef4444",
          950: "#2c0b0b",
        },
        activity: {
          400: "#60a5fa",
        },
      },
      fontFamily: {
        display: ["'Space Grotesk'", "sans-serif"],
        body: ["'Inter'", "sans-serif"],
        mono: ["'JetBrains Mono'", "ui-monospace", "monospace"],
      },
      boxShadow: {
        console: "0 0 0 1px rgba(148,163,184,0.06), 0 20px 60px -20px rgba(0,0,0,0.6)",
      },
      keyframes: {
        scan: {
          "0%": { transform: "translateY(-100%)" },
          "100%": { transform: "translateY(100%)" },
        },
        "pulse-ring": {
          "0%": { transform: "scale(0.6)", opacity: "0.9" },
          "100%": { transform: "scale(2.2)", opacity: "0" },
        },
      },
      animation: {
        scan: "scan 8s linear infinite",
        "pulse-ring": "pulse-ring 1.6s cubic-bezier(0.2,0.6,0.4,1) infinite",
      },
    },
  },
  plugins: [],
};