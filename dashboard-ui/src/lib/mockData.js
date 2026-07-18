const HOUSEHOLDS = [
  "MAC000123",
  "MAC000551",
  "MAC000217",
  "MAC000884",
  "MAC000042",
  "MAC000309",
];

const PRICING_TIERS = ["dToU_High", "Standard"];

function diurnalBaseline(hour) {
  return (
    0.6 * Math.sin(((hour - 7) / 24) * 2 * Math.PI) +
    0.4 * Math.sin(((hour - 18) / 24) * 2 * Math.PI)
  );
}

let tick = 0;

export function generateTelemetryTick() {
  tick += 1;
  const now = new Date();
  const hour = now.getHours() + now.getMinutes() / 60;
  const householdId = HOUSEHOLDS[tick % HOUSEHOLDS.length];
  const baseline = diurnalBaseline(hour);
  const noise = (Math.random() - 0.5) * 0.35;

  // ~6% chance of a demand spike / technical fault, echoing the anomaly
  // injection rate used in model training.
  const isAnomaly = Math.random() < 0.06;
  const spike = isAnomaly ? (Math.random() > 0.5 ? 1 : -1) * (2.5 + Math.random() * 3) : 0;
  const kwConsumed = Number((baseline + noise + spike).toFixed(3));
  const anomalyScore = isAnomaly
    ? Number((-0.05 - Math.random() * 0.15).toFixed(4))
    : Number((0.02 + Math.random() * 0.15).toFixed(4));
  const confidenceScore = isAnomaly
    ? Number((0.7 + Math.random() * 0.29).toFixed(4))
    : Number((Math.random() * 0.15).toFixed(4));

  return {
    household_id: householdId,
    kw_consumed: kwConsumed,
    expected_baseline: Number(baseline.toFixed(3)),
    anomaly_score: anomalyScore,
    anomaly_detected: isAnomaly,
    confidence_score: confidenceScore,
    pricing_tier: PRICING_TIERS[tick % PRICING_TIERS.length],
    timestamp: now.toISOString(),
  };
}

export function tickToAlert(t) {
  const alertType =
    t.pricing_tier === "dToU_High"
      ? "Non-Adherence (dToU_High)"
      : t.kw_consumed > 5
        ? "Grid Overload"
        : "Technical Anomaly";
  const severity =
    t.confidence_score >= 0.85 ? "Severe" : t.confidence_score >= 0.5 ? "Moderate" : "Minor";

  return {
    household_id: t.household_id,
    severity,
    alert_type: alertType,
    kw_consumed: t.kw_consumed,
    anomaly_score: t.anomaly_score,
    confidence_score: t.confidence_score,
    timestamp: t.timestamp,
  };
}