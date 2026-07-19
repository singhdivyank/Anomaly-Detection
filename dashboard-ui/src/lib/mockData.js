const HOUSEHOLDS = [
  "MAC000123",
  "MAC000551",
  "MAC000217",
  "MAC000884",
  "MAC000042",
  "MAC000309",
];

const PRICING_TIERS = ["dToU_High", "Standard"];

// Mirrors GeospatialLookupService's BOROUGH_BOUNDS on the backend, so demo
// mode plots households in the same kind of tight central-London bounding
// boxes a real backend response would use.
const BOROUGH_BOUNDS = {
  Westminster: [51.493, 51.522, -0.177, -0.12],
  Camden: [51.515, 51.565, -0.19, -0.115],
  Islington: [51.53, 51.565, -0.125, -0.085],
  Hackney: [51.53, 51.57, -0.085, -0.035],
  "Tower Hamlets": [51.505, 51.535, -0.07, -0.005],
  Southwark: [51.47, 51.51, -0.115, -0.05],
  Lambeth: [51.44, 51.495, -0.145, -0.085],
  "Kensington and Chelsea": [51.485, 51.515, -0.21, -0.175],
};
const BOROUGHS = Object.keys(BOROUGH_BOUNDS);

// Java's String.hashCode() algorithm, reimplemented so demo-mode geospatial
// assignment follows the same deterministic-hash approach as the backend
// (not required to match exactly -- these are independent code paths -- but
// keeping the same method avoids two conflicting "sources of truth").
function stableHash(str) {
  let h = 0;
  for (let i = 0; i < str.length; i++) {
    h = (Math.imul(31, h) + str.charCodeAt(i)) | 0;
  }
  return h;
}

function stableFraction(seed) {
  const unsigned = stableHash(seed) >>> 0;
  return (unsigned % 100000) / 100000;
}

const geospatialCache = new Map();

function resolveGeospatial(householdId) {
  if (geospatialCache.has(householdId)) return geospatialCache.get(householdId);

  const borough = BOROUGHS[Math.abs(stableHash(householdId)) % BOROUGHS.length];
  const [minLat, maxLat, minLng, maxLng] = BOROUGH_BOUNDS[borough];
  const latitude = minLat + stableFraction(`${householdId}::lat`) * (maxLat - minLat);
  const longitude = minLng + stableFraction(`${householdId}::lng`) * (maxLng - minLng);

  const info = {
    borough,
    latitude: Number(latitude.toFixed(6)),
    longitude: Number(longitude.toFixed(6)),
  };
  geospatialCache.set(householdId, info);
  return info;
}

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
    geospatial: resolveGeospatial(householdId),
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
    geospatial: t.geospatial ?? resolveGeospatial(t.household_id),
  };
}