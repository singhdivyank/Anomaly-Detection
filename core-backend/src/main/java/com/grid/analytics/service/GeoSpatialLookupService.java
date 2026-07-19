package com.grid.analytics.service;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

import com.grid.analytics.dto.GeoSpatialInfo;
import com.grid.analytics.model.GridAsset;
import com.grid.analytics.repository.GridAssetRepository;
import org.springframework.dao.DataIntegrityViolationException;
import org.springframework.stereotype.Service;

/**
 * Resolves a household_id to a geospatial location by looking it up in the
 * `grid_assets` Core Asset Registry.
 *
 * IMPORTANT: the source London Smart Meter dataset (LCLid) is anonymized
 * and carries no real address/geocoordinate data. There is no real
 * geocoding happening here. On first sight of a household_id, this service
 * deterministically derives a stable borough + coordinate from a hash of
 * the ID (so the same household always lands in the same place across
 * restarts), persists it to `grid_assets`, and serves it from an in-memory
 * cache thereafter. In a real deployment, `grid_assets` would instead be
 * pre-populated by an actual meter-installation / asset-management system,
 * and this service would become a pure read-through cache.
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class GeoSpatialLookupService {

    private final GridAssetRepository gridAssetRepository;
    private final Map<String, GeoSpatialInfo> cache = new ConcurrentHashMap<>();
    private static final Map<String, double[]> BOROUGH_BOUNDS = new LinkedHashMap<>();

    static {
        BOROUGH_BOUNDS.put("Westminster", new double[] { 51.4930, 51.5220, -0.1770, -0.1200 });
        BOROUGH_BOUNDS.put("Camden", new double[] { 51.5150, 51.5650, -0.1900, -0.1150 });
        BOROUGH_BOUNDS.put("Islington", new double[] { 51.5300, 51.5650, -0.1250, -0.0850 });
        BOROUGH_BOUNDS.put("Hackney", new double[] { 51.5300, 51.5700, -0.0850, -0.0350 });
        BOROUGH_BOUNDS.put("Tower Hamlets", new double[] { 51.5050, 51.5350, -0.0700, -0.0050 });
        BOROUGH_BOUNDS.put("Southwark", new double[] { 51.4700, 51.5100, -0.1150, -0.0500 });
        BOROUGH_BOUNDS.put("Lambeth", new double[] { 51.4400, 51.4950, -0.1450, -0.0850 });
        BOROUGH_BOUNDS.put("Kensington and Chelsea", new double[] { 51.4850, 51.5150, -0.2100, -0.1750 });
    }
    private static final List<String> BOROUGHS = List.copyOf(BOROUGH_BOUNDS.keySet());

    public GeoSpatialInfo resolve(String householdId) {
        GeoSpatialInfo cached = cache.get(householdId);
        if (cached != null)
            return cached;

        synchronized (this) {
            cached = cache.get(householdId);
            if (cached != null)
                return cached;

            return gridAssetRepository.findById(householdId)
                    .map(asset -> cacheAndReturn(householdId,
                            new GeoSpatialInfo(asset.getBorough(), asset.getLatitude(), asset.getLongitude())))
                    .orElseGet(() -> provisionAsset(householdId));
        }
    }

    private GeoSpatialInfo provisionAsset(String householdId) {
        GeoSpatialInfo generated = deriveDeterministicLocation(householdId);
        GridAsset asset = GridAsset.builder()
                .householdId(householdId)
                .borough(generated.borough())
                .latitude(generated.latitude())
                .longitude(generated.longitude())
                .build();

        try {
            gridAssetRepository.save(asset);
            log.info("Provisioned new grid_assets entry for household={} borough={}",
                    householdId, generated.borough());
        } catch (DataIntegrityViolationException raceLost) {
            return gridAssetRepository.findById(householdId)
                    .map(a -> new GeoSpatialInfo(a.getBorough(), a.getLatitude(), a.getLongitude()))
                    .orElse(generated);
        }
        return cacheAndReturn(householdId, generated);
    }

    private GeoSpatialInfo cacheAndReturn(String householdId, GeoSpatialInfo info) {
        cache.put(householdId, info);
        return info;
    }

    private GeoSpatialInfo deriveDeterministicLocation(String householdId) {
        String borough = BOROUGHS.get(Math.floorMod(householdId.hashCode(), BOROUGHS.size()));
        double[] box = BOROUGH_BOUNDS.get(borough);

        double latFraction = stableFraction(householdId + "::lat");
        double lngFraction = stableFraction(householdId + "::lng");
        double latitude = box[0] + latFraction * (box[1] - box[0]);
        double longitude = box[2] + lngFraction * (box[3] - box[2]);

        return new GeoSpatialInfo(borough, round6(latitude), round6(longitude));
    }

    /** Deterministic pseudo-random value in [0, 1) derived from a string seed. */
    private double stableFraction(String seed) {
        long unsigned = seed.hashCode() & 0xffffffffL;
        return (unsigned % 100_000) / 100_000.0;
    }

    private double round6(double value) {
        return Math.round(value * 1_000_000.0) / 1_000_000.0;
    }
}
