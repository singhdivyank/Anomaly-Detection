package com.grid.analytics.model;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

/**
 * Maps to the relational `grid_assets` table -- the Core Asset Registry
 * that resolves an anonymized smart meter ID (household_id) to a
 * geospatial location. The source smart meter dataset (LCLid) has no real
 * address data, so this registry is provisioned on demand by
 * GeospatialLookupService using a deterministic borough/coordinate
 * assignment rather than a real address lookup -- see that class for
 * details.
 */
@Entity
@Table(name = "grid_assets")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class GridAsset {

    @Id
    @Column(name = "household_id", length = 50)
    private String householdId;

    @Column(name = "borough", nullable = false, length = 100)
    private String borough;

    @Column(name = "latitude", nullable = false)
    private double latitude;

    @Column(name = "longitude", nullable = false)
    private double longitude;
}
