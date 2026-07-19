package com.grid.analytics.repository;

import com.grid.analytics.model.GridAsset;
import org.springframework.data.jpa.repository.JpaRepository;

public interface GridAssetRepository extends JpaRepository<GridAsset, String> {
}
