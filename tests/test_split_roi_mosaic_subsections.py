#!/usr/bin/env python3
"""
Test script for split_roi_mosaic_subsections.py
"""

import os
import sys
import json
import shutil
import tempfile
import numpy as np
import pandas as pd
import tifffile
from pathlib import Path

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from task_specific.Xenium_profile.split_roi_mosaic_subsections import (
    split_roi_table_into_mosaic_tiles,
    build_mosaic_grid_layout,
    discover_tiff_tiles,
)


def create_synthetic_gui_rois(output_path: Path, canvas_w: int = 1900, canvas_h: int = 1900):
    """Generate synthetic cells across the canvas, including in overlap areas."""
    records = []
    
    # Cell 1: In top-left quadrant (x: 100-200, y: 100-200)
    pts1 = np.array([[100, 100], [200, 100], [200, 200], [100, 200], [100, 100]])
    records.append({
        "Sample_ID": "SYNTH_01",
        "ROI_Number": 1,
        "ROI_instance": 1,
        "Region_Index": 1,
        "Label": "Glomerulus",
        "Fluorescence_Label": "Glomerulus",
        "Vertex_Count": len(pts1),
        "Geometry_WKT": "POLYGON ((100 100, 200 100, 200 200, 100 200, 100 100))",
        "Vertices_JSON": json.dumps(pts1.tolist()),
        "Centroid_X": 150.0,
        "Centroid_Y": 150.0,
        "Area_px": 10000.0,
        "UID": "cell_001",
        "cell_id": "cell_001",
        "Image_Name": "synthetic_canvas",
        "Image_Path": "synthetic.tif",
    })

    # Cell 2: Straddling boundary between Col 0 and Col 1
    # Tile size 1000, 10% overlap -> step_x = 900.
    # Tile 0: [0, 1000], Tile 1: [900, 1900]. Overlap region: [900, 1000].
    # Cell 2 placed at [880, 100] to [950, 150] (intersects both Tile 0 and Tile 1)
    pts2 = np.array([[880, 100], [950, 100], [950, 150], [880, 150], [880, 100]])
    records.append({
        "Sample_ID": "SYNTH_01",
        "ROI_Number": 2,
        "ROI_instance": 1,
        "Region_Index": 2,
        "Label": "Tubule",
        "Fluorescence_Label": "Tubule",
        "Vertex_Count": len(pts2),
        "Geometry_WKT": "POLYGON ((880 100, 950 100, 950 150, 880 150, 880 100))",
        "Vertices_JSON": json.dumps(pts2.tolist()),
        "Centroid_X": 915.0,
        "Centroid_Y": 125.0,
        "Area_px": 3500.0,
        "UID": "cell_002_overlap",
        "cell_id": "cell_002_overlap",
        "Image_Name": "synthetic_canvas",
        "Image_Path": "synthetic.tif",
    })

    # Cell 3: In bottom-right tile (x: 1200-1300, y: 1200-1300)
    pts3 = np.array([[1200, 1200], [1300, 1200], [1300, 1300], [1200, 1300], [1200, 1200]])
    records.append({
        "Sample_ID": "SYNTH_01",
        "ROI_Number": 3,
        "ROI_instance": 2,
        "Region_Index": 3,
        "Label": "Glomerulus",
        "Fluorescence_Label": "Glomerulus",
        "Vertex_Count": len(pts3),
        "Geometry_WKT": "POLYGON ((1200 1200, 1300 1200, 1300 1300, 1200 1300, 1200 1200))",
        "Vertices_JSON": json.dumps(pts3.tolist()),
        "Centroid_X": 1250.0,
        "Centroid_Y": 1250.0,
        "Area_px": 10000.0,
        "UID": "cell_003",
        "cell_id": "cell_003",
        "Image_Name": "synthetic_canvas",
        "Image_Path": "synthetic.tif",
    })

    df = pd.DataFrame(records)
    df.to_parquet(output_path, index=False)
    return df


def test_split_workflow():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        input_pq = tmp_path / "test_gui_rois.parquet"
        out_dir = tmp_path / "subsections"

        create_synthetic_gui_rois(input_pq)

        # 1. Test Default Cohesive Output (Manual 2x2 Grid with 10% overlap, no individual files)
        print("\n--- Testing Default Cohesive Output (Manual Grid Specification) ---")
        manifest = split_roi_table_into_mosaic_tiles(
            input_path=input_pq,
            output_dir=out_dir,
            grid_rows=2,
            grid_cols=2,
            tile_height=1000,
            tile_width=1000,
            overlap_percent=10.0,
            coord_mode="both",
            crop_mode="intersect",
            clip_vertices=True,
        )

        assert manifest["total_source_cells"] == 3
        assert len(manifest["tiles"]) == 4

        # By default, individual parquet files should NOT be saved
        tile0 = manifest["tiles"][0]
        assert tile0["cell_count"] == 2, f"Expected 2 cells in tile 0, got {tile0['cell_count']}"
        assert "parquet" not in tile0["output_files"], "Individual parquet should not be exported by default"

        # Cohesive parquet and csv must exist
        assert "parquet" in manifest["cohesive_output_files"]
        assert "csv" in manifest["cohesive_output_files"]
        cohesive_pq = Path(manifest["cohesive_output_files"]["parquet"])
        assert cohesive_pq.exists()

        df_cohesive = pd.read_parquet(cohesive_pq)
        # Total records in cohesive output: tile 0 (2) + tile 1 (1) + tile 2 (0) + tile 3 (1) = 4 records
        assert len(df_cohesive) == 4, f"Expected 4 cohesive records, got {len(df_cohesive)}"
        assert manifest["total_subsections_cells"] == 4

        # Verify Tile 0 cells in cohesive dataframe
        df_t0 = df_cohesive[df_cohesive["Image_Name"] == tile0["tile_name"]]
        assert len(df_t0) == 2
        assert "cell_001" in df_t0["cell_id"].values
        assert "cell_002_overlap" in df_t0["cell_id"].values

        # Verify Tile 1 cells in cohesive dataframe
        tile1 = manifest["tiles"][1]
        assert tile1["cell_count"] == 1, f"Expected 1 cell in tile 1, got {tile1['cell_count']}"
        df_t1 = df_cohesive[df_cohesive["Image_Name"] == tile1["tile_name"]]
        assert len(df_t1) == 1
        assert "cell_002_overlap" in df_t1["cell_id"].values

        # Verify local coordinate transform on Tile 1 for cell_002_overlap:
        # Global x range [880, 950] clipped to Tile 1 [900, 1900] is [900, 950].
        # In Tile 1 local coords [0, 50] -> Local Centroid_X is 25.0, Canvas Centroid_X is 925.0!
        row_t1 = df_t1[df_t1["cell_id"] == "cell_002_overlap"].iloc[0]
        assert abs(row_t1["Centroid_X"] - 25.0) < 1e-2, f"Expected local Centroid_X 25.0, got {row_t1['Centroid_X']}"
        assert abs(row_t1["Centroid_X_canvas"] - 925.0) < 1e-2

        print("Manual Grid & Cohesive Overlap Assignment verified successfully!")

        # 2. Test Tile Directory Automatic Discovery with Image Names
        print("\n--- Testing Automatic Tile Directory Discovery & Image Names ---")
        tiles_dir = tmp_path / "raw_tiles"
        tiles_dir.mkdir()
        # Create 4 dummy TIFF tiles
        for i in range(1, 5):
            dummy_img = np.zeros((1000, 1000), dtype=np.uint16)
            tifffile.imwrite(tiles_dir / f"ROI_HSI_tile_{i:02d}.tif", dummy_img)

        discovered = discover_tiff_tiles(tiles_dir)
        assert len(discovered) == 4
        assert [p.name for p in discovered] == [
            "ROI_HSI_tile_01.tif",
            "ROI_HSI_tile_02.tif",
            "ROI_HSI_tile_03.tif",
            "ROI_HSI_tile_04.tif"
        ]

        out_dir_auto = tmp_path / "subsections_auto"
        manifest_auto = split_roi_table_into_mosaic_tiles(
            input_path=input_pq,
            output_dir=out_dir_auto,
            tile_dir=tiles_dir,
            grid_rows=2,
            grid_cols=2,
            overlap_percent=10.0,
            coord_mode="both",
            crop_mode="intersect",
        )

        assert manifest_auto["tiles"][0]["tile_name"] == "ROI_HSI_tile_01"
        assert manifest_auto["tiles"][1]["tile_name"] == "ROI_HSI_tile_02"
        # In snake mode, row 1 is reversed (tile 04 then tile 03)
        assert manifest_auto["tiles"][2]["tile_name"] == "ROI_HSI_tile_04"
        assert manifest_auto["tiles"][3]["tile_name"] == "ROI_HSI_tile_03"

        # Check cohesive parquet output from auto-discovery
        df_auto_cohesive = pd.read_parquet(manifest_auto["cohesive_output_files"]["parquet"])
        unique_image_names = set(df_auto_cohesive["Image_Name"].unique())
        # Tile 0 has cell_001 & cell_002_overlap -> ROI_HSI_tile_01
        # Tile 1 has cell_002_overlap -> ROI_HSI_tile_02
        # Tile 3 (bottom right) has cell_003 -> ROI_HSI_tile_03
        assert "ROI_HSI_tile_01" in unique_image_names
        assert "ROI_HSI_tile_02" in unique_image_names
        assert "ROI_HSI_tile_03" in unique_image_names

        # Check that Image_Path is populated with real file path
        row_img = df_auto_cohesive[df_auto_cohesive["Image_Name"] == "ROI_HSI_tile_01"].iloc[0]
        assert "ROI_HSI_tile_01.tif" in row_img["Image_Path"]

        print("Tile Directory Discovery & Image Names verified successfully!")

        # 3. Test Save Individual Option
        print("\n--- Testing save_individual=True Option ---")
        out_dir_indiv = tmp_path / "subsections_individual"
        manifest_indiv = split_roi_table_into_mosaic_tiles(
            input_path=input_pq,
            output_dir=out_dir_indiv,
            grid_rows=2,
            grid_cols=2,
            tile_height=1000,
            tile_width=1000,
            overlap_percent=10.0,
            save_individual=True,
        )
        assert "parquet" in manifest_indiv["tiles"][0]["output_files"]
        assert Path(manifest_indiv["tiles"][0]["output_files"]["parquet"]).exists()
        assert "parquet" in manifest_indiv["cohesive_output_files"]
        print("Save individual toggle verified successfully!")

        # 4. Test Step 3 Integration (transform_and_crop_rois with split_mosaic=True)
        print("\n--- Testing Step 3 transform_and_crop_rois with Integrated Splitting ---")
        from task_specific.Xenium_profile.transform_and_crop_rois import transform_and_crop_rois

        # Create dummy identity transform JSON
        transform_json_path = tmp_path / "test_transform.i2r.json"
        with open(transform_json_path, "w") as f:
            json.dump({
                "matrix_xy_px": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "fixed_paths": [{"image_shape": [1900, 1900], "pixel_size_um": 1.0}],
            }, f, indent=2)

        df_res, saved = transform_and_crop_rois(
            input_path=str(input_pq),
            transform_json=str(transform_json_path),
            output_prefix=str(tmp_path / "step3_out"),
            offset_x=0.0,
            offset_y=0.0,
            flip_y=False,
            generate_mask=False,
            split_mosaic=True,
            split_grid_rows=2,
            split_grid_cols=2,
            split_tile_height=1000,
            split_tile_width=1000,
            split_overlap_percent=10.0,
        )

        assert "split_manifest" in saved
        assert len(saved["split_manifest"]["tiles"]) == 4
        assert "cohesive_parquet" in saved
        assert Path(saved["cohesive_parquet"]).exists()
        print("Step 3 transform_and_crop_rois integrated splitting verified successfully!")


if __name__ == "__main__":
    test_split_workflow()
    print("\nALL TESTS PASSED!")
