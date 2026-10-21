# Xenium ROI Generation & GUI Conversion Guide

## Executive Summary

This document explains the end-to-end historical creation pipeline that produced:
1. Individual ROI cell tables such as:
   `D:\ADATA Backup\HuBMAP\HuBMAP Xenium\Xenium_data\Xen24_3990_ckd-selected\ckd-roi1-HSI_cells_gui_rois.parquet`
2. The unified multi-sample master table:
   `D:\ADATA Backup\HuBMAP\HuBMAP Xenium\Xenium_data\Xenium-round1-napari_rois.parquet`

It details the two distinct stages of this pipeline:
- **Stage 1**: The transformation and canvas cropping logic implemented in [`read_parquet.ipynb`](file:///c:/Users/jvill/Documents/GitHub/hsi_machine_learning/task_specific/Xenium_profile/read_parquet.ipynb).
- **Stage 2**: The aggregation and formatting logic executed by [`convert_roi_cells_to_gui_format.py`](file:///c:/Users/jvill/Documents/GitHub/hsi_machine_learning/task_specific/Xenium_profile/convert_roi_cells_to_gui_format.py).

---

## 1. Context: Direct vs. Indirect Transformation Files

A critical distinction in this dataset is the origin of the `.i2r.json` transformation files:

- **Direct Cell Segmentation Transform** (e.g. `ckd_cell_segmentation-roi1_transform.i2r.json`):
  - **Moving Image**: `ckd_cell_segmentation.ome.tiff` (raw slide space, matching `ckd_cell_segmentation.csv`).
  - **Fixed Image**: `Protein-ckd-roi1-HSI.tif` (the target ROI canvas: 4408 x 1973 px).
  - **Direction**: Forward affine matrix (`use_inverse=False`).
  - **Purpose**: Maps raw segmented cell polygons directly to the local HSI ROI canvas.

- **Indirect WSI Stitch Transform** (e.g. `Protein-ckd-roi1-HSI_transform.i2r.json`):
  - **Moving Image**: `Protein-ckd-roi1-HSI.tif` (the local ROI).
  - **Fixed Image**: `protein-result.tif` (the large whole-slide stitched SRS image).
  - **Direction**: Inverted matrix (`use_inverse=True`).
  - **Purpose**: Maps between the full stitched mosaic and the local ROI canvas.

The original round 1 ROIs were created using the **Direct Cell Segmentation Transform**.

---

## 2. Stage 1: The Logic Behind `read_parquet.ipynb`

The core coordinate transformation and cropping were performed in Cell 7 of [`read_parquet.ipynb`](file:///c:/Users/jvill/Documents/GitHub/hsi_machine_learning/task_specific/Xenium_profile/read_parquet.ipynb).

```
   Raw Tissue Coordinates                         Target ROI Canvas Space
[ckd_cell_segmentation.csv]                      [ckd-roi1-HSI_cells.parquet]
(vertex_x, vertex_y, x, y)                       (x_roi, y_roi [0..W, 0..H])
           │                                                  ▲
           ▼                                                  │
┌─────────────────────────┐                        ┌──────────────────────┐
│  Affine Transformation  │ ───────► Cropping & ───►  Vertical Y-Inversion│
│ (3x3 Matrix Projection) │          Clamping      │ (y = height - y_roi) │
└─────────────────────────┘                        └──────────────────────┘
```

### Step-by-Step Processing:

1. **Input Data Ingestion**:
   - **Cell Segments**: `ckd_cell_segmentation.csv` (contains per-vertex records with cell identifiers and raw coordinates).
   - **Metadata Annotations**: `_metadata.csv` (contains cell type classifications, such as `levelKID_refined`: `altCNT`, `PT`, `POD`, etc.).
   - **Transform JSON**: `ckd_cell_segmentation-roi1_transform.i2r.json` (extracts the 3x3 pixel matrix `matrix_xy_px` and target canvas dimensions `width=4408, height=1973`).

2. **Affine Coordinate Mapping & Integer Rounding**:
   - The cell vertices are projected using [`apply_roi_transform_from_json`](file:///c:/Users/jvill/Documents/GitHub/hsi_machine_learning/core/roi_coordinate_transformer.py):
     $$\begin{bmatrix} x_{\text{roi}} \\ y_{\text{roi}} \\ 1 \end{bmatrix} = \mathbf{M}_{3\times3} \begin{bmatrix} x_{\text{raw}} \\ y_{\text{raw}} \\ 1 \end{bmatrix}$$
   - Coordinates are rounded to the nearest integer pixel (`round_coordinates=True`):
     ```python
     transformed_coords = np.round(transformed).astype(np.int32)
     result_df['x_roi'] = transformed_coords[:, 0]
     result_df['y_roi'] = transformed_coords[:, 1]
     ```

3. **Intersect Cropping & Canvas Margin Clamping**:
   - Cells touching or intersecting the target canvas are identified:
     ```python
     roi_cells_transformed_cropped = roi_cells_transformed[
         (roi_cells_transformed['x_roi'] >= 0) & 
         (roi_cells_transformed['y_roi'] >= 0) &
         (roi_cells_transformed['x_roi'] < roi_bounds['width']) &
         (roi_cells_transformed['y_roi'] < roi_bounds['height'])
     ]
     cells_in_roi = roi_cells_transformed_cropped['cell_id'].unique()
     ```
   - All vertices belonging to those intersecting cells are retained, and any vertices lying outside the image border are clamped to the exact canvas margins:
     ```python
     total_roi_cells = roi_cells_transformed[roi_cells_transformed['cell_id'].isin(cells_in_roi)].copy()
     total_roi_cells[['x_roi', 'y_roi']] = total_roi_cells[['x_roi', 'y_roi']].clip(
         lower=[0, 0], upper=[roi_bounds['width'], roi_bounds['height']], axis=1
     )
     ```

4. **Metadata Merging & Coordinate Orientation (Vertical Flip)**:
   - Cell annotations are joined:
     ```python
     total_roi_cells = total_roi_cells.merge(df_metadata, on='cell_id', how='left')
     ```
   - The Y axis is inverted to match microscope image orientation (top-left origin):
     ```python
     total_roi_cells['y_roi'] = roi_bounds['height'] - total_roi_cells['y_roi']
     ```

5. **Stage 1 Output**:
   - Saved as an intermediate per-vertex table:
     `D:\ADATA Backup\HuBMAP\HuBMAP Xenium\Xenium_data\Xen24_3990_ckd-selected\ckd-roi1-HSI_cells.parquet` (300,955 rows representing 12,046 distinct cells).

---

## 3. Stage 2: What `convert_roi_cells_to_gui_format.py` Does

[`convert_roi_cells_to_gui_format.py`](file:///c:/Users/jvill/Documents/GitHub/hsi_machine_learning/task_specific/Xenium_profile/convert_roi_cells_to_gui_format.py) bridges the raw per-vertex table (`*_cells.parquet`) into the unified tabular format required by the Napari GUI (`multimodal_napari_gui` / `SegmentationWidget`).

```
  Per-Vertex Table                       Napari GUI Aggregated Table
[ckd-roi1-HSI_cells.parquet]        [ckd-roi1-HSI_cells_gui_rois.parquet]
(300,955 rows: 1 row per vertex)     (12,046 rows: 1 row per cell polygon)
             │                                        ▲
             ▼                                        │
  ┌──────────────────────┐               ┌────────────────────────┐
  │ Consecutive Vertex   │ ────────────► │ Polygon Area, Centroid,│
  │ Splitting by cell_id │               │ WKT & JSON Generation  │
  └──────────────────────┘               └────────────────────────┘
```

### Step-by-Step Processing:

1. **Auto-Detection of Columns**:
   - Detects `cell_id` from `['cell_id', 'UID', ...]`.
   - Detects coordinates from `['x_roi', 'y_roi']`.
   - Detects cell label from `['levelKID_refined', 'Label', 'group']`.

2. **Vertex Grouping & Splitting**:
   - Identifies boundaries between cells where `cell_id` changes:
     ```python
     change_idx = np.where(c_ids[:-1] != c_ids[1:])[0] + 1
     splits_xy = np.split(np.column_stack((vx, vy)), change_idx)
     splits_c = np.split(c_ids, change_idx)
     splits_l = np.split(lbls, change_idx)
     ```

3. **Polygon Closure & Geometry Calculations**:
   - Loops through each cell's vertex array `pts`:
     - **Closure**: If the first and last points are not identical, appends `pts[0]` to form a closed loop:
       ```python
       if len(pts) >= 2 and not np.allclose(pts[0], pts[-1]):
           pts_closed = np.vstack([pts, pts[0]])
       else:
           pts_closed = pts
       ```
     - **Well-Known Text (WKT)**: Formats standard geospatial WKT string:
       `POLYGON ((566.000 1284.000, 568.000 1288.000, ...))`
     - **Vertices JSON**: Formats JSON list of vertex pairs:
       `[[566.0, 1284.0], [568.0, 1288.0], ...]`
     - **Centroid**:
       $$\text{Centroid\_X} = \frac{1}{N}\sum x_i, \quad \text{Centroid\_Y} = \frac{1}{N}\sum y_i$$
     - **Shoelace Area**:
       $$\text{Area\_px} = \frac{1}{2}\left|\sum_{i=0}^{N-1} (x_i y_{i+1} - x_{i+1} y_i)\right|$$

4. **Standard GUI Schema Generation**:
   Produces a single table row per cell containing:
   | Column | Description | Example |
   | :--- | :--- | :--- |
   | `Sample_ID` | Identifier of sample or slide | `ckd-roi1-HSI` / `3990` |
   | `ROI_Number` | Sequential integer index | `1`, `2`, `...` |
   | `ROI_instance` | Per-label occurrence counter | `1`, `2`, `...` |
   | `Region_Index` | Region index for GUI selection | `1` |
   | `Label` | Biological cell annotation | `altCNT`, `PT`, `POD` |
   | `Fluorescence_Label` | Mirror of Label for fluorescence overlay | `altCNT` |
   | `Vertex_Count` | Number of closed vertices | `25` |
   | `Geometry_WKT` | Geospatial WKT representation | `POLYGON ((...))` |
   | `Vertices_JSON` | Array of `[x, y]` coordinates | `[[566.0, 1284.0], ...]` |
   | `Centroid_X` | Mean X coordinate | `576.64` |
   | `Centroid_Y` | Mean Y coordinate | `1277.68` |
   | `Area_px` | Shoelace polygon area | `490.0` |
   | `UID` | Unique cell identifier | `aldfibib-1` |
   | `cell_id` | Matching UID | `aldfibib-1` |
   | `Image_Name` | Target image filename for GUI binding | `ckd-roi1-HSI.tif` |
   | `Image_Path` | Filepath to associated TIFF canvas | `.../Protein-ckd-roi1-HSI.tif` |

5. **Stage 2 Output**:
   - Saved as GUI-ready Parquet and CSV:
     `D:\ADATA Backup\HuBMAP\HuBMAP Xenium\Xenium_data\Xen24_3990_ckd-selected\ckd-roi1-HSI_cells_gui_rois.parquet` (12,046 cell polygons).

---

## 4. Multi-Sample Aggregation into `Xenium-round1-napari_rois.parquet`

The master file `Xenium-round1-napari_rois.parquet` (74,475 total cells) was generated by concatenating the individual ROI tables created by the above pipeline across all six round 1 kidney regions:

| Region / Sample | Image_Name | Cell Count |
| :--- | :--- | :--- |
| KPMP_038 Ref ROI 1 | `ref-roi1-HSI.tif` | 13,874 |
| KPMP_038 Ref ROI 2 | `ref-roi2-HSI.tif` | 13,639 |
| Xen24_3782 AKI ROI 1 | `aki-roi1-HSI.tif` | 11,858 |
| Xen24_3782 AKI ROI 2 | `aki-roi2-HSI.tif` | 11,281 |
| Xen24_3990 CKD ROI 1 | `ckd-roi1-HSI.tif` | 12,046 |
| Xen24_3990 CKD ROI 2 | `ckd-roi2-HSI.tif` | 11,777 |
| **Total** | **6 Regions** | **74,475 cells** |

### Why This Format Avoids Line Artifacts in Napari:
1. **Self-Contained Polygons**: Each cell has a discrete, closed loop of coordinates without referencing adjacent or distant cells.
2. **Strict Margin Clamping**: Vertices never fall outside $[0, W]$ or $[0, H]$, preventing Napari shape layers from creating out-of-bounds cross-canvas lines.
3. **Pre-Rounded Integer Offsets**: Coordinate points align naturally with physical pixel grids, eliminating subpixel boundary collapse.
