#!/usr/bin/env bash
# ==============================================================================
# run_step4_split_mosaic_rois.sh
# ------------------------------------------------------------------------------
# Splits transformed/coregistered ROI cell tables (e.g. ref-roi1-HSI_cells_gui_rois.parquet)
# into smaller (possibly overlapping) subsections matching an N x M mosaic grid
# of raw HSI / image tiles.
#
# USAGE:
#   ./run_step4_split_mosaic_rois.sh
#
# Override variables via environment or CLI:
#   INPUT_FILE="/path/to/ref-roi1-HSI_cells_gui_rois.parquet" \
#   TILE_DIR="/path/to/processed-tifs" \
#   OUTPUT_DIR="/path/to/subsections" \
#   ./run_step4_split_mosaic_rois.sh
#
# Or with manual grid parameters:
#   GRID_ROWS=9 GRID_COLS=4 OVERLAP_PERCENT=5.0 ./run_step4_split_mosaic_rois.sh
# ==============================================================================

set -e

# --- CONFIGURABLE VARIABLES (Edit these or pass as environment variables) -----
INPUT_FILE="${INPUT_FILE:-D:/ADATA Backup/HuBMAP/HuBMAP Xenium/Xenium_data/Xen24_KPMP_038_ref-selected/ref-roi1-HSI_cells_gui_rois.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-D:/ADATA Backup/HuBMAP/HuBMAP Xenium/Xenium_data/Xen24_KPMP_038_ref-selected/ref-roi1-HSI_cells_gui_rois_subsections}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-ref-roi1}"

# Option A: Automatic discovery from tile directory
TILE_DIR="${TILE_DIR:-E:/Data/Jorge/01_HumanKidney_HuBMAP/Xenium_datasets/Ref/ROI1}"

# Option B: Manual grid parameters (used if TILE_DIR is not provided or to override)
GRID_ROWS="${GRID_ROWS:-}"
GRID_COLS="${GRID_COLS:-}"
TILE_HEIGHT="${TILE_HEIGHT:-}"
TILE_WIDTH="${TILE_WIDTH:-}"
OVERLAP_PERCENT="${OVERLAP_PERCENT:-5.0}"
ORDER="${ORDER:-snake}"

# Coordinate mode: 'local' (0 to tile_w/h), 'global' (canvas coords), or 'both' (both stored, primary local)
COORD_MODE="${COORD_MODE:-local}"

# Boundary inclusion mode: 'intersect' (overlapping margins included in both tiles), 'centroid', or 'strict'
CROP_MODE="${CROP_MODE:-intersect}"

# Clip polygon vertices strictly to tile boundaries [0, W] x [0, H]
CLIP_VERTICES="${CLIP_VERTICES:-true}"

SAMPLE_ID="${SAMPLE_ID:-1}"
FORMAT="${FORMAT:-both}"
GENERATE_MASKS="${GENERATE_MASKS:-true}"
PIXEL_SIZE="${PIXEL_SIZE:-0.4971847}"

# Cohesive vs individual tile parquet/csv output
SAVE_INDIVIDUAL="${SAVE_INDIVIDUAL:-false}"
COHESIVE_NAME="${COHESIVE_NAME:-}"
STITCHED_IMAGE="${STITCHED_IMAGE:-}"
TARGET_CANVAS_SHAPE="${TARGET_CANVAS_SHAPE:-}"
# ------------------------------------------------------------------------------

# Detect Python interpreter
if [ -n "$PYTHON_EXEC" ] && [ -x "$PYTHON_EXEC" ]; then
    : # User specified PYTHON_EXEC
elif [ -f "/opt/miniconda3/envs/gui_analysis_312/bin/python" ]; then
    PYTHON_EXEC="/opt/miniconda3/envs/gui_analysis_312/bin/python"
elif [ -f "/opt/miniconda3/envs/gui_analysis/bin/python" ]; then
    PYTHON_EXEC="/opt/miniconda3/envs/gui_analysis/bin/python"
elif [ -f "/opt/miniconda3/envs/hsi_machine_learning/bin/python" ]; then
    PYTHON_EXEC="/opt/miniconda3/envs/hsi_machine_learning/bin/python"
elif [ -n "$CONDA_PREFIX" ] && [ -f "$CONDA_PREFIX/bin/python" ]; then
    PYTHON_EXEC="$CONDA_PREFIX/bin/python"
else
    PYTHON_EXEC="python3"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/split_roi_mosaic_subsections.py"

echo "======================================================================"
echo "RUNNING STEP 4: GENERATE COHESIVE MOSAIC SUBSECTIONS"
echo "======================================================================"
echo "Python Exec     : $PYTHON_EXEC"
echo "Input ROI Table : $INPUT_FILE"
echo "Output Directory: $OUTPUT_DIR"
echo "Tile Directory  : $TILE_DIR"
echo "Grid Layout     : ${GRID_ROWS:-auto} rows x ${GRID_COLS:-auto} cols (overlap ${OVERLAP_PERCENT}%)"
echo "Coordinate Mode : $COORD_MODE"
echo "Crop/Overlap    : $CROP_MODE"
echo "Save Individual : $SAVE_INDIVIDUAL"
echo "Format          : $FORMAT"
echo "======================================================================"

ARGS=(
  --input "$INPUT_FILE"
  --output-dir "$OUTPUT_DIR"
  --overlap-percent "$OVERLAP_PERCENT"
  --order "$ORDER"
  --coord-mode "$COORD_MODE"
  --crop-mode "$CROP_MODE"
  --sample-id "$SAMPLE_ID"
  --format "$FORMAT"
  --pixel-size "$PIXEL_SIZE"
)

if [ -n "$OUTPUT_PREFIX" ]; then
    ARGS+=(--output-prefix "$OUTPUT_PREFIX")
fi

if [ -n "$TILE_DIR" ]; then
    ARGS+=(--tile-dir "$TILE_DIR")
fi

if [ -n "$GRID_ROWS" ]; then
    ARGS+=(--grid-rows "$GRID_ROWS")
fi

if [ -n "$GRID_COLS" ]; then
    ARGS+=(--grid-cols "$GRID_COLS")
fi

if [ -n "$TILE_HEIGHT" ]; then
    ARGS+=(--tile-height "$TILE_HEIGHT")
fi

if [ -n "$TILE_WIDTH" ]; then
    ARGS+=(--tile-width "$TILE_WIDTH")
fi

if [ "$CLIP_VERTICES" = "true" ]; then
    ARGS+=(--clip-vertices)
fi

if [ "$GENERATE_MASKS" = "true" ]; then
    ARGS+=(--generate-masks)
fi

if [ "$SAVE_INDIVIDUAL" = "true" ]; then
    ARGS+=(--save-individual)
fi

if [ -n "$COHESIVE_NAME" ]; then
    ARGS+=(--cohesive-name "$COHESIVE_NAME")
fi

if [ -n "$STITCHED_IMAGE" ]; then
    ARGS+=(--stitched-image "$STITCHED_IMAGE")
fi

if [ -n "$TARGET_CANVAS_SHAPE" ]; then
    ARGS+=(--target-canvas-shape "$TARGET_CANVAS_SHAPE")
fi

"$PYTHON_EXEC" "$PYTHON_SCRIPT" "${ARGS[@]}" "$@"
