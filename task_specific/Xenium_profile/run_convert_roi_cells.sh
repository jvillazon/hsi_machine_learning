#!/usr/bin/env bash
# ==============================================================================
# run_convert_roi_cells.sh
# ------------------------------------------------------------------------------
# Converts cell segmentation Parquet/CSV files (e.g. from read_parquet.ipynb)
# into GUI-compatible Parquet/CSV files directly importable into
# multimodal_napari_gui (run_app.py).
#
# USAGE:
#   ./run_convert_roi_cells.sh
#   Or override variables via environment or CLI:
#   INPUT_FILE="/path/to/my_cells.parquet" ./run_convert_roi_cells.sh
#   ./run_convert_roi_cells.sh -i "/path/to/dir" --pattern "*_cells.parquet"
# ==============================================================================

set -e

# --- CONFIGURABLE VARIABLES (Edit these as needed) ----------------------------
INPUT_FILE="${INPUT_FILE:-/Volumes/ADATA SE880/ADATA Backup/HuBMAP/HuBMAP Xenium/Xenium_data/Xen24_3782_aki-selected/aki_cell_segmentation-roi1_cells.parquet}"
OUTPUT_PATH="${OUTPUT_PATH:-/Volumes/ADATA SE880/ADATA Backup/HuBMAP/HuBMAP Xenium/Xenium_data/Xen24_3782_aki-selected/aki_cell_segmentation-roi1_cell_rois.parquet}"

SAMPLE_ID="${SAMPLE_ID:-3782}"
ROI_NAME="${ROI_NAME:-aki-roi1-HSI.tif}"
IMAGE_PATH="${IMAGE_PATH:-}"

# Optional label filtering (leave empty to keep all)
EXCLUDE_LABELS="${EXCLUDE_LABELS:-}" # e.g. "low_quality,unassigned"
INCLUDE_LABELS="${INCLUDE_LABELS:-}"

# Coordinate orientation: set to "true" if Y coordinates need to be inverted
FLIP_Y="${FLIP_Y:-false}"

# Generate optional labeled mask TIFF and legend JSON
GENERATE_MASK="${GENERATE_MASK:-true}"

# Output format: 'parquet', 'csv', or 'both'
FORMAT="${FORMAT:-parquet}"
# ------------------------------------------------------------------------------

# Detect Python interpreter
if [ -n "$PYTHON_EXEC" ] && [ -x "$PYTHON_EXEC" ]; then
    : # User specified PYTHON_EXEC
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
PYTHON_SCRIPT="$SCRIPT_DIR/convert_roi_cells_to_gui_format.py"

echo "======================================================================"
echo "RUNNING ROI CELL CONVERTER FOR NAPARI GUI"
echo "======================================================================"
echo "Python Exec    : $PYTHON_EXEC"
echo "Input File/Dir : $INPUT_FILE"
echo "Output Path    : ${OUTPUT_PATH:-<auto_suffix>}"
echo "Sample ID      : ${SAMPLE_ID:-<auto_detect>}"
echo "ROI Name       : ${ROI_NAME:-<auto_detect>}"
echo "Exclude Labels : ${EXCLUDE_LABELS:-<none>}"
echo "Flip Y         : $FLIP_Y"
echo "Format         : $FORMAT"
echo "======================================================================"

ARGS=(
  --input "$INPUT_FILE"
  --format "$FORMAT"
)

if [ -n "$OUTPUT_PATH" ]; then
    ARGS+=(--output "$OUTPUT_PATH")
fi

if [ -n "$SAMPLE_ID" ]; then
    ARGS+=(--sample-id "$SAMPLE_ID")
fi

if [ -n "$ROI_NAME" ]; then
    ARGS+=(--image-name "$ROI_NAME")
fi

if [ -n "$IMAGE_PATH" ]; then
    ARGS+=(--image-path "$IMAGE_PATH")
fi

if [ -n "$EXCLUDE_LABELS" ]; then
    ARGS+=(--exclude-labels "$EXCLUDE_LABELS")
fi

if [ -n "$INCLUDE_LABELS" ]; then
    ARGS+=(--include-labels "$INCLUDE_LABELS")
fi

if [ "$FLIP_Y" = "true" ]; then
    ARGS+=(--flip-y)
fi

if [ "$GENERATE_MASK" = "true" ]; then
    ARGS+=(--generate-mask)
fi

"$PYTHON_EXEC" "$PYTHON_SCRIPT" "${ARGS[@]}" "$@"
