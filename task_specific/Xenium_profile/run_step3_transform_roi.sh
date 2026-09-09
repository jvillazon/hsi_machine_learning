#!/usr/bin/env bash
# ==============================================================================
# run_step3_transform_roi.sh
# ------------------------------------------------------------------------------
# Applies affine transformation from transform JSON (*.i2r.json), crops to target
# ROI bounds, flips Y axis for coordinate orientation, and outputs GUI-ready
# Parquet/CSV files directly loadable by multimodal_napari_gui (run_app.py).
#
# USAGE:
#   ./run_step3_transform_roi.sh
#   Or override variables via environment or CLI flags:
#   TRANSFORM_JSON="/path/to/my_transform.i2r.json" ./run_step3_transform_roi.sh
#   ./run_step3_transform_roi.sh --crop-mode strict
# ==============================================================================

set -e

# --- CONFIGURABLE VARIABLES (Edit these as needed) ----------------------------
INPUT_FILE="${INPUT_FILE:-/Users/jorgevillazon/Downloads/metadata/xen65_3811/xen65_3811_merged_vertices.parquet}"
TRANSFORM_JSON="${TRANSFORM_JSON:-/Users/jorgevillazon/Downloads/D/Protein-D-DM-ROI4-HSI-781_transform.i2r.json}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/Users/jorgevillazon/Downloads/D/xen65_3811_ROI4}"

SAMPLE_ID="${SAMPLE_ID:-3811}"
ROI_NAME="${ROI_NAME:-D-DM-ROI4-HSI-781.3-806.5nm-500mW-450IR-HV500-20us-zoom2_00<01-80>.tif}"
IMAGE_PATH="${IMAGE_PATH:-""}"

# Optional coordinate offsets (default 'auto' detects alignment to fixed canvas)
OFFSET_X="${OFFSET_X:-auto}"
OFFSET_Y="${OFFSET_Y:-auto}"
REFERENCE_MASK="${REFERENCE_MASK:-}"

# Cropping mode: 'intersect' (cells touching ROI, clipped to bounds), 'strict' (cells 100% inside)
CROP_MODE="${CROP_MODE:-intersect}"

# Coordinate orientation: set to "true" to invert Y axis (roi_height - y)
FLIP_Y="${FLIP_Y:-false}"

# Use inverse transformation matrix (e.g. matrix_xy_px_inv or np.linalg.inv)
USE_INVERSE="${USE_INVERSE:-true}"

# Output format: 'parquet', 'csv', or 'both'
FORMAT="${FORMAT:-both}"

# Generate transformed ROI labeled mask TIFF directly on target canvas
GENERATE_MASK="${GENERATE_MASK:-true}"
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
elif [ -f "/opt/miniconda3/envs/gui_analysis/bin/python" ]; then
    PYTHON_EXEC="/opt/miniconda3/envs/gui_analysis/bin/python"
else
    PYTHON_EXEC="python3"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/transform_and_crop_rois.py"

echo "======================================================================"
echo "RUNNING STEP 3: TRANSFORM & CROP CELLS TO TARGET ROI"
echo "======================================================================"
echo "Python Exec    : $PYTHON_EXEC"
echo "Input File     : $INPUT_FILE"
echo "Transform JSON : $TRANSFORM_JSON"
echo "Output Prefix  : $OUTPUT_PREFIX"
echo "ROI Name       : $ROI_NAME"
echo "Crop Mode      : $CROP_MODE"
echo "Flip Y         : $FLIP_Y"
echo "Use Inverse    : $USE_INVERSE"
echo "======================================================================"

ARGS=(
  --input "$INPUT_FILE"
  --transform-json "$TRANSFORM_JSON"
  --output-prefix "$OUTPUT_PREFIX"
  --sample-id "$SAMPLE_ID"
  --roi-name "$ROI_NAME"
  --crop-mode "$CROP_MODE"
  --format "$FORMAT"
  --offset-x "$OFFSET_X"
  --offset-y "$OFFSET_Y"
)

if [ -n "$REFERENCE_MASK" ]; then
    ARGS+=(--reference-mask "$REFERENCE_MASK")
fi

if [ -n "$IMAGE_PATH" ]; then
    ARGS+=(--image-path "$IMAGE_PATH")
fi

if [ "$FLIP_Y" = "true" ]; then
    ARGS+=(--flip-y)
else
    ARGS+=(--no-flip-y)
fi

if [ "$USE_INVERSE" = "true" ]; then
    ARGS+=(--use-inverse)
fi

if [ "$GENERATE_MASK" = "true" ]; then
    ARGS+=(--generate-mask)
fi

"$PYTHON_EXEC" "$PYTHON_SCRIPT" "${ARGS[@]}" "$@"
