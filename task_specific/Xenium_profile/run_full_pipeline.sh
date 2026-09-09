#!/usr/bin/env bash
# ==============================================================================
# run_full_pipeline.sh
# ------------------------------------------------------------------------------
# Executes the entire end-to-end pipeline:
#   1. Merges Parquet boundaries with CSV cell labels
#   2. Generates integer-labeled mask TIFF + label legend JSON
#   3. Transforms & crops cells into target ROI space (GUI-compatible Parquet/CSV)
#
# USAGE:
#   ./run_full_pipeline.sh
#   Or override variables inline:
#   SAMPLE_ID="xen65_3811" ./run_full_pipeline.sh
#   ./run_full_pipeline.sh --steps "1,2"
# ==============================================================================

set -e

# --- CONFIGURABLE VARIABLES (Edit these as needed) ----------------------------
BOUNDARIES="${BOUNDARIES:-/Users/jorgevillazon/Downloads/metadata/xen65_KPMP057_ABC/cell_boundaries.parquet}"
LABELS="${LABELS:-/Users/jorgevillazon/Downloads/metadata/xen65_KPMP057_ABC/xen65_KPMP057_C_cell_labels_KPMP3_SRS_mincounts20_allgenes_taccofix.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-/Users/jorgevillazon/Downloads/metadata/xen65_KPMP057_ABC/pipeline_output}"

# Optional transformation JSON (leave empty to skip step 3)
TRANSFORM_JSON="${TRANSFORM_JSON:-}"

# Optional metadata mappings for mask generation (leave empty to auto-enumerate)
METADATA_DICT="${METADATA_DICT:-}"
METADATA_LIST="${METADATA_LIST:-}"

# Identifiers
SAMPLE_ID="${SAMPLE_ID:-xen65_KPMP057}"
ROI_NAME="${ROI_NAME:-ROI1}"
IMAGE_PATH="${IMAGE_PATH:-}"

# Steps to run: "1,2,3" (or "1,2" / "1" / "3")
STEPS="${STEPS:-1,2,3}"

# Output format: 'parquet', 'csv', or 'both'
FORMAT="${FORMAT:-both}"
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
PYTHON_SCRIPT="$SCRIPT_DIR/xenium_pipeline.py"

echo "======================================================================"
echo "RUNNING FULL XENIUM -> GUI PIPELINE"
echo "======================================================================"
echo "Python Exec    : $PYTHON_EXEC"
echo "Boundaries     : $BOUNDARIES"
echo "Labels         : $LABELS"
echo "Output Dir     : $OUTPUT_DIR"
echo "Sample ID      : $SAMPLE_ID"
echo "ROI Name       : $ROI_NAME"
echo "Steps to Run   : $STEPS"
echo "======================================================================"

ARGS=(
  --boundaries "$BOUNDARIES"
  --labels "$LABELS"
  --output-dir "$OUTPUT_DIR"
  --sample-id "$SAMPLE_ID"
  --roi-name "$ROI_NAME"
  --steps "$STEPS"
  --format "$FORMAT"
)

if [ -n "$TRANSFORM_JSON" ]; then
    ARGS+=(--transform-json "$TRANSFORM_JSON")
fi

if [ -n "$METADATA_DICT" ]; then
    ARGS+=(--metadata-dict "$METADATA_DICT")
fi

if [ -n "$METADATA_LIST" ]; then
    ARGS+=(--metadata-list "$METADATA_LIST")
fi

if [ -n "$IMAGE_PATH" ]; then
    ARGS+=(--image-path "$IMAGE_PATH")
fi

"$PYTHON_EXEC" "$PYTHON_SCRIPT" "${ARGS[@]}" "$@"
