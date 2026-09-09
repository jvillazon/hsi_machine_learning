#!/usr/bin/env bash
# ==============================================================================
# run_step1_merge.sh
# ------------------------------------------------------------------------------
# Merges cell boundary coordinates (Parquet/CSV) with cell labels (CSV/Parquet).
# Outputs both per-vertex coordinates and aggregated GUI-compatible polygon ROIs.
#
# USAGE:
#   ./run_step1_merge.sh
#   Or override any variable via environment variable or CLI arguments:
#   SAMPLE_ID="xen65_3673" ./run_step1_merge.sh
#   ./run_step1_merge.sh --boundaries "/path/to/custom_boundaries.parquet"
# ==============================================================================

set -e

# --- CONFIGURABLE VARIABLES (Edit these as needed) ----------------------------
BOUNDARIES="${BOUNDARIES:-/Users/jorgevillazon/Downloads/metadata/xen65_3673/cell_boundaries.parquet}"
LABELS="${LABELS:-/Users/jorgevillazon/Downloads/metadata/xen65_3673/xen65_3673_cell_labels_KPMP3_SRS_mincounts20_allgenes_taccofix.csv}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/Users/jorgevillazon/Downloads/metadata/xen65_3673/xen65_3673}"

BOUNDARY_CELL_COL="${BOUNDARY_CELL_COL:-cell_id}"
BOUNDARY_X_COL="${BOUNDARY_X_COL:-vertex_x}"
BOUNDARY_Y_COL="${BOUNDARY_Y_COL:-vertex_y}"

LABEL_CELL_COL="${LABEL_CELL_COL:-cell_id}"
LABEL_GROUP_COL="${LABEL_GROUP_COL:-group}"

UNLABELED_HANDLING="${UNLABELED_HANDLING:-keep}" # 'filter' or 'keep'
DEFAULT_UNLABELED="${DEFAULT_UNLABELED:-Unassigned}"

SAMPLE_ID="${SAMPLE_ID:-3673}"
IMAGE_NAME="${IMAGE_NAME:-}"
IMAGE_PATH="${IMAGE_PATH:-}"

FORMAT="${FORMAT:-parquet}" # 'parquet', 'csv', or 'both'
MODE="${MODE:-both}"     # 'vertices', 'regions', or 'both'
# ------------------------------------------------------------------------------

# Detect Python interpreter
if [ -n "$PYTHON_EXEC" ] && [ -x "$PYTHON_EXEC" ]; then
    : # User specified PYTHON_EXEC
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
PYTHON_SCRIPT="$SCRIPT_DIR/merge_boundaries_and_labels.py"

echo "======================================================================"
echo "RUNNING STEP 1: MERGE BOUNDARIES & LABELS"
echo "======================================================================"
echo "Python Exec    : $PYTHON_EXEC"
echo "Boundaries     : $BOUNDARIES"
echo "Labels         : $LABELS"
echo "Output Prefix  : $OUTPUT_PREFIX"
echo "Sample ID      : $SAMPLE_ID"
echo "======================================================================"

"$PYTHON_EXEC" "$PYTHON_SCRIPT" \
  --boundaries "$BOUNDARIES" \
  --labels "$LABELS" \
  --output-prefix "$OUTPUT_PREFIX" \
  --boundary-cell-col "$BOUNDARY_CELL_COL" \
  --boundary-x-col "$BOUNDARY_X_COL" \
  --boundary-y-col "$BOUNDARY_Y_COL" \
  --label-cell-col "$LABEL_CELL_COL" \
  --label-group-col "$LABEL_GROUP_COL" \
  --unlabeled-handling "$UNLABELED_HANDLING" \
  --default-unlabeled "$DEFAULT_UNLABELED" \
  --sample-id "$SAMPLE_ID" \
  --image-name "$IMAGE_NAME" \
  --image-path "$IMAGE_PATH" \
  --mode "$MODE" \
  --format "$FORMAT" \
  "$@"
