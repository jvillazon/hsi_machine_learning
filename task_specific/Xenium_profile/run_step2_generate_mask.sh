#!/usr/bin/env bash
# ==============================================================================
# run_step2_generate_mask.sh
# ------------------------------------------------------------------------------
# Generates integer-labeled mask TIFF from cell boundary polygons and metadata.
#
# USAGE:
#   ./run_step2_generate_mask.sh
#   Or override variables via environment or pass through flags:
#   MASK_TYPE="all" ./run_step2_generate_mask.sh
#   ./run_step2_generate_mask.sh --output "/custom/path/mask.tif"
# ==============================================================================

set -e

# --- CONFIGURABLE VARIABLES (Edit these as needed) ----------------------------
INPUT_FILE="${INPUT_FILE:-/Users/jorgevillazon/Downloads/metadata/xen65_KPMP057_ABC/xen65_KPMP057_C_merged_vertices.parquet}"
OUTPUT_MASK="${OUTPUT_MASK:-/Users/jorgevillazon/Downloads/metadata/xen65_KPMP057_ABC/xen65_KPMP057_C_binary-mask.tif}"

# Label metadata mapping (leave empty to auto-generate from unique labels)
METADATA_DICT="${METADATA_DICT:-}" # Path to JSON file or inline JSON string: '{"POD":1,"PT":2}'
METADATA_LIST="${METADATA_LIST:-ATL,B,C-IC-A,C-PC,C-TAL,C/M-TAL,CNT,DC,
       DCT,DTL1,DTL2,DTL3,EC-AA,EC-AVR,EC-DVR,
       EC-EA,EC-GC,EC-LYM,EC-PCV,EC-PTC,EC-V,ERY,
       FIB,IC-B,IMCD,M-FIB,M-IC-A,M-PC,M-TAL,MAST,
       MC,MD,MON,MYOF,N,NK,PEC,PL,POD,PT-S1,
       PT-S2,PT-S3,REN,SC/NEU,T,VSMC,VSMC/P,aCNT,
       aDCT,aDTL,aEC-GC,aPT,aTAL,angEC-PTC,cycEC,
       cycMAC,cycPT,cycT,cycTAL,dATL,dC-IC-A,dC-TAL,
       dCNT,dDCT,dDTL,dEC-AVR,dEC-GC,dEC-PTC,dFIB,
       dIMCD,dM-FIB,dM-IC-A,dM-PC,dM-TAL,dPOD,dPT,
       dVSMC,frPT,frTAL,infEC-AVR,infEC-PTC,infFIB,
       moMAC,moMAC-INF,pvFIB,resMAC,tPC-IC}"

# Mask type: 'binary' (all cells 255, background 0), 'labeled' (2D int), 'instance', 'stack' (3D), 'split' (per-label), 'all'
MASK_TYPE="${MASK_TYPE:-binary}"

# Pixel size: 'auto' (calculated as min_x/min_y from processing_mask.ipynb) or float in microns
PIXEL_SIZE="${PIXEL_SIZE:-auto}"
PADDING="${PADDING:-10}"          # Margin around cells
MIN_SIZE="${MIN_SIZE:-0}"          # Minimum connected component area (for split masks)

# Optional canvas dimensions (leave empty for automatic calculation)
CANVAS_WIDTH="${CANVAS_WIDTH:-}"
CANVAS_HEIGHT="${CANVAS_HEIGHT:-}"
# ------------------------------------------------------------------------------

# Auto-detect input file if user specified .parquet but only .csv exists (or vice versa)
if [ ! -f "$INPUT_FILE" ] && [ -f "${INPUT_FILE%.parquet}.csv" ]; then
    INPUT_FILE="${INPUT_FILE%.parquet}.csv"
elif [ ! -f "$INPUT_FILE" ] && [ -f "${INPUT_FILE%.csv}.parquet" ]; then
    INPUT_FILE="${INPUT_FILE%.csv}.parquet"
fi

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
PYTHON_SCRIPT="$SCRIPT_DIR/generate_labeled_mask.py"

echo "======================================================================"
echo "RUNNING STEP 2: GENERATE LABELED MASK TIFF"
echo "======================================================================"
echo "Python Exec    : $PYTHON_EXEC"
echo "Input File     : $INPUT_FILE"
echo "Output Mask    : $OUTPUT_MASK"
echo "Mask Type      : $MASK_TYPE"
echo "Pixel Size     : $PIXEL_SIZE um"
echo "======================================================================"

ARGS=(
  --input "$INPUT_FILE"
  --output "$OUTPUT_MASK"
  --mask-type "$MASK_TYPE"
  --pad "$PADDING"
  --pixel-size "$PIXEL_SIZE"
  --min-size "$MIN_SIZE"
)

if [ -n "$METADATA_DICT" ]; then
    ARGS+=(--metadata-dict "$METADATA_DICT")
fi

if [ -n "$METADATA_LIST" ]; then
    ARGS+=(--metadata-list "$METADATA_LIST")
fi

if [ -n "$CANVAS_WIDTH" ]; then
    ARGS+=(--width "$CANVAS_WIDTH")
fi

if [ -n "$CANVAS_HEIGHT" ]; then
    ARGS+=(--height "$CANVAS_HEIGHT")
fi

"$PYTHON_EXEC" "$PYTHON_SCRIPT" "${ARGS[@]}" "$@"
