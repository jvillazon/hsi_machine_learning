<#
.SYNOPSIS
  Splits transformed/coregistered ROI cell tables into smaller mosaic subsections.

.DESCRIPTION
  Runs split_roi_mosaic_subsections.py with native PowerShell support on Windows.

.EXAMPLE
  .\run_step4_split_mosaic_rois.ps1
  $env:TILE_DIR="E:\Data\Jorge\01_HumanKidney_HuBMAP\Xenium_datasets\Ref\roi2"; .\run_step4_split_mosaic_rois.ps1
#>

$ErrorActionPreference = "Stop"

# --- CONFIGURABLE VARIABLES (Edit these or pass as environment variables) -----
$INPUT_FILE = if ($env:INPUT_FILE) { $env:INPUT_FILE } else { "C:\Users\jvill\OneDrive - UC San Diego\Xenium-SRS\Xenium_data\Xen24_KPMP_038_ref-selected\ref-roi2-HSI_cells_gui_rois.parquet" }
$OUTPUT_DIR = if ($env:OUTPUT_DIR) { $env:OUTPUT_DIR } else { "C:\Users\jvill\OneDrive - UC San Diego\Xenium-SRS\Xenium_data\Xen24_KPMP_038_ref-selected\ref-roi2-HSI_all_mosaics" }
$OUTPUT_PREFIX = if ($env:OUTPUT_PREFIX) { $env:OUTPUT_PREFIX } else { "" }

# Option A: Automatic discovery from tile directory
$TILE_DIR = if ($env:TILE_DIR) { $env:TILE_DIR } else { "C:\Users\jvill\OneDrive - UC San Diego\Xenium-SRS\total_hSRS\KPMP_038-roi2" }

# Option B: Manual grid parameters (used if TILE_DIR is not provided or to override)
$GRID_ROWS = if ($env:GRID_ROWS) { $env:GRID_ROWS } else { "" }
$GRID_COLS = if ($env:GRID_COLS) { $env:GRID_COLS } else { "" }
$TILE_HEIGHT = if ($env:TILE_HEIGHT) { $env:TILE_HEIGHT } else { "" }
$TILE_WIDTH = if ($env:TILE_WIDTH) { $env:TILE_WIDTH } else { "" }
$OVERLAP_PERCENT = if ($env:OVERLAP_PERCENT) { $env:OVERLAP_PERCENT } else { "5.0" }
$ORDER = if ($env:ORDER) { $env:ORDER } else { "snake" }

# Coordinate mode: 'local' (0 to tile_w/h), 'global' (canvas coords), or 'both'
$COORD_MODE = if ($env:COORD_MODE) { $env:COORD_MODE } else { "both" }

# Boundary inclusion mode: 'intersect', 'centroid', or 'strict'
$CROP_MODE = if ($env:CROP_MODE) { $env:CROP_MODE } else { "intersect" }

# Clip polygon vertices strictly to tile boundaries [0, W] x [0, H]
$CLIP_VERTICES = if ($env:CLIP_VERTICES) { $env:CLIP_VERTICES } else { "true" }

$SAMPLE_ID = if ($env:SAMPLE_ID) { $env:SAMPLE_ID } else { "KPMP_038" }
$FORMAT = if ($env:FORMAT) { $env:FORMAT } else { "both" }
$GENERATE_MASKS = if ($env:GENERATE_MASKS) { $env:GENERATE_MASKS } else { "false" }
$PIXEL_SIZE = if ($env:PIXEL_SIZE) { $env:PIXEL_SIZE } else { "0.4971847" }

# Cohesive vs individual tile parquet/csv output
$SAVE_INDIVIDUAL = if ($env:SAVE_INDIVIDUAL) { $env:SAVE_INDIVIDUAL } else { "false" }
$COHESIVE_NAME = if ($env:COHESIVE_NAME) { $env:COHESIVE_NAME } else { "" }
$STITCHED_IMAGE = if ($env:STITCHED_IMAGE) { $env:STITCHED_IMAGE } else { "D:\ADATA Backup\HuBMAP\HuBMAP Xenium\Xenium HSI\Normalized_Images_xenium-HSI\ref-roi2-HSI\Protein-ref-roi2-HSI.tif" }
$TARGET_CANVAS_SHAPE = if ($env:TARGET_CANVAS_SHAPE) { $env:TARGET_CANVAS_SHAPE } else { "" }
# ------------------------------------------------------------------------------

# Detect Python interpreter
$PYTHON_EXEC = "python"
if ($env:PYTHON_EXEC -and (Test-Path $env:PYTHON_EXEC)) {
  $PYTHON_EXEC = $env:PYTHON_EXEC
}
elseif (Test-Path "C:\Users\$env:USERNAME\miniconda3\envs\gui_analysis_312\python.exe") {
  $PYTHON_EXEC = "C:\Users\$env:USERNAME\miniconda3\envs\gui_analysis_312\python.exe"
}
elseif (Test-Path "C:\Users\$env:USERNAME\miniconda3\envs\gui_analysis\python.exe") {
  $PYTHON_EXEC = "C:\Users\$env:USERNAME\miniconda3\envs\gui_analysis\python.exe"
}
elseif (Test-Path "C:\Users\$env:USERNAME\miniconda3\envs\hsi_machine_learning\python.exe") {
  $PYTHON_EXEC = "C:\Users\$env:USERNAME\miniconda3\envs\hsi_machine_learning\python.exe"
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonScript = Join-Path $ScriptDir "split_roi_mosaic_subsections.py"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "RUNNING STEP 4: GENERATE COHESIVE MOSAIC SUBSECTIONS" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Python Exec     : $PYTHON_EXEC"
Write-Host "Input ROI Table : $INPUT_FILE"
Write-Host "Output Directory: $OUTPUT_DIR"
Write-Host "Tile Directory  : $TILE_DIR"
Write-Host "Coordinate Mode : $COORD_MODE"
Write-Host "Crop/Overlap    : $CROP_MODE"
Write-Host "Save Individual : $SAVE_INDIVIDUAL"
Write-Host "Generate Masks  : $GENERATE_MASKS"
Write-Host "Format          : $FORMAT"
Write-Host "======================================================================" -ForegroundColor Cyan

$ArgsList = @(
  "--input", $INPUT_FILE,
  "--output-dir", $OUTPUT_DIR,
  "--overlap-percent", $OVERLAP_PERCENT,
  "--order", $ORDER,
  "--coord-mode", $COORD_MODE,
  "--crop-mode", $CROP_MODE,
  "--sample-id", $SAMPLE_ID,
  "--format", $FORMAT,
  "--pixel-size", $PIXEL_SIZE
)

if ($OUTPUT_PREFIX) { $ArgsList += @("--output-prefix", $OUTPUT_PREFIX) }
if ($TILE_DIR) { $ArgsList += @("--tile-dir", $TILE_DIR) }
if ($GRID_ROWS) { $ArgsList += @("--grid-rows", $GRID_ROWS) }
if ($GRID_COLS) { $ArgsList += @("--grid-cols", $GRID_COLS) }
if ($TILE_HEIGHT) { $ArgsList += @("--tile-height", $TILE_HEIGHT) }
if ($TILE_WIDTH) { $ArgsList += @("--tile-width", $TILE_WIDTH) }
if ($CLIP_VERTICES -eq "true") { $ArgsList += @("--clip-vertices") }
if ($GENERATE_MASKS -eq "true") { $ArgsList += @("--generate-masks") }
if ($SAVE_INDIVIDUAL -eq "true") { $ArgsList += @("--save-individual") }
if ($COHESIVE_NAME) { $ArgsList += @("--cohesive-name", $COHESIVE_NAME) }
if ($STITCHED_IMAGE) { $ArgsList += @("--stitched-image", $STITCHED_IMAGE) }
if ($TARGET_CANVAS_SHAPE) { $ArgsList += @("--target-canvas-shape", $TARGET_CANVAS_SHAPE) }

& $PYTHON_EXEC $PythonScript @ArgsList
