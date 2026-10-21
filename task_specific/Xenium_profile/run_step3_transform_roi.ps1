<#
.SYNOPSIS
  Transforms and crops cell polygon ROIs according to a registration matrix JSON (*.i2r.json).

.DESCRIPTION
  Runs transform_and_crop_rois.py on Windows using PowerShell with geometric polygon clipping,
  un-looping safeguards, and 1D line / horizontal stretch prevention.

.EXAMPLE
  .\run_step3_transform_roi.ps1
  $env:USE_INVERSE="true"; .\run_step3_transform_roi.ps1
#>

$ErrorActionPreference = "Stop"

# --- CONFIGURABLE VARIABLES (Edit these or pass as environment variables) -----
$INPUT_FILE = if ($env:INPUT_FILE) { $env:INPUT_FILE } else { "D:\ADATA Backup\HuBMAP\HuBMAP Xenium\Xenium_data\Xen24_3990_ckd-selected\ckd-wsi-wsi_cells_gui_rois.parquet" }
$TRANSFORM_JSON = if ($env:TRANSFORM_JSON) { $env:TRANSFORM_JSON } else { "D:\ADATA Backup\HuBMAP\HuBMAP Xenium\Xenium_data\Xen24_3990_ckd-selected\Protein-ckd-roi1-HSI_transform.i2r.json" }
$OUTPUT_PREFIX = if ($env:OUTPUT_PREFIX) { $env:OUTPUT_PREFIX } else { "D:\ADATA Backup\HuBMAP\HuBMAP Xenium\Xenium_data\Xen24_3990_ckd-selected\ckd-roi1-" }

$SAMPLE_ID = if ($env:SAMPLE_ID) { $env:SAMPLE_ID } else { "3990" }
$ROI_NAME = if ($env:ROI_NAME) { $env:ROI_NAME } else { "roi1" }
$IMAGE_PATH = if ($env:IMAGE_PATH) { $env:IMAGE_PATH } else { "C:\Users\jvill\OneDrive - UC San Diego\Xenium-SRS\stitch_hSRS_data\ckd-roi1-hsi.tif" }

# Coordinate offsets ('auto' calculates offset to canvas)
$OFFSET_X = if ($env:OFFSET_X) { $env:OFFSET_X } else { "auto" }
$OFFSET_Y = if ($env:OFFSET_Y) { $env:OFFSET_Y } else { "auto" }
$REFERENCE_MASK = if ($env:REFERENCE_MASK) { $env:REFERENCE_MASK } else { "" }

# Cropping mode: 'intersect' (cells touching ROI), 'strict' (100% inside), or 'none'
$CROP_MODE = if ($env:CROP_MODE) { $env:CROP_MODE } else { "intersect" }

# Invert Y axis for target coordinate orientation
$FLIP_Y = if ($env:FLIP_Y) { $env:FLIP_Y } else { "false" }

# Apply inverse transform matrix (e.g. matrix_xy_px_inv or np.linalg.inv)
$USE_INVERSE = if ($env:USE_INVERSE) { $env:USE_INVERSE } else { "true" }

# Generate transformed ROI labeled mask TIFF
$GENERATE_MASK = if ($env:GENERATE_MASK) { $env:GENERATE_MASK } else { "true" }
$FORMAT = if ($env:FORMAT) { $env:FORMAT } else { "both" }

# Optional integrated mosaic tile splitting
$SPLIT_MOSAIC = if ($env:SPLIT_MOSAIC) { $env:SPLIT_MOSAIC } else { "false" }
$SPLIT_TILE_DIR = if ($env:SPLIT_TILE_DIR) { $env:SPLIT_TILE_DIR } else { "" }
$SPLIT_GRID_ROWS = if ($env:SPLIT_GRID_ROWS) { $env:SPLIT_GRID_ROWS } else { "" }
$SPLIT_GRID_COLS = if ($env:SPLIT_GRID_COLS) { $env:SPLIT_GRID_COLS } else { "" }
$SPLIT_OVERLAP_PERCENT = if ($env:SPLIT_OVERLAP_PERCENT) { $env:SPLIT_OVERLAP_PERCENT } else { "5.0" }
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
$PythonScript = Join-Path $ScriptDir "transform_and_crop_rois.py"

if (-not $INPUT_FILE -or -not $TRANSFORM_JSON -or -not $OUTPUT_PREFIX) {
  Write-Host "Please provide INPUT_FILE, TRANSFORM_JSON, and OUTPUT_PREFIX either in the script or via environment variables." -ForegroundColor Yellow
  Write-Host "Example:" -ForegroundColor Gray
  Write-Host "  `$env:INPUT_FILE='path\to\cells.parquet'" -ForegroundColor Gray
  Write-Host "  `$env:TRANSFORM_JSON='path\to\transform.i2r.json'" -ForegroundColor Gray
  Write-Host "  `$env:OUTPUT_PREFIX='path\to\output_roi'" -ForegroundColor Gray
  Write-Host "  .\run_step3_transform_roi.ps1" -ForegroundColor Gray
  exit 1
}

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "RUNNING STEP 3: TRANSFORM & CROP CELLS TO TARGET ROI" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Python Exec    : $PYTHON_EXEC"
Write-Host "Input File     : $INPUT_FILE"
Write-Host "Transform JSON : $TRANSFORM_JSON"
Write-Host "Output Prefix  : $OUTPUT_PREFIX"
Write-Host "Crop Mode      : $CROP_MODE"
Write-Host "Flip Y         : $FLIP_Y"
Write-Host "Use Inverse    : $USE_INVERSE"
Write-Host "Generate Mask  : $GENERATE_MASK"
Write-Host "======================================================================" -ForegroundColor Cyan

$ArgsList = @(
  "--input", $INPUT_FILE,
  "--transform-json", $TRANSFORM_JSON,
  "--output-prefix", $OUTPUT_PREFIX,
  "--sample-id", $SAMPLE_ID,
  "--crop-mode", $CROP_MODE,
  "--format", $FORMAT,
  "--offset-x", $OFFSET_X,
  "--offset-y", $OFFSET_Y
)

if ($ROI_NAME) { $ArgsList += @("--roi-name", $ROI_NAME) }
if ($IMAGE_PATH) { $ArgsList += @("--image-path", $IMAGE_PATH) }
if ($REFERENCE_MASK) { $ArgsList += @("--reference-mask", $REFERENCE_MASK) }

if ($FLIP_Y -eq "true") {
  $ArgsList += @("--flip-y")
}
else {
  $ArgsList += @("--no-flip-y")
}

if ($USE_INVERSE -eq "true") {
  $ArgsList += @("--use-inverse")
}

if ($GENERATE_MASK -eq "true") {
  $ArgsList += @("--generate-mask")
}

if ($SPLIT_MOSAIC -eq "true") {
  $ArgsList += @("--split-mosaic")
  if ($SPLIT_TILE_DIR) { $ArgsList += @("--split-tile-dir", $SPLIT_TILE_DIR) }
  if ($SPLIT_GRID_ROWS) { $ArgsList += @("--split-grid-rows", $SPLIT_GRID_ROWS) }
  if ($SPLIT_GRID_COLS) { $ArgsList += @("--split-grid-cols", $SPLIT_GRID_COLS) }
  if ($SPLIT_OVERLAP_PERCENT) { $ArgsList += @("--split-overlap-percent", $SPLIT_OVERLAP_PERCENT) }
}

& $PYTHON_EXEC $PythonScript @ArgsList

if ($LASTEXITCODE -eq 0) {
  $ConfigSuffix = if ($OUTPUT_PREFIX.EndsWith("-") -or $OUTPUT_PREFIX.EndsWith("_")) { "transform_config.json" } else { "_transform_config.json" }
  $ConfigPath = "${OUTPUT_PREFIX}${ConfigSuffix}"
  Write-Host "======================================================================" -ForegroundColor Green
  Write-Host "STEP 3 COMPLETED SUCCESSFULLY" -ForegroundColor Green
  Write-Host "Run configuration exported to: $ConfigPath" -ForegroundColor Green
  Write-Host "======================================================================" -ForegroundColor Green
}
