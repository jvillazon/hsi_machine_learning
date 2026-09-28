<#
.SYNOPSIS
  Converts cell segmentation Parquet/CSV files into standard Napari GUI-compatible ROI format.

.DESCRIPTION
  Runs convert_roi_cells_to_gui_format.py with native PowerShell support on Windows and Anaconda.

.EXAMPLE
  .\run_convert_roi_cells.ps1
  
.EXAMPLE
  $env:INPUT_FILE="D:\ADATA Backup\HuBMAP\HuBMAP Xenium\Xenium_data\Xen24_3782_aki-selected\aki-wsi-wsi_cells.parquet"; .\run_convert_roi_cells.ps1
  
.EXAMPLE
  .\run_convert_roi_cells.ps1 -InputFile "D:\path\to\cells.parquet" -SampleId "3782" -RoiName "aki-wsi-wsi.tif"
#>

[CmdletBinding()]
param(
    [string]$InputFile,
    [string]$OutputPath,
    [string]$SampleId,
    [string]$RoiName,
    [string]$ImagePath,
    [string]$ExcludeLabels,
    [string]$IncludeLabels,
    [string]$FlipY,
    [string]$GenerateMask,
    [string]$Format
)

$ErrorActionPreference = "Stop"

# --- CONFIGURABLE VARIABLES (Edit these or pass via parameters/environment) -----
$INPUT_FILE = if ($InputFile) { $InputFile }
              elseif ($env:INPUT_FILE) { $env:INPUT_FILE }
              else { "D:\ADATA Backup\HuBMAP\HuBMAP Xenium\Xenium_data\Xen24_3782_aki-selected\aki-wsi-wsi_cells.parquet" }

$OUTPUT_PATH = if ($OutputPath) { $OutputPath }
               elseif ($env:OUTPUT_PATH) { $env:OUTPUT_PATH }
               else { "D:\ADATA Backup\HuBMAP\HuBMAP Xenium\Xenium_data\Xen24_3782_aki-selected\aki-wsi-wsi_cell_rois.parquet" }

$SAMPLE_ID = if ($SampleId) { $SampleId }
             elseif ($env:SAMPLE_ID) { $env:SAMPLE_ID }
             else { "3782" }

$ROI_NAME = if ($RoiName) { $RoiName }
            elseif ($env:ROI_NAME) { $env:ROI_NAME }
            else { "aki-wsi-wsi.tif" }

$IMAGE_PATH = if ($ImagePath) { $ImagePath }
              elseif ($env:IMAGE_PATH) { $env:IMAGE_PATH }
              else { "" }

$EXCLUDE_LABELS = if ($ExcludeLabels) { $ExcludeLabels }
                  elseif ($env:EXCLUDE_LABELS) { $env:EXCLUDE_LABELS }
                  else { "" }

$INCLUDE_LABELS = if ($IncludeLabels) { $IncludeLabels }
                  elseif ($env:INCLUDE_LABELS) { $env:INCLUDE_LABELS }
                  else { "" }

$FLIP_Y = if ($FlipY) { $FlipY }
          elseif ($env:FLIP_Y) { $env:FLIP_Y }
          else { "false" }

$GENERATE_MASK = if ($GenerateMask) { $GenerateMask }
                 elseif ($env:GENERATE_MASK) { $env:GENERATE_MASK }
                 else { "true" }

$FORMAT = if ($Format) { $Format }
          elseif ($env:FORMAT) { $env:FORMAT }
          else { "parquet" }
# ------------------------------------------------------------------------------

# Detect Python interpreter (Anaconda / Miniconda / Active Conda Environment)
$PYTHON_EXEC = ""
if ($env:PYTHON_EXEC -and (Test-Path $env:PYTHON_EXEC)) {
    $PYTHON_EXEC = $env:PYTHON_EXEC
}
elseif ($env:CONDA_DEFAULT_ENV -and $env:CONDA_DEFAULT_ENV -ne "base" -and $env:CONDA_PREFIX -and (Test-Path "$env:CONDA_PREFIX\python.exe")) {
    $PYTHON_EXEC = "$env:CONDA_PREFIX\python.exe"
}
elseif (Test-Path "C:\Users\$env:USERNAME\miniconda3\envs\hsi_machine_learning\python.exe") {
    $PYTHON_EXEC = "C:\Users\$env:USERNAME\miniconda3\envs\hsi_machine_learning\python.exe"
}
elseif (Test-Path "C:\Users\$env:USERNAME\miniconda3\envs\gui_analysis\python.exe") {
    $PYTHON_EXEC = "C:\Users\$env:USERNAME\miniconda3\envs\gui_analysis\python.exe"
}
elseif (Test-Path "C:\Users\$env:USERNAME\anaconda3\envs\hsi_machine_learning\python.exe") {
    $PYTHON_EXEC = "C:\Users\$env:USERNAME\anaconda3\envs\hsi_machine_learning\python.exe"
}
elseif (Test-Path "C:\ProgramData\miniconda3\envs\hsi_machine_learning\python.exe") {
    $PYTHON_EXEC = "C:\ProgramData\miniconda3\envs\hsi_machine_learning\python.exe"
}
elseif (Test-Path "C:\ProgramData\anaconda3\envs\hsi_machine_learning\python.exe") {
    $PYTHON_EXEC = "C:\ProgramData\anaconda3\envs\hsi_machine_learning\python.exe"
}
elseif ($env:CONDA_PREFIX -and (Test-Path "$env:CONDA_PREFIX\python.exe")) {
    $PYTHON_EXEC = "$env:CONDA_PREFIX\python.exe"
}
else {
    $PYTHON_EXEC = "python"
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonScript = Join-Path $ScriptDir "convert_roi_cells_to_gui_format.py"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "RUNNING ROI CELL CONVERTER FOR NAPARI GUI (Windows PowerShell)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Python Exec    : $PYTHON_EXEC"
Write-Host "Input File/Dir : $INPUT_FILE"
Write-Host "Output Path    : $(if ($OUTPUT_PATH) { $OUTPUT_PATH } else { '<auto_suffix>' })"
Write-Host "Sample ID      : $(if ($SAMPLE_ID) { $SAMPLE_ID } else { '<auto_detect>' })"
Write-Host "ROI Name       : $(if ($ROI_NAME) { $ROI_NAME } else { '<auto_detect>' })"
Write-Host "Exclude Labels : $(if ($EXCLUDE_LABELS) { $EXCLUDE_LABELS } else { '<none>' })"
Write-Host "Flip Y         : $FLIP_Y"
Write-Host "Generate Mask  : $GENERATE_MASK"
Write-Host "Format         : $FORMAT"
Write-Host "======================================================================" -ForegroundColor Cyan

$ArgsList = @(
    "--input", $INPUT_FILE,
    "--format", $FORMAT
)

if ($OUTPUT_PATH) { $ArgsList += @("--output", $OUTPUT_PATH) }
if ($SAMPLE_ID) { $ArgsList += @("--sample-id", $SAMPLE_ID) }
if ($ROI_NAME) { $ArgsList += @("--image-name", $ROI_NAME) }
if ($IMAGE_PATH) { $ArgsList += @("--image-path", $IMAGE_PATH) }
if ($EXCLUDE_LABELS) { $ArgsList += @("--exclude-labels", $EXCLUDE_LABELS) }
if ($INCLUDE_LABELS) { $ArgsList += @("--include-labels", $INCLUDE_LABELS) }
if ($FLIP_Y -eq "true") { $ArgsList += @("--flip-y") }
if ($GENERATE_MASK -eq "true") { $ArgsList += @("--generate-mask") }

# Pass any extra trailing arguments to the python script
if ($args.Count -gt 0) {
    $ArgsList += $args
}

& $PYTHON_EXEC $PythonScript @ArgsList
