# SRS SAM Matching

This folder contains a standalone script for matching hyperspectral image pixels against reference spectra using Spectral Angle Mapping (SAM).

## Process

- Loads reference spectra from a CSV file
- Loads a background spectrum from a CSV file
- Estimates a global spectral shift
- Computes SAM-based similarity for each pixel
- Writes similarity maps, classification maps, and prediction CSV/Parquet outputs

## Requirements

Install the dependencies from the companion requirements file:

```bash
pip install -r requirements.txt
```

## Configuration

Edit the configuration block at the top of the script and update:

- `image_dir` for the input HSI TIFF files
- `CONFIG` for granular/optional changes

The script expects a folder of TIFF images and will create output subfolders for each image.

## Run

```bash
python srs_sam_matching.py
```

## Notes

- The script uses large TIFF-backed memmaps for memory efficiency.
- Output files are written per input image into an output folder named from the image basename.
- If you want to exclude specific molecules or group molecules into classes, adjust the `exclude_molecules` and `mol_groups` settings in the config.
