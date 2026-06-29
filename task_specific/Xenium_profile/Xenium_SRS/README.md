# Xenium to SRS Coordinate Transformation Pipeline

Jupyter notebook and utility python file designed to align 2D Xenium cell boundary coordinates with Stimulated Raman Scattering images (or pseudo-histology H&E images).

## Detailed Pipeline Steps

### Step 1: Create Cell Segmentation Mask and Grid CSV
Raw Xenium cell boundaries are loaded from a `.parquet` file in micrometer coordinates.
1. The resolution step size (`min_diff`) is calculated for the x and y axes.
2. Coordinates are converted to pixel space on a grid aligned at `(0, 0)`.
3. A multi-label cell segmentation mask (`.tif`) is generated where each cell polygon is filled with its mapped metadata category index. This mask can be utilized to align with the SRS images using image2image.
4. The pixel-aligned coordinates are saved as a `.csv` file.

### Step 2: Image-to-Image Registration
The generated cell segmentation mask (or the H&E image) is registered to the target SRS image.
1. The segmentation mask is defined as the **Moving Image**.
2. The SRS image is defined as the **Fixed Image**.
3. Corresponding landmark fiducials are set at aligned anatomical coordinates.
4. An affine registration transformation is computed and exported as a JSON file (e.g., `*_transform.i2r.json`).

### Step 3: Register and Filter Coordinates
Using the registration JSON file:
1. The registration transformation matrix is loaded.
2. The pixel-coordinate cell boundary vertices from Step 1 are mapped into the SRS coordinate space using the affine matrix.
3. Transformed coordinates are filtered or cropped to fit within the SRS image dimensions (`img_bounds`).
5. The final aligned cell vertices are exported as a registered `.parquet` file.

## Installation & Setup

Ensure all dependencies are installed before executing the notebook using requirements.txt file.


## Configuration & Usage

Inside `transform_Xenium_coords.ipynb`:

1. Define the cell categories in `metadata_list` in the order matching your classification indices.
2. Set paths to your raw input files in Step 1:
   ```python
   file_path = r"path/to/cell_boundaries.parquet"
   metadata_path = r"path/to/_metadata.csv"
   ```
3. Execute Step 1 cells to generate the TIFF mask and CSV.
4. Perform registration using your preferred image-alignment software to generate the transform JSON.
5. Set the JSON and CSV paths in Step 3:
   ```python
   transform_json = r"path/to/*_transform.i2r.json"
   cell_seg_csv = r"path/to/*_cell_segmentation.csv"
   ```
6. Execute Step 3 cells to generate the registered `*_img_segmented_cells.parquet` file containing final mapped coordinates (`x_roi`, `y_roi`).
