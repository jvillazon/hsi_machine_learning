## Written by Gemini 3.7

# Napari Interactive ROI Segmentation Tool

An interactive, high-performance segmentation and polygon annotation GUI built on top of **Napari** and **Qt**. Designed for multi-channel fluorescence, hyperspectral, and microscopy TIFF imaging datasets (supporting standard TIFF, ImageJ TIFF stacks, and OME-TIFF), batch region queueing, and exports to CSV & instance-labeled binary mask TIFFs.

---

## 📦 Installation

Ensure you have Python 3.9+ and install the dependencies:

```bash
pip install -r requirements.txt
```

*(Optional) If using conda:*
```bash
conda create -n <environment_name> python>=3.9
conda activate <environment_name>
pip install -r requirements.txt
```

---

## 🖥️ Usage

Run the GUI with:

```bash
python napari_segmentation.py
```

### Supported Image Formats
- **TIFF / ImageJ / OME-TIFF** (`.tif`, `.tiff`, `.ome.tif`, `.ome.tiff`)
*(Note: For proprietary microscopy formats such as Olympus `.oir`, Zeiss `.czi`, or Leica `.lif`, convert them to TIFF / ImageJ TIFF first using Fiji/ImageJ or Bio-Formats exporter.)*

### Step-by-Step Workflow

1. **Load Images**:
   - Click **"Load Base Image"** to select your primary TIFF image. If multi-channel/3D, use the **Base Channel/Z** slider to select the slice.
   - (Optional) Click **"Load Overlay Image"** to load co-registered fluorescence or reference TIFF images and scrub through channels.
2. **Set Image Metadata**:
   - Verify or edit the **Sample ID** (auto-populated from filename) and **ROI #**.
   - Image pixel resolutions and units are automatically extracted from ImageJ tags (`Pixel_Size_X`, `Pixel_Size_Y`, `ResolutionUnit`).
3. **Assign Label & Draw ROIs**:
   - Enter your target marker/structure name under **Assign Label** (e.g., `DAPI`, `Glomerulus`, `Early PT`, `Late PT`, `Lipid`).
   - In the Napari toolbar, select the **Add polygons** tool.
   - Click on the image to place vertices. **Double-click** or press **Esc** to finish closing the polygon.
4. **Queue to Batch**:
   - Click **"Add Label Regions to Batch (Deletes current polygons)"**.
   - The polygons are saved to the queue and cleared from the canvas, ready for you to draw regions for the next label or load a new image.
5. **Export**:
   - Click **"Export CSV (All Batched Regions)"** to save all geometries and metadata.
   - Click **"Export Batch Masks TIF(s)"** to generate instance-labeled mask TIFF files into a target directory.

---

## 🎨 Napari Drawing Controls & Shortcuts

| Action | Shortcut / Interaction |
| :--- | :--- |
| **Add Polygon** | Click on canvas to add points |
| **Finish Polygon** | Double-click or press <kbd>Esc</kbd> |
| **Select / Edit Shapes** | Press <kbd>2</kbd> or click the pointer tool |
| **Move Polygon Vertex** | Select vertex and drag |
| **Delete Selected Polygon** | Press <kbd>Backspace</kbd> / <kbd>Delete</kbd> |
| **Pan & Zoom** | Press <kbd>1</kbd> or hold <kbd>Space</kbd> + drag / scroll wheel |
