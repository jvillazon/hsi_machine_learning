# Hyperspectral Image Analysis & Integrated Multimodal Pipeline

Modular analysis tools for hyperspectral stimulated Raman scattering (SRS) microscopy, collagen fiber morphology, and multimodal tissue masking. Developed in the Laboratory of Optical Bioimaging and Spectroscopy under Dr. Lingyan Shi @ UCSD.

---

## Repository Structure

```text
hsi_machine_learning/
├── README.md
├── environment.yml
├── requirements.txt
├── .gitignore
└── pipeline/
    ├── __init__.py
    ├── srs_normalization.py           # Spectral standardization, background subtraction & macromolecule maps
    ├── srs_sam_matching.py            # Spectral Angle Mapper (SAM) pixel classification
    ├── references/                    # Reference spectra for SAM matching
    │   ├── averaged_spectra.csv
    │   └── water_HSI_71.csv
    ├── kmeans_hsrs.py                 # Multi-image SRS spectral standardization & MiniBatchKMeans clustering
    ├── collagen_structure_tensor.py   # Structure tensor collagen analysis (orientation, coherence, thickness)
    └── image_masking.py               # Multimodal instance masking & regional feature extraction
```

---

## Pipeline Overview

### 1. SRS Normalization (`pipeline/srs_normalization.py`)
- Standardizes hyperspectral spectra across samples.
- Removes non-resonant background.
- Computes single-macromolecule intensity maps (Protein, Saturated Lipid, Lipid, Unsaturated Lipid) and chemical ratio maps.

### 2. Spectral Angle Mapper Matching (`pipeline/srs_sam_matching.py`)
- Matches pixel spectra against known molecular libraries (`pipeline/references/averaged_spectra.csv`).
- Compensates for spectral shift against background (`pipeline/references/water_HSI_71.csv`).
- Generates similarity maps, classification masks, and top-contender prediction tables.

### 3. K-Means Clustering (`pipeline/kmeans_hsrs.py`)
- Performs multi-image spectral standardization and batch clustering with `MiniBatchKMeans`.
- Includes optimal $k$ search (silhouette score and elbow evaluation).
- Reconstructs color-mapped cluster TIFF images and centroid profile plots.

### 4. Collagen Fiber Analysis (`pipeline/collagen_structure_tensor.py`)
- Computes local fiber orientation and anisotropy/coherence using structure tensors.
- Calculates fiber thickness maps using Euclidean distance transformations.
- Exports preferential orientation vector maps and standalone publication colorbars.

### 5. Multimodal Masking (`pipeline/image_masking.py`)
- Aggregates pixel-level measurements (SAM classes, ratio maps, collagen metrics) into instance- and zone-level feature matrices.
- Supports dilated instance masks and cluster annotations for downstream quantitative analysis.

---

## Setup & Installation

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate hsi_ml
```

Or install dependencies via pip:

```bash
pip install -r requirements.txt
```
