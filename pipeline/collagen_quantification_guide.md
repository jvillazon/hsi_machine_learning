# Collagen Fiber Analysis: Thickness, Structure Tensor Coherence, and Physical Scale Calibration

This guide details the mathematical foundation, image processing best practices, and code implementation for robustly quantifying **collagen thickness** and **collagen signal coherence** using structure tensors, calibrated directly to physical fiber dimensions (microns and pixels).

---

## Table of Contents
1. [Overview & Core Architecture](#1-overview--core-architecture)
2. [Collagen Thickness Quantification](#2-collagen-thickness-quantification)
3. [Structure Tensor & Collagen Signal Coherence](#3-structure-tensor--collagen-signal-coherence)
4. [Physical Scale Calibration: Correlating Gaussian Filters to Fiber Dimensions](#4-physical-scale-calibration-correlating-gaussian-filters-to-fiber-dimensions)
5. [Complete Refactored Python Implementation](#5-complete-refactored-python-implementation)
6. [Summary of Key Parameter Recommendations](#6-summary-of-key-parameter-recommendations)

---

## 1. Overview & Core Architecture

In collagen imaging (e.g., Second Harmonic Generation [SHG], Stimulated Raman Scattering [SRS], or polarized light microscopy), quantitative structural analysis centers around two primary metrics:
1. **Fiber Thickness (Diameter)**: Measured via Euclidean Distance Transform (EDT) sampled along the medial axis / centerline (skeleton).
2. **Fiber Orientation & Coherence (Anisotropy)**: Measured via the 2D Structure Tensor (Directional Gradient Tensor, equivalent to OrientationJ).

```
Raw Image (SRS / SHG)
        │
        ▼
Pre-processing & Thresholding (Otsu / Percentile)
        │
        ├───► Binary Mask (Hole-filling + Denoising)
        │             │
        │             ▼
        │     Distance Transform (EDT) + Skeletonization
        │             │
        │             ▼
        │     True Fiber Thickness (Median, Mean, IQR in µm)
        │             │
        │             ▼ [Calibrate Physical Scales: σ_gradient & σ_tensor]
        │             │
        └───► Structure Tensor Analysis
                      │
                      ├──► Analytical Gaussian Derivatives (d/dx, d/dy)
                      ├──► Outer Product & Tensor Smoothing (Integration Scale)
                      ├──► Eigenvalue Decomposition (λ_max, λ_min)
                      │
                      ▼
              Collagen Signal Coherence & Orientation Field
              - Coherence = (λ_max - λ_min) / (λ_max + λ_min)
              - Masked strictly to collagen foreground
              - Intensity-weighted alignment index
```

---

## 2. Collagen Thickness Quantification

### Understanding the Euclidean Distance Transform (EDT)
When a binary collagen mask is passed to `scipy.ndimage.distance_transform_edt`, each foreground pixel receives a value equal to the Euclidean distance (in pixels) to the nearest background pixel. Multiplying by `pixel_size_um` converts this distance to microns (µm).

### Comparing Map Types:
- **`distance_um`**: Stores the radius (distance to boundary). Center pixels store R; boundary pixels approach 0.
- **`diameter_um` (2 * distance_um)**: Multiplies distance by 2 for every foreground pixel.
  - *Pitfall*: For a cylindrical fiber of diameter D, the center line is D, but pixels towards the boundary drop linearly to 0. Averaging `diameter_um` across all foreground pixels yields approximately D / 2, heavily underestimating the true fiber thickness.
- **`skeleton_diameter_um` (Gold Standard)**: Evaluates 2 * distance_um **strictly on the skeleton / medial axis**.
  - Because the skeleton runs along the center of the fiber, every skeleton point records the true local diameter D.

### Essential Fixes for Robust Thickness:
1. **Hole Filling (`binary_fill_holes`)**:
   - In thresholded images, dark pixels inside a thick collagen bundle act as internal background. Without hole filling, a single hole drops the distance transform to 0, artificially splitting a thick 10 µm fiber bundle into tiny false fragments.
2. **Skeleton Summary Statistics**:
   - Do not merely output a TIF image; calculate quantitative distribution metrics on `skeleton_diameter[skeleton > 0]`:
     - **Median thickness (µm)**: Robust against fiber crossings and junction nodes.
     - **Mean ± Standard Deviation (µm)**.
     - **Interquartile Range (IQR, 25th - 75th percentiles)**.
3. **Cleaning the Unused Parameter**:
   - The parameter `threshold_sigma=1.0` in the existing function signature was never utilized in the function body.

---

## 3. Structure Tensor & Collagen Signal Coherence

### Structure Tensor Formulation
For a 2D image I(x, y), the structure tensor J at each pixel is formed by the smoothed outer product of image gradients:

```
J = [ Txx  Txy ]
    [ Txy  Tyy ]

where:
  Gx  = dI/dx  (smoothed at differentiation scale σ_gradient)
  Gy  = dI/dy  (smoothed at differentiation scale σ_gradient)
  Txx = Gaussian_filter(Gx * Gx, sigma = σ_tensor)
  Tyy = Gaussian_filter(Gy * Gy, sigma = σ_tensor)
  Txy = Gaussian_filter(Gx * Gy, sigma = σ_tensor)
```

### Eigenvalue Decomposition
The eigenvalues λ_max and λ_min of this 2x2 symmetric matrix represent the dominant and minor gradient energy:

```
trace = Txx + Tyy
discriminant = sqrt( max( (Txx - Tyy)^2 + 4 * Txy^2, 0 ) )

λ_max = (trace + discriminant) / 2
λ_min = (trace - discriminant) / 2
```

### Linear Coherence vs. Squared Coherence
- **OrientationJ / Standard Formulation (Linear)**:
  ```
  Coherence = (λ_max - λ_min) / (λ_max + λ_min)
  ```
  - Ranges from 0.0 (perfectly isotropic / disordered / crossing fibers) to 1.0 (perfectly parallel, unidirectional alignment).
- **Current Code Issue**:
  - The comment states `(lambda_max - lambda_min) / (lambda_max + lambda_min)`, but line 396 applied `np.square(...)`.
  - Squaring artificially penalizes intermediate alignment: a biological alignment of 0.60 drops to 0.36, and 0.40 drops to 0.16.
  - **Recommendation**: Use linear coherence by default to maintain consistency with published OrientationJ literature.

### Restricting Coherence to the Collagen Signal
In the existing script:
- `coherence` was computed globally and returned without masking background pixels.
- Gaussian smoothing of tensor components diffuses gradients outward across fiber borders into empty space, generating artifactual coherence values outside the tissue.
- **Fixes**:
  1. **Strict Signal Masking**:
     ```python
     collagen_mask = image > 0
     coherence = np.where(collagen_mask, coherence, 0.0)
     orientation = np.where(collagen_mask, orientation, np.nan)
     ```
  2. **Intensity-Weighted Coherence (Alignment Index)**:
     To summarize tissue organization across an ROI or whole image without faint background noise dragging down the score, compute the **intensity-weighted mean coherence**:
     ```
     Weighted_Coherence = Sum( Coherence * Intensity ) / Sum( Intensity )
     ```
     This gives prominent, well-defined collagen bundles proportional weight relative to weak background haze.

---

## 4. Physical Scale Calibration: Correlating Gaussian Filters to Fiber Dimensions

A structure tensor requires setting **two spatial scales**:
1. **Inner Scale (σ_gradient)**: The differentiation / noise scale.
2. **Outer Scale (σ_tensor)**: The integration / neighborhood alignment scale.

Both must be calibrated to physical fiber dimensions (µm) and image pixel size (µm/pixel).

```
|<------------------- Fiber Diameter D ------------------->|
|==================== Boundary Edge =======================|  --> σ_gradient ~ 0.25 * D
|                                                          |
|==================== Boundary Edge =======================|
|<----------------- Integration Window ------------------->|  --> σ_tensor ~ 1.5 to 3.0 * D
```

### Conversion Rule: Microns to Pixels
For any physical scale σ specified in microns:
```
sigma_pixels = sigma_microns / pixel_size_um
```

---

### Step A: Tuning the Inner Scale (σ_gradient)
- **Physical Meaning**: Regularizes differentiation and suppresses high-frequency sensor noise without blurring together adjacent parallel fibers.
- **Formula**:
  ```
  σ_gradient(µm) ≈ 0.25 × D_fiber(µm)   (or 0.5 × fiber_radius)
  σ_gradient(px) = σ_gradient(µm) / pixel_size_um
  ```
- **Nyquist / Sampling Constraint**:
  - σ_gradient(px) must **never be less than 0.8 - 1.0 pixel**. Below 0.8 px, discrete kernels cannot properly sample the continuous Gaussian, introducing grid artifacts.
- **Example**: If fiber diameter D = 2.0 µm and `pixel_size_um = 0.65`:
  - σ_gradient(µm) = 0.5 µm
  - σ_gradient(px) = 0.5 / 0.65 = 0.77 px --> clamp to approximately 0.8 - 1.0 px.

---

### Step B: Tuning the Outer Scale (σ_tensor)
- **Physical Meaning**: Defines the neighborhood over which gradient orientations are integrated to evaluate alignment.
- **Why scale matters**:
  - At any single point on a fiber edge, the gradient vector points perpendicular to that edge. A single gradient has λ_min = 0, meaning **a single point is mathematically 100% anisotropic (C = 1.0)**.
  - If σ_tensor is smaller than the fiber diameter D_fiber, the integration window only captures one side of a single fiber edge, falsely reporting 100% coherence even in completely random, disorganized meshworks!
  - For meaningful coherence, the window must span across the entire fiber thickness and include neighboring fibers:
  ```
  σ_tensor(µm) ≈ (1.5 to 3.0) × D_fiber(µm)
  σ_tensor(px) = σ_tensor(µm) / pixel_size_um
  ```
- **Example**: If fiber diameter D = 2.0 µm and `pixel_size_um = 0.65`:
  - σ_tensor(µm) = 3.0 µm
  - σ_tensor(px) = 3.0 / 0.65 ≈ 4.6 px.
  *(Note: The original setting of 1.0 / 0.65 = 1.54 px was too small because it did not even span a single fiber).*

---

### Step C: Analytical Gaussian Derivatives vs. Sobel
In the original code:
```python
smoothed = gaussian_filter(image, sigma=sigma_gradient)
Gx = sobel(smoothed, axis=1)
Gy = sobel(smoothed, axis=0)
```
- Sobel applies a fixed 3x3 discrete difference kernel with its own hardcoded smoothing. This introduces directional bias along diagonal axes and alters the effective smoothing scale.
- **Superior Approach**: Use `scipy.ndimage.gaussian_filter` with derivative orders `order=[0, 1]` for X and `order=[1, 0]` for Y:
  ```python
  Gx = gaussian_filter(image, sigma=sigma_gradient_px, order=[0, 1], mode='reflect')
  Gy = gaussian_filter(image, sigma=sigma_gradient_px, order=[1, 0], mode='reflect')
  ```
  - Directly computes d/dx (G_sigma * I) in a single step.
  - Perfectly rotationally symmetric (no 3x3 grid artifacts).
  - The smoothing scale is exactly σ_gradient, directly tied to your micron calibration.

---

## 5. Complete Refactored Python Implementation

Below are the updated, drop-in replacement functions for `compute_thickness` and `compute_structure_tensor`.

```python
import os
import numpy as np
import tifffile
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage import morphology

def compute_thickness(
    image,
    pixel_size_um=1.0,
    map_type='skeleton_diameter_um',
    file_name="thickness.tif",
    output_dir=None
):
    """
    Compute quantitative fiber thickness maps and summary statistics.

    Args:
        image (ndarray): 2D grayscale image (foreground > 0).
        pixel_size_um (float): Pixel calibration in microns per pixel.
        map_type (str):
            - 'skeleton_diameter_um': Local diameter along fiber centerlines (recommended).
            - 'diameter_um': Local diameter across all foreground pixels (2 * distance).
            - 'distance_um': Distance to nearest background (radius).
        file_name (str): Filename for saved TIF map.
        output_dir (str): Directory where TIF map will be saved.

    Returns:
        tuple: (out_map, stats_dict)
    """
    if image.ndim != 2:
        raise ValueError("Input image must be 2D grayscale.")

    # 1. Clean binary mask: fill internal holes to prevent artificial splitting
    binary_raw = image > 0
    binary = ndimage.binary_fill_holes(binary_raw)

    # 2. Euclidean distance transform converted to microns
    distance_px = ndimage.distance_transform_edt(binary)
    distance_um = distance_px.astype(np.float32) * float(pixel_size_um)

    # 3. Extract fiber centerlines (skeleton)
    skeleton = morphology.skeletonize(binary)

    if map_type == 'distance_um':
        out_map = distance_um
    elif map_type == 'diameter_um':
        out_map = 2.0 * distance_um
    elif map_type == 'skeleton_diameter_um':
        out_map = 2.0 * distance_um * skeleton
    else:
        raise ValueError(f"Unknown map_type: '{map_type}'")

    out_map = np.where(binary, out_map, 0.0).astype(np.float32)

    # 4. Extract summary statistics strictly along the skeleton
    skeleton_diameters = (2.0 * distance_um)[skeleton]
    stats = {}
    if len(skeleton_diameters) > 0:
        stats = {
            'mean_um': float(np.mean(skeleton_diameters)),
            'median_um': float(np.median(skeleton_diameters)),
            'std_um': float(np.std(skeleton_diameters)),
            'p25_um': float(np.percentile(skeleton_diameters, 25)),
            'p75_um': float(np.percentile(skeleton_diameters, 75)),
            'count_skeleton_px': int(np.count_nonzero(skeleton)),
        }
        print(f"Fiber Thickness (on skeleton):")
        print(f"  Median = {stats['median_um']:.2f} um")
        print(f"  Mean   = {stats['mean_um']:.2f} ± {stats['std_um']:.2f} um")
        print(f"  IQR    = [{stats['p25_um']:.2f}, {stats['p75_um']:.2f}] um")

    # 5. Save TIF map if directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        full_path = os.path.join(output_dir, file_name)
        tifffile.imwrite(full_path, out_map)
        print(f"Saved thickness map: {full_path}")

    return out_map, stats


def compute_structure_tensor(
    image,
    sigma_gradient=1.0,
    sigma_tensor=3.0,
    linear_coherence=True,
    output_dir=None,
    coherence_file_name=None,
    orientation_file_name=None
):
    """
    Compute structure tensor, fiber orientation, and collagen signal coherence.

    Args:
        image (ndarray): Input 2D grayscale image.
        sigma_gradient (float): Inner differentiation scale in pixels.
        sigma_tensor (float): Outer integration scale in pixels.
        linear_coherence (bool): If True, use standard OrientationJ definition (C).
                                 If False, use squared anisotropy (C^2).
        output_dir (str): Output directory for TIF files.
        coherence_file_name (str): Filename for coherence map TIF.
        orientation_file_name (str): Filename for orientation map TIF (in degrees).

    Returns:
        tuple: (image, orientation, coherence, metrics_dict)
    """
    # Downsample large images safely if needed (preserves memory)
    image_tup, ds = _auto_downsample_2d(image, max_pixels=120_000_000)
    image = image_tup[0].astype(np.float32, copy=False)

    img_max = image.max()
    if img_max > 0:
        image = image / img_max

    collagen_mask = image > 0

    print(f"Computing gradients (analytical Gaussian derivatives, sigma={sigma_gradient:.2f} px)...")
    # Step 1: Analytical Gaussian derivatives (order=[0,1] for X, [1,0] for Y)
    Gx = gaussian_filter(image, sigma=sigma_gradient, order=[0, 1], mode='reflect')
    Gy = gaussian_filter(image, sigma=sigma_gradient, order=[1, 0], mode='reflect')

    print(f"Computing structure tensor components and integration smoothing (sigma={sigma_tensor:.2f} px)...")
    # Step 2: Outer product
    Gxx = Gx * Gx
    Gyy = Gy * Gy
    Gxy = Gx * Gy

    # Step 3: Neighborhood integration smoothing
    Txx = gaussian_filter(Gxx, sigma=sigma_tensor, mode='reflect')
    Tyy = gaussian_filter(Gyy, sigma=sigma_tensor, mode='reflect')
    Txy = gaussian_filter(Gxy, sigma=sigma_tensor, mode='reflect')

    # Step 4: Eigenvalue decomposition
    trace = Txx + Tyy
    discriminant = np.sqrt(np.maximum((Txx - Tyy)**2 + 4.0 * (Txy**2), 0.0))
    lambda_max = (trace + discriminant) / 2.0
    lambda_min = (trace - discriminant) / 2.0

    # Step 5: Coherence restricted to collagen foreground
    denominator = lambda_max + lambda_min
    valid = (denominator > 1e-10) & collagen_mask

    coherence = np.zeros_like(image, dtype=np.float32)
    diff = lambda_max[valid] - lambda_min[valid]

    if linear_coherence:
        # Standard OrientationJ: (lambda_max - lambda_min) / (lambda_max + lambda_min)
        coherence[valid] = (diff / denominator[valid]).astype(np.float32)
    else:
        # Squared anisotropy
        coherence[valid] = np.square(diff / denominator[valid]).astype(np.float32)

    # Step 6: Orientation angle (perpendicular to gradient, along fiber direction)
    orientation = (0.5 * np.arctan2(-2.0 * Txy, Txx - Tyy) + np.pi / 2.0) % np.pi
    orientation = np.where(collagen_mask, orientation, np.nan).astype(np.float32)

    # Step 7: Calculate signal alignment index (intensity-weighted coherence)
    metrics = {}
    if np.any(valid):
        weights = image[valid]
        coh_valid = coherence[valid]
        metrics = {
            'mean_coherence': float(np.mean(coh_valid)),
            'median_coherence': float(np.median(coh_valid)),
            'intensity_weighted_coherence': float(np.sum(coh_valid * weights) / (np.sum(weights) + 1e-8)),
        }
        print(f"Collagen Coherence Results:")
        print(f"  Mean               = {metrics['mean_coherence']:.3f}")
        print(f"  Intensity-Weighted = {metrics['intensity_weighted_coherence']:.3f}")

    # Step 8: Save TIF maps if filenames provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        if coherence_file_name:
            full_path_coh = os.path.join(output_dir, coherence_file_name)
            tifffile.imwrite(full_path_coh, coherence)
            print(f"Saved coherence map: {full_path_coh}")

        if orientation_file_name:
            # Save orientation in degrees [0 - 180] for valid collagen pixels
            ori_deg = np.where(collagen_mask, np.degrees(orientation), 0.0).astype(np.float32)
            full_path_ori = os.path.join(output_dir, orientation_file_name)
            tifffile.imwrite(full_path_ori, ori_deg)
            print(f"Saved orientation map: {full_path_ori}")

    return image, orientation, coherence, metrics
```

---

### Connecting Thickness to Structure Tensor in `__main__`
This snippet demonstrates how the measured fiber diameter dynamically and robustly calibrates the Gaussian filter scales:

```python
if __name__ == "__main__":
    pixel_size_um = CONFIG['pixel_size_um']  # e.g., 0.65 um/px

    # 1. Load image and mask
    image = load_image(
        image_path=CONFIG['image_path'],
        threshold_method=CONFIG['threshold_method'],
        threshold_value=CONFIG['threshold_value']
    )

    # 2. Compute true fiber thickness along skeleton
    thickness_map, thickness_stats = compute_thickness(
        image,
        pixel_size_um=pixel_size_um,
        map_type='skeleton_diameter_um',
        file_name=f"{CONFIG['image_name']}_skeleton_thickness.tif",
        output_dir=output_dir
    )

    # 3. Correlate Gaussian filter scales to measured median fiber diameter
    median_diam_um = thickness_stats.get('median_um', 2.0)

    # Differentiation scale (inner): ~0.25 * fiber diameter (clamped to at least 1.0 px)
    sigma_grad_um = max(0.25 * median_diam_um, 1.0 * pixel_size_um)
    sigma_grad_px = sigma_grad_um / pixel_size_um

    # Integration scale (outer): ~1.5 to 2.0 * fiber diameter (spans across the fiber bundle)
    sigma_tensor_um = max(1.75 * median_diam_um, 2.0 * pixel_size_um)
    sigma_tensor_px = sigma_tensor_um / pixel_size_um

    print(f"\n--- Physically Calibrated Scale Selection ---")
    print(f"Median Fiber Thickness: {median_diam_um:.2f} um")
    print(f"Gradient Scale (Inner): {sigma_grad_um:.2f} um ({sigma_grad_px:.2f} px)")
    print(f"Tensor Scale   (Outer): {sigma_tensor_um:.2f} um ({sigma_tensor_px:.2f} px)")

    # 4. Compute structure tensor and coherence of collagen signal
    image, orientation, coherence, coh_metrics = compute_structure_tensor(
        image,
        sigma_gradient=sigma_grad_px,
        sigma_tensor=sigma_tensor_px,
        linear_coherence=True,
        output_dir=output_dir,
        coherence_file_name=f"{CONFIG['image_name']}_coherence.tif",
        orientation_file_name=f"{CONFIG['image_name']}_orientation.tif"
    )
```

---

## 6. Summary of Key Parameter Recommendations

| Parameter | Recommended Setting | Rationale |
| :--- | :--- | :--- |
| **`map_type`** | `'skeleton_diameter_um'` | Samples local thickness strictly along the medial axis (centerline), avoiding the edge fall-off bias of whole-mask distance averaging. |
| **Binary Mask Prep** | `ndimage.binary_fill_holes` | Eliminates single-pixel voids inside thick fibers that falsely collapse distance transforms to zero. |
| **Coherence Formula** | Linear `(λ_max - λ_min) / (λ_max + λ_min)` | Standard OrientationJ definition; avoids squaring which overly depresses moderate alignment scores. |
| **Coherence Masking** | Mask to `image > 0` | Prevents blurred gradient boundary halos in background noise from corrupting quantification. |
| **Coherence Metric** | Intensity-Weighted Mean | Gives highest biological weight to bright, intact collagen bundles while discounting dim background noise. |
| **σ_gradient (Inner Scale)** | 0.25 * D_fiber (minimum 1.0 px) | Detects fiber edges cleanly without blurring adjacent parallel fibers. |
| **σ_tensor (Outer Scale)** | 1.5 - 3.0 * D_fiber (~4 - 8 px) | Integrates both boundaries of the fiber and neighboring fibers to measure genuine bundle alignment. |
| **Gradient Operator** | `gaussian_filter(..., order=[0,1])` | Rotationally invariant scale-space derivative; eliminates discrete 3x3 Sobel directional bias. |
