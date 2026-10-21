"""
Structure Tensor-based Collagen Fiber Analysis

This script uses the structure tensor method (similar to OrientationJ in Fiji) to analyze collagen fibers.
Much faster and more robust than Gabor filters while preserving directional continuity.

Key Functions:
1. compute_structure_tensor(image): Computes orientation and coherence using structure tensor
2. compute_thickness(image): Computes thickness of collagen fibers (distance to background)
3. visualize_preferential_orientation_vector_map(image, orientation, coherence): Visualizes the preferential orientation of collagen fibers


"""
import json
import os
import sys

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import tifffile
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
from scipy import ndimage
from scipy.ndimage import gaussian_filter, sobel
from scipy.interpolate import griddata
from skimage import filters, io, morphology
from skimage.color import rgb2gray
from skimage.measure import block_reduce
from tqdm import tqdm
import time

import tkinter as tk
from tkinter import filedialog

image_path = r"C:\Users\jvill\OneDrive - UC San Diego\integrated_pipeline\WSI_data\SRS\collagen\S21-K21_198_3-collagen.ome.tiff"
image_name = os.path.basename(image_path).split('.')[0]
output_dir = os.path.join(os.path.dirname(image_path), f"{image_name}_collagen_10-20-26-output")

# Choose thresholding method based on your ROI:
# - 'otsu': Best for small, clean ROIs with clear background/signal distinction
# - 'li': Better for larger ROIs with more complex intensity distributions
# - 'yen': Good for unimodal distributions
# - 'percentile': Manual control - e.g., threshold_value=50 removes bottom 50% of intensities
# - 'triangle' : Good for unimodal distributions
# - 'none': No thresholding (analyze raw image)
# image = load_image(image_path, threshold_method='otsu')

# For larger ROIs with poor Otsu performance, try one of:
# image = load_image(image_path, threshold_method='li')
# image = load_image(image_path, threshold_method='percentile', threshold_value=75)

CONFIG = {
    'image_path': image_path,
    'output_dir': output_dir,
    'image_name': image_name,
    'pixel_size_um': 0.65,
    'map_type': 'propagated_diameter_um',  # 'propagated_diameter_um', 'skeleton_diameter_um', 'diameter_um', 'distance_um'
    'tophat_radius_um':None,              
    'auto_scale_from_thickness': True,    # Automatically calibrate sigma_gradient and sigma_tensor to fiber width
    'sigma_gradient': 0.5,                # Fallback inner scale in microns (if auto_scale is False)
    'sigma_tensor': 2.0,                  # Fallback outer scale in microns (if auto_scale is False)
    'threshold_method': 'percentile',
    'threshold_value': 10,
    'plot_images': True,
    'export_config_only': False,           # If True, exports config JSON and exits immediately without running pipeline
}


# Global plotting style configuration for consistency
PLOT_STYLE = {
    'width': 12,
    'cbar_width': 2.0,  # Standalone colorbar width in inches
    'title_size': 18,
    'label_size': 14,
    'tick_size': 12,
    'cbar_fraction': 0.046,
    'cbar_pad': 0.04
}

def save_standalone_colorbar(mappable, filename, label, ticks=None, tick_labels=None):
    """
    Save a standalone colorbar as a separate PNG file.
    
    Args:
        mappable: The ScalarMappable or image to create the colorbar from
        filename (str): Output filename
        label (str): Label for the colorbar
        ticks (list): Optional custom ticks
        tick_labels (list): Optional custom tick labels
    """
    fig_cbar = plt.figure(figsize=(PLOT_STYLE['cbar_width'], 8))
    ax_cbar = fig_cbar.add_axes([0.3, 0.1, 0.2, 0.8])
    
    cbar = fig_cbar.colorbar(mappable, cax=ax_cbar)
    cbar.set_label(label, fontsize=PLOT_STYLE['label_size'], fontweight='bold', labelpad=10)
    cbar.ax.tick_params(labelsize=PLOT_STYLE['tick_size'])
    
    if ticks is not None:
        cbar.set_ticks(ticks)
    if tick_labels is not None:
        cbar.ax.set_yticklabels(tick_labels)
        
    plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig_cbar)
    print(f"Saved colorbar: {filename}")

def get_figsize(shape, base_width=None):
    """
    Calculate figure size that matches the aspect ratio of the input data.
    
    Args:
        shape (tuple): Shape of the image/data (height, width)
        base_width (float): Desired width of the figure in inches (none = use PLOT_STYLE)
        
    Returns:
        tuple: (width, height) in inches
    """
    if base_width is None:
        base_width = PLOT_STYLE['width']
        
    h, w = shape[:2]
    aspect = w / h
    return (base_width, base_width / aspect)


def _auto_downsample_2d(*arrays, max_pixels=80_000_000):
    """Downsample one or more 2D arrays by integer stride to stay under max_pixels.

    Returns
    -------
    tuple
        (downsampled_arrays, stride)
    """
    if not arrays:
        return (), 1

    h, w = arrays[0].shape[:2]
    total_pixels = h * w
    if total_pixels <= max_pixels:
        return arrays, 1

    stride = int(np.ceil(np.sqrt(total_pixels / float(max_pixels))))
    downs = tuple(arr[::stride, ::stride] for arr in arrays)
    return downs, stride


def load_image(image_path, threshold_method='otsu', threshold_value=None):
    """
    Load and convert image to grayscale.
    Apply thresholding to remove background noise before structure tensor.
    
    Args:
        image_path (str): Path to image file
        threshold_method (str): Thresholding method:
            - 'otsu': Otsu's method (good for bimodal distributions)
            - 'li': Li's method (more robust for complex histograms)
            - 'yen': Yen's method (better for unimodal distributions)
            - 'percentile': Manual percentile threshold (use threshold_value parameter)
            - 'none': No thresholding, returns raw image
        threshold_value (float): For 'percentile' method, the percentile (0-100) to use as threshold
        
    Returns:
        image (ndarray): Grayscale image with background removed
    """
    try:
        image = tifffile.imread(image_path)
        print(f"Loaded image: {image.shape}, dtype: {image.dtype}")
    except Exception as e:
        print(f"Error reading image: {e}")
        return None
    
    if image.ndim == 3:
        print("Converting multi-channel to grayscale")
        image = rgb2gray(image)
    
    # Use float16 for large images to reduce memory pressure.
    if image.dtype != np.float32:
        image = image.astype(np.float32)
        print(f"Converted to float16: range [{image.min():.3f}, {image.max():.3f}]")
    
    # Apply thresholding based on selected method
    if threshold_method.lower() == 'none':
        print("No thresholding applied")
        print(f"  Background pixels (0): {np.sum(image == 0)}")
        print(f"  Signal pixels (>0): {np.sum(image > 0)}")
        return image
    
    elif threshold_method.lower() == 'otsu':
        
        threshold = filters.threshold_otsu(image)
        image[image <= threshold] = 0
        print(f"Applied Otsu threshold: {threshold:.3f}")

    elif threshold_method.lower() == 'triangle':
        nonzero_pixels = image[image > 10]
        threshold = filters.threshold_triangle(nonzero_pixels)
        image[image <= threshold] = 0
        print(f"Applied Triangle threshold: {threshold:.3f}")
    
    elif threshold_method.lower() == 'li':
        otsu_threshold = filters.threshold_otsu(image)
        threshold = filters.threshold_li(image, initial_guess=otsu_threshold)
        image[image <= threshold] = 0
        print(f"Applied Li's threshold: {threshold:.3f}")
    
    elif threshold_method.lower() == 'yen':
        threshold = filters.threshold_yen(image)
        image[image <= threshold] = 0
        print(f"Applied Yen's threshold: {threshold:.3f}")
    
    elif threshold_method.lower() == 'percentile':
        if threshold_value is None:
            print("ERROR: percentile method requires threshold_value parameter (0-100)")
            return None
        threshold = np.percentile(image[image>1e-4], threshold_value)
        image[image <= threshold] = 0
        print(f"Applied percentile {threshold_value}% threshold: {threshold:.3f}")
    
    else:
        print(f"ERROR: Unknown threshold_method '{threshold_method}'")
        return None
    
    # Set values below a certain threshold to 0
    image = image > 1e-4
    selem = morphology.disk(2)
    image = ndimage.binary_opening(image, selem)

    print(f"  Background pixels: {np.sum(image == 0)}")
    print(f"  Signal pixels: {np.sum(image > 0)}")
    print(f"Threshold BEFORE structure tensor prevents boundary artifacts")

    return image



def compute_thickness(
    image,
    pixel_size_um=1.0,
    map_type='propagated_diameter_um',
    tophat_radius_um=4.0,
    file_name="thickness.tif",
    output_dir=None,
    return_stats=False
):
    """
    Compute quantitative distance/thickness maps from collagen mask.

    Args:
        image (ndarray): Input grayscale image of collagen fibers.
        pixel_size_um (float): Pixel size in microns per pixel.
        map_type (str):
            - 'propagated_diameter_um': Local diameter propagated across full fibril footprint (recommended)
            - 'skeleton_diameter_um': Local diameter evaluated strictly along skeleton / medial axis
            - 'diameter_um': Local full thickness (2 * distance) across all foreground pixels
            - 'distance_um': Per-pixel distance to nearest background in microns (radius)
        tophat_radius_um (float): Scale in um for morphological White Top-Hat filtering to separate
                                  individual fibrils in dense sheets. Set None to disable.
        file_name (str): Filename to save quantitative map.
        output_dir (str): Output directory path.
        return_stats (bool): If True, returns (out_map, stats_dict). If False, returns out_map.

    Returns:
        ndarray or tuple: Quantitative map in microns (and stats dict if return_stats=True).
    """

    if len(image.shape) > 2:
        raise ValueError("Input image must be 2D grayscale")

    print(f"Computing quantitative map: {map_type} (pixel_size_um={pixel_size_um})...")

    # Tophat to seperate fibrils
    if tophat_radius_um is not None and tophat_radius_um > 0 and np.any(image > 0):
        radius_px = max(1, int(np.round((tophat_radius_um / 2.0) / pixel_size_um)))
        filter_size = 2 * radius_px + 1
        print(f"  Applying White Top-Hat filter to unmerge dense fibrils (scale={tophat_radius_um:.2f} um / filter_size={filter_size} px)...")
        
        fibril_enhanced = ndimage.white_tophat(image, size=filter_size)
        fg_vals = fibril_enhanced[image > 0]
        
        if len(fg_vals) > 0 and fg_vals.max() > 0:
            thresh = filters.threshold_otsu(fg_vals)
            binary_raw = (fibril_enhanced > thresh) & (image > 0)
            print(f"  Top-Hat Otsu threshold: {thresh:.3f} (fibril pixels: {np.count_nonzero(binary_raw)})")
        else:
            binary_raw = image > 0
        del fibril_enhanced
    else:
        binary_raw = image > 0

    # Clean binary mask: fill internal voids to prevent artificial thickness collapse
    # binary = ndimage.binary_fill_holes(binary_raw)

    # 2. Euclidean distance transform converted to microns (memory-safe float32)
    if HAS_CV2:
        # cv2.distanceTransform computes exact Euclidean distance directly in float32
        # Avoids Scipy's intermediate (2, H, W) float64 arrays (5.53+ GiB) that cause ArrayMemoryError on 300+ MP WSIs
        binary_uint8 = binary_raw.astype(np.uint8, copy=False)
        distance_px = cv2.distanceTransform(binary_uint8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        del binary_uint8
        distance_um = distance_px  # Already float32
        if pixel_size_um != 1.0:
            distance_um *= np.float32(pixel_size_um)
    else:
        distance_px = ndimage.distance_transform_edt(binary_raw)
        distance_um = distance_px.astype(np.float32, copy=False) * float(pixel_size_um)
        del distance_px 

    skeleton = morphology.skeletonize(binary_raw)
    skeleton_diam = (2.0 * distance_um * skeleton).astype(np.float32, copy=False)

    # 3. Compute robust summary statistics on skeleton centerlines
    skeleton_diameters = skeleton_diam[skeleton > 0]
    stats = {}
    if len(skeleton_diameters) > 0:
        stats = {
            'median_um': float(np.median(skeleton_diameters)),
            'mean_um': float(np.mean(skeleton_diameters)),
            'std_um': float(np.std(skeleton_diameters)),
            'p25_um': float(np.percentile(skeleton_diameters, 25)),
            'p75_um': float(np.percentile(skeleton_diameters, 75)),
            'count_skeleton_px': int(np.count_nonzero(skeleton)),
            'count_fiber_px': int(np.count_nonzero(binary_raw)),
        }
        print(f"  Fibril Thickness Stats (on skeleton centerlines):")
        print(f"    Median = {stats['median_um']:.2f} um")
        print(f"    Mean   = {stats['mean_um']:.2f} +/- {stats['std_um']:.2f} um")
        print(f"    IQR    = [{stats['p25_um']:.2f}, {stats['p75_um']:.2f}] um")

    # 4. Construct thickness map according to map_type
    if map_type == 'distance_um':
        # Radius (distance to nearest background)
        out_map = distance_um
    elif map_type == 'diameter_um':
        # Local diameter (2 * distance)
        out_map = 2.0 * distance_um
    elif map_type == 'skeleton_diameter_um':
        # True diameter strictly on skeleton
        out_map = skeleton_diam
    elif map_type == 'propagated_diameter_um':
        # Nearest-centerline Voronoi projection across full fibril
        if np.any(skeleton):
            print("  Propagating centerline diameters across full fibril (Voronoi projection)...")
            if HAS_CV2:
                # Fast, memory-efficient Voronoi using cv2.distanceTransformWithLabels (half the RAM of Scipy)
                cv_in = (~skeleton).astype(np.uint8, copy=False)
                _, labels = cv2.distanceTransformWithLabels(
                    cv_in, cv2.DIST_L2, cv2.DIST_MASK_5, labelType=cv2.DIST_LABEL_PIXEL
                )
                del cv_in

                skeleton_vals = skeleton_diam[skeleton]
                lookup = np.empty(len(skeleton_vals) + 1, dtype=np.float32)
                lookup[0] = 0.0
                lookup[1:] = skeleton_vals
                del skeleton_vals

                # Reuse distance_um buffer as out_map to avoid extra 1.4+ GB allocation on 300+ MP images
                h = distance_um.shape[0]
                chunk_size = 2500
                for r in range(0, h, chunk_size):
                    r_end = min(r + chunk_size, h)
                    distance_um[r:r_end, :] = lookup[labels[r:r_end, :]]
                del labels, lookup
                out_map = distance_um
            else:
                nearest_idx = ndimage.distance_transform_edt(
                    ~skeleton, 
                    return_distances=False, 
                    return_indices=True
                )
                out_map = np.empty_like(skeleton_diam)
                h = skeleton_diam.shape[0]
                chunk_size = 2500
                for r in range(0, h, chunk_size):
                    r_end = min(r + chunk_size, h)
                    out_map[r:r_end, :] = skeleton_diam[nearest_idx[0, r:r_end, :], nearest_idx[1, r:r_end, :]]
                del nearest_idx
        else:
            out_map = np.zeros_like(distance_um)
    else:
        raise ValueError(
            "map_type must be one of ['propagated_diameter_um', 'skeleton_diameter_um', 'diameter_um', 'distance_um']"
        )

    # Keep background strictly at zero
    out_map = np.where(binary_raw, out_map, 0.0).astype(np.float32, copy=False)

    print(
        f"  Map stats: min={out_map.min():.3f} um, max={out_map.max():.3f} um, "
        f"nonzero={np.count_nonzero(out_map)}"
    )
    
    # Save as float32 TIF
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    else:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        output_dir = filedialog.askdirectory(title="Select Folder")
        root.destroy()

    if output_dir:
        full_path = os.path.join(output_dir, f"{file_name}")
        tifffile.imwrite(full_path, out_map)
        print(f"Saved quantitative map: {map_type} to {full_path}")
    else:
        print("Save cancelled: No output directory selected.")

    if return_stats:
        return out_map, stats
    return out_map



def compute_structure_tensor(
    image,
    sigma_gradient=1.0,
    sigma_tensor=2.0,
    linear_coherence=True,
    output_dir=None,
    coherence_file_name=None,
    orientation_file_name=None,
    return_metrics=False
):
    """
    Compute fiber orientation and anisotropy using structure tensor method.
    This is the mathematical foundation of OrientationJ.
    
    Args:
        image (ndarray): Input grayscale image of collagen fibers
        sigma_gradient (float): Gaussian differentiation scale in pixels
        sigma_tensor (float): Gaussian integration scale in pixels
        linear_coherence (bool): If True, use standard OrientationJ definition (C).
                                 If False, use squared anisotropy (C^2).
        output_dir (str): Folder to save output maps
        coherence_file_name (str): Filename for saved coherence TIF
        orientation_file_name (str): Filename for saved orientation TIF
        return_metrics (bool): If True, returns (image, orientation, coherence, metrics).
                               If False, returns (image, orientation, coherence).
        
    Returns:
        tuple: (image, orientation, coherence) [and metrics dict if return_metrics=True]
    """

    image_tup, ds = _auto_downsample_2d(image, max_pixels=120000000)
    image = image_tup[0]
    # Normalize to 0-1 in float32 to keep memory bounded on large images.
    image = image.astype(np.float32, copy=False)
    if image.max() > 1.0:
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    collagen_mask = image > 0
    print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")
    
    # Step 1: Compute gradients using analytical Gaussian derivatives
    # Exact continuous scale-space derivatives: order=[0, 1] for d/dx, [1, 0] for d/dy
    print(f"Computing gradients (analytical Gaussian derivatives, sigma={sigma_gradient:.2f} px)...")
    if sigma_gradient > 0:
        Gx = gaussian_filter(image, sigma=sigma_gradient, order=[0, 1], mode='reflect')
        Gy = gaussian_filter(image, sigma=sigma_gradient, order=[1, 0], mode='reflect')
    else:
        Gx = sobel(image, axis=1)
        Gy = sobel(image, axis=0)
    
    print(f"  Gx range: [{Gx.min():.4f}, {Gx.max():.4f}]")
    print(f"  Gy range: [{Gy.min():.4f}, {Gy.max():.4f}]")
    
    # Step 2: Construct structure tensor elements (outer product)
    print(f"\nComputing structure tensor components...")
    Gxx = Gx * Gx
    Gyy = Gy * Gy
    Gxy = Gx * Gy
    
    # Step 3: Smooth tensor components across integration window
    print(f"\nSmoothing structure tensor components (integration sigma={sigma_tensor:.2f} px)...")
    if sigma_tensor > 0:
        Txx = gaussian_filter(Gxx, sigma=sigma_tensor, mode='reflect')
        Tyy = gaussian_filter(Gyy, sigma=sigma_tensor, mode='reflect')
        Txy = gaussian_filter(Gxy, sigma=sigma_tensor, mode='reflect')
        print(f"  Applied Gaussian integration smoothing with sigma={sigma_tensor:.2f} px")
    else:
        Txx = Gxx
        Tyy = Gyy
        Txy = Gxy
        print(f"  No smoothing applied (sigma=0) - using raw tensor components")
    
    # Step 4: Eigenvalue decomposition at each pixel
    print("\nComputing eigenvalue decomposition at each pixel...")
    trace = Txx + Tyy  # trace = sum of eigenvalues
    det = Txx * Tyy - Txy * Txy  # determinant
    discriminant = np.sqrt(np.maximum((Txx - Tyy)**2 + 4.0 * (Txy**2), 0.0))
    
    lambda_max = (trace + discriminant) / 2.0  # larger eigenvalue
    lambda_min = (trace - discriminant) / 2.0  # smaller eigenvalue

    lambda_min = np.nan_to_num(lambda_min, nan=0.0)
    lambda_max = np.nan_to_num(lambda_max, nan=0.0)
    
    # Step 5: Coherence restricted strictly to collagen signal
    # Coherence = (lambda_max - lambda_min) / (lambda_max + lambda_min)
    # Ranges from 0 (isotropic / crossing / noise) to 1 (perfectly aligned)
    denominator = lambda_max + lambda_min
    valid = (denominator > 1e-10) & collagen_mask
    diff = lambda_max[valid] - lambda_min[valid]

    coherence = np.zeros_like(denominator, dtype=np.float32)
    if linear_coherence:
        # Standard linear OrientationJ definition
        coherence[valid] = (diff / denominator[valid]).astype(np.float32)
    else:
        # Squared anisotropy
        coherence[valid] = np.square(diff / denominator[valid]).astype(np.float32)
    
    # Explicitly ensure background pixels are 0.0
    coherence = np.where(collagen_mask, coherence, 0.0).astype(np.float32)
    
    # Step 6: Orientation from eigenvector of largest eigenvalue
    # Rotate by 90 degrees = pi/2 to obtain fiber direction (perpendicular to gradient)
    # -2*Txy flips to a Right-Handed Cartesian system (y-up relative to horizontal)
    orientation = (0.5 * np.arctan2(-2.0 * Txy, Txx - Tyy) + np.pi / 2.0).astype(np.float32)
    orientation = orientation % np.pi  # Normalize to [0, pi]
    orientation = np.where(collagen_mask, orientation, np.nan).astype(np.float32)

    # Step 7: Calculate signal alignment index (intensity-weighted coherence)
    metrics = {}
    if np.any(valid):
        signal_weights = image[valid]
        coh_valid = coherence[valid]
        metrics = {
            'mean_coherence': float(np.mean(coh_valid)),
            'median_coherence': float(np.median(coh_valid)),
            'intensity_weighted_coherence': float(np.sum(coh_valid * signal_weights) / (np.sum(signal_weights) + 1e-8)),
        }
        print(f"\nCollagen Signal Coherence Metrics:")
        print(f"  Mean Coherence               = {metrics['mean_coherence']:.3f}")
        print(f"  Median Coherence             = {metrics['median_coherence']:.3f}")
        print(f"  Intensity-Weighted Coherence = {metrics['intensity_weighted_coherence']:.3f}")

    # Free large intermediates early on large images
    del Gx, Gy, Gxx, Gyy, Gxy, Txx, Tyy, Txy, trace, det, discriminant, lambda_max, lambda_min, denominator, valid
    
    print(f"\nResults:")
    print(f"  Coherence range: [{coherence.min():.3f}, {coherence.max():.3f}]")
    print(f"  Orientation range: [{np.nanmin(orientation):.3f}, {np.nanmax(orientation):.3f}] rad")
    
    # Save TIF files if file names are provided
    if coherence_file_name is not None or orientation_file_name is not None:
        if output_dir is None:
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            output_dir = filedialog.askdirectory(title="Select Folder to Save Structure Tensor Maps")
            root.destroy()
            
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            if coherence_file_name is not None:
                full_path_coh = os.path.join(output_dir, coherence_file_name)
                tifffile.imwrite(full_path_coh, coherence)
                print(f"Saved coherence heatmap TIF to: {full_path_coh}")
                
            if orientation_file_name is not None:
                # Convert valid orientation from radians (0-pi) to degrees (0-180)
                orientation_degrees = np.where(collagen_mask, np.degrees(orientation), 0.0)
                orientation_degrees = np.clip(orientation_degrees, 0.0, 180.0).astype(np.float32)
                
                full_path_ori = os.path.join(output_dir, orientation_file_name)
                tifffile.imwrite(full_path_ori, orientation_degrees)
                print(f"Saved orientation TIF to: {full_path_ori}")
        else:
            print("Save cancelled: No output directory selected.")
            
    if return_metrics:
        return image, orientation, coherence, metrics
    return image, orientation, coherence

def visualize_orientation_heatmap(image, orientation, coherence=None, coherence_threshold=0.1, 
                                  filename_png="orientation_heatmap_viz.png", 
                                  figsize=None, sigma_smooth=0.5):
    """
    Create and save a high-contrast orientation heatmap using HSV colormap.
    
    Args:
        image (ndarray): Original image for masking background
        orientation (ndarray): Orientation map in radians (0 to pi)
        coherence (ndarray): Coherence map for optional additional filtering
        coherence_threshold (float): Minimum coherence to display (0-1)
        filename_png (str): Output filename
        figsize (tuple): Figure size
        sigma_smooth (float): Gaussian smoothing sigma
        
    Returns:
        fig: Matplotlib figure object
    """
    
    # Downsample for visualization if image is very large.
    if coherence is not None:
        (image_ds, orientation_ds, coherence_ds), ds = _auto_downsample_2d(image, orientation, coherence, max_pixels=120000000)
    else:
        (image_ds, orientation_ds), ds = _auto_downsample_2d(image, orientation, max_pixels=120000000)
        coherence_ds = None

    if ds > 1:
        print(f"Orientation heatmap downsampled by {ds}x for memory safety")

    # Define overall mask: intensity signal AND well-defined orientation (coherence)
    mask = image_ds > 0
    if coherence_ds is not None:
        mask = mask & (coherence_ds >= coherence_threshold)
    
    # Mask background and low-anisotropy regions without creating duplicate where-arrays.
    orientation_masked = orientation_ds.astype(np.float32, copy=True)
    orientation_masked[~mask] = np.nan
    
    # Smooth before visualization
    if sigma_smooth > 0:
        # Note: smoothed nan is tricky, handle carefully
        mask = np.isnan(orientation_masked)
        orient_filled = np.nan_to_num(orientation_masked)
        smooth_orient = gaussian_filter(orient_filled, sigma=sigma_smooth)
        # Re-mask
        smooth_orient[mask] = np.nan
    else:
        smooth_orient = orientation_masked
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Display with HSV colormap (circular)
    # Map [0, pi] to [0, 1] for colormap
    im = ax.imshow(smooth_orient, cmap='hsv', vmin=0, vmax=np.pi, aspect='auto')
    
    # ax.set_title('Fiber Orientation Heatmap', fontsize=PLOT_STYLE['title_size'], fontweight='bold')
    ax.axis('off') 
    
    # Save standalone colorbar
    cbar_filename = filename_png.replace('.png', '_colorbar.png')
    save_standalone_colorbar(im, cbar_filename, 'Orientation (radians)', 
                            ticks=[0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
                            tick_labels=['0', 'π/4', 'π/2', '3π/4', 'π'])

    fig.patch.set_facecolor('#f8f8f8')
    plt.tight_layout()
    plt.savefig(filename_png, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename_png}")
    plt.close(fig)
    
    return fig

def visualize_coherence_heatmap(coherence, filename_png="coherence_heatmap_viz.png", figsize=None, sigma_smooth=0, image=None):
    """
    Create and save coherence heatmap visualization for quick inspection.
    
    Args:
        coherence (ndarray): Coherence/anisotropy map (values 0-1)
        filename_png (str): Output filename for PNG visualization
        figsize (tuple): Figure size
        sigma_smooth (float): Gaussian smoothing sigma for visualization (default 0.5 for fine detail)
        image (ndarray): Original image for masking background (optional)
    Returns:
        fig: Matplotlib figure object
    """
    
    # Downsample for visualization if image is very large.
    if image is not None:
        (coherence_ds, image_ds), ds = _auto_downsample_2d(coherence, image, max_pixels=120000000)
    else:
        (coherence_ds,), ds = _auto_downsample_2d(coherence, max_pixels=120000000)
        image_ds = None

    if ds > 1:
        print(f"Coherence visualization downsampled by {ds}x for memory safety")

    # Mask background in-place on a float16 working copy.
    coherence_work = coherence_ds.astype(np.float32, copy=True)
    if image_ds is not None:
        coherence_work[image_ds <= 0] = 0
    
    if figsize is None:
        figsize = get_figsize(coherence.shape, base_width=10)
        
    fig, ax = plt.subplots(figsize=figsize)
    # Apply Gaussian smoothing for visualization (sigma=0 for no smoothing, increase for more blur)
    coherence_smooth = gaussian_filter(coherence_work, sigma=sigma_smooth)
    # Display with hot colormap
    im = ax.imshow(coherence_smooth, cmap='hot', vmin=0, vmax=1, aspect='auto')
    # ax.set_title('Fiber Coherence Map', fontsize=PLOT_STYLE['title_size'], fontweight='bold')
    ax.axis('off')
    
    # Save standalone colorbar
    cbar_filename = filename_png.replace('.png', '_colorbar.png')
    save_standalone_colorbar(im, cbar_filename, 'Coherence [0-1]')

    fig.patch.set_facecolor('#f8f8f8')
    plt.tight_layout()
    plt.savefig(filename_png, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename_png}")
    plt.close(fig) 

    return fig

def visualize_preferential_orientation_vector_map(orientation, coherence, image=None, 
                                                  bin_size=32, figsize=None, 
                                                  overlay_image=True, coherence_threshold=0.1,
                                                  vector_length_scale=1.0, filename_png="vector_map.png",
                                                  max_display_dim=3500):
    """
    Visualize preferential fiber orientation as a vector map with spatial binning.
    Bins nearby pixels and computes their median orientation using circular statistics.
    
    This provides a cleaner view of the dominant orientation field compared to 
    plotting every individual pixel's orientation.
    
    Args:
        orientation (ndarray): Orientation map in radians (0 to pi)
        coherence (ndarray): Coherence/anisotropy map (0 to 1) - used for coloring
        image (ndarray): Original image (for masking and optional overlay)
        bin_size (int): Spatial bin size in pixels (default 32)
        figsize (tuple): Figure size
        overlay_image (bool): If True, show original image in background
        coherence_threshold (float): Minimum coherence to display vectors (0-1)
        vector_length_scale (float): Scale factor for vector sizes
        filename (str): Output filename for PNG
        
    Returns:
        fig: Matplotlib figure object
    """
    
    h, w = orientation.shape
    
    # Create bin grid
    y_bins = np.arange(0, h, bin_size)
    x_bins = np.arange(0, w, bin_size)
    
    bin_orientations = []
    bin_coherences = []
    bin_positions_y = []
    bin_positions_x = []
    
    print(f"Creating vector field with bin_size={bin_size}px...")
    
    # For each bin, compute the median orientation using circular statistics
    for i, y_start in enumerate(y_bins[:-1]):
        y_end = min(y_start + bin_size, h)
        
        for j, x_start in enumerate(x_bins[:-1]):
            x_end = min(x_start + bin_size, w)
            
            # Extract orientations and coherences in this bin
            bin_orient = orientation[y_start:y_end, x_start:x_end]
            bin_coher = coherence[y_start:y_end, x_start:x_end]
            
            # Mask by original image if provided
            if image is not None:
                bin_img = image[y_start:y_end, x_start:x_end]
                valid_mask = bin_img > 0
            else:
                valid_mask = np.ones_like(bin_orient, dtype=bool)
            
            # Only process bins with valid (non-background) pixels
            if valid_mask.sum() > 0:
                valid_orient = bin_orient[valid_mask]
                valid_coher = bin_coher[valid_mask]
                
                # Compute circular mean of orientation using resultant vector method
                # This is more robust for circular data than standard mean
                # Convert angle to unit vector: x = cos(2*θ), y = sin(2*θ)
                # (multiply by 2 because orientation is π-periodic, not 2π)
                cos_vals = np.cos(2 * valid_orient)
                sin_vals = np.sin(2 * valid_orient)
                
                mean_cos = cos_vals.mean()
                mean_sin = sin_vals.mean()
                
                # Compute mean orientation (and divide by 2 to undo the doubling)
                mean_orient = 0.5 * np.arctan2(mean_sin, mean_cos)
                mean_orient = mean_orient % np.pi  # Normalize to [0, pi]
                
                # Mean coherence in bin
                mean_coher = valid_coher.mean()
                
                # Only add if coherence exceeds threshold
                if mean_coher >= coherence_threshold:
                    bin_orientations.append(mean_orient)
                    bin_coherences.append(mean_coher)
                    bin_positions_y.append((y_start + y_end) / 2.0)
                    bin_positions_x.append((x_start + x_end) / 2.0)
    
    bin_orientations = np.array(bin_orientations)
    bin_coherences = np.array(bin_coherences)
    bin_positions_y = np.array(bin_positions_y)
    bin_positions_x = np.array(bin_positions_x)
    
    print(f"  Created {len(bin_orientations)} valid bins")
    
    # Create figure with exact pixel aspect ratio matching the input image
    dpi = 600
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # Span the entire figure (no margins)
    
    # Optionally show original image in background
    if overlay_image and image is not None:
        # Render as single-channel float 16 with colormap, avoiding explicit RGBA expansion.
        ds = 1
        if max_display_dim is not None and max(h, w) > max_display_dim:
            ds = int(np.ceil(max(h, w) / float(max_display_dim)))

        img_display = image[::ds, ::ds].astype(np.float32, copy=False)
        valid = img_display > 0
        if np.any(valid):
            p_low, p_high = np.percentile(img_display[valid], [1, 99])
        else:
            p_low, p_high = 0.0, 1.0
        img_display = np.clip((img_display - p_low) / (p_high - p_low + 1e-8), 0, 1)

        magenta_cmap = mcolors.LinearSegmentedColormap.from_list(
            'magenta', ['black', 'magenta', 'white']
        )

        # Mask background so it stays black without creating a full RGBA ndarray.
        img_masked = np.ma.masked_where(img_display <= 0, img_display)
        ax.set_facecolor('black')
        ax.axis('off')
        ax.imshow(
            img_masked,
            cmap=magenta_cmap,
            aspect='auto',
            interpolation='nearest',
            extent=(0, w, h, 0),
        )
    
    # Draw vectors colored by coherence
    # Normalize coherence for colormap
    if len(bin_coherences) > 0:
        coher_norm = (bin_coherences - bin_coherences.min()) / (bin_coherences.max() - bin_coherences.min() + 1e-8)
        cmap = plt.cm.jet  # Use jet colormap for coherence
        
        # Draw each vector
        for idx in range(len(bin_orientations)):
            x = bin_positions_x[idx]
            y = bin_positions_y[idx]
            angle = bin_orientations[idx]
            coher = bin_coherences[idx]
            
            # Vector direction (2 perpendicular directions since orientation is π-periodic)
            length = (bin_size / 2) * coher * vector_length_scale
            dx = length * np.cos(angle)
            dy = -length * np.sin(angle)  # Negative because positive Cartesian angle points "UP" (smaller image-Y)
            
            # Color by coherence
            color = cmap(coher_norm[idx])
            
            # Dynamic arrow styling proportional to bin_size
            h_width = max(0.5, bin_size * 0.15)
            h_length = max(0.2, bin_size * 0.08)
            l_width = max(0.2, bin_size * 0.03)
            
            # Draw line segment
            ax.arrow(x - dx, y - dy, 2*dx, 2*dy, 
                    head_width=h_width, head_length=h_length, fc=color, ec=color, 
                    alpha=0.8, linewidth=l_width, length_includes_head=True)
            
    # Customize plot boundaries (spines)
    for spine in ax.spines.values():
        if not overlay_image:
            spine.set_visible(False)
        else:
            spine.set_linewidth(2)   # Thickness of the boundary
            spine.set_color("black") # Color of the boundary

    # Set axis properties to cover the full image area precisely
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)  # Invert y-axis for image coordinates
    ax.set_aspect('equal')
    # ax.set_title(f'Fiber Orientation Vector Map', 
    #             fontsize=PLOT_STYLE['title_size'], fontweight='bold')
    if not overlay_image:
        ax.axis('off')
    
    # Add colorbar for coherence
    if len(bin_coherences) > 0:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=bin_coherences.min(), 
                                                                   vmax=bin_coherences.max()))
        sm.set_array([])
        # Save standalone colorbar
        cbar_filename = filename_png.replace('.png', '_colorbar.png')
        save_standalone_colorbar(sm, cbar_filename, 'Mean Coherence')
    
    if not overlay_image:
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
    else:
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        
    plt.savefig(filename_png, dpi=600, transparent=True)
    print(f"Saved: {filename_png}")
    plt.close(fig)
    
    return fig


def plot_polar_orientation_histogram(image, orientation, coherence=None, coherence_threshold=0.1, 
                                     bin_count=36, figsize=None, filename_png="orientation_rose_plot.png"):
    """
    Create a polar rose plot of fiber orientation distribution.
    Mirror the 0-180 degree distribution to 180-360 degrees to show bidirectional nature.
    
    Args:
        image (ndarray): Original image (for masking orientation)
        orientation (ndarray): Orientation map in radians (0 to pi)
        coherence (ndarray): Coherence/anisotropy map (for filtering)
        coherence_threshold (float): Minimum coherence to include (0-1)
        bin_count (int): Number of bins for the half-circle (default 36)
        figsize (tuple): Figure size
        filename_png (str): Output filename
        
    Returns:
        fig: Matplotlib figure object
    """
    
    # Downsample for rose plot calculations on very large images.
    if coherence is not None:
        (image_ds, orientation_ds, coherence_ds), ds = _auto_downsample_2d(image, orientation, coherence, max_pixels=120000000)
    else:
        (image_ds, orientation_ds), ds = _auto_downsample_2d(image, orientation, max_pixels=120000000)
        coherence_ds = None

    if ds > 1:
        print(f"Rose plot input downsampled by {ds}x for memory safety")

    # Define overall mask: intensity signal AND well-defined orientation (coherence)
    mask = image_ds > 0
    if coherence_ds is not None:
        mask = mask & (coherence_ds >= coherence_threshold)
    
    # Extract orientation for validated signal pixels only
    valid_orient = orientation_ds[mask].flatten()
    if len(valid_orient) == 0:
        print("Warning: No orientation data to plot in rose diagram.")
        return None
    
    # Create bins for [0, pi]
    bins = np.linspace(0, np.pi, bin_count + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    counts, _ = np.histogram(valid_orient, bins=bins)
    
    # Mirror the data to [pi, 2pi] for full bidirectional representation
    full_bin_centers = np.concatenate([bin_centers, bin_centers + np.pi])
    full_counts = np.concatenate([counts, counts])
    
    # Create polar figure
    if figsize is None:
        # Polar plots should be square
        width = PLOT_STYLE['width']
        figsize = (width, width)
        
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='polar')
    
    # Use HSV colormap mirrored across the 180-degree axis
    # (so that 90° and 270° share the same color, etc.)
    cmap = plt.cm.hsv
    colors = cmap((full_bin_centers % np.pi) / np.pi)
    
    # Draw bars
    width = (2 * np.pi) / (bin_count * 2)
    bars = ax.bar(full_bin_centers, full_counts, width=width, color=colors, 
                   edgecolor='black', linewidth=0.5, alpha=0.8)
    
    # Customize polar plot
    ax.set_theta_zero_location('E')  # 0 degrees is East (right)
    ax.set_theta_direction(1)        # Counter-clockwise
    ax.set_title('Fiber Orientation Rose Diagram\n(Bidirectional Symmetry)', 
                 fontsize=PLOT_STYLE['title_size'], fontweight='bold', pad=20)
    
    # Add angle labels in degrees
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    ax.set_xticklabels([f'{d}°' for d in np.arange(0, 360, 45)], fontsize=PLOT_STYLE['tick_size'])
    
    # Remove radial labels (counts) as requested
    ax.set_yticklabels([])
    
    plt.tight_layout()
    plt.savefig(filename_png, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename_png}")
    plt.close(fig)
    
    return fig

if __name__ == "__main__":
    try:
        # 1. Immediately export configuration parameters to JSON
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, f"{CONFIG['image_name']}_config.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(CONFIG, f, indent=4)
        print(f"Exported configuration parameters JSON to: {json_path}")
        print("\n--- Configuration Parameters ---")
        print(json.dumps(CONFIG, indent=4))

        # If export_config_only is set, exit immediately without executing heavy image processing
        if CONFIG.get('export_config_only', False) or ('--config-only' in sys.argv) or ('--export-config' in sys.argv):
            print("\n'export_config_only' is enabled. Exiting without running processing pipeline.")
            sys.exit(0)

        # Compute structure tensor with optimized parameters
        start_time = time.time()
        pixel_size_um = CONFIG['pixel_size_um']
        
        image = load_image(
            image_path=CONFIG['image_path'], 
            threshold_method=CONFIG['threshold_method'], 
            threshold_value=CONFIG['threshold_value']
        )

        # 1. Quantitative per-pixel distance/thickness (in microns)
        distance_um_map, thickness_stats = compute_thickness(
            image,
            pixel_size_um=pixel_size_um,
            map_type=CONFIG['map_type'],
            tophat_radius_um=CONFIG.get('tophat_radius_um', 4.0),
            file_name=f'{CONFIG["image_name"]}_{CONFIG["map_type"]}.tif',
            output_dir=output_dir,
            return_stats=True
        )

        # 2. Correlate Gaussian filter scales to actual fiber thickness or use config
        if CONFIG.get('auto_scale_from_thickness', True) and thickness_stats.get('median_um'):
            median_diam_um = thickness_stats['median_um']
            # Inner differentiation scale: ~0.25 * fiber diameter (clamped to at least 1.0 px)
            sigma_grad_um = max(0.25 * median_diam_um, 1.0 * pixel_size_um)
            sigma_grad_px = sigma_grad_um / pixel_size_um

            # Outer integration scale: ~1.75 * fiber diameter (integrates across fiber bundle)
            sigma_tensor_um = max(1.75 * median_diam_um, 2.0 * pixel_size_um)
            sigma_tensor_px = sigma_tensor_um / pixel_size_um

            print(f"\n--- Physically Calibrated Scale Selection ---")
            print(f"Median Fiber Thickness: {median_diam_um:.2f} um")
            print(f"Gradient Scale (Inner): {sigma_grad_um:.2f} um ({sigma_grad_px:.2f} px)")
            print(f"Tensor Scale   (Outer): {sigma_tensor_um:.2f} um ({sigma_tensor_px:.2f} px)")
        else:
            sigma_grad_px = CONFIG['sigma_gradient'] / pixel_size_um
            sigma_tensor_px = CONFIG['sigma_tensor'] / pixel_size_um

        # 3. Compute structure tensor and coherence of collagen signal
        image, orientation, coherence = compute_structure_tensor(
            image, 
            sigma_gradient=sigma_grad_px,
            sigma_tensor=sigma_tensor_px,
            linear_coherence=True,
            coherence_file_name=f'{CONFIG["image_name"]}_coherence.tif',
            orientation_file_name=f'{CONFIG["image_name"]}_orientation.tif',
            output_dir=output_dir
        )

        elapsed = time.time() - start_time
        print(f"\nComputation time: {elapsed:.2f} seconds")
        
        if CONFIG['plot_images']:
            # Calculate figure size to match image aspect ratio
            image_figsize = get_figsize(image.shape)

            # Create coherence heatmap visualization png
            print("\nCreating coherence heatmap...")
            visualize_coherence_heatmap(
                coherence,
                filename_png=os.path.join(output_dir, "coherence_heatmap_viz.png"),
                image=image
            )

            # Create orientation heatmap visualization png
            print("\nCreating orientation heatmap...")
            visualize_orientation_heatmap(
                image,
                orientation,
                coherence=coherence,
                coherence_threshold=0.1,
                filename_png=os.path.join(output_dir, "orientation_heatmap_viz.png")
            )

            # Create preferential orientation vector map (transparent)
            print("\nCreating preferential orientation vector map...")
            visualize_preferential_orientation_vector_map(
                orientation, coherence, image=image, 
                bin_size= int(64 * (image.size/ 50000000)),  # Adjust this for different levels of detail
                overlay_image=False,
                coherence_threshold=0.1,  # Only show vectors with coherence > 0.1
                vector_length_scale=1.0,
                filename_png=os.path.join(output_dir, "preferential_orientation_vector_map.png"),
                figsize=image_figsize
            )

            # Create polar rose plot of orientation (polar)
            print("\nPlotting rose plot of polar orientation...")
            plot_polar_orientation_histogram(
                image, orientation, coherence=coherence, coherence_threshold=0.1,
                filename_png=os.path.join(output_dir, "orientation_rose_plot.png"))
            
        # Update run configuration parameters and summary metrics to JSON
        config_to_save = dict(CONFIG)
        config_to_save['elapsed_seconds'] = round(elapsed, 2)
        if 'thickness_stats' in locals() and thickness_stats:
            config_to_save['thickness_stats'] = thickness_stats
        config_to_save['calculated_scales'] = {
            'sigma_gradient_px': round(float(sigma_grad_px), 3),
            'sigma_tensor_px': round(float(sigma_tensor_px), 3),
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(config_to_save, f, indent=4)
        print(f"Updated configuration and run statistics JSON at: {json_path}")

        print(f"\nAll plots and figures saved to: {output_dir}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
