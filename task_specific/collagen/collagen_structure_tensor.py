"""
Structure Tensor-based Collagen Fiber Analysis

This script uses the structure tensor method (similar to OrientationJ in Fiji) to analyze collagen fibers.
Much faster and more robust than Gabor filters while preserving directional continuity.

Key Functions:
1. compute_structure_tensor(image_path): Computes orientation and coherence using structure tensor
2. visualize_structure_tensor_analysis(): Creates similar visualizations to the Gabor version
3. compare_methods(): Side-by-side comparison of Gabor vs Structure Tensor

"""
import os

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import ndimage
from scipy.ndimage import gaussian_filter, sobel
from scipy.interpolate import griddata
from skimage import filters, io, morphology
from skimage.color import rgb2gray
from skimage.measure import block_reduce
from tqdm import tqdm
import time

import time

image_path = r"D:\ADATA Backup\HuBMAP\HuBMAP Xenium\SRS_data\05-14-25 Xenium S-2211-0003960 Ref\WSI-SHG-1031nm-700mw-4us-500V-zoom1.5_Cycle_03\SHG.tif"
output_dir = os.path.join(os.path.dirname(image_path), f"{os.path.basename(image_path).split('.')[0]}_collagen_output")
os.makedirs(output_dir, exist_ok=True)


CONFIG = {
    'image_path': image_path,
    'output_dir': output_dir,
    'pixel_size_um': 1.3258,
    'sigma_gradient': 0.5,
    'sigma_tensor': 1.0,
    'bin_size': None,
    'threshold_method': 'percentile',
    'threshold_value': 50,
    'distance_map_threshold_sigma': 1.0,
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
    
    cbar = plt.colorbar(mappable, cax=ax_cbar)
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

def compute_thickness(
    image,
    pixel_size_um=1.0,
    map_type='distance_um',
    threshold_sigma=1.0,
):
    """
    Compute quantitative distance/thickness maps from collagen mask.

    Args:
        image (ndarray): Input grayscale image of collagen fibers.
        pixel_size_um (float): Pixel size in microns per pixel.
        map_type (str):
            - 'distance_um': per-pixel distance to nearest background in microns
            - 'diameter_um': local full thickness (2 * distance) in microns
            - 'skeleton_diameter_um': diameter on skeleton only (legacy style)
        threshold_sigma (float): Gaussian smoothing sigma before thresholding.

    Returns:
        ndarray: Quantitative map in microns.
    """

    if len(image.shape) > 2:
        raise Exception("Input image must be grayscale")

    print(f"Computing quantitative map: {map_type} (pixel_size_um={pixel_size_um})...")
    # smoothed = filters.gaussian(image, sigma=threshold_sigma)
    # threshold = filters.threshold_li(image)
    # binary = image > threshold

    binary = image > 0

    # Distance to background in pixels, converted to microns.
    distance_px = ndimage.distance_transform_edt(binary)
    distance_um = distance_px.astype(np.float32, copy=False) * float(pixel_size_um)

    if map_type == 'distance_um':
        # Each foreground pixel stores distance to nearest background.
        out_map = distance_um
    elif map_type == 'diameter_um':
        # Local diameter estimate at each foreground pixel.
        out_map = 2.0 * distance_um
    elif map_type == 'skeleton_diameter_um':
        skeleton = morphology.skeletonize(binary)
        out_map = 2.0 * distance_um * skeleton
    else:
        raise ValueError(
            "map_type must be one of ['distance_um', 'diameter_um', 'skeleton_diameter_um']"
        )

    # Keep background at zero explicitly.
    out_map = np.where(binary, out_map, 0).astype(np.float32, copy=False)

    print(
        f"  Map stats: min={out_map.min():.3f} um, max={out_map.max():.3f} um, "
        f"nonzero={np.count_nonzero(out_map)}"
    )
    return out_map



def compute_structure_tensor(image, sigma_gradient=1.0, sigma_tensor=2.0, bin_size=None):
    """
    Compute fiber orientation and anisotropy using structure tensor method.
    This is the mathematical foundation of OrientationJ.
    
    Args:
        image (ndarray): Input grayscale image of collagen fibers
        sigma_gradient (float): Gaussian smoothing for gradient computation
        sigma_tensor (float): Gaussian smoothing for tensor. Larger = more robust to noise
        bin_size (int): Optional spatial averaging bin size
        
    Returns:
        orientation (ndarray): Dominant fiber angle at each pixel (0 to pi)
        coherence (ndarray): Anisotropy measure (0=isotropic, 1=perfectly aligned)
    """

    image_tup, ds =  _auto_downsample_2d(image, max_pixels=120000000)
    image = image_tup[0]
    # Normalize to 0-1 in float32 to keep memory bounded on large images.
    image = image.astype(np.float32, copy=False)
    if image.max() > 1.0:
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    height, width = image.shape
    print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")
    
    # Step 1: Compute gradients
    print(f"Computing gradients (sigma={sigma_gradient})...")
    
    # Smooth image first for more stable gradients
    if sigma_gradient > 0:
        smoothed = gaussian_filter(image, sigma=sigma_gradient)
        print(f"  Applied Gaussian smoothing with sigma={sigma_gradient}")
    else:
        smoothed = image
        print(f"  No smoothing applied (sigma=0)")
    
    # Compute gradients using Sobel filters
    Gx = sobel(smoothed, axis=1)  # x-direction
    Gy = sobel(smoothed, axis=0)  # y-direction
    
    print(f"  Gx range: [{Gx.min():.4f}, {Gx.max():.4f}]")
    print(f"  Gy range: [{Gy.min():.4f}, {Gy.max():.4f}]")
    
    # Step 2: Construct structure tensor elements
    print(f"\nComputing structure tensor components...")
    
    # Tensor components (outer product of gradient)
    Gxx = Gx * Gx
    Gyy = Gy * Gy
    Gxy = Gx * Gy
    
    # Step 3: Smooth tensor components (critical for robustness)
    print(f"\nSmoothing structure tensor components (sigma={sigma_tensor})...")
    
    if sigma_tensor > 0:
        Txx = gaussian_filter(Gxx, sigma=sigma_tensor)
        Tyy = gaussian_filter(Gyy, sigma=sigma_tensor)
        Txy = gaussian_filter(Gxy, sigma=sigma_tensor)
        print(f"  Applied Gaussian smoothing with sigma={sigma_tensor}")
    else:
        Txx = Gxx
        Tyy = Gyy
        Txy = Gxy
        print(f"  No smoothing applied (sigma=0) - using raw tensor components")
    
    # Step 4: Eigenvalue decomposition at each pixel
    print("\nComputing eigenvalue decomposition at each pixel...")
    
    # Vectorized eigenvalue computation for speed
    # For 2x2 matrix [[a, b], [b, c]], eigenvalues are:
    # lambda = (a + c ± sqrt((a-c)^2 + 4*b^2)) / 2
    
    trace = Txx + Tyy  # trace = sum of eigenvalues
    det = Txx * Tyy - Txy * Txy  # determinant
    discriminant = np.sqrt(np.maximum((Txx - Tyy)**2 + 4*Txy**2, 0))
    
    lambda_max = (trace + discriminant) / 2.0  # larger eigenvalue
    lambda_min = (trace - discriminant) / 2.0  # smaller eigenvalue

    lambda_min = np.nan_to_num(lambda_min, nan=0.0)
    lambda_max = np.nan_to_num(lambda_max, nan=0.0)

    
    # Coherence/Anisotropy = (lambda_max - lambda_min) / (lambda_max + lambda_min)
    # This ranges from 0 (isotropic) to 1 (perfectly aligned)
    denominator = lambda_max + lambda_min
    coherence = np.zeros_like(denominator, dtype=np.float32)
    valid = denominator > 1e-10
    coherence[valid] = np.square((lambda_max[valid] - lambda_min[valid]) / denominator[valid]).astype(np.float32)
    
    # Orientation from eigenvector of largest eigenvalue
    # For 2x2 symmetric matrix, if lambda is eigenvalue, eigenvector is proportional to
    # [Txy, lambda - Txx] or [lambda - Tyy, Txy]
    # Orientation angle = atan2(Gy_dominant, Gx_dominant) / 2 (divided by 2 because orientation is pi-periodic)
    
    # NOTE: arctan2(2*Txy, Txx - Tyy) gives the direction of MAXIMUM gradient change
    # For fiber analysis, we want the PERPENDICULAR direction (along which fibers align)
    # So we rotate by 90 degrees = π/2
    # We use -2*Txy to flip to a Right-Handed Cartesian system (y-up relative to horizontal)
    orientation = (0.5 * np.arctan2(-2*Txy, Txx - Tyy) + np.pi/2).astype(np.float32)
    # Normalize to [0, pi]
    orientation = orientation % np.pi

    # Free large intermediates early on large images.
    del Gx, Gy, Gxx, Gyy, Gxy, Txx, Tyy, Txy, trace, det, discriminant, lambda_max, lambda_min, denominator, valid
    
    print(f"\nResults:")
    print(f"  Coherence range: [{coherence.min():.3f}, {coherence.max():.3f}]")
    print(f"  Orientation range: [{orientation.min():.3f}, {orientation.max():.3f}] rad")
    
    return image, orientation, coherence


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
    
    # Use float32 for large images to reduce memory pressure.
    if image.dtype != np.float32:
        image = image.astype(np.float32)
        print(f"Converted to float32: range [{image.min():.3f}, {image.max():.3f}]")
    
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
    
    print(f"  Background pixels: {np.sum(image == 0)}")
    print(f"  Signal pixels: {np.sum(image > 0)}")
    print(f"Threshold BEFORE structure tensor prevents boundary artifacts")

    return image



def visualize_structure_tensor_results(image, orientation, coherence, eigenvalues_max, 
                                       bin_size=32, figsize=None, scale_factor=0.25, sigma_smooth=0.5):
    """
    Visualize structure tensor results.
    
    Args:
        image (ndarray): Original image
        orientation (ndarray): Orientation map
        coherence (ndarray): Coherence/anisotropy map
        eigenvalues_max (ndarray): Density-like measure
        bin_size (int): Downsampling factor for vector visualization
        figsize (tuple): Figure size
        scale_factor (float): Downscaling factor for high-resolution images (0.0-1.0)
        sigma_smooth (float): Gaussian smoothing sigma for visualization (default 0.5 for fine detail)
    """
    
    # Downscale image for memory efficiency
    original_image = image
    if scale_factor < 1.0:
        h, w = original_image.shape
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        original_image = block_reduce(original_image, block_size=(int(1/scale_factor), int(1/scale_factor)), func=np.mean)
        coherence_viz = block_reduce(coherence, block_size=(int(1/scale_factor), int(1/scale_factor)), func=np.mean)
    else:
        coherence_viz = coherence
    
    # Create figure with original and coherence maps
    if figsize is None:
        # Total width 12, each panel is 6. Aspect handles height.
        h, w = original_image.shape[:2]
        aspect = (2 * w) / h  # Double width due to two panels
        figsize = (PLOT_STYLE['width'], PLOT_STYLE['width'] / aspect)
        
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Panel A: Original image with magenta colormap
    magenta_cmap = mcolors.LinearSegmentedColormap.from_list('magenta', ['black', 'purple', 'magenta'])
    p_low, p_high = np.percentile(original_image, [1, 99])
    original_enhanced = np.clip((original_image - p_low) / (p_high - p_low), 0, 1)
    
    axes[0].imshow(original_enhanced, cmap=magenta_cmap, aspect='auto')
    # axes[0].set_title('Original SHG Image', fontsize=PLOT_STYLE['title_size'])
    axes[0].axis('off')
    
    # Panel B: Coherence map (anisotropy)
    smoothed_coherence = gaussian_filter(coherence_viz, sigma=sigma_smooth)
    im1 = axes[1].imshow(smoothed_coherence, cmap='jet', aspect='auto',
                        vmin=np.nanpercentile(smoothed_coherence, 5),
                        vmax=np.nanpercentile(smoothed_coherence, 95))
    # axes[1].set_title('Structural Coherence', fontsize=PLOT_STYLE['title_size'])
    axes[1].axis('off')
    # Colorbar removed from main plot - separate saving not implemented for this multi-panel view
    # to keep it simple, but can be added if needed.
    
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.savefig("structure_tensor_anisotropy.png", dpi=300, bbox_inches='tight')
    print("Saved: structure_tensor_anisotropy.png")
    plt.close(fig)
    
    return fig


def save_coherence_heatmap_tif(coherence, filename="coherence_heatmap.tif", image=None):
    """
    Save coherence (anisotropy) heatmap as full-resolution TIF image.
    Same spatial dimensions as original image for direct overlay.
    
    Args:
        coherence (ndarray): Coherence/anisotropy map (values 0-1)
        filename (str): Output filename for TIF image
        image (ndarray): Original image for masking (optional)
    Returns:
        None. Saves file to disk.
    """
    # Work on a float32 copy and mask in-place to avoid an extra full-size array.
    coherence_float32 = coherence.astype(np.float32, copy=True)
    if image is not None:
        coherence_float32[image <= 0] = 0

    # Ensure values are in range [0, 1]
    np.clip(coherence_float32, 0, 1, out=coherence_float32)
    
    # Save as float32 TIF for precision and easy colormap application
    tifffile.imwrite(filename, coherence_float32)
    
    print(f"\nCoherence heatmap TIF saved: {filename}")
    print(f"  Shape: {coherence_float32.shape}")
    print(f"  Range: [{coherence_float32.min():.3f}, {coherence_float32.max():.3f}]")
    print(f"  Use with colormap: plt.imshow(tifffile.imread('{filename}'), cmap='hot', vmin=0, vmax=1)")
    
    # Clean up large arrays from memory
    del coherence_float32


def save_thickness_tif(thickness, filename="thickness_map.tif", quantity_label="thickness", unit_label="um"):
    """
    Save thickness map as full-resolution TIF image.
    Same spatial dimensions as original image for direct overlay.
    
    Args:
        thickness (ndarray): Thickness map (values in pixels)
        filename (str): Output filename for TIF image
        
    Returns:
        None. Saves file to disk.
    """
    
    # Convert to float32 to preserve precision
    thickness_float32 = thickness.astype(np.float32)
    
    # Save as float32 TIF
    tifffile.imwrite(filename, thickness_float32)
    
    print(f"\n{quantity_label.capitalize()} map TIF saved: {filename}")
    print(f"  Shape: {thickness_float32.shape}")
    print(f"  Range: [{thickness_float32.min():.2f}, {thickness_float32.max():.2f}] {unit_label}")
    print(
        f"  Use with colormap: plt.imshow(tifffile.imread('{filename}'), "
        f"cmap='turbo', vmin=0, vmax={thickness_float32.max():.1f})"
    )
    
    # Clean up large arrays from memory
    del thickness_float32

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

    # Mask background in-place on a float32 working copy.
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



def compare_bin_sizes_structure_tensor(orientation, coherence, 
                                       bin_sizes=[16, 32, 64], figsize=(15, 5)):
    """
    Compare structure tensor vector maps at different bin sizes.
    """
    
    fig, axes = plt.subplots(1, len(bin_sizes), figsize=figsize)
    if len(bin_sizes) == 1:
        axes = [axes]
    
    for idx, bin_size in enumerate(bin_sizes):
        small_coherence = block_reduce(coherence, block_size=(bin_size, bin_size), func=np.mean)
        small_orientation = block_reduce(orientation, block_size=(bin_size, bin_size), func=np.mean)
        
        ax = axes[idx]
        ax.set_facecolor('white')
        
        mask = small_coherence > np.percentile(small_coherence, 20)
        y_pos, x_pos = np.where(mask)
        coherence_values = small_coherence[mask]
        
        # Draw vectors
        for i in range(len(x_pos)):
            x_center = x_pos[i]
            y_center = y_pos[i]
            
            angle = small_orientation[y_center, x_center]
            coherence_val = small_coherence[y_center, x_center]
            
            norm_coherence = (coherence_val - coherence_values.min()) / (coherence_values.max() - coherence_values.min())
            color = plt.cm.coolwarm(np.power(norm_coherence, 0.5))
            
            # Simple line segment
            length = 0.7 + 0.2 * norm_coherence
            dx = length * np.cos(angle)
            dy = length * np.sin(angle)
            
            ax.plot([x_center - dx/2, x_center + dx/2], 
                   [y_center - dy/2, y_center + dy/2], 
                   color=color, linewidth=2, alpha=0.9)
        
        ax.set_xlim(-0.5, small_coherence.shape[1] - 0.5)
        ax.set_ylim(-0.5, small_coherence.shape[0] - 0.5)
        ax.set_title(f'Bin Size: {bin_size}px', fontsize=20, fontweight='bold')
        ax.axis('off')
        ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig("structure_tensor_bin_comparison.png", dpi=300, bbox_inches='tight')
    print("Saved: structure_tensor_bin_comparison.png")
    plt.close(fig)
    
    return fig


def save_orientation_tif(orientation, filename="orientation_map.tif", image=None):
    """
    Save orientation map as a TIF image with values in degrees (0-180).
    Same spatial dimensions as original image for direct overlay with colormap.
    
    Args:
        orientation (ndarray): Orientation map in radians (0 to pi)
        filename (str): Output filename for TIF image
        image (ndarray): Original image for masking (optional)
    Returns:
        None. Saves file to disk.
    """
    orientation_work = orientation.astype(np.float32, copy=True)
    if image is not None:
        orientation_work[image <= 0] = 0
        
    
    # Convert from radians (0-pi) to degrees (0-180)
    orientation_degrees = np.degrees(orientation_work)
    
    # Ensure values are in range [0, 180]
    orientation_degrees = np.clip(orientation_degrees, 0, 180)
    
    # Convert to float32 to preserve precision
    orientation_float32 = orientation_degrees.astype(np.float32)
    
    # Save as TIF
    tifffile.imwrite(filename, orientation_float32)
    
    print(f"\nOrientation TIF saved: {filename}")
    print(f"  Shape: {orientation_float32.shape}")
    print(f"  Range: [{orientation_float32.min():.1f}, {orientation_float32.max():.1f}] degrees")
    print(f"  Use with colormap: plt.imshow(tifffile.imread('{filename}'), cmap='hsv', vmin=0, vmax=180)")
    
    # Clean up large arrays from memory
    del orientation_degrees, orientation_float32


def visualize_preferential_orientation_vector_map(orientation, coherence, image=None, 
                                                  bin_size=32, figsize=None, 
                                                  overlay_image=True, coherence_threshold=0.1,
                                                  vector_length_scale=1.0, filename="vector_map.png",
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
        # Render as single-channel float32 with colormap, avoiding explicit RGBA expansion.
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
        cbar_filename = filename.replace('.png', '_colorbar.png')
        save_standalone_colorbar(sm, cbar_filename, 'Mean Coherence')
    
    if not overlay_image:
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
    else:
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        
    plt.savefig(filename, dpi=600, transparent=True)
    print(f"Saved: {filename}")
    plt.close(fig)
    
    return fig


def plot_orientation_histogram(image, orientation, coherence=None, coherence_threshold=0.1, 
                               bin_count=36, figsize=None, filename="orientation_histogram.png"):
    """
    Create histogram of fiber orientation distribution with colormap overlay.
    Shows pixel count per orientation angle, with colors matching HSV colormap.
    
    Args:
        image (ndarray): Original image (for validation)
        orientation (ndarray): Orientation map in radians (0 to pi)
        coherence (ndarray): Coherence/anisotropy map (for optional filtering)
        coherence_threshold (float): Minimum coherence to include in histogram (0-1)
        bin_count (int): Number of bins for histogram (default 36 = 5 degree bins)
        figsize (tuple): Figure size
        filename (str): Output filename
        
    Returns:
        fig: Matplotlib figure object
    """
    
    # Downsample for histogram calculations on very large images.
    if coherence is not None:
        (image_ds, orientation_ds, coherence_ds), ds = _auto_downsample_2d(image, orientation, coherence, max_pixels=120000000)
    else:
        (image_ds, orientation_ds), ds = _auto_downsample_2d(image, orientation, max_pixels=120000000)
        coherence_ds = None

    if ds > 1:
        print(f"Orientation histogram downsampled by {ds}x for memory safety")

    # Define overall mask: intensity signal AND well-defined orientation (coherence)
    mask = image_ds > 0
    if coherence_ds is not None:
        mask = mask & (coherence_ds >= coherence_threshold)
    
    orientation = orientation[mask]

    # Convert to degrees (0-180)
    orientation_degrees = np.degrees(orientation_ds).flatten()
    
    # Create bins from 0 to 180
    bins = np.linspace(0, 180, bin_count + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Compute histogram
    counts, bin_edges = np.histogram(orientation_degrees, bins=bins)
    
    # Create figure
    if figsize is None:
        width = PLOT_STYLE['width']
        figsize = (width, width * 0.6)  # Standard landscape aspect for histogram
        
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use HSV colormap for bins (matches typical orientation visualization)
    cmap = plt.cm.hsv
    colors = cmap(bin_centers / 180.0)  # Normalize to [0, 1]
    
    # Create bar plot
    bars = ax.bar(bin_centers, counts, width=(180 / bin_count), color=colors, 
                   edgecolor='black', linewidth=0.5, alpha=0.8)
    
    ax.set_xlabel('Orientation (degrees)', fontsize=PLOT_STYLE['label_size'], fontweight='bold')
    ax.set_ylabel('Pixel Count', fontsize=PLOT_STYLE['label_size'], fontweight='bold')
    # ax.set_title('Fiber Orientation Distribution', fontsize=PLOT_STYLE['title_size'], fontweight='bold')
    ax.set_xlim(0, 180)
    
    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add colorbar to show orientation color mapping
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=180))
    sm.set_array([])
    # Save standalone colorbar
    cbar_filename = filename.replace('.png', '_colorbar.png')
    save_standalone_colorbar(sm, cbar_filename, 'Orientation (degrees)')
    print(f"\nSaved: {filename}")
    plt.close(fig)
    
    return fig


def circular_mean(angles_rad):
    """
    Compute circular mean of angles in radians.
    Accounts for the circular nature of orientation angles.
    
    Args:
        angles_rad (ndarray): Angles in radians
        
    Returns:
        float: Circular mean in radians
    """
    sin_sum = np.sin(angles_rad).sum()
    cos_sum = np.cos(angles_rad).sum()
    mean_angle = np.arctan2(sin_sum, cos_sum)
    return mean_angle % np.pi  # Ensure within [0, pi]


def plot_polar_orientation_histogram(image, orientation, coherence=None, 
                                     coherence_threshold=0.1, bin_count=36, 
                                     figsize=None, filename="orientation_rose_plot.png"):
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
        filename (str): Output filename
        
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
    # ax.set_title('Fiber Orientation Rose Diagram\n(Bidirectional Symmetry)', 
    #              fontsize=PLOT_STYLE['title_size'], fontweight='bold', pad=20)
    
    # Add angle labels in degrees
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    ax.set_xticklabels([f'{d}°' for d in np.arange(0, 360, 45)], fontsize=PLOT_STYLE['tick_size'])
    
    # Remove radial labels (counts) as requested
    ax.set_yticklabels([])
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close(fig)
    
    return fig


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

if __name__ == "__main__":

    try:
        
        # Compute structure tensor with optimized parameters
        start_time = time.time()

        # Choose thresholding method based on your ROI:
        # - 'otsu': Best for small, clean ROIs with clear background/signal distinction
        # - 'li': Better for larger ROIs with more complex intensity distributions
        # - 'yen': Good for unimodal distributions
        # - 'percentile': Manual control - e.g., threshold_value=50 removes bottom 50% of intensities
        # - 'none': No thresholding (analyze raw image)
        # image = load_image(image_path, threshold_method='otsu')
        
        # For larger ROIs with poor Otsu performance, try one of:
        # image = load_image(image_path, threshold_method='li')
        # image = load_image(image_path, threshold_method='percentile', threshold_value=75)
        image = load_image(image_path, threshold_method=CONFIG['threshold_method'], threshold_value=CONFIG['threshold_value'])

        pixel_size_um = CONFIG['pixel_size_um']



        #         # Quantitative per-pixel distance to background (in microns)
        # distance_um_map = compute_thickness(
        #     image,
        #     pixel_size_um=CONFIG['pixel_size_um'],
        #     map_type='distance_um',
        # )

        # image = load_image(image_path, threshold_method='yen')
        image, orientation, coherence = compute_structure_tensor(
            image, 
            sigma_gradient=CONFIG['sigma_gradient'] / pixel_size_um,  # Gradient smoothing (stabilize Sobel) in pixels
            sigma_tensor=CONFIG['sigma_tensor'] / pixel_size_um,    # ← Tensor smoothing in pixels
            bin_size=None
        )


        elapsed = time.time() - start_time
        print(f"\nComputation time: {elapsed:.2f} seconds")
        
        if orientation is not None and image is not None:


            # # Save quantitative distance-to-background map as TIF
            # print("\nSaving distance-to-background map (um) as TIF...")
            # save_thickness_tif(
            #     distance_um_map,
            #     filename=os.path.join(output_dir, "ckd-wsi-wsi-Collagen_Thickness-ratio.tif"),
            #     quantity_label="distance-to-background",
            #     unit_label="um",
            # )

            # Calculate figure size to match image aspect ratio
            image_figsize = get_figsize(image.shape)

            #             # Visualization 2: Save coherence as full-resolution TIF for overlay
            # print("\nSaving coherence heatmap as TIF...")
            # save_coherence_heatmap_tif(coherence, filename=os.path.join(output_dir, "ckd-wsi-wsi-Collagen_Coherence-ratio.tif"), image=image)

            # # Visualization 2b: Visualize coherence heatmap
            # print("\nCreating coherence heatmap visualization...")
            # visualize_coherence_heatmap(coherence, filename_png=os.path.join(output_dir, "coherence_heatmap_viz.png"), 
            #                             image=image, figsize=image_figsize)


            # # Visualization 4: Save orientation as TIF
            # print("\nSaving orientation map as TIF...")
            # save_orientation_tif(orientation, filename=os.path.join(output_dir, "orientation_map.tif"), image=image)

            # # Visualization 5: Orientation histograms
            # print("\nCreating orientation histogram...")
            # plot_orientation_histogram(image, orientation, coherence=coherence, 
            #                            coherence_threshold=0.1, bin_count=36, 
            #                            filename=os.path.join(output_dir, "orientation_histogram.png"))
            
            # print("\nCreating orientation rose plot...")
            # plot_polar_orientation_histogram(image, orientation, coherence=coherence, 
            #                                coherence_threshold=0.1, bin_count=36,
            #                                filename=os.path.join(output_dir, "orientation_rose_plot.png"))

            # # Visualization 5b: Orientation heatmap visualization
            # print("\nCreating orientation heatmap visualization...")
            # visualize_orientation_heatmap(image, orientation, coherence=coherence, coherence_threshold=0.1,
            #                               filename_png=os.path.join(output_dir, "orientation_heatmap_viz.png"),
            #                               figsize=image_figsize)

            # Visualization 6: Preferential orientation vector map
            print("\nCreating preferential orientation vector map...")
            visualize_preferential_orientation_vector_map(
                orientation, coherence, image=image, 
                bin_size= int(32 * (image.size/ 70000000)),  # Adjust this for different levels of detail
                overlay_image=False,
                coherence_threshold=0.1,  # Only show vectors with coherence > 0.1
                vector_length_scale=1.0,
                filename=os.path.join(output_dir, "preferential_orientation_vector_map.png"),
                figsize=image_figsize
            )
        print(f"\nPlots and figures saved to: {output_dir}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
