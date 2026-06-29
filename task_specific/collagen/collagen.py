"""
Collagen Fiber Analysis Tool

This script provides comprehensive analysis and visualization of collagen fibers in SHG (Second Harmonic Generation) images.

Key Functions:
1. analyze_collagen(image_path): Computes density, anisotropy, and orientation of collagen fibers
2. visualize_collagen_anisotropy_and_vectors(image_path, bin_size=32): 
   Creates three-panel visualization similar to the reference figure:
   - Panel A: Original SHG image (magenta colormap)
   - Panel B: Collagen Anisotropy Signal Interpolated (jet colormap)
   - Panel C: Polar Vector Map with line segments (white background)
3. compare_bin_sizes(image_path, bin_sizes=[16, 32, 64]): 
   Compares polar vector maps at different bin sizes using line segments
4. visualize_varied_polar_vectors(image_path, bin_size=32):
   Creates varied polar vector map with different shapes based on anisotropy levels:
   - Low anisotropy: X-shapes/crosses
   - Medium anisotropy: Short line segments  
   - High anisotropy: Long line segments with perpendicular marks
5. create_detailed_polar_vector_analysis(image_path, bin_size=32):
   Creates comprehensive 2x2 analysis showing:
   - Original SHG image with analysis grid overlay
   - Anisotropy map
   - Fiber orientation map  
   - Final polar vector map


Usage Examples:
# Basic anisotropy and vector visualization (3-panel figure)
fig, results = visualize_collagen_anisotropy_and_vectors("image.tif", bin_size=32)

# Compare different bin sizes for line segment vectors
fig_compare = compare_bin_sizes("your_image.tif", bin_sizes=[16, 32, 64])

# Varied vector shapes based on anisotropy levels
fig_varied = visualize_varied_polar_vectors("image.tif", bin_size=32)

# Detailed analysis with 2x2 workflow visualization
fig_detailed = create_detailed_polar_vector_analysis("image.tif", bin_size=32)

# Custom bin size for any function
fig, results = visualize_collagen_anisotropy_and_vectors("image.tif", bin_size=16)

Parameters:
- bin_size: Controls the density of vectors in the polar map (smaller = more dense)
- figsize: Controls the size of the output figure

Author: Modified for anisotropy and polar vector visualization
"""

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
from scipy.interpolate import griddata
from skimage import filters, io, morphology
from skimage.color import rgb2gray
from skimage.filters import gabor_kernel
from skimage.measure import block_reduce

from tqdm import tqdm



def analyze_collagen(image_path):
    try:
        image = tifffile.memmap(image_path, mode="r")
        print(f"Successfully loaded image: {image.shape}, dtype: {image.dtype}")
        print(f"Image value range: [{image.min():.3f}, {image.max():.3f}]")
        if image.size > 1e8:
            print("Warning: Image is very large, processing may be slow. Downsampling..")
            image = image[::2, ::2]  # Simple downsampling by a factor of 2
            print(f"Downsampled image shape: {image.shape}")
        else:
            print("Image size is manageable, processing at full resolution.")


    except Exception as e:
        print(f"Error reading image: {e}")
        return None, None, None

    if image.ndim == 3:
        print("Converting multi-channel image to grayscale")
        image = rgb2gray(image)
    
    # Normalize image to 0-1 range if needed
    if image.max() > 1.0:
        print("Normalizing image values to 0-1 range")
        image = image.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min()+1e-8)  # Add small epsilon to avoid division by zero
    
    # Thresholding to remove background (optional, can be adjusted or removed)
    threshold = filters.threshold_otsu(image)
    image = np.where(image > threshold, image, 0)

    # Threshold again by nonzero pixels to focus on fibers
    mean_threshold = image[image > 0].mean() + 1 * image[image > 0].std() if np.any(image > 0) else 0
    image = np.where(image > mean_threshold, image, 0)
    


    try:
        print("Computing Meijering filter...")
        enhanced = filters.meijering(image, sigmas=range(1, 10, 2), mode="reflect", black_ridges=False)
        density = gaussian_filter(enhanced, sigma=5)
        print(f"Density computed successfully: range [{density.min():.3f}, {density.max():.3f}]")


        theta = np.linspace(0, np.pi, 8, endpoint=False)
        kernels = [gabor_kernel(frequency=0.1, theta=t) for t in theta]
        feats = np.zeros((image.shape[0], image.shape[1], len(kernels)), dtype=np.float32)
        for k, kernel in tqdm(enumerate(kernels), total=len(kernels), desc="Computing Gabor responses"):
            filtered = convolve2d(image, np.real(kernel), mode="same", boundary="wrap")
            feats[:, :, k] = filtered

        anisotropy = np.std(feats, axis=-1)
        orientation = theta[np.argmax(feats, axis=-1)]
        
        print(f"Anisotropy computed: range [{anisotropy.min():.3f}, {anisotropy.max():.3f}]")
        print(f"Orientation computed: range [{orientation.min():.3f}, {orientation.max():.3f}]")
        
        return density, anisotropy, orientation
        
    except Exception as e:
        print(f"Error in image processing: {e}")
        return None, None, None


def visualize_collagen(image, density, anisotropy, orientation):
    scale = 100  # Adjust this to change the density of arrows
    small_density = block_reduce(density, block_size=(scale, scale), func=np.mean)
    small_anisotropy = block_reduce(anisotropy, block_size=(scale, scale), func=np.mean)
    small_orientation = block_reduce(orientation, block_size=(scale, scale), func=np.mean)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Original image
    ax1.imshow(image, cmap="gray", aspect="auto")
    ax1.set_title("Original Collagen Image")
    ax1.set_xlabel("μm")
    ax1.set_ylabel("μm")

    y, x = np.mgrid[0 : small_density.shape[0], 0 : small_density.shape[1]]
    u = np.cos(small_orientation)
    v = np.sin(small_orientation)

    # Normalize density for color mapping
    norm_density = (small_density - small_density.min()) / (small_density.max() - small_density.min())

    # Create a custom colormap
    cmap = matplotlib.colormaps["jet"]
    colors = cmap(norm_density.flatten())

    # Plot arrows
    ax2.quiver(
        x, y, u, v, color=colors, pivot="middle", scale=30, headwidth=4, headlength=4, headaxislength=4
    )

    ax2.set_title("Spatial Collagen Fiber Analysis")
    ax2.set_xlabel("μm")
    ax2.set_ylabel("μm")
    ax2.invert_yaxis()

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=density.min(), vmax=density.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax2)
    cbar.set_label("Fiber Density", rotation=270, labelpad=15)

    plt.tight_layout()
    plt.show()


def analyze_collagen_density(image_path):
    try:
        image = io.imread(image_path)
        print(f"Image shape: {image.shape}, dtype: {image.dtype}")

        # Ensure image is 2D
        if len(image.shape) > 2:
            image = image[:, :, 0]  # Take first channel if it's a multi-channel image

        # Apply Gaussian filter to reduce noise
        smoothed = filters.gaussian(image, sigma=2)

        # Threshold the image to separate collagen fibers
        threshold = filters.threshold_otsu(smoothed)
        binary = smoothed > threshold

        # Calculate local density using a sliding window
        density_map = ndimage.generic_filter(binary.astype(float), np.mean, size=20)

        print(f"Density map shape: {density_map.shape}, dtype: {density_map.dtype}")
        return density_map

    except Exception as e:
        print(f"Error in analyze_collagen_density: {e}")
        return None


def analyze_fiber_thickness(image_path):
    try:
        image = io.imread(image_path)
        print(f"Image shape: {image.shape}, dtype: {image.dtype}")

        # Ensure image is 2D
        if len(image.shape) > 2:
            image = image[:, :, 0]  # Take first channel if it's a multi-channel image

        smoothed = filters.gaussian(image, sigma=2)
        threshold = filters.threshold_otsu(smoothed)
        binary = smoothed > threshold

        skeleton = morphology.skeletonize(binary)

        distance = ndimage.distance_transform_edt(binary)
        thickness_map = 2 * distance * skeleton

        filled_thickness = ndimage.gaussian_filter(thickness_map, sigma=3)

        print(f"Thickness map shape: {filled_thickness.shape}, dtype: {filled_thickness.dtype}")
        return filled_thickness

    except Exception as e:
        print(f"Error in analyze_fiber_thickness: {e}")
        return None


def plot_spatial_collagen_analysis(healthy_image_path, bpd_image_path):
    fig = plt.figure(figsize=(20, 15))

    for i, (image_path, title) in enumerate(
        [(healthy_image_path, "Healthy Lung"), (bpd_image_path, "BPD Lung")]
    ):
        print(f"Analyzing {title}")

        # Analyze density
        density_map = analyze_collagen_density(image_path)

        # Analyze thickness
        thickness_map = analyze_fiber_thickness(image_path)

        if density_map is None or thickness_map is None:
            print(f"Error processing {title} image")
            continue

        # Plot density map
        ax1 = fig.add_subplot(2, 2, i * 2 + 1)
        im_density = ax1.imshow(density_map, cmap="viridis", aspect="auto")
        ax1.set_title(f"{title} Spatial Collagen Density")
        plt.colorbar(im_density, ax=ax1, label="Density")

        # Plot 3D thickness map
        ax2 = fig.add_subplot(2, 2, i * 2 + 2, projection="3d")
        x = np.arange(0, thickness_map.shape[1])
        y = np.arange(0, thickness_map.shape[0])
        X, Y = np.meshgrid(x, y)

        # Mask very low values to improve visibility
        mask = thickness_map > np.percentile(thickness_map, 95)
        Z = np.ma.masked_where(~mask, thickness_map)

        surf = ax2.plot_surface(X, Y, Z, cmap="plasma", linewidth=0, antialiased=False)
        ax2.set_title(f"{title} Spatial Fiber Thickness")
        plt.colorbar(surf, ax=ax2, label="Thickness (μm)")

        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Thickness (μm)")

    plt.tight_layout()
    plt.savefig("spatial_collagen_analysis.tif", dpi=300)


def compare_collagen_metrics(healthy_images, bpd_images):
    healthy_density = []
    healthy_thickness = []
    bpd_density = []
    bpd_thickness = []

    for image_path in healthy_images:
        density_map = analyze_collagen_density(image_path)
        thickness_map = analyze_fiber_thickness(image_path)
        if density_map is not None and thickness_map is not None:
            healthy_density.append(np.mean(density_map))
            healthy_thickness.append(np.mean(thickness_map[thickness_map > 0]))

    for image_path in bpd_images:
        density_map = analyze_collagen_density(image_path)
        thickness_map = analyze_fiber_thickness(image_path)
        if density_map is not None and thickness_map is not None:
            bpd_density.append(np.mean(density_map))
            bpd_thickness.append(np.mean(thickness_map[thickness_map > 0]))

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.bar(
        ["Healthy", "BPD"],
        [np.mean(healthy_density), np.mean(bpd_density)],
        yerr=[np.std(healthy_density), np.std(bpd_density)],
        capsize=3,
        color=["#1f77b4", "#ff7f0e"],
    )
    ax1.set_ylabel("Mean Collagen Density")
    ax1.set_title("Collagen Density Comparison")

    ax2.bar(
        ["Healthy", "BPD"],
        [np.mean(healthy_thickness), np.mean(bpd_thickness)],
        yerr=[np.std(healthy_thickness), np.std(bpd_thickness)],
        capsize=3,
        color=["#1f77b4", "#ff7f0e"],
    )
    ax2.set_ylabel("Mean Fiber Thickness (μm)")
    ax2.set_title("Collagen Fiber Thickness Comparison")

    plt.tight_layout()
    plt.savefig("collagen_metrics_comparison.tif", dpi=300)


def visualize_collagen_anisotropy_and_vectors(image_path, bin_size=32, figsize=(15, 6)):
    """
    Create visualizations showing collagen anisotropy signal interpolated and polar vector map.
    
    Args:
        image_path (str): Path to the collagen SHG image
        bin_size (int): Size of bins for downsampling (default: 32)
        figsize (tuple): Figure size (width, height)
    """
    density, anisotropy, orientation = analyze_collagen(image_path)
    
    if density is None or anisotropy is None or orientation is None:
        print("Failed to analyze the image.")
        return
    
    # Load original image
    try:
        original_image = tifffile.imread(image_path)
        if original_image.ndim == 3:
            original_image = rgb2gray(original_image)
        
        original_image = original_image.astype(np.float64)
        
        print(f"Raw image for display: range [{original_image.min():.3f}, {original_image.max():.3f}]")
        print(f"Image percentiles: P1={np.percentile(original_image, 1):.3f}, P5={np.percentile(original_image, 5):.3f}, P95={np.percentile(original_image, 95):.3f}, P99={np.percentile(original_image, 99):.3f}")
        
    except Exception as e:
        print(f"Error reading image: {e}")
        return None
    
    height, width = anisotropy.shape

    smoothed_anisotropy = gaussian_filter(anisotropy, sigma=3)
    
    # Create high-resolution grid for interpolation
    x_orig = np.arange(width)
    y_orig = np.arange(height)
    x_new = np.linspace(0, width-1, width*2)
    y_new = np.linspace(0, height-1, height*2)
    X_new, Y_new = np.meshgrid(x_new, y_new)
    
    # Interpolate anisotropy values
    points = np.column_stack([np.repeat(x_orig, height), np.tile(y_orig, width)])
    values = smoothed_anisotropy.flatten()
    anisotropy_interpolated = griddata(points, values, (X_new, Y_new), method='cubic')
    
    scale = bin_size
    small_anisotropy = block_reduce(anisotropy, block_size=(scale, scale), func=np.mean)
    small_orientation = block_reduce(orientation, block_size=(scale, scale), func=np.mean)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Panel A: Original SHG image
    magenta_cmap = mcolors.LinearSegmentedColormap.from_list('magenta', ['black', 'purple', 'magenta'])
    
    # Use percentile-based normalization for better contrast
    p_low, p_high = np.percentile(original_image, [1, 99])
    original_image_enhanced = np.clip((original_image - p_low) / (p_high - p_low), 0, 1)
    
    print(f"Enhanced image range: [{original_image_enhanced.min():.3f}, {original_image_enhanced.max():.3f}]")
    
    axes[0].imshow(original_image_enhanced, cmap=magenta_cmap, aspect='auto')
    axes[0].set_title('Collagen Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Panel B: Anisotropy Signal Interpolated
    im1 = axes[1].imshow(anisotropy_interpolated, cmap='jet', aspect='auto', 
                        vmin=np.nanpercentile(anisotropy_interpolated, 5),
                        vmax=np.nanpercentile(anisotropy_interpolated, 95))
    axes[1].set_title('Collagen Anisotropy Signal Interpolated', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.set_label('Collagen Anisotropy [a.u.]', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.savefig("anisotropy.tif", dpi=300)
    plt.close()
    
    return fig, (density, anisotropy, orientation)
    



def visualize_varied_polar_vectors(image_path, bin_size=32, figsize=(8, 8)):
    """
    Create polar vector map with varied vector shapes similar to the reference image.
    Different anisotropy levels are represented by different vector shapes.
    
    Args:
        image_path (str): Path to the collagen SHG image
        bin_size (int): Size of bins for downsampling (default: 32)
        figsize (tuple): Figure size (width, height)
    """
    density, anisotropy, orientation = analyze_collagen(image_path)
    
    if density is None or anisotropy is None or orientation is None:
        print("Failed to analyze the image.")
        return
    
    # Create downsampled arrays for vector plotting
    scale = bin_size
    small_anisotropy = block_reduce(anisotropy, block_size=(scale, scale), func=np.mean)
    small_orientation = block_reduce(orientation, block_size=(scale, scale), func=np.mean)
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_facecolor('white')
    
    # Create a mask to only show vectors where anisotropy is significant
    mask = small_anisotropy > np.percentile(small_anisotropy, 20)
    
    y_pos, x_pos = np.where(mask)
    # Define anisotropy thresholds for different vector types
    aniso_values = small_anisotropy[mask]
    low_thresh = np.percentile(aniso_values, 20)
    high_thresh = np.percentile(aniso_values, 70)
    
    # Create varied vector representations
    for i in range(len(x_pos)):
        x_center = x_pos[i]
        y_center = y_pos[i]
        
        # Get the orientation and anisotropy at this position
        angle = small_orientation[y_center, x_center]
        aniso_val = small_anisotropy[y_center, x_center]
        
        norm_aniso = (aniso_val - aniso_values.min()) / (aniso_values.max() - aniso_values.min())
        enhanced_norm = np.power(norm_aniso, 0.5)  # Square root to enhance higher values
        
        color = plt.cm.coolwarm(enhanced_norm)
        
        # Different vector representations based on anisotropy level
        if aniso_val < low_thresh:
            # Low anisotropy: X-shape
            length = 0.6
            dx = length * np.cos(angle)
            dy = length * np.sin(angle)
            dx_perp = length * np.cos(angle + np.pi/2)
            dy_perp = length * np.sin(angle + np.pi/2)
            
            # Draw cross with enhanced visibility
            ax.plot([x_center - dx/2, x_center + dx/2], 
                   [y_center - dy/2, y_center + dy/2], 
                   color=color, linewidth=2, alpha=0.9)
            ax.plot([x_center - dx_perp/2, x_center + dx_perp/2], 
                   [y_center - dy_perp/2, y_center + dy_perp/2], 
                   color=color, linewidth=2, alpha=0.9)
            
        elif aniso_val < high_thresh:
            # Medium anisotropy: short line segments
            length = 0.7
            dx = length * np.cos(angle)
            dy = length * np.sin(angle)
            
            ax.plot([x_center - dx/2, x_center + dx/2], 
                   [y_center - dy/2, y_center + dy/2], 
                   color=color, linewidth=2.5, alpha=0.9)
            
        else:
            # High anisotropy: longer line segments with prominent perpendicular marks
            length = 0.9
            dx = length * np.cos(angle)
            dy = length * np.sin(angle)
            
            # Draw main line with enhanced visibility
            ax.plot([x_center - dx/2, x_center + dx/2], 
                   [y_center - dy/2, y_center + dy/2], 
                   color=color, linewidth=3.5, alpha=1.0)
            
            # Add more prominent perpendicular marks
            perp_length = 0.25  # Slightly longer for better visibility
            dx_perp = perp_length * np.cos(angle + np.pi/2)
            dy_perp = perp_length * np.sin(angle + np.pi/2)
            
            ax.plot([x_center - dx_perp/2, x_center + dx_perp/2], 
                   [y_center - dy_perp/2, y_center + dy_perp/2], 
                   color=color, linewidth=2, alpha=0.9)  # Thicker perpendicular marks
    
    # Set axis properties
    ax.set_xlim(-0.5, small_anisotropy.shape[1] - 0.5)
    ax.set_ylim(-0.5, small_anisotropy.shape[0] - 0.5)
    ax.set_title('Varied Polar Vector Map', fontsize=14, fontweight='bold')
    ax.axis('off')
    ax.invert_yaxis()
    
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Collagen Anisotropy [a.u.]', rotation=270, labelpad=15)
    
    # Set custom colorbar ticks to show actual anisotropy values
    actual_ticks = np.linspace(aniso_values.min(), aniso_values.max(), 5)
    enhanced_ticks = np.power((actual_ticks - aniso_values.min()) / (aniso_values.max() - aniso_values.min()), 0.5)
    cbar.set_ticks(enhanced_ticks)
    cbar.set_ticklabels([f'{tick:.4f}' for tick in actual_ticks])
    
    plt.tight_layout()
    plt.savefig("polar_vector_map.tif", dpi=300)
    plt.close()
    
    return fig



if __name__ == "__main__":
    image_path = r"C:\Users\jvill\Documents\integrated_pipeline\WSI_data\SRS\S21-K21_198_3-collagen.tif"
    try:

        density, orientation, anisotropy = analyze_collagen(image_path)
        print(f"Density range: [{density.min():.3f}, {density.max():.3f}]")
        print(f"Orientation range: [{orientation.min():.3f}, {orientation.max():.3f}]")
        print(f"Anisotropy range: [{anisotropy.min():.3f}, {anisotropy.max():.3f}]")


        
        # fig, results = visualize_collagen_anisotropy_and_vectors(image_path, bin_size=32, figsize=(15, 6))
        # print("Successfully created anisotropy and polar vector map visualization!")
        
        # # Varied polar vector visualization
        # print("Creating varied polar vector map...")
        # fig_varied = visualize_varied_polar_vectors(image_path, bin_size=32, figsize=(8, 8))
        # print("Successfully created varied polar vector map!")
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
        print("Please make sure your image file exists and is readable.")
