## Multi-Image Spectral Standardization & K-Means Clustering

import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import silhouette_score
import glob

# Find project root by locating the 'core' directory
project_root = os.getcwd()
while not os.path.exists(os.path.join(project_root, 'core')):
    parent = os.path.dirname(project_root)
    if parent == project_root:  # reached filesystem root
        raise RuntimeError("Could not locate project root (looking for 'core' directory)")
    project_root = parent


# ==================== Configuration ====================
# Hyperspectral image parameters
wn_1 = 2700
wn_2 = 3100
num_samp = 61
ch_start = int((2800 - wn_1) / (wn_2 - wn_1) * num_samp)

# Data directories
image_dir = r"D:\ADATA Backup\HuBMAP\HuBMAP Xenium\Xenium HSI\data"
srs_params_path = 'params_dataset/srs_params_61.npz'

# Load background from SRS parameters
srs_data = np.load(srs_params_path)
background_spectrum = srs_data['background']

# Suppress matplotlib GUI event loop warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*event loop.*')


def normalize(array, max_val=None, min_val=None, axis=None):
    """
    Normalize array to specified range.
    
    Args:
        array: Input array (1D or 2D) - numpy array
        max_val: Maximum value for normalized output (computed from data if None)
        min_val: Minimum value for normalized output (computed from data if None)
        axis: For 2D arrays only - axis along which to normalize
              axis=0: normalize each column, axis=1: normalize each row
              axis=None: global normalization
              For 1D arrays, axis parameter is ignored
    
    Returns:
        Normalized numpy array
    """
    array = np.asarray(array, dtype=np.float32)

    # For 1D arrays, ignore axis parameter
    if array.ndim == 1:
        axis = None
    
    # Calculate min/max if not provided
    if axis is None:
        # Global normalization
        if max_val is None:
            max_val = np.max(array)
            max_val = max_val.astype(np.float32)
        if min_val is None:
            min_val = np.min(array)
            min_val = min_val.astype(np.float32)
        
        diff = (max_val - min_val).astype(np.float32)
        normalized_spectra = (array - min_val) / diff + 1e-6

        return normalized_spectra
    else:
        # Axis-based normalization for 2D arrays
        if max_val is None:
            max_val = np.max(array, axis=axis, keepdims=True)
            max_val = max_val.astype(np.float32)
        if min_val is None:
            min_val = np.min(array, axis=axis, keepdims=True)
            min_val = min_val.astype(np.float32)
        
        diff = (max_val - min_val).astype(np.float32)
        return (array - min_val) / diff + 1e-6




## Parallelized Image Loading and Metadata Extraction

def load_image_metadata(image_paths):
    """
    Load image metadata using memmap without loading actual data into memory.
    Returns image_dict with lazy references.
    Parameters:
    ----------
    image_paths: List
        List of image file paths to load metadata from.
    """
    image_dict = {
        "name": [],
        "idx": [],
        "pixels": [],
        "shape": [],
    }
    
    for img_idx, img_name in enumerate(tqdm(image_paths, desc="Loading metadata")):
        # Use memmap=True to avoid loading entire image into memory
        image_data = tifffile.memmap(img_name, mode='r')

        _, height, width = image_data.shape
        num_pixels = height * width
        
        image_dict["name"].append(os.path.basename(img_name))
        image_dict["idx"].append(img_idx)
        image_dict["pixels"].append(num_pixels)
        image_dict["shape"].append((height, width))

        print(f"Image {img_idx} of shape: {height, width}") 
    
    return image_dict

def load_and_reshape_image(img_path):
    """
    Load single image using lazy memmap and reshape for standardization.
    Uses memmap to avoid loading entire image into memory at once.
    Returns numpy array.
    """
    # Use memmap for lazy loading (data stays on disk until accessed)
    image_memmap = tifffile.memmap(img_path, mode='r')
    
    # Reshape: (channels, height, width) -> (height*width, channels)
    # Convert to numpy array
    reshaped = image_memmap.reshape((image_memmap.shape[0], -1)).T
    reshaped = np.asarray(reshaped)
    
    # Flip to match original orientation
    result = np.flip(reshaped, axis=1)
    
    return result

## Parallelized Spectral Standardization Functions

def spectral_standardization(data, wavenum_1, wavenum_2, num_samp, background, ch_start=None):
    """
    Apply spectral standardization to hyperspectral data.
    
    Parameters
    ----------
    data : numpy.ndarray of shape (N, num_samp)
        Hyperspectral data where N is number of pixels
    wavenum_1 : float
        Starting wavenumber
    wavenum_2 : float
        Ending wavenumber
    num_samp : int
        Number of samples
    background : numpy.ndarray of shape (num_samp,)
        Background spectrum
    ch_start : int, optional
        Channel index for silent region
    
    Returns
    -------
    spectra : numpy.ndarray of shape (N, num_samp)
        Standardized spectra
    """
    if ch_start is None:
        ch_start = int((2800 - wavenum_1) / (wavenum_2 - wavenum_1) * num_samp)
    
    # Normalize input data
    data = np.asarray(data, dtype=np.float32)
    temp_norm = normalize(data)
    
    # Extract tail and head regions
    temp_end = temp_norm[:, -1:-4:-1]
    temp_start = temp_norm[:, :ch_start]
    
    # Remove baseline from silent region
    temp = temp_norm-np.mean(temp_start,axis=1)[:, np.newaxis]
    
    # Estimate background magnitude
    spectra_magnitude = np.mean(temp_end, axis=1) - np.mean(temp_start, axis=1)
    background_arr = np.outer(spectra_magnitude, background)
    
    # Subtract background
    spectra_standard = temp - background_arr
    
    # Normalize to background-removed spectrum
    spectra_max_idx = np.argmax(np.mean(spectra_standard, axis=0))
    spectra_norm = normalize(
        spectra_standard,
        max_val=np.mean(spectra_standard[:, spectra_max_idx]) + 3 * np.std(spectra_standard[:, spectra_max_idx]),
        min_val=0,
        axis=1
    )
    
    # Final baseline removal
    spectra = spectra_norm - np.median(spectra_norm[:, :ch_start], axis=1)[:, np.newaxis]
    
    return spectra.astype(np.float32)


def standardize_image(image_paths, wn_1, wn_2, num_samp, background, ch_start, image_pixels):
    """
    Standardize multiple images using numpy.
    Loads and processes images sequentially to manage memory.
    
    Returns:
        combined_spectra : numpy.ndarray of shape (total_pixels, num_samp)
    """
    all_spectra = []
    
    for img_idx, img_path in enumerate(tqdm(image_paths, desc="Standardizing images")):
        print(f"\nProcessing image {img_idx}: {os.path.basename(img_path)}")
        
        # Load and reshape image
        image_memmap = tifffile.memmap(img_path, mode='r')
        
        # Reshape: (channels, height, width) -> (height*width, channels)
        # Convert to numpy array
        reshaped = image_memmap.reshape((image_memmap.shape[0], -1)).T
        reshaped = np.asarray(reshaped)
        
        # Flip to match original orientation
        raw_data = np.flip(reshaped, axis=1)
        print(f"  Shape: {raw_data.shape}")
        
        # Standardize spectra for this image
        standardized = spectral_standardization(raw_data, wn_1, wn_2, num_samp, background, ch_start)
        all_spectra.append(standardized)
        
        print(f"  Standardized shape: {standardized.shape}")
    
    # Combine all images into single array
    combined = np.vstack(all_spectra)
    return combined.astype(np.float32)



## K-Means Clustering Functions
def parallel_kmeans_clustering(spectral_array, n_clusters, n_init=20, batch_size=5000, max_iter=100):
    """
    Perform K-means on large numpy array using MiniBatchKMeans for memory efficiency.
    """
    print(f"\nStarting K-means with {n_clusters} clusters...")
    print(f"  Batch size: {batch_size}")
    print(f"  Max iterations: {max_iter}")
    
    # Use MiniBatchKMeans for large datasets - processes in batches
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        init_size=batch_size * 3,
        batch_size=batch_size,
        max_iter=max_iter,
        n_init=n_init,
        random_state=42,
        verbose=1
    )
    
    # Fit the model
    labels = kmeans.fit_predict(spectral_array)
    centers = kmeans.cluster_centers_
    
    return labels, centers, kmeans


def normalize_spectra(spectral_array, wn_1=2700, wn_2=3100, num_samp=61, method='trapezoid'):
    """
    Perform area normalization using trapezoidal integration.
    
    Parameters
    ----------
    spectral_array : ndarray
        Spectral data array of shape (n_pixels, n_channels)
    wn_1 : float
        Starting wavenumber
    wn_2 : float
        Ending wavenumber
    num_samp : int
        Number of spectral channels
    method : str
        Method for normalization ('trapezoid' or 'minmax')
    
    Returns
    -------
    normalized : ndarray
        Area-normalized spectra
    """
    # Calculate wavenumber spacing
    delta_wn = (wn_2 - wn_1) / (num_samp - 1)
    ch_start = int((2800 - wn_1) / (wn_2 - wn_1) * num_samp)
    
    if method == 'trapezoid':
        # Calculate area under curve for each spectrum using trapezoidal rule
        areas = np.trapezoid(spectral_array, axis=1, dx=delta_wn)
        
        # Reshape areas for broadcasting
        areas = areas[:, np.newaxis]
        
        # Avoid division by zero for background pixels
        areas = np.where(areas == 0, 1, areas)
        
        # Normalize: each spectrum divided by its area
        normalized = spectral_array / areas
        
    elif method == 'minmax':
        # Find min and max values for normalization
        min_val = np.min(spectral_array, axis=1, keepdims=True)
        max_val = np.max(spectral_array, axis=1, keepdims=True)
        
        # Normalize each spectrum to [0, 1]
        range_val = max_val - min_val
        range_val = np.where(range_val == 0, 1, range_val)
        normalized = (spectral_array - min_val) / range_val
        
        # Identify non-zero spectra (skip background zeros)
        mask_nonzero = np.any(spectral_array != 0, axis=1)
        
        # Only subtract baseline from non-zero spectra
        if np.any(mask_nonzero):
            baseline = np.median(normalized[mask_nonzero, :ch_start], axis=1, keepdims=True)
            normalized[mask_nonzero] = normalized[mask_nonzero] - baseline
    
    return normalized

## Visualization Functions

def visualize_random_standardized_spectra(spectral_array, wavenum_1, wavenum_2, num_samp, num_samples=3):
    """
    Visualize random standardized spectra for validation.
    
    Parameters
    ----------
    spectral_array : ndarray
        Combined standardized spectra array of shape (n_pixels, num_samp)
    wavenum_1 : float
        Starting wavenumber
    wavenum_2 : float  
        Ending wavenumber
    num_samp : int
        Number of spectral channels
    num_samples : int
        Number of random spectra to plot (default=3)
    """
    # Select random indices
    total_pixels = spectral_array.shape[0]
    random_index = np.random.randint(0, total_pixels)
    wavenumbers = np.linspace(wavenum_1, wavenum_2, num_samp)

    start_idx = max(0, random_index - (num_samples // 2))
    end_idx = min(total_pixels, random_index + (num_samples // 2))
    print(f"Random index: {random_index}, Sample range: {start_idx} to {end_idx}")

    sample_spectra = spectral_array[start_idx:end_idx]
    print(f"Sample shape: {sample_spectra.shape}")

    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, spectra in enumerate(sample_spectra):
        ax.plot(wavenumbers, spectra, alpha=0.7, linewidth=2, label=f"Spectrum {start_idx + idx}")

    ax.set_title('Spectral Standardization Validation - Random Spectra', fontsize=14, fontweight='bold')
    ax.set_xlabel('Wavenumbers (cm⁻¹)', fontsize=12)
    ax.set_ylabel('Normalized Intensity (A.U.)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
   

def optimize_silhouette_score(spectral_array, k_range=(2, 11), batch_size=1000):
    """
    Calculate silhouette scores for K-means optimization using a sample of the data.
    
    Parameters
    ----------
    spectral_array : ndarray
        Combined standardized spectra array
    k_range : tuple
        Range of cluster numbers to test (start, end exclusive)
    batch_size : int
        Batch size for MiniBatchKMeans if data is large
    
    Returns
    -------
    optimal_k : int
        Optimal number of clusters from silhouette analysis
    silhouette_scores : list
        Silhouette scores for each k value
    k_values : list
        List of k values tested
    """
    # Sample a small subset for silhouette score optimization
    total_pixels = spectral_array.shape[0]
    sample_size = min(3000, total_pixels)
    random_indices = np.random.choice(total_pixels, sample_size, replace=False)
    sample_data = spectral_array[random_indices]
    print(f"Sampled data shape: {sample_data.shape}")
    
    silhouette_scores = []
    k_values = list(range(k_range[0], k_range[1]))
    for k in tqdm(k_values, desc="Silhouette Optimization"):
        kmeans_temp = MiniBatchKMeans(
            n_clusters=k,
            random_state=42,
            n_init=5,
            batch_size=batch_size,
            max_no_improvement=3
        )
        labels_temp = kmeans_temp.fit_predict(sample_data)
        score = silhouette_score(sample_data, labels_temp)
        silhouette_scores.append(score)
    
    # Find optimal k
    optimal_k = k_values[np.argmax(silhouette_scores)]
    
    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_values, silhouette_scores, 'bo-', linewidth=2, markersize=8)
    ax.axvline(x=optimal_k, color='r', linestyle='--', linewidth=2, label=f'Optimal k={optimal_k}')
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title('K-Means Silhouette Score Optimization', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.show()
    
    print(f"\nOptimal k (from silhouette analysis): {optimal_k}")
    print(f"  Max silhouette score: {max(silhouette_scores):.4f}")
    
    return optimal_k, silhouette_scores, k_values

# Reconstruct clustered images
def reconstruct_images(final_labels, image_dict, rgb_colors):
    """
    Reconstruct clustered images by mapping labels to RGB colors.
    
    Parameters
    ----------
    final_labels : ndarray
        Cluster label for each pixel (from K-means)
    image_dict : dict
        Dictionary with 'name', 'pixels', 'shape' for each image
    rgb_colors : list of tuples
        RGB color tuples for each cluster (values 0-1)
    
    Returns
    -------
    reconstructed_images : list of ndarray
        RGB images with shape (height, width, 3) for each input image
    """
    # Create color map: cluster_id -> RGB
    color_map = np.array(rgb_colors)  # Shape: (n_clusters, 3)
    
    # Map labels to RGB colors
    print("Converting labels to RGB colors...")
    rgb_array = color_map[final_labels]
    
    # Reconstruct individual images
    print("Reconstructing individual images...")
    reconstructed_images = []
    pixel_start_idx = 0
    
    for img_idx in range(len(image_dict['name'])):
        num_pixels = image_dict['pixels'][img_idx]
        pixel_end_idx = pixel_start_idx + num_pixels
        
        # Extract RGB data for this image
        image_rgb = rgb_array[pixel_start_idx:pixel_end_idx]
        
        # Get original dimensions
        height, width = image_dict['shape'][img_idx]
        
        # Reshape to 2D image
        image_2d = image_rgb.reshape((height, width, 3))
        
        reconstructed_images.append(image_2d)
        pixel_start_idx = pixel_end_idx
    
    return reconstructed_images


def display_reconstructed_images(reconstructed_images, image_dict, figsize_per_image=(5, 5)):
    """
    Display reconstructed clustered images.
    
    Parameters
    ----------
    reconstructed_images : list of ndarray
        List of RGB images from reconstruct_images()
    image_dict : dict
        Dictionary with image metadata
    figsize_per_image : tuple
        Figure size per image
    """
    num_images = len(reconstructed_images)
    cols = min(3, num_images)
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * figsize_per_image[0], rows * figsize_per_image[1]))
    
    if num_images == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()
    
    for img_idx, image_2d in enumerate(reconstructed_images):
        ax = axes[img_idx]
        ax.imshow(image_2d, aspect='auto', interpolation='nearest')
        ax.set_title(f'{image_dict["name"][img_idx]} - Clustered', fontsize=12)
        ax.set_xlabel('Width (pixels)')
        ax.set_ylabel('Height (pixels)')
    
    # Hide unused subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('clustered_images.png', dpi=300)
    plt.show()
    
    print(f"Displayed {num_images} clustered images")


def main():
    # ==================== Main Workflow ====================
    # Print configuration
    print("\n" + "="*60)
    print("Configuration:")
    print(f"  Wavenumber range: {wn_1}-{wn_2} cm⁻¹")
    print(f"  Number of samples: {num_samp}")
    print(f"  Channel start: {ch_start}")
    print(f"  Background spectrum shape: {background_spectrum.shape}")
    print("="*60 + "\n")

    # Find images
    image_paths = glob.glob(os.path.join(image_dir, '*.tif'))

    # Load metadata
    print("Loading image metadata...")
    image_dict = load_image_metadata(image_paths)
    print(f"Found {len(image_paths)} images with metadata loaded.")

    # Standardize spectra
    print("\nStarting spectral standardization...")
    combined_spectra = standardize_image(image_paths, wn_1, wn_2, num_samp, background_spectrum, ch_start, image_dict['pixels'])

    print(f"Combined array shape: {combined_spectra.shape}")
    print("Standardized spectra combined into numpy array.")

    # Visualize random standardized spectra for validation
    # visualize_random_standardized_spectra(combined_spectra, wn_1, wn_2, num_samp, num_samples=6)

    # # Optimize silhouette score to find best k
    # optimal_k, silhouette_scores, k_values = optimize_silhouette_score(
    #     combined_spectra,
    #     k_range=(2, 11),
    #     batch_size=1000
    # )

    # K-means clustering in parallel 
    k_initial = int(input("Enter number of clusters for initial k-means: "))
    initial_labels, initial_centers, kmeans_models = parallel_kmeans_clustering(
        combined_spectra,
        n_clusters=k_initial,
        batch_size=5000
    )

    print("Completed initial k-means clustering")

    # Reorganize cluster centers
    reorg_idx = np.argsort(np.mean(initial_centers, axis=1))
    reorg_centers = initial_centers[reorg_idx]

    # Default color palette
    default_colors = [
        '#000000', '#FF0000', '#00FF00', '#0000FF', 
        '#FFFF00', '#FF00FF', '#00FFFF', '#FFA500',
        '#800080', '#008080'
    ]

    # Color selection
    print("Select hexadecimal colors for each cluster:")

    wavenumbers = np.linspace(wn_1, wn_2, num_samp)
    fig, ax = plt.subplots(figsize=(12, 6))

    color_list = []
    for i in range(k_initial):
        default_color = default_colors[i % len(default_colors)]
        user_input = input(f'Enter hexcode for cluster {i} (default: {default_color}): ').strip()
        color = user_input if user_input else default_color
        color_list.append(color)
        # Plot cluster center
        ax.plot(wavenumbers, reorg_centers[i], color=color, linewidth=2, label=f'Cluster {i}')

    ax.set_xlabel('Wavenumbers (cm⁻¹)', fontsize=12)
    ax.set_ylabel('Normalized Intensity (A.U.)', fontsize=12)
    ax.set_title('Final K-Means Cluster Centers', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('kmeans_cluster_centers.png', dpi=300)
    print("Saved cluster centers plot as 'kmeans_cluster_centers.png'")

    print("\nCluster colors assigned:")
    for i, color in enumerate(color_list):
        print(f"  Cluster {i}: {color}")


    # Reconstruct and visualize images

    # Convert hex colors to RGB for visualization
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

    # Get image colors for visualization
    fig_colors = ['#FFFFFF', '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#FFA500', '#800080', '#008080']

    print("\nSelect hexadecimal colors for image visualization:")
    image_color_list = []
    for i in range(k_initial):
        default_color = fig_colors[i % len(fig_colors)]
        user_input = input(f'Enter hexcode for cluster {i} in images (default: {default_color}): ').strip()
        color = user_input if user_input else default_color
        image_color_list.append(color)

    rgb_colors = [hex_to_rgb(c) for c in image_color_list]

    print(f"\nReconstructing clustered images")

    # Reorganize cluster labels
    reorg_labels = np.zeros_like(initial_labels)
    for idx, label_idx in enumerate(reorg_idx):
        temp = initial_labels.copy()
        reorg_labels[temp==label_idx] = np.where(reorg_idx==label_idx)[0]
        print(f"Label {label_idx} is now {np.where(reorg_idx==label_idx)[0]}")

    # Reconstruct image
    reconstructed_images = reconstruct_images(
        reorg_labels,
        image_dict,
        rgb_colors,
    )

    print(f"Successfully reconstructed {len(reconstructed_images)} images")

    # Display all images
    display_reconstructed_images(reconstructed_images, image_dict, figsize_per_image=(6, 6))
    
    print("\nWorkflow completed successfully!")

if __name__ == "__main__":
    main()