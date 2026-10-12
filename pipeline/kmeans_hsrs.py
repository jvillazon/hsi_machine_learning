import os
import sys
import glob
import gc
import shutil
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tifffile
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

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
    
    # Validate shape
    if data.shape[1] != num_samp:
        raise ValueError(
            f"Data channel mismatch: Expected {num_samp} channels, got {data.shape[1]}. "
            f"Check that the correct image files are being loaded. "
            f"Image shape: {data.shape}"
        )
    
    temp_norm = normalize(data)
    
    # Extract tail and head regions
    temp_end = temp_norm[:, -1:-4:-1]
    temp_start = temp_norm[:, :ch_start]
    
    # Remove baseline from silent region
    temp = temp_norm-np.mean(temp_start,axis=1)[:, np.newaxis]
    
    # # Estimate background magnitude
    spectra_magnitude = np.mean(temp_end, axis=1) - np.mean(temp_start, axis=1)
    background_arr = np.outer(spectra_magnitude, background)
    
    # Subtract background
    spectra_standard = (temp - background_arr).astype(np.float32)

    # spectra_standard = temp
    
    # Normalize to background-removed spectrum
    spectra_max_idx = np.argmax(np.mean(spectra_standard, axis=0))
    max_val = np.mean(spectra_standard[:, spectra_max_idx]) + 3 * np.std(spectra_standard[:, spectra_max_idx]).astype(np.float32)  
    min_val = np.mean(spectra_standard[:, :ch_start]) - 3 * np.std(spectra_standard[:, :ch_start]).astype(np.float32)
    spectra_norm = normalize(
        spectra_standard,
        max_val=max_val,
        min_val=min_val,
        axis=1
    )
    
    # # Final baseline removal
    # spectra = spectra_norm - np.median(spectra_norm[:, :ch_start], axis=1)[:, np.newaxis]
    
    return spectra_norm.astype(np.float32)


def standardize_and_save_images(image_paths, wn_1, wn_2, num_samp, background, ch_start, output_dir=r"clustering/standardized_data"):
    """
    Standardize images and save to disk to avoid memory issues.
    Also returns metadata about pixel counts per image.
    
    Parameters
    ----------
    image_paths : list
        List of image file paths
    output_dir : str
        Directory to save standardized data
    
    Returns
    -------
    standardized_info : dict
        Dictionary with 'paths', 'pixels', and 'shapes' for each standardized image
    """
    os.makedirs(output_dir, exist_ok=True)
    
    standardized_info = {
        "name": [],
        "paths": [],
        "pixels": [],
        "shapes": []
    }
    
    total_pixels = 0
    
    for img_idx, img_path in enumerate(tqdm(image_paths, desc="Standardizing and saving images")):
        print(f"\nProcessing image {img_idx}: {os.path.basename(img_path)}")
        
        # Load and reshape image
        image_memmap = tifffile.memmap(img_path, mode='r')
        
        # Handle different TIFF formats
        # If shape is (height, width, channels), transpose to (channels, height, width)
        if image_memmap.ndim == 3:
            if image_memmap.shape[0] > image_memmap.shape[2]:  # Likely (height, width, channels)
                image_memmap = np.transpose(image_memmap, (2, 0, 1))
        
        # Reshape: (channels, height, width) -> (height*width, channels)
        reshaped = image_memmap.reshape((image_memmap.shape[0], -1)).T
        reshaped = np.asarray(reshaped)
        
        # Flip to match original orientation
        raw_data = np.flip(reshaped, axis=1)
        print(f"  Shape: {raw_data.shape}")
        
        # Standardize spectra for this image
        standardized = spectral_standardization(reshaped, wn_1, wn_2, num_samp, background, ch_start)

        # standardized = normalize(reshaped)
        
        # Save standardized data to disk
        output_path = os.path.join(output_dir, f"standardized_{img_idx:03d}.npy")
        np.save(output_path, standardized.astype(np.float32))
        
        num_pixels = standardized.shape[0]
        height, width = image_memmap.shape[1:]
        
        standardized_info["name"].append(os.path.basename(img_path))
        standardized_info["paths"].append(output_path)
        standardized_info["pixels"].append(num_pixels)
        standardized_info["shapes"].append((height, width))
        
        total_pixels += num_pixels
        
        print(f"  Standardized shape: {standardized.shape}")
    
    print(f"\nTotal pixels standardized and saved: {total_pixels:,}")
    return standardized_info
    

# def spectral_sample(info):
#     """
#     Sample spectra from standardized images and create chunks for clustering.
    
#     Parameters
#     ----------
#     info : dict
#         Dictionary with 'paths' and 'pixels' for each standardized image
#     num_samples_per_image : int
#         Number of spectra to sample from each image
    
#     Returns
#     -------
#     sampled_array : numpy.ndarray
#     """
#     sample_pixels = []
#     for file_path in info["paths"]:
#         data_mmap = np.load(file_path, mmap_mode='r')
#         sample_pixels.append(np.array(data_mmap[:]))
#         del data_mmap
#         gc.collect()
#     sample_array = np.vstack(sample_pixels)

#     return sample_array.astype(np.float32)

# def PCA_batch(sample_array, n_components=10):
#     """
#     Apply PCA to sample array for dimensionality reduction.
    
#     Parameters
#     ----------
#     sample_array : numpy.ndarray
#         Array of shape (N, num_samp) with sampled spectra
#     n_components : int
#         Number of PCA components to retain
#     Returns
#     -------
#     pca_array : numpy.ndarray
#         PCA-transformed array of shape (N, n_components)
#     """       
#     pca = PCA(n_components=n_components, random_state=42)
#     pca_data = pca.fit_transform(sample_array)
#     return pca_data.astype(np.float32)


def batch_cluster_model(init_ar=None, n_clusters=10, batch_size=10000, random_state=42):
    """
    Apply MiniBatchKMeans clustering to sample array.
    
    Parameters
    ----------
    init_ar : numpy.ndarray, optional
        Initial cluster centers for KMeans
    n_clusters : int
        Number of clusters for KMeans
    batch_size : int
        Batch size for MiniBatchKMeans
    random_state : int
        Random state for reproducibility
    
    Returns
    -------
    kmeans_model : sklearn.cluster.MiniBatchKMeans
        The fitted KMeans model
    """
    if init_ar is not None:
        kmeans_model = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=random_state, init=init_ar)
    else:
        kmeans_model = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=random_state, init='k-means++')

    return kmeans_model


def load_batch_data(info, batch_size=50000):
    """
    Generator to load standardized data in batches for clustering.
    Yields batches of spectra from the standardized images to reduce memory laod.
    Parameters
    ----------
    info : dict
        Dictionary with 'paths' and 'pixels' for each standardized image
    batch_size : int
        Number of spectra to yield in each batch
    Returns
    -------
    batch_data : numpy.ndarray
        Batch of spectra for clustering
    """
    buffer = []  # Buffer to accumulate pixels across batches
    for file_path in info["paths"]:
        data_mmap = np.load(file_path, mmap_mode='r')
        num_pixels = data_mmap.shape[0]
        chunk_size = min(batch_size * 2, 100000)  # Process in reasonable chunks
        for chunk_start in range(0, num_pixels, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_pixels)
            chunk = np.array(data_mmap[chunk_start:chunk_end])
            buffer.append(chunk)
            while len(buffer) > 0:
                combined = np.vstack(buffer) if len(buffer) > 1 else buffer[0]
                if combined.shape[0] >= batch_size:
                    num_full_batches = combined.shape[0] // batch_size
                    for batch_idx in range(num_full_batches):
                        start = batch_idx * batch_size
                        end = start + batch_size
                        yield combined[start:end]
                    remainder = combined[num_full_batches * batch_size:]
                    if len(remainder) > 0:
                        buffer = [remainder]
                    else:
                        buffer = []
                    break
                else:
                    break
            del chunk
            gc.collect()
        del data_mmap
        gc.collect()
    if len(buffer) > 0:
        remaining = np.vstack(buffer) if len(buffer) > 1 else buffer[0]
        if remaining.shape[0] > 0:
            yield remaining

def cluster_spectra(kmeans_model, info, batch_size=50000):
    """
    Cluster spectra using the provided KMeans model and standardized data.
    
    Parameters
    ----------
    kmeans_model : sklearn.cluster.MiniBatchKMeans
        The fitted KMeans model
    info : dict
        Dictionary with 'paths' and 'pixels' for each standardized image
    batch_size : int
        Number of spectra to process in each batch
    
    Returns
    -------
    all_labels : numpy.ndarray
        Cluster labels for all spectra across images
    """

    total_pixels = sum(info["pixels"])

    # First pass: Fit KMeans model in batches
    for batch_data in tqdm(load_batch_data(info, batch_size), desc="Fitting batches"):
        non_zero_mask = np.where(np.std(batch_data, axis=1) > 1e-2)
        batch_data = batch_data[non_zero_mask]
        kmeans_model.partial_fit(batch_data)
        del batch_data
        gc.collect()
    
    # Second pass: Predict cluster labels in batches
    labels = np.zeros(total_pixels, dtype=np.int32)
    pixel_count = 0
    for batch_data in tqdm(load_batch_data(info, batch_size), desc='CLustering batches'):
        non_zero_mask = np.where(np.std(batch_data, axis=1) > 1e-2)
        batch_non_bg = batch_data[non_zero_mask]
        batch_data = batch_non_bg[pixel_count:pixel_count + batch_size]
        labels[pixel_count:pixel_count + batch_size][non_zero_mask] = kmeans_model.predict(batch_non_bg)
        pixel_count += batch_size
        del batch_data
        gc.collect()
    
    centers = kmeans_model.cluster_centers_

    return labels, centers


def cluster_normalized_spectra(kmeans_model, info, reorg_centers, reorg_labels, ch_start, batch_size=50000):
    """
    Re-normalize all non-background spectra (where reorg_labels != 0) in batches.
    Each spectrum is normalized individually based on its own statistics.
    
    Parameters
    ----------
    kmeans_model : MiniBatchKMeans
        KMeans model to fit on normalized data
    info : dict
        Dictionary with paths and pixels for each image
    reorg_centers : ndarray
        Reorganized cluster centers
    reorg_labels : ndarray
        Current cluster labels (0 is background)
    ch_start : int
        Channel index for silent region
    batch_size : int
        Batch size for processing
    
    Returns
    -------
    norm_labels : ndarray
        New cluster labels from re-normalized data
    norm_centers : ndarray
        New cluster centers
    """
    total_pixels = sum(info["pixels"])

    # Debugging: Print reorg_label center stats
    unique_labels, counts = np.unique(reorg_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"  Label {label}: {count} pixels")
        print(f"Std. of center {label} is {np.std(reorg_centers[label])}")
        plt.plot(reorg_centers[label], label=f'Cluster {label}')
    plt.legend(fontsize=26)
    plt.title("KMeans Cluster Centers Before Re-normalization")
    plt.xlabel("Channel")
    plt.ylabel("Intensity")
    plt.grid(True, alpha=0.3)
    plt.show()

    
    # First pass: Fit KMeans on normalized, non-background spectra
    print("\nFirst pass: Fitting KMeans on re-normalized spectra...")
    pixel_count = 0
    for batch_data in tqdm(load_batch_data(info, batch_size), desc="Fitting on normalized spectra"):
        batch_size_actual = batch_data.shape[0]
        batch_labels = reorg_labels[pixel_count:pixel_count + batch_size_actual]


        
        # Filter to non-background pixels
        non_bg_mask = batch_labels != 0
        if np.any(non_bg_mask):
            batch_non_bg = batch_data[non_bg_mask]
            
            # Normalize each spectrum individually using vectorized operations
            max_vals = np.max(batch_non_bg, axis=1, keepdims=True)
            min_vals = np.mean(batch_non_bg[:, :ch_start], axis=1, keepdims=True)
            
            # Normalize using per-spectrum max/min
            batch_normalized = normalize(batch_non_bg, max_val=max_vals, min_val=min_vals, axis=1)
            kmeans_model.partial_fit(batch_normalized)
            
            del batch_non_bg, batch_normalized, max_vals, min_vals
        
        pixel_count += batch_size_actual
        del batch_data
        gc.collect()
    
    # Second pass: Predict labels on normalized spectra
    print("\nSecond pass: Predicting labels on normalized spectra...")
    norm_labels = np.full(total_pixels, np.nan, dtype=np.int32)
    pixel_count = 0
    for batch_data in tqdm(load_batch_data(info, batch_size), desc="Predicting normalized clusters"):
        batch_size_actual = batch_data.shape[0]
        batch_labels = reorg_labels[pixel_count:pixel_count + batch_size_actual]

        # Filter to non-background pixels
        non_bg_mask = batch_labels != 0
        if np.any(non_bg_mask):
            batch_non_bg = batch_data[non_bg_mask]
            
            # Normalize each spectrum individually using vectorized operations
            max_vals = np.max(batch_non_bg, axis=1, keepdims=True)
            min_vals = np.mean(batch_non_bg[:, :ch_start], axis=1, keepdims=True)
            
            batch_normalized = normalize(batch_non_bg, max_val=max_vals, min_val=min_vals, axis=1)
            predicted_labels = kmeans_model.predict(batch_normalized)
            norm_labels[pixel_count:pixel_count + batch_size_actual][non_bg_mask] = predicted_labels
            
            del batch_non_bg, batch_normalized, max_vals, min_vals
        
        pixel_count += batch_size_actual
        del batch_data
        gc.collect()
    
    norm_centers = kmeans_model.cluster_centers_
    return norm_labels, norm_centers

def reorganize_labels_clusters(labels, centers):
    """
    Reorganize cluster labels based on cluster center similarity to create a more interpretable order.
    Parameters
    ----------
    labels : numpy.ndarray
        Cluster labels for all spectra
    centers : numpy.ndarray
        Cluster centers from KMeans model
    Returns
    -------
    reorg_labels : numpy.ndarray
        Reorganized cluster labels
    reorg_centers : numpy.ndarray
        Reorganized cluster centers
    """
    # Reorganize clusters based on similarity of centers (e.g., by standard deviation or mean)
    reorg_idx = np.argsort(np.std(centers, axis=1))
    reorg_centers = centers[reorg_idx]
    reorg_labels = np.zeros_like(labels)
    for new_idx, old_idx in enumerate(reorg_idx):
        reorg_labels[labels == old_idx] = new_idx

    return reorg_labels, reorg_centers

def plot_cluster_spectra(num_clusters, reorg_centers, wn_1, wn_2, num_samp, file_name, save_dir):
    """
    Choose a color palette for visualizing clusters.
    
    Parameters
    ----------
    num_clusters : int
        Number of clusters to visualize
    reorg_centers : numpy.ndarray
        Reorganized cluster centers for plotting
    wn_1 : float
        Starting wavenumber
    wn_2 : float
        Ending wavenumber
    num_samp : int
        Number of samples (channels)
    Returns
    -------
    colors : list
        List of color codes for each cluster
    """
    
    default_colors = [
            '#000000', '#FF0000', '#00FF00', '#0000FF', 
            '#FFFF00', '#FF00FF', '#00FFFF', '#FFA500',
            '#800080', '#008080'
        ]
    
    print(f"\nSelect hexadecimal colors for cluster centers:")
    wavenumbers = np.linspace(wn_1, wn_2, num_samp)
    fig, ax = plt.subplots(figsize=(14, 10))
    color_list = []

    # # Normalize centers (except for background cluster 0) for better visualization
    # for i in range(1, num_clusters):
    #     center = reorg_centers[i]
    #     center_min = np.min(center)
    #     center_max = np.max(center)
    #     if center_max > center_min:
    #         reorg_centers[i] = (center - center_min) / (center_max - center_min)
    #     else:
    #         reorg_centers[i] = np.zeros_like(center)
    # reorg_centers[0] = reorg_centers[0] - np.min(reorg_centers[0])  # Ensure background cluster is also non-negative

    # Add color for cluster 0 (background) - always use first default color (black)
    color_list.append(default_colors[0])
    
    for i in range(1, num_clusters):
        default_color = default_colors[i % len(default_colors)]
        user_input = input(f'  Cluster {i} (default: {default_color}): ').strip()
        color = user_input if user_input else default_color
        color_list.append(color)
        ax.plot(wavenumbers, reorg_centers[i]-reorg_centers[i][0], color=color, linewidth=2, label=f'Cluster {i}')
    ax.set_xlabel('Wavenumber (cm$^{-1}$)')
    ax.set_xlabel('Wavenumbers (cm⁻¹)', fontsize=30, fontweight='bold')
    ax.set_ylabel('Normalized Intensity (A.U.)', fontsize=30, fontweight='bold')
    ax.set_title('Spectral Clusters', fontsize=30, fontweight='bold')
    ax.legend(loc='best', fontsize=20)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=18, colors='black')
    # Make axis labels darker
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    plt.tight_layout()
    filename = file_name
    plt.savefig(os.path.join(save_dir,filename), dpi=300)
    print(f"Saved cluster centers plot as '{filename}'")
    print(f"\nCluster colors assigned:")
    for i, color in enumerate(color_list):
        print(f"  Cluster {i}: {color}")

    # Ask user if they want to change colors
    response = input(f"\nDo you want to reassign colors? (yes/no) (View current cluster colors in {save_dir}): ").strip().lower()
    
    if response in ('yes', 'y'):
        print("\nEnter new hexadecimal colors for each cluster (or press Enter to keep current):")
        new_color_list = []
        for i in range(num_clusters):
            current_color = color_list[i] if i < len(color_list) else default_colors[i % len(default_colors)]
            user_input = input(f'  Cluster {i} (current: {current_color}): ').strip()
            new_color = user_input if user_input else current_color
            new_color_list.append(new_color)
        
        # Display new colors
        fig, ax = plt.subplots(figsize=(14, 10))
        print("\nNew cluster center plot with updated colors:")
        for i in range(num_clusters):
            ax.plot(wavenumbers, reorg_centers[i], color=new_color_list[i], linewidth=2, label=f'Cluster {i}')
        
        ax.set_xlabel('Wavenumbers (cm⁻¹)', fontsize=30, fontweight='bold')
        ax.set_ylabel('Normalized Intensity (A.U.)', fontsize=30, fontweight='bold')
        ax.set_title('Cluster Centers - Updated Colors', fontsize=30, fontweight='bold')
        ax.legend(loc='best', fontsize=20)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        filename = file_name
        plt.savefig(os.path.join(save_dir,filename), dpi=300)
        print(f"Saved cluster centers plot as '{filename}'")
        print(f"\nCluster colors assigned:")
        for i, color in enumerate(new_color_list):
            print(f"  Cluster {i}: {color}")
        return new_color_list
    else:
        return color_list

def reconstruct_cluster_images(reorg_labels, info, color_list, batch_size=50000):
    """
    Reconstruct and colorize clustered images using batches.
    Uses same batch size as clustering for consistency.
    
    Parameters
    ----------
    labels : ndarray
        Cluster labels for each pixel
    image_dict : dict
        Dictionary with name, pixels, shape for each image
    color_list : list
        Hex color codes for each cluster
    batch_size : int
        Batch size (should match clustering batch size)
    
    Returns
    -------
    reconstructed_images : list
        List of RGB reconstructed images
    """
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return np.array([int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float32)
    
    # Validate color_list has correct number of colors
    if len(color_list) != np.max(reorg_labels[~np.isnan(reorg_labels)]).astype(int) + 1:
        print(f"Warning: color_list has {len(color_list)} colors but max label is {np.max(reorg_labels[~np.isnan(reorg_labels)]).astype(int)}")
        print(f"Ensure plot_cluster_spectra returns {np.max(reorg_labels[~np.isnan(reorg_labels)]).astype(int) + 1} colors")
    
    color_map = np.array([hex_to_rgb(c) for c in color_list])
    reconstructed_images = [None] * len(info['name'])
    
    print("Reconstructing images in batches...")
    # If labels are nan for background, set to 0 for color mapping
    reorg_labels = np.where(np.isnan(reorg_labels), 0, reorg_labels).astype(int)
    total_pixels = len(reorg_labels)
    
    for batch_start in tqdm(range(0, total_pixels, batch_size), desc="Reconstructing images"):
        batch_end = min(batch_start + batch_size, total_pixels)
        batch_labels = reorg_labels[batch_start:batch_end]
        
        # Clip labels to valid range for color_map
        batch_labels = np.clip(batch_labels, 0, len(color_list) - 1)
        
        # Map labels to RGB colors
        batch_rgb = color_map[batch_labels.astype(int)]
        
        # Distribute batch pixels to corresponding images
        current_pixel = batch_start
        for img_idx in range(len(info['name'])):
            img_pixel_start = sum(info['pixels'][:img_idx])
            img_pixel_end = img_pixel_start + info['pixels'][img_idx]
            
            # Check overlap between batch and image
            overlap_start = max(current_pixel, img_pixel_start)
            overlap_end = min(batch_end, img_pixel_end)
            
            if overlap_start < overlap_end:
                # Initialize image if needed
                if reconstructed_images[img_idx] is None:
                    height, width = info['shapes'][img_idx]
                    reconstructed_images[img_idx] = np.zeros((height * width, 3), dtype=np.float32)
                
                # Copy RGB data to image
                batch_offset_start = overlap_start - batch_start
                batch_offset_end = overlap_end - batch_start
                img_offset_start = overlap_start - img_pixel_start
                img_offset_end = overlap_end - img_pixel_start
                
                reconstructed_images[img_idx][img_offset_start:img_offset_end] = batch_rgb[batch_offset_start:batch_offset_end]
    
    # Reshape to 2D images
    final_images = []
    for img_idx, img_pixels in enumerate(reconstructed_images):
        height, width = info['shapes'][img_idx]
        img_2d = img_pixels.reshape((height, width, 3))
        final_images.append(img_2d)
    
    return final_images

def display_reconstructed_images(reconstructed_images, info, save_dir, figsize_per_image=(5, 5), ):
    """
    Display reconstructed clustered images.
    
    Parameters
    ----------
    reconstructed_images : list of ndarray
        List of RGB images from reconstruct_images()
    info : dict
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
        ax.set_title(f'{info["name"][img_idx]} - Clustered', fontsize=12)
        ax.set_xlabel('Width (pixels)')
        ax.set_ylabel('Height (pixels)')
    
    # Hide unused subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'clustered_images.png'), dpi=300)
    plt.show()
    
    print(f"Displayed {num_images} clustered images")


def sample_spectra_for_k_search(info, max_samples=30000, random_state=42):
    rng = np.random.default_rng(random_state)
    n_images = len(info["paths"])
    samples_per_image = max(1, max_samples // n_images)

    sampled_chunks = []
    for file_path in info["paths"]:
        data = np.load(file_path, mmap_mode='r')
        n = data.shape[0]
        idx = np.sort(rng.choice(n, size=min(samples_per_image, n), replace=False))
        chunk = np.array(data[idx], dtype=np.float32)
        mask = np.std(chunk, axis=1) > 1e-2
        sampled_chunks.append(chunk[mask])
        del data, chunk
        gc.collect()

    sample = np.vstack(sampled_chunks)
    if sample.shape[0] > max_samples:
        idx2 = rng.choice(sample.shape[0], size=max_samples, replace=False)
        sample = sample[idx2]

    return sample.astype(np.float32)


def find_optimal_k(data, k_range, batch_size=10000, sil_sample_size=5000, random_state=42, label=""):
    rng = np.random.default_rng(random_state)
    sil_n = min(sil_sample_size, data.shape[0])
    sil_idx = rng.choice(data.shape[0], size=sil_n, replace=False)
    sil_data = data[sil_idx]

    inertias, sil_scores = [], []
    for k in tqdm(k_range, desc=f"Optimal k ({label})"):
        km = MiniBatchKMeans(
            n_clusters=k,
            batch_size=batch_size,
            random_state=random_state,
            n_init=3,
        )
        km.fit(data)
        inertias.append(km.inertia_)
        labels_sil = km.predict(sil_data)
        sil_scores.append(silhouette_score(sil_data, labels_sil))
        del km
        gc.collect()

    return inertias, sil_scores


def main():
    parser = argparse.ArgumentParser(description="SRS Hyperspectral K-Means Clustering")
    parser.add_argument("--image_dir", type=str, default=r"d:\ADATA Backup\HuBMAP\HuBMAP Xenium\Xenium HSI\xenium-HSI")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--srs_params_path", type=str, default=None)
    parser.add_argument("--wn_start", type=float, default=2700)
    parser.add_argument("--wn_end", type=float, default=3100)
    parser.add_argument("--num_samples", type=int, default=61)
    parser.add_argument("--n_clusters", type=int, default=7)
    parser.add_argument("--run_optimal_k", action="store_true")
    parser.add_argument("--batch_size", type=int, default=50000)
    args = parser.parse_args()

    wn_1 = args.wn_start
    wn_2 = args.wn_end
    num_samp = args.num_samples
    ch_start = int((2800 - wn_1) / (wn_2 - wn_1) * num_samp)

    image_dir = args.image_dir
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_base = os.path.basename(os.path.normpath(image_dir)) + '_kmeans_output'
        output_dir = os.path.join(os.path.dirname(image_dir), output_base)
    os.makedirs(output_dir, exist_ok=True)

    standardized_dir = os.path.join(output_dir, 'standardized_temp')

    background_spectrum = None
    if args.srs_params_path and os.path.exists(args.srs_params_path):
        srs_data = np.load(args.srs_params_path)
        background_spectrum = srs_data['background']
    else:
        ref_bg = os.path.join(os.path.dirname(__file__), 'references', 'water_HSI_71.csv')
        if os.path.exists(ref_bg):
            import pandas as pd
            bg_df = pd.read_csv(ref_bg)
            background_spectrum = bg_df.iloc[:, 1].values[:num_samp]
        else:
            background_spectrum = np.ones(num_samp, dtype=np.float32)

    image_paths = sorted(glob.glob(os.path.join(image_dir, '*.tif')))
    if not image_paths:
        print(f"No .tif files found in {image_dir}")
        return

    print(f"Found {len(image_paths)} images in {image_dir}")
    standardized_info = standardize_and_save_images(
        image_paths, wn_1, wn_2, num_samp, background_spectrum, ch_start,
        output_dir=standardized_dir
    )

    if args.run_optimal_k:
        print("Running optimal k search...")
        sample_unnorm = sample_spectra_for_k_search(standardized_info, max_samples=30000)
        _max_v = np.max(sample_unnorm, axis=1, keepdims=True)
        _min_v = np.mean(sample_unnorm[:, :ch_start], axis=1, keepdims=True)
        _valid = (_max_v - _min_v).squeeze() > 1e-6
        sample_norm = normalize(
            sample_unnorm[_valid],
            max_val=_max_v[_valid],
            min_val=_min_v[_valid],
            axis=1
        ).astype(np.float32)
        k_search_range = range(2, 13)
        inertias_norm, sil_norm = find_optimal_k(sample_norm, k_search_range, label="normalized")

    n_clusters = args.n_clusters
    kmeans_model = batch_cluster_model(n_clusters=n_clusters, batch_size=10000, random_state=42)
    labels, centers = cluster_spectra(kmeans_model=kmeans_model, info=standardized_info, batch_size=args.batch_size)
    reorg_labels, reorg_centers = reorganize_labels_clusters(labels, centers)

    color_list = [
        '#000000', '#FF0000', '#00FF00', '#0000FF',
        '#FFFF00', '#FF00FF', '#00FFFF', '#FFA500',
        '#800080', '#008080'
    ][:n_clusters]

    images = reconstruct_cluster_images(
        reorg_labels=reorg_labels,
        info=standardized_info,
        color_list=color_list,
        batch_size=args.batch_size
    )

    for idx, img in enumerate(images):
        img_name = os.path.splitext(os.path.basename(standardized_info['name'][idx]))[0]
        out_path = os.path.join(output_dir, f"{img_name}_clustered.tif")
        tifffile.imwrite(out_path, (img * 255).astype(np.uint8))
        print(f"Saved clustered image: {out_path}")

    if os.path.exists(standardized_dir):
        shutil.rmtree(standardized_dir)

if __name__ == '__main__':
    main()
