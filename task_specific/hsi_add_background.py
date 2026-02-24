"""
HSI Background Addition Script

This script processes hyperspectral images by:
1. Normalizing each image along the spectral (Z) dimension
2. Subtracting the median of the silent region
3. Adding background from an srs_params.npz dataset
4. Saving the processed images

Usage:
    python hsi_add_background.py --input_dir <path> --params_file <path> --output_dir <path>
"""
import os
import sys
import numpy as np
import tifffile
from tqdm import tqdm
import matplotlib.pyplot as plt
core_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'core'))
if core_path not in sys.path:
    sys.path.insert(0, core_path)
from hsi_load_data import normalize






def normalize_hyperspectral_image(image, ch_start, visualize=False):
    """
    Normalize a 3D hyperspectral image using global baseline and maximum mean.
    
    Uses the global baseline (median of all silent region pixels) and the maximum
    of each pixel's mean value to normalize the entire image stack.
    
    Parameters
    ----------
    image : ndarray of shape (channels, height, width)
        3D hyperspectral image to normalize
    ch_start : int
        Index marking the start of the CH-stretching region (end of silent region)
    visualize : bool
        If True, show before/after normalization plots
        
    Returns
    -------
    normalized_image : ndarray of shape (channels, height, width)
        Normalized hyperspectral image
    max_mean : float
        Maximum of the mean values across all pixel spectra
    """
    # For visualization, sample a few pixels
    if visualize:
        h, w = image.shape[1], image.shape[2]
        sample_pixels = [
            (h//2, w//2),      # Center
            (h//4, w//4),      # Upper left
            (3*h//4, 3*w//4),  # Lower right
        ]
        pre_norm_spectra = [image[:, y, x] for y, x in sample_pixels]
    
    # Calculate global baseline (median of entire silent region)
    baseline = np.median(image[:ch_start, :, :])
    
    # Calculate the mean for each pixel, then take the maximum
    mean_per_pixel = np.mean(image, axis=0)  # Shape: (height, width)
    max_mean = np.max(mean_per_pixel)
    
    # Normalize: (image - baseline) / (max_mean - baseline)
    # This sets baseline to 0 and max mean to 1
    diff = max_mean - baseline
    normalized = (image - baseline) / (diff + 1e-10)
    
    # Diagnostic: check after normalization
    print(f"Global baseline: {baseline:.4f}, Maximum mean: {max_mean:.4f}")
    print(f"After normalization - max: {np.max(normalized):.4f}, min: {np.min(normalized):.4f}")
    print(f"Silent region after normalization - median: {np.median(normalized[:ch_start, :, :]):.4f}")
    
    # Visualization
    if visualize:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle('Normalization Comparison: Before (top) vs After (bottom)', fontsize=14, fontweight='bold')
        
        for i, (y, x) in enumerate(sample_pixels):
            # Pre-normalization
            axes[0, i].plot(pre_norm_spectra[i], linewidth=2)
            axes[0, i].axvline(ch_start, color='r', linestyle='--', alpha=0.5, label='CH start')
            axes[0, i].set_title(f'Pre-norm: Pixel ({y},{x})')
            axes[0, i].set_xlabel('Channel')
            axes[0, i].set_ylabel('Intensity')
            axes[0, i].grid(True, alpha=0.3)
            axes[0, i].legend()
            
            # Post-normalization
            axes[1, i].plot(normalized[:, y, x], linewidth=2, color='orange')
            axes[1, i].axvline(ch_start, color='r', linestyle='--', alpha=0.5, label='CH start')
            axes[1, i].set_title(f'Post-norm: Pixel ({y},{x})')
            axes[1, i].set_xlabel('Channel')
            axes[1, i].set_ylabel('Normalized Intensity')
            axes[1, i].grid(True, alpha=0.3)
            axes[1, i].legend()
        
        plt.tight_layout()
        plt.show()
    
    return normalized, max_mean


def add_background_to_image(image, background, bg_scale_factor=1.0):
    """
    Add background spectrum to a hyperspectral image.
    
    Parameters
    ----------
    image : ndarray of shape (channels, height, width)
        Normalized hyperspectral image
    background : ndarray of shape (channels,)
        Background spectrum to add
    bg_scale_factor : float
        Scaling factor for background intensity (default 1.0)
        
    Returns
    -------
    image_with_bg : ndarray of shape (channels, height, width)
        Image with background added
    """
    # Reshape background to broadcast across spatial dimensions
    # From (channels,) to (channels, 1, 1)
    bg_reshaped = background[:, np.newaxis, np.newaxis]
    
    print(f"Background range: [{np.min(background):.4f}, {np.max(background):.4f}]")
    print(f"Background scale factor: {bg_scale_factor}")
    
    # Add scaled background to image
    image_with_bg = image + (bg_scale_factor * bg_reshaped)
    
    print(f"After adding background - max: {np.max(image_with_bg):.4f}, min: {np.min(image_with_bg):.4f}")
    
    return image_with_bg


def process_directory(input_dir, params_file, output_dir, bg_scale_factor=1.0, file_suffix='_bg', visualize=False):
    """
    Process all hyperspectral images in a directory.
    
    Parameters
    ----------
    input_dir : str
        Directory containing input .tif hyperspectral images
    params_file : str
        Path to srs_params.npz file containing background and parameters
    output_dir : str
        Directory to save processed images
    bg_scale_factor : float
        Scaling factor for background intensity (default 1.0)
    file_suffix : str
        Suffix to add to output filenames (default '_bg')
    visualize : bool
        If True, show before/after plots for first image
    """
    # Load SRS parameters
    print(f"Loading SRS parameters from: {params_file}")
    params = np.load(params_file)
    
    background = params['background']
    bg_scale_vec = params['bg_scale_vec']
    ratio_scale_vec = params['ratio_scale_vec']
    ch_start = int(params['ch_start'])
    num_samples = int(params['num_samples'])
    
    print(f"Background shape: {background.shape}")
    print(f"Number of samples: {num_samples}")
    print(f"CH start index: {ch_start}")
    print(f"Background scale factor: {bg_scale_factor}")
    print(f"Background scale vector range: [{np.min(bg_scale_vec):.4f}, {np.max(bg_scale_vec):.4f}]")
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of .tif files
    file_list = [f for f in os.listdir(input_dir) if f.endswith('.tif') or f.endswith('.tiff')]
    
    if not file_list:
        print(f"No .tif files found in {input_dir}")
        return
    
    print(f"\nProcessing {len(file_list)} images...")
    
    processed_count = 0
    skipped_count = 0
    
    for filename in tqdm(file_list, desc="Processing images"):
        try:
            # Load image
            filepath = os.path.join(input_dir, filename)
            image = tifffile.imread(filepath)
            
            # Check if image has correct number of channels
            if image.shape[0] != num_samples:
                print(f"\nWarning: {filename} has {image.shape[0]} channels, expected {num_samples}. Skipping.")
                skipped_count += 1
                continue
            
            # Flip image (match coordinate convention from hsi_load_data.py)
            image = np.flip(image, axis=0)



            
            # Normalize image (visualize only for first valid image)
            show_viz = visualize and processed_count == 0
            normalized_image, _ = normalize_hyperspectral_image(image, ch_start, visualize=show_viz)
            noise = np.std(normalized_image[:ch_start, :, :])
            print(f"Image noise level (standard deviation of silent region): {noise:.4f}")

            snr_scale = max(np.random.choice(ratio_scale_vec) * noise, 
                       2* noise)
            print(f"SNR scale factor: {snr_scale:.4f}")
            
            normalized_image *= snr_scale

            # Recalculate max_mean after scaling
            mean_per_pixel = np.mean(normalized_image, axis=0)  # Shape: (height, width)
            max_mean = np.max(mean_per_pixel)

            
            # Background scale factor is 0.5 (half of normalized range)
            bg_scale = np.random.choice(bg_scale_vec) * snr_scale * max_mean
            
            # Add background
            image_with_bg = add_background_to_image(normalized_image, background, bg_scale)
            
            # Normalize again to ensure values are within expected range
            normalized_image, max_mean = normalize_hyperspectral_image(image_with_bg, ch_start, visualize=show_viz)

            # Flip back before saving
            image_with_bg = np.flip(image_with_bg, axis=0)
            
            # Save processed image
            name_without_ext = os.path.splitext(filename)[0]
            output_filename = f"{name_without_ext}{file_suffix}.tif"
            output_path = os.path.join(output_dir, output_filename)
            
            tifffile.imwrite(output_path, image_with_bg.astype(np.float32))
            processed_count += 1
            
        except Exception as e:
            print(f"\nError processing {filename}: {e}")
            skipped_count += 1
    
    print(f"\n{'='*80}")
    print(f"Processing complete!")
    print(f"  Processed: {processed_count} images")
    print(f"  Skipped: {skipped_count} images")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*80}")


def main():

    image_dir = r"/Volumes/ADATA SE880/ADATA Backup/Lipid Reference Library/HSI_data"
    parent_dir = os.path.dirname(image_dir)
    output_dir = os.path.join(parent_dir, "processed")

    params_file = r"params_dataset/srs_params_61.npz"


    process_directory(
        input_dir=image_dir,
        params_file=params_file,
        output_dir=output_dir,
        bg_scale_factor=1.0,
        file_suffix='_bg',
        visualize=True
    )


if __name__ == '__main__':
    main()
