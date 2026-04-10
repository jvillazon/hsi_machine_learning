import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from tqdm import tqdm
import torch
from torch.utils.data import Dataset


plt.switch_backend('TkAgg')  # Use TkAgg backend for display


base_directory = "/Users/jorgevillazon/Documents/files/codex-srs/HuBMAP .tif files for Jorge Part 1/data"

class HSI_Unlabeled_Dataset(Dataset):
    def __init__(self, img_dir, ch_start, transform=None, image_normalization=False, min_max_normalization=False,
                 wavenumber_start=2700, wavenumber_end=3100, num_samples=61, compute_stats=True):
        """
        Initialize HSI Dataset
        Args:
            img_dir: Directory containing .tif files
            ch_start: Channel index for silent region
            transform: Optional transforms
            wavenumber_start: Starting wavenumber for molecule dataset (default 2700)
            wavenumber_end: Ending wavenumber for molecule dataset (default 3100)
            num_samples: Number of samples in wavenumber range (default 61)
            compute_stats: Whether to compute global normalization stats (expensive scan)
        """
        self.wavenumber_start = wavenumber_start
        self.wavenumber_end = wavenumber_end
        self.num_samples = num_samples
        self.compute_stats = compute_stats
        # Find image files with both .tif and .tiff extensions.
        # On case-insensitive file systems (e.g., Windows), multiple glob
        # patterns can match the same file path, so deduplicate by normalized path.
        discovered_paths = []
        for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF']:
            discovered_paths.extend(glob.glob(os.path.join(img_dir, ext)))

        unique_paths = {}
        for path in discovered_paths:
            unique_paths[os.path.normcase(os.path.normpath(path))] = path
        self.img_list = sorted(unique_paths.values())
        self.ch_start = ch_start
        self.image_normalization = image_normalization
        self.min_max_normalization = min_max_normalization
        self.transform = transform
        
        # Initialize per-image statistics dictionary
        self.image_stats = {}
        
        # Compute image sizes and statistics
        self.img_size = [0]
        current_size = 0
        
        
        print("\nSaving image information...")
        for img_path in tqdm(self.img_list):
            image = tifffile.memmap(img_path, mode='r')
            with tifffile.TiffFile(img_path) as tif:
                if tif.is_imagej:
                    page = tif.pages[0]
                    pixel_size_x = 1/page.resolution[1]
                    pixel_size_y = 1/page.resolution[0]
                else:
                    pixel_size_x = 1
                    pixel_size_y = 1

            if len(image.shape) >= 2:
                height = image.shape[-2]
                width = image.shape[-1]
                size = (height * width)
                
                # Compute statistics for this image
                # Store image shape for  ion
                height, width = image.shape[-2], image.shape[-1]

                # Check if we need to compute global statistics
                image_min = None
                image_max = None
                
                if self.compute_stats:
                    # Use a chunked approach for statistics calculation to save memory
                    n_pixels = height * width
                    chunk_size = 1000000  # 1 million pixels per chunk
                    
                    # We'll calculate the mean of the silent region (image_min) 
                    # and the global mean/std for robust max (image_max)
                    sum_silent = 0.0
                    sum_all = 0.0
                    sum_sq_all = 0.0
                    
                    # Reshape memmap to (channels, pixels) for linear chunking
                    # Note: memmap.reshape is generally safe as it doesn't load whole data into RAM
                    image_flat = image.reshape(image.shape[0], -1)
                    
                    print(f"  Calculating statistics in chunks for {os.path.basename(img_path)}...")
                    for start_p in range(0, n_pixels, chunk_size):
                        end_p = min(start_p + chunk_size, n_pixels)
                        # Load chunk from flattened memmap: shape (channels, chunk_size)
                        chunk = image_flat[:, start_p:end_p].T
                        # Convert to float32 and flip wavenumbers
                        chunk = np.flip(chunk, axis=1).astype(np.float32)
                        
                        # Update silent region statistic
                        sum_silent += np.sum(np.mean(chunk[:, :self.ch_start], axis=1))
                        
                        # Update global statistics for mean/std
                        sum_all += np.sum(chunk)
                        sum_sq_all += np.sum(chunk**2)
                    
                    avg_silent = sum_silent / n_pixels
                    global_mean = sum_all / (n_pixels * image.shape[0])
                    global_std = np.sqrt((sum_sq_all / (n_pixels * image.shape[0])) - (global_mean**2))
                    
                    if self.min_max_normalization:
                        # In this mode, we previously used pixel-wise min/max which is very slow/memory-intensive
                        # For consistency with the requested fix, we'll store the scalar averages
                        image_min = avg_silent
                        image_max = global_mean + 3 * global_std
                        self.image_normalization = False
                    elif self.image_normalization:
                        # image_min is scalar subtracted from everything
                        image_min = avg_silent
                        image_max = global_mean + 3 * global_std
                    
                self.image_stats[img_path] = {
                    'image_min': image_min,
                    'image_max': image_max,
                    'pixel_size_x': pixel_size_x,
                    'pixel_size_y': pixel_size_y,
                    'height': height,
                    'width': width,
                    'start_idx': current_size
                }
            else:
                size = 0
            
            current_size += size
            self.img_size.append(current_size)


    def __len__(self):
        return self.img_size[-1]
    
    def __getitem__(self, idx):
        """Get a single normalized sample using per-image statistics"""
        # Find correct image and pixel
        diff = np.array([int(idx)-size for size in self.img_size[:-1]])
        mask = diff >= 0
        pix_val = int(np.min(diff[mask]))
        img_idx = np.where(diff == pix_val)[0][0]
        
        # Get image path and stats
        img_path = self.img_list[img_idx]
        img_stats = self.image_stats[img_path]
        
        # Load spectrum
        mmap = tifffile.memmap(img_path, mode='r')
        pixel_idx = np.unravel_index(pix_val, (mmap.shape[-2], mmap.shape[-1]))
        spectra = np.array(mmap[:, pixel_idx[0], pixel_idx[1]], dtype=np.float32)
        spectra = np.flip(spectra).copy()  # Make a contiguous copy after flipping
        
        # Convert to tensor
        spectra = torch.from_numpy(spectra)
    

        if self.image_normalization:
            # Get the min/max from entire image
            spectra = spectra - torch.mean(spectra[:self.ch_start])
            image_min = img_stats['image_min']
            image_max = img_stats['image_max']
            spectra = (spectra - image_min) / (image_max - image_min + 1e-6)
        else:
            # Normalize using pixel-specific min/max
            spectra_min = torch.mean(spectra[:self.ch_start])
            spectra_max = torch.max(spectra)
            spectra = (spectra - spectra_min) / (spectra_max - spectra_min + 1e-6)

        
        if self.transform:
            spectra = self.transform(spectra)
            
        return spectra, img_idx  # Return both spectrum and its image index
    
    def load_and_process_image(self, img_path):
        """
        Load and process a single image from the dataset
        
        Args:
            img_path: Path to the image file
            
        Returns:
            image_spectra: numpy array of shape (n_pixels, n_channels) containing processed spectra
        """
        # Load image using tifffile
        image = tifffile.memmap(img_path, mode='r')
        
        # Reshape to (pixels, wavenumbers)
        image_spectra = image.reshape(image.shape[0], -1).T
        
        # Flip spectra if needed (depending on dataset orientation)
        image_spectra = np.flip(image_spectra, axis=1).astype(np.float32)

        if self.image_normalization or self.min_max_normalization:
            # Get the min/max from stored image stats
            stats = self.image_stats[img_path]
            image_min = stats['image_min']
            image_max = stats['image_max']
            
            # Step 1: Mean center the silent region in-place
            # This is equivalent to spectra - mean(silent)
            silent_mean = np.mean(image_spectra[:, :self.ch_start], axis=1, keepdims=True)
            image_spectra -= silent_mean
            
            # Step 2: Normalize by range in-place
            # Final formula: (spectra - image_min) / (range + 1e-6)
            # Since we already subtracted silent_mean, we might need to adjust image_min 
            # but in this implementation image_min is actually the avg_silent from __init__,
            # so subtracting silent_mean effectively sets the baseline.
            
            # Actually, to follow the original logic exactly:
            # spectra = spectra - mean(silent)
            # spectra = (spectra - 0) / (image_max - 0 + 1e-6)  [if image_min was 0]
            
            denom = (image_max - image_min + 1e-6)
            # image_spectra already has silent region at ~0
            image_spectra /= denom
        
        return image_spectra  # Return as (n_pixels, n_wavenumbers)
    
    

    