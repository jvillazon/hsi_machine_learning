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
                 wavenumber_start=2700, wavenumber_end=3100, num_samples=61):
        """
        Initialize HSI Dataset
        Args:
            img_dir: Directory containing .tif files
            ch_start: Channel index for silent region
            transform: Optional transforms
            wavenumber_start: Starting wavenumber for molecule dataset (default 2700)
            wavenumber_end: Ending wavenumber for molecule dataset (default 3100)
            num_samples: Number of samples in wavenumber range (default 61)
        """
        self.wavenumber_start = wavenumber_start
        self.wavenumber_end = wavenumber_end
        self.num_samples = num_samples
        self.img_list = glob.glob(os.path.join(img_dir, '*.tif'))
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

                # Reshape to (wavenumbers, pixels)
                img_data = image.reshape(image.shape[0], -1).T
                img_data = np.flip(img_data, axis=1).astype(np.float32)

                if self.min_max_normalization:
                    # Compute pixel-specific min and max for normalization
                    image_min = np.mean(img_data[:,:self.ch_start], axis=1, keepdims=True)
                    image_max = np.max(img_data, axis=1, keepdims=True)
                    self.image_normalization = False  # Disable image normalization if min-max is used
                elif self.image_normalization:
                    # Compute image min and max for normalization
                    img_data = img_data - np.mean(img_data[:,:self.ch_start], axis=1, keepdims=True)
                    image_min = 0 # np.median(img_data[:,:self.ch_start]) --- IGNORE ---
                    image_max = np.mean(img_data) + 3*np.std(img_data)   # robust global max estimate
                else:
                    # Image normalization disabled
                    image_min = None
                    image_max = None
                    
                self.image_stats[img_path] = {
                    'image_min': image_min,
                    'image_max': image_max,
                    'pixel_size_x': pixel_size_x,
                    'pixel_size_y': pixel_size_y,
                    'height': height,
                    'width': width,
                    'start_idx': current_size  # Store starting index for this image
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

        if self.image_normalization:
            # Get the min/max from entire image
            image_spectra = image_spectra - np.mean(image_spectra[:,:self.ch_start], axis=1, keepdims=True)
            image_min = self.image_stats[img_path]['image_min']
            image_max = self.image_stats[img_path]['image_max']
            # Normalize using per-wavenumber min/max
            image_spectra = (image_spectra - image_min) / (image_max - image_min + 1e-6)
        
        if self.min_max_normalization:
            # Get channel min/max values for this image
            stats = self.image_stats[img_path]
            image_min = stats['image_min']
            image_max = stats['image_max']
        
            # Normalize using per-wavenumber min/max
            image_spectra = (image_spectra - image_min) / (image_max - image_min + 1e-6)
        
        return image_spectra  # Return as (n_pixels, n_wavenumbers)
    
    

    