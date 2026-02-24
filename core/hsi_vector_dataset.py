# =============================================================================
# SRS Vector Dataset - Denoised HSI pixels paired with molecule vectors
# =============================================================================
import os
import glob
import numpy as np
import pandas as pd
import torch
import tifffile
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class HSI_SRS_Vector_Dataset(Dataset):
    """
    PyTorch Dataset that pairs denoised hyperspectral pixel data with corresponding molecule vectors.
    
    The dataset loads masked .tif files (each representing a specific molecule) and pairs
    the noisy pixel spectra with the corresponding class vectors and ID from LIPIDS MAPS dataset
    and lipidome projection.

    Reference:
    Olzhabaev T, Müller L, Krause D, Schwudke D, Torda AE. 
    Lipidome visualisation, comparison, and analysis in a vector space. 
    PLoS Comput Biol. 2025 Apr 15;21(4):e1012892. 
    doi: 10.1371/journal.pcbi.1012892. PMID: 40233092; PMCID: PMC12058142.
    
    Usage:
    ------
    dataset = HSI_SRS_Vector_Dataset(
        lipid_vector_dataset_path='molecule_dataset/lipid_id',
        hsi_image_dir='path/to/masked/tif/files',
        num_samples_per_class=2000
    )
    
    # Use with DataLoader
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    """
    
    def __init__(self, lipid_vector_dataset_path, hsi_image_dir,
                wavenumbers=61, wavenumber_start=2700, wavenumber_end=3100, exclude_lipids=None):
        """
        Initialize dataset by loading molecule data, SRS params, and HSI images.
        
        Parameters
        ----------
        lipid_vector_dataset_path : str
            Path to saved lipid vector .csv file (with or without extension)
        hsi_image_dir : str
            Directory containing masked hyperspectral .tif files named after molecules
        wavenumbers : int
            Number of wavenumbers/channels in the spectra (default 61)
        wavenumber_start : int
            Starting wavenumber for the spectral range (default 2700)
        wavenumber_end : int
            Ending wavenumber for the spectral range (default 3100)
        exclude_lipids : list or None
            List of lipid names to exclude from the dataset (default None)
        """
        # Add .npz extension if needed
        mol_path = lipid_vector_dataset_path if lipid_vector_dataset_path.endswith('.csv') else lipid_vector_dataset_path + '.csv'

        # Load molecule data
        print(f"Loading molecules from: {mol_path}")
        lipid_df = pd.read_csv(mol_path)
        lipid_ids = lipid_df['LipidID'].values
        lipid_names = lipid_df['ImageName'].values
        lipid_vectors = lipid_df.iloc[:, lipid_df.columns.get_loc('LipidID'):].values
        
        
        # Filter out excluded lipids
        if exclude_lipids is not None:
            exclude_set = set(exclude_lipids)
            keep_indices = [i for i, name in enumerate(lipid_ids) if name not in exclude_set]
            lipid_vectors = lipid_vectors[keep_indices]
            self.lipid_names = lipid_names[keep_indices]
            self.lipid_ids = lipid_ids[keep_indices]
        else:
            self.lipid_ids = lipid_ids
            self.lipid_names = lipid_names
        

        # Spectral data
        self.n_wavenumbers = wavenumbers
        self.ch_start = int((2800 - wavenumber_start) / (wavenumber_end - wavenumber_start) * wavenumbers)

        # Lipid vectors
        self.lipid_vectors = torch.from_numpy(lipid_vectors).float()
        self.n_molecules = len(self.lipid_ids)

        # Create lipid name to index mapping
        self.lipid_name_to_idx = {name: idx for idx, name in enumerate(self.lipid_names)}

        # Load HSI images from directory
        print(f"\nLoading HSI images from: {hsi_image_dir}")
        self.hsi_image_dir = hsi_image_dir
        self.tif_files = glob.glob(os.path.join(hsi_image_dir, '*.tif'))
        self.samples = []
        self._load_hsi_images()
        
        print("\nDataset initialized:")
        print(f"  Total molecules available: {len(self.lipid_names)}")
        print(f"  Molecules with HSI data: {len(set([lip_idx for _, lip_idx in self.samples]))}")
        print(f"  Total pixel samples: {len(self.samples)}")
    
    def _load_hsi_images(self):
        """
        Load all .tif files and extract pixel data.
        
        Maps each .tif filename to a molecule name, loads the image,
        and stores all pixel spectra along with their corresponding molecule index.
        """
        print("Processing .tif files...")
        
        lipids_found = []
        lipid_not_found = []
        
        for tif_path in tqdm(self.tif_files):
            # Extract molecule name from filename (remove .tif extension)
            filename = os.path.basename(tif_path)
            lipid_name = os.path.splitext(filename)[0]
            
            # Check if this molecule exists in our molecule dataset
            if lipid_name not in self.lipid_name_to_idx:
                lipid_not_found.append(lipid_name)
                continue
            
            lipids_found.append(lipid_name)
            lipid_idx = self.lipid_name_to_idx[lipid_name]
            
            # Load the .tif file
            image = tifffile.imread(tif_path)
            
            # Expected shape: (n_wavenumbers, height, width)
            if len(image.shape) != 3:
                print(f"  Warning: Skipping {filename} - unexpected shape {image.shape}")
                continue
            
            n_channels, height, width = image.shape
            
            if n_channels != self.n_wavenumbers:
                print(f"  Warning: Skipping {filename} - expected {self.n_wavenumbers} channels, got {n_channels}")
                continue
            
            # Reshape to (n_pixels, n_wavenumbers)
            pixels = image.reshape(n_channels, -1).T  # Shape: (height*width, n_wavenumbers)
            
            # Filter out zero/background pixels (assuming masked regions are non-zero)
            pixel_sums = np.sum(pixels, axis=1)
            valid_pixels = pixels[pixel_sums > 0]
            
            # Store each valid pixel with its corresponding molecule index
            for pixel_spectrum in valid_pixels:
                self.samples.append((pixel_spectrum, lipid_idx))
        
        print(f"  Lipids in .tif files: {sorted(lipids_found)}")
        if lipid_not_found:
            print(f"  Warning: .tif files without matching lipids: {sorted(lipid_not_found)}")
    
    def __len__(self):
        """Return total number of pixel samples."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a single training pair.
        
        Returns
        -------
        pixel_spectrum : torch.Tensor
            Pixel spectrum (min-max normalized)
        lipid_vector : torch.Tensor
            Embedding vector of the corresponding lipid
        """
        pixel_spectrum, lip_idx = self.samples[idx]
        
        # Convert pixel spectrum to tensor
        pixel_spectrum = torch.from_numpy(pixel_spectrum.copy()).float()
        
        # Flip spectrum (to match expected orientation)
        pixel_spectrum = torch.flip(pixel_spectrum, dims=[0])
        
        
        # Normalize: subtract baseline (mean of silent region) and min-max normalize
        baseline = torch.mean(pixel_spectrum[:self.ch_start])
        pixel_spectrum = pixel_spectrum - baseline
        
        # Min-max normalization
        spec_min = torch.min(pixel_spectrum)
        spec_max = torch.max(pixel_spectrum)
        pixel_spectrum = (pixel_spectrum - spec_min) / (spec_max - spec_min + 1e-6)
        
        # Get lipid vector
        lipid_vector = self.lipid_vectors[lip_idx]   
        
        return pixel_spectrum, lipid_vector
    
    def get_lipid_name(self, idx):
        """Get the lipid name for a given sample index."""
        _, lip_idx = self.samples[idx]
        return self.lipid_names[lip_idx]
    
    def get_sample_idx_by_lipid(self, lipid_name):
        """
        Get all sample indices for a specific lipid.
        
        Parameters
        ----------
        lipid_name : str
            Name of the lipid
            
        Returns
        -------
        list of int
            Indices of all samples belonging to this lipid
        """
        if lipid_name not in self.lipid_name_to_idx:
            return []
        
        lip_idx = self.lipid_name_to_idx[lipid_name]
        return [i for i, (_, l_idx) in enumerate(self.samples) if l_idx == lip_idx]

def create_dataloaders(dataset, batch_size=32, num_samples=2000, train_ratio=0.7, val_ratio=0.15, seed=42):
        """
        Create a DataLoader for this dataset.
        
        Parameters
        ----------
        batch_size : int
            Number of samples per batch (default 32)
        num_samples : int
            Number of samples to draw per class (default 2000)
        train_ratio : float
            Proportion of data to use for training (default 0.7)
        val_ratio : float
            Proportion of data to use for validation (default 0.15)
        seed : int
            Random seed for reproducibility (default 42)
        
        Returns
        -------
        DataLoader
            PyTorch DataLoader for this dataset
        """

        np.random.seed(seed)
        n_classes = dataset.n_molecules
        samples_per_class = dataset.num_samples_per_class
        min_class_size = samples_per_class  # All classes assumed equal in this dataset

        # If n_samples > 0.75 * min_class_size, use all samples for all classes
        if n_samples > 0.75 * min_class_size:
            n_samples = min_class_size

        indices = []
        for class_idx in range(n_classes):
            class_start = class_idx * samples_per_class
            class_indices = np.arange(class_start, class_start + samples_per_class)
            np.random.shuffle(class_indices)
            indices.extend(class_indices[:n_samples])

        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        return loader