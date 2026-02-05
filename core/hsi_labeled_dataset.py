"""
PyTorch Dataset for creating labeled artificial SRS datasets from saved files.
"""
import os
import glob
import numpy as np
import pandas as pd
import torch
import tifffile
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from tqdm import tqdm

class HSI_Labeled_Dataset(Dataset):
    """
    PyTorch Dataset that generates labeled synthetic SRS data using saved experimental parameters.
    
    Usage:
    ------
    dataset = HSI_Labeled_Dataset(
        molecule_dataset_path='molecule_dataset/lipid_subtype_CH_61',
        srs_params_path='params_dataset/srs_params_61',
        num_samples_per_class=2000
    )
    
    # Use with DataLoader
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    """
    
    def __init__(self, molecule_dataset_path, srs_params_path,
                 num_samples_per_class=20000, normalize_per_molecule=False,
                 compute_min_max=True, exclude_molecules=None, noise_multiplier=1.0):
        """
        Initialize dataset by loading from saved .npz files.
        
        Parameters
        ----------
        molecule_dataset_path : str
            Path to saved molecule dataset .npz file (with or without extension)
        srs_params_path : str
            Path to saved SRS parameters .npz file (with or without extension)
        num_samples_per_class : int
            Total synthetic samples per molecule (default 20000)
        normalize_per_molecule : bool
            Whether to normalize each molecule by the max value computed from 1000 augmented samples
        compute_min_max : bool
            Whether to apply min-max normalization (default True)
        exclude_molecules : list of str, optional
            List of molecule names to exclude from the dataset (default None)
        noise_multiplier : float
            Multiplier for noise magnitude (default 1.0). Use 2.0 for double noise, 0.5 for half, 0.0 for no noise
        """
        # Add .npz extension if needed
        mol_path = molecule_dataset_path if molecule_dataset_path.endswith('.npz') else molecule_dataset_path + '.npz'
        srs_path = srs_params_path if srs_params_path.endswith('.npz') else srs_params_path + '.npz'
        
        # Load molecule data
        print(f"Loading molecules from: {mol_path}")

        mol_data = np.load(mol_path, allow_pickle=True)
        mol_spectra = mol_data['normalized_molecules']
        molecule_names = mol_data['molecule_names']
        
        # Filter out excluded molecules
        if exclude_molecules is not None:
            exclude_set = set(exclude_molecules)
            keep_indices = [i for i, name in enumerate(molecule_names) if name not in exclude_set]
            mol_spectra = mol_spectra[keep_indices]
            self.molecule_names = molecule_names[keep_indices]
        else:
            self.molecule_names = molecule_names
        
        # Load SRS parameters
        print(f"Loading SRS parameters from: {srs_path}")
        srs_data = np.load(srs_path)
        bg_scale_vec = srs_data['bg_scale_vec']
        ratio_scale_vec = srs_data['ratio_scale_vec']
        noise_scale_vec = srs_data['noise_scale_vec']
        ch_start = srs_data['ch_start']
        background = srs_data['background']
        
        # Convert to torch tensors
        self.mol_spectra = torch.from_numpy(mol_spectra).float()
        self.bg_scale_vec = np.minimum(bg_scale_vec, 0.75*np.mean(bg_scale_vec))# Ensure min background scaling of 0.75*max
        self.ratio_scale_vec = ratio_scale_vec  
        self.noise_scale_vec = noise_scale_vec
        self.background = torch.from_numpy(background.copy()).float()
        self.num_samples_per_class = num_samples_per_class
        self.noise_param = np.mean(noise_scale_vec) 
        self.noise_multiplier = noise_multiplier
        self.ch_start = int(ch_start)
        
        self.n_molecules = mol_spectra.shape[0]
        self.n_wavenumbers = mol_spectra.shape[1]

        # Store flag: whether to normalize by per-molecule max computed from augmented samples
        self.normalize_per_molecule = normalize_per_molecule
        self.compute_min_max = compute_min_max
        
        # Compute per-molecule max values from augmented samples
        if self.normalize_per_molecule:
            print("\nComputing per-molecule max values from 1000 augmented samples per class...")
            self.compute_molecule_max_values(num_samples=1000)

        print("Dataset initialized:")
        print(f"  Molecules: {self.n_molecules}")
        print(f"  Molecule names: {list(self.molecule_names)}")
        print(f"  Wavenumbers: {self.n_wavenumbers}")
        print(f"  Samples per class: {num_samples_per_class}")
        print(f"  Total samples: {len(self)}")
        print(f"  Normalize per molecule (by max): {self.normalize_per_molecule}")
        print(f"  Min-Max normalization: {self.compute_min_max}")

    
    def compute_molecule_max_values(self, num_samples=1000):
        """Compute max value for each molecule class from augmented samples.
        
        For each molecule class, generate num_samples augmented spectra (with noise,
        background, SNR scaling) and record the maximum intensity observed across all
        samples. This max is used for normalization in __getitem__.
        """
        self.molecule_max_vals = torch.zeros(self.n_molecules)
        
        for mol_idx in range(self.n_molecules):
            max_val = 0.0
            for _ in range(num_samples):
                # Sample experimental parameters
                bg_amplitude = np.random.choice(self.bg_scale_vec)
                snr_scale = max(np.random.choice(self.ratio_scale_vec) * self.noise_param, 
                               self.noise_param)
                noise = torch.randn(self.n_wavenumbers) * self.noise_param * self.noise_multiplier
                
                # Synthesize spectrum (same logic as __getitem__)
                spectrum = (self.mol_spectra[mol_idx] * snr_scale + 
                           self.background * bg_amplitude + 
                           noise)
                
                current_max = torch.max(spectrum).item()
                if current_max > max_val:
                    max_val = current_max
            
            self.molecule_max_vals[mol_idx] = max_val
        
        print(f"  Per-molecule max range: [{self.molecule_max_vals.min():.4f}, {self.molecule_max_vals.max():.4f}]")

    def __len__(self):
        return self.n_molecules * self.num_samples_per_class
    
    def __getitem__(self, idx):
        """Generate synthetic spectrum on-the-fly."""
        # Determine which molecule and which sample
        mol_idx = idx // self.num_samples_per_class
        
        # Sample experimental parameters
        bg_amplitude = np.random.choice(self.bg_scale_vec)
        snr_scale = max(np.random.choice(self.ratio_scale_vec) * self.noise_param, 
                       self.noise_param)
        
        # Generate noise
        noise = torch.randn(self.n_wavenumbers) * self.noise_param * self.noise_multiplier
        
        # Synthesize spectrum
        spectrum = (self.mol_spectra[mol_idx] * snr_scale + 
                   self.background * bg_amplitude + 
                   noise)
        
        # Apply normalization
        if self.normalize_per_molecule:
            # Normalize by the pre-computed max for this molecule class
            eps = 1e-8
            mol_max = self.molecule_max_vals[mol_idx]
            spectrum = spectrum / (mol_max + eps)
        elif self.compute_min_max:
            # Min-max normalization: normalize each individual spectrum by its own max and baseline
            spectrum_max = torch.max(spectrum)
            spectrum_min = torch.mean(spectrum[:self.ch_start])
            spectrum = (spectrum - spectrum_min) / (spectrum_max - spectrum_min + 1e-6)
        # else: No normalization
        
        return spectrum, mol_idx
    
    # Visualize sample spectra from training and validation sets
    def visualize_dataset_samples(self, train_ratio=0.7, val_ratio=0.15, num_samples_per_class=3, seed=42):
        """
        Visualize sample spectra from training and validation sets for each class.
        Uses stratified sampling to ensure all classes are represented.
        
        Parameters
        ----------
        dataset : HSI_Labeled_Dataset
            The dataset to visualize
        train_ratio : float
            Training split ratio
        val_ratio : float
            Validation split ratio
        num_samples_per_class : int
            Number of samples to plot per class
        seed : int
            Random seed for reproducibility
        """
        n_classes = self.n_molecules
        train_size_per_class = int(self.num_samples_per_class * train_ratio)
        val_size_per_class = int(self.num_samples_per_class * val_ratio)

        # Create stratified indices
        np.random.seed(seed)
        train_indices = []
        val_indices = []

        for mol_idx in range(self.n_molecules):
            class_start = mol_idx * self.num_samples_per_class
            class_indices = np.arange(class_start, class_start + self.num_samples_per_class)
            np.random.shuffle(class_indices)
            train_indices.extend(class_indices[:train_size_per_class])
            val_indices.extend(class_indices[train_size_per_class:train_size_per_class + val_size_per_class])
        
        # Create figure with subplots for each class
        n_cols = 4
        n_rows = (n_classes + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        axes = axes.flatten()
        
        for class_idx in range(n_classes):
            ax = axes[class_idx]
            
            # Get indices for this class
            class_start = class_idx * self.num_samples_per_class
            class_train_indices = [i for i in train_indices if class_start <= i < class_start + self.num_samples_per_class]
            class_val_indices = [i for i in val_indices if class_start <= i < class_start + self.num_samples_per_class]
            
            # Plot training samples
            for i in range(min(num_samples_per_class, len(class_train_indices))):
                idx = class_train_indices[i]
                spectrum, label = self[idx]
                ax.plot(spectrum.numpy(), alpha=0.5, color='blue', linewidth=0.8)
            
            # Plot validation samples
            for i in range(min(num_samples_per_class, len(class_val_indices))):
                idx = class_val_indices[i]
                spectrum, label = self[idx]
                ax.plot(spectrum.numpy(), alpha=0.5, color='red', linewidth=0.8)
            
            ax.set_title(f'{self.molecule_names[class_idx]}', fontsize=8)
            ax.set_xlabel('Channel', fontsize=7)
            ax.set_ylabel('Intensity', fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_classes, len(axes)):
            axes[idx].axis('off')
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', lw=2, label='Training'),
            Line2D([0], [0], color='red', lw=2, label='Validation')
        ]
        fig.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('dataset_visualization.png', dpi=150, bbox_inches='tight')
        print("\nDataset visualization saved to: dataset_visualization.png")
        plt.show()


# Dataloader for torch model
def create_dataloaders(dataset, batch_size=32, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Split dataset and create train/val/test loaders.
    
    Returns
    -------
    train_loader, val_loader, test_loader
    """
    from torch.utils.data import DataLoader, Subset
    
    # Calculate split sizes per class
    train_size = int(dataset.num_samples_per_class * train_ratio)
    val_size = int(dataset.num_samples_per_class * val_ratio)
    test_size = dataset.num_samples_per_class - train_size - val_size
    
    print("\nCreating splits:")
    print(f"  Train: {train_size * dataset.n_molecules} samples")
    print(f"  Val:   {val_size * dataset.n_molecules} samples")
    print(f"  Test:  {test_size * dataset.n_molecules} samples")
    
    # Create indices for each split (per-class to maintain balance)
    np.random.seed(seed)
    train_indices = []
    val_indices = []
    test_indices = []
    
    for mol_idx in range(dataset.n_molecules):
        # All indices for this class
        class_start = mol_idx * dataset.num_samples_per_class
        class_indices = np.arange(class_start, class_start + dataset.num_samples_per_class)
        
        # Shuffle and split
        np.random.shuffle(class_indices)
        train_indices.extend(class_indices[:train_size])
        val_indices.extend(class_indices[train_size:train_size + val_size])
        test_indices.extend(class_indices[train_size + val_size:])
    
    # Create subsets
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)
    
    # Create loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


# =============================================================================
# SRS Denoising Dataset - Real HSI pixels paired with molecule class annotations
# =============================================================================

class HSI_SRS_Vector_Dataset(Dataset):
    """
    PyTorch Dataset that pairs noisy hyperspectral pixel data with clean molecule spectra.
    
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
        srs_params_path='params_dataset/srs_params_61',
        lipid_vector_dataset_path='molecule_dataset/lipid_id',
        hsi_image_dir='path/to/masked/tif/files'
    )
    
    # Use with DataLoader
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    """
    
    def __init__(self, srs_params_path, lipid_vector_dataset_path, hsi_image_dir, wavenumbers=61,
                 wavenumber_start=2700, wavenumber_end=3100, exclude_molecules=None,
                 noise_multiplier=1.0):
        """
        Initialize dataset by loading molecule data, SRS params, and HSI images.
        
        Parameters
        ----------
        lipid_vector_dataset_path : str
            Path to saved lipid vector .csv file (with or without extension)
        srs_params_path : str
            Path to saved SRS parameters .npz file (with or without extension)
        hsi_image_dir : str
            Directory containing masked hyperspectral .tif files named after molecules
        wavenumbers : int
            Number of wavenumbers/channels in the spectra (default 61)
        wavenumber_start : int
            Starting wavenumber for the spectral range (default 2700)
        wavenumber_end : int
            Ending wavenumber for the spectral range (default 3100)
        exclude_molecules : list or None
            List of molecule names to exclude from the dataset (default None)
        noise_multiplier : float
            Multiplier for noise magnitude (default 1.0). Use 2.0 for double noise, 0.5 for half, 0.0 for no noise
        """
        # Add .npz extension if needed
        mol_path = lipid_vector_dataset_path if lipid_vector_dataset_path.endswith('.csv') else lipid_vector_dataset_path + '.csv'
        srs_path = srs_params_path if srs_params_path.endswith('.npz') else srs_params_path + '.npz'

        # Load molecule data
        print(f"Loading molecules from: {mol_path}")
        lipid_df = pd.read_csv(mol_path)
        lipid_ids = lipid_df['LipidID'].values
        lipid_names = lipid_df['ImageName'].values
        lipid_vectors = lipid_df.iloc[:, lipid_df.columns.get_loc('LipidID'):].values
        
        
        # Filter out excluded molecules
        if exclude_molecules is not None:
            exclude_set = set(exclude_molecules)
            keep_indices = [i for i, name in enumerate(lipid_ids) if name not in exclude_set]
            lipid_vectors = lipid_vectors[keep_indices]
            self.lipid_names = lipid_names[keep_indices]
            self.lipid_ids = lipid_ids[keep_indices]
        else:
            self.lipid_ids = lipid_ids
            self.lipid_names = lipid_names
        
        
        # Load SRS parameters
        print(f"Loading SRS parameters from: {srs_path}")
        srs_data = np.load(srs_path)
        bg_scale_vec = srs_data['bg_scale_vec']
        ratio_scale_vec = srs_data['ratio_scale_vec']
        noise_scale_vec = srs_data['noise_scale_vec']
        ch_start = srs_data['ch_start']
        background = srs_data['background']

        # Convert to torch tensors
        self.lipid_vectors = torch.from_numpy(lipid_vectors).float()
        self.bg_scale_vec = bg_scale_vec
        self.ratio_scale_vec = ratio_scale_vec  
        self.noise_scale_vec = noise_scale_vec
        self.background = torch.from_numpy(background.copy()).float()
        self.noise_param = np.mean(noise_scale_vec)
        self.ch_start = int(ch_start)
        self.n_molecules = len(self.lipid_ids)
        self.n_wavenumbers = wavenumbers


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
        noisy_spectrum : torch.Tensor
            Processed pixel spectrum (flipped, normalized, background-added)
        clean_spectrum : torch.Tensor
            Clean molecule spectrum (ground truth)
        """
        pixel_spectrum, mol_idx = self.samples[idx]
        
        # Convert pixel spectrum to tensor
        noisy_spectrum = torch.from_numpy(pixel_spectrum.copy()).float()
        
        # Flip spectrum (to match expected orientation)
        noisy_spectrum = torch.flip(noisy_spectrum, dims=[0])
        
        
        # Normalize: subtract baseline (mean of silent region) and min-max normalize
        baseline = torch.mean(noisy_spectrum[:self.ch_start])
        noisy_spectrum = noisy_spectrum - baseline
        
        # Min-max normalization
        spec_min = torch.min(noisy_spectrum)
        spec_max = torch.max(noisy_spectrum)
        noisy_spectrum = (noisy_spectrum - spec_min) / (spec_max - spec_min + 1e-6)
        
        # Get clean molecule spectrum
        clean_spectrum = self.mol_spectra[mol_idx]
        
        return noisy_spectrum, clean_spectrum
    
    def get_molecule_name(self, idx):
        """Get the molecule name for a given sample index."""
        _, mol_idx = self.samples[idx]
        return self.molecule_names[mol_idx]
    
    def get_samples_by_molecule(self, molecule_name):
        """
        Get all sample indices for a specific molecule.
        
        Parameters
        ----------
        molecule_name : str
            Name of the molecule
            
        Returns
        -------
        list of int
            Indices of all samples belonging to this molecule
        """
        if molecule_name not in self.molecule_name_to_idx:
            return []
        
        mol_idx = self.molecule_name_to_idx[molecule_name]
        return [i for i, (_, m_idx) in enumerate(self.samples) if m_idx == mol_idx]


# =============================================================================
# Denoising Dataset - Target is clean normalized molecule spectrum
# =============================================================================

class HSI_Denoising_Dataset(Dataset):
    """
    PyTorch Dataset for denoising autoencoder training.
    
    Returns:
    --------
    (noisy_spectrum, clean_spectrum, class_idx)
    - noisy_spectrum: Molecule + background + noise (min-max normalized)
    - clean_spectrum: Pure molecule spectrum (min-max normalized, 0-1 range)
    - class_idx: Integer index of the molecule class
    
    Usage:
    ------
    dataset = HSI_Denoising_Dataset(
        molecule_dataset_path='molecule_dataset/lipid_subtype_CH_61',
        srs_params_path='params_dataset/srs_params_61',
        num_samples_per_class=2000
    )
    
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for noisy, clean, class_idx in loader:
        output = model(noisy)
        loss = criterion(output, clean)
    """
    
    def __init__(self, molecule_dataset_path, srs_params_path,
                 num_samples_per_class=20000, exclude_molecules=None, noise_multiplier=1.0,
                 create_mixtures=False, mixture_pairs=None):
        """
        Initialize denoising dataset.
        
        Parameters
        ----------
        molecule_dataset_path : str
            Path to saved molecule dataset .npz file
        srs_params_path : str
            Path to saved SRS parameters .npz file
        num_samples_per_class : int
            Total synthetic samples per molecule (default 20000)
        exclude_molecules : list of str, optional
            List of molecule names to exclude from the dataset (default None)
        noise_multiplier : float
            Multiplier for noise magnitude (default 1.0). Use 2.0 for double noise, 0.5 for half, 0.0 for no noise
        create_mixtures : bool
            If True, create artificial mixture classes by combining molecule pairs (default False)
        mixture_pairs : list of tuples, optional
            List of (index1, index2) or (name1, name2) pairs to mix. If None and create_mixtures=True,
            creates mixtures for all unique pairs (default None)
        """
        # Add .npz extension if needed
        mol_path = molecule_dataset_path if molecule_dataset_path.endswith('.npz') else molecule_dataset_path + '.npz'
        srs_path = srs_params_path if srs_params_path.endswith('.npz') else srs_params_path + '.npz'
        
        # Load molecule data
        print(f"Loading molecules from: {mol_path}")
        mol_data = np.load(mol_path, allow_pickle=True)
        mol_spectra = mol_data['normalized_molecules']
        molecule_names = mol_data['molecule_names']

        # No Match index
        try: 
            no_match_idx = np.where(molecule_names == 'No Match')[0]
            if len(no_match_idx) > 0:
                no_match_spectra = mol_spectra[no_match_idx]
                mol_spectra = np.delete(mol_spectra, no_match_idx, axis=0)
                molecule_names = np.delete(molecule_names, no_match_idx, axis=0)
        except:
            no_match_idx = []
        
        
        # Filter out excluded molecules
        if exclude_molecules is not None:
            exclude_set = set(exclude_molecules)
            keep_indices = [i for i, name in enumerate(molecule_names) if name not in exclude_set]
            mol_spectra = mol_spectra[keep_indices]
            self.molecule_names = molecule_names[keep_indices]
        else:
            self.molecule_names = molecule_names
        
        # Create mixture classes if requested
        if create_mixtures:
            mol_spectra, self.molecule_names = self._create_mixture_classes(
                mol_spectra, self.molecule_names, mixture_pairs
            )
        # Add No Match back if it was removed
        if len(no_match_idx) > 0:
            mol_spectra = np.vstack([mol_spectra, no_match_spectra])  # Add No Match back at the end
            self.molecule_names = np.concatenate([self.molecule_names, ['No Match']])
        
        # Load SRS parameters
        print(f"Loading SRS parameters from: {srs_path}")
        srs_data = np.load(srs_path)
        bg_scale_vec = srs_data['bg_scale_vec']
        ratio_scale_vec = srs_data['ratio_scale_vec']
        noise_scale_vec = srs_data['noise_scale_vec']
        ch_start = srs_data['ch_start']
        background = srs_data['background']
        
        # Convert to torch tensors
        self.mol_spectra = torch.from_numpy(mol_spectra).float()
        self.bg_scale_vec = bg_scale_vec
        self.ratio_scale_vec = ratio_scale_vec  
        self.noise_scale_vec = noise_scale_vec
        self.background = torch.from_numpy(background.copy()).float()
        self.num_samples_per_class = num_samples_per_class
        self.noise_param = np.mean(noise_scale_vec) 
        self.noise_multiplier = noise_multiplier
        self.ch_start = int(ch_start)
        
        self.n_molecules = mol_spectra.shape[0]
        self.n_wavenumbers = mol_spectra.shape[1]

        print("Denoising Dataset initialized:")
        print(f"  Spectra: {self.n_molecules}")
        print(f"  Wavenumbers: {self.n_wavenumbers}")
        print(f"  Samples per class: {num_samples_per_class}")
        print(f"  Total samples: {len(self)}")
        print("  Noisy normalization: Min-max (baseline as min)")
        print("  Clean normalization: Min-max (0-1)")
    
    def _create_mixture_classes(self, mol_spectra, molecule_names, mixture_pairs=None):
        """
        Create artificial mixture classes by combining pairs of pure molecules.
        
        Parameters
        ----------
        mol_spectra : np.ndarray
            Original molecule spectra (n_molecules, n_wavenumbers)
        molecule_names : np.ndarray
            Original molecule names
        mixture_pairs : list of tuples, optional
            Pairs to mix. Can be indices or names. If None, creates all unique pairs.
        
        Returns
        -------
        combined_spectra : np.ndarray
            Original + mixture spectra
        combined_names : np.ndarray
            Original + mixture names
        """
        n_original = len(mol_spectra)
        
        # Determine which pairs to create
        if mixture_pairs is None:
            # Create all unique pairs
            pairs = []
            for i in range(n_original):
                for j in range(i + 1, n_original):
                    pairs.append((i, j))
            print(f"\n  Creating mixtures for {len(pairs)} unique pairs...")
        else:
            # Convert pairs to indices if they're names
            pairs = []
            name_to_idx = {name: idx for idx, name in enumerate(molecule_names)} # Ignore No Match
            for pair in mixture_pairs:
                if isinstance(pair[0], str):
                    # Names provided
                    idx1 = name_to_idx.get(pair[0])
                    idx2 = name_to_idx.get(pair[1])
                    if idx1 is not None and idx2 is not None:
                        pairs.append((idx1, idx2))
                    else:
                        print(f"  Warning: Skipping pair {pair} - molecule not found")
                else:
                    # Indices provided
                    pairs.append(pair)
            print(f"\n  Creating {len(pairs)} mixture classes...")
        
        # Create mixtures
        mixture_spectra = []
        mixture_names = []
        
        for idx1, idx2 in pairs:
            # Add the two spectra and normalize to [0, 1]
            mixture = mol_spectra[idx1] + mol_spectra[idx2]
            mixture_min = mixture.min()
            mixture_max = mixture.max()
            mixture_normalized = (mixture - mixture_min) / (mixture_max - mixture_min + 1e-8)
            
            mixture_spectra.append(mixture_normalized)
            
            # Create mixture name
            name1 = molecule_names[idx1]
            name2 = molecule_names[idx2]
            mixture_name = f"{name1} + {name2}"
            mixture_names.append(mixture_name)
            
        
        # Combine original and mixtures
        if len(mixture_spectra) > 0:
            mixture_spectra = np.array(mixture_spectra)
            combined_spectra = np.vstack([mol_spectra, mixture_spectra])
            combined_names = np.concatenate([molecule_names, mixture_names])
            
            print(f"  Total molecules (original + mixtures): {len(combined_names)}")
        else:
            combined_spectra = mol_spectra
            combined_names = molecule_names
        
        return combined_spectra, combined_names

    def __len__(self):
        return self.n_molecules * self.num_samples_per_class
    
    def __getitem__(self, idx):
        """
        Generate noisy input and clean target.
        
        Returns
        -------
        noisy_spectrum : torch.Tensor, shape (n_wavenumbers,)
            Molecule + background + noise, min-max normalized using baseline as min
            (same as HSI_Labeled_Dataset with compute_min_max=True)
        clean_spectrum : torch.Tensor, shape (n_wavenumbers,)
            Pure molecule spectrum, min-max normalized to [0, 1]
        mol_idx : int
            Class index of the molecule
        """
        # Determine which molecule
        mol_idx = idx // self.num_samples_per_class
        
        # Sample experimental parameters
        bg_amplitude = np.random.choice(self.bg_scale_vec)
        snr_scale = max(np.random.choice(self.ratio_scale_vec) * self.noise_param, 
                       self.noise_param)
        
        # Generate noise
        noise = torch.randn(self.n_wavenumbers) * self.noise_param * self.noise_multiplier
        
        # Synthesize noisy spectrum
        noisy_spectrum = (self.mol_spectra[mol_idx] * snr_scale + 
                         self.background * bg_amplitude + 
                         noise)
        
        # Get clean molecule spectrum (just the molecule, scaled by SNR)
        clean_spectrum = self.mol_spectra[mol_idx] * snr_scale
        
        # Min-max normalize noisy spectrum (SAME AS HSI_Labeled_Dataset with compute_min_max=True)
        # Use baseline (mean of first ch_start channels) as min, max as max
        noisy_max = torch.max(noisy_spectrum)
        noisy_min = torch.mean(noisy_spectrum[:self.ch_start])  # Baseline
        noisy_normalized = (noisy_spectrum - noisy_min) / (noisy_max - noisy_min + 1e-6)
        
        # Skip No Match class for clean normalization
        if self.molecule_names[mol_idx] == 'No Match':
            clean_normalized = clean_spectrum  # No normalization
        else:
            # Min-max normalize clean spectrum to [0, 1]
            clean_min = torch.min(clean_spectrum)
            clean_max = torch.max(clean_spectrum)
            clean_normalized = (clean_spectrum - clean_min) / (clean_max - clean_min + 1e-8)
        
        return noisy_spectrum, clean_spectrum, mol_idx # Switching back to unnormalized versions for training


def create_denoising_dataloaders(dataset, batch_size=32, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Create train/val/test loaders for denoising dataset.
    
    Parameters
    ----------
    dataset : HSI_Denoising_Dataset
        The denoising dataset
    batch_size : int
        Batch size for dataloaders
    train_ratio : float
        Fraction for training (default 0.7)
    val_ratio : float
        Fraction for validation (default 0.15)
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    """
    from torch.utils.data import DataLoader, Subset
    
    # Calculate split sizes per class
    train_size = int(dataset.num_samples_per_class * train_ratio)
    val_size = int(dataset.num_samples_per_class * val_ratio)
    test_size = dataset.num_samples_per_class - train_size - val_size
    
    print("\nCreating denoising splits:")
    print(f"  Train: {train_size * dataset.n_molecules} samples")
    print(f"  Val:   {val_size * dataset.n_molecules} samples")
    print(f"  Test:  {test_size * dataset.n_molecules} samples")
    
    # Create indices for each split (stratified by molecule class)
    np.random.seed(seed)
    train_indices = []
    val_indices = []
    test_indices = []
    
    for mol_idx in range(dataset.n_molecules):
        class_start = mol_idx * dataset.num_samples_per_class
        class_indices = np.arange(class_start, class_start + dataset.num_samples_per_class)
        
        # Shuffle and split
        np.random.shuffle(class_indices)
        train_indices.extend(class_indices[:train_size])
        val_indices.extend(class_indices[train_size:train_size + val_size])
        test_indices.extend(class_indices[train_size + val_size:])
    
    # Create subsets
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)
    
    # Create loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

