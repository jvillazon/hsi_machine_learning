import os
import numpy as np
from scipy.interpolate import CubicSpline
import pandas as pd
import tifffile
from tqdm import tqdm
import matplotlib.pyplot as plt

def normalize(array, max_val=None, min_val=None, axis=None):
    """
    Normalize array to specified range.
    
    Args:
        array: Input array (1D or 2D)
        max_val: Maximum value for normalized output (computed from data if None)
        min_val: Minimum value for normalized output (computed from data if None)
        axis: For 2D arrays only - axis along which to normalize
              axis=0: normalize each column, axis=1: normalize each row
              axis=None: global normalization
              For 1D arrays, axis parameter is ignored
    
    Returns:
        Normalized array
    """
    # For 1D arrays, ignore axis parameter
    if array.ndim == 1:
        axis = None
    
    # Calculate min/max if not provided
    if axis is None:
        # Global normalization
        if max_val is None:
            max_val = np.max(array)
        if min_val is None:
            min_val = np.min(array)
        
        diff = max_val - min_val
        return (array - min_val) / diff + 1e-6
    else:
        # Axis-based normalization for 2D arrays
        if max_val is None:
            max_val = np.max(array, axis=axis, keepdims=True)
        if min_val is None:
            min_val = np.min(array, axis=axis, keepdims=True)
        
        diff = max_val - min_val
        return (array - min_val) / diff + 1e-6


def create_background(wavenumber_1, wavenumber_2, num_samp, background_df, br_shift):

    """Generate background (PBS) baseline from background .csv file using interpolation
    
    Uses cubic spline interpolation to generate a background spectrum of arbitrary length
    from the original background data, allowing flexible output sizes regardless of the
    input background_df length.
    
    Parameters
    ----------
    wavenumber_1 : int
        Initial Raman wavenumber (lower wavenumber) captured by hyperspectral image (measured in cm^-1).
    wavenumber_2 : int
        Final Raman wavenumber (higher wavenumber) captured by hyperspectral image (measured in cm^-1).
    num_samp : int
        Number of hyperspectral slices (desired output length)
    background_df : Pandas DataFrame
        Dataframe generated from background .csv file corresponding to PBS baseline
    br_shift : int
        Shift parameter for background alignment

    Outputs
    ----------
    background : ndarray of shape (num_samp,)
        1-dimensional array corresponding to the PBS background between the two wavenumbers
    """

    # Target wavenumber points for the output background
    CH_wavenumber = np.linspace(wavenumber_1, wavenumber_2, num_samp - br_shift)
    
    # Extract background intensity values from DataFrame
    temp = background_df[:].to_numpy() 
    temp = temp[:, 0]
    
    # Original wavenumber points (assuming background_df spans the same wavenumber range)
    x_original = np.linspace(wavenumber_1, wavenumber_2, len(temp))
    y_original = np.array(temp)
    
    # Create cubic spline interpolation
    spline = CubicSpline(x_original, y_original)
    
    # Interpolate to target wavenumber points
    background = spline(CH_wavenumber)
    background = normalize(background)
    background = np.flip(background)
    
    # Apply shift if needed
    back_temp = np.zeros(num_samp)
    back_temp[br_shift:] = background
    background = back_temp

    return background


class HSI_Loader:
    def __init__(self, background_df, wavenumber_start=2700, wavenumber_end=3100, num_samples=61, shift=0):
        """
        Initialize HSI Data Loader
        
        Args:
            background_df: pandas DataFrame with background spectrum data
            wavenumber_start: Starting wavenumber for molecule dataset (default 2700)
            wavenumber_end: Ending wavenumber for molecule dataset (default 3100)
            num_samples: Number of samples in wavenumber range (default 61)
            shift: Shift parameter for background (default 0)
        """
        self.wavenumber_start = wavenumber_start
        self.wavenumber_end = wavenumber_end
        self.num_samples = num_samples
        self.ch_start = int((2800 - wavenumber_start) / ((wavenumber_end - wavenumber_start) / (num_samples - 1)))
        self.shift = shift
        self.background_df = background_df
        

    def save_molecule_dataset(self, molecule_file, exclude_molecules=None, output_path=None):
        """
        Class method to process molecule dataset from CSV file.
        
        Args:
            molecule_csv_path: Path to CSV file with molecule spectra
            save_molecule_dataset: If True, save processed data to file (default False)
            output_path: Path where to save (required if save_molecule_dataset=True)
            
        Returns:
            loader: HSI_Loader instance with processed molecules
        """
        if output_path is None:
            raise ValueError("output_path must be provided.")
        
        print(f"Processing molecule dataset from: {molecule_file}")
        
        # Load molecule data (CSV or Excel)
        if molecule_file.endswith('.xlsx') or molecule_file.endswith('.xls'):
            molecule_df = pd.read_excel(molecule_file)
        else:
            molecule_df = pd.read_csv(molecule_file)

        # Remove columns that are all NaN
        molecule_df = molecule_df.dropna(axis='columns', how='all')
        molecule_df_size = molecule_df.columns.shape[0]

        if molecule_df_size % 2 != 0:
            raise ValueError("DataFrame must have pairs of wavenumber-intensity columns")
            
        # Convert to numpy for processing
        molecule_arr = molecule_df.to_numpy()
        num_mol = int(molecule_df_size / 2)
        temp_names = np.array(molecule_df.columns[0:molecule_df_size:2])

        # Initialize arrays
        temp = np.empty((num_mol, self.num_samples), dtype='float32')
        remove_nan = []
        
        # Generate wavenumber points for interpolation
        wavenumber = np.linspace(self.wavenumber_start, self.wavenumber_end, self.num_samples)
        
        # Process each molecule's spectrum
        for i in list(range(0, molecule_df_size, 2)):
            molx = molecule_arr[:, i]  # Wavenumbers
            moly = molecule_arr[:, i + 1]  # Intensities
            
            # Remove NaN values
            valid_mask = ~np.isnan(molx) & ~np.isnan(moly)
            molx = molx[valid_mask]
            moly = moly[valid_mask]
            
            # Check wavenumber coverage
            if molx.min() <= self.wavenumber_start and molx.max() >= self.wavenumber_end:
                temp_spline = CubicSpline(molx, moly, extrapolate=True)
                temp[int(i / 2)] = temp_spline(wavenumber)
            else:
                remove_nan.append(int(i / 2))
                print(f"Warning: Molecule {temp_names[int(i/2)]} doesn't cover required range "
                      f"[{self.wavenumber_start}, {self.wavenumber_end}], skipping")
        
        # Keep only valid molecules
        keep = [i for i in range(temp.shape[0]) if i not in remove_nan]
        if not keep:
            raise ValueError("No valid molecules found in the dataset")
        
        # Remove excluded molecules
        if exclude_molecules is not None:
            keep = [keep[i] for i in range(len(keep)) if temp_names[keep[i]] not in exclude_molecules]  
            print(f"Excluding molecules: {exclude_molecules}")
            
        molecules = temp[keep, :]
        
        # Normalize spectra
        molecules = molecules + 1e-6  # Avoid zero values
        molecules_max = np.max(molecules, axis=1, keepdims=True)
        molecules_min = np.min(molecules, axis=1, keepdims=True)
        mol_norm = (molecules - molecules_min) / (molecules_max - molecules_min + 1e-6)
        mol_norm = mol_norm.astype('float32')
        
        # Apply shift to molecule spectra (shift to the right)
        if self.shift > 0:
            # Create zero-padded array
            mol_shifted = np.zeros((mol_norm.shape[0], self.num_samples), dtype='float32')
            # Place original spectra starting at index 'shift'
            mol_shifted[:, self.shift:] = mol_norm[:, :(self.num_samples - self.shift)]
            mol_norm = mol_shifted
            print(f"Applied shift of {self.shift} indices to molecule spectra (shifted right)")
        
        # Add background spectrum (slight normal variations around 0)
        # background = np.random.normal(loc=0.0, scale=1e-6, size=self.num_samples).astype('float32')

        background = np.zeros(self.num_samples, dtype='float32')
        mol_norm = np.vstack((mol_norm, background))
        mol_names = temp_names[keep]
        mol_names = np.hstack((mol_names, 'No Match'))
        
        print(f"Processed {len(mol_names)-1} molecules (+1 background)")
        print(f"Molecule names: {mol_names[:-1]}")
        
        # Save molecule dataset
        if output_path is not None:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            save_path = output_path if output_path.endswith('.npz') else output_path + '.npz'
            np.savez_compressed(save_path,
                                normalized_molecules=mol_norm,
                                molecule_names=mol_names)
            print(f"Saved to: {save_path}")

    def save_srs_params(self, data_dir, output_path=None):
        """
        Extract SRS spectral parameters from real experimental images for realistic synthetic data generation.
        
        This function processes real SRS hyperspectral images to extract statistical parameters
        that characterize the noise, background, and signal properties. These parameters are then
        used to generate realistic synthetic training data.
        
        Parameters
        ----------
        data_dir : str
            Directory path containing experimental .tif SRS hyperspectral images
            
        Returns
        -------
        image_vec : ndarray of shape (n_spectra, num_samp)
            Filtered and normalized spectra from all images (transposed for compatibility)
        noise_scale_vec : ndarray of shape (n_spectra,)
            Standard deviation of the silent/baseline region for each spectrum (noise estimate)
        bg_scale_vec : ndarray of shape (n_spectra,)
            Background signal amplitude for each spectrum (last channel minus baseline median)
        ratio_scale_vec : ndarray of shape (n_filtered,)
            Signal-to-noise ratios (SNR) for spectra after outlier removal
            Calculated as (max_signal - baseline) / noise_std
            
        Process Steps
        -------------
        1. Load all images from directory and flip vertically
        2. Subtract baseline (mean of silent region per pixel)
        3. Remove invalid values (inf, nan)
        4. Flatten spatial dimensions to create spectrum vectors
        5. Subtract median of silent region from all channels
        6. Normalize each spectrum
        7. Filter: keep only spectra where first channel < last channel (physical constraint)
        8. Filter: keep only spectra where silent region is within ±3σ of mean (noise filter)
        9. Extract parameters:
            - noise_scale_vec: std of silent region
            - bg_scale_vec: last channel - median(silent region)
            - ratio_scale_vec: SNR, with outliers (>3σ) removed
            
        Notes
        -----
        - Silent region is defined as channels 0 to self.ch_start
        - The filtering steps ensure only physically reasonable spectra are used
        - SNR outlier removal ensures synthetic data parameters are robust
        """
        if output_path is None:
            raise ValueError("output_path must be provided.")

        print(f"\nLoading experimental SRS images from: {data_dir}")
        print(f"Silent region: First {self.ch_start} wavenumber")
        
        boolean = True  # Flag for first concatenation
        data_list = os.listdir(data_dir)
        processed_count = 0
        
        for name in tqdm(data_list, desc="Processing images"):
            # Skip hidden files
            if name.startswith("."):
                continue
                
            # Load image
            image = tifffile.imread(data_dir + name)
            image = np.flip(image, axis=0)  # Flip to match coordinate convention
            
            # Baseline subtraction: remove mean of silent region per pixel
            baseline = np.mean(image[:self.ch_start], axis=0)
            image = image - baseline
            
            # Remove invalid values
            image[np.isinf(image)] = 0
            image[np.isnan(image)] = 0
            
            # Reshape: (channels, height, width) -> (channels, height*width)
            image_vector = np.reshape(image, (image.shape[0], image.shape[1] * image.shape[2]))
            
            # Further baseline adjustment: subtract median of silent region
            silent_median = np.median(image_vector[:self.ch_start, :])
            image_vector = image_vector - silent_median
            
            # Concatenate all images
            if boolean:
                temp = image_vector
                boolean = False
            else:
                temp = np.concatenate((temp, image_vector), axis=1)
            
            processed_count += 1
        
        print(f"Processed {processed_count} images, total spectra: {temp.shape[1]}")
        
        # Normalize all spectra
        image_spec = normalize(temp)
        image_spec = image_spec - np.median(image_spec[:self.ch_start], axis=0, keepdims=True)
        print(f"Normalized spectra shape: {image_spec.shape}")

        # Filter 1: Keep spectra where first wavenumber < last wavenumber (physical constraint)
        valid_mask = np.logical_not(image_spec[0, :] > image_spec[-1, :])
        image_spec = image_spec[:, valid_mask]
        print(f"After physical constraint filter: {image_spec.shape[1]} spectra")
        
        # Filter 2: Keep spectra with silent region within ±3σ (noise filter)
        spec_start = image_spec[:self.ch_start]
        mean_silent = np.mean(spec_start)
        std_silent = np.std(spec_start)
        
        # Check each spectrum's silent region
        silent_check = np.all(
            np.logical_and(
                spec_start < mean_silent + 3 * std_silent,
                spec_start > mean_silent - 3 * std_silent
            ),
            axis=0
        )
        image_vec = image_spec[:, silent_check]
        print(f"After noise filter (±3σ): {image_vec.shape[1]} spectra")

        # Normalize
        image_vec = normalize(image_vec)
        
        # Extract parameters for synthetic data generation
        start_vec = image_vec[:self.ch_start]
        
        # Noise scale: std deviation of silent region per spectrum
        noise_scale_vec = np.std(start_vec, axis=0)
        
        # Background scale: last channel - median of silent region
        bg_scale_vec = image_vec[-1] - np.median(start_vec, axis=0)
        bg_scale_vec = np.maximum(bg_scale_vec, 0)  # Ensure non-negative background
        bg_scale_vec = np.minimum(bg_scale_vec, 0.75*np.mean(bg_scale_vec))  # Avoid extreme values
        
        # Signal-to-noise ratio
        max_vec = np.max(image_vec, axis=0)
        ratio_scale_vec = (max_vec - np.median(start_vec, axis=0)) / noise_scale_vec
        
        # Remove SNR outliers (<3σ and >3σ)
        snr_median = np.median(ratio_scale_vec)
        snr_std = np.std(ratio_scale_vec)
        ratio_scale_vec = np.delete(
            ratio_scale_vec,
            np.where(ratio_scale_vec > snr_median + 3 * snr_std)
        )
        ratio_scale_vec = np.delete(
            ratio_scale_vec,
            np.where(ratio_scale_vec < snr_median - 3 * snr_std)
        )
        
        print("Extracted parameters:")
        print(f"  Noise scale: mean={np.mean(noise_scale_vec):.4f}, std={np.std(noise_scale_vec):.4f}")
        print(f"  Background scale: mean={np.mean(bg_scale_vec):.4f}, std={np.std(bg_scale_vec):.4f}")
        print(f"  Background range: [{np.min(bg_scale_vec):.4f}, {np.max(bg_scale_vec):.4f}]")
        print(f"  Peak range: [{np.min(max_vec):.4f}, {np.max(max_vec):.4f}]")
        print(f"  SNR: mean={np.mean(ratio_scale_vec):.2f}, std={np.std(ratio_scale_vec):.2f}")
        print(f"  SNR range: [{np.min(ratio_scale_vec):.2f}, {np.max(ratio_scale_vec):.2f}]")

        # Background shift
        self.background = create_background(self.wavenumber_start, self.wavenumber_end, self.num_samples, self.background_df, br_shift=20)

        # Save parameters to file
        if output_path is not None:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            save_path = output_path if output_path.endswith('.npz') else output_path + '.npz'
            np.savez_compressed(save_path,
                                image_vec=image_vec,
                                noise_scale_vec=noise_scale_vec,
                                bg_scale_vec=bg_scale_vec,
                                ratio_scale_vec=ratio_scale_vec,
                                wavenumber_start=self.wavenumber_start,
                                wavenumber_end=self.wavenumber_end,
                                num_samples=self.num_samples,
                                ch_start=self.ch_start,
                                background=self.background)
            print(f"Saved to: {save_path}")


if __name__ == '__main__':
    """
    Run this script to process and save datasets.
    
    This will:
    1. Process molecule datasets and save to molecule_dataset/
    2. Extract SRS parameters and save to params_dataset/
    
    Edit the configurations below for your specific datasets.
    """
    
    print("=" * 80)
    print("HSI DATA PROCESSING SCRIPT")
    print("=" * 80)
    
    # =========================================================================
    # CONFIGURATION - Edit these for your datasets
    # =========================================================================
    
    # Background file
    BACKGROUND_FILE = 'unprocessed_data/water_HSI_76.csv'
    
    # Datasets to process
    CONFIGS = [
        {
            'molecule_file': 'unprocessed_data/lipid_subtype_reworked.xlsx',
            'experimental_dir': 'training_data/61-2700-3100/',
            'wavenumber_start': 2700,
            'wavenumber_end': 3100,
            'num_samples': 61,
            'shift': 0,
            'molecule_output': 'lipid_subtype_CH_61_0_shift',
            'params_output': 'srs_params_61_0_shift',
            'exclude_molecules': None
        },
        # Add more molecule datasets here as needed
        {
            'molecule_file': 'unprocessed_data/lipid_subtype_reworked.xlsx',
            'experimental_dir': 'training_data/61-2700-3100/',
            'wavenumber_start': 2700,
            'wavenumber_end': 3100,
            'num_samples': 61,
            'shift': 0,
            'molecule_output': 'lipid_subtype_organic_61_0_shift',
            'params_output': 'srs_params_organic_61_0_shift',
            'exclude_molecules': ['Cer(m18:1(4E)/24:1(15Z)', 'Cer(m18:1(4E)/16:0)']  # Add molecules to exclude here

        },
    ]
    
    
    # =========================================================================
    # PROCESS DATA
    # =========================================================================
    
    print("\n" + "=" * 80)
    print("STEP 1: PROCESSING MOLECULE DATASETS")
    print("=" * 80)
    
    background_df = pd.read_csv(BACKGROUND_FILE)
    print(f"Loaded background from: {BACKGROUND_FILE}")
    
    for i, config in enumerate(CONFIGS, 1):
        print(f"\n[{i}/{len(CONFIGS)}]")
        print("-" * 80)
        
        try:
            loader = HSI_Loader(
                background_df=background_df,
                wavenumber_start=config['wavenumber_start'],
                wavenumber_end=config['wavenumber_end'],
                num_samples=config['num_samples'],
                shift=config['shift']
            )

            loader.save_molecule_dataset(
                molecule_file=config['molecule_file'],
                output_path=f"molecule_dataset/{config['molecule_output']}",
                exclude_molecules=config['exclude_molecules']
            )
            print(f"✓ Saved to: molecule_dataset/{config['molecule_output']}.npz")

            loader.save_srs_params(
                config['experimental_dir'],
                output_path=f"params_dataset/{config['params_output']}"
            )
            print(f"✓ Saved to: params_dataset/{config['params_output']}.npz")

        except Exception as e:
            print(f"✗ Failed to process {config['experimental_dir']}: {e}")

    
    # =========================================================================
    # SUMMARY
    # =========================================================================

    print("\nOutput directories:")
    print("  - molecule_dataset/  (molecule spectra)")
    print("  - params_dataset/    (SRS parameters)")
    print("\nYou can now use these saved files in your workflows:")
    print("  - HSI_Loader.load_molecule_dataset('molecule_dataset/...')")
    print("  - HSI_Loader.load_srs_params('params_dataset/...')")
