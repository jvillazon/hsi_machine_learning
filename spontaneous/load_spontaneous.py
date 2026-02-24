
import os
import glob
import numpy as np

import rampy 

import matplotlib.pyplot as plt



def baseline_removal(wavenumbers, spectrum, display= True, lam=1e5, eta=0.5, ratio=1e-2, niter=20):
    """
    Apply Doubly Reweighted Asymmetric Least Squares (drPLS) baseline correction to a spectrum.
    
    Parameters:
    -----------
    spectrum : np.ndarray
        1D array of intensity values for the spectrum
    wavenumbers : np.ndarray
        1D array of wavenumber values corresponding to the spectrum
    lam : float
        Smoothness parameter (lambda) for the baseline. Higher values result in a smoother baseline.
    eta : float
        Robustness parameter for the drPLS algorithm. Higher values make the algorithm more robust to outliers.
    ratio : float
        Ratio parameter for the drPLS algorithm.
    niter : int
        Number of iterations for the drPLS algorithm. More iterations can improve baseline estimation but increase computation time.

    Returns:
    --------
    np.ndarray : Baseline-corrected spectrum
    """
    corrected_spectrum, baseline = rampy.baseline(wavenumbers, spectrum, method='drPLS', lam=lam, eta=eta, ratio=ratio, niter=niter)
    corrected_spectrum = corrected_spectrum.squeeze()
    baseline = baseline.squeeze()   

    if display:
        plt.figure(figsize=(10, 6))
        plt.plot(corrected_spectrum, label='Corrected Spectrum')
        plt.plot(spectrum, label='Original Spectrum')
        plt.plot(baseline, label='Baseline')
        plt.legend()
        plt.xlabel('Wavenumber (cm$^{-1}$)')
        plt.ylabel('Intensity (A.U.)')
        plt.title('Baseline Correction using drPLS')
        plt.show()

    
    return corrected_spectrum, baseline

def calculate_peak_ratios(wavenumbers, spectra, peak_pairs):
    """
    Calculate ratios of specified peaks for all spectra.
    
    Parameters:
    -----------
    wavenumbers : np.ndarray
        Array of wavenumber values (1D)
    spectra : np.ndarray
        2D array where each column is a spectrum (wavenumbers x n_spectra)
    peak_pairs : list of tuples
        List of tuples [(wn1, wn2), ...] specifying which peaks to ratio (wn1/wn2)
        Example: [(2850, 2930), (2880, 2960)]
    
    Returns:
    --------
    dict : Dictionary mapping peak pair to array of ratios for each spectrum
        Example: {(2850, 2930): [ratio_spec1, ratio_spec2, ...], ...}
    """
    ratios = {}
    for wn1, wn2 in peak_pairs:
        idx1 = np.argmin(np.abs(wavenumbers - wn1))
        idx2 = np.argmin(np.abs(wavenumbers - wn2))
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = spectra[idx1, :] / spectra[idx2, :]
            ratio = np.where(np.isfinite(ratio), ratio, np.nan)
        ratios[(wn1, wn2)] = ratio
    return ratios

def normalize_by_value(array, min_value=np.array([]), max_value=np.array([]), axis=1):

    if len(array.shape) == 2:
        if axis != 1:
            array = array.T
            transpose = True
        else:
            transpose = False
        if min_value.size == 0:
            min_value = np.min(array, 1)
        if max_value.size == 0:
            max_value = np.max(array, 1)
        diff = max_value - min_value
        if np.any(diff < 0):
            raise Exception("min_value > max_value")
        norm = (array - min_value[:, np.newaxis]) / diff[:, np.newaxis]

        if transpose:
            norm = norm.T

    elif len(array.shape) == 1:
        if min_value.size == 0:
            min_value = np.min(array)
        if max_value.size == 0:
            max_value = np.max(array)
        diff = max_value - min_value
        if diff < 0:
            raise Exception("min_value > max_value")

        norm = (array - min_value) / (diff)

    else:
        raise Exception ("Array shape is not compatible")

    return norm


def snv_normalize(spectra, axis=0):
    """
    Standard Normal Variate (SNV) normalization.
    
    Centers each spectrum to zero mean and unit standard deviation.
    Useful for removing multiplicative scatter effects in spectroscopy.
    
    Parameters:
    -----------
    spectra : np.ndarray
        Array of spectra. If 2D, each column (axis=0) or row (axis=1) is a spectrum.
    axis : int
        Axis along which to normalize (0 for column-wise, 1 for row-wise)
    
    Returns:
    --------
    np.ndarray : SNV-normalized spectra
    """
    if len(spectra.shape) == 1:
        # Single spectrum
        mean = np.mean(spectra)
        std = np.std(spectra, ddof=1)
        if std == 0:
            return spectra - mean
        return (spectra - mean) / std
    
    elif len(spectra.shape) == 2:
        # Multiple spectra
        mean = np.mean(spectra, axis=axis, keepdims=True)
        std = np.std(spectra, axis=axis, ddof=1, keepdims=True)
        # Avoid division by zero
        std[std == 0] = 1
        return (spectra - mean) / std
    
    else:
        raise ValueError("Input must be 1D or 2D array")


def load_txt_folder_to_array(folder_dir, skip_first_column=True, wavenumbers=None, remove_spectra=[]):
    """
    Loads all .txt files in a folder into a single numpy array.
    Assumes each .txt file has the same number of rows (wavenumbers).
    If shapes differ, interpolates to match the first file's wavenumber grid.
    
    Parameters:
    -----------
    folder_dir: str
        Directory containing .txt files
    skip_first_column: bool
        If True, skips the first column (wavenumbers) when concatenating spectra
    wavenumbers: np.ndarray or None
        If provided, interpolates all spectra to this wavenumber grid
    remove_spectra: list
        List of column indices to remove from the final array
    Returns:
    --------
    np.ndarray : Combined array of wavenumbers and spectra
    spectra_dict : dict
        Dictionary of spectra metadata

    """

    files = glob.glob(os.path.join(folder_dir, '*.txt'))
    # Find the file with the most rows (longest wavenumber grid)
    max_len = 0
    ref_file = None
    if wavenumbers is not None:
        ref_wavenumbers = wavenumbers
        arr= ref_wavenumbers[:, np.newaxis]  # Make wavenumbers first column
    else:
        for file in files:
            with open(file, 'r') as f:
                n_lines = sum(1 for _ in f)
            if n_lines > max_len:
                max_len = n_lines
                ref_file = file

        # Load the reference file (longest)
        ref_arr = np.loadtxt(ref_file)
        print(f"Using reference file for wavenumber grid: {os.path.basename(ref_file)}")
        ref_wavenumbers = ref_arr[:, 0]
        arr = ref_wavenumbers[:, np.newaxis]  # Start with wavenumbers as first column

    spectra_dict = {
        "name" : [],
        "length": [],
    }
    # Load and interpolate all files to the reference wavenumber grid
    for idx, file in enumerate(files):
        print(f"Spectra {idx+1}: {os.path.basename(file)}")
        spectra_dict["name"].append(os.path.basename(file))
        temp = np.loadtxt(file)
        temp_wavenumbers = temp[:, 0]
        
        # # Apply baseline removal to the spectra
        # blr_spectra, baseline = baseline_removal(temp_wavenumbers, temp[:,1], display=False, lam=1e5, eta=0.5, ratio=1e-2, niter=20)
        # temp[:, 1] = blr_spectra  # Replace spectra with baseline-corrected spectra
        

        spectra_dict["length"].append(len(temp_wavenumbers))
        
        if temp.shape[0] != arr.shape[0]:
            # Interpolate each spectrum column (skip first column which is wavenumbers)
            temp_interpolated = np.zeros((arr.shape[0], temp.shape[1]))
            temp_interpolated[:, 0] = ref_wavenumbers
            temp_interpolated[:, 1] = np.interp(ref_wavenumbers, temp_wavenumbers, temp[:, 1])
            temp = temp_interpolated  # already included as arr
            
        if skip_first_column:
            arr = np.concatenate((arr, temp[:, 1:]), axis=1)
        else:
            arr = np.concatenate((arr, temp), axis=1)

    if len(remove_spectra) > 0:
        arr = np.delete(arr, remove_spectra, axis=1)
    return arr, spectra_dict


def crop_and_normalize_region(txt_arr,  spectra_dict, start_wavenumber, end_wavenumber, 
                              region_name="Region", plot=False, 
                              peak_labels=None, normalization='minmax',):
    """
    Crop and normalize spectra for a specific wavenumber region.
    
    Parameters:
    -----------
    txt_arr : np.ndarray
        Array with wavenumbers in first column and spectra in subsequent columns
    start_wavenumber : float
        Starting wavenumber for the region
    end_wavenumber : float
        Ending wavenumber for the region
    region_name : str
        Name of the region (e.g., "CH Stretching", "Fingerprint", "CD")
    plot : bool
        Whether to plot the normalized spectra
    peak_labels : dict
        Dictionary mapping wavenumber values (float, str, or tuple/list for range) to peak labels
        Example: {2930: 'CH3', (2850,2870): 'CH2 band', 1655: 'Amide I'}
    normalization : str
        Type of normalization: 'minmax' (default) or 'snv' (Standard Normal Variate)
    
    Returns:
    --------
    dict : Dictionary containing:
        - 'wavenumbers': wavenumber array for the region
        - 'normalized_spectra': normalized spectra array
    """
    # Find indices for cropping
    first_idx = np.argmin(np.abs(txt_arr[:, 0] - start_wavenumber))
    last_idx = np.argmin(np.abs(txt_arr[:, 0] - end_wavenumber))
    
    # Ensure proper ordering
    if first_idx > last_idx:
        first_idx, last_idx = last_idx, first_idx
    
    # Crop the array
    cropped_arr = txt_arr[first_idx:last_idx, :]
    wavenumbers = cropped_arr[:, 0]
    
    # Normalize spectra
    if normalization == 'snv':
        norm_arr_crop = snv_normalize(cropped_arr[:, 1:], axis=0)
    else:  # default to minmax
        norm_arr_crop = normalize_by_value(cropped_arr[:, 1:], axis=0)
    
    # Plot if requested
    if plot:
        plt.figure(figsize=(10, 6))
        for idx, arr in enumerate(norm_arr_crop.T):
            

            plt.plot(wavenumbers, arr, label=f'Spectra {spectra_dict["name"][idx]}')
        
        # Get current y-axis limits
        y_min, y_max = plt.ylim()
        y_range = y_max - y_min
        
        # Add peak labels if provided (supporting ranges)
        if peak_labels:
            max_label_height = 0
            for key, label in peak_labels.items():
                # Handle range (tuple/list) or single value
                if isinstance(key, (tuple, list)) and len(key) == 2:
                    wn_start, wn_end = float(key[0]), float(key[1])
                    # Only plot if range overlaps region
                    if (start_wavenumber <= wn_end and end_wavenumber >= wn_start) or (end_wavenumber <= wn_end and start_wavenumber >= wn_start):
                        idx_start = np.argmin(np.abs(wavenumbers - wn_start))
                        idx_end = np.argmin(np.abs(wavenumbers - wn_end))
                        idx_min, idx_max = min(idx_start, idx_end), max(idx_start, idx_end)
                        # Get max intensity in this range
                        max_intensity = np.max(norm_arr_crop[idx_min:idx_max+1, :])
                        max_label_height = max(max_label_height, max_intensity)
                        # Shade region
                        plt.axvspan(wn_start, wn_end, color='gray', alpha=0.2, zorder=1)
                else:
                    if not isinstance(key, (tuple, list)):
                        try:
                            wn = float(key)
                        except Exception:
                            continue
                        if start_wavenumber <= wn <= end_wavenumber or end_wavenumber <= wn <= start_wavenumber:
                            wn_idx = np.argmin(np.abs(wavenumbers - wn))
                            max_intensity = np.max(norm_arr_crop[wn_idx, :])
                            max_label_height = max(max_label_height, max_intensity)
                            plt.axvline(x=wn, color='gray', linestyle='--', alpha=0.5, zorder=1)
            # Extend y-axis if labels would be too close to the top
            if max_label_height > y_max - 0.2 * y_range:
                new_y_max = max_label_height + 0.4 * y_range
                plt.ylim(y_min, new_y_max)
                y_max = new_y_max
                y_range = y_max - y_min
            # Add text labels, all horizontal and spaced vertically
            # Dynamically space labels based on proximity
            label_positions = []
            min_spacing = 0.08  # minimum offset
            max_spacing = 0.40  # maximum offset
            for i, (key, label) in enumerate(peak_labels.items()):
                # Get wavenumber position
                if isinstance(key, (tuple, list)) and len(key) == 2:
                    wn_pos = (float(key[0]) + float(key[1])) / 2
                else:
                    wn_pos = float(key)
                label_positions.append(wn_pos)
            # Sort by wavenumber
            sorted_indices = np.argsort(label_positions)
            sorted_positions = np.array(label_positions)[sorted_indices]
            offsets = np.full(len(peak_labels), min_spacing)
            # Increase offset for labels that are close together
            for i in range(1, len(sorted_positions)):
                if abs(sorted_positions[i] - sorted_positions[i-1]) < 30:  # threshold in wavenumber units
                    offsets[sorted_indices[i]] = min(max_spacing, offsets[sorted_indices[i-1]] + min_spacing)
            for i, (key, label) in enumerate(peak_labels.items()):
                offset = offsets[i]
                if isinstance(key, (tuple, list)) and len(key) == 2:
                    wn_start, wn_end = float(key[0]), float(key[1])
                    if (start_wavenumber <= wn_end and end_wavenumber >= wn_start) or (end_wavenumber <= wn_end and start_wavenumber >= wn_start):
                        idx_start = np.argmin(np.abs(wavenumbers - wn_start))
                        idx_end = np.argmin(np.abs(wavenumbers - wn_end))
                        idx_min, idx_max = min(idx_start, idx_end), max(idx_start, idx_end)
                        max_intensity = np.max(norm_arr_crop[idx_min:idx_max+1, :])
                        text_x = (wn_start + wn_end) / 2
                        text_y = max_intensity + offset * y_range
                        plt.text(text_x, text_y, label, rotation=0, verticalalignment='bottom', horizontalalignment='center',
                                 fontsize=11, fontweight='bold', alpha=1.0,
                                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9, linewidth=0.5),
                                 zorder=5, clip_on=True)
                else:
                    if not isinstance(key, (tuple, list)):
                        try:
                            wn = float(key)
                        except Exception:
                            continue
                        if start_wavenumber <= wn <= end_wavenumber or end_wavenumber <= wn <= start_wavenumber:
                            wn_idx = np.argmin(np.abs(wavenumbers - wn))
                            max_intensity = np.max(norm_arr_crop[wn_idx, :])
                            text_y = max_intensity + offset * y_range
                            plt.text(wn, text_y, label, rotation=0, verticalalignment='bottom', horizontalalignment='center',
                                     fontsize=11, fontweight='bold', alpha=1.0,
                                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9, linewidth=0.5),
                                     zorder=5, clip_on=True)
        
        plt.xlabel('Wavenumber (cm$^{-1}$)')
        plt.ylabel('Normalized Intensity (A.U.)')
        plt.title(f"{region_name}")
        plt.legend().set_visible(True)
        plt.tight_layout()
        
        # Save figure to base directory
        safe_filename = region_name.replace(' ', '_').replace('/', '_').replace('\\', '_') + '.png'
        save_path = os.path.join(data_directory, "images", safe_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # print(f"Saved: {save_path}")
        
        plt.show()
    
    return {
        'wavenumbers': wavenumbers,
        'normalized_spectra': norm_arr_crop,
    }


def calculate_spectral_variability(wavenumbers, spectra):
    """
    Calculate mean, standard deviation, and coefficient of variation at each wavenumber.
    
    Parameters:
    -----------
    wavenumbers : np.ndarray
        Array of wavenumber values
    spectra : np.ndarray
        2D array where each column is a spectrum (wavenumbers x n_spectra)
    
    Returns:
    --------
    dict : Dictionary containing:
        - 'mean': mean intensity at each wavenumber
        - 'std': standard deviation at each wavenumber
        - 'cv': coefficient of variation at each wavenumber (std/mean * 100)
        - 'wavenumbers': wavenumber array
    """
    mean_spectrum = np.mean(spectra, axis=1)
    std_spectrum = np.std(spectra, axis=1, ddof=1)
    
    # Calculate coefficient of variation (CV = std/mean * 100%)
    # Avoid division by zero
    cv_spectrum = np.zeros_like(mean_spectrum)
    nonzero_mask = np.abs(mean_spectrum) > 1e-10
    cv_spectrum[nonzero_mask] = (std_spectrum[nonzero_mask] / np.abs(mean_spectrum[nonzero_mask])) * 100
    
    return {
        'wavenumbers': wavenumbers,
        'mean': mean_spectrum,
        'std': std_spectrum,
        'cv': cv_spectrum
    }


def plot_spectral_variability(wavenumbers, mean_spectrum, std_spectrum, cv_spectrum,
                              region_name="Region", peak_labels=None,
                              save_path=None):
    """
    Plot mean spectrum with standard deviation, std plot, and coefficient of variation.
    
    Parameters:
    -----------
    wavenumbers : np.ndarray
        Array of wavenumber values
    mean_spectrum : np.ndarray
        Mean intensity at each wavenumber
    std_spectrum : np.ndarray
        Standard deviation at each wavenumber
    cv_spectrum : np.ndarray
        Coefficient of variation at each wavenumber (in %)
    region_name : str
        Name of the spectral region
    peak_labels : dict
        Dictionary mapping wavenumber values (float, str, or tuple/list for range) to peak labels
    save_path : str
        Directory path to save the figure
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
    
    # Top plot: Mean spectrum with shaded std region
    ax1.plot(wavenumbers, mean_spectrum, 'b-', linewidth=2, label='Mean', zorder=3)
    ax1.fill_between(wavenumbers, 
                     mean_spectrum - std_spectrum, 
                     mean_spectrum + std_spectrum,
                     alpha=0.3, color='blue', label='±1 SD', zorder=2)
    
    # Add peak labels if provided (supporting ranges)
    if peak_labels:
        y_min, y_max = ax1.get_ylim()
        y_range = y_max - y_min
        # Dynamically space labels based on proximity
        label_positions = []
        min_spacing = 0.08
        max_spacing = 0.40
        for i, (key, label) in enumerate(peak_labels.items()):
            if isinstance(key, (tuple, list)) and len(key) == 2:
                wn_pos = (float(key[0]) + float(key[1])) / 2
            else:
                wn_pos = float(key)
            label_positions.append(wn_pos)
        sorted_indices = np.argsort(label_positions)
        sorted_positions = np.array(label_positions)[sorted_indices]
        offsets = np.full(len(peak_labels), min_spacing)
        for i in range(1, len(sorted_positions)):
            if abs(sorted_positions[i] - sorted_positions[i-1]) < 30:
                offsets[sorted_indices[i]] = min(max_spacing, offsets[sorted_indices[i-1]] + min_spacing)
        for i, (key, label) in enumerate(peak_labels.items()):
            offset = offsets[i]
            if isinstance(key, (tuple, list)) and len(key) == 2:
                wn_start, wn_end = float(key[0]), float(key[1])
                if (wavenumbers[0] <= wn_end and wavenumbers[-1] >= wn_start) or (wavenumbers[-1] <= wn_end and wavenumbers[0] >= wn_start):
                    idx_start = np.argmin(np.abs(wavenumbers - wn_start))
                    idx_end = np.argmin(np.abs(wavenumbers - wn_end))
                    idx_min, idx_max = min(idx_start, idx_end), max(idx_start, idx_end)
                    peak_height = np.max(mean_spectrum[idx_min:idx_max+1])
                    ax1.axvspan(wn_start, wn_end, color='gray', alpha=0.2, zorder=1)
                    text_x = (wn_start + wn_end) / 2
                    text_y = peak_height + offset * y_range
                    ax1.text(text_x, text_y, label, rotation=0, verticalalignment='bottom', horizontalalignment='center',
                             fontsize=9, fontweight='bold', alpha=0.8,
                             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.7, linewidth=0.5),
                             zorder=4)
            else:
                if not isinstance(key, (tuple, list)):
                    try:
                        wn = float(key)
                    except Exception:
                        continue
                    if wavenumbers[0] <= wn <= wavenumbers[-1] or wavenumbers[-1] <= wn <= wavenumbers[0]:
                        wn_idx = np.argmin(np.abs(wavenumbers - wn))
                        peak_height = mean_spectrum[wn_idx]
                        ax1.axvline(x=wn, color='gray', linestyle='--', alpha=0.5, linewidth=1, zorder=1)
                        ax1.text(wn, peak_height + offset * y_range, label,
                                rotation=0, verticalalignment='bottom', horizontalalignment='center',
                                fontsize=9, fontweight='bold', alpha=0.8,
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.7, linewidth=0.5),
                                zorder=4)
    
    ax1.set_ylabel('Normalized Intensity (A.U.)', fontsize=12, fontweight='bold')
    ax1.set_title(f"{region_name} - Mean ± Std Dev", fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Middle plot: Standard deviation
    ax2.plot(wavenumbers, std_spectrum, 'r-', linewidth=2, label='Std Dev')
    ax2.fill_between(wavenumbers, 0, std_spectrum, alpha=0.3, color='red')
    
    # Mark peaks with high variability
    high_var_threshold = np.mean(std_spectrum) + np.std(std_spectrum)
    high_var_indices = std_spectrum > high_var_threshold
    if np.any(high_var_indices):
        ax2.scatter(wavenumbers[high_var_indices], std_spectrum[high_var_indices],
                   color='darkred', s=30, alpha=0.6, zorder=3,
                   label=f'High variability (>{high_var_threshold:.3f})')
    
    ax2.set_ylabel('Standard Deviation', fontsize=12, fontweight='bold')
    ax2.set_title(f"{region_name} - Spectral Variability (Std Dev)", fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Bottom plot: Coefficient of Variation
    ax3.plot(wavenumbers, cv_spectrum, 'g-', linewidth=2, label='CV (%)')
    ax3.fill_between(wavenumbers, 0, cv_spectrum, alpha=0.3, color='green')
    
    # Mark peaks with high CV
    high_cv_threshold = np.mean(cv_spectrum) + np.std(cv_spectrum)
    high_cv_indices = cv_spectrum > high_cv_threshold
    if np.any(high_cv_indices):
        ax3.scatter(wavenumbers[high_cv_indices], cv_spectrum[high_cv_indices],
                   color='darkgreen', s=30, alpha=0.6, zorder=3,
                   label=f'High CV (>{high_cv_threshold:.1f}%)')
    
    ax3.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Coefficient of Variation (%)', fontsize=12, fontweight='bold')
    ax3.set_title(f"{region_name} - Coefficient of Variation", fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        safe_filename = region_name.replace(' ', '_').replace('/', '_').replace('\\', '_') + '_variability.png'
        full_path = os.path.join(save_path, safe_filename)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        # print(f"Saved variability plot: {full_path}")
    
    plt.show()


def save_variability_data(wavenumbers, mean_spectrum, std_spectrum, cv_spectrum,
                          region_name, save_dir):
    """
    Save mean, standard deviation, and coefficient of variation arrays to text files.
    
    Parameters:
    -----------
    wavenumbers : np.ndarray
        Array of wavenumber values
    mean_spectrum : np.ndarray
        Mean intensity at each wavenumber
    std_spectrum : np.ndarray
        Standard deviation at each wavenumber
    cv_spectrum : np.ndarray
        Coefficient of variation at each wavenumber (in %)
    region_name : str
        Name of the spectral region
    save_dir : str
        Directory path to save the data files
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Combine data into a single array
    data = np.column_stack((wavenumbers, mean_spectrum, std_spectrum, cv_spectrum))
    
    # Create safe filename
    safe_filename = region_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
    
    # Save to text file
    filepath = os.path.join(save_dir, f"{safe_filename}_variability_data.txt")
    np.savetxt(filepath, data, 
               header='Wavenumber(cm-1)\tMean\tStd_Dev\tCV(%)',
               delimiter='\t', fmt='%.6f', comments='')
    # print(f"Saved variability data: {filepath}")
    
    return filepath


def main():

    base_directory = r"/Volumes/ADATA SE880/Molecule Identification/data (not finished)"
    data_directories = glob.glob(os.path.join(base_directory, "*"))

    for data_directory in data_directories:
        molecule = os.path.basename(data_directory).strip()
        print(f"Reading from {data_directory}...")
        # for idx, glob_file in enumerate(glob.glob(os.path.join(data_directory, '*.txt'))):
        #     print(f"Found Spectra {idx+1}: {os.path.basename(glob_file)}")
        txt_arr, spectra_dict = load_txt_folder_to_array(data_directory, remove_spectra=[])
        os.makedirs(os.path.join(data_directory, "images"), exist_ok=True)
        plot = False # Set to True to enable plotting of spectra and variability
        normalization = 'minmax'
        
        # Choose normalization method: 'minmax' or 'snv'
        full_spectrum_normalization = 'minmax'  # Change to 'minmax' for min-max normalization
        
        # Define peak labels for different regions
        ch_peaks = {
            2845: 'CH2 sym',
            2880: 'CH3 sym', 
            2960: 'CH3 asym',
            2930: 'CH2 asym',
        }
        
        fingerprint_peaks = {
            1670: 'Amide I',
            1640: 'C=C stretch',
            (1445, 1475): 'CH3 deform',
            (1370, 1385): 'CH3 sym deform',
            1295: 'CH2 twist',
            # 1120: 'C-O-C ring',
            (950, 1150): 'C-C stretch (glucose)',
            885: 'C-C skeletal stretch'
        }
        
        cd_peaks = {
            2135: 'CD lipid',
            2185: 'CD protein'
        }
        
        # Process CH stretching region (2700-3100 cm⁻¹)
        print("Processing CH stretching region...")
        ch_region = crop_and_normalize_region(
            txt_arr, 
            spectra_dict,
            start_wavenumber=2700, 
            end_wavenumber=3100,
            region_name=f"{molecule} CH Stretching Region",
            plot=plot,
            peak_labels=ch_peaks,
            normalization=normalization
        )
        
        # Calculate and visualize variability for CH region
        print("Analyzing CH stretching variability...")
        ch_variability = calculate_spectral_variability(
            ch_region['wavenumbers'],
            ch_region['normalized_spectra']
        )
        if plot:
            plot_spectral_variability(
                ch_variability['wavenumbers'],
                ch_variability['mean'],
                ch_variability['std'],
                ch_variability['cv'],
                region_name=f"{molecule} CH Stretching",
                peak_labels=ch_peaks,
                save_path=os.path.join(data_directory, "images")
            )
        save_variability_data(
            ch_variability['wavenumbers'],
            ch_variability['mean'],
            ch_variability['std'],
            ch_variability['cv'],
            region_name=f"{molecule}_CH_Stretching",
            save_dir=os.path.join(data_directory, "variability_data")
        )
        
        # Process fingerprint region (400-1800 cm⁻¹)
        print("Processing fingerprint region...")
        fingerprint_region = crop_and_normalize_region(
            txt_arr,
            spectra_dict,
            start_wavenumber=400,
            end_wavenumber=1800,
            region_name=f"{molecule} Fingerprint Region",
            plot=plot,
            peak_labels=fingerprint_peaks,
            normalization=normalization
        )
        
        # Calculate and visualize variability for fingerprint region
        print("Analyzing fingerprint variability...")
        fp_variability = calculate_spectral_variability(
            fingerprint_region['wavenumbers'],
            fingerprint_region['normalized_spectra']
        )

        if plot:
            plot_spectral_variability(
                fp_variability['wavenumbers'],
                fp_variability['mean'],
                fp_variability['std'],
                fp_variability['cv'],
                region_name=f"{molecule} Fingerprint",
                peak_labels=fingerprint_peaks,
                save_path=os.path.join(data_directory, "images")
            )
        save_variability_data(
            fp_variability['wavenumbers'],
            fp_variability['mean'],
            fp_variability['std'],
            fp_variability['cv'],
            region_name=f"{molecule}_Fingerprint",
            save_dir=os.path.join(data_directory, "variability_data")
        )
        
        # Process CD region (2000-2410 cm⁻¹)
        print("Processing CD region...")
        cd_region = crop_and_normalize_region(
            txt_arr,
            spectra_dict,
            start_wavenumber=1940,
            end_wavenumber=2410,
            region_name=f"{molecule} CD Region",
            plot=plot,
            peak_labels=cd_peaks,
            normalization=normalization
        )
        
        # Calculate and visualize variability for CD region
        print("Analyzing CD variability...")
        cd_variability = calculate_spectral_variability(
            cd_region['wavenumbers'],
            cd_region['normalized_spectra']
        )

        if plot:
            plot_spectral_variability(
                cd_variability['wavenumbers'],
                cd_variability['mean'],
                cd_variability['std'],
                cd_variability['cv'],
                region_name=f"{molecule} CD",
                peak_labels=cd_peaks,
                save_path=os.path.join(data_directory, "images")
            )
        save_variability_data(
            cd_variability['wavenumbers'],
            cd_variability['mean'],
            cd_variability['std'],
            cd_variability['cv'],
            region_name=f"{molecule}_CD",
            save_dir=os.path.join(data_directory, "variability_data")
        )
        
        # Calculate and plot average spectrum
        print("Creating average spectrum...")

        if full_spectrum_normalization == 'snv':
            norm_arr = snv_normalize(txt_arr[:, 1:], axis=0)
        else:  # minmax
            first_idx = np.argmin(np.abs(txt_arr[:, 0] - 2700))
            last_idx = np.argmin(np.abs(txt_arr[:, 0] - 3100))
            norm_arr = normalize_by_value(
                txt_arr[:, 1:],
                min_value=np.min(txt_arr[first_idx:last_idx, 1:], axis=0),
                max_value=np.max(txt_arr[first_idx:last_idx, 1:], axis=0),
                axis=0
            )
        average_spectrum = np.mean(norm_arr, axis=1)
        std_spectrum = np.std(norm_arr, axis=1, ddof=1)


        if plot:
            # Plot all normalized spectra (not cropped)
            print("Plotting all normalized spectra...")
            plt.figure(figsize=(12, 6))
            for idx, arr in enumerate(norm_arr.T):
                plt.plot(txt_arr[:, 0], arr, label=f'Spectra {idx+1}')
            plt.xlabel('Wavenumber (cm$^{-1}$)')
            plt.ylabel('Normalized Intensity (A.U.)')
            plt.title(f"{molecule} All Normalized Spectra (Full Range) - {full_spectrum_normalization.upper()}")
            plt.legend(loc='best')
            plt.tight_layout()
            
            # Save all normalized spectra plot
            all_norm_filename = f"{molecule.replace(' ', '_')}_All_Normalized_Spectra.png"
            all_norm_path = os.path.join(data_directory, "images", all_norm_filename)
            plt.savefig(all_norm_path, dpi=300, bbox_inches='tight')
            # print(f"Saved: {all_norm_path}")
            plt.show()

            # Plot average spectrum with shaded std region
            plt.figure(figsize=(12, 6))
            plt.plot(txt_arr[:, 0], average_spectrum)
            plt.fill_between(txt_arr[:, 0], 
                            average_spectrum - std_spectrum, 
                            average_spectrum + std_spectrum,
                            alpha=0.3, color='blue', label='±1 SD')
            plt.legend()
            plt.xlabel('Wavenumber (cm$^{-1}$)')
            plt.ylabel('Normalized Intensity (A.U.)')
            plt.title(f"{molecule} Average Spectrum (Mean)")
            plt.tight_layout()
            
            # Save average spectrum plot
            avg_filename = f"{molecule.replace(' ', '_')}_Average_Spectrum.png"
            avg_path = os.path.join(data_directory, "images",   avg_filename)
            plt.savefig(avg_path, dpi=300, bbox_inches='tight')
            plt.show()
            
        # Save processed data
        spectrum = np.concatenate((txt_arr[:, 0][:, np.newaxis], average_spectrum[:, np.newaxis]), axis=1)
        os.makedirs(os.path.join(data_directory, "processed_data"), exist_ok=True)
        with open(os.path.join(data_directory, "processed_data", f"{molecule.replace(' ', '_')}_averaged.txt"), "w") as f:
            for row in spectrum:
                f.write(f"{row[0]}\t{row[1]}\n")

        # # Calculate and export peak ratios for CH and fingerprint regions
        # ch_peak_pairs = [(2850, 2930), (2960, 2880), (2880, 2850), (2960, 2850)]
        # fingerprint_peak_pairs = [(1670, 1295), (1640, 885)]

        # ch_ratios = calculate_peak_ratios(
        #     ch_region['wavenumbers'],
        #     ch_region['normalized_spectra'],
        #     ch_peak_pairs
        # )
        # ch_csv_path = os.path.join(data_directory, "processed_data", f"{molecule}_CH_peak_ratios.csv")
        # with open(ch_csv_path, "w") as f:
        #     header = [f"ratio_{p1}_{p2}" for p1, p2 in ch_peak_pairs]
        #     f.write("spectrum," + ",".join(header) + "\n")
        #     n_spectra = ch_region['normalized_spectra'].shape[1]
        #     for i in range(n_spectra):
        #         row = [str(i+1)] + [str(ch_ratios[pair][i]) for pair in ch_peak_pairs]
        #         f.write(",".join(row) + "\n")

        # fp_ratios = calculate_peak_ratios(
        #     fingerprint_region['wavenumbers'],
        #     fingerprint_region['normalized_spectra'],
        #     fingerprint_peak_pairs
        # )
        # fp_csv_path = os.path.join(data_directory, "processed_data", f"{molecule}_Fingerprint_peak_ratios.csv")
        # with open(fp_csv_path, "w") as f:
        #     header = [f"ratio_{p1}_{p2}" for p1, p2 in fingerprint_peak_pairs]
        #     f.write("spectrum," + ",".join(header) + "\n")
        #     n_spectra = fingerprint_region['normalized_spectra'].shape[1]
        #     for i in range(n_spectra):
        #         row = [str(i+1)] + [str(fp_ratios[pair][i]) for pair in fingerprint_peak_pairs]
        #         f.write(",".join(row) + "\n")



        print('Done! Processed regions: CH stretching, Fingerprint, and CD')

if __name__ == "__main__":
    main()


