import os
import glob
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import pybaselines
from pybaselines import whittaker

import matplotlib.pyplot as plt

base_directory = r"E:\Data\Jorge\Vallon Kidney\3_31_26 15-1 kidney"


def baseline_correct(spectrum, wavenumbers, lam=1e3, p=0.0001, niter=100, mode='als', eta=0.0005, show_plot=False):
    """
    Perform baseline correction using pybaselines (Asymmetric Least Squares or drPLS).
    
    Parameters:
    -----------
    spectrum : np.ndarray
        1D array of intensity values corresponding to the spectrum.
    wavenumbers : np.ndarray
        1D array of wavenumber values corresponding to the spectrum.
    lam : float
        Smoothness parameter (default: 1e4). Higher values make the baseline smoother.
    p : float
        Asymmetry parameter for 'als' (default: 0.0001).
    niter : int
        Number of iterations for the algorithm (default: 100).
    mode : str
        Algorithm to use: 'als' or 'drpls' (default: 'als').
    eta : float
        Smoothness parameter for 'drpls' (default: 0.05).
    show_plot : bool
        Whether to plot the results (default: False).
    
    Returns:
    --------
    np.ndarray : Baseline-corrected spectrum
    """
    
    # Ensure 1D to avoid broadcasting issues
    y = np.asanyarray(spectrum).flatten()
    
    if mode == 'als':
        # Asymmetric Least Squares
        baseline, _ = whittaker.asls(y, lam=lam, p=p, max_iter=niter)
    elif mode == 'drpls':
        # Dually Reweighted Penalized Least Squares
        baseline, _ = whittaker.drpls(y, lam=lam, eta=eta, max_iter=niter)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'als' or 'drpls'.")

    corrected_spectrum = y - baseline

    if show_plot:
        plt.figure(figsize=(10, 4))
        plt.plot(wavenumbers, y, label='Original Spectrum', color='blue')    
        plt.plot(wavenumbers, baseline, label='Estimated Baseline', color='red', linestyle='--')
        plt.plot(wavenumbers, corrected_spectrum, label='Baseline-Corrected Spectrum', color='green')
        plt.title(f'Baseline Correction ({mode.upper()})')
        plt.legend()
        plt.show()

    return corrected_spectrum


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


def load_txt_folder_to_array(folder_dir, skip_first_column=True, remove_spectra=[], baseline_correction=None):
    """Load all .txt files in a folder into a single array, optionally skipping the first column and applying baseline correction.
    
    Parameters:
    -----------
    folder_dir : str
        Directory containing .txt files.
    skip_first_column : bool
        Whether to skip the first column (wavenumbers) when loading.
    remove_spectra : list
        List of spectrum indices to remove.
    baseline_correction : callable
        Function to apply baseline correction (default: None).

    Returns:
    --------
    np.ndarray : Combined array of spectra.
    """

    files = glob.glob(os.path.join(folder_dir, '*.txt'))
    arr = np.loadtxt(files[0])

    if baseline_correction is not None:
        # Correct only the first spectrum (which is in column 1, after wavenumbers in column 0)
        # We use flatten to ensure it's 1D, and the corrected version is re-inserted as a column.
        arr[:, 1:] = baseline_correct(arr[:, 1:], arr[:, 0], mode='als', show_plot=True)[:, np.newaxis]
    

    for idx, file in enumerate(files[1:]):
        temp = np.loadtxt(file)
        
        # If shapes don't match, interpolate to match arr's wavenumber grid
        if temp.shape[0] != arr.shape[0]:
            # Extract wavenumber columns (first column)
            arr_wavenumbers = arr[:, 0]
            temp_wavenumbers = temp[:, 0]
            
            # Create interpolated array with arr's wavenumber grid
            temp_interpolated = np.zeros((arr.shape[0], temp.shape[1]))
            temp_interpolated[:, 0] = arr_wavenumbers  # Use arr's wavenumbers
            
            # Interpolate each spectrum column (skip first column which is wavenumbers)
            for col in range(1, temp.shape[1]):
                interp_spectrum = np.interp(arr_wavenumbers, temp_wavenumbers, temp[:, col])
                    
                temp_interpolated[:, col] = interp_spectrum   

            temp = temp_interpolated  # Use the interpolated array for further processing  
        
        if baseline_correction is not None:

            temp_spec = baseline_correct(temp[:, 1:], arr[:, 0], mode='als', show_plot=True)  # Apply baseline correction to spectra (skip wavenumbers)
            temp[:, 1:] = temp_spec[:, np.newaxis]  # Add back the wavenumber column for concatenation

        
        arr = np.concatenate((arr, temp[:, 1:]), axis=1)

    if len(remove_spectra) > 0:
        arr = np.delete(arr, remove_spectra, axis=1)
    return arr


def crop_and_normalize_region(txt_arr, start_wavenumber, end_wavenumber, 
                              region_name="Region", plot=False, 
                              peak_labels=None, normalization='minmax'):
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
        Dictionary mapping wavenumber values (float or str) to peak labels
        Example: {2930: 'CH3', 2850: 'CH2', '1655': 'Amide I'}
    normalization : str
        Type of normalization: 'minmax' (default) or 'snv' (Standard Normal Variate)
    
    Returns:
    --------
    dict : Dictionary containing:
        - 'wavenumbers': wavenumber array for the region
        - 'normalized_spectra': normalized spectra array
        - 'cropped_array': original cropped array (wavenumbers + spectra)
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
            plt.plot(wavenumbers, arr, label=f'Spectra {idx+1}')
        
        # Get current y-axis limits
        y_min, y_max = plt.ylim()
        y_range = y_max - y_min
        
        # Add peak labels if provided
        if peak_labels:
            max_label_height = 0
            for wavenumber, label in peak_labels.items():
                wn = float(wavenumber)
                # Find closest wavenumber in the region
                if start_wavenumber <= wn <= end_wavenumber or end_wavenumber <= wn <= start_wavenumber:
                    wn_idx = np.argmin(np.abs(wavenumbers - wn))
                    # Get max intensity at this wavenumber
                    max_intensity = np.max(norm_arr_crop[wn_idx, :])
                    max_label_height = max(max_label_height, max_intensity)
                    
                    plt.axvline(x=wn, color='gray', linestyle='--', alpha=0.5, zorder=1)
            
            # Extend y-axis if labels would be too close to the top
            # Reserve 20% of the plot height for labels
            if max_label_height > y_max - 0.2 * y_range:
                new_y_max = max_label_height + 0.4 * y_range
                plt.ylim(y_min, new_y_max)
                y_max = new_y_max
                y_range = y_max - y_min
            
            # Now add text labels with updated margins
            for wavenumber, label in peak_labels.items():
                wn = float(wavenumber)
                if start_wavenumber <= wn <= end_wavenumber or end_wavenumber <= wn <= start_wavenumber:
                    wn_idx = np.argmin(np.abs(wavenumbers - wn))
                    max_intensity = np.max(norm_arr_crop[wn_idx, :])
                    # Position text above the highest peak with padding
                    text_y = max_intensity + 0.05 * y_range
                    
                    plt.text(wn, text_y, label, 
                            rotation=90, verticalalignment='bottom', horizontalalignment='center',
                            fontsize=11, fontweight='bold', alpha=1.0,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                     edgecolor='gray', alpha=0.9, linewidth=0.5),
                            zorder=5, clip_on=True)
        
        plt.xlabel('Wavenumber (cm$^{-1}$)')
        plt.ylabel('Normalized Intensity (A.U.)')
        plt.title(f"{region_name}")
        plt.legend().set_visible(True)
        plt.tight_layout()
        
        # Save figure to base directory
        safe_filename = region_name.replace(' ', '_').replace('/', '_').replace('\\', '_') + '.png'
        save_path = os.path.join(base_directory, "images", safe_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        
        plt.show()
    
    return {
        'wavenumbers': wavenumbers,
        'normalized_spectra': norm_arr_crop,
        'cropped_array': cropped_arr
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
        Dictionary mapping wavenumber values to peak labels
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
    
    # Add peak labels if provided
    if peak_labels:
        y_min, y_max = ax1.get_ylim()
        y_range = y_max - y_min
        
        for wavenumber, label in peak_labels.items():
            wn = float(wavenumber)
            if wavenumbers[0] <= wn <= wavenumbers[-1] or wavenumbers[-1] <= wn <= wavenumbers[0]:
                wn_idx = np.argmin(np.abs(wavenumbers - wn))
                peak_height = mean_spectrum[wn_idx]
                
                ax1.axvline(x=wn, color='gray', linestyle='--', alpha=0.5, linewidth=1, zorder=1)
                ax1.text(wn, peak_height + 0.05 * y_range, label,
                        rotation=90, verticalalignment='bottom', horizontalalignment='center',
                        fontsize=9, fontweight='bold', alpha=0.8,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                 edgecolor='gray', alpha=0.7, linewidth=0.5),
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
        print(f"Saved variability plot: {full_path}")
    
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
    print(f"Saved variability data: {filepath}")
    
    return filepath


def main():
    molecule = os.path.basename(base_directory).strip()
    print(f"Reading from {base_directory}...")
    txt_arr = load_txt_folder_to_array(base_directory, remove_spectra=[], baseline_correction=True)
    os.makedirs(os.path.join(base_directory, "images"), exist_ok=True)

    
    # Choose normalization method: 'minmax' or 'snv'
    full_spectrum_normalization = 'snv'  # Change to 'minmax' for min-max normalization
    
    # Define peak labels for different regions
    ch_peaks = {
        2850: 'CH2 sym',
        2880: 'CH3 sym', 
        2930: 'CH3 asym',
        2960: 'CH2 asym'
    }
    
    fingerprint_peaks = {
        1655: 'Amide I',
        1450: 'CH2 bend',
        1300: 'Amide III',
        1080: 'C-O stretch'
    }
    
    cd_peaks = {
        2135: 'CD lipid',
        2185: 'CD protein'
    }
    
    # Process CH stretching region (2700-3100 cm⁻¹)
    print("Processing CH stretching region...")
    ch_region = crop_and_normalize_region(
        txt_arr, 
        start_wavenumber=2700, 
        end_wavenumber=3100,
        region_name=f"{molecule} CH Stretching Region",
        plot=True,
        peak_labels=ch_peaks,
        normalization='snv'
    )
    
    # Calculate and visualize variability for CH region
    print("Analyzing CH stretching variability...")
    ch_variability = calculate_spectral_variability(
        ch_region['wavenumbers'],
        ch_region['normalized_spectra']
    )
    plot_spectral_variability(
        ch_variability['wavenumbers'],
        ch_variability['mean'],
        ch_variability['std'],
        ch_variability['cv'],
        region_name=f"{molecule} CH Stretching",
        peak_labels=ch_peaks,
        save_path=os.path.join(base_directory, "images")
    )
    save_variability_data(
        ch_variability['wavenumbers'],
        ch_variability['mean'],
        ch_variability['std'],
        ch_variability['cv'],
        region_name=f"{molecule}_CH_Stretching",
        save_dir=os.path.join(base_directory, "variability_data")
    )
    
    # Process fingerprint region (400-1800 cm⁻¹)
    print("Processing fingerprint region...")
    fingerprint_region = crop_and_normalize_region(
        txt_arr,
        start_wavenumber=400,
        end_wavenumber=1800,
        region_name=f"{molecule} Fingerprint Region",
        plot=True,
        peak_labels=fingerprint_peaks,
        normalization='snv'
    )
    
    # Calculate and visualize variability for fingerprint region
    print("Analyzing fingerprint variability...")
    fp_variability = calculate_spectral_variability(
        fingerprint_region['wavenumbers'],
        fingerprint_region['normalized_spectra']
    )
    plot_spectral_variability(
        fp_variability['wavenumbers'],
        fp_variability['mean'],
        fp_variability['std'],
        fp_variability['cv'],
        region_name=f"{molecule} Fingerprint",
        peak_labels=fingerprint_peaks,
        save_path=os.path.join(base_directory, "images")
    )
    save_variability_data(
        fp_variability['wavenumbers'],
        fp_variability['mean'],
        fp_variability['std'],
        fp_variability['cv'],
        region_name=f"{molecule}_Fingerprint",
        save_dir=os.path.join(base_directory, "variability_data")
    )
    
    # Process CD region (2000-2410 cm⁻¹)
    print("Processing CD region...")
    cd_region = crop_and_normalize_region(
        txt_arr,
        start_wavenumber=1940,
        end_wavenumber=2410,
        region_name=f"{molecule} CD Region",
        plot=True,
        peak_labels=cd_peaks,
        normalization='snv'
    )
    
    # Calculate and visualize variability for CD region
    print("Analyzing CD variability...")
    cd_variability = calculate_spectral_variability(
        cd_region['wavenumbers'],
        cd_region['normalized_spectra']
    )
    plot_spectral_variability(
        cd_variability['wavenumbers'],
        cd_variability['mean'],
        cd_variability['std'],
        cd_variability['cv'],
        region_name=f"{molecule} CD",
        peak_labels=cd_peaks,
        save_path=os.path.join(base_directory, "images")
    )
    save_variability_data(
        cd_variability['wavenumbers'],
        cd_variability['mean'],
        cd_variability['std'],
        cd_variability['cv'],
        region_name=f"{molecule}_CD",
        save_dir=os.path.join(base_directory, "variability_data")
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
    all_norm_path = os.path.join(base_directory, "images", all_norm_filename)
    plt.savefig(all_norm_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {all_norm_path}")
    plt.show()
    
    average_spectrum = np.median(norm_arr, axis=1)
    plt.figure(figsize=(12, 6))
    plt.plot(txt_arr[:, 0], average_spectrum)
    plt.xlabel('Wavenumber (cm$^{-1}$)')
    plt.ylabel('Normalized Intensity (A.U.)')
    plt.title(f"{molecule} Average Spectrum")
    plt.tight_layout()
    
    # Save average spectrum plot
    avg_filename = f"{molecule.replace(' ', '_')}_Average_Spectrum.png"
    avg_path = os.path.join(base_directory, "images",   avg_filename)
    plt.savefig(avg_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {avg_path}")
    
    plt.show()
    
    # Save processed data
    spectrum = np.concatenate((txt_arr[:, 0][:, np.newaxis], average_spectrum[:, np.newaxis]), axis=1)
    os.makedirs(os.path.join(base_directory, "processed_data"), exist_ok=True)
    with open(os.path.join(base_directory, "processed_data", f"{molecule.replace(' ', '_')}_averaged.txt"), "w") as f:
        for row in spectrum:
            f.write(f"{row[0]}\t{row[1]}\n")

    print('Done! Processed regions: CH stretching, Fingerprint, and CD')

if __name__ == "__main__":
    main()


