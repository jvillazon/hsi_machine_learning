import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import pandas as pd

import rampy as rp

from load_spontaneous import load_txt_folder_to_array, normalize_by_value, snv_normalize, crop_and_normalize_region, calculate_spectral_variability

def scan_all_spectra(molecule_dir, start_idx=390, end_idx=3150):
    """
    Scans all spectra in the specified base directory and returns the longest
    wavenumber range and the corresponding wavenumber array.

    Parameters
    ----------
    molecule_dir: list
        List of directories containing subdirectories of spectra files.
    start_idx: int
        The starting wavenumber for cropping.
    end_idx: int
        The ending wavenumber for cropping.
    Returns
    -------
    wavenumbers: np.ndarray
        The wavenumber array corresponding to the longest spectrum.
    """

    max_length = 0
    wavenumbers = None

    for molecule_path in molecule_dir:
        for file in glob.glob(os.path.join(molecule_path, "*.txt")):
            txt_arr = np.loadtxt(file)

            start_crop_idx = np.argmin(np.abs(txt_arr[:, 0] - start_idx))
            end_crop_idx = np.argmin(np.abs(txt_arr[:, 0] - end_idx)) + 1
            cropped_arr = txt_arr[start_crop_idx:end_crop_idx, :]

            if cropped_arr.shape[0] > max_length:
                max_length = cropped_arr.shape[0]
                wavenumbers = cropped_arr[:, 0]

    return wavenumbers


def concatenate_and_save_spectra(data_dirs, start_wavenumber=400, end_wavenumber=3150, output_dir=None):
    """
    Preprocess spectra: scan for longest wavenumber, crop, interpolate, concatenate, and index by molecule name.
    Parameters
    ----------
    data_dirs: list
        List of directories containing spectra files.
    start_wavenumber: int
        Start of wavenumber range.
    end_wavenumber: int
        End of wavenumber range.
    output_dir: str or None
        Directory to save the concatenated spectra CSV. If None, does not save.
    Returns
    -------
    concatenated_spectra: np.ndarray
        2D array of all spectra, interpolated to the longest wavenumber.
    molecule_names: list
        List of molecule names for each spectrum.
    wavenumbers: np.ndarray
        The reference wavenumber array.
    """

    wavenumbers = np.linspace(start_wavenumber, end_wavenumber, num=2751)  # Assuming 1 cm^-1 resolution
    all_spectra = []
    all_names = []
    for data_dir in data_dirs:
        molecule_name = os.path.basename(data_dir).strip()
        files = glob.glob(os.path.join(data_dir, '*.txt'))
        for file in files:
            txt_arr = np.loadtxt(file)
            start_crop_idx = np.argmin(np.abs(txt_arr[:, 0] - start_wavenumber))
            end_crop_idx = np.argmin(np.abs(txt_arr[:, 0] - end_wavenumber)) + 1
            cropped_arr = txt_arr[start_crop_idx:end_crop_idx, :]
            cropped_wn = cropped_arr[:, 0]
            cropped_spectrum = cropped_arr[:, 1]
            interp_spectrum = np.interp(wavenumbers, cropped_wn, cropped_spectrum)
            all_spectra.append(interp_spectrum)
            all_names.append(molecule_name)

    concatenated_spectra = np.vstack(all_spectra)
    if output_dir is None:
        output_dir = os.path.join(data_dirs[0], "../processed_data")

    os.makedirs(output_dir, exist_ok=True)
    raw_df = pd.DataFrame(concatenated_spectra, columns=wavenumbers)
    raw_df.set_index(pd.Index(all_names, name="Molecule"), inplace=True)
    raw_df.to_csv(os.path.join(output_dir, "raw_spectra.csv"))

def test_processing_pipeline(molecule_df, start_wavenumber=400, end_wavenumber=1800):
    """
    Test function to verify the processing pipeline on a subset of spectra.
    Parameters
    ----------
    molecule_df: pd.DataFrame
        DataFrame containing raw spectra indexed by molecule name.
    start_wavenumber: int
        Start of wavenumber range for testing.
    end_wavenumber: int
        End of wavenumber range for testing.
    Returns
    -------
    """
    column_wavenumbers = molecule_df.columns.astype(float)

    start_crop_idx = np.argmin(np.abs(column_wavenumbers - start_wavenumber))
    end_crop_idx = np.argmin(np.abs(column_wavenumbers - end_wavenumber)) + 1
    wavenumbers = np.array(column_wavenumbers[start_crop_idx:end_crop_idx].astype(np.float32))

    # Process the first 5 spectra for testing
    norm_spectrums = np.empty((min(5, len(molecule_df)), len(wavenumbers)))

    for i in range(min(5, len(molecule_df))):
        # Split indices for regions
        region1 = (wavenumbers < 2000)
        region2 = (wavenumbers >= 2000)

        raw_spectrum = molecule_df.iloc[i, start_crop_idx:end_crop_idx].values.astype(np.float32)


        # Apply baseline removal separately
        corrected_spectrum1, baseline1 = rp.baseline(wavenumbers[region1], raw_spectrum[region1], method='drPLS', lam=1e7, eta=0.5, max_iter=100)
        plt.figure(figsize=(10, 4))
        plt.plot(wavenumbers[region1], raw_spectrum[region1], label="Raw Spectrum - Region 1", linestyle='--')
        plt.plot(wavenumbers[region1], baseline1, label="Baseline - Region 1", linestyle=':')
        plt.plot(wavenumbers[region1], corrected_spectrum1, label="Baseline Corrected - Region 1", linestyle='-')
        plt.title(f"Test Spectrum - {molecule_df.index[0]} (Region 1)")
        plt.xlabel("Wavenumber (cm$^{-1}$)")
        plt.ylabel("Intensity")
        plt.legend()
        plt.show()

        corrected_spectrum2, baseline2 = rp.baseline(wavenumbers[region2], raw_spectrum[region2], method='drPLS', lam=1e6, eta=0.5, max_iter=100)
        plt.figure(figsize=(10, 4))
        plt.plot(wavenumbers[region2], raw_spectrum[region2], label="Raw Spectrum - Region 2", linestyle='--')
        plt.plot(wavenumbers[region2], baseline2, label="Baseline - Region 2", linestyle=':')
        plt.plot(wavenumbers[region2], corrected_spectrum2, label="Baseline Corrected - Region 2", linestyle='-')
        plt.title(f"Test Spectrum - {molecule_df.index[0]} (Region 2)")
        plt.xlabel("Wavenumber (cm$^{-1}$)")
        plt.ylabel("Intensity")
        plt.legend()
        plt.show()

        # Calculate offset at the boundary
        offset = corrected_spectrum1[-1] - corrected_spectrum2[0]
        corrected_spectrum2 += offset  # Shift region 2 to match region 1

        # Concatenate
        corrected_spectrum = np.concatenate([corrected_spectrum1, corrected_spectrum2])
        corrected_spectrum = corrected_spectrum.squeeze()  # Ensure it's 1D after baseline correction
        wavenumbers = np.concatenate([wavenumbers[region1], wavenumbers[region2]])
        wavenumbers = wavenumbers.squeeze()  # Ensure wavenumbers is also 1D


        # Normalization (e.g., area normalization)
        area = np.trapz(corrected_spectrum, wavenumbers)
        norm_spectrums[i] = corrected_spectrum / (area + 1e-8)  # Add small value to avoid division by zero

    plt.figure(figsize=(10, 4))
    for i in range(min(5, len(molecule_df))):
        plt.plot(wavenumbers, norm_spectrums[i], label=f"Area Normalized - Spectrum {i+1}", linestyle='--')
    plt.title(f"Test Spectrum - {molecule_df.index[0]} (Area Normalized)")
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.show()

    print("Processing pipeline test complete.")


def load_and_process_spectra(raw_csv_path, normalization_method='area', baseline_removal=True):
    """
    Loads the concatenated spectra from CSV, applies normalization and optional baseline removal.
    Parameters
    ----------
    raw_csv_path: str
        Path to the raw spectra CSV file.
    normalization_method: str
        Normalization method to apply ('minmax' or 'snv').
    baseline_removal: bool
        Whether to apply baseline removal (not implemented in this function).
    Returns
    -------
    processed_spectra: pd.DataFrame
        DataFrame of processed spectra indexed by molecule name.
    """

    raw_df = pd.read_csv(raw_csv_path, index_col='Molecule')
    wavenumbers = np.array(raw_df.columns.astype(np.float32))

    norm_spectra = np.empty_like(raw_df.values, dtype=np.float32)

    for i in range(raw_df.shape[0]):
        raw_spectrum = np.array(raw_df.iloc[i].values.astype(np.float32))

        if baseline_removal:
            # Split indices for regions
            region1 = (wavenumbers < 2000)
            region2 = (wavenumbers >= 2000)

            # Apply baseline removal separately
            corrected_spectrum1, baseline1 = rp.baseline(wavenumbers[region1], raw_spectrum[region1], method='drPLS', lam=1e7, eta=0.5, max_iter=100)
            corrected_spectrum2, baseline2 = rp.baseline(wavenumbers[region2], raw_spectrum[region2], method='drPLS', lam=1e6, eta=0.5, max_iter=100)

            # Calculate offset at the boundary
            offset = corrected_spectrum1[-1] - corrected_spectrum2[0]
            corrected_spectrum2 += offset  # Shift region 2 to match region 1

            # Concatenate
            corrected_spectrum = np.concatenate([corrected_spectrum1, corrected_spectrum2])
            corrected_spectrum = corrected_spectrum.squeeze()  # Ensure it's 1D after baseline correction
        else:
            corrected_spectrum = raw_spectrum

        if normalization_method == 'minmax':
            norm_spectra[i] = (corrected_spectrum - np.min(corrected_spectrum)) / (np.max(corrected_spectrum) - np.min(corrected_spectrum) + 1e-8)
        elif normalization_method == 'snv':
            norm_spectra[i] = (corrected_spectrum - np.mean(corrected_spectrum)) / (np.std(corrected_spectrum) + 1e-8)
        elif normalization_method == 'area':
            area = np.trapezoid(corrected_spectrum, wavenumbers)
            norm_spectra[i] = corrected_spectrum / (area + 1e-8)  # Add small value to avoid division by zero
        else:
            raise ValueError("Unsupported normalization method. Use 'minmax' or 'snv'.")
    norm_df = pd.DataFrame(norm_spectra, columns=raw_df.columns, index=raw_df.index)

    norm_df.to_csv(raw_csv_path.replace("raw_spectra.csv", "processed_spectra.csv"))



def main():
    # Loop through all spectra and metadata
    base_directory = r"/Volumes/ADATA SE880/Molecule Identification/data (not finished)"
    data_dirs = glob.glob(os.path.join(base_directory, "*"))

    output_dir = r"/Users/jorgevillazon/Documents/GitHub/hsi_machine_learning/spontaneous/datasets"
    
    if not os.path.exists(os.path.join(output_dir, "raw_spectra.csv")): 
        print("Concatenating spectra and saving to CSV...")
        concatenate_and_save_spectra(data_dirs, start_wavenumber=400, end_wavenumber=3150, output_dir=output_dir)

    if not os.path.exists(os.path.join(output_dir, "processed_spectra.csv")):
        print("Loading and processing spectra...")
        load_and_process_spectra(os.path.join(output_dir, "raw_spectra.csv"), normalization_method='area', baseline_removal=True)
    
    print("Preprocessing complete. Concatenated spectra saved to:", os.path.join(output_dir, "raw_spectra.csv"))

if __name__ == "__main__":
    main()