import os
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt

from load_spontaneous import load_txt_folder_to_array, normalize_by_value, snv_normalize, crop_and_normalize_region

def load_all_spectra(base_directory, normalization="snv"):
    """
    Loads and processes all spectra from the specified base directory.

    Parameters
    ----------
    base_directory: str
        The base directory containing subdirectories of spectra files.
    Returns
    -------
    molecule_dict: dict
        A dictionary where each key is a molecule name and the value is another 
        dictionary with processed (cropped and normalized) 'fingerprint' and 'ch' spectra.
    """

    molecule_dir = glob.glob(base_directory)

    molecule_dict = {}

    for molecule_path in molecule_dir:
        molecule = os.path.basename(molecule_path).strip()
        print(f"Processing {molecule}...")
        txt_arr = load_txt_folder_to_array(molecule_path, remove_spectra=[])

        # Normalize spectra
        fingerprint_dict = crop_and_normalize_region(
            txt_arr,
            start_wavenumber=400,
            end_wavenumber=1800,
            plot=False,
            peak_labels=None,
            normalization=normalization
        )

        ch_dict = crop_and_normalize_region(
            txt_arr,
            start_wavenumber=2800,
            end_wavenumber=3100,
            plot=False,
            peak_labels=None,
            normalization=normalization
        )

        molecule_dict[molecule] = {
            "fingerprint": fingerprint_dict,
            "ch": ch_dict
        }

    return molecule_dict

def main():
    base_directory = r"/Volumes/ADATA SE880/20260120 Jorge's sample/*"
    molecule_spectra = load_all_spectra(base_directory, normalization="snv")

    # Directory to save plots
    save_dir = os.path.join(os.path.dirname(base_directory.split('*')[0]), "images")
    os.makedirs(save_dir, exist_ok=True)

    def detect_peaks(wavenumbers, mean, std, cv, window=5, std_thresh=1.0, mean_thresh=0.7):
        """
        Detect peak regions based on high mean intensity and/or high std/cv.
        Returns a dict of wavenumber: label for plotting.
        """
        peaks = {}
        # Normalize mean and std for thresholding
        mean_norm = (mean - np.min(mean)) / (np.max(mean) - np.min(mean))
        std_norm = (std - np.min(std)) / (np.max(std) - np.min(std))
        cv_norm = (cv - np.min(cv)) / (np.max(cv) - np.min(cv))
        # Find indices where mean or std/cv is above threshold
        for i in range(window, len(wavenumbers)-window):
            if mean_norm[i] > mean_thresh or std_norm[i] > std_thresh or cv_norm[i] > std_thresh:
                wn = wavenumbers[i]
                label = f"Peak @ {int(wn)}"
                peaks[wn] = label
        return peaks

    for molecule, spectra_dict in molecule_spectra.items():
        print(f"Analyzing {molecule}...")
        # Fingerprint region
        fp = spectra_dict["fingerprint"]
        fp_var = calculate_spectral_variability(fp['wavenumbers'], fp['normalized_spectra'])
        fp_peaks = detect_peaks(fp_var['wavenumbers'], fp_var['mean'], fp_var['std'], fp_var['cv'], window=5, std_thresh=0.7, mean_thresh=0.7)
        plot_spectral_variability(
            fp_var['wavenumbers'], fp_var['mean'], fp_var['std'], fp_var['cv'],
            region_name=f"{molecule} Fingerprint", peak_labels=fp_peaks, save_path=save_dir
        )

        # CH region
        ch = spectra_dict["ch"]
        ch_var = calculate_spectral_variability(ch['wavenumbers'], ch['normalized_spectra'])
        ch_peaks = detect_peaks(ch_var['wavenumbers'], ch_var['mean'], ch_var['std'], ch_var['cv'], window=3, std_thresh=0.7, mean_thresh=0.7)
        plot_spectral_variability(
            ch_var['wavenumbers'], ch_var['mean'], ch_var['std'], ch_var['cv'],
            region_name=f"{molecule} CH", peak_labels=ch_peaks, save_path=save_dir
        )

    print(f"All variability plots saved to {save_dir}")


