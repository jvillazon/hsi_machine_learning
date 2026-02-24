import os

import numpy as np
import pandas as pd
import scipy
import scipy.signal as signal
import scipy.optimize as optimize
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt


def find_initial_peaks(wavenumbers, intensities, height=None, distance=None, prominence=None):
    """
    Identifies initial peak positions in the spectrum using scipy's find_peaks function.

    Parameters
    ----------
    wavenumbers: np.ndarray
        Array of wavenumber values corresponding to the spectrum.
    intensities: np.ndarray
        Array of intensity values corresponding to the spectrum.
    height: float or tuple, optional
        Required height of peaks. Can be a single value or a tuple (min, max).
    distance: float, optional
        Required minimum horizontal distance (in number of samples) between neighboring peaks.
    prominence: float or tuple, optional
        Required prominence of peaks. Can be a single value or a tuple (min, max).

    Returns
    -------
    peaks: np.ndarray
        Indices of the identified peaks in the input arrays.
    properties: dict
        Properties of the identified peaks as returned by scipy's find_peaks function.
    window: tuple
        Suggested window around each peak for fitting, based on the distance parameter.
    """
    min_intensity = np.min(intensities)
    max_intensity = np.max(intensities)

    normalized_intensities = (intensities - min_intensity) / (max_intensity - min_intensity + 1e-8)

    peaks, properties = signal.find_peaks(normalized_intensities, height=height, distance=distance, prominence=prominence)
    width_data = signal.peak_widths(normalized_intensities, peaks, rel_height=0.5)
    widths = width_data[0]
    
    properties['widths'] = widths

    windows = []    
    for idx, peak in enumerate(peaks):
        start = max(0, int(width_data[2][idx] - widths[idx] // 2))
        end = min(len(wavenumbers), int(width_data[3][idx] + widths[idx] // 2))
        windows.append((start, end))
    return peaks, properties, windows


def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

def lorentzian(x, A, mu, gamma):
    return A * (gamma ** 2 / ((x - mu) ** 2 + gamma ** 2))

def voigt(x, A, mu, sigma, gamma):
    z = ((x - mu) + 1j * gamma) / (sigma * np.sqrt(2))
    return A * np.real(scipy.special.wofz(z)) / (sigma * np.sqrt(2 * np.pi))

def fit_model(model, x, y, p0):
    popt, _ = optimize.curve_fit(model, x, y, p0=p0, maxfev=50000)
    residuals = y - model(x, *popt)
    rss = np.sum(residuals**2)
    return popt, rss

def aic(rss, n, k):
    return 2*k + n*np.log(rss/n)

def determine_peak_type(x_window, y_window, mu_guess, width_guess):
    # Initial guesses
    p0_gauss = [y_window.max(), mu_guess, width_guess]
    p0_lorentz = [y_window.max(), mu_guess, width_guess]
    p0_voigt = [y_window.max(), mu_guess, width_guess, width_guess]

    # Fit all models
    popt_g, rss_g = fit_model(gaussian, x_window, y_window, p0_gauss)
    popt_l, rss_l = fit_model(lorentzian, x_window, y_window, p0_lorentz)
    popt_v, rss_v = fit_model(voigt, x_window, y_window, p0_voigt)

    # Compute AIC
    n = len(x_window)
    aic_g = aic(rss_g, n, 3)
    aic_l = aic(rss_l, n, 3)
    aic_v = aic(rss_v, n, 4)

    # Choose best
    aics = {"gaussian": aic_g, "lorentzian": aic_l, "voigt": aic_v}
    best_type = min(aics, key=aics.get)

    return best_type, popt_g if best_type == "gaussian" else popt_l if best_type == "lorentzian" else popt_v, aics[best_type]

def get_peak_info_windows(wavenumbers, intensities, peaks, properties, windows):
    peak_info = []
    for i, peak in enumerate(peaks):
        mu_guess = wavenumbers[peak]
        width_guess = properties['widths'][i]
        x_window = wavenumbers[windows[i][0]:windows[i][1]]
        y_window = intensities[windows[i][0]:windows[i][1]]
        peak_type, params, aic = determine_peak_type(
            x_window, 
            y_window, 
            mu_guess, 
            width_guess
        )
        peak_info.append({
            "type": peak_type,
            "initial_params": params,
            "aic": aic,
            "wavenumber_window": x_window,
            "intensity_window": y_window,
            "mu_guess": mu_guess,
            "width_guess": width_guess,
        })
    return peak_info

def fit_global_mixture(wavenumbers, intensities, peak_info):
    """
    Fits a global mixture model to the spectrum using the identified peaks and their windows.

    Parameters
    ----------
    wavenumbers: np.ndarray
        Array of wavenumber values corresponding to the spectrum.
    intensities: np.ndarray
        Array of intensity values corresponding to the spectrum.
    peak_info: list of dicts
        List of dictionaries containing peak information (including peak types and initial parameters).

    Returns
    -------
    fitted_params: list of dicts
        List of fitted parameters for each peak, including type and model parameters.
    """
    # Extract peak types and initial parameters for global fitting
    peak_types = [info['type'] for info in peak_info]
    initial_params = []
    for info in peak_info:
        initial_params.extend(info['initial_params'])

    # Define a global mixture model that sums the contributions of all peaks based on their types and parameters
    def mixture_model(x, *params):
        y = np.zeros_like(x)
        idx = 0
        for peak in peak_types:
            if peak == "gaussian":
                A, mu, sigma = params[idx:idx+3]
                y += gaussian(x, A, mu, sigma)
                idx += 3
            elif peak == "lorentzian":
                A, mu, gamma = params[idx:idx+3]
                y += lorentzian(x, A, mu, gamma)
                idx += 3
            elif peak == "voigt":
                A, mu, sigma, gamma = params[idx:idx+4]
                y += voigt(x, A, mu, sigma, gamma)
                idx += 4
        return y

    # Fit the global mixture model to the entire spectrum
    popt, _ = optimize.curve_fit(
        mixture_model, 
        wavenumbers, 
        intensities, 
        p0=initial_params,
        maxfev=50000)
    return popt, peak_types

def extract_peak_parameters(popt, peak_types):
    """
    Extracts peak parameters from the fitted global mixture model.

    Parameters
    ----------
    popt: list of float
        Fitted parameters from the global mixture model.
    peak_types: list of str
        List of peak types (e.g., "gaussian", "lorentzian", "voigt").

    Returns
    -------
    extracted_params: list of dicts
        List of dictionaries containing extracted parameters for each peak.
    """
    extracted_params = []
    idx = 0
    for peak_type in peak_types:
        if peak_type == "gaussian":
            A, mu, sigma = popt[idx:idx+3]
            fwhm = 2.355 * sigma
            area = A * sigma * np.sqrt(2 * np.pi)
            extracted_params.append({
                "type": "gaussian",
                "A": A, 
                "mu": mu, 
                "sigma": sigma,
                "fwhm": fwhm,
                "area": area
                })
            idx += 3
        elif peak_type == "lorentzian":
            A, mu, gamma = popt[idx:idx+3]
            fwhm = 2 * gamma
            area = A * np.pi * gamma
            extracted_params.append({
                "type": "lorentzian", 
                "A": A, 
                "mu": mu, 
                "gamma": gamma, 
                "fwhm": fwhm, 
                "area": area
                })
            idx += 3
        elif peak_type == "voigt":
            A, mu, sigma, gamma = popt[idx:idx+4]
            fwhm_gauss = 2.355 * sigma
            fwhm_lorentz = 2 * gamma
            fwhm = 0.5346 * fwhm_lorentz + np.sqrt(0.2166 * fwhm_lorentz**2 + fwhm_gauss**2)
            area = A * sigma * np.sqrt(2 * np.pi)  # Approximate area for Voigt
            extracted_params.append({
                "type": "voigt", 
                "A": A, 
                "mu": mu, 
                "sigma": sigma, 
                "gamma": gamma, 
                "fwhm": fwhm, 
                "area": area
                })
            idx += 4
    return extracted_params

def aggregate_peak_parameters(peak_info, extracted_params, spectrum_id):
    """
    Aggregates peak information and extracted parameters into a structured format.

    Parameters
    ----------
    peak_info: list of dicts
        List of dictionaries containing initial peak information.
    extracted_params: list of dicts
        List of dictionaries containing extracted parameters for each peak.
    spectrum_id: str
        Identifier for the spectrum (e.g., filename or sample ID).

    Returns
    -------
    aggregated_info: list of dicts
        List of dictionaries combining initial peak information and extracted parameters.
    """
    aggregated_info = []
    for info, params in zip(peak_info, extracted_params):
        combined_info = {
            "x_local": info['wavenumber_window'],
            "y_local": info['intensity_window'],
            "mu_guess": info['mu_guess'],
            "width_guess": info['width_guess'],
            "parameters": params,
            "spectrum_id": spectrum_id,
        }
        aggregated_info.append(combined_info)

    return aggregated_info

def display_peak_model(aggregated_info, wavenumbers):

    for info in aggregated_info:
        params = info['parameters']
        if params['type'] == "gaussian":
            model_y = gaussian(wavenumbers, params['A'], params['mu'], params['sigma'])
        elif params['type'] == "lorentzian":
            model_y = lorentzian(wavenumbers, params['A'], params['mu'], params['gamma'])
        elif params['type'] == "voigt":
            model_y = voigt(wavenumbers, params['A'], params['mu'], params['sigma'], params['gamma'])
        else:
            continue
        plt.plot(wavenumbers, model_y)
    
    plt.legend()
    plt.show()





def main():
    
    # Load preprocessed spectra

    dataset_path = r"/Users/jorgevillazon/Documents/GitHub/hsi_machine_learning/spontaneous/datasets/processed_spectra.csv"
    molecule_df = pd.read_csv(dataset_path)
    wavenumbers = np.array(molecule_df.columns[1:].to_numpy().astype(np.float32))

    peak_dataset = []
    for name, row in zip(molecule_df['Molecule'], molecule_df.iloc[:, 1:].itertuples(index=False)):
        # Convert spectrum to numpy array
        spectra = np.array(row, dtype=np.float32)

        # Detect peaks and determine their types
        peaks, properties, windows = find_initial_peaks(wavenumbers, spectra, height=0.01, distance=5, prominence=0.005)
        peak_info = get_peak_info_windows(wavenumbers, spectra, peaks, properties, windows)
        popt, peak_types = fit_global_mixture(wavenumbers, spectra, peak_info)

        # Extract and save peak parameters
        extracted_params = extract_peak_parameters(popt, peak_types)

        # Aggregate all information into a structured format
        aggregated_info = aggregate_peak_parameters(peak_info, extracted_params, spectrum_id=name)

        # Display the fitted model for visual inspection
        display_peak_model(aggregated_info, wavenumbers)

        # Append aggregated info to the dataset
        peak_dataset.extend(aggregated_info)

    # Convert to DataFrame and save
    peak_df = pd.DataFrame(peak_dataset, index=[info["spectrum_id"] for info in peak_dataset])

    output_dir = os.path.join(os.path.dirname(dataset_path), "..")
    peak_df.to_csv(os.path.join(output_dir, "extracted_peak_parameters.csv"))

if __name__ == "__main__":
    main()  