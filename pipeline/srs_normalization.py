import numpy as np
import os
import sys
from typing import Optional, Tuple, Union, Dict
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import glob
import tifffile
import torch
import torch.nn.functional as F
from scipy.interpolate import CubicSpline


base_directories = [
    # r"/Users/jorgevillazon/Downloads/A2/HSI/hSRS",
    # r"/Users/jorgevillazon/Downloads/A1/HSI/hSRS",
    # r"/Users/jorgevillazon/Downloads/A3/HSI/hSRS",
    # r"/Users/jorgevillazon/Downloads/B/HSI/hSRS",
    # r"/Users/jorgevillazon/Downloads/C/HSI/hSRS",
    # r"D:\ADATA Backup\HuBMAP\HuBMAP Xenium\Xenium HSI\09.11.26_test\data"
    r"C:\Users\jvill\OneDrive - UC San Diego\Vallon Kidney\HSI data"
]

CONFIG = {
    'wn_1': 2700,             # first cm-1
    'wn_2': 3100,             # last cm-1
    'background_csv': r'C:\Users\jvill\Documents\GitHub\hsi_machine_learning\unprocessed_data\water_HSI_71.csv', # water baseline profile
    'spectra_csv': r'C:\Users\jvill\Documents\GitHub\hsi_machine_learning\unprocessed_data\averaged_spectra.csv', # molecule reference spectra (e.g., Kidney Background)
    'quant': True,            # create macromolecule images
    'flip': True,            # reverse order to match cm-1 (2700 -> 3100)
    'bg_subtract': True,      # subtract water background 
    'estimate_shift': True,   # estimate best shift using correlation with background reference spectra to locate peak indices
    'max_shift_cm': 50,       # maximum shift search in cm^-1
    'shift': 15,              # fallback manual shift in cm^-1 if estimate_shift is False
    'show_plots': False,       # show plots of background subtraction and peak assignment (debugging)
    'chunk_size': 1_000_000,  # chunk size for memory efficient processing
}


def macro_idx(wavenumber: float, num_samp: int, wavenum_1: float = 2700, wavenum_2: float = 3100) -> int:
    """Calculate the channel index for a given wavenumber in a spectrum of size num_samp."""
    idx = int(((wavenumber - wavenum_1) / (wavenum_2 - wavenum_1)) * num_samp)
    return idx


def normalize_spectra(spectra: np.ndarray) -> np.ndarray:
    """Min-max normalize spectra (n_samples, n_channels) or (n_channels,)."""
    if spectra.ndim == 1:
        min_v = np.min(spectra)
        max_v = np.max(spectra)
        return (spectra - min_v) / (max_v - min_v + 1e-10)
    else:
        min_v = np.min(spectra, axis=1, keepdims=True)
        max_v = np.max(spectra, axis=1, keepdims=True)
        return (spectra - min_v) / (max_v - min_v + 1e-10)


def shift_spectra(spectra: np.ndarray, shift_channels: int) -> np.ndarray:
    """
    Shift spectra along the channel axis by shift_channels.
    Positive shift_channels moves spectra to higher channel indices.
    Negative shift_channels moves spectra to lower channel indices.
    """
    if shift_channels == 0:
        return spectra.copy()

    shifted = np.zeros_like(spectra)
    if spectra.ndim == 1:
        if shift_channels > 0:
            shifted[shift_channels:] = spectra[:-shift_channels]
            shifted[:shift_channels] = spectra[0]
        else:
            s = abs(shift_channels)
            shifted[:-s] = spectra[s:]
            shifted[-s:] = spectra[-1]
    else:
        if shift_channels > 0:
            shifted[:, shift_channels:] = spectra[:, :-shift_channels]
            shifted[:, :shift_channels] = spectra[:, :1]
        else:
            s = abs(shift_channels)
            shifted[:, :-s] = spectra[:, s:]
            shifted[:, -s:] = spectra[:, -1:]
    return shifted


def load_reference_splines(csv_path: str) -> Dict[str, CubicSpline]:
    """Load reference spectra from CSV and return continuous CubicSpline interpolators."""
    df = pd.read_csv(csv_path)
    df = df.dropna(axis='columns', how='all')

    col_names = df.columns
    ref_splines = {}

    for i in range(0, len(col_names), 2):
        name = col_names[i].strip()
        mol_wn = df.iloc[:, i].values
        mol_int = df.iloc[:, i + 1].values

        # Remove NaNs
        valid = ~np.isnan(mol_wn) & ~np.isnan(mol_int)
        mol_wn = mol_wn[valid]
        mol_int = mol_int[valid]

        if len(mol_wn) < 2:
            continue

        # Sort by wavenumber if needed
        idx = np.argsort(mol_wn)
        mol_wn = mol_wn[idx]
        mol_int = mol_int[idx]

        try:
            cs = CubicSpline(mol_wn, mol_int, extrapolate=True)
            ref_splines[name] = cs
        except Exception as e:
            print(f"Failed to create spline for {name}: {e}")

    return ref_splines


def get_channel_calibration(
    c_in: int,
    wn_1: float,
    wn_2: float,
    raw_water_bg: np.ndarray,
    ref_splines: Dict[str, CubicSpline],
    bg_ref_name: str,
    max_shift_cm: float,
    cache: dict,
    flip: bool = False,
) -> dict:
    """
    Retrieve or compute the 1D background profile, anchor reference spectrum,
    and channel grid parameters for a specific channel count (c_in).
    Results are cached to avoid repeated 1D interpolations.
    """
    if c_in in cache:
        return cache[c_in]

    wn_step = (wn_2 - wn_1) / (c_in - 1) if c_in > 1 else 1.0
    ch_start = int((2800 - wn_1) / (wn_2 - wn_1) * c_in)
    max_shift_ch = int(np.ceil(max_shift_cm / wn_step))
    target_wn = np.linspace(wn_1, wn_2, c_in)

    # 1. Resample water baseline 1D profile to match c_in
    if len(raw_water_bg) != c_in:
        water_bg = np.interp(
            np.linspace(0, len(raw_water_bg) - 1, c_in),
            np.arange(len(raw_water_bg)),
            raw_water_bg
        ).astype(np.float32)
    else:
        water_bg = raw_water_bg.astype(np.float32)

    if flip:
        water_bg = np.flip(water_bg)

    # 2. Resample anchor reference spectrum to match c_in
    if bg_ref_name in ref_splines:
        ref_raw = ref_splines[bg_ref_name](target_wn)
        bg_ref_norm = normalize_spectra(ref_raw.astype(np.float32))
    else:
        bg_ref_norm = np.zeros(c_in, dtype=np.float32)

    calib = {
        'c_in': c_in,
        'wn_step': wn_step,
        'ch_start': ch_start,
        'max_shift_ch': max_shift_ch,
        'water_background': water_bg,
        'bg_ref_norm': bg_ref_norm,
        'target_wn': target_wn,
    }
    cache[c_in] = calib
    return calib


def estimate_best_shift(
    image_mean: np.ndarray,
    background_ref: np.ndarray,
    ch_start: int,
    max_shift_ch: int
) -> Tuple[int, float]:
    """
    Estimate the shift of the image relative to reference by correlating background_ref with image_mean.
    Window: High intensity region (> 2800 cm^-1, index ch_start:).
    Returns best_shift (relative displacement of image peaks relative to ref).
    """
    best_corr = -1.0
    best_shift = 0

    # Focus on the region after CH_START (the CH stretch region)
    region_mask = slice(ch_start, None)
    img_region = image_mean[region_mask]
    img_region = (img_region - np.mean(img_region)) / (np.std(img_region) + 1e-10)

    for s in range(-max_shift_ch, max_shift_ch + 1):
        shifted_ref = shift_spectra(background_ref, s)
        ref_region = shifted_ref[region_mask]
        ref_region = (ref_region - np.mean(ref_region)) / (np.std(ref_region) + 1e-10)

        corr = float(np.mean(img_region * ref_region))
        if corr > best_corr:
            best_corr = corr
            best_shift = s

    return best_shift, best_corr


def gpu_vectorized_interpolation(
    spectra_2d: np.ndarray,
    target_channels: int = 61,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Optional helper: Vectorized 1D linear spectral interpolation using PyTorch acceleration.
    Resamples (N_pixels, C_in) -> (N_pixels, target_channels).
    """
    if spectra_2d.shape[1] == target_channels:
        return spectra_2d.astype(np.float32)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        tensor_in = torch.from_numpy(spectra_2d).to(device=device, dtype=torch.float32).unsqueeze(1)
        tensor_out = F.interpolate(
            tensor_in,
            size=target_channels,
            mode="linear",
            align_corners=True,
        ).squeeze(1)

        if device.type == "cuda":
            return tensor_out.cpu().numpy()
        return tensor_out.numpy()


def main():
    wn_1 = CONFIG['wn_1']
    wn_2 = CONFIG['wn_2']
    max_shift_cm = CONFIG.get('max_shift_cm', 50)
    chunk_size = CONFIG.get('chunk_size', 1_000_000)

    if not CONFIG['bg_subtract']:
        CONFIG['quant'] = True

    # 1. Load raw water baseline profile
    raw_water_background = pd.read_csv(CONFIG['background_csv'], header=None).values.flatten()

    # 2. Load molecule reference splines from CSV
    ref_splines = load_reference_splines(CONFIG['spectra_csv'])

    # Determine anchor reference spectrum name
    bg_ref_name = 'Kidney Background'
    if bg_ref_name not in ref_splines:
        bg_ref_name = next((k for k in ref_splines.keys() if 'background' in k.lower()), list(ref_splines.keys())[0])
        print(f"Warning: '{bg_ref_name}' not found, using '{bg_ref_name}' as calibration anchor.")

    # Cache for dynamically computed calibration parameters per unique channel count (c_in)
    channel_cache = {}

    for base_directory in tqdm(base_directories, desc="Processing Base Directories"):
        if not os.path.exists(base_directory):
            print(f"Directory not found: {base_directory}")
            continue

        img_list = glob.glob(os.path.join(base_directory, '*.tif'))
        for img_path in tqdm(img_list, desc="Processing images"):
            sample = os.path.basename(img_path)

            with tifffile.TiffFile(img_path) as tif:
                page = tif.pages[0]
                if tif.is_imagej:
                    pixel_size_x = 1 / page.resolution[1]
                    pixel_size_y = 1 / page.resolution[0]
                else:
                    pixel_size_x = 1.0
                    pixel_size_y = 1.0

            # Load as memmap
            img_mmap = tifffile.memmap(img_path, mode='r')
            if img_mmap.ndim == 3:
                s0, s1, s2 = img_mmap.shape
                if s0 < s1 and s0 < s2:
                    c_in, h, w = s0, s1, s2
                    image_flat = img_mmap.reshape(c_in, -1).T
                elif s2 < s0 and s2 < s1:
                    h, w, c_in = s0, s1, s2
                    image_flat = img_mmap.reshape(-1, c_in)
                else:
                    c_in, h, w = s0, s1, s2
                    image_flat = img_mmap.reshape(c_in, -1).T
            else:
                print(f"Skipping {sample}: expected 3D image stack, got shape {img_mmap.shape}")
                continue

            n_pixels = h * w

            # Retrieve cached or newly computed 1D calibration parameters for this channel length
            calib = get_channel_calibration(
                c_in=c_in,
                wn_1=wn_1,
                wn_2=wn_2,
                raw_water_bg=raw_water_background,
                ref_splines=ref_splines,
                bg_ref_name=bg_ref_name,
                max_shift_cm=max_shift_cm,
                cache=channel_cache,
                flip=CONFIG['flip'],
            )

            wn_step = calib['wn_step']
            ch_start = calib['ch_start']
            max_shift_ch = calib['max_shift_ch']
            water_background = calib['water_background']
            bg_ref_norm = calib['bg_ref_norm']

            print(f"\nProcessing {sample} ({h}x{w}, {c_in} channels native), pixel_size_x: {pixel_size_x:.4f}, pixel_size_y: {pixel_size_y:.4f}...")

            # Flip channels if needed (spectra order inversion)
            if CONFIG['flip']:
                image_flat = np.flip(image_flat, axis=1)

            # Estimate best shift against background reference spectrum (in native c_in channels)
            if CONFIG.get('estimate_shift', True):
                print("  Estimating global shift across image...")
                sum_spectra = np.zeros(c_in, dtype=np.float64)
                for start_p in range(0, n_pixels, chunk_size):
                    end_p = min(start_p + chunk_size, n_pixels)
                    sum_spectra += np.sum(image_flat[start_p:end_p].astype(np.float32), axis=0)
                image_mean = (sum_spectra / n_pixels).astype(np.float32)

                # Baseline and water baseline correction on mean spectrum
                temp_start_mean = float(np.mean(image_mean[:ch_start]))
                temp_mean = image_mean - temp_start_mean
                spectra_magnitude_mean = float(np.mean(image_mean[-1:]) - temp_start_mean)

                if CONFIG['bg_subtract']:
                    water_sub_mean = temp_mean - (spectra_magnitude_mean * water_background)
                else:
                    water_sub_mean = temp_mean

                processed_mean = water_sub_mean - np.median(water_sub_mean[:ch_start])
                image_mean_norm = normalize_spectra(processed_mean)

                # Estimate best shift of the image relative to reference
                best_shift, best_corr = estimate_best_shift(image_mean_norm, bg_ref_norm, ch_start, max_shift_ch)
                shift_cm = best_shift * wn_step
                print(f"  Estimated best shift: {best_shift} channels (~{shift_cm:.2f} cm^-1, corr: {best_corr:.4f})")
            else:
                shift_cm = CONFIG.get('shift', 0)
                best_shift = int(round(shift_cm / wn_step))
                print(f"  Using manual shift: {shift_cm} cm^-1 (~{best_shift} channels)")

            # Compute macromolecule peak indices directly for c_in
            unsat_idx = macro_idx(3005 + shift_cm, c_in, wn_1, wn_2)
            protein_idx = macro_idx(2930 + shift_cm, c_in, wn_1, wn_2)
            sat_idx = macro_idx(2880 + shift_cm, c_in, wn_1, wn_2)
            lipid_idx = macro_idx(2850 + shift_cm, c_in, wn_1, wn_2)

            # Clamp indices within valid window boundaries
            unsat_idx = max(2, min(unsat_idx, c_in - 3))
            protein_idx = max(2, min(protein_idx, c_in - 3))
            sat_idx = max(2, min(sat_idx, c_in - 3))
            lipid_idx = max(2, min(lipid_idx, c_in - 3))

            if CONFIG['quant']:
                print(f"  Assigned peak indices ({c_in} ch): unsat={unsat_idx} ({3005 + shift_cm:.1f} cm^-1), protein={protein_idx} ({2930 + shift_cm:.1f} cm^-1), sat={sat_idx} ({2880 + shift_cm:.1f} cm^-1), lipid={lipid_idx} ({2850 + shift_cm:.1f} cm^-1)")

            if CONFIG['show_plots']:
                plt.figure(figsize=(10, 5))
                plt.plot(normalize_spectra(image_mean), label='Raw Image Mean', alpha=0.5, linestyle=':')
                plt.plot(image_mean_norm, label='Water-Subtracted Image Mean', linewidth=2)
                plt.plot(bg_ref_norm, label=f'Ref: {bg_ref_name}', linestyle='--', color='black', alpha=0.7)
                plt.axvline(protein_idx, color='r', linestyle='--', label=f'Protein ({protein_idx})')
                plt.axvline(sat_idx, color='g', linestyle='--', label=f'Sat Lipid ({sat_idx})')
                plt.axvline(lipid_idx, color='b', linestyle='--', label=f'Lipid ({lipid_idx})')
                plt.axvline(unsat_idx, color='m', linestyle='--', label=f'Unsat Lipid ({unsat_idx})')
                plt.axvline(ch_start, color='grey', linestyle=':', label='CH Start')
                plt.title(f"{sample} - Shift Estimation & Peak Assignments (Shift: {best_shift} ch / ~{shift_cm:.1f} cm^-1)")
                plt.xlabel('Channel Index')
                plt.ylabel('Normalized Intensity')
                plt.legend()
                plt.show()

            if CONFIG['bg_subtract']:
                normalized_dir = os.path.join(os.path.dirname(os.path.dirname(img_path)), f'BackgroundRemoved_{os.path.basename(os.path.dirname(img_path))}')
                os.makedirs(normalized_dir, exist_ok=True)
                norm_img_path = os.path.join(normalized_dir, f"{sample}")

                # Pre-allocate output TIFF stack with native channels (CYX)
                norm_stack = tifffile.memmap(
                    norm_img_path, shape=(c_in, h, w), dtype='float32', imagej=True,
                    metadata={'axes': 'CYX', 'unit': 'um'}
                )

            if CONFIG['quant']:
                # Pre-allocate macro-maps in memory
                protein_map = np.zeros((h, w), dtype='float32')
                sat_map = np.zeros((h, w), dtype='float32')
                lipid_map = np.zeros((h, w), dtype='float32')
                unsat_map = np.zeros((h, w), dtype='float32')
                mask_map = np.zeros((h, w), dtype='uint8')

            for i in tqdm(range(0, n_pixels, chunk_size), desc="Writing chunks", leave=False):
                end_i = min(i + chunk_size, n_pixels)
                # Process native channel spectra directly without per-pixel interpolation
                chunk_raw = image_flat[i : end_i].astype(np.float32)

                temp_end = chunk_raw[:, -1:-2:-1]
                temp_start = chunk_raw[:, :ch_start]
                temp_start_mean = np.mean(temp_start, axis=1)[:, np.newaxis]
                temp = chunk_raw - temp_start_mean

                spectra_magnitude = np.mean(temp_end, axis=1) - np.mean(temp_start, axis=1)

                if CONFIG['bg_subtract']:
                    background_arr = np.outer(spectra_magnitude, water_background)
                    spectra_standard = temp - background_arr
                else:
                    spectra_standard = temp

                if CONFIG['quant'] and CONFIG['bg_subtract']:
                    spectral_std = spectra_standard.std(axis=1)
                    threshold = np.mean(spectra_standard[:, :ch_start].std(axis=1)) + 1.75 * np.std(spectra_standard[:, :ch_start].std(axis=1))
                    valid_mask = spectral_std > threshold
                    spectra_standard[~valid_mask] = 0  # Zero out low-variance spectra

                spectra = spectra_standard - np.median(spectra_standard[:, :ch_start], axis=1)[:, np.newaxis]

                if CONFIG['bg_subtract']:
                    # Write directly to 3D stack in native channels
                    for ch_idx in range(c_in):
                        norm_stack.reshape(c_in, -1)[ch_idx, i:end_i] = spectra[:, ch_idx]
                    norm_stack.flush()

                if CONFIG['quant']:
                    # Map extraction using the shift-adjusted peak indices scaled to c_in
                    protein_map.reshape(-1)[i:end_i] = np.max(spectra[:, protein_idx-2:protein_idx+2], axis=1)
                    sat_map.reshape(-1)[i:end_i] = np.max(spectra[:, sat_idx-2:sat_idx+2], axis=1)
                    lipid_map.reshape(-1)[i:end_i] = np.max(spectra[:, lipid_idx-2:lipid_idx+2], axis=1)
                    unsat_map.reshape(-1)[i:end_i] = np.max(spectra[:, unsat_idx-2:unsat_idx+2], axis=1)

                    if CONFIG['bg_subtract']:
                        mask_map.reshape(-1)[i:end_i] = valid_mask.astype(np.uint8)
                    else:
                        mask_map.reshape(-1)[i:end_i] = 1

            if CONFIG['quant']:
                folder_prefix = "BR_Images" if CONFIG['bg_subtract'] else "Images"
                output_dir = os.path.join(os.path.dirname(os.path.dirname(img_path)), os.path.join(f'{folder_prefix}_{os.path.basename(os.path.dirname(img_path))}', f"{sample.replace('.tif','')}"))
                os.makedirs(output_dir, exist_ok=True)

                protein_map = np.maximum(protein_map, 0)
                sat_map = np.maximum(sat_map, 0)
                lipid_map = np.maximum(lipid_map, 0)
                unsat_map = np.maximum(unsat_map, 0)

                # Helper for resolution
                res = (1 / pixel_size_x, 1 / pixel_size_y)
                meta = {'axes': 'YX', 'unit': 'um', 'PhysicalSizeX': pixel_size_x, 'PhysicalSizeY': pixel_size_y}
                tifffile.imwrite(os.path.join(output_dir, f"Protein-{sample}"), protein_map, resolution=res, imagej=True, metadata=meta)
                tifffile.imwrite(os.path.join(output_dir, f"Sat-{sample}"), sat_map, resolution=res, imagej=True, metadata=meta)
                tifffile.imwrite(os.path.join(output_dir, f"Lipid-{sample}"), lipid_map, resolution=res, imagej=True, metadata=meta)
                tifffile.imwrite(os.path.join(output_dir, f"Unsat-{sample}"), unsat_map, resolution=res, imagej=True, metadata=meta)
                tifffile.imwrite(os.path.join(output_dir, f"Mask-{sample}"), mask_map, resolution=res, imagej=True, metadata=meta)

                # Ratios
                p_img = np.maximum(protein_map - lipid_map, 0)
                l_img = np.maximum(5 * lipid_map - 0.4 * protein_map, 0)
                tifffile.imwrite(os.path.join(output_dir, f"P_image-{sample}"), p_img, resolution=res, imagej=True, metadata=meta)
                tifffile.imwrite(os.path.join(output_dir, f"L_image-{sample}"), l_img, resolution=res, imagej=True, metadata=meta)

                unsat_ratio = np.where(sat_map == 0, 0, unsat_map / (sat_map + unsat_map + 1e-6))
                lipid_ratio = np.where(protein_map == 0, 0, lipid_map / (protein_map + lipid_map + 1e-6))

                ratio_folder_prefix = "BR_Ratios" if CONFIG['bg_subtract'] else "Ratios"
                ratio_dir = os.path.join(os.path.dirname(os.path.dirname(img_path)), f'{ratio_folder_prefix}_{os.path.basename(os.path.dirname(img_path))}', f"{sample.replace('.tif','')}")
                os.makedirs(ratio_dir, exist_ok=True)
                tifffile.imwrite(os.path.join(ratio_dir, f"{sample.replace('.tif','')}-Lipid_Unsaturation-ratio.tif"), unsat_ratio.astype('float32'), resolution=res, imagej=True, metadata=meta)
                tifffile.imwrite(os.path.join(ratio_dir, f"{sample.replace('.tif','')}-Lipid_to_Protein-ratio.tif"), lipid_ratio.astype('float32'), resolution=res, imagej=True, metadata=meta)

            # Cleanup
            del img_mmap
            import gc
            gc.collect()


if __name__ == "__main__":
    main()
    print("Processing complete.")