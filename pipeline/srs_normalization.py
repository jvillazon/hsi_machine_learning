import numpy as np
import os
import sys
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import glob
import tifffile

base_directories =[
    # r"E:\Data\Jorge\01_HumanKidney_HuBMAP\05-13-25 Xenium AKI 3782 K2200093_5\ROI left side\tifs",
    # r"E:\Data\Jorge\01_HumanKidney_HuBMAP\05-13-25 Xenium AKI 3782 K2200093_5\ROI1X left side\tifs",
    # r"E:\Data\Jorge\01_HumanKidney_HuBMAP\05-13-25 Xenium AKI 3782 K2200093_5\ROI2X bottom center\tifs",
    # r"E:\Data\Jorge\01_HumanKidney_HuBMAP\05-14-25 Xenium Ref KPMP_038 S-2211-0003960\ROI1X glomeruli top left\tifs",
    # r"E:\Data\Jorge\01_HumanKidney_HuBMAP\05-14-25 Xenium Ref KPMP_038 S-2211-0003960\ROI2X glomeruli middle\tifs",
    # r"E:\Data\Jorge\01_HumanKidney_HuBMAP\05-15-25 Xenium CKD 3990\ROI1 glomeruli bottom right\tifs",
    # r"E:\Data\Jorge\01_HumanKidney_HuBMAP\05-15-25 Xenium CKD 3990\ROI2 glomeruli middle\tifs",
    r"D:\ADATA Backup\HuBMAP\HuBMAP Xenium\Xenium HSI\data"
]

CONFIG = {
    'num_samples' : 61,
    'wn_1' : 2700,
    'wn_2' : 3100,
    'srs_params_path' : 'params_dataset/srs_params_61.npz',
    'background_csv': r'C:\Users\jvill\OneDrive\Documents\GitHub\hsi_machine_learning\unprocessed_data\water_HSI_76.csv',
    'quant' : True,
    'flip' : True,
    'bg_subtract' : False,
    'show_plots' : False
}


def macro_idx(wavenumber, num_samp, wavenum_1=2700, wavenum_2=3100):
    idx = int(((wavenumber - wavenum_1) / (wavenum_2 - wavenum_1)) * num_samp)
    return idx

def get_image_global_stats(image_flat, chunk_size):
    """Pass 0: Get global min and max of raw data."""
    raw_min = float('inf')
    raw_max = float('-inf')
    for i in range(0, len(image_flat), chunk_size):
        chunk = image_flat[i : i + chunk_size].astype(np.float32)
        raw_min = min(raw_min, np.min(chunk))
        raw_max = max(raw_max, np.max(chunk))
    return raw_min, raw_max

def main():
    # Setup parameters
    # base_directory = r"E:\Data\Jorge\01_HumanKidney_HuBMAP\05-13-25 Xenium AKI 3782 K2200093_5\ROI1X left side\tifs"
    num_samp = CONFIG['num_samples']
    wn_1 = CONFIG['wn_1']
    wn_2 = CONFIG['wn_2']
    ch_start = int((2800 - wn_1) / (wn_2 - wn_1) * num_samp)
    
    
    if not CONFIG['bg_subtract']:
        CONFIG['quant'] = True
    
    background = pd.read_csv(CONFIG['background_csv'])
    background = background.values.flatten()
    if CONFIG['flip']:
        background = np.flip(background)  # Flip to match spectra order if needed
    #Shift background
    shift = 10
    # if shift > 0:
    #     background = np.pad(background, (shift, 0), mode='constant')
    # Interpolate background to match num_samp if needed
    if background.shape[0] != num_samp:
        background_interp = np.interp(
            np.linspace(0, background.shape[0]-1, num_samp),
            np.arange(background.shape[0]),
            background.flatten()
        )
        background = background_interp

  

    for base_directory in tqdm(base_directories, desc="Processing Base Directories"):

        CHUNK_SIZE = 1_000_000

        img_list = glob.glob(os.path.join(base_directory, '*.tif'))
        for img_path in tqdm(img_list, desc="Processing images"):
            sample = os.path.basename(img_path)

            with tifffile.TiffFile(img_path) as tif:
                page = tif.pages[0]
                if tif.is_imagej:
                    pixel_size_x = 1/page.resolution[1]
                    pixel_size_y = 1/page.resolution[0]
                else:
                    pixel_size_x = 1.0
                    pixel_size_y = 1.0
                h, w = page.shape[-2], page.shape[-1]

            n_pixels = h * w
            print(f"\nProcessing {sample} ({h}x{w}), pixel_size_x: {pixel_size_x:.4f}, pixel_size_y: {pixel_size_y:.4f}...")
            
            # Load as memmap
            img_mmap = tifffile.memmap(img_path, mode='r')
            # Handle channel axis (assume CYX or YXC based on dataset convention)
            if img_mmap.shape[0] == num_samp:
                image_flat = img_mmap.reshape(num_samp, -1).T
            else:
                image_flat = img_mmap.reshape(-1, num_samp)

            # Flip channels if needed (spectra order inversion)
            if CONFIG['flip']:
                image_flat = np.flip(image_flat, axis=1)

            if CONFIG['bg_subtract']:
                normalized_dir = os.path.join(os.path.dirname(os.path.dirname(img_path)), f'BackgroundRemoved_{os.path.basename(os.path.dirname(img_path))}')
                os.makedirs(normalized_dir, exist_ok=True)
                norm_img_path = os.path.join(normalized_dir, f"{sample}")
            
                # Pre-allocate output TIFF stack (CYX)
                norm_stack = tifffile.memmap(
                    norm_img_path, shape=(num_samp, h, w), dtype='float32', imagej=True,
                    metadata={'axes': 'CYX', 'unit': 'um'}
                )

            if CONFIG['quant']:
                unsat_idx = macro_idx(3050 + shift, num_samp, wn_1, wn_2)
                protein_idx = macro_idx(2930 + shift, num_samp, wn_1, wn_2)
                sat_idx = macro_idx(2880 + shift, num_samp, wn_1, wn_2)
                lipid_idx = macro_idx(2850 + shift, num_samp, wn_1, wn_2)
                print(f"unsat idx: {unsat_idx}")
                print(f"protein idx: {protein_idx}")
                print(f"sat idx: {sat_idx}")
                print(f"lipid idx: {lipid_idx}")
                # Pre-allocate macro-maps in memory (usually small enough)
                protein_map = np.zeros((h, w), dtype='float32')
                sat_map = np.zeros((h, w), dtype='float32')
                lipid_map = np.zeros((h, w), dtype='float32')
                unsat_map = np.zeros((h, w), dtype='float32')
                
            for i in tqdm(range(0, n_pixels, CHUNK_SIZE), desc="Writing chunks", leave=False):
                end_i = min(i + CHUNK_SIZE, n_pixels)
                chunk_raw = image_flat[i : end_i].astype(np.float32)
                chunk_norm = chunk_raw
                
                # temp_end = chunk_norm[:, -1:-4:-1]
                temp_end = chunk_norm[:, -1:-2:-1]
                temp_start = chunk_norm[:, :ch_start]
                temp_start_mean = np.mean(temp_start, axis=1)[:, np.newaxis]
                temp = chunk_norm - temp_start_mean
                
                spectra_magnitude = np.mean(temp_end, axis=1) - np.mean(temp_start, axis=1)


                if CONFIG['bg_subtract']:
                    if background.shape[0] != num_samp:
                        background_resized = np.zeros(num_samp)
                        background_resized[-background.shape[0]:] = background
                    else:
                        background_resized = background
                    background_arr = np.outer(spectra_magnitude, background_resized)
                    spectra_standard = temp - background_arr
                else:
                    spectra_standard = temp

                if i == 0 and CONFIG['show_plots']:  # Plot example from first chunk
                    temp_mean = np.mean(temp, axis=0)
                    spec_mean = np.mean(spectra_standard, axis=0)
                    plt.figure(figsize=(10, 6))
                    plt.plot(temp_mean, label='Original Spectrum')
                    if CONFIG['bg_subtract']:
                        background_mean = np.mean(background_arr, axis=0)
                        plt.plot(background_mean, label='Background Estimate')
                        plt.plot(spec_mean, label='Standardized Spectrum')
                        plt.title('Spectral Standardization Example')
                        plt.legend(['Original Spectrum', 'Background Estimate', 'Standardized Spectrum'])
                    else:
                        plt.title('Spectral Standardization Example (No Background Subtraction)')
                        plt.legend(['Original/Standardized Spectrum'])
                    plt.show()

                if CONFIG['quant'] and CONFIG['bg_subtract']:
                    spectral_std = spectra_standard.std(axis=1)
                    threshold = np.mean(spectra_standard[:, :ch_start].std(axis=1)) + 2 * np.std(spectra_standard[:, :ch_start].std(axis=1))
                    valid_mask = spectral_std > threshold
                    spectra_standard[~valid_mask] = 0  # Zero out low-variance spectra

                spectra = spectra_standard - np.median(spectra_standard[:, :ch_start], axis=1)[:, np.newaxis]
                
                # Visualize macromolecule indices on the first chunk
                if i == 0 and CONFIG['quant'] and CONFIG['show_plots']:
                    spectra_mean = np.mean(spectra, axis=0)
                    plt.figure(figsize=(10, 6))
                    plt.plot(spectra_mean, label='Standardized Spectrum')
                    plt.axvline(protein_idx, color='r', linestyle='--', label='Protein Index')
                    plt.axvline(sat_idx, color='g', linestyle='--', label='Saturated Lipid Index')
                    plt.axvline(lipid_idx, color='b', linestyle='--', label='Lipid Index')
                    plt.axvline(unsat_idx, color='m', linestyle='--', label='Unsaturated Lipid Index')
                    plt.title('Standardized Spectrum with Macromolecule Indices')
                    plt.xlabel('Channel Index')
                    plt.ylabel('Intensity')
                    plt.legend()
                    plt.show()

                if CONFIG['bg_subtract']:
                    # Write to 3D stack
                    for ch_idx in range(num_samp):
                        norm_stack.reshape(num_samp, -1)[ch_idx, i:end_i] = spectra[:, ch_idx]
                    norm_stack.flush()

                if CONFIG['quant']:
                    # Map extraction (Max in range)
                    protein_map.reshape(-1)[i:end_i] = np.max(spectra[:, protein_idx-2:protein_idx+2], axis=1)
                    sat_map.reshape(-1)[i:end_i] = np.max(spectra[:, sat_idx-2:sat_idx+2], axis=1)
                    lipid_map.reshape(-1)[i:end_i] = np.max(spectra[:, lipid_idx-2:lipid_idx+2], axis=1)
                    unsat_map.reshape(-1)[i:end_i] = np.max(spectra[:, unsat_idx-2:unsat_idx+2], axis=1)

                    # Determine macromolecule output dir prefix based on parameters
                    if CONFIG['bg_subtract']:
                        folder_prefix = "BackgroundRemoved_Images"
                    else:
                        folder_prefix = "Images"

                    output_dir = os.path.join(os.path.dirname(os.path.dirname(img_path)), os.path.join(f'{folder_prefix}_{os.path.basename(os.path.dirname(img_path))}', f"{sample.replace('.tif','')}"))
                    os.makedirs(output_dir, exist_ok=True)
                    
                    protein_map = np.maximum(protein_map, 0)
                    sat_map = np.maximum(sat_map, 0)
                    lipid_map = np.maximum(lipid_map, 0)
                    unsat_map = np.maximum(unsat_map, 0)
                    
                    # Helper for resolution
                    res = (1/pixel_size_x, 1/pixel_size_y)
                    meta = {'axes': 'YX', 'unit': 'um', 'PhysicalSizeX': pixel_size_x, 'PhysicalSizeY': pixel_size_y}

                    tifffile.imwrite(os.path.join(output_dir, f"Protein-{sample}"), protein_map, resolution=res, imagej=True, metadata=meta)
                    tifffile.imwrite(os.path.join(output_dir, f"Sat-{sample}"), sat_map, resolution=res, imagej=True, metadata=meta)
                    tifffile.imwrite(os.path.join(output_dir, f"Lipid-{sample}"), lipid_map, resolution=res, imagej=True, metadata=meta)
                    tifffile.imwrite(os.path.join(output_dir, f"Unsat-{sample}"), unsat_map, resolution=res, imagej=True, metadata=meta)
                    
                    # Ratios
                    p_img = np.maximum(protein_map - lipid_map, 0)
                    # p_img[p_img > 50] = 0
                    l_img = np.maximum(5 * lipid_map - 0.4 * protein_map, 0)
                    # l_img[l_img > 50] = 0
                    
                    tifffile.imwrite(os.path.join(output_dir, f"P_image-{sample}"), p_img, resolution=res, imagej=True, metadata=meta)
                    tifffile.imwrite(os.path.join(output_dir, f"L_image-{sample}"), l_img, resolution=res, imagej=True, metadata=meta)
                    
                    unsat_ratio = np.where(sat_map == 0, 0, unsat_map / (sat_map + unsat_map + 1e-6))
                    lipid_ratio = np.where(protein_map == 0, 0, lipid_map / (protein_map + lipid_map + 1e-6))

                    # Determine ratio output dir based on parameters
                    if CONFIG['bg_subtract']:
                        ratio_folder_prefix = "Ratio_BackgroundRemoved"
                    else:
                        ratio_folder_prefix = "Ratio"

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