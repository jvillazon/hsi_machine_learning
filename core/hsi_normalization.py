import numpy as np
import os
import sys
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt


core_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'core'))
if core_path not in sys.path:
    sys.path.insert(0, core_path)
    
from hsi_unlabeled_dataset import HSI_Unlabeled_Dataset
from hsi_load_data import normalize, noise_normalize
import tifffile


def macro_idx(wavenumber, num_samp, wavenum_1=2700, wavenum_2=3100):
    idx = int(((wavenumber - wavenum_1) / (wavenum_2 - wavenum_1)) * num_samp)
    return idx

def spectral_standardization(data, wavenum_1=2700, wavenum_2=3100, num_samp=61, background=None, plot_example=False):
    ch_start = int((2800 - wavenum_1) / (wavenum_2 - wavenum_1) * num_samp)
    if background is None:
        raise ValueError("background cannot be None. Load from srs_params_path.")
    temp_norm = noise_normalize(data, ch_start)
    temp_end = temp_norm[:,-1:-4:-1]
    temp_start = temp_norm[:,:ch_start]
    temp = temp_norm-np.mean(temp_start,axis=1)[:, np.newaxis]
    spectra_magnitude = np.mean(temp_end, axis=1)-np.mean(temp_start, axis=1)
    background_arr = np.outer(spectra_magnitude, background)

    spectra_standard = temp-background_arr
    if plot_example:
        spec_mean = np.mean(spectra_standard, axis=0)
        background_mean = np.mean(background_arr, axis=0)
        temp_mean = np.mean(temp, axis=0)
        plt.figure(figsize=(10, 6))
        plt.plot(spec_mean, label='Standardized Spectrum')
        plt.plot(background_mean, label='Background Estimate')
        plt.plot(temp_mean, label='Original Spectrum')
        plt.title('Spectral Standardization Example')
        plt.legend(['Original Spectrum', 'Background Estimate', 'Standardized Spectrum'])
        plt.show()

    spectra_max_idx = np.argmax(np.mean(spectra_standard,axis=0))
    spectra_norm = normalize(spectra_standard, max_val=np.mean(spectra_standard[:,spectra_max_idx])+3*np.std(spectra_standard[:,spectra_max_idx]),min_val=0)
    spectra = spectra_norm-np.median(spectra_norm[:,:ch_start],axis=1)[:,np.newaxis]

    return spectra

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
    num_samp = 81
    wn_1 = 2700
    wn_2 = 3100
    ch_start = int((2800 - wn_1) / (wn_2 - wn_1) * num_samp)
    
    srs_params_path = 'params_dataset/srs_params_61.npz'

    quant = True
    normalize = False
    flip = True
    bg_subtract = False
    show_plots = True
    
    if not bg_subtract:
        quant = True
    
    # Load SRS parameters to get background dataframe
    srs_data = np.load(srs_params_path)
    # background = srs_data['background']

    background = pd.read_csv(r'C:\Users\jvill\OneDrive\Documents\GitHub\hsi_machine_learning\unprocessed_data\water_HSI_76.csv')
    background = background.values.flatten()
    if flip:
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




    base_directories = [
        # r"E:\Data\Jorge\01_HumanKidney_HuBMAP\05-13-25 Xenium AKI 3782 K2200093_5\ROI1X left side\tifs",
        # r"E:\Data\Jorge\01_HumanKidney_HuBMAP\05-13-25 Xenium AKI 3782 K2200093_5\ROI2X bottom center\tifs",
        # r"E:\Data\Jorge\01_HumanKidney_HuBMAP\05-14-25 Xenium Ref KPMP_038 S-2211-0003960\ROI2X glomeruli middle\tifs",
        # r"E:\Data\Jorge\01_HumanKidney_HuBMAP\05-14-25 Xenium Ref KPMP_038 S-2211-0003960\ROI1X glomeruli top left\tifs",
        # r"E:\Data\Jorge\01_HumanKidney_HuBMAP\05-15-25 Xenium CKD 3990\ROI1 glomeruli bottom right\tifs",
        # r"E:\Data\Jorge\01_HumanKidney_HuBMAP\05-15-25 Xenium CKD 3990\ROI2 glomeruli middle\tifs"
        # r"D:\ADATA Backup\HuBMAP\HuBMAP Xenium\Xenium HSI\xenium-HSI",
        r"C:\Users\jvill\Downloads\KPMP_QC_images\HSI"


    ]

    for base_directory in tqdm(base_directories, desc="Processing Base Directories"):

        dataset = HSI_Unlabeled_Dataset(
            base_directory,
            ch_start,
            transform=None,
            image_normalization=True,
            min_max_normalization=False,
            num_samples=num_samp,
            wavenumber_start=wn_1,
            wavenumber_end=wn_2,
            compute_stats=True
        )
        # print('Saving macromolecule channels...')



        CHUNK_SIZE = 1_000_000

        for img_path in tqdm(dataset.img_list[:], desc="Processing images"):
            sample = os.path.basename(img_path)
            stats = dataset.image_stats[img_path]
            h, w = stats['height'], stats['width']
            n_pixels = h * w
            
            # Load as memmap
            img_mmap = tifffile.memmap(img_path, mode='r')
            # Handle channel axis (assume CYX or YXC based on dataset convention)
            if img_mmap.shape[0] == num_samp:
                image_flat = img_mmap.reshape(num_samp, -1).T
            else:
                image_flat = img_mmap.reshape(-1, num_samp)

            # Flip channels if needed (spectra order inversion)
            if flip:
                image_flat = np.flip(image_flat, axis=1)

            print(f"\nProcessing {sample} ({h}x{w})...")
            
            # # --- Pass 0: Global Raw Stats ---
            # print("  Pass 0: Calculating raw min/max...")
            # raw_min, raw_max = get_image_global_stats(image_flat, CHUNK_SIZE)
            # raw_diff = raw_max - raw_min + 1e-10

            if normalize:
                raw_min = stats['image_min']
                raw_max = stats['image_max']
                raw_diff = raw_max - raw_min + 1e-10

                # --- Pass 1: Standardized Stats ---
                print("  Pass 1: Calculating standardized channel stats...")
                sum_standard = np.zeros(num_samp, dtype=np.float64)
                sum_standard_sq = np.zeros(num_samp, dtype=np.float64)
                
                for i in range(0, n_pixels, CHUNK_SIZE):
                    chunk = (image_flat[i : i + CHUNK_SIZE].astype(np.float32) - raw_min) / raw_diff + 1e-6
                    # Standardization steps from spectral_standardization
                    # temp_end = chunk[:, -1:-4:-1]
                    temp_end = chunk[:, -1:-2:-1]
                    temp_start = chunk[:, :ch_start]
                    temp_start_mean = np.mean(temp_start, axis=1)[:, np.newaxis]
                    temp = chunk - temp_start_mean
                    
                    spectra_magnitude = np.mean(temp_end, axis=1) - np.mean(temp_start, axis=1)
                    if background.shape[0] != num_samp:
                        # Extrapolate background to match num_samp if needed by adding zeros to beginning
                        background_resized = np.zeros(num_samp)
                        background_resized[-background.shape[0]:] = background
                    else:
                        background_resized = background
                    
                    if bg_subtract:
                        background_arr = np.outer(spectra_magnitude, background_resized)
                        spectra_standard = temp - background_arr
                    else:
                        spectra_standard = temp
                    
                    sum_standard += np.sum(spectra_standard, axis=0)
                    sum_standard_sq += np.sum(spectra_standard**2, axis=0)
                    
                mean_standard = sum_standard / n_pixels
                std_standard = np.sqrt(np.maximum(sum_standard_sq / n_pixels - mean_standard**2, 0))
                
                spectra_max_idx = np.argmax(mean_standard)
                max_val = mean_standard[spectra_max_idx] + 3 * std_standard[spectra_max_idx]
                
            
            # --- Pass 2: Final Process and Save ---
            print("  Pass 2: Finalizing and writing output...")
            if bg_subtract:
                if normalize:
                    normalized_dir = os.path.join(os.path.dirname(os.path.dirname(img_path)), f'Normalized_BackgroundRemoved_{os.path.basename(os.path.dirname(img_path))}')
                else:
                    normalized_dir = os.path.join(os.path.dirname(os.path.dirname(img_path)), f'BackgroundRemoved_{os.path.basename(os.path.dirname(img_path))}')
                os.makedirs(normalized_dir, exist_ok=True)
                norm_img_path = os.path.join(normalized_dir, f"{sample}")
            
                # Pre-allocate output TIFF stack (CYX)
                norm_stack = tifffile.memmap(
                    norm_img_path, shape=(num_samp, h, w), dtype='float32', imagej=True,
                    metadata={'axes': 'CYX', 'unit': 'um'}
                )


            if quant:
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
                if normalize:
                    chunk_norm = (chunk_raw - raw_min) / raw_diff + 1e-6
                else:
                    chunk_norm = chunk_raw
                
                # temp_end = chunk_norm[:, -1:-4:-1]
                temp_end = chunk_norm[:, -1:-2:-1]
                temp_start = chunk_norm[:, :ch_start]
                temp_start_mean = np.mean(temp_start, axis=1)[:, np.newaxis]
                temp = chunk_norm - temp_start_mean


                
                spectra_magnitude = np.mean(temp_end, axis=1) - np.mean(temp_start, axis=1)
                if background.shape[0] != num_samp:
                    background_resized = np.zeros(num_samp)
                    background_resized[-background.shape[0]:] = background
                else:
                    background_resized = background

                if bg_subtract:
                    background_arr = np.outer(spectra_magnitude, background_resized)
                    spectra_standard = temp - background_arr
                else:
                    spectra_standard = temp

                # if quant:
                #     spectral_std = spectra_standard.std(axis=1)
                #     threshold = np.mean(spectra_standard[:, :ch_start].std(axis=1)) + 2 * np.std(spectra_standard[:, :ch_start].std(axis=1))

                #     valid_mask = spectral_std > threshold
                #     spectra_standard[~valid_mask] = 0  # Zero out low-variance spectra

                if i == 0 and show_plots:  # Plot example from first chunk
                    temp_mean = np.mean(temp, axis=0)
                    spec_mean = np.mean(spectra_standard, axis=0)
                    plt.figure(figsize=(10, 6))
                    plt.plot(temp_mean, label='Original Spectrum')
                    if bg_subtract:
                        background_mean = np.mean(background_arr, axis=0)
                        plt.plot(background_mean, label='Background Estimate')
                        plt.plot(spec_mean, label='Standardized Spectrum')
                        plt.title('Spectral Standardization Example')
                        plt.legend(['Original Spectrum', 'Background Estimate', 'Standardized Spectrum'])
                    else:
                        plt.title('Spectral Standardization Example (No Background Subtraction)')
                        plt.legend(['Original/Standardized Spectrum'])
                    plt.show()


                
                if normalize:
                # Final max-normalization
                    spectra_norm = (spectra_standard - 0) / (max_val - 0) + 1e-6
                    spectra = spectra_norm - np.median(spectra_norm[:, :ch_start], axis=1)[:, np.newaxis]
                else:
                    spectra = spectra_standard - np.median(spectra_standard[:, :ch_start], axis=1)[:, np.newaxis]
                    # spectra = np.maximum(spectra, 0)  # Ensure non-negativity if not normalizing
                
                #Visualize macromolecule indeces on the first chunk
                if i == 0 and quant and show_plots:
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

                if quant:
                    # Map extraction (Max in range)
                    protein_map.reshape(-1)[i:end_i] = np.max(spectra[:, protein_idx-2:protein_idx+2], axis=1)
                    sat_map.reshape(-1)[i:end_i] = np.max(spectra[:, sat_idx-2:sat_idx+2], axis=1)
                    lipid_map.reshape(-1)[i:end_i] = np.max(spectra[:, lipid_idx-2:lipid_idx+2], axis=1)
                    unsat_map.reshape(-1)[i:end_i] = np.max(spectra[:, unsat_idx-2:unsat_idx+2], axis=1)

                if bg_subtract:
                    # Write to 3D stack
                    for ch_idx in range(num_samp):
                        norm_stack.reshape(num_samp, -1)[ch_idx, i:end_i] = spectra[:, ch_idx]
                    norm_stack.flush()
                
                if quant:
                    # Determine macromolecule output dir prefix based on parameters
                    if bg_subtract:
                        if normalize:
                            folder_prefix = "Normalized_BackgroundRemoved_Images"
                        else:
                            folder_prefix = "BackgroundRemoved_Images"
                    else:
                        if normalize:
                            folder_prefix = "Normalized_Images"
                        else:
                            folder_prefix = "Images"

                    output_dir = os.path.join(os.path.dirname(os.path.dirname(img_path)), os.path.join(f'{folder_prefix}_{os.path.basename(os.path.dirname(img_path))}', f"{sample.replace('.tif','')}"))
                    os.makedirs(output_dir, exist_ok=True)
                    
                    protein_map = np.maximum(protein_map, 0)
                    sat_map = np.maximum(sat_map, 0)
                    lipid_map = np.maximum(lipid_map, 0)
                    unsat_map = np.maximum(unsat_map, 0)
                    
                    # Helper for resolution
                    pixel_size_x = stats['pixel_size_x']
                    pixel_size_y = stats['pixel_size_y']
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
                    if bg_subtract:
                        if normalize:
                            ratio_folder_prefix = "Ratio_Normalized_BackgroundRemoved"
                        else:
                            ratio_folder_prefix = "Ratio_BackgroundRemoved"
                    else:
                        if normalize:
                            ratio_folder_prefix = "Ratio_Normalized"
                        else:
                            ratio_folder_prefix = "Ratio"

                    ratio_dir = os.path.join(os.path.dirname(os.path.dirname(img_path)), f'{ratio_folder_prefix}_{os.path.basename(os.path.dirname(img_path))}', f"{sample.replace('.tif','')}")
                    os.makedirs(ratio_dir, exist_ok=True)
                    tifffile.imwrite(os.path.join(ratio_dir, f"{sample.replace('.tif','')}-Lipid_Unsaturation-ratio.tif"), unsat_ratio.astype('float32'), resolution=res, imagej=True, metadata=meta)
                    tifffile.imwrite(os.path.join(ratio_dir, f"{sample.replace('.tif','')}-Lipid_to_Protein-ratio.tif"), lipid_ratio.astype('float32'), resolution=res, imagej=True, metadata=meta)

            # Cleanup
            if bg_subtract:
                del norm_stack
            del img_mmap
            import gc
            gc.collect()

if __name__ == "__main__":
    main()
    print("Processing complete.")