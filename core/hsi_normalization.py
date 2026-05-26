import numpy as np
import os
import sys
from tqdm import tqdm
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
        index = np.random.randint(0, spectra_standard.shape[0], size=1)
        plt.plot(temp[index[0]])
        plt.plot(background_arr[index[0]])
        plt.plot(spectra_standard[index[0]])
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
    base_directory = r"C:\Users\jvill\Documents\integrated_pipeline\HSI_data\data_processed"
    num_samp = 61
    wn_1 = 2700
    wn_2 = 3100
    ch_start = int((2800 - wn_1) / (wn_2 - wn_1) * num_samp)
    shift = 20  
    
    srs_params_path = 'params_dataset/srs_params_61.npz'

    # Load SRS parameters to get background dataframe
    srs_data = np.load(srs_params_path)
    background = srs_data['background']

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
    print('Saving macromolecule channels...')

    unsat_idx = macro_idx(3015 + shift, num_samp, wn_1, wn_2)
    protein_idx = macro_idx(2930 + shift, num_samp, wn_1, wn_2)
    sat_idx = macro_idx(2885 + shift, num_samp, wn_1, wn_2)
    lipid_idx = macro_idx(2850 + shift, num_samp, wn_1, wn_2)
    print(f"unsat idx: {unsat_idx}")
    print(f"protein idx: {protein_idx}")
    print(f"sat idx: {sat_idx}")
    print(f"lipid idx: {lipid_idx}")

    CHUNK_SIZE = 1_000_000

    for img_path in tqdm(dataset.img_list, desc="Processing images"):
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
        image_flat = np.flip(image_flat, axis=1)

        print(f"\nProcessing {sample} ({h}x{w})...")
        
        # # --- Pass 0: Global Raw Stats ---
        # print("  Pass 0: Calculating raw min/max...")
        # raw_min, raw_max = get_image_global_stats(image_flat, CHUNK_SIZE)
        # raw_diff = raw_max - raw_min + 1e-10

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
            temp_end = chunk[:, -1:-4:-1]
            temp_start = chunk[:, :ch_start]
            temp_start_mean = np.mean(temp_start, axis=1)[:, np.newaxis]
            temp = chunk - temp_start_mean
            
            spectra_magnitude = np.mean(temp_end, axis=1) - np.mean(temp_start, axis=1)
            background_arr = np.outer(spectra_magnitude, background)
            spectra_standard = temp - background_arr
            
            sum_standard += np.sum(spectra_standard, axis=0)
            sum_standard_sq += np.sum(spectra_standard**2, axis=0)
            
        mean_standard = sum_standard / n_pixels
        std_standard = np.sqrt(np.maximum(sum_standard_sq / n_pixels - mean_standard**2, 0))
        
        spectra_max_idx = np.argmax(mean_standard)
        max_val = mean_standard[spectra_max_idx] + 3 * std_standard[spectra_max_idx]
        
        
        # --- Pass 2: Final Process and Save ---
        print("  Pass 2: Finalizing and writing output...")
        normalized_dir = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'Normalized')
        os.makedirs(normalized_dir, exist_ok=True)
        norm_img_path = os.path.join(normalized_dir, f"{sample}")
        
        # Pre-allocate output TIFF stack (CYX)
        norm_stack = tifffile.memmap(
            norm_img_path, shape=(num_samp, h, w), dtype='float32', imagej=True,
            metadata={'axes': 'CYX', 'unit': 'um'}
        )
        
        # Pre-allocate macro-maps in memory (usually small enough)
        protein_map = np.zeros((h, w), dtype='float32')
        sat_map = np.zeros((h, w), dtype='float32')
        lipid_map = np.zeros((h, w), dtype='float32')
        unsat_map = np.zeros((h, w), dtype='float32')
        
        for i in tqdm(range(0, n_pixels, CHUNK_SIZE), desc="Writing chunks", leave=False):
            end_i = min(i + CHUNK_SIZE, n_pixels)
            chunk_raw = image_flat[i : end_i].astype(np.float32)
            chunk_norm = (chunk_raw - raw_min) / raw_diff + 1e-6
            
            temp_end = chunk_norm[:, -1:-4:-1]
            temp_start = chunk_norm[:, :ch_start]
            temp_start_mean = np.mean(temp_start, axis=1)[:, np.newaxis]
            temp = chunk_norm - temp_start_mean
            
            spectra_magnitude = np.mean(temp_end, axis=1) - np.mean(temp_start, axis=1)
            background_arr = np.outer(spectra_magnitude, background)
            spectra_standard = temp - background_arr
            
            # Final max-normalization
            spectra_norm = (spectra_standard - 0) / (max_val - 0) + 1e-6
            spectra = spectra_norm - np.median(spectra_norm[:, :ch_start], axis=1)[:, np.newaxis]
            
            # Write to 3D stack
            for ch_idx in range(num_samp):
                norm_stack.reshape(num_samp, -1)[ch_idx, i:end_i] = spectra[:, ch_idx]
                
            # Update macro-maps
            chunk_h = (end_i - i) // w # This might be tricky for partial rows, simpler to use indices
            
            # Indices in the 2D maps
            y_coords = np.arange(i, end_i) // w
            x_coords = np.arange(i, end_i) % w
            
            # Map extraction (Max in range)
            protein_map.reshape(-1)[i:end_i] = np.max(spectra[:, protein_idx-2:protein_idx+2], axis=1)
            sat_map.reshape(-1)[i:end_i] = np.max(spectra[:, sat_idx-2:sat_idx+2], axis=1)
            lipid_map.reshape(-1)[i:end_i] = np.max(spectra[:, lipid_idx-2:lipid_idx+2], axis=1)
            unsat_map.reshape(-1)[i:end_i] = np.max(spectra[:, unsat_idx-2:unsat_idx+2], axis=1)

        norm_stack.flush()
        
        # Save macromolecule images
        output_dir = os.path.join(os.path.dirname(os.path.dirname(img_path)), os.path.join('Normalized_Images', f"{sample.replace('.tif','')}"))
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
        p_img[p_img > 50] = 0
        l_img = np.maximum(5 * lipid_map - 0.4 * protein_map, 0)
        l_img[l_img > 50] = 0
        
        tifffile.imwrite(os.path.join(output_dir, f"P_image-{sample}"), p_img, resolution=res, imagej=True, metadata=meta)
        tifffile.imwrite(os.path.join(output_dir, f"L_image-{sample}"), l_img, resolution=res, imagej=True, metadata=meta)
        
        unsat_ratio = np.where(sat_map == 0, 0, unsat_map / (sat_map + unsat_map + 1e-6))
        lipid_ratio = np.where(protein_map == 0, 0, lipid_map / (protein_map + 1e-6))
        
        ratio_dir = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'processed-Ratio', f"{sample.replace('.tif','')}")
        os.makedirs(ratio_dir, exist_ok=True)
        tifffile.imwrite(os.path.join(ratio_dir, f"{sample.replace('.tif','')}-Lipid_Unsaturation-ratio.tif"), unsat_ratio.astype('float32'), resolution=res, imagej=True, metadata=meta)
        tifffile.imwrite(os.path.join(ratio_dir, f"{sample.replace('.tif','')}-Lipid_to_Protein-ratio.tif"), lipid_ratio.astype('float32'), resolution=res, imagej=True, metadata=meta)

        # Cleanup
        del img_mmap, norm_stack
        import gc
        gc.collect()

if __name__ == "__main__":
    main()
    print("Processing complete.")