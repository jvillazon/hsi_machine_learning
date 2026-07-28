import os
import re
import numpy as np
import pandas as pd
import tifffile
from scipy.interpolate import CubicSpline
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns


# =========================================================================
# CONFIGURATION
# =========================================================================

image_dir = r"E:\integrated_pipeline\HSI_data\data_processed"
base_dir = os.path.dirname(image_dir)
output_dir = os.path.join(base_dir, f'{os.path.basename(image_dir)}_sam_outputs_molecules')
os.makedirs(output_dir, exist_ok=True)

CONFIG = {
    'image_dir': image_dir,
    'output_dir': output_dir,
    'spectra_csv': r'C:\Users\jvill\OneDrive\Documents\GitHub\hsi_machine_learning\unprocessed_data\averaged_spectra.csv',
    'background_csv' : r'C:\Users\jvill\OneDrive\Documents\GitHub\hsi_machine_learning\unprocessed_data\water_HSI_71.csv',
    'wavenumber_start': 2700,
    'wavenumber_end': 3100,
    'num_samples': 61,
    'max_shift_cm': 50,  # Max shift in cm^-1
    'similarity_threshold': 0.8, # Min similarity (0-1) to be considered a match
    'chunk_size': 1_000_000, # Number of pixels per batch for large images
    'max_csv_pixels': 10_000_000, # Skip full CSV if image is larger than this
    'export_individual_stack_images': True,
    'exclude_molecules': [],
    'mol_groups': None,
    # 'mol_groups': {
    #     'Sterols': [
    #         'Cholesterol (ovine)'
    #         ],
    #     'Sterol Esters': [
    #         '18:1 Cholesterol Ester'
    #     ],
    #     'Cardiolipins': [
    #         '16:0 Cardiolipin'
    #         ],
    #     'Phospholipids': [
    #         'PC Mix', 
    #         'DOPC', 
    #         'DSPC',
    #         'PE Mix', 
    #         'DOPE',
    #         'PS Mix',
    #         'PI Mix',
    #         'PG Mix'
    #         ],
    #     'Lysophospholipids': [
    #         '18:0 Lyso PA', 
    #         '18:1 LPA'
    #         ],
    #     'Diglycerides': [
    #         'DAG 16:0', 
    #         'DAG 18:0 and 24:0 '
    #         '16:0 CDP DG'
    #         ],
    #     'Triglycerides': [
    #         'TAG Mix'
    #         'TAG 16:0', 
    #         'TAG 18:1',
    #         ],
    #     'Sphingolipids': [
    #         'Sphingosine'
    #         'Cer 18:1-12:0'
    #         'Cer 18:1-18:0', 
    #         'Cer 18:1-18:1'
    #         'Cer 18:1-22:0', 
    #         'Cer 18:1-24:0', 
    #         'Cer 18:1-24:1'
    #         ],
    #     'Glucosylsphingolipids': [
    #         "Glucosylceramide (Gaucher's Spleen)"
    #         ],
    #     'Dihydroceramides': [
    #         'Cer 18:0-14:0', 
    #         'Cer 18:1-24:1(15Z)'
    #         ],
    #     'Deoxyceramides' : ['Cer m18:1-16:0', 'Cer m18:1-24:1'],
    #     'SFA': ['Stearic Acid', 'Palmitic Acid'],
    #     'PUFA': ['Docosahexaenoic Acid', 'Tetracosapentaenoic Acid'],
    #     'Carnitines': [
    #         'C12 Carnitine', 
    #         'C16 Carnitine', 
    #         'C18 Carnitine (d9-cis)'
    #         ],
    #     'Background' : ['Kidney Background']
    # },
    # 'mol_groups': {
    #     'Background': ['Kidney Background'],
    #     'Sterols': [
    #         '18:1 Cholesterol Ester', 
    #         'Cholesterol (ovine)'
    #         ],
    #     'Phospholipids': [
    #         '16:0 Cardiolipin', 
    #         'PC Mix', 
    #         'PE Mix', 
    #         'PS Mix', 
    #         'PI Mix', 
    #         'PG Mix', 
    #         'DOPC', 
    #         'DOPE', 
    #         'DSPC', 
    #         '18:0 Lyso PA', 
    #         '18:1 LPA'
    #         ],
    #     'Glycerols': [
    #         '16:0 CDP DG',
    #         'DAG 16:0',
    #         'DAG 18:0 and 24:0 ',
    #         'TAG Mix',
    #         'TAG 16:0',
    #         'TAG 18:1'
    #     ],
    #     'Sphingolipids': [
    #         'Sphingosine',
    #         'Cer 18:1-12:0',
    #         'Cer 18:1-18:0',
    #         'Cer 18:1-18:1',
    #         'Cer 18:1-22:0',
    #         'Cer 18:1-24:0',
    #         'Cer 18:1-24:1',
    #         "Glucosylceramide (Gaucher's Spleen)",
    #         'Cer 18:0-14:0',
    #         'Cer 18:1-24:1(15Z)',
    #         'Cer m18:1-16:0',
    #         'Cer m18:1-24:1'
    #     ],
    #     'Fatty Acids': [
    #         'Stearic Acid',
    #         'Palmitic Acid',
    #         'Docosahexaenoic Acid',
    #         'Tetracosapentaenoic Acid'
    #     ],
    #     'Carnitines': [
    #         'C8 L-Carnitine',
    #         'C12 Carnitine',
    #         'C16 Carnitine',
    #         'C18 Carnitine (d9-cis)'
    #     ]
    # },
}



# Derived configuration
WN_STEP = (CONFIG['wavenumber_end'] - CONFIG['wavenumber_start']) / (CONFIG['num_samples'] - 1)
MAX_SHIFT_CHANNELS = int(np.ceil(CONFIG['max_shift_cm'] / WN_STEP))
CH_START = int((2800 - CONFIG['wavenumber_start']) / WN_STEP)

# =========================================================================
# UTILITIES
# =========================================================================

def normalize_spectra(spectra):
    """Min-max normalize spectra (n_samples, n_channels) or (n_channels,)"""
    if spectra.ndim == 1:
        min_v = np.min(spectra)
        max_v = np.max(spectra)
        return (spectra - min_v) / (max_v - min_v + 1e-10)
    else:
        min_v = np.min(spectra, axis=1, keepdims=True)
        max_v = np.max(spectra, axis=1, keepdims=True)
        return (spectra - min_v) / (max_v - min_v + 1e-10)

def load_reference_spectra(csv_path, wn_start, wn_end, num_samp):
    """Load and interpolate reference spectra from CSV."""
    df = pd.read_csv(csv_path)
    df = df.dropna(axis='columns', how='all')
    
    col_names = df.columns
    num_mol = len(col_names) // 2
    
    target_wn = np.linspace(wn_start, wn_end, num_samp)
    ref_spectra = {}
    
    for i in range(0, len(col_names), 2):
        name = col_names[i].strip()
        mol_wn = df.iloc[:, i].values
        mol_int = df.iloc[:, i+1].values
        
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
        
        # Interpolate
        try:
            cs = CubicSpline(mol_wn, mol_int, extrapolate=True)
            interp_int = cs(target_wn)
            ref_spectra[name] = normalize_spectra(interp_int)
        except Exception as e:
            print(f"Failed to interpolate {name}: {e}")
            
    return ref_spectra

def calculate_sam(query, references):
    """
    Calculate Spectral Angle (SAM) between query and references.
    query: (n_pixels, n_channels)
    references: (n_molecules, n_channels)
    Returns: (n_pixels, n_molecules) angles in radians
    """
    # Normalize vectors to unit length
    q_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-10)
    r_norm = references / (np.linalg.norm(references, axis=1, keepdims=True) + 1e-10)
    
    # Dot product
    cos_theta = np.dot(q_norm, r_norm.T)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    return np.arccos(cos_theta) # Small angles mean very similar (good matches) 

def estimate_best_shift(image_mean, background_ref, max_shift_ch):
    """
    Estimate best global shift by correlating background_ref with image_mean.
    Window: High intensity region (> 2800 cm^-1).
    """
    best_corr = -1
    best_shift = 0
    
    # We focus on the region after CH_START (the CH stretch region)
    region_mask = slice(CH_START, None)
    img_region = image_mean[region_mask]
    img_region = (img_region - np.mean(img_region)) / (np.std(img_region) + 1e-10)
    
    for s in range(-max_shift_ch, max_shift_ch + 1):
        # Shift background ref
        shifted_ref = np.zeros_like(background_ref)
        if s > 0:
            shifted_ref[s:] = background_ref[:-s]
        elif s < 0:
            shifted_ref[:s] = background_ref[-s:]
        else:
            shifted_ref = background_ref
            
        ref_region = shifted_ref[region_mask]
        ref_region = (ref_region - np.mean(ref_region)) / (np.std(ref_region) + 1e-10)
        
        corr = np.mean(img_region * ref_region)
        if corr > best_corr:
            best_corr = corr
            best_shift = s
            
    return best_shift

def plot_diagnostic_samples(out_dir, image_mmap, shifted_ref_matrix, mol_names, ch_order_flip=False):
    """
    Pick best, worst, and random matches to visualize SAM performance for large images.
    Since angles/similarity are not globally stored, we perform a quick scan of 1M random pixels.
    """
    print("Generating diagnostic plots from sampled pixels...")
    h, w = image_mmap.shape[-2], image_mmap.shape[-1]
    n_pixels = h * w
    
    # Sample 1M pixels (or all if fewer)
    sample_size = min(n_pixels, 1_000_000)
    indices = np.random.choice(n_pixels, sample_size, replace=False)
    
    # Load and process sample
    image_flat = image_mmap.reshape(image_mmap.shape[0], -1).T
    chunk = image_flat[indices]
    if ch_order_flip:
        chunk = np.flip(chunk, axis=1)
    chunk = chunk.astype(np.float32)
    
    # Baseline subtraction
    silent_baseline = np.mean(chunk[:, :CH_START], axis=1, keepdims=True)
    chunk_proc = chunk - silent_baseline
    
    # SAM
    angles = calculate_sam(chunk_proc, shifted_ref_matrix)
    similarity = 1.0 - (angles / np.pi)
    best_match_idx = np.argmin(angles, axis=1)
    best_similarity = np.max(similarity, axis=1)
    
    # Pick 3 samples from the chunk
    diag_indices = [
        np.argmax(best_similarity), # Best in sample
        np.argmin(best_similarity), # Worst in sample
        np.random.randint(0, sample_size) # Random in sample
    ]
    diag_labels = ['Best_Match', 'Worst_Match', 'Random_Match']
    
    wn = np.linspace(CONFIG['wavenumber_start'], CONFIG['wavenumber_end'], CONFIG['num_samples'])
    
    for local_idx, label in zip(diag_indices, diag_labels):
        global_p_idx = indices[local_idx]
        q = chunk_proc[local_idx]
        m_idx = best_match_idx[local_idx]
        r = shifted_ref_matrix[m_idx]
        angle = angles[local_idx, m_idx]
        
        # Unit normalization for visual comparison
        q_unit = q / (np.linalg.norm(q) + 1e-10)
        r_unit = r / (np.linalg.norm(r) + 1e-10)
        
        plt.figure(figsize=(10, 5))
        plt.plot(wn, q_unit, 'b-', label='Pixel Spectrum (Unit Vector)')
        plt.plot(wn, r_unit, 'r--', label=f'Matched Ref: {mol_names[m_idx]} (Unit Vector)')
        
        plt.title(f"{label} (Pixel {global_p_idx})\nAngle: {angle:.4f} rad ({np.degrees(angle):.2f} deg)")
        plt.xlabel("Wavenumber (cm^-1)")
        plt.ylabel("Unit Normalized Intensity")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        out_path = os.path.join(out_dir, f"diagnostic_{label.lower()}.png")
        plt.savefig(out_path, dpi=100)
        plt.close()
        
    return indices, best_match_idx, similarity # Return samples for summary plot if needed


def sanitize_name_for_filename(name):
    """Sanitize class/target names to safe filenames."""
    sanitized = re.sub(r'[^\w\-\. ]', '', str(name)).strip()
    sanitized = sanitized.replace(' ', '_')
    return sanitized if sanitized else 'unnamed'


def export_stack_slices_as_tiff(stack_mmap, target_names, out_dir, subfolder):
    """Export each target slice from a stack as an individual TIFF image."""
    slices_dir = os.path.join(out_dir, subfolder)
    os.makedirs(slices_dir, exist_ok=True)

    for t_idx, target_name in enumerate(target_names):
        safe_name = sanitize_name_for_filename(target_name)
        out_path = os.path.join(slices_dir, f"{safe_name}.tif")
        tifffile.imwrite(out_path, stack_mmap[t_idx])



if __name__ == '__main__':

    ref_dict = load_reference_spectra(
        CONFIG['spectra_csv'], 
        CONFIG['wavenumber_start'], 
        CONFIG['wavenumber_end'], 
        CONFIG['num_samples']
    )
    
    # calibration anchor
    bg_ref_name = 'Kidney Background'
    if bg_ref_name not in ref_dict:
        bg_ref_name = next((k for k in ref_dict.keys() if 'background' in k.lower()), list(ref_dict.keys())[0])
        print(f"Warning: '{bg_ref_name}' not found, using '{bg_ref_name}' as calibration anchor.")
    
    bg_ref = ref_dict[bg_ref_name]
    mol_names = sorted(list(ref_dict.keys()))
    
    # Pre-calculate grouping mapping
    if len(CONFIG['exclude_molecules']) > 0:
        mol_names = [m for m in mol_names if m not in CONFIG['exclude_molecules']]
        ref_matrix = np.array([ref_dict[name] for name in mol_names])
        mol_to_idx_global = {name: i for i, name in enumerate(mol_names)}
    else:
        ref_matrix = np.array([ref_dict[name] for name in mol_names])
        mol_to_idx_global = {name: i for i, name in enumerate(mol_names)}

    use_groups = 'mol_groups' in CONFIG and CONFIG['mol_groups']
    if use_groups:
        target_names = list(CONFIG['mol_groups'].keys())
        group_to_mols = {i: [mol_to_idx_global[m] for m in CONFIG['mol_groups'][name] if m in mol_to_idx_global]
                        for i, name in enumerate(target_names)}
    else:
        target_names = mol_names

    image_files = [f for f in os.listdir(CONFIG['image_dir']) 
                if f.endswith(('.tif', '.tiff')) and not f.startswith('._')]
    print(f"Found {len(image_files)} images.")
    PALETTE = sns.color_palette("husl", len(target_names))

    # Initialize background:
    background = pd.read_csv(CONFIG['background_csv'])
    background = np.flip(background.values.flatten()) # Flip to orient with images

    for img_file in image_files:
        img_path = os.path.join(CONFIG['image_dir'], img_file)


        img_base = os.path.splitext(img_file)[0]
        img_base = sanitize_name_for_filename(img_base)

        img_out_dir = os.path.normpath(os.path.join(output_dir, img_base))
        os.makedirs(img_out_dir, exist_ok=True)

        print(f"\nProcessing {img_file}...")

        # Load image as memmap
        image_mmap = tifffile.memmap(img_path, mode='r')
        with tifffile.TiffFile(img_path) as tif:
            if tif.is_imagej:
                page = tif.pages[0]
                pixel_size_x = 1 / page.resolution[1]
                pixel_size_y = 1 / page.resolution[0]
            else:
                pixel_size_x = 1
                pixel_size_y = 1
        res = (1 / pixel_size_x, 1 / pixel_size_y)
        meta = {'axes': 'YX', 'unit': 'um', 'PhysicalSizeX': pixel_size_x, 'PhysicalSizeY': pixel_size_y}
        print(f"  Image shape: {image_mmap.shape}")

        # Identify channel order
        ch_axis = -1
        if image_mmap.ndim == 3:
            if image_mmap.shape[0] == CONFIG['num_samples']:
                ch_axis = 0
                ch, h, w = image_mmap.shape
            elif image_mmap.shape[2] == CONFIG['num_samples']:
                ch_axis = 2
                h, w, ch = image_mmap.shape
            else:
                print(f"  Skipping: Could not find {CONFIG['num_samples']} channels.")
                continue
        else:
            print(f"  Skipping: Expected 3D HSI stack.")
            continue

        n_pixels = h * w
        image_flat = image_mmap.reshape(ch, -1).T if ch_axis == 0 else image_mmap.reshape(-1, ch)
        image_flat = np.flip(image_flat)


        # Stage 2: Batched Shift Estimation
        print("  Estimating global shift in batches...")
        sum_spectra = np.zeros(CONFIG['num_samples'], dtype=np.float64)
        for start_p in range(0, n_pixels, CONFIG['chunk_size']):
            end_p = min(start_p + CONFIG['chunk_size'], n_pixels)
            sum_spectra += np.sum(image_flat[start_p:end_p].astype(np.float32), axis=0)

        image_mean = (sum_spectra / n_pixels).astype(np.float32)
        image_mean_norm = normalize_spectra(image_mean)

        best_shift = estimate_best_shift(image_mean_norm, bg_ref, MAX_SHIFT_CHANNELS)
        print(f"  Estimated best shift: {best_shift} channels (~{best_shift * WN_STEP:.2f} cm^-1)")



        # Shift background
        if best_shift > 0:
            background = np.pad(background, (best_shift, 0), mode='constant')
        # Interpolate background to match num_samp if needed
        if background.shape[0] != CONFIG['num_samples']:
            background_interp = np.interp(
                np.linspace(0, background.shape[0]-1, CONFIG['num_samples']),
                np.arange(background.shape[0]),
                background.flatten()
            )
        else:
            background_interp = background


        # Apply shift to references
        shifted_ref_matrix = np.zeros_like(ref_matrix)
        for i in range(len(mol_names)):
            if best_shift > 0:
                shifted_ref_matrix[i, best_shift:] = ref_matrix[i, :-best_shift]
            elif best_shift < 0:
                shifted_ref_matrix[i, :best_shift] = ref_matrix[i, -best_shift:]
            else:
                shifted_ref_matrix[i] = ref_matrix[i]

        # Batched SAM and Classification
        print(f"  Calculating SAM and maps in chunks of {CONFIG['chunk_size']} pixels...")

        # Prepare output paths
        sim_stack_path = os.path.normpath(os.path.join(img_out_dir, f"{img_base}_sam_similarity.tif"))
        class_map_path = os.path.normpath(os.path.join(img_out_dir, f"{img_base}_sam_classification.tif"))
        class_stack_path = os.path.normpath(os.path.join(img_out_dir, f"{img_base}_sam_class_stack.tif"))
        rgb_stack_path = os.path.normpath(os.path.join(img_out_dir, f"{img_base}_sam_rgb.tif"))

        sim_best_path = os.path.normpath(os.path.join(img_out_dir, f"{img_base}_sam_best_similarity.tif"))

        # Create TIFF-backed memmaps for outputs (no separate .dat temp files)
        sim_stack_mmap = tifffile.memmap(sim_stack_path, shape=(len(target_names), h, w), dtype='float32')
        class_stack_mmap = tifffile.memmap(class_stack_path, shape=(len(target_names), h, w), dtype='uint8')
        rgb_stack_mmap = tifffile.memmap(rgb_stack_path, shape=(3, h, w), dtype='uint8')
        class_map_mmap = tifffile.memmap(class_map_path, shape=(h, w), dtype='uint16')
        best_sim_mmap = tifffile.memmap(sim_best_path, shape=(h, w), dtype='float32')

        # Ensure deterministic output values where pixels are below threshold.
        sim_stack_mmap[:] = 0
        class_stack_mmap[:] = 0
        rgb_stack_mmap[:] = 0
        class_map_mmap[:] = 0
        best_sim_mmap[:] = 0

        top_k = 5
        all_top_indices = np.zeros((n_pixels, top_k), dtype=np.uint16)
        all_top_sims = np.zeros((n_pixels, top_k), dtype=np.float32)
        save_csv = n_pixels < CONFIG['max_csv_pixels']

        for start_p in tqdm(range(0, n_pixels, CONFIG['chunk_size']), desc="        Processing batches"):
            end_p = min(start_p + CONFIG['chunk_size'], n_pixels)
            chunk = image_flat[start_p:end_p].astype(np.float32)

            temp_end = chunk[:, -1:-2:-1]
            temp_start = chunk[:, :CH_START]
            spectra_magnitude = np.mean(temp_end, axis=1) - np.mean(temp_start, axis=1)

            silent_baseline = np.mean(temp_start, axis=1, keepdims=True)
            chunk_proc = chunk - silent_baseline

            # Subtract background
            background_arr = np.outer(spectra_magnitude, background_interp)
            chunk_bg_r = chunk_proc - background_arr

            chunk_angles = calculate_sam(chunk_bg_r, shifted_ref_matrix)

            if use_groups:
                n_groups = len(target_names)
                effective_angles = np.full((len(chunk_angles), n_groups), np.pi, dtype=np.float32)
                for g_idx, m_indices in group_to_mols.items():
                    if m_indices:
                        effective_angles[:, g_idx] = np.min(chunk_angles[:, m_indices], axis=1)
            else:
                effective_angles = chunk_angles

            chunk_similarity = 1.0 - (effective_angles / np.pi)
            # Choose best candidate, optionally skipping the explicit background class.
            sorted_indices = np.argsort(effective_angles, axis=1)
            background_candidates = [name for name in target_names if 'background' in name.lower()]
            if background_candidates:
                background_idx = target_names.index(background_candidates[0])
                first_idx = sorted_indices[:, 0]
                second_idx = sorted_indices[:, 1] if sorted_indices.shape[1] > 1 else sorted_indices[:, 0]
                candidate_idx = np.where(first_idx == background_idx, second_idx, first_idx)
            else:
                candidate_idx = sorted_indices[:, 0]

            candidate_sim = chunk_similarity[np.arange(len(candidate_idx)), candidate_idx]
            valid_mask = candidate_sim > CONFIG['similarity_threshold']

            # Class 0 is reserved for "No Match".
            chunk_best_idx = np.zeros(len(candidate_idx), dtype=np.uint16)
            chunk_best_idx[valid_mask] = (candidate_idx[valid_mask] + 1).astype(np.uint16)

            chunk_best_sim = np.zeros(len(candidate_idx), dtype=np.float32)
            chunk_best_sim[valid_mask] = candidate_sim[valid_mask].astype(np.float32)

            top_k_indices = np.argsort(effective_angles, axis=1)[:, :top_k]
            all_top_indices[start_p:end_p] = top_k_indices
            all_top_sims[start_p:end_p] = 1.0 - (effective_angles[np.arange(len(effective_angles))[:, None], top_k_indices] / np.pi)

            # Write into TIFF memmaps
            # class_map stored as 2D (h, w)
            class_map_mmap.reshape(-1)[start_p:end_p] = chunk_best_idx
            best_sim_mmap.reshape(-1)[start_p:end_p] = chunk_best_sim

            sim_stack_flat = sim_stack_mmap.reshape(len(target_names), -1)
            class_stack_flat = class_stack_mmap.reshape(len(target_names), -1)
            rgb_chunk = np.zeros((len(candidate_idx), 3), dtype=np.uint8)
            palette_u8 = (np.array(PALETTE) * 255).astype(np.uint8)

            for t_idx in range(len(target_names)):
                sim_stack_flat[t_idx, start_p:end_p] = chunk_similarity[:, t_idx]
                best_match_mask = valid_mask & (candidate_idx == t_idx)
                class_stack_flat[t_idx, start_p:end_p] = best_match_mask.astype(np.uint8)
                rgb_chunk[best_match_mask] = palette_u8[t_idx]

            rows = np.arange(start_p, end_p)
            ys = (rows // w).astype(int)
            xs = (rows % w).astype(int)
            rgb_stack_mmap[0, ys, xs] = rgb_chunk[:, 0]
            rgb_stack_mmap[1, ys, xs] = rgb_chunk[:, 1]
            rgb_stack_mmap[2, ys, xs] = rgb_chunk[:, 2]

        # Flush memmaps
        sim_stack_mmap.flush()
        class_stack_mmap.flush()
        rgb_stack_mmap.flush()
        class_map_mmap.flush()
        best_sim_mmap.flush()

        if CONFIG.get('export_individual_stack_images', False):
            export_stack_slices_as_tiff(sim_stack_mmap, target_names, img_out_dir, 'similarity_slices')
            export_stack_slices_as_tiff(class_stack_mmap, target_names, img_out_dir, 'classification_slices')

        # Save target names and RGB to CSV
        target_names_id = target_names.copy()
        target_names_id.insert(0, "No Match")
        target_names_hexcodes = [f"#{int(PALETTE[i][0]*255):02x}{int(PALETTE[i][1]*255):02x}{int(PALETTE[i][2]*255):02x}" for i, name in enumerate(target_names)]
        target_names_hexcodes.insert(0, "#000000")
        target_names_df = pd.DataFrame({"ID Number": list(range(len(target_names_id))), "Molecules": target_names_id, "Hexcodes": target_names_hexcodes})
        target_names_df.to_csv(os.path.normpath(os.path.join(img_out_dir, f"{img_base}_target_names.csv")), index=False)

        indices, best_match_idx_sample, similarity_sample = plot_diagnostic_samples(
            img_out_dir, image_mmap, shifted_ref_matrix, mol_names
        )

        print(f"  Generating summary result ({'CSV' if save_csv else 'Parquet'}) with Top 5 contenders...")
        mol_names_ext = ["No Match"] + target_names
        df_records = {
            'x': np.tile(np.arange(w), h),
            'y': np.repeat(np.arange(h), w),
            'coords': [(int(y), int(x)) for y, x in zip(np.repeat(np.arange(h), w), np.tile(np.arange(w), h))],
            'Best_Match': [mol_names_ext[idx] for idx in class_map_mmap.ravel()],
            'Best_Similarity': best_sim_mmap.ravel()
        }

        for k in range(top_k):
            k_indices = all_top_indices[:, k]
            df_records[f'Contender_{k+1}'] = [target_names[idx] for idx in k_indices]
            df_records[f'Contender_Similarity_{k+1}'] = all_top_sims[:, k]

        summary_df = pd.DataFrame(df_records)

        if save_csv:
            csv_out = os.path.join(img_out_dir, f"{img_base}_sam_predictions.csv")
            summary_df.to_csv(csv_out, index=False)
            print(f"  Saved CSV: {csv_out}")
        else:
            pq_out = os.path.join(img_out_dir, f"{img_base}_sam_predictions.parquet")
            try:
                summary_df.to_parquet(pq_out, index=False)
                print(f"  Saved Parquet (Image too large for CSV): {pq_out}")
            except ImportError:
                print("  Warning: pyarrow or fastparquet not found. Falling back to CSV (might be very large!).")
                csv_out = os.path.join(img_out_dir, f"{img_base}_sam_predictions.csv")
                summary_df.to_csv(csv_out, index=False)

        del all_top_indices, all_top_sims, summary_df

        plt.figure(figsize=(12, 5))
        cmap_colors = plt.cm.tab20(np.linspace(0, 1, len(target_names)))
        cmap_colors = np.vstack([cmap_colors, [0.5, 0.5, 0.5, 1.0]])
        cmap = mcolors.ListedColormap(cmap_colors)

        plt.subplot(1, 2, 1)
        plt.imshow(class_map_mmap, cmap=cmap, vmin=0, vmax=len(target_names))
        plt.title("SAM Classification Map\n(Grey = No Match)")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        wn = np.linspace(CONFIG['wavenumber_start'], CONFIG['wavenumber_end'], CONFIG['num_samples'])
        plt.plot(wn, image_mean_norm, 'k-', label='Image Mean', linewidth=2, alpha=0.5)
        max_sim_per_mol = np.max(similarity_sample, axis=0)
        top_3_indices = np.argsort(max_sim_per_mol)[-3:][::-1]
        for idx in top_3_indices:
            plt.plot(wn, shifted_ref_matrix[idx], color=cmap_colors[idx % len(target_names)], linestyle='--',
                        label=f"{mol_names[idx]} (max sim: {max_sim_per_mol[idx]:.3f})")
        plt.title(f"Spectral Overlay (Shift: {best_shift})")
        plt.legend(fontsize=8, loc='upper left')
        plt.xlabel("Wavenumber (cm^-1)")
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(img_out_dir, f"{img_base}_sam_summary.png"), dpi=150)
        plt.close()

        del sim_stack_mmap, rgb_stack_mmap, class_map_mmap, best_sim_mmap, class_stack_mmap
        import gc
        gc.collect()

