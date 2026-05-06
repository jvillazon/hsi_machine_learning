import os
import re
import numpy as np
import pandas as pd
import tifffile
from glob import glob
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks
from scipy import ndimage as ndi
from skimage import measure
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# =========================================================================
# CONFIGURATION
# =========================================================================
CONFIG = {
    'spectra_csv': r'C:\Users\jvill\OneDrive\Documents\GitHub\hsi_machine_learning\unprocessed_data\averaged_spectra.csv',
    'image_dir': r"C:\Users\jvill\Downloads\jorge_kidney_data_annotated_yang\jorge_kidney_data_annotated_yang\normalized",
    'wavenumber_start': 2700,
    'wavenumber_end': 3100,
    'num_samples': 61,
    'max_shift_cm': 30,  # Max shift in cm^-1
    'similarity_threshold': 0.85, # Min similarity (0-1) to be considered a match
    'chunk_size': 1_000_000, # Number of pixels per batch for large images
    'max_csv_pixels': 10_000_000, # Skip full CSV if image is larger than this
    'exclude_molecules': [
        'Sphingosine',
        'Cer 18:1-12:0',
        'Cer 18:1-18:0',
        'Cer 18:1-18:1',
        'Cer 18:1-22:0',
        'Cer 18:1-24:0',
        'Cer 18:1-24:1',
        "Glucosylceramide (Gaucher's Spleen)",
        'Cer 18:0-14:0',
        'Cer 18:1-24:1(15Z)',
        'Cer m18:1-16:0',
        'Cer m18:1-24:1'
    ],
        # 'mol_groups': {
    #     'Sterols': [
    #         '18:1 Cholesterol Ester', 
    #         'Cholesterol (ovine)'
    #         ],
    #     'Cardiolipins': [
    #         '16:0 Cardiolipin'
    #         ],
    #     'Phosphatidylcholines': [
    #         'PC Mix', 
    #         'DOPC', 
    #         'DSPC'
    #         ],
    #     'Phosphatidylethanolamines': [
    #         'PE Mix', 
    #         'DOPE'
    #         ],
    #     'Phosphatidylserines': [
    #         'PS Mix'
    #         ],
    #     'Phosphatidylinositols': [
    #         'PI Mix'
    #         ],
    #     'Phosphatidylglycerols': [
    #         'PG Mix'
    #         ],
    #     'Lysophosphatidic Acids': [
    #         '18:0 Lyso PA', 
    #         '18:1 LPA'
    #         ],
    #     'CDP-Diaglycerols': [
    #         '16:0 CDP DG'
    #         ],
    #     'Diaglycerols (L/VL)': [
    #         'DAG 16:0', 
    #         'DAG 18:0 and 24:0 '
    #         ],
    #     'Triglycerols (M)': [
    #         'TAG Mix'
    #         ],
    #     'Triglycerols (L)': [
    #         'TAG 16:0', 
    #         'TAG 18:1'
    #         ],
    #     'Sphingomyelins': [
    #         'Sphingosine'
    #         ],
    #     'Ceramides (M)': [
    #         'Cer 18:1-12:0'
    #         ],
    #     'Ceramides (L)': [
    #         'Cer 18:1-18:0', 
    #         'Cer 18:1-18:1'
    #         ],
    #     'Ceramides (VL)': [
    #         'Cer 18:1-22:0', 
    #         'Cer 18:1-24:0', 
    #         'Cer 18:1-24:1'
    #         ],
    #     'Glucosylceramides': [
    #         "Glucosylceramide (Gaucher's Spleen)"
    #         ],
    #     'Dihydroceramides': [
    #         'Cer 18:0-14:0', 
    #         'Cer 18:1-24:1(15Z)'
    #         ],
    #     'Deoxyceramides' : ['Cer m18:1-16:0', 'Cer m18:1-24:1'],
    #     'Fatty Acids (L)': ['Stearic Acid', 'Palmitic Acid'],
    #     'Fatty Acids (PUFA)': ['Docosahexaenoic Acid', 'Tetracosapentaenoic Acid'],
    #     'Carnitines (S)': [
    #         'C8 L-Carnitine',
    #       ],
    #     'Carnitines (M/L)': [
    #         'C12 Carnitine', 
    #         'C16 Carnitine', 
    #         'C18 Carnitine (d9-cis)'
    #         ],
    # },
    'mol_groups': {
        'Background': ['Kidney Background'],
        'Sterols': [
            '18:1 Cholesterol Ester', 
            'Cholesterol (ovine)'
            ],
        'Phospholipids': [
            '16:0 Cardiolipin', 
            'PC Mix', 
            'PE Mix', 
            'PS Mix', 
            'PI Mix', 
            'PG Mix', 
            'DOPC', 
            'DOPE', 
            'DSPC', 
            '18:0 Lyso PA', 
            '18:1 LPA'
            ],
        'Glycerols': [
            '16:0 CDP DG',
            'DAG 16:0',
            'DAG 18:0 and 24:0 ',
            'TAG Mix',
            'TAG 16:0',
            'TAG 18:1'
        ],
        'Sphingolipids': [
            'Sphingosine',
            'Cer 18:1-12:0',
            'Cer 18:1-18:0',
            'Cer 18:1-18:1',
            'Cer 18:1-22:0',
            'Cer 18:1-24:0',
            'Cer 18:1-24:1',
            "Glucosylceramide (Gaucher's Spleen)",
            'Cer 18:0-14:0',
            'Cer 18:1-24:1(15Z)',
            'Cer m18:1-16:0',
            'Cer m18:1-24:1'
        ],
        'Fatty Acids': [
            'Stearic Acid',
            'Palmitic Acid',
            'Docosahexaenoic Acid',
            'Tetracosapentaenoic Acid'
        ],
        'Carnitines': [
            'C8 L-Carnitine',
            'C12 Carnitine',
            'C16 Carnitine',
            'C18 Carnitine (d9-cis)'
        ]
    },
}

MASKING_CONFIG = {
    'perform_masking': True,
    'mask_dir': r"C:\Users\jvill\Downloads\jorge_kidney_data_annotated_yang\jorge_kidney_data_annotated_yang\Mask",
    'min_instance_size': 100,
    'classes_to_ignore': ['No Match', 'Kidney Background'],
    'extract_similarity': True,
    'dilations': 0,

    'count_lipid_droplets': True,
    'droplet_classes': [
        'Sterols', 'Phospholipids', 'Glycerols', 'Sphingolipids', 
        'Fatty Acids'
    ], # If empty, counts all non-ignored molecule pixels as droplets
    'droplet_min_size': 5,
    'droplet_max_size': 150,
    'droplet_min_roundness': 0.8, # 1.0 is a perfect circle
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

def apply_sam_masking(img_base, img_out_dir, mask_dir, mol_names, mol_groups=None, pixel_size=(1.0, 1.0), norm_img_dir=None):
    """
    Apply masks to SAM classification and similarity results.
    Mask pattern: *-<Unit>-<Index>-mask.tif
    """

    img_mask_dir = os.path.join(mask_dir, img_base)
    if not os.path.exists(img_mask_dir):
        print(f"    Warning: Mask directory not found: {img_mask_dir}")
        return []
    
    mask_files = glob(os.path.join(img_mask_dir, "*-mask.tif"))
    if not mask_files:
        print(f"    Warning: No mask files matching *-mask.tif in {img_mask_dir}")
        return []

    # Load classification TIFF
    class_map_path = os.path.join(img_out_dir, f"{img_base}_sam_classification.tif")
    if not os.path.exists(class_map_path):
        print(f"    Error: Classification TIFF not found: {class_map_path}")
        return []
    class_map = tifffile.imread(class_map_path)
    h, w = class_map.shape

    # Optionally load similarity stack
    sim_stack = None
    if MASKING_CONFIG['extract_similarity']:
        sim_stack_path = os.path.join(img_out_dir, f"{img_base}_sam_similarity.tif")
        if os.path.exists(sim_stack_path):
            sim_stack = tifffile.imread(sim_stack_path)
        else:
            print(f"    Warning: Similarity stack not found for extraction: {sim_stack_path}")

    # Molecule to index mapping for fast lookup
    mol_to_idx = {name: i + 1 for i, name in enumerate(mol_names)}
    
    # Regex to parse Unit and Index
    # Pattern: *-<Unit>-<Index>-mask.tif
    mask_pattern = re.compile(r".*-(?P<unit>[^-]+)-(?P<index>[^-]+)-mask\.tif$")
    if MASKING_CONFIG.get('count_lipid_droplets') and norm_img_dir is not None:
        structure = np.ones((3, 3), dtype=int)
        total_droplet_mask = np.zeros((h, w), dtype=bool) if MASKING_CONFIG.get('count_lipid_droplets') else None
        l_memmap = tifffile.memmap(os.path.join(norm_img_dir, img_base, f"Lipid-{img_base}.tif"))
        p_memmap = tifffile.memmap(os.path.join(norm_img_dir, img_base, f"Protein-{img_base}.tif"))
        ld_mask = (l_memmap - p_memmap)  > 0.1  # Assuming lipid droplets are marked as >0.2 in this mask
        # Apply binary opening: erosion followed by dilation
        # ld_mask = ndi.binary_closing(ld_mask, structure=structure)
        ld_mask = ndi.binary_opening(ld_mask, structure=structure)
    
    records = []
    for mask_file in tqdm(mask_files, desc="    Quantifying masks", leave=False):
        # Initialize lipid droplet mask for entire image
        droplet_mask = np.zeros((h, w), dtype=bool) if MASKING_CONFIG.get('count_lipid_droplets') else None
        fname = os.path.basename(mask_file)
        match = mask_pattern.match(fname)
        if not match:
            print(f"    Warning: Mask {fname} does not match pattern")
            continue
            # # Fallback if pattern doesn't match perfectly
            # unit = "Unknown"
            # index = fname.replace("-mask.tif", "")
        else:
            unit = match.group("unit")
            index = match.group("index")
        
        mask = tifffile.imread(mask_file)
        if mask.shape != (h, w):
            # This might happen if image was downsampled or shape changed
            print(f"    Warning: Mask {fname} shape {mask.shape} mismatch with image {(h, w)}")
            # Try resizing if needed, but for now just skip or warn
            continue

        

        if MASKING_CONFIG['dilations'] > 0:
            mask = ndi.binary_dilation(mask, iterations=MASKING_CONFIG['dilations'])

        # Find connected components if multiple exist in one file, 
        # though pattern suggests one instance per file.
        labeled_mask, num_feat = ndi.label(mask > 0)
        slices = ndi.find_objects(labeled_mask)
        
        for i, slc in enumerate(slices):
            if slc is None: continue
            
            inst_mask = labeled_mask[slc] == (i + 1)
            n_pixels = np.sum(inst_mask)
            
            if n_pixels < MASKING_CONFIG['min_instance_size']:
                continue
            
            # Extract classification data
            class_patch = class_map[slc][inst_mask]
            
            # Centroid
            y_coords, x_coords = np.where(labeled_mask == (i + 1))
            cy, cx = np.mean(y_coords), np.mean(x_coords)
            
            # Counts
            labels, counts = np.unique(class_patch, return_counts=True)
            mol_names_ext = ["No Match"] + mol_names
            name_counts = {mol_names_ext[lbl]: cnt for lbl, cnt in zip(labels, counts)}
            
            # Filter ignored classes
            ignore_set = set(MASKING_CONFIG.get('classes_to_ignore', ['No Match']))
            total_valid = sum(cnt for name, cnt in name_counts.items() if name not in ignore_set)
            
            if total_valid == 0: continue
            
            # Initialize wide record for this instance
            record = {
                'Image_Name': img_base,
                'Unit': unit,
                'Image Index': int(index),
                'Instance_Label': index if num_feat == 1 else f"{index}_{i+1}",
                'centroid_x': cx,
                'centroid_y': cy,
                'n_pixels': n_pixels
            }
            
            # 1. Fill individual molecule percentages (ONLY if no groups are defined)
            if not mol_groups:
                for lbl, cnt in zip(labels, counts):
                    name = mol_names_ext[lbl]
                    if name in ignore_set:
                        continue
                    
                    percentage = (cnt / total_valid) * 100.0
                    record[name] = percentage
                    
                    # Extract mean similarity
                    if sim_stack is not None and lbl > 0:
                        sim_patch = sim_stack[lbl-1][slc][inst_mask & (class_map[slc] == lbl)]
                        record[f"{name}_Similarity"] = np.mean(sim_patch)
                
            # 2. Process groups
            if mol_groups:
                for group_name, members in mol_groups.items():
                    group_cnt = sum(name_counts.get(m, 0) for m in members)
                    if group_cnt > 0:
                        percentage = (group_cnt / total_valid) * 100.0
                        record[group_name] = percentage
                        
                        # Weighted mean similarity for group
                        if sim_stack is not None:
                            total_sim_sum = 0
                            total_pixel_count = 0
                            for m in members:
                                if m in mol_to_idx:
                                    m_idx = mol_to_idx[m]
                                    if m_idx in labels:
                                        pixel_mask = inst_mask & (class_map[slc] == m_idx)
                                        sim_pixels = sim_stack[m_idx-1][slc][pixel_mask]
                                        total_sim_sum += np.sum(sim_pixels)
                                        total_pixel_count += len(sim_pixels)
                            
                            if total_pixel_count > 0:
                                record[f"{group_name}_Similarity"] = total_sim_sum / total_pixel_count
            
            # 3. Lipid Droplet Counting
            if MASKING_CONFIG.get('count_lipid_droplets'):

                # droplet_classes = MASKING_CONFIG.get('droplet_classes', [])

                # if not droplet_classes:
                #     droplet_indices = [j+1 for j, m_name in enumerate(mol_names) if m_name not in ignore_set]
                # else:
                #     droplet_indices = [mol_to_idx[m] for m in droplet_classes if m in mol_to_idx]
            
                # Use 2D classification map for the instance bounding box
                patch_2d = class_map[slc]
                # droplet_pixel_mask = np.isin(patch_2d, droplet_indices) & ld_mask[slc]  # Only consider pixels marked as lipid droplets in the mask
                droplet_pixel_mask = ld_mask[slc]
                
                if droplet_pixel_mask.any():
                    labeled_droplets, n_droplets = ndi.label(droplet_pixel_mask)

                    droplet_slices = ndi.find_objects(labeled_droplets)
                    props = measure.regionprops(labeled_droplets)
                    
                    valid_droplet_areas = []
                    valid_droplet_majorities = []
                    all_droplets_data = [] # To store list of percentages for each droplet
                    mask_droplet_counts = {m: 0 for m in mol_names}
                    mask_droplet_px_total_area = 0

                    for d_idx, d_slc in enumerate(droplet_slices, 1):
                        if d_slc is None: continue
                        d_mask = labeled_droplets[d_slc] == d_idx
                        d_size = np.sum(d_mask)
                       
                        if d_size >= MASKING_CONFIG.get('droplet_min_size', 50) and d_size < MASKING_CONFIG.get('droplet_max_size', 1000):
                            prop = props[d_idx-1]
                            # Calculate roundness (circularity): 4 * pi * area / perimeter^2
                            # if prop.perimeter > 0:
                            #     roundness = (4 * np.pi * prop.area) / (prop.perimeter ** 2)
                            # else:
                            #     roundness = 0
                                
                            # if roundness < MASKING_CONFIG.get('droplet_min_roundness', 0.0):
                            #     continue

                            # Map back to global coordinates (Add location of instance bounding box and droplet bounding box)
                            global_d_slc = (
                                slice(d_slc[0].start + slc[0].start, d_slc[0].stop + slc[0].start),
                                slice(d_slc[1].start + slc[1].start, d_slc[1].stop + slc[1].start)
                            )
                            droplet_mask[global_d_slc][d_mask] = True
                            total_droplet_mask[global_d_slc][d_mask] = True

                            # Area
                            area_um2 = np.float16(d_size * pixel_size[0] * pixel_size[1])
                            valid_droplet_areas.append(area_um2)
                            mask_droplet_px_total_area += d_size
                            
                            # Composition
                            d_class_patch = patch_2d[d_slc][d_mask]
                            d_labels, d_counts = np.unique(d_class_patch, return_counts=True)
                            
                            # Calculate valid pixels for this droplet (excluding ignored classes)
                            d_valid_px = sum(cnt for lbl, cnt in zip(d_labels, d_counts) if mol_names_ext[lbl] not in ignore_set)
                            
                            if d_valid_px > 0:
                                d_pcts = {m: 0.0 for m in mol_names}
                                for lbl, cnt in zip(d_labels, d_counts):
                                    name = mol_names_ext[lbl]
                                    if name in ignore_set:
                                        continue
                                    mask_droplet_counts[name] += cnt # Keep for global weighted avg
                                    d_pcts[name] = (cnt / d_valid_px) * 100.0
                                all_droplets_data.append(d_pcts)
                                
                                d_max_local_idx = np.argmax(d_counts)
                                d_max_name = mol_names_ext[d_labels[d_max_local_idx]]
                                valid_droplet_majorities.append(d_max_name)
                        else:
                            continue

                    
                    if valid_droplet_areas:
                        record['Lipid Droplet Count'] = len(valid_droplet_areas)
                        # record['Lipid Droplet Areas um2'] = valid_droplet_areas
                        record['Lipid Droplet Majority Molecules'] = valid_droplet_majorities
                        # record['Lipid Droplet Molecular Percentages'] = valid_droplet_compositions  # Full composition
                        record['Lipid Droplet Avg Area um2'] = np.mean(valid_droplet_areas)
                        for name in mol_names:
                            # Pixel-weighted average (original)
                            record[f'Lipid Droplet {name} Weighted'] = (mask_droplet_counts[name] / mask_droplet_px_total_area * 100.0) if mask_droplet_px_total_area > 0 else 0.0
                            
                            # Per-droplet unweighted average and std
                            if all_droplets_data:
                                pct_values = [d[name] for d in all_droplets_data]
                                record[f'Lipid Droplet {name} Avg'] = np.mean(pct_values)
                                record[f'Lipid Droplet {name} Std'] = np.std(pct_values)
                            else:
                                record[f'Lipid Droplet {name} Avg'] = 0.0
                                record[f'Lipid Droplet {name} Std'] = 0.0

            records.append(record)

        # Save lipid droplet mask
        if droplet_mask is not None and droplet_mask.any():
            droplet_mask_path = os.path.join(img_out_dir, f"{os.path.splitext(fname)[0]}_sam_masked_droplets.tif")
            tifffile.imwrite(droplet_mask_path, droplet_mask.astype(np.uint8))
            print(f"    Saved droplet mask: {droplet_mask_path}")

        del droplet_mask

    # Save cumulative lipid droplet mask
    if total_droplet_mask is not None and total_droplet_mask.any():
        total_droplet_mask_path = os.path.join(img_out_dir, f"cumulative_sam_masked_droplets.tif")
        tifffile.imwrite(total_droplet_mask_path, total_droplet_mask.astype(np.uint8))
        print(f"    Saved cumulative droplet mask: {droplet_mask_path}")


    if records:
        df = pd.DataFrame(records)
        csv_path = os.path.join(img_out_dir, f"{img_base}_sam_masked_quantification.csv")
        df.to_csv(csv_path, index=False)
        print(f"    Saved masked quantification: {csv_path}")
        
    return records

# =========================================================================
# PIPELINE EXECUTION
# =========================================================================

def main():
    base_dir = os.path.dirname(CONFIG['image_dir'])

    if MASKING_CONFIG['count_lipid_droplets']:
        norm_img_dir = os.path.join(base_dir, 'Normalized_Images')


    output_dir = os.path.join(base_dir, 'sam_outputs-no-sm')
    os.makedirs(output_dir, exist_ok=True)
    
    all_mask_records = []

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
    
    for img_file in image_files:
        img_path = os.path.join(CONFIG['image_dir'], img_file)
        # Sanitize img_base to avoid hidden characters or illegal symbols
        img_base = os.path.splitext(img_file)[0]
        img_base = re.sub(r'[^\w\-_\. ]', '', img_base).strip()
        
        img_out_dir = os.path.normpath(os.path.join(output_dir, img_base))
        os.makedirs(img_out_dir, exist_ok=True)
        
        print(f"\nProcessing {img_file}...")
        
        # Load image as memmap
        image_mmap = tifffile.memmap(img_path, mode='r')
        with tifffile.TiffFile(img_path) as tif:
            if tif.is_imagej:
                page = tif.pages[0]
                pixel_size_x = 1/page.resolution[1]
                pixel_size_y = 1/page.resolution[0]
            else:
                pixel_size_x = 1
                pixel_size_y = 1
        res = (1/pixel_size_x, 1/pixel_size_y)
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
        
        # Apply shift to references
        shifted_ref_matrix = np.zeros_like(ref_matrix)
        for i in range(len(mol_names)):
            if best_shift > 0:
                shifted_ref_matrix[i, best_shift:] = ref_matrix[i, :-best_shift]
            elif best_shift < 0:
                shifted_ref_matrix[i, :best_shift] = ref_matrix[i, -best_shift:]
            else:
                shifted_ref_matrix[i] = ref_matrix[i]
        
        # Stage 3 & 4: Batched SAM and Classification
        print(f"  Calculating SAM and maps in chunks of {CONFIG['chunk_size']} pixels...")
        
        # Prepare output paths
        sim_stack_path = os.path.normpath(os.path.join(img_out_dir, f"{img_base}_sam_similarity.tif"))
        class_map_path = os.path.normpath(os.path.join(img_out_dir, f"{img_base}_sam_classification.tif"))
        class_stack_path = os.path.normpath(os.path.join(img_out_dir, f"{img_base}_sam_class_stack.tif"))
        
        # Temp file paths
        sim_stack_temp_path = os.path.normpath(os.path.join(img_out_dir, "temp_sim_stack.dat"))
        class_stack_temp_path = os.path.normpath(os.path.join(img_out_dir, "temp_class_stack.dat"))
        class_temp_path = os.path.normpath(os.path.join(img_out_dir, "temp_class.dat"))
        sim_temp_path = os.path.normpath(os.path.join(img_out_dir, "temp_sim.dat"))
        
        # Cleanup any existing temp files from previous failed runs to avoid locks
        for p in [sim_stack_temp_path, class_temp_path, sim_temp_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except:
                    pass
        
        # Create memmaps - based on target_names (Groups or Individuals)
        sim_stack_mmap = np.memmap(
            sim_stack_temp_path, shape=(len(target_names), h, w), dtype='float32', mode='w+'
        )
        class_stack_mmap = np.memmap(
            class_stack_temp_path, shape=(len(target_names), h, w), dtype='float32', mode='w+'
        )
        # Initialize class_stack_mmap with NaN
        class_stack_mmap[:] = np.nan
        
        class_map_mmap = np.memmap(
            class_temp_path, shape=(n_pixels,), dtype='uint16', mode='w+'
        )
        best_sim_mmap = np.memmap(
            sim_temp_path, shape=(n_pixels,), dtype='float32', mode='w+'
        )

        # Pre-allocate for CSV/Parquet contenders
        top_k = 5
        all_top_indices = np.zeros((n_pixels, top_k), dtype=np.uint16)
        all_top_sims = np.zeros((n_pixels, top_k), dtype=np.float32)
        save_csv = n_pixels < CONFIG['max_csv_pixels']

        for start_p in tqdm(range(0, n_pixels, CONFIG['chunk_size']), desc="Processing batches"):
            end_p = min(start_p + CONFIG['chunk_size'], n_pixels)
            chunk = image_flat[start_p:end_p].astype(np.float32)
            
            # Baseline subtraction
            silent_baseline = np.mean(chunk[:, :CH_START], axis=1, keepdims=True)
            chunk_proc = chunk - silent_baseline
            
            # SAM similarity
            chunk_angles = calculate_sam(chunk_proc, shifted_ref_matrix)
            
            if use_groups:
                # Group angle is the minimum angle among its members
                n_groups = len(target_names)
                effective_angles = np.full((len(chunk_angles), n_groups), np.pi, dtype=np.float32)
                for g_idx, m_indices in group_to_mols.items():
                    if m_indices:
                        effective_angles[:, g_idx] = np.min(chunk_angles[:, m_indices], axis=1)

            else:
                effective_angles = chunk_angles

            chunk_similarity = 1.0 - (effective_angles / np.pi)
        
            
            # Classification: 0 = No Match, 1..N = target indices
            chunk_best_idx = np.argmin(effective_angles, axis=1)
            chunk_best_sim = np.max(chunk_similarity, axis=1)

            top_k_indices = np.argsort(effective_angles, axis=1)[:, :top_k]
            all_top_indices[start_p:end_p] = top_k_indices
            all_top_sims[start_p:end_p] = 1.0 - (effective_angles[np.arange(len(effective_angles))[:, None], top_k_indices] / np.pi)

            # Apply threshold
            no_match_mask = chunk_best_sim < CONFIG['similarity_threshold']
            chunk_best_idx = chunk_best_idx + 1
            chunk_best_idx[no_match_mask] = 0

            class_map_mmap[start_p:end_p] = chunk_best_idx
            best_sim_mmap[start_p:end_p] = chunk_best_sim

            # Write to similarity stack (C, H, W) and class stack
            class_stack_flat = class_stack_mmap.reshape(len(target_names), -1)
            for t_idx in range(len(target_names)):
                sim_stack_mmap.reshape(len(target_names), -1)[t_idx, start_p:end_p] = chunk_similarity[:, t_idx]
                # Mark pixels where this target has the best match (before threshold) with 1.0
                best_match_mask = (chunk_best_idx - 1 == t_idx)
                class_stack_flat[t_idx, start_p:end_p][best_match_mask] = 1.0

        # Flush and save maps
        sim_stack_mmap.flush()
        class_stack_mmap.flush()
        class_map_mmap.flush()
        best_sim_mmap.flush()
        
        # Save similarity and classification TIFFs
        tifffile.imwrite(sim_stack_path, sim_stack_mmap, resolution=res, imagej=True, 
                         metadata={'axes': 'ZYX', 'Labels': target_names})
        tifffile.imwrite(class_map_path, class_map_mmap.reshape(h, w), resolution=res, imagej=True, metadata=meta)
        tifffile.imwrite(class_stack_path, class_stack_mmap, resolution=res, imagej=True, 
                         metadata={'axes': 'ZYX', 'Labels': target_names})
        
        # Stage 4.3: Diagnostic Plots
        # Sample pixels for diagnostics (we don't keep all similarity in RAM)
        indices, best_match_idx_sample, similarity_sample = plot_diagnostic_samples(
            img_out_dir, image_mmap, shifted_ref_matrix, mol_names
        )
        
        # Stage 4.4: Result Output (CSV or Parquet)
        print(f"  Generating summary result ({'CSV' if save_csv else 'Parquet'}) with Top 5 contenders...")
        # Index 0 = No Match; indices 1..N map to target_names[0..N-1]
        mol_names_ext = ["No Match"] + target_names
        df_records = {
            'x': np.tile(np.arange(w), h),
            'y': np.repeat(np.arange(h), w),
            'coords': [(int(y),int(x)) for y,x in zip(np.repeat(np.arange(h), w), np.tile(np.arange(w), h))],
            'Best_Match': [mol_names_ext[idx] for idx in class_map_mmap],
            'Best_Similarity': best_sim_mmap
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

        # Stage 4.5: Summary Plot (using sampled data)
        plt.figure(figsize=(12, 5))
        cmap_colors = plt.cm.tab20(np.linspace(0, 1, len(target_names)))
        cmap_colors = np.vstack([cmap_colors, [0.5, 0.5, 0.5, 1.0]])
        cmap = mcolors.ListedColormap(cmap_colors)
        
        plt.subplot(1, 2, 1)
        plt.imshow(class_map_mmap.reshape(h, w), cmap=cmap, vmin=0, vmax=len(target_names))
        plt.title("SAM Classification Map\n(Grey = No Match)")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        wn = np.linspace(CONFIG['wavenumber_start'], CONFIG['wavenumber_end'], CONFIG['num_samples'])
        plt.plot(wn, image_mean_norm, 'k-', label='Image Mean', linewidth=2, alpha=0.5)
        # Note: similarity_sample based on individual mols, plot for visualization only
        max_sim_per_mol = np.max(similarity_sample, axis=0)
        top_3_indices = np.argsort(max_sim_per_mol)[-3:][::-1]
        for idx in top_3_indices:
            plt.plot(wn, shifted_ref_matrix[idx], color=cmap_colors[idx % len(target_names)], linestyle='--', 
                     label=f"{mol_names[idx]} (max sim: {max_sim_per_mol[idx]:.3f})")
        plt.title(f"Spectral Overlay (Shift: {best_shift})")
        plt.legend(fontsize=8, loc='upper left'); plt.xlabel("Wavenumber (cm^-1)")
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(img_out_dir, f"{img_base}_sam_summary.png"), dpi=150); plt.close()
        
        # Cleanup temp files
        del class_map_mmap, best_sim_mmap
        import gc
        gc.collect() # Force cleanup on Windows to release file handles
        
        try:
            os.remove(os.path.join(img_out_dir, "temp_class.dat"))
            os.remove(os.path.join(img_out_dir, "temp_sim.dat"))
            os.remove(os.path.join(img_out_dir, "temp_sim_stack.dat"))
            os.remove(os.path.join(img_out_dir, "temp_class_stack.dat"))
        except Exception as e:
            print(f"  Note: Could not remove temp files: {e}")

        # Stage 5: Masking (Per Image)
        if MASKING_CONFIG['perform_masking']:
            records = apply_sam_masking(
                img_base, img_out_dir, MASKING_CONFIG['mask_dir'], 
                target_names,
                pixel_size=(pixel_size_x, pixel_size_y),
                norm_img_dir=norm_img_dir if MASKING_CONFIG.get('count_lipid_droplets') else None
            )
            all_mask_records.extend(records)

    # Final Aggregated Masking Summary
    if MASKING_CONFIG['perform_masking'] and all_mask_records:
        summary_df = pd.DataFrame(all_mask_records)
        summary_path = os.path.join(output_dir, "sam_masked_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nFinal Aggregated Masking Summary saved to: {summary_path}")

    print("\nSAM Pipeline Complete.")

if __name__ == "__main__":
    main()
