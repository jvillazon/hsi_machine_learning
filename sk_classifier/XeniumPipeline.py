import os
import re
from functools import reduce
import sys
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
import seaborn as sns

###################################################################
# =========================================================================
# CONFIGURATION
# =========================================================================
CONFIG = {
    'spectra_csv': r'C:\Users\jvill\OneDrive\Documents\GitHub\hsi_machine_learning\unprocessed_data\averaged_spectra.csv',
    'image_dir': r"D:\ADATA Backup\HuBMAP\HuBMAP Xenium\Xenium WSI\data",
    'wavenumber_start': 2700,
    'wavenumber_end': 3100,
    'num_samples': 61,
    'exclude_molecules': [],
    # 'mol_groups': None,
    'mol_groups': {
        'Sterols': [
            '18:1 Cholesterol Ester', 
            'Cholesterol (ovine)'
            ],
        'Cardiolipins': [
            '16:0 Cardiolipin'
            ],
        'Phosphatidylcholines': [
            'PC Mix', 
            'DOPC', 
            'DSPC'
            ],
        'Phosphatidylethanolamines': [
            'PE Mix', 
            'DOPE'
            ],
        'Phosphatidylserines': [
            'PS Mix'
            ],
        'Phosphatidylinositols': [
            'PI Mix'
            ],
        'Phosphatidylglycerols': [
            'PG Mix'
            ],
        'Lysophosphatidate': [
            '18:0 Lyso PA', 
            '18:1 LPA'
            ],
        'Glycerols': [
            'DAG 16:0', 
            'DAG 18:0 and 24:0 '
            'TAG Mix'
            'TAG 16:0', 
            'TAG 18:1',
            '16:0 CDP DG'
            ],
        'Ceramides': [
            'Sphingosine'
            'Cer 18:1-12:0'
            'Cer 18:1-18:0', 
            'Cer 18:1-18:1'
            'Cer 18:1-22:0', 
            'Cer 18:1-24:0', 
            'Cer 18:1-24:1'
            ],
        'Glucosylceramides': [
            "Glucosylceramide (Gaucher's Spleen)"
            ],
        'Dihydroceramides': [
            'Cer 18:0-14:0', 
            'Cer 18:1-24:1(15Z)'
            ],
        'Deoxyceramides' : ['Cer m18:1-16:0', 'Cer m18:1-24:1'],
        'SFA': ['Stearic Acid', 'Palmitic Acid'],
        'PUFA': ['Docosahexaenoic Acid', 'Tetracosapentaenoic Acid'],
        'Carnitines': [
            # 'C8 L-Carnitine',
            'C12 Carnitine', 
            'C16 Carnitine', 
            'C18 Carnitine (d9-cis)'
            ],
        'Background' : ['Kidney Background']
       },

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

MASKING_CONFIG = {
    'perform_masking': True,
    'use_parquet_cells': True,
    'parquet_dir': r"D:\ADATA Backup\HuBMAP\HuBMAP Xenium\Xenium_data",
    'export_individual_csvs': True,
    'mask_dir': r"D:\ADATA Backup\HuBMAP\HuBMAP Xenium\Xenium HSI\Xenium_masks",
    'prefix': 'Xenium',
    'min_instance_size': 20,
    'classes_to_ignore': ['No Match', 'Kidney Background', 'Tissue Background'],
    'extract_similarity': True,
    'dilations': 2,
    'mask_dict': {
        '0': 'low_quality',
        '1': 'POD',
        '2': 'PT',
        '3': 'CNT',
        '4': 'DCT',
        '5': 'TAL',
        '6': 'DTL',
        '7': 'ATL',
        '8': 'IC',
        '9': 'altPOD',
        '10': 'altPT',
        '11': 'altCNT',
        '12': 'altDCT',
        '13': 'altTAL',
        '14': 'altDTL',
        '15': 'altATL',
        '16': 'altIC',
        '17': 'PC',
        '18': 'PEC',
        '19': 'MD',
        '20': 'EC',
        '21': 'EC-GC',
        '22': 'FIB',
        '23': 'infFIB',
        '24': 'pvFIB',
        '25': 'aFIB',
        '26': 'B',
        '27': 'DC',
        '28': 'MON',
        '29': 'N',
        '30': 'NK',
        '31': 'PL',
        '32': 'T',
        '33': 'MAC',
        '34': 'resMAC',
        '35': 'moMAC',
        '36': 'moMAC-INF',
        '37': 'MAST',
        '38': 'Lymphoid',
        '39': 'Myeloid',
        '40': 'NEU',
        '41': 'SC/NEU',
        '42': 'MC',
        '43': 'MYOF',
        '44': 'REN',
        '45': 'VSM/P',
        '46': 'unassigned'
    },
    'subgroups': None,

    # ── Update / resume mode ────────────────────────────────────────────────
    # When update_mode=True and a unified CSV already exists for an image,
    # only the modes listed in update_modes are re-extracted.  Their columns
    # are dropped from the existing CSV and replaced with the fresh values.
    # All other columns (e.g. SAM, RF, HSI) are left untouched.
    # Valid mode names: 'ratios', 'sam_classmap', 'predictions', 'hsi'
    'update_mode':  True,
    'update_modes': ['collagen'],
}

# Derived configuration
WN_STEP = (CONFIG['wavenumber_end'] - CONFIG['wavenumber_start']) / (CONFIG['num_samples'] - 1)
CH_START = int((2800 - CONFIG['wavenumber_start']) / WN_STEP)

# =================================================================
# FUNCTIONS
# =================================================================


def _get_mode_columns(df: pd.DataFrame, mode: str, meta_cols: list) -> list:
    """
    Return the column names in *df* that were produced by a given data mode.
    Used by update-mode logic to identify which columns to drop before
    merging in freshly-extracted values.

    Parameters
    ----------
    df        : DataFrame to inspect
    mode      : one of 'ratios', 'sam_classmap', 'predictions', 'hsi'
    meta_cols : list of metadata column names (always excluded)
    """
    if mode == 'ratios':
        # Ratio columns: everything that is not a metadata col, not RF_/SAM_/
        # collagen_/CH<n>
        return [
            c for c in df.columns
            if c not in meta_cols
            and not c.startswith(('RF_', 'SAM_', 'Collagen_'))
            and not re.match(r'^CH\d+$', c)
        ]
    elif mode == 'collagen':
        return [c for c in df.columns if c.startswith('Collagen_')]
    elif mode == 'sam_classmap':
        return [c for c in df.columns if c.startswith('SAM_')]
    elif mode == 'predictions':
        return [c for c in df.columns if c.startswith('RF_')]
    elif mode == 'hsi':
        return [c for c in df.columns if re.match(r'^CH\d+$', c)]
    return []


def _parse_mask_basename(basename, prefix, subgroups):
    """Return cleaned unit_name and optional subgroup from a mask basename."""
    unit_name = basename
    if prefix and unit_name.startswith(f"{prefix}-"):
        unit_name = unit_name[len(prefix) + 1:]

    for suffix in ['-mask.tif', '-Mask.tif']:
        if unit_name.endswith(suffix):
            unit_name = unit_name[:-len(suffix)]

    subgroup = None
    if "_" in unit_name and subgroups is not None:
        unit_parts = unit_name.split("_")
        if len(unit_parts) == 2:
            match0 = next((s for s in subgroups if s.lower() == unit_parts[0].lower()), None)
            match1 = next((s for s in subgroups if s.lower() == unit_parts[1].lower()), None)
            if match0:
                subgroup = match0
                unit_name = unit_parts[1]
            elif match1:
                subgroup = match1
                unit_name = unit_parts[0]

        if subgroup is None:
            for i, part in enumerate(unit_parts):
                match = next((s for s in subgroups if s.lower() == part.lower()), None)
                if match:
                    subgroup = match
                    remaining = [p for j, p in enumerate(unit_parts) if i != j]
                    unit_name = "_".join(remaining)
                    break

    return unit_name, subgroup


def apply_unified_masking(*, data_configs, mask_list_path, target_names=None, mol_groups=None, prefix=None, mask_dict=None,
                          subgroups=None, img_name=None, classes_to_ignore=None, min_instance_size=100, output_flat_csv=None):
    """
    Unified masking quantification for all data types in data_configs in a single pass.
    Avoids repeatedly loading and segmenting masks by generating instances once.
    
    Returns a wide pandas DataFrame with one row per mask instance and columns for all
    quantified values.
    """
    mask_files = glob(os.path.join(mask_list_path, '*-mask.tif'))
    if not mask_files:
        raise ValueError(f'No mask TIFF files found in {mask_list_path}')

    if mask_list_path is None:
        raise ValueError('mask_list_path is required')

    # Defaults
    if classes_to_ignore is None:
        classes_to_ignore = ['Masked', 'Kidney Background', 'No Match']
    classes_to_ignore_set = {c.strip() for c in classes_to_ignore}

    local_img_name = img_name
    
    # Pre-load/open data sources to avoid redundant file IO
    loaded_sources = []
    try:
        for config in data_configs:
            mode = config['mode']
            files = config['files']
            dilations = config.get('dilations', 2)
            col_prefix = config.get('column_prefix', '')
            
            for f in files:
                if not os.path.exists(f):
                    print(f"Warning: File not found {f}, skipping from masking analysis.")
                    continue
                    
                if mode == 'ratios':
                    # Use TiffFile for robust, lazy page access
                    tif_handle = tifffile.TiffFile(f)
                    num_pages = len(tif_handle.pages)
                    ratio_basename = os.path.splitext(os.path.basename(f))[0]
                    if '-ratio' in ratio_basename:
                        ratio_basename = ratio_basename.split('-')[-2]
                    
                    if local_img_name is None:
                        local_img_name = os.path.splitext(os.path.basename(f))[0]
                        if '-ratio' in local_img_name:
                            local_img_name = local_img_name.split('-')[0]
                            
                    loaded_sources.append({
                        'type': 'ratios',
                        'file_path': f,
                        'handle': tif_handle,
                        'num_pages': num_pages,
                        'ratio_basename': ratio_basename,
                        'dilations': dilations,
                        'column_prefix': col_prefix
                    })

                elif mode == 'collagen':
                    # Same extraction as ratios but with automatic 'collagen_' prefix.
                    # Expects files named: {img_base}-{MetricName}-collagen.tif
                    tif_handle  = tifffile.TiffFile(f)
                    num_pages   = len(tif_handle.pages)
                    base_no_ext = os.path.splitext(os.path.basename(f))[0]
                    if base_no_ext.endswith('-collagen'):
                        base_no_ext = base_no_ext[:-len('-collagen')]
                    ratio_basename = base_no_ext.split('-')[-1]  # e.g. 'Coherence'

                    if local_img_name is None:
                        local_img_name = os.path.splitext(os.path.basename(f))[0].split('-')[0]

                    loaded_sources.append({
                        'type': 'ratios',          # reuse ratio extraction logic
                        'file_path': f,
                        'handle': tif_handle,
                        'num_pages': num_pages,
                        'ratio_basename': ratio_basename,
                        'dilations': dilations,
                        'column_prefix': 'Collagen_'
                    })

                elif mode == 'sam_classmap':
                    sam_class_map = tifffile.imread(f)
                    if sam_class_map.ndim > 2:
                        sam_class_map = np.squeeze(sam_class_map)

                    if local_img_name is None:
                        local_img_name = os.path.splitext(os.path.basename(f))[0]

                    class_names = []
                    if target_names is not None:
                        class_names = [target.strip() for target in target_names if target.strip() not in classes_to_ignore_set]

                    loaded_sources.append({
                        'type': 'sam_classmap',
                        'file_path': f,
                        'matrix': sam_class_map,
                        'class_names': class_names,
                        'dilations': dilations,
                        'column_prefix': col_prefix
                    })

                elif mode == 'predictions':
                    # Random Forest predictions from CSV
                    data_df = pd.read_csv(f, header=None, index_col=False)
                    rf_class_map = data_df.values
                    if rf_class_map.ndim > 2:
                        rf_class_map = np.squeeze(rf_class_map)

                    if local_img_name is None:
                        local_img_name = os.path.basename(f).replace('_rf_predictions.csv', '')

                    loaded_sources.append({
                        'type': 'predictions',
                        'file_path': f,
                        'matrix': rf_class_map,
                        'dilations': dilations,
                        'column_prefix': col_prefix
                    })

                elif mode == 'hsi':

                    hsi_data = tifffile.imread(f)
                    if hsi_data.ndim > 2:
                        if hsi_data.shape[0] > hsi_data.shape[2]:
                            hsi_data = np.transpose(hsi_data, (2, 0, 1))
                    else:
                        hsi_data = np.expand_dims(hsi_data, axis=0)
                        
                    if local_img_name is None:
                        local_img_name = os.path.splitext(os.path.basename(f))[0]
                        
                    loaded_sources.append({
                        'type': 'hsi',
                        'file_path': f,
                        'matrix': hsi_data,
                        'dilations': dilations,
                        'column_prefix': col_prefix
                    })

        # Process each mask file
        records = []
        for mask_file in tqdm(mask_files, desc='Processing masks...'):
            mask_basename = os.path.basename(mask_file)
            mask_img = tifffile.imread(mask_file)
            if mask_img.ndim > 2:
                mask_img = np.squeeze(mask_img)

            # Determine mask slices & units
            slices_to_process = []
            if mask_img.ndim == 3 and mask_img.shape[0] > 1:
                if mask_dict is None:
                    raise Exception(f"Mask image {mask_basename} has multiple channels but no mask_dict provided for indexing.")
                for idx, unit in mask_dict.items():
                    mask_slice = mask_img[int(idx)]
                    unit_name = unit
                    subgroup = None
                    if subgroups is not None:
                        for sg in subgroups:
                            if sg.lower() in unit_name.lower():
                                unit_name = unit_name.replace(sg, '')[:-1]
                                subgroup = sg
                                break
                    slices_to_process.append((mask_slice, unit_name, subgroup))
            else:
                unit_name, subgroup = _parse_mask_basename(mask_basename, prefix, subgroups)
                if subgroups is not None and subgroup is None:
                    continue
                slices_to_process.append((mask_img, unit_name, subgroup))

            # Segment and process each slice
            for mask_slice, unit_name, subgroup in slices_to_process:
                labeled_mask, num_features = ndi.label(mask_slice > 0)
                if num_features == 0:
                    continue

                object_slices = ndi.find_objects(labeled_mask)
                for label, obj_slice in enumerate(object_slices, start=1):
                    if obj_slice is None:
                        continue

                    # Calculate original centroid coords
                    rows, cols = np.nonzero(labeled_mask[obj_slice] == label)
                    rows += obj_slice[0].start
                    cols += obj_slice[1].start
                    centroid_y = int(np.mean(rows))
                    centroid_x = int(np.mean(cols))

                    # Base metadata record
                    inst_record = {
                        'Image_Name': local_img_name,
                        'Unit': unit_name,
                        'Instance_Label': f"{int(label)}",
                        'Source_ID': local_img_name.split('-')[0] if local_img_name else None,
                        'Group_ID': subgroup,
                        'centroid_x': centroid_x,
                        'centroid_y': centroid_y,
                    }
                    
                    has_any_data = False

                    # Extract data from each loaded source
                    for source in loaded_sources:
                        dilations = source['dilations']
                        y_slice, x_slice = obj_slice
                        y_start = max(0, y_slice.start - dilations)
                        y_end = min(labeled_mask.shape[0], y_slice.stop + dilations)
                        x_start = max(0, x_slice.start - dilations)
                        x_end = min(labeled_mask.shape[1], x_slice.stop + dilations)
                        padded_slice = (slice(y_start, y_end), slice(x_start, x_end))

                        local_labels = labeled_mask[padded_slice]
                        instance_mask = local_labels == label

                        npx = int(np.sum(instance_mask))
                        if npx < min_instance_size:
                            continue

                        if dilations > 0:
                            instance_mask = ndi.binary_dilation(instance_mask, iterations=dilations-1)

                        # Set pixels count if not already set (at first valid source)
                        if 'n_pixels' not in inst_record:
                            inst_record['n_pixels'] = npx

                        if source['type'] == 'ratios':
                            for slice_idx in range(source['num_pages']):
                                data_slice = source['handle'].pages[slice_idx].asarray()
                                ratio_basename = source['ratio_basename']
                                r_type_slice = f"{ratio_basename}_{slice_idx + 1}" if source['num_pages'] > 1 else ratio_basename
                                col_name = f"{source['column_prefix']}{r_type_slice}"
                                
                                masked_vals = data_slice[padded_slice][instance_mask]
                                vals = masked_vals[(masked_vals > 0) & np.isfinite(masked_vals)]
                                if vals.size > 0:
                                    mean_ratio = float(np.nanmean(vals))
                                    inst_record[col_name] = mean_ratio
                                    has_any_data = True
                                    
                        elif source['type'] == 'sam_classmap':
                            patch = source['matrix'][padded_slice][instance_mask]
                            labels, counts = np.unique(patch, return_counts=True)
                            name_counts = {}
                            for lbl, cnt in zip(labels, counts):
                                lbl_int = int(lbl)
                                if lbl_int <= 0:
                                    continue
                                class_idx = lbl_int - 1
                                if class_idx < len(source['class_names']):
                                    name_counts[source['class_names'][class_idx]] = int(cnt)
                            total = sum(name_counts.values())
                            if total > 0:
                                for mol, cnt in name_counts.items():
                                    pct = (cnt / total) * 100.0
                                    col_name = f"SAM_{mol}"
                                    inst_record[col_name] = float(pct)
                                    has_any_data = True
                                    
                        elif source['type'] == 'predictions':
                            preds = source['matrix'][padded_slice][instance_mask]
                            unique, counts = np.unique(preds, return_counts=True)
                            class_counts = {}
                            for k, v in zip(unique, counts):
                                k_str = str(k).strip()
                                if mol_groups is not None and k_str in mol_groups:
                                    mapped_name = mol_groups[k_str]
                                else:
                                    mapped_name = k_str
                                class_counts[mapped_name] = class_counts.get(mapped_name, 0) + int(v)
                                
                            for c in classes_to_ignore_set:
                                if c in class_counts:
                                    del class_counts[c]
                                    
                            total = sum(class_counts.values())
                            if total > 0:
                                for mol, cnt in class_counts.items():
                                    pct = (cnt / total) * 100.0
                                    col_name = f"RF_{mol}"
                                    inst_record[col_name] = float(pct)
                                    has_any_data = True
                                    
                        elif source['type'] == 'hsi':
                            hsi_data = source['matrix']
                            num_channels = hsi_data.shape[0]
                            for ch_idx in range(num_channels):
                                col_name = f"CH{ch_idx + 1}"
                                masked_vals = hsi_data[ch_idx][padded_slice][instance_mask]
                                vals = masked_vals[np.isfinite(masked_vals)]
                                if vals.size > 0:
                                    inst_record[col_name] = float(np.nanmean(vals))
                                    has_any_data = True

                    if has_any_data:
                        records.append(inst_record)

    finally:
        # Guarantee all opened TiffFiles are properly closed
        for source in loaded_sources:
            if source['type'] == 'ratios' and 'handle' in source:
                try:
                    source['handle'].close()
                except Exception as e:
                    print(f"Failed to close TIFF handle: {e}")

    if not records:
        raise ValueError('No valid instances found across all masks.')

    df_out = pd.DataFrame.from_records(records)
    
    # Fill percentage columns with 0.0 where NaN
    pct_cols = [c for c in df_out.columns if c.startswith(('RF_', 'SAM_'))]
    if pct_cols:
        df_out[pct_cols] = df_out[pct_cols].fillna(0.0)

    # Optionally write flat output
    if output_flat_csv:
        df_out.to_csv(output_flat_csv, index=False)

    return df_out


def apply_unified_parquet_masking(*, data_configs, parquet_path, target_names=None, mol_groups=None, roi_bounds=None,
                                  min_instance_size=20, classes_to_ignore=None, img_name=None, output_flat_csv=None):
    """
    Direct Parquet-based single-cell masking quantification.
    Uses polygon coordinates directly from the parquet table to extract signals on-the-fly.
    Preserves single-cell identities and resolves all cell overlaps.
    """
    import cv2
    
    df_cells = pd.read_parquet(parquet_path)
    if 'cell_ID' in df_cells.columns and 'cell_id' not in df_cells.columns:
        df_cells = df_cells.rename(columns={'cell_ID': 'cell_id'})
        
    if classes_to_ignore is None:
        classes_to_ignore = ['Masked', 'Kidney Background', 'No Match']
    classes_to_ignore_set = {c.strip() for c in classes_to_ignore}

    local_img_name = img_name
    
    # Pre-load/open data sources to avoid redundant file IO
    loaded_sources = []
    try:
        for config in data_configs:
            mode = config['mode']
            files = config['files']
            dilations = config.get('dilations', 2)
            col_prefix = config.get('column_prefix', '')
            
            for f in files:
                if not os.path.exists(f):
                    print(f"Warning: File not found {f}, skipping from masking analysis.")
                    continue
                    
                if mode == 'ratios':
                    # Use TiffFile for robust, lazy page access
                    tif_handle = tifffile.TiffFile(f)
                    num_pages = len(tif_handle.pages)
                    ratio_basename = os.path.splitext(os.path.basename(f))[0]
                    if '-ratio' in ratio_basename:
                        ratio_basename = ratio_basename.split('-')[-2]
                    
                    if local_img_name is None:
                        local_img_name = os.path.splitext(os.path.basename(f))[0]
                        if '-ratio' in local_img_name:
                            local_img_name = local_img_name.split('-')[0]
                            
                    loaded_sources.append({
                        'type': 'ratios',
                        'file_path': f,
                        'handle': tif_handle,
                        'num_pages': num_pages,
                        'ratio_basename': ratio_basename,
                        'dilations': dilations,
                        'column_prefix': col_prefix
                    })

                elif mode == 'collagen':
                    # Same extraction as ratios but with automatic 'collagen_' prefix.
                    # Expects files named: {img_base}-{MetricName}-collagen.tif
                    tif_handle  = tifffile.TiffFile(f)
                    num_pages   = len(tif_handle.pages)
                    base_no_ext = os.path.splitext(os.path.basename(f))[0]
                    if base_no_ext.endswith('-collagen'):
                        base_no_ext = base_no_ext[:-len('-collagen')]
                    ratio_basename = base_no_ext.split('-')[-1]  # e.g. 'Coherence'

                    if local_img_name is None:
                        local_img_name = os.path.splitext(os.path.basename(f))[0].split('-')[0]

                    loaded_sources.append({
                        'type': 'ratios',          # reuse ratio extraction logic
                        'file_path': f,
                        'handle': tif_handle,
                        'num_pages': num_pages,
                        'ratio_basename': ratio_basename,
                        'dilations': dilations,
                        'column_prefix': 'Collagen_'
                    })

                elif mode == 'sam_classmap':
                    sam_class_map = tifffile.imread(f)
                    if sam_class_map.ndim > 2:
                        sam_class_map = np.squeeze(sam_class_map)

                    if local_img_name is None:
                        local_img_name = os.path.splitext(os.path.basename(f))[0]

                    class_names = []
                    if target_names is not None:
                        class_names = [target.strip() for target in target_names if target.strip() not in classes_to_ignore_set]

                    loaded_sources.append({
                        'type': 'sam_classmap',
                        'file_path': f,
                        'matrix': sam_class_map,
                        'class_names': class_names,
                        'dilations': dilations,
                        'column_prefix': col_prefix
                    })

                elif mode == 'predictions':

                    # Random Forest predictions from CSV
                    data_df = pd.read_csv(f, header=None, index_col=False)
                    rf_class_map = data_df.values
                    if rf_class_map.ndim > 2:
                        rf_class_map = np.squeeze(rf_class_map)
                        
                    if local_img_name is None:
                        local_img_name = os.path.basename(f).replace('_rf_predictions.csv', '')
                        
                    loaded_sources.append({
                        'type': 'predictions',
                        'file_path': f,
                        'matrix': rf_class_map,
                        'dilations': dilations,
                        'column_prefix': col_prefix
                    })
                    
                elif mode == 'hsi':
                    hsi_data = tifffile.imread(f)
                    if hsi_data.ndim > 2:
                        if hsi_data.shape[0] > hsi_data.shape[2]:
                            hsi_data = np.transpose(hsi_data, (2, 0, 1))
                    else:
                        hsi_data = np.expand_dims(hsi_data, axis=0)
                        
                    if local_img_name is None:
                        local_img_name = os.path.splitext(os.path.basename(f))[0]
                        
                    loaded_sources.append({
                        'type': 'hsi',
                        'file_path': f,
                        'matrix': hsi_data,
                        'dilations': dilations,
                        'column_prefix': col_prefix
                    })

        # Determine spatial dimensions of the target images
        h_bound, w_bound = None, None
        for src in loaded_sources:
            if src['type'] == 'sam_classmap':
                h_bound, w_bound = src['matrix'].shape
                break
            elif src['type'] == 'predictions':
                h_bound, w_bound = src['matrix'].shape
                break
            elif src['type'] == 'hsi':
                h_bound, w_bound = src['matrix'].shape[1:]
                break
            elif src['type'] == 'ratios':
                first_page = src['handle'].pages[0].asarray()
                h_bound, w_bound = first_page.shape
                break

        if h_bound is None or w_bound is None:
            if roi_bounds is not None:
                h_bound = roi_bounds.get('height', 10000)
                w_bound = roi_bounds.get('width', 10000)
            else:
                h_bound, w_bound = 10000, 10000

        # Perform localized masking cell-by-cell
        records = []
        global_mask = np.zeros((h_bound, w_bound), dtype=np.uint8)
        
        for cell_id, group in tqdm(df_cells.groupby('cell_id'), desc="Quantifying parquet cells"):
            # Ensure coordinates are in int32 format for cv2.fillPoly
            vertices = group[['x_roi', 'y_roi']].values.astype(np.int32)
            unit_name = group['levelKID_refined'].iloc[0] if 'levelKID_refined' in group.columns else 'unassigned'
            subgroup = group['Group_ID'].iloc[0] if 'Group_ID' in group.columns else None
            
            # Calculate local bounding box
            min_x, min_y = vertices.min(axis=0)
            max_x, max_y = vertices.max(axis=0)
            
            # Bound inside the image grid
            min_x = max(0, min_x)
            min_y = max(0, min_y)
            max_x = min(w_bound - 1, max_x)
            max_y = min(h_bound - 1, max_y)
            
            if min_x >= max_x or min_y >= max_y:
                continue
                
            centroid_x = int(np.mean(vertices[:, 0]))
            centroid_y = int(np.mean(vertices[:, 1]))
            
            inst_record = {
                'Image_Name': local_img_name,
                'Unit': unit_name,
                'Instance_Label': str(cell_id),
                'Source_ID': local_img_name.split('-')[0] if local_img_name else None,
                'Group_ID': subgroup,
                'centroid_x': centroid_x,
                'centroid_y': centroid_y,
            }
            
            has_any_data = False
            
            # Extract data from each loaded source
            for source in loaded_sources:
                dilations = source['dilations']
                
                # Expand bounding box locally for dilation
                y_start = max(0, min_y - dilations)
                y_end = min(h_bound, max_y + dilations + 1)
                x_start = max(0, min_x - dilations)
                x_end = min(w_bound, max_x + dilations + 1)
                
                patch_h = y_end - y_start
                patch_w = x_end - x_start
                if patch_h <= 0 or patch_w <= 0:
                    continue
                    
                # Draw local polygon mask
                local_mask = np.zeros((patch_h, patch_w), dtype=np.uint8)
                local_vertices = vertices.copy()
                local_vertices[:, 0] -= x_start
                local_vertices[:, 1] -= y_start
                
                cv2.fillPoly(local_mask, [local_vertices], 1)
                
                npx = int(np.sum(local_mask))
                if npx < min_instance_size:
                    continue
                    
                if dilations > 0:
                    instance_mask = ndi.binary_dilation(local_mask > 0, iterations=max(0, dilations - 1))
                else:
                    instance_mask = local_mask > 0
                    
                padded_slice = (slice(y_start, y_end), slice(x_start, x_end))
                
                if 'n_pixels' not in inst_record:
                    inst_record['n_pixels'] = npx
                    
                # Accumulate the dilated mask of the cell into the global layout for transparent debug plotting
                if source['type'] == 'hsi':
                    global_mask[padded_slice][instance_mask] = 255
                
                if source['type'] == 'ratios':
                    for slice_idx in range(source['num_pages']):
                        data_slice = source['handle'].pages[slice_idx].asarray()
                        ratio_basename = source['ratio_basename']
                        r_type_slice = f"{ratio_basename}_{slice_idx + 1}" if source['num_pages'] > 1 else ratio_basename
                        col_name = f"{source['column_prefix']}{r_type_slice}"
                        
                        masked_vals = data_slice[padded_slice][instance_mask]
                        vals = masked_vals[(masked_vals > 0) & np.isfinite(masked_vals)]
                        if vals.size > 0:
                            mean_ratio = float(np.nanmean(vals))
                            inst_record[col_name] = mean_ratio
                            has_any_data = True
                            
                elif source['type'] == 'sam_classmap':
                    patch = source['matrix'][padded_slice][instance_mask]
                    labels, counts = np.unique(patch, return_counts=True)
                    name_counts = {}
                    for lbl, cnt in zip(labels, counts):
                        lbl_int = int(lbl)
                        if lbl_int <= 0:
                            continue
                        class_idx = lbl_int - 1
                        if class_idx < len(source['class_names']):
                            name_counts[source['class_names'][class_idx]] = int(cnt)
                    total = sum(name_counts.values())
                    if total > 0:
                        for mol, cnt in name_counts.items():
                            pct = (cnt / total) * 100.0
                            col_name = f"SAM_{mol}"
                            inst_record[col_name] = float(pct)
                            has_any_data = True
                            
                elif source['type'] == 'predictions':
                    preds = source['matrix'][padded_slice][instance_mask]
                    unique, counts = np.unique(preds, return_counts=True)
                    class_counts = {}
                    for k, v in zip(unique, counts):
                        k_str = str(k).strip()
                        if mol_groups is not None and k_str in mol_groups:
                            mapped_name = mol_groups[k_str]
                        else:
                            mapped_name = k_str
                        class_counts[mapped_name] = class_counts.get(mapped_name, 0) + int(v)
                        
                    for c in classes_to_ignore_set:
                        if c in class_counts:
                            del class_counts[c]
                            
                    total = sum(class_counts.values())
                    if total > 0:
                        for mol, cnt in class_counts.items():
                            pct = (cnt / total) * 100.0
                            col_name = f"RF_{mol}"
                            inst_record[col_name] = float(pct)
                            has_any_data = True
                            
                elif source['type'] == 'hsi':
                    hsi_data = source['matrix']
                    num_channels = hsi_data.shape[0]
                    for ch_idx in range(num_channels):
                        col_name = f"CH{ch_idx + 1}"
                        masked_vals = hsi_data[ch_idx][padded_slice][instance_mask]
                        vals = masked_vals[np.isfinite(masked_vals)]
                        if vals.size > 0:
                            inst_record[col_name] = float(np.nanmean(vals))
                            has_any_data = True

            if has_any_data:
                records.append(inst_record)
                
        # Generate and save transparent overlap debug plot
        hsi_src = next((s for s in loaded_sources if s['type'] == 'hsi'), None)
        if hsi_src is not None:
            try:
                hsi_mip = np.max(hsi_src['matrix'], axis=0)
                # Normalize MIP for display
                mip_min = hsi_mip.min()
                mip_max = hsi_mip.max()
                if mip_max > mip_min:
                    mip_norm = (hsi_mip - mip_min) / (mip_max - mip_min)
                else:
                    mip_norm = np.zeros_like(hsi_mip, dtype=float)
                
                # Make RGB base
                overlay_rgb = np.stack([mip_norm, mip_norm, mip_norm], axis=-1)
                
                # Overlay global mask transparently in red
                # Blend: 70% original HSI MIP + 30% pure red wherever mask > 0
                mask_idx = global_mask > 0
                overlay_rgb[mask_idx] = 0.7 * overlay_rgb[mask_idx] + 0.3 * np.array([1.0, 0.0, 0.0])
                
                plt.figure(figsize=(10, 10))
                plt.imshow(overlay_rgb)
                plt.title("Total HSI MIP with Transparent Mask Overlay (Red)")
                plt.axis('off')
                
                out_dir = os.path.dirname(output_flat_csv) if output_flat_csv else '.'
                os.makedirs(out_dir, exist_ok=True)
                debug_img_path = os.path.join(out_dir, "debug_alignment.png")
                plt.savefig(debug_img_path, dpi=200, bbox_inches='tight')
                plt.close()
                print(f"\n[DEBUG] Diagnostic transparent alignment overlap saved to: {debug_img_path}")
            except Exception as ex:
                print(f"Failed to generate global debug alignment plot: {ex}")
                
    finally:
        # Guarantee all opened TiffFiles are properly closed
        for source in loaded_sources:
            if source['type'] == 'ratios' and 'handle' in source:
                try:
                    source['handle'].close()
                except Exception as e:
                    print(f"Failed to close TIFF handle: {e}")

    if not records:
        raise ValueError('No valid instances found across all masks.')

    df_out = pd.DataFrame.from_records(records)
    
    # Fill percentage columns with 0.0 where NaN
    pct_cols = [c for c in df_out.columns if c.startswith(('RF_', 'SAM_'))]
    if pct_cols:
        df_out[pct_cols] = df_out[pct_cols].fillna(0.0)

    # Optionally write flat output
    if output_flat_csv:
        df_out.to_csv(output_flat_csv, index=False)

    return df_out




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



def main():
    pass

    # =========================================================================
    # Load reference spectra and prepare for analysis
    # =========================================================================

    base_dir = os.path.dirname(CONFIG['image_dir'])


    output_dir = os.path.join(base_dir, 'sam_outputs')
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

    PALETTE = sns.color_palette("husl", len(target_names))

    # =========================================================================
    # MASKING AND INSTANCE ANALYSIS (OPTIONAL)
    # =========================================================================
    from pathlib import Path
    from functools import reduce
    import sys
    sys.path.insert(0, str(Path.cwd().parent))


    masking_output_dir = os.path.join(base_dir, 'masking_outputs')
    os.makedirs(masking_output_dir, exist_ok=True)

    if MASKING_CONFIG['perform_masking']:
        print("\nPerforming masking and instance analysis...")

        ratios_dir = os.path.join(os.path.dirname(CONFIG['image_dir']), 'Ratio')
        sam_outputs_dir = os.path.join(os.path.dirname(CONFIG['image_dir']), 'sam_outputs_grouped')
        rf_outputs_dir = os.path.join(os.path.dirname(CONFIG['image_dir']), 'rf_outputs')
        collagen_outputs_dir = os.path.join(os.path.dirname(CONFIG['image_dir']), 'collagen_outputs')
        merge_keys = ['centroid_x', 'centroid_y', 'Instance_Label', 'Unit', 'Image_Name']
        df_list = []

        for img_file in image_files:
            img_path = os.path.join(CONFIG['image_dir'], img_file)
            img_base = os.path.splitext(img_file)[0]
            img_base = re.sub(r'[^\w\-_\. ]', '', img_base).strip()

            mask_list_path = os.path.join(MASKING_CONFIG['mask_dir'], f"{img_base}")

            # Prepare data configurations to extract
            data_configs = []

            if os.path.exists(img_path) and "HSI" in img_base:
                # Add the raw hyperspectral image as a source to extract channel means
                data_configs.append({
                    'mode': 'hsi',
                    'files': [img_path],
                    'dilations': MASKING_CONFIG['dilations']
                })

            if os.path.exists(ratios_dir):
                ratio_files = glob(os.path.join(ratios_dir, img_base, "*-ratio.tif"))
                if ratio_files:
                    data_configs.append({
                        'mode': 'ratios',
                        'files': ratio_files,
                        'dilations': MASKING_CONFIG['dilations']
                    })

            if os.path.exists(sam_outputs_dir):
                sam_files = glob(os.path.join(sam_outputs_dir, img_base, f"*_sam_classification.tif"))
                if sam_files:
                    data_configs.append({
                        'mode': 'sam_classmap',
                        'files': sam_files,
                        'dilations': MASKING_CONFIG['dilations']
                    })

            if os.path.exists(rf_outputs_dir):
                rf_files = glob(os.path.join(rf_outputs_dir, img_base, f"*_predictions.csv"))
                if rf_files:
                    data_configs.append({
                        'mode': 'predictions',
                        'files': rf_files,
                        'dilations': MASKING_CONFIG['dilations']
                    })

            if os.path.exists(collagen_outputs_dir):
                collagen_files = glob(os.path.join(collagen_outputs_dir, img_base, f"*-collagen.tif"))
                if collagen_files:
                    data_configs.append({
                        'mode': 'collagen',
                        'files': collagen_files,
                        'dilations': np.maximum(10, MASKING_CONFIG['dilations']),
                    })

            if data_configs:
                # ── Locate parquet cell boundary file ─────────────────────
                parquet_path = None
                if MASKING_CONFIG.get('use_parquet_cells', False):
                    parquet_pattern = os.path.join(MASKING_CONFIG['parquet_dir'], "**", f"*{img_base}_cells.parquet")
                    parquet_files = glob(parquet_pattern, recursive=True)
                    if not parquet_files:
                        roi_name = img_base.split('-')[-2] if len(img_base.split('-')) == 3 else img_base
                        sample_name = img_base.split('-')[0] if len(img_base.split('-')) == 2 else img_base
                        fallback_pattern = os.path.join(MASKING_CONFIG['parquet_dir'], "**", f"*{sample_name}*{roi_name}*_cells.parquet")
                        parquet_files = glob(fallback_pattern, recursive=True)
                    if parquet_files:
                        parquet_path = parquet_files[0]
                        print(f"Found parquet cell boundaries: {parquet_path}")
                    else:
                        print(f"Warning: Cell boundary parquet file not found for {img_base}. Skipping parquet extraction.")

                # ── Check for existing output CSV (update-mode support) ────
                unified_csv_path = os.path.join(masking_output_dir, f"{img_base}_unified.csv")
                update_mode_active = (
                    MASKING_CONFIG.get('update_mode', False)
                    and os.path.exists(unified_csv_path)
                )

                meta_cols = ['centroid_x', 'centroid_y', 'Instance_Label', 'Unit',
                             'Image_Name', 'Source_ID', 'Group_ID', 'n_pixels']

                if update_mode_active:
                    # ── UPDATE MODE: only re-extract the requested modes ───
                    update_modes = set(MASKING_CONFIG.get('update_modes', ['ratios']))
                    update_configs = [c for c in data_configs if c['mode'] in update_modes]

                    if not update_configs:
                        print(f"[Update] No matching data configs for modes {update_modes}; "
                              f"loading existing CSV for {img_base}.")
                        unified_df = pd.read_csv(unified_csv_path)
                    else:
                        print(f"[Update] Re-extracting modes {update_modes} for {img_base} "
                              f"(existing CSV found, all other columns preserved).")

                        # Run masking for only the update modes
                        if parquet_path is not None:
                            update_df = apply_unified_parquet_masking(
                                data_configs=update_configs,
                                parquet_path=parquet_path,
                                target_names=target_names,
                                mol_groups=CONFIG.get('mol_groups', None),
                                min_instance_size=MASKING_CONFIG['min_instance_size'],
                                classes_to_ignore=MASKING_CONFIG['classes_to_ignore'],
                                img_name=img_base,
                            )
                        else:
                            update_df = apply_unified_masking(
                                data_configs=update_configs,
                                mask_list_path=mask_list_path,
                                target_names=target_names,
                                mol_groups=CONFIG.get('mol_groups', None),
                                prefix=MASKING_CONFIG['prefix'],
                                mask_dict=MASKING_CONFIG['mask_dict'],
                                subgroups=MASKING_CONFIG['subgroups'],
                                img_name=img_base,
                                classes_to_ignore=MASKING_CONFIG['classes_to_ignore'],
                                min_instance_size=MASKING_CONFIG['min_instance_size'],
                            )

                        existing_df = pd.read_csv(unified_csv_path)

                        # Drop stale columns for every updated mode
                        cols_to_drop = []
                        for mode in update_modes:
                            cols_to_drop.extend(_get_mode_columns(existing_df, mode, meta_cols))
                        cols_to_drop = [c for c in cols_to_drop if c in existing_df.columns]
                        if cols_to_drop:
                            print(f"[Update] Dropping {len(cols_to_drop)} stale column(s): "
                                  f"{cols_to_drop[:8]}{'...' if len(cols_to_drop) > 8 else ''}")
                        base_df = existing_df.drop(columns=cols_to_drop)

                        # Identify the new data columns produced by the update
                        new_data_cols = [c for c in update_df.columns
                                         if c not in meta_cols and c != 'Instance_Label']
                        update_subset = update_df[['Instance_Label'] + new_data_cols]

                        # Left-join: keep all existing cells; update only matched ones
                        unified_df = base_df.merge(update_subset, on='Instance_Label', how='left')

                        # Persist the updated unified CSV
                        unified_df.to_csv(unified_csv_path, index=False)
                        print(f"[Update] Saved updated unified CSV: {unified_csv_path}")

                else:
                    # ── FULL RUN: extract all configured sources ──────────
                    if parquet_path is not None:
                        unified_df = apply_unified_parquet_masking(
                            data_configs=data_configs,
                            parquet_path=parquet_path,
                            target_names=target_names,
                            mol_groups=CONFIG.get('mol_groups', None),
                            min_instance_size=MASKING_CONFIG['min_instance_size'],
                            classes_to_ignore=MASKING_CONFIG['classes_to_ignore'],
                            img_name=img_base,
                            output_flat_csv=unified_csv_path,
                        )
                    else:
                        unified_df = apply_unified_masking(
                            data_configs=data_configs,
                            mask_list_path=mask_list_path,
                            target_names=target_names,
                            mol_groups=CONFIG.get('mol_groups', None),
                            prefix=MASKING_CONFIG['prefix'],
                            mask_dict=MASKING_CONFIG['mask_dict'],
                            subgroups=MASKING_CONFIG['subgroups'],
                            img_name=img_base,
                            classes_to_ignore=MASKING_CONFIG['classes_to_ignore'],
                            min_instance_size=MASKING_CONFIG['min_instance_size'],
                            output_flat_csv=unified_csv_path,
                        )

                df_list.append(unified_df)

                # Export individual CSVs if toggle is enabled
                if MASKING_CONFIG.get('export_individual_csvs', True):
                    actual_meta_cols = [c for c in meta_cols if c in unified_df.columns]

                    # 1. Ratios CSV
                    ratio_cols = [c for c in unified_df.columns if c not in meta_cols and not c.startswith(('RF_', 'SAM_', 'Collagen_')) and not re.match(r'^CH\d+$', c)]
                    if ratio_cols:
                        ratios_csv_path = os.path.join(masking_output_dir, f"{img_base}_ratios.csv")
                        unified_df[actual_meta_cols + ratio_cols].to_csv(ratios_csv_path, index=False)

                    # 2. SAM CSV
                    sam_cols = [c for c in unified_df.columns if c.startswith('SAM_')]
                    if sam_cols:
                        sam_csv_path = os.path.join(masking_output_dir, f"{img_base}_sam.csv")
                        unified_df[actual_meta_cols + sam_cols].to_csv(sam_csv_path, index=False)

                    # 3. RF CSV
                    rf_cols = [c for c in unified_df.columns if c.startswith('RF_')]
                    if rf_cols:
                        rf_csv_path = os.path.join(masking_output_dir, f"{img_base}_rf.csv")
                        unified_df[actual_meta_cols + rf_cols].to_csv(rf_csv_path, index=False)

                    # 4. Collagen CSV
                    collagen_cols = [c for c in unified_df.columns if c.startswith('Collagen_')]
                    if collagen_cols:
                        collagen_df = unified_df[actual_meta_cols + collagen_cols].copy()
                        collagen_df = collagen_df.rename(columns={c: c[len('Collagen_'):] for c in collagen_cols})
                        collagen_csv_path = os.path.join(masking_output_dir, f"{img_base}_collagen.csv")
                        collagen_df.to_csv(collagen_csv_path, index=False)

                    # 5. HSI Channels CSV
                    hsi_cols = [c for c in unified_df.columns if re.match(r'^CH\d+$', c)]
                    if hsi_cols:
                        hsi_csv_path = os.path.join(masking_output_dir, f"{img_base}_hsi.csv")
                        unified_df[actual_meta_cols + hsi_cols].to_csv(hsi_csv_path, index=False)

        if df_list:
            final_df = pd.concat(df_list, ignore_index=True)
            final_csv_path = os.path.join(masking_output_dir, "summary_masked_instances.csv")
            final_df.to_csv(final_csv_path, index=False)
            print(f"\nSaved final masked instance CSV: {final_csv_path}")


if __name__ == "__main__":
    main()