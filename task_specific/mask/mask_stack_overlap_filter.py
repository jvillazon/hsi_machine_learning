"""
Filter a query mask stack based on its overlap with a reference mask stack.
Outputs individual 2D masks for each overlapping combination.

Goal
----
Identifies connected components (instances) in the 'query' mask stack (e.g. cell types)
and retains those that overlap with specific channels in the 'reference' mask stack (e.g. collagen).
Saves individual 8-bit TIFF files for each successful overlap.
"""

import os
from pathlib import Path

import numpy as np
import tifffile
import scipy.ndimage as ndi
from tqdm import tqdm


def filter_overlap_slice(query_slice: np.ndarray, ref_slice: np.ndarray) -> np.ndarray:
    """
    Filter connected components in query_slice that overlap with ref_slice.
    Returns 8-bit mask (0 or 255).
    """
    if np.sum(query_slice) == 0 or np.sum(ref_slice) == 0:
        return None
    
    # Label connected components in query
    labeled_query, num_features = ndi.label(query_slice > 0)
    if num_features == 0:
        return None
    
    # Find label IDs that intersect with the reference mask
    overlapping_labels = np.unique(labeled_query[ref_slice > 0])
    overlapping_labels = overlapping_labels[overlapping_labels > 0]
    
    if len(overlapping_labels) == 0:
        return None
    
    # Reconstruct the mask containing only overlapping labels
    if len(overlapping_labels) == 1:
        filtered_mask = (labeled_query == overlapping_labels[0])
    else:
        filtered_mask = np.isin(labeled_query, overlapping_labels)
        
    return (filtered_mask.astype(np.uint8) * 255)


def subtract_claimed_instances(filtered_slice: np.ndarray, claimed_mask: np.ndarray) -> np.ndarray:
    """
    Remove any connected components from filtered_slice that overlap with claimed_mask.
    Operates at the instance level: if any pixel of a component touches the claimed region,
    the entire component is removed.

    Parameters
    ----------
    filtered_slice : np.ndarray
        8-bit mask (0 or 255) of overlapping query instances for the current ref channel.
    claimed_mask : np.ndarray
        Boolean array of pixels already assigned to prior ref channels.

    Returns
    -------
    np.ndarray or None
        8-bit mask with previously-claimed instances removed, or None if nothing remains.
    """
    if not np.any(claimed_mask):
        return filtered_slice  # nothing claimed yet -- return as-is

    labeled, _ = ndi.label(filtered_slice > 0)

    # Find component labels that touch the claimed region
    claimed_labels = np.unique(labeled[claimed_mask])
    claimed_labels = claimed_labels[claimed_labels > 0]

    if len(claimed_labels) == 0:
        return filtered_slice  # no overlap with claimed -- keep all

    # Remove those components
    remaining_mask = np.isin(labeled, claimed_labels, invert=True) & (labeled > 0)
    if not np.any(remaining_mask):
        return None

    return (remaining_mask.astype(np.uint8) * 255)


def main():
    # ========== CONFIGURATION ==========
    ref_path = r"C:\Users\jvill\Documents\integrated_pipeline\WSI_data\CODEX\cluster_data\K21_198_3\Immune-Cluster-Stack.tif"
    query_path = r"C:\Users\jvill\Documents\integrated_pipeline\WSI_data\WSI_data\MALDI_CODEX_masks\S21-K21_198_3\PT-clusters.tif"
    output_dir = r"C:\Users\jvill\Documents\integrated_pipeline\WSI_data\WSI_data\MALDI_Immune_tubules_masks\S21-K21_198_3"

    # Map query slice indices to names (from mask_images.py)
    # query_labels = {
    #     "20": "Cluster1",
    #     "15": "Cluster2",
    #     "19": "Cluster3",
    #     "16": "Cluster4",
    #     "9": "Cluster5",
    #     "18": "Cluster6",
    #     "26": "Cluster7",
    #     "1": "Cluster8",
    #     "0": "Cluster9",
    #     "23": "Cluster10",
    #     "25": "Cluster11",
    #     "2": "Cluster12",
    #     "3": "Cluster13",
    #     "24": "Cluster14",
    #     "12": "Cluster15",
    #     "21": "Cluster16",
    #     "17": "Cluster17",
    #     "10": "Cluster18",
    #     "4": "Cluster19",
    #     "5": "Cluster20",
    #     "14": "Cluster21",
    #     "22": "Cluster22",
    #     "8": "Cluster23",
    #     "11": "Cluster24",
    #     "13": "Cluster25",
    #     "6": "Cluster26",
    #     "7": "Cluster27",
    # }

    # query_labels = {
    # "0": "ProxTub",
    # }

    query_labels = {
        "0": "Cluster2",
        "1": "Cluster3",
        "2": "Cluster4",
        "3": "Cluster5",
    }

    # query_labels = {
    #     "0": "Glom",
    #     "1": "ProxTub",
    #     "2": "DistNeph",
    #     "3": "TAL",
    #     "4": "DistTub"
    # }

    # query_labels = {
    #     "0": "DAPI",
    #     "1": "Glom",
    #     "2": "ProxTub",
    #     "3": "TDL",
    #     "4": "TAL",
    #     "5": "DistTub",
    #     "6": "DistNeph",
    #     "7": "Vasc"
    # }
    # query_labels = {
    #     "0": "Scale10_collagen",
    #     "1": "Scale50_collagen",
    #     "2": "Scale100_collagen",
    #     "3": "Scale150_collagen",
    # }

    # r_labels = [
    #     "Cluster2",
    #     "Cluster3",
    #     "Cluster4",
    #     "Cluster5",
    # ]

    # r_labels = [
    #     '10um',
    #     '50um',
    #     '100um',
    #     '150um',
    #     '+150um'
    # ]

    # r_labels = [
    #     'Medulla',
    #     'Cortex'
    # ]

    # r_labels =[
    #     "C1",
    #     "C2",
    #     "C3",
    #     "CM",
    #     "M2",
    #     "M"
    # ]

    # r_labels = [
    #     "Cluster9",
    #     "Cluster8",
    #     "Cluster12",
    #     "Cluster13",
    #     "Cluster19",
    #     "Cluster20",
    #     "Cluster26",
    #     "Cluster27",
    #     "Cluster23",
    #     "Cluster5",
    #     "Cluster18",
    #     "Cluster24",
    #     "Cluster15",
    #     "Cluster25",
    #     "Cluster21",
    #     "Cluster2",
    #     "Cluster4",
    #     "Cluster17",
    #     "Cluster6",
    #     "Cluster3",
    #     "Cluster1",
    #     "Cluster16",
    #     "Cluster22",
    #     "Cluster10",
    #     "Cluster14",
    #     "Cluster11",
    #     "Cluster7",

    # ]

    r_labels = [
        "CD3,CD4",
        "CD3,CD8",
        "CD45,CD206",
        "WSI"
    ]


    exclude_prior_overlaps= False
    # ===================================

    query_path = Path(query_path)
    ref_path = Path(ref_path)
    os.makedirs(output_dir, exist_ok=True)

    image_name = query_path.stem.split('.')[0]  # Handle .ome.tiff
    if '-' in image_name:
        image_name = image_name.split('-')[0]

    # 1. Inspect metadata
    with tifffile.TiffFile(query_path) as tif_q:
        q_pages_count = len(tif_q.pages)
        q_shape = tif_q.series[0].shape

    # # with tifffile.TiffFile(ref_path) as tif_r:
    #     r_pages_count = len(tif_r.pages)
    #     r_metadata = tif_r.imagej_metadata if tif_r.imagej_metadata else {}
    #     # Try to get channel labels for the reference stack
    #     # r_labels = r_metadata.get('Labels', [f"scale_{i}" for i in range(r_pages_count)])

    print(f"Image Name: {image_name}")
    # print(f"Processing {q_pages_count} query channels vs {r_pages_count} reference channels...")

    # Per-query cumulative claimed mask: tracks pixels already saved for earlier ref channels.
    # Lazily initialized on first use; keyed by q_idx.
    claimed_masks: dict = {}  # q_idx -> bool ndarray

    # 2. Iterate through each Reference Channel (Outer Loop)
    with tifffile.TiffFile(ref_path) as tif_r:
        for r_idx, r_page in enumerate(tif_r.pages):
            ref_label = r_labels[r_idx] if r_idx < len(r_labels) else f"ch{r_idx}"
            ref_label_safe = str(ref_label).replace(' ', '_').replace('/', '_').replace(':', '')

            ref_slice = r_page.asarray()
            if not np.any(ref_slice):
                continue

            # 3. Iterate through each Query Slice (Inner Loop)
            with tifffile.TiffFile(query_path) as tif_q:
                for q_idx, q_page in enumerate(tqdm(tif_q.pages, desc=f"Ref: {ref_label}", leave=False)):
                    query_label = query_labels.get(str(q_idx), query_labels.get(q_idx, f"slice{q_idx:02d}"))

                    query_slice = q_page.asarray()
                    filtered_slice = filter_overlap_slice(query_slice, ref_slice)
                    if filtered_slice is None or not int(np.sum(filtered_slice)) > 0:
                        continue

                    # --- Exclusion of previously claimed instances ---
                    if exclude_prior_overlaps and q_idx in claimed_masks:
                        filtered_slice = subtract_claimed_instances(filtered_slice, claimed_masks[q_idx])
                        if filtered_slice is None or not np.any(filtered_slice):
                            continue

                    if filtered_slice is not None:
                        # Save individual 8-bit TIFF
                        out_filename = f"MALDI-{query_label}_{ref_label_safe}-mask.tif"
                        out_path = os.path.join(output_dir, out_filename)
                        tifffile.imwrite(out_path, filtered_slice, compression='ADOBE_DEFLATE')

                        # Update claimed mask for this query slice
                        if exclude_prior_overlaps:
                            if q_idx not in claimed_masks:
                                claimed_masks[q_idx] = np.zeros(filtered_slice.shape, dtype=bool)
                            claimed_masks[q_idx] |= (filtered_slice > 0)

    print(f"Done. Individual masks saved to: {output_dir}")


if __name__ == "__main__":
    main()
