"""
Script to apply masks from a directory of mask files to a directory of images.

Masks and images are matched by filename (e.g., image1.tif is masked by image1.tif).
Supports 8-bit or 32-bit masks.
"""

import os
import glob
import numpy as np
import tifffile
from tqdm import tqdm


def load_mask(mask_path: str, slice_idx: int = None) -> np.ndarray:
    """
    Load a .tif mask file (2D or 3D stack).
    
    Args:
        mask_path: Path to the .tif mask file
        slice_idx: Optional index of the slice to load (for 3D stacks)
        
    Returns:
        np.ndarray: Mask array (2D) with binary uint8 values (0 or 1)
    """
    # Load the .tif file and return as numpy array
    if slice_idx is not None:
        mask = tifffile.imread(mask_path, key=slice_idx)
    else:
        mask = tifffile.imread(mask_path)

    # Verify if 8-bit (0-255 value) or 32-bit (0-1 value)
    if mask.dtype != np.uint8 and mask.dtype != np.float32:
        # Some masks might be bool or other types, let's be more flexible but warn
        if mask.dtype == np.bool_:
            return mask.astype(np.uint8)
        raise ValueError(f"Mask has unexpected dtype: {mask.dtype}. Expected uint8 or float32.")
    
    mask = mask > 0  # Convert to boolean mask

    return mask.astype(np.uint8)

def main():
    """
    Main execution function.
    """
    # ========== CONFIGURATION ==========
    mask_path = r"C:\Users\jvill\Documents\integrated_pipeline\WSI_data\WSI_data\mask_to_process\K21_198_3\K21_198_3-Clusters-transformed.ome.tiff"       # Directory containing mask files
    image_dir = r"C:\Users\jvill\Documents\integrated_pipeline\WSI_data\WSI_data\mask_to_process\K21_198_3-collagen"     # Directory containing images to be masked
    output_dir = os.path.join(os.path.dirname(image_dir), "output_masks")  # Output directory for masked images
    os.makedirs(output_dir, exist_ok=True)

    # (Optional) Label 3D mask slices if needed
    slice_labels = {
        "20": "Cluster1",
        "15": "Cluster2",
        "19": "Cluster3",
        "16": "Cluster4",
        "9": "Cluster5",
        "18": "Cluster6",
        "26": "Cluster7",
        "1": "Cluster8",
        "0": "Cluster9",
        "23": "Cluster10",
        "25": "Cluster11",
        "2": "Cluster12",
        "3": "Cluster13",
        "24": "Cluster14",
        "12": "Cluster15",
        "21": "Cluster16",
        "17": "Cluster17",
        "10": "Cluster18",
        "4": "Cluster19",
        "5": "Cluster20",
        "14": "Cluster21",
        "22": "Cluster22",
        "8": "Cluster23",
        "11": "Cluster24",
        "13": "Cluster25",
        "6": "Cluster26",
        "7": "Cluster27",
    }

    # slice_labels = {
    #     "0" : "Vasc",
    #     "1" : "DistNeph",
    #     "2" : "DistTub",
    #     "3" : "Glom",
    #     "4" : "ProxTub",
    #     "5" : "TAL",
    #     "6" : "TDL"
    # }
    # slice_labels = {
    #     "0": "Glom",
    #     "1": "ProxTub",
    #     "2": "DistNeph",
    #     "3": "TAL",
    #     "4": "DistTub"
    # }

    # ===================================
    
    # Validate inputs
    # if not os.path.isdir(mask_dir):
    #     raise NotADirectoryError(f"Mask directory not found: {mask_dir}")
    
    if not os.path.isdir(image_dir):
        raise NotADirectoryError(f"Image directory not found: {image_dir}")
    
    # Process images and match with corresponding masks
    processed_count = 0
    for image_path in tqdm(glob.glob(os.path.join(image_dir, "*.tif")), desc="Processing images"):
        image_filename = os.path.basename(image_path)
        image_name = os.path.splitext(image_filename)[0]  # Get name without extension
        
        # Check if corresponding mask exists
        if not os.path.isfile(mask_path):
            print(f"Warning: No matching mask found for {image_filename}, skipping...")
            continue
        
        # Create output folder for this image
        image_output_dir = os.path.join(output_dir, image_name)
        os.makedirs(image_output_dir, exist_ok=True)
        
        # Get mask metadata without loading the whole array
        with tifffile.TiffFile(mask_path) as tif:
            # Handle both OME-TIFF and standard TIFF
            series = tif.series[0]
            mask_shape = series.shape
            mask_ndim = len(mask_shape)
            
        image_array = tifffile.imread(image_path)
        
        # Check if mask is 2D or 3D
        if mask_ndim == 2:
            # Apply single 2D mask
            mask = load_mask(mask_path)
            masked_array = np.where(mask, image_array, 0)
            output_path = os.path.join(image_output_dir, f"{image_name}.tif")
            tifffile.imwrite(output_path, masked_array)
            processed_count += 1
            
        elif mask_ndim == 3:
            num_slices = mask_shape[0]
            if slice_labels is not None and num_slices != len(slice_labels):
                print(f"Warning: Number of mask slices ({num_slices}) does not match number of labels ({len(slice_labels)}) for {image_filename}, skipping...")
                continue

            # Apply each slice of 3D mask one by one
            for slice_idx in range(num_slices):
                mask_slice = load_mask(mask_path, slice_idx=slice_idx)
                
                # Check if slice dimensions match image
                if mask_slice.shape != image_array.shape[:2]:
                    # Handle potential dimension mismatch (e.g. OME-TIFF metadata vs series shape)
                    print(f"Warning: Mask slice {slice_idx} shape {mask_slice.shape} does not match image shape {image_array.shape[:2]}")
                    continue

                masked_array = np.where(mask_slice, image_array, 0)

                if np.sum(masked_array)==0:
                    continue
                
                if slice_labels is not None:
                    # The dictionary slice_labels uses string keys in the script, 
                    # but slice_idx is an int. Let's handle both just in case.
                    label = slice_labels.get(str(slice_idx), slice_labels.get(slice_idx, f"slice{slice_idx:03d}"))
                    output_path = os.path.join(image_output_dir, f"CODEX-{label}_{image_name}-mask.tif")
                else:
                    output_path = os.path.join(image_output_dir, f"{image_name}_slice{slice_idx:03d}.tif")
                
                tifffile.imwrite(output_path, masked_array)
                
            processed_count += 1
            print(f"Processed {image_filename} with {num_slices} slices")
        else:
            print(f"Warning: Unexpected mask dimensions ({mask_ndim}) for {image_filename}, skipping...")
            continue
    
    # Print summary
    print(f"Processed {processed_count} images")
    print(f"Masked images saved to: {output_dir}")
    

if __name__ == "__main__":
    main()
