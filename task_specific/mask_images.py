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


def load_mask(mask_path: str) -> np.ndarray:
    """
    Load a .tif mask file (2D or 3D stack).
    
    Args:
        mask_path: Path to the .tif mask file
        
    Returns:
        np.ndarray: Mask array (2D or 3D) with boolean values
    """
    # Load the .tif file and return as numpy array
    mask = tifffile.imread(mask_path)

    # Verify if 8-bit (0-255 value) or 32-bit (0-1 value)
    if mask.dtype != np.uint8 and mask.dtype != np.float32:
        raise ValueError("Mask must be either 8-bit (uint8) or 32-bit (float32) format.")
    
    mask = mask > 0  # Convert to boolean mask

    return mask

def main():
    """
    Main execution function.
    """
    # ========== CONFIGURATION ==========
    # TODO: Set your paths here
    mask_dir = r"/Volumes/ADATA SE880/ADATA Backup/Lipid Reference Library/Mask"       # Directory containing mask files
    image_dir = r"/Volumes/ADATA SE880/ADATA Backup/Lipid Reference Library/HSI_data"     # Directory containing images to be masked
    output_dir = os.path.join(os.path.dirname(image_dir), "output_masks")  # Output directory for masked images
    os.makedirs(output_dir, exist_ok=True)

    # (Optional) Label 3D mask slices if needed
    slice_labels = None
    # slice_labels = {
    #     0: "nuclei",
    #     1: "cytoplasm",
    #     2: "background"
    # }
    # ===================================
    
    # Validate inputs
    if not os.path.isdir(mask_dir):
        raise NotADirectoryError(f"Mask directory not found: {mask_dir}")
    
    if not os.path.isdir(image_dir):
        raise NotADirectoryError(f"Image directory not found: {image_dir}")
    
    # Process images and match with corresponding masks
    processed_count = 0
    for image_path in tqdm(glob.glob(os.path.join(image_dir, "*.tif")), desc="Processing images"):
        image_filename = os.path.basename(image_path)
        image_name = os.path.splitext(image_filename)[0]  # Get name without extension
        mask_path = os.path.join(mask_dir, image_filename)
        
        # Check if corresponding mask exists
        if not os.path.isfile(mask_path):
            print(f"Warning: No matching mask found for {image_filename}, skipping...")
            continue
        
        # Create output folder for this image
        image_output_dir = os.path.join(output_dir, image_name)
        os.makedirs(image_output_dir, exist_ok=True)
        
        # Load mask and image
        mask = load_mask(mask_path)
        image_array = tifffile.imread(image_path)
        


        # Check if mask is 2D or 3D
        if mask.ndim == 2:
            # Apply single 2D mask
            masked_array = np.where(mask, image_array, 0)
            output_path = os.path.join(image_output_dir, f"{image_name}.tif")
            tifffile.imwrite(output_path, masked_array)
            processed_count += 1
            
        elif mask.ndim == 3:
            if slice_labels is not None and mask.shape[0] != len(slice_labels):
                print(f"Warning: Number of mask slices does not match number of labels for {image_filename}, skipping...")
                continue

            # Apply each slice of 3D mask
            for slice_idx in range(mask.shape[0]):
                mask_slice = mask[slice_idx]
                masked_array = np.where(mask_slice, image_array, 0)
                if slice_labels is not None:
                    label = slice_labels.get(slice_idx, f"slice{slice_idx:03d}")
                    output_path = os.path.join(image_output_dir, f"{image_name}_{label}.tif")
                else:
                    output_path = os.path.join(image_output_dir, f"{image_name}_slice{slice_idx:03d}.tif")
                tifffile.imwrite(output_path, masked_array)
            processed_count += 1
            print(f"Processed {image_filename} with {mask.shape[0]} slices")
        else:
            print(f"Warning: Unexpected mask dimensions for {image_filename}, skipping...")
            continue
    
    # Print summary
    print(f"Processed {processed_count} images")
    print(f"Masked images saved to: {output_dir}")
    

if __name__ == "__main__":
    main()
