"""
Process CODEX mask images and multiply them with corresponding slices from hyperstack images.

This script:
1. Iterates through each folder in the Mask directory
2. Reads mask images (CODEX-<unit_name>-mask.tif)
3. Extracts unit name (ignoring _cortex/_medu suffixes)
4. Multiplies the mask with the corresponding slice from the hyperstack image

Before running this script, activate the conda environment:
    conda activate hsi_machine_learning

Then run:
    python process_codex_masks.py
"""

import os
import re
from pathlib import Path
from glob import glob
import numpy as np
from tifffile import imread, imwrite


# Unit name to slice index mapping
UNIT_SLICE_MAPPING = {
    'Vasc': 7,      # 8th slice (0-indexed)
    'ProxTub': 2,
    'Glom': 1,
    'TAL': 4,
    'ThAL': 3,
    'DistNeph': 6,
    'DistTub': 5,
}


def extract_unit_name(filename):
    """
    Extract unit name from mask filename, ignoring _cortex/_medu suffix.
    
    Args:
        filename: Filename like "CODEX-Vasc_cortex-mask.tif"
    
    Returns:
        Unit name like "Vasc"
    """
    match = re.search(r'CODEX-([^_-]+)(?:_(?:cortex|medu))?-mask\.tif', filename)
    if match:
        return match.group(1)
    return None


def process_masks(mask_base_dir, normalized_base_dir, output_dir=None):
    """
    Process all mask images and multiply with hyperstack slices.
    
    Args:
        mask_base_dir: Base directory containing Mask/<image_name> folders
        normalized_base_dir: Base directory containing Normalized_Images/<image_name> folders
        output_dir: Optional output directory for results (defaults to <mask_base_dir>/Processed)
    """
    mask_base_path = Path(mask_base_dir)
    normalized_base_path = Path(normalized_base_dir)
    
    if output_dir is None:
        output_path = mask_base_path.parent / "Processed_Masks"
    else:
        output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image name folders in the Mask directory
    image_folders = [d for d in mask_base_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    print(f"Found {len(image_folders)} image folders to process")
    
    for image_folder in image_folders:
        image_name = image_folder.name
        print(f"\nProcessing: {image_name}")
        
        # Find all mask files in this folder
        mask_files = list(image_folder.glob("CODEX-*-mask.tif"))
        
        if not mask_files:
            print(f"  No mask files found in {image_name}")
            continue
        
        # Find the corresponding hyperstack file
        hyperstack_pattern = normalized_base_path / image_name / "inv_transform" / "*_DAPI_Masks_transformed_inverse.ome.tiff"
        hyperstack_files = list(Path(normalized_base_path / image_name / "inv_transform").glob("*_DAPI_Masks_transformed_inverse.ome.tiff"))
        
        if not hyperstack_files:
            print(f"  WARNING: No hyperstack file found for {image_name}")
            continue
        
        hyperstack_file = hyperstack_files[0]
        print(f"  Loading hyperstack: {hyperstack_file.name}")
        
        try:
            # Load the hyperstack image
            hyperstack = imread(hyperstack_file)
            print(f"  Hyperstack shape: {hyperstack.shape}")
            
            # Create output folder for this image: output_path/<image_name>/
            image_output_path = output_path / image_name
            image_output_path.mkdir(parents=True, exist_ok=True)
            
            # Process each mask file
            for mask_file in mask_files:
                mask_filename = mask_file.name
                unit_name = extract_unit_name(mask_filename)
                
                if unit_name is None:
                    print(f"  WARNING: Could not extract unit name from {mask_filename}")
                    continue
                
                if unit_name not in UNIT_SLICE_MAPPING:
                    print(f"  WARNING: No slice mapping for unit '{unit_name}' in {mask_filename}")
                    continue
                
                slice_idx = UNIT_SLICE_MAPPING[unit_name]
                
                # Check if slice index is valid
                if slice_idx >= hyperstack.shape[0]:
                    print(f"  WARNING: Slice index {slice_idx} out of range for {mask_filename}")
                    continue
                
                print(f"  Processing {mask_filename} -> slice {slice_idx} ({unit_name})")
                
                # Load the mask
                mask = imread(mask_file)
                
                # Threshold mask to binary (0 or 255)
                mask_binary = (mask > 0).astype(np.uint8) * 255
                
                # Get the corresponding slice from hyperstack
                hyperstack_slice = hyperstack[slice_idx]
                
                # Check shape compatibility
                if mask_binary.shape != hyperstack_slice.shape:
                    print(f"    WARNING: Shape mismatch - mask: {mask_binary.shape}, slice: {hyperstack_slice.shape}")
                    # Try to resize or handle differently if needed
                    continue
                
                # Multiply binary mask with hyperstack slice
                result = (mask_binary > 0) * hyperstack_slice
                
                # Convert to 8-bit image
                result = result.astype(np.uint8)
                
                # Save result with the same filename as the original mask
                output_file = image_output_path / mask_filename
                imwrite(output_file, result)
                print(f"    Saved: {mask_filename}")
                
        except Exception as e:
            print(f"  ERROR processing {image_name}: {str(e)}")
            continue
    
    print(f"\n✓ Processing complete. Results saved to: {output_path}")


def main():
    """Main function to run the processing."""
    
    # Define base directories
    base_dir = "/Users/jorgevillazon/Documents/files/codex-srs/HuBMAP .tif files for Jorge Part 1"
    mask_dir = os.path.join(base_dir, "Mask")
    normalized_dir = os.path.join(base_dir, "Normalized_Images")
    
    # Optional: specify custom output directory
    # output_dir = "/path/to/custom/output"
    output_dir = None
    
    # Process all masks
    process_masks(mask_dir, normalized_dir, output_dir)


if __name__ == "__main__":
    main()
