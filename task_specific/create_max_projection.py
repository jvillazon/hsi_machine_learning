"""
Create max projection images from mask stacks in subfolders of image directories.

Usage:
    python create_max_projection.py <image_dir> <mask_stack_subfolder> <mask_pattern> <output_name>

- <image_dir>: Directory containing image folders (each with a mask stack in a subfolder)
- <mask_stack_subfolder>: Name of subfolder inside each image folder containing the mask stack .tif
- <mask_pattern>: Pattern to match mask stack files (e.g. '*mask.tif')
- <output_name>: Name for the output max projection file (e.g. 'max_projection.tif')
"""

import os
import glob
import sys
from tifffile import imread, imwrite
from tqdm import tqdm

def main(image_dir, mask_stack_subfolder, mask_pattern):
    image_dir = os.path.abspath(image_dir)
    # For each image folder
    for image_folder_name in tqdm([os.path.basename(d) for d in glob.glob(os.path.join(image_dir, "*")) if os.path.isdir(os.path.join(image_dir, d))]):
        image_folder = os.path.join(image_dir, image_folder_name)
        mask_stack_dir = os.path.join(image_folder, mask_stack_subfolder)
        if not os.path.isdir(mask_stack_dir):
            print(f"No mask stack subfolder found: {mask_stack_dir}")
            continue
        mask_stacks = glob.glob(os.path.join(mask_stack_dir, mask_pattern))
        if not mask_stacks:
            print(f"No mask stack found in {mask_stack_dir} matching {mask_pattern}")
            continue
        mask_stack_path = mask_stacks[0]
        mask_stack = imread(mask_stack_path)
        print(f"Processing {mask_stack_path} with shape {mask_stack.shape}")
        # Compute max projection along axis 0
        max_proj = mask_stack.max(axis=0)

        output_name = f"{os.path.splitext(mask_stack_path)[0]}_max_projection.tif"
        out_path = os.path.join(mask_stack_dir, output_name)
        imwrite(out_path, max_proj)
        print(f"Saved max projection to {out_path}")
    print(f"\n✓ All max projections created.")

if __name__ == "__main__":


    image_dir = "D:\\integrated_pipeline\\HSI_data\\Normalized_Images"
    mask_pattern = "*Clusters-transformed_transformed_inverse.ome.tiff"
    mask_stack_subfolder = "inv_transform" 


    main(image_dir, mask_stack_subfolder, mask_pattern)
