"""
Stack and reindex .tif images from a folder, rotate the stack 90° counterclockwise, and save as a single .tif stack.

- Loads all .tif images from a folder.
- Uses a dictionary to map new stack indices to filenames (key: new index, value: filename).
- Stacks images in the order of the dictionary keys.
- Rotates the stack 90° counterclockwise.
- Saves the result as a .tif stack.

Usage:
    python stack_and_rotate_masks.py <input_folder> <output_path>

Edit the index_to_file dictionary in the script as needed.
"""

import os
import sys
import numpy as np
from tifffile import imread, imwrite


def main(input_folder, output_path, index_to_file):
    # input_folder: folder with .tif images
    # output_path: path to save the stacked and rotated .tif
    # index_to_file: dict mapping new index to filename (relative to input_folder)

    # Load images in the order of sorted keys
    stack = []
    for idx in sorted(index_to_file, key=lambda x: int(x)):
        fname = index_to_file[idx]
        img_path = os.path.join(input_folder, f"{fname}")
        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")
            continue
        img = imread(img_path)
        if img.ndim != 2:  # If single channel, add channel dimension
            print(f"Warning: {img_path} is not 2D. Skipping.")
            continue
        stack.append(img)
        print(f"Loaded {img_path}")

    if not stack:
        print("No images loaded. Exiting.")
        return

    stack_arr = np.stack(stack, axis=0)
    print(f"Stacked array shape: {stack_arr.shape}")

    # Rotate stack 90° counterclockwise (axis 1 and 2)
    stack_rot = np.rot90(stack_arr, k=1, axes=(1, 2)).astype(np.uint8)
    print(f"Rotated stack shape: {stack_rot.shape}")

    # Save the rotated stack as a .tif
    imwrite(output_path, stack_rot)
    print(f"Saved stacked and rotated image to {output_path}")

    # Save max projection for quick visualization
    max_proj = np.max(stack_rot, axis=0)
    max_proj_path = os.path.splitext(output_path)[0] + "_max_projection.tif"
    imwrite(max_proj_path, max_proj)
    print(f"Saved max projection to {max_proj_path}")


if __name__ == "__main__":
    # Example usage:
    # python stack_and_rotate_masks.py <input_folder> <output_path>
    # Edit the dictionary below as needed for your use case
    # if len(sys.argv) != 3:
    #     print("Usage: python stack_and_rotate_masks.py <input_folder> <output_path>")
    #     sys.exit(1)
    # # Example: index_to_file = {"0": "CODEX-Cluster1-mask.tif", "1": "CODEX-Cluster2-mask.tif", ...}

    input_folder = r"C:\Users\jvill\Downloads\Clusters\Clusters"
    output_path = r"C:\Users\jvill\Downloads\Clusters\CODEX_clusters_stack.tif"



    index_to_file = {
        "0": "Export_Cluster20.tif",
        "1": "Export_Cluster15.tif",
        "2": "Export_Cluster19.tif",
        "3": "Export_Cluster16.tif",
        "4": "Export_Cluster9.tif",
        "5": "Export_Cluster18.tif",
        "6": "Export_Cluster26.tif",
        "7": "Export_Cluster1.tif",
        "8": "Export_Cluster0.tif",
        "9": "Export_Cluster23.tif",
        "10": "Export_Cluster25.tif",
        "11": "Export_Cluster24.tif",
        "12": "Export_Cluster2.tif",
        "13": "Export_Cluster3.tif",
        "14": "Export_Cluster12.tif",
        "15": "Export_Cluster21.tif",
        "16": "Export_Cluster17.tif",
        "17": "Export_Cluster4.tif",
        "18": "Export_Cluster10.tif",
        "19": "Export_Cluster5.tif",
        "20": "Export_Cluster14.tif",
        "21": "Export_Cluster22.tif",
        "22": "Export_Cluster8.tif",
        "23": "Export_Cluster11.tif",
        "24": "Export_Cluster13.tif",
        "25": "Export_Cluster6.tif",
        "26": "Export_Cluster7.tif",
    }
    main(input_folder=input_folder, output_path=output_path, index_to_file=index_to_file)

    print("\n✓ All masks stacked, rotated, and saved successfully.")
