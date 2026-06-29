"""
Split mask stack (.tif) into individual mask images with naming convention <prefix>-<unit_name>-mask.tif.
Mirrors the folder structure from the image directory into a new output directory (sibling to the image directory).

Usage:
    python process_codex_mask_stack.py <image_dir> <mask_stack_subfolder> <index_to_unit_name.json> <prefix>

- <image_dir>: Directory containing image folders (each with a mask stack in a subfolder)
- <mask_stack_subfolder>: Name of subfolder inside each image folder containing the mask stack .tif
- <index_to_unit_name.json>: JSON file mapping index to unit_name
- <prefix>: Prefix for output mask filenames
"""


import os
import glob
import sys
from tifffile import imread, imwrite
from tqdm import tqdm


def main(image_dir, mask_name, mask_pattern, mask_stack_subfolder, index_to_unit, prefix):
    # image_dir: path to directory containing image folders
    # mask_name: used for output directory naming
    # mask_pattern: pattern to match mask files
    # mask_stack_subfolder: subfolder inside each image folder containing the mask stack .tif
    # index_to_unit: dictionary mapping index (int or str) to unit_name
    # prefix: prefix for output mask filenames

    image_dir = os.path.abspath(image_dir)
    output_dir = os.path.join(os.path.dirname(image_dir), f"{mask_name}_masks")
    os.makedirs(output_dir, exist_ok=True)

    # For each image folder
    for image_folder_name in tqdm([os.path.basename(d) for d in glob.glob(os.path.join(image_dir, "*")) if os.path.isdir(os.path.join(image_dir, d))]):
        image_folder = os.path.join(image_dir, image_folder_name)
        mask_stack_dir = os.path.join(image_folder, mask_stack_subfolder)
        if not os.path.isdir(mask_stack_dir):
            print(f"No mask stack subfolder found: {mask_stack_dir}")
            continue
        mask_stacks = glob.glob(os.path.join(mask_stack_dir, f"{mask_pattern}"))
        if not mask_stacks:
            print(f"No mask stack found in {mask_stack_dir}")
            continue
        mask_stack_path = os.path.join(mask_stack_dir, mask_stacks[0])
        mask_stack = imread(mask_stack_path)
        print(f"Processing {mask_stack_path} with shape {mask_stack.shape}")

        # Prepare output folder
        out_img_folder = os.path.join(output_dir, image_folder_name)
        os.makedirs(out_img_folder, exist_ok=True)

        # Save each mask
        for idx, unit_name in index_to_unit.items():
            idx_int = int(idx)
            if idx_int >= mask_stack.shape[0]:
                print(f"Index {idx_int} out of range for {mask_stack_path}")
                continue
            mask = mask_stack[idx_int]
            if mask.sum() == 0:
                print(f"Mask for {unit_name} (index {idx}) is empty, skipping.")
                continue
            out_name = f"{prefix}-{unit_name}-mask.tif"
            out_path = os.path.join(out_img_folder, out_name)
            imwrite(out_path, mask)
            print(f"Saved {out_path}")

    print(f"\n✓ All masks split and saved to {output_dir}")


if __name__ == "__main__":

    # args = argparse.ArgumentParser(description="Split mask stack into individual mask images.")
    # args.add_argument("image_dir",
    #                   '-i', '--image_dir',
    #                    help="Directory containing image folders")
    # args.add_argument("mask_name",
    #                   '-m', '--mask_name',
    #                    help="Name for output mask directory")
    # args.add_argument("mask_pattern", 
    #                   '-p', '--mask_pattern',
    #                   help="Pattern to match mask stack files (e.g. 'CODEX-*-mask.tif')")
    # args.add_argument("mask_stack_subfolder", 
    #                   '-s', '--mask_stack_subfolder',
    #                   help="Subfolder inside each image folder containing the mask stack .tif")
    # args.add_argument("dictionary", 
    #                   '-d', '--dictionary',
    #                   help="dictionary mapping index to unit_name in JSON format (e.g. '{\"0\": \"Vasc\", \"1\": \"ProxTub\"}')")
    # args.add_argument("prefix", 
    #                     '-x', '--prefix',
    #                   help="Prefix for output mask filenames (e.g. 'CODEX')")



    # # Example usage:
    # # python process_codex_mask_stack.py <image_dir> <mask_name> <mask_pattern> <mask_stack_subfolder> <prefix>
    # # Edit the dictionary below as needed for your use case
    # if len(sys.argv) != 6:
    #     print("Usage: python process_codex_mask_stack.py <image_dir> <mask_name> <mask_pattern> <mask_stack_subfolder> <prefix>")
    #     sys.exit(1)
    # # Example: index_to_unit = {"0": "Vasc", "1": "ProxTub", ...}
    # index_to_unit = {
    #     "0": "Vasc",
    #     "1": "ProxTub",
    #     "2": "Glom",
    #     "3": "TAL",
    #     "4": "ThAL",
    #     "5": "DistNeph",
    #     "6": "DistTub"
    # }

    # args['dictionary'] = str(index_to_unit)  # Convert dict to string for command line argument

    # args = args.parse_args()
    # main(args.image_dir, args.mask_name, args.mask_pattern, args.mask_stack_subfolder, eval(args.dictionary), args.prefix)
    # print("\n✓ All masks split and saved successfully.")

    image_dir = r"D:\integrated_pipeline\registered_data\CODEX\Clusters"
    mask_name = "CODEX_clusters"
    mask_pattern = "*Clusters-transformed.tif"
    mask_stack_subfolder = "inv_transform"
    
    # Xenium AKI
    # index_to_unit = {
    #     "0": "altPT",
    #     "1": "altDTL",
    #     "2": "PT",
    #     "3": "DTL",
    #     "4": "EC",
    #     "5": "TAL",
    #     "6": "unassigned",
    #     "7": "altTAL",
    #     "8": "low_quality",
    #     "9": "PC",
    #     "10": "IC",
    #     "11": "POD",
    #     "12": "PEC",
    #     "13": "Lymphoid",
    #     "14": "Myeloid",
    #     "15": "T",
    #     "16": "ATL",
    #     "17": "DCT",
    #     "18": "altATL",
    #     "19": "VSM,P",
    #     "20": "FIB",
    #     "21": "CNT",
    #     "22": "altCNT",
    #     "23": "altIC",
    #     "24": "MON",
    #     "25": "DC",
    #     "26": "infFIB",
    #     "27": "SC,NEU",
    #     "28": "moMAC",
    #     "29": "altDCT",
    #     "30": "MYOF",
    #     "31": "pvFIB",
    #     "32": "resMAC",
    #     "33": "aFIB",
    #     "34": "moMAC,INF",
    #     "35": "MD",
    #     "36": "NK",
    #     "37": "N",
    #     "38": "MAST",
    #     "39": "REN",
    #     "40": "EC,GC",
    #     "41": "NEU",
    #     "42": "altPOD",
    #     "43": "B",
    #     "44": "MC"
    # }




    # Xenium CKD
    # index_to_unit = {
    #     "0": "PC",
    #     "1": "altIC",
    #     "2": "T",
    #     "3": "altTAL",
    #     "4": "low_quality",
    #     "5": "altDTL",
    #     "6": "altATL",
    #     "7": "FIB",
    #     "8": "IC",
    #     "9": "TAL",
    #     "10": "NK",
    #     "11": "DTL",
    #     "12": "ATL",
    #     "13": "PEC",
    #     "14": "unassigned",
    #     "15": "EC",
    #     "16": "DC",
    #     "17": "MON",
    #     "18": "altPT",
    #     "19": "N",
    #     "20": "Lymphoid",
    #     "21": "MD",
    #     "22": "VSM,P",
    #     "23": "Myeloid",
    #     "24": "B",
    #     "25": "POD",
    #     "26": "PT",
    #     "27": "EC,GC",
    #     "28": "altDCT",
    #     "29": "altCNT",
    #     "30": "DCT",
    #     "31": "resMAC",
    #     "32": "CNT",
    #     "33": "PL",
    #     "34": "MAST",
    #     "35": "infFIB",
    #     "36": "MYOF",
    #     "37": "moMAC,INF",
    #     "38": "aFIB",
    #     "39": "altPOD",
    #     "40": "NEU",
    #     "41": "MC",
    #     "42": "pvFIB",
    #     "43": "moMAC",
    #     "44": "SC,NEU",
    #     "45": "REN"
    # }


    # # Xenium Ref
    # index_to_unit = {
    #     "0": "TAL",
    #     "1": "altTAL",
    #     "2": "T",
    #     "3": "DTL",
    #     "4": "PC",
    #     "5": "FIB",
    #     "6": "altDTL",
    #     "7": "IC",
    #     "8": "altATL",
    #     "9": "low_quality",
    #     "10": "altIC",
    #     "11": "Lymphoid",
    #     "12": "Myeloid",
    #     "13": "EC",
    #     "14": "PEC",
    #     "15": "aFIB",
    #     "16": "altDCT",
    #     "17": "resMAC",
    #     "18": "DCT",
    #     "19": "ATL",
    #     "20": "altPT",
    #     "21": "PT",
    #     "22": "moMAC",
    #     "23": "MON",
    #     "24": "DC",
    #     "25": "POD",
    #     "26": "pvFIB",
    #     "27": "unassigned",
    #     "28": "EC,GC",
    #     "29": "MD",
    #     "30": "NK",
    #     "31": "moMAC,INF",
    #     "32": "MYOF",
    #     "33": "altCNT",
    #     "34": "PL",
    #     "35": "CNT",
    #     "36": "B",
    #     "37": "infFIB",
    #     "38": "N",
    #     "39": "MAST",
    #     "40": "VSM,P",
    #     "41": "MC",
    #     "42": "NEU",
    #     "43": "altPOD",
    #     "44": "REN"
    # }


    # CODEX clusters
    index_to_unit = {
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


    prefix = "CODEX"

    main(image_dir, mask_name, mask_pattern, mask_stack_subfolder, index_to_unit, prefix)
    print("\n✓ All masks split and saved successfully.")   
