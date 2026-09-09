#!/usr/bin/env python3
"""
xenium_pipeline.py
==================
All-in-one Master Pipeline for Xenium Spatial Transcriptomics & GUI Integration:

Steps:
  1. Merge Boundaries (Parquet/CSV) + Cell Labels (CSV/Parquet)
  2. Generate Labeled Integer Mask TIFF (using metadata dict or auto-enumerated legend)
  3. Transform & Crop to Target ROI (using transform JSON & core.roi_coordinate_transformer),
     outputting GUI-compatible Parquet/CSV for multimodal_napari_gui.

Run any step individually or all steps end-to-end!
"""

import os
import sys
import argparse
from pathlib import Path

# Add current folder to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from merge_boundaries_and_labels import merge_boundaries_and_labels
from generate_labeled_mask import generate_labeled_mask
from transform_and_crop_rois import transform_and_crop_rois
from convert_roi_cells_to_gui_format import convert_single_file, batch_convert


def run_pipeline(
    boundaries_path: Optional[str] = None,
    labels_path: Optional[str] = None,
    output_dir: str = "./output",
    transform_json: Optional[str] = None,
    convert_input: Optional[str] = None,
    metadata_dict: Optional[str] = None,
    metadata_list: Optional[str] = None,
    sample_id: str = "1",
    roi_name: str = "",
    image_path: str = "",
    run_steps: List[str] = ["1", "2", "3"],
    export_format: str = "both",
):
    os.makedirs(output_dir, exist_ok=True)
    sample_tag = sample_id if sample_id else "sample"
    step1_prefix = os.path.join(output_dir, f"{sample_tag}")

    df_vertices = None
    df_regions = None

    # Step 1
    if "1" in run_steps:
        print("\n>>> EXECUTING STEP 1: MERGE BOUNDARIES & LABELS <<<")
        df_vertices, df_regions = merge_boundaries_and_labels(
            boundaries_path=boundaries_path,
            labels_path=labels_path,
            output_prefix=step1_prefix,
            sample_id=sample_id,
            image_name=roi_name,
            image_path=image_path,
            save_mode="both",
            export_format=export_format,
        )

    # Step 2
    if "2" in run_steps:
        print("\n>>> EXECUTING STEP 2: GENERATE LABELED MASK TIFF <<<")
        input_for_step2 = f"{step1_prefix}_merged_vertices.parquet"
        if not os.path.exists(input_for_step2):
            input_for_step2 = f"{step1_prefix}_labeled_regions.parquet"

        mask_out = os.path.join(output_dir, f"{sample_tag}_mask.tif")
        generate_labeled_mask(
            input_path=input_for_step2,
            output_path=mask_out,
            metadata_dict_arg=metadata_dict,
            metadata_list_arg=metadata_list,
            mask_type="all",
        )

    # Step 3
    if "3" in run_steps:
        if not transform_json:
            print("\n>>> SKIPPING STEP 3: No --transform-json specified. <<<")
        else:
            print("\n>>> EXECUTING STEP 3: TRANSFORM & CROP TO TARGET ROI <<<")
            input_for_step3 = f"{step1_prefix}_merged_vertices.parquet"
            if not os.path.exists(input_for_step3):
                input_for_step3 = f"{step1_prefix}_labeled_regions.parquet"

            roi_prefix = os.path.join(output_dir, f"{sample_tag}_{roi_name if roi_name else 'roi'}")
            transform_and_crop_rois(
                input_path=input_for_step3,
                transform_json=transform_json,
                output_prefix=roi_prefix,
                sample_id=sample_id,
                roi_name=roi_name,
                image_path=image_path,
                export_format=export_format,
            )

    # Conversion Step
    if convert_input or "convert" in run_steps:
        print("\n>>> EXECUTING CONVERT CELLS TO GUI FORMAT <<<")
        inp = convert_input if convert_input else (f"{step1_prefix}_merged_vertices.parquet" if os.path.exists(f"{step1_prefix}_merged_vertices.parquet") else f"{step1_prefix}_labeled_regions.parquet")
        convert_single_file(
            input_path=inp,
            output_path=os.path.join(output_dir, f"{sample_tag}_{roi_name if roi_name else 'roi'}_gui_rois"),
            sample_id=sample_id,
            roi_name=roi_name,
            image_path=image_path,
            export_format=export_format,
        )

    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETE! All outputs saved to: {output_dir}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Master Pipeline: Merge, Generate Masks, and Transform ROIs for multimodal_napari_gui."
    )
    parser.add_argument("--boundaries", "-b", default=None, help="Cell boundaries file (.parquet/.csv)")
    parser.add_argument("--labels", "-l", default=None, help="Cell labels metadata file (.csv/.parquet)")
    parser.add_argument("--convert-input", "-c", default=None, help="Directly convert existing cells parquet file to GUI format")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory for all pipeline products")
    parser.add_argument("--transform-json", "-t", help="Optional transform JSON (*.i2r.json) for Step 3")
    parser.add_argument("--metadata-dict", help="Optional JSON dict mapping labels to int IDs for Step 2")
    parser.add_argument("--metadata-list", help="Optional text file or list of labels for Step 2")
    parser.add_argument("--sample-id", default="xen65", help="Sample identifier string")
    parser.add_argument("--roi-name", default="roi1", help="ROI name identifier")
    parser.add_argument("--image-path", default="", help="Optional Image_Path for GUI table")
    parser.add_argument("--steps", default="1,2,3", help="Comma-separated steps to run: '1', '2', '3', 'convert' or '1,2,3'")
    parser.add_argument("--format", choices=["parquet", "csv", "both"], default="both", help="Export format")

    args = parser.parse_args()
    steps_to_run = [s.strip() for s in args.steps.split(",") if s.strip()]

    run_pipeline(
        boundaries_path=args.boundaries,
        labels_path=args.labels,
        output_dir=args.output_dir,
        transform_json=args.transform_json,
        convert_input=args.convert_input,
        metadata_dict=args.metadata_dict,
        metadata_list=args.metadata_list,
        sample_id=args.sample_id,
        roi_name=args.roi_name,
        image_path=args.image_path,
        run_steps=steps_to_run,
        export_format=args.format,
    )


if __name__ == "__main__":
    main()
