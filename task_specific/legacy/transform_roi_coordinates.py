"""
Batch ROI Coordinate Transformation Script

Transform cell coordinates from tissue space to ROI-local space for all samples and ROIs.
Uses SimpleITK/elastix transformation matrices stored in JSON files.

Usage:
    python transform_roi_coordinates.py

Configuration:
    - Modify SAMPLES dictionary to adjust data paths
    - Modify OUTPUT_BASE_DIR to change output location
"""

import sys
from pathlib import Path
import pandas as pd
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.roi_coordinate_transformer import BatchROIProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('roi_transform.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration for samples and data paths
SAMPLES = {
    'ckd': {
        'sample_n': '3990',
        'cell_seg_csv': r'D:\ADATA Backup\HuBMAP\HuBMAP Xenium\Xenium_data\Xen24_3990_ckd-selected\Xen24_3990-ckd_cell_segmentation.csv',
    },
    'aki': {
        'sample_n': '3782',
        'cell_seg_csv': r'D:\ADATA Backup\HuBMAP\HuBMAP Xenium\Xenium_data\Xen24_3782_aki-selected\aki_cell_segmentation.csv',
    },
    'ref': {
        'sample_n': 'KPMP_038',
        'cell_seg_csv': r'D:\ADATA Backup\HuBMAP\HuBMAP Xenium\Xenium_data\Xen24_KPMP_038_ref-selected\ref_cell_segmentation.csv',
    }
}

# Base directory containing transformation JSON files
TRANSFORM_JSON_BASE = Path(r'D:\ADATA Backup\HuBMAP\HuBMAP Xenium\Xenium_data')

# Output directory for transformed CSVs
OUTPUT_BASE_DIR = Path(r'D:\ADATA Backup\HuBMAP\ROI_Transformed_Coordinates')


def process_sample(sample_name: str, sample_config: dict) -> pd.DataFrame:
    """
    Process all ROIs for a single sample.
    
    Parameters
    ----------
    sample_name : str
        Sample identifier (ckd, aki, ref)
    sample_config : dict
        Configuration dict with sample_n and cell_seg_csv
    
    Returns
    -------
    pd.DataFrame
        Processing summary
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing sample: {sample_name}")
    logger.info(f"{'='*60}")
    
    sample_n = sample_config['sample_n']
    cell_seg_csv = sample_config['cell_seg_csv']
    
    # Construct paths
    transform_json_dir = TRANSFORM_JSON_BASE / f"Xen24_{sample_n}_{sample_name}-selected"
    output_dir = OUTPUT_BASE_DIR / sample_name
    
    # Verify paths exist
    if not Path(cell_seg_csv).exists():
        logger.error(f"Cell segmentation CSV not found: {cell_seg_csv}")
        return pd.DataFrame()
    
    if not transform_json_dir.exists():
        logger.error(f"Transform JSON directory not found: {transform_json_dir}")
        return pd.DataFrame()
    
    logger.info(f"Cell seg CSV: {cell_seg_csv}")
    logger.info(f"Transform JSON dir: {transform_json_dir}")
    logger.info(f"Output dir: {output_dir}")
    
    # Initialize processor
    processor = BatchROIProcessor(
        sample_name=sample_name,
        cell_seg_csv=cell_seg_csv,
        transform_json_dir=transform_json_dir,
        output_dir=output_dir
    )
    
    # Process all ROIs
    summary_df = processor.process_all_rois()
    
    # Log summary
    if not summary_df.empty:
        logger.info(f"\nProcessing Summary for {sample_name}:")
        logger.info(summary_df[['roi_id', 'status', 'num_cells_in_roi', 'num_cells_clipped']].to_string())
        
        # Save summary
        summary_path = output_dir / f"{sample_name}_processing_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Summary saved to: {summary_path}")
    else:
        logger.warning(f"No ROIs processed for {sample_name}")
    
    return summary_df


def main():
    """Main entry point."""
    logger.info("Starting batch ROI coordinate transformation")
    logger.info(f"Processing {len(SAMPLES)} samples: {', '.join(SAMPLES.keys())}")
    
    all_summaries = {}
    
    for sample_name, sample_config in SAMPLES.items():
        try:
            summary = process_sample(sample_name, sample_config)
            all_summaries[sample_name] = summary
        except Exception as e:
            logger.error(f"Failed to process {sample_name}: {e}", exc_info=True)
            continue
    
    # Aggregate results
    logger.info(f"\n{'='*60}")
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    
    total_rois = sum(len(df) for df in all_summaries.values() if not df.empty)
    successful_rois = sum((df['status'] == 'success').sum() for df in all_summaries.values() if not df.empty)
    
    logger.info(f"Total ROIs processed: {total_rois}")
    logger.info(f"Successful transformations: {successful_rois}")
    
    # Save consolidated summary
    consolidated = pd.concat(all_summaries.values(), ignore_index=True)
    if not consolidated.empty:
        summary_path = OUTPUT_BASE_DIR / "ALL_SAMPLES_processing_summary.csv"
        consolidated.to_csv(summary_path, index=False)
        logger.info(f"\nConsolidated summary saved to: {summary_path}")
    
    logger.info("\nLog saved to: roi_transform.log")


if __name__ == '__main__':
    main()
