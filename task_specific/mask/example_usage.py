"""
Example usage of the ImageMaskProcessor for various HSI analysis scenarios.
"""

import numpy as np
from pathlib import Path
from stack_mask import (
    ImageMaskProcessor,
    process_image_with_mask_stack,
    mean_intensity,
    intensity_stats,
    region_properties,
)


# ============================================================================
# EXAMPLE 1: Basic 2D image with mean intensity
# ============================================================================

def example_1_basic_2d():
    """Simple 2D example with mean intensity calculation."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic 2D Processing")
    print("="*60)
    
    # Load your actual images (here using synthetic data)
    query_image = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    mask_stack = np.random.randint(0, 5, (512, 512), dtype=np.uint8)
    
    # Process with mean intensity
    df = process_image_with_mask_stack(
        query_image,
        mask_stack,
        mean_intensity,
        output_csv='results/example1_mean_intensity.csv',
    )
    
    print(df)


# ============================================================================
# EXAMPLE 2: 3D stack with per-slice annotations
# ============================================================================

def example_2_3d_with_annotations():
    """3D image processing with custom annotations per mask."""
    print("\n" + "="*60)
    print("EXAMPLE 2: 3D with Annotations")
    print("="*60)
    
    # 3D image and mask
    depths, height, width = 10, 256, 256
    query_image = np.random.randint(0, 256, (depths, height, width), dtype=np.uint8)
    mask_stack = np.random.randint(0, 5, (depths, height, width), dtype=np.uint8)
    
    # Create per-slice annotations
    # Structure: {slice_idx: {mask_id: {annotation_key: annotation_value, ...}}}
    annotations = {
        slice_idx: {
            mask_id: {
                'cell_type': f'type_{mask_id}',
                'batch': 'batch_A',
                'z_position': slice_idx,
                'condition': 'control',
            }
            for mask_id in range(1, 5)
        }
        for slice_idx in range(depths)
    }
    
    df = process_image_with_mask_stack(
        query_image,
        mask_stack,
        intensity_stats,
        slice_annotations=annotations,
        output_csv='results/example2_3d_annotated.csv',
    )
    
    print(df.head(10))
    print(f"\nTotal results: {len(df)} mask instances")


# ============================================================================
# EXAMPLE 3: Custom processing function
# ============================================================================

def example_3_custom_function():
    """Define and use a custom processing function."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Custom Processing Function")
    print("="*60)
    
    # Your custom processing function
    def custom_analysis(masked_image: np.ndarray) -> dict:
        """
        Custom analysis: calculate multiple metrics on masked region.
        
        Parameters
        ----------
        masked_image : np.ndarray
            1D array of pixel values from the masked region
            
        Returns
        -------
        dict
            Dictionary with calculated metrics
        """
        return {
            'pixel_count': len(masked_image),
            'mean': float(np.mean(masked_image)),
            'variance': float(np.var(masked_image)),
            'percentile_95': float(np.percentile(masked_image, 95)),
            'percentile_05': float(np.percentile(masked_image, 5)),
            'range': float(np.max(masked_image) - np.min(masked_image)),
            'sum': float(np.sum(masked_image)),
        }
    
    query_image = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    mask_stack = np.random.randint(0, 4, (512, 512), dtype=np.uint8)
    
    df = process_image_with_mask_stack(
        query_image,
        mask_stack,
        custom_analysis,
        output_csv='results/example3_custom.csv',
    )
    
    print(df)


# ============================================================================
# EXAMPLE 4: Advanced - Use ImageMaskProcessor class directly
# ============================================================================

def example_4_advanced_direct_class():
    """Advanced usage directly with ImageMaskProcessor class."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Advanced Direct Class Usage")
    print("="*60)
    
    query_image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    mask_stack = np.random.randint(0, 5, (256, 256), dtype=np.uint8)
    
    # Initialize processor
    processor = ImageMaskProcessor(query_image, mask_stack)
    
    # Get unique masks
    masks_per_slice = processor.get_unique_masks_per_slice()
    print(f"Masks per slice: {masks_per_slice}")
    
    # Process specific masks manually if needed
    result = processor.process_slice(
        slice_idx=0,
        mask_value=1,
        processing_fn=intensity_stats,
        annotation={'cell_id': 'C001', 'status': 'valid'}
    )
    print(f"\nProcessed slice 0, mask 1:")
    print(f"  Results: {result.results}")
    print(f"  Annotations: {result.annotations}")
    
    # Process all masks
    processor.process_all_masks(intensity_stats)
    
    # Convert to DataFrame and save
    df = processor.results_to_dataframe()
    processor.save_results('results/example4_advanced.csv')
    
    print(f"\nTotal masks processed: {len(df)}")


# ============================================================================
# EXAMPLE 5: Batch processing - Mix of 2D and 3D
# ============================================================================

def example_5_batch_processing():
    """Process multiple images in batch."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Batch Processing")
    print("="*60)
    
    results_list = []
    
    # Process multiple experiments
    for exp_id in range(3):
        depth = 10 if exp_id > 0 else 1
        shape = (depth, 256, 256) if depth > 1 else (256, 256)
        
        query = np.random.randint(0, 256, shape, dtype=np.uint8)
        masks = np.random.randint(0, 4, shape, dtype=np.uint8)
        
        # Annotations include experiment ID
        annotations = {
            slice_idx: {
                mask_id: {'exp_id': exp_id, 'replicate': mask_id}
                for mask_id in range(1, 4)
            }
            for slice_idx in range(depth)
        }
        
        df = process_image_with_mask_stack(
            query, masks, intensity_stats,
            slice_annotations=annotations,
            verbose=False,
        )
        df['experiment_id'] = exp_id
        results_list.append(df)
    
    # Combine all results
    combined_df = __import__('pandas').concat(results_list, ignore_index=True)
    
    # Save combined results
    combined_df.to_csv('results/example5_batch_combined.csv', index=False)
    
    print(f"\nProcessed {len(results_list)} experiments")
    print(f"Total mask instances: {len(combined_df)}")
    print("\nFirst few rows:")
    print(combined_df.head())


# ============================================================================
# EXAMPLE 6: Real-world scenario - Load from files
# ============================================================================

def example_6_load_from_files():
    """Template for loading actual image files."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Load from Files (Template)")
    print("="*60)
    
    # This is a template - modify paths as needed
    print("""
    # Load query image (e.g., grayscale HSI intensity)
    from skimage import io
    query_image = io.imread('path/to/query_image.tif')
    
    # Load mask stack (uint8, each slice is a different mask)
    import tifffile
    mask_stack = tifffile.imread('path/to/mask_stack.tif')
    
    # Define annotations
    annotations = {
        0: {
            1: {'region': 'A', 'condition': 'treated'},
            2: {'region': 'B', 'condition': 'control'},
        }
    }
    
    # Process
    df = process_image_with_mask_stack(
        query_image,
        mask_stack,
        intensity_stats,
        slice_annotations=annotations,
        output_csv='results/processed_results.csv',
    )
    
    print(df)
    """)


# ============================================================================
# EXAMPLE 7: Handling different file formats
# ============================================================================

def example_7_file_format_tips():
    """Tips for loading different image formats."""
    print("\n" + "="*60)
    print("EXAMPLE 7: File Format Tips")
    print("="*60)
    
    print("""
    # TIFF (single slice or stack)
    from skimage import io
    image = io.imread('image.tiff')  # Automatically handles stacks
    
    # Multi-slice TIFF
    import tifffile
    image = tifffile.imread('stack.tif')  # Returns (D, H, W) for stacks
    
    # NPZ (NumPy)
    data = np.load('image.npz')
    query_image = data['arr_0']  # Access individual arrays
    mask_stack = data['arr_1']
    
    # PNG (single 2D images only)
    from PIL import Image
    image = np.array(Image.open('image.png'))
    
    # HDF5
    import h5py
    with h5py.File('data.h5', 'r') as f:
        query_image = f['query'][:]
        mask_stack = f['masks'][:]
    """)


# ============================================================================
# Run all examples
# ============================================================================

if __name__ == '__main__':
    # Create output directory
    Path('results').mkdir(exist_ok=True)
    
    example_1_basic_2d()
    example_2_3d_with_annotations()
    example_3_custom_function()
    example_4_advanced_direct_class()
    example_5_batch_processing()
    example_6_load_from_files()
    example_7_file_format_tips()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("Check 'results/' directory for CSV outputs")
    print("="*60)
