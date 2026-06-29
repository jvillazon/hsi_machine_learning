"""
Image masking pipeline for large HSI images with stack-based masks.

Supports 2D and 3D query images with corresponding mask stacks where each 
slice is an 8-bit mask. Applies processing functions to masked regions and 
saves results to CSV.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MaskSliceResult:
    """Container for results from a single mask slice."""
    slice_idx: int
    mask_id: int
    annotations: Dict[str, Any]
    results: Dict[str, float]


class ImageMaskProcessor:
    """
    Process large images with multi-slice mask stacks.
    
    Applies processing functions to each masked region and aggregates results.
    """
    
    def __init__(self, query_image: np.ndarray, mask_stack: np.ndarray):
        """
        Initialize processor with query and mask images.
        
        Parameters
        ----------
        query_image : np.ndarray
            2D or 3D query image to be masked (e.g., grayscale HSI intensity).
            Shape: (H, W) for 2D or (D, H, W) for 3D.
        mask_stack : np.ndarray
            Stack of 8-bit masks, same spatial dims as query_image.
            Shape must match query_image. Each slice contains a mask 
            (can be different masks on same slice or same mask across slices).
        """
        self.query_image = query_image
        self.mask_stack = mask_stack
        
        # Validate shapes
        if query_image.shape[1:] != mask_stack.shape[1:]:
            raise ValueError(
                f"Query image shape {query_image.shape} must match "
                f"mask stack shape {mask_stack.shape}"
            )
        
        self.is_3d = query_image.ndim == 3
        self.results: List[MaskSliceResult] = []
        
    def get_unique_masks_per_slice(self) -> Dict[int, np.ndarray]:
        """
        Get unique non-zero mask values for each slice.
        
        Returns
        -------
        Dict[int, np.ndarray]
            Dictionary mapping slice index to array of unique mask values.
        """
        masks_per_slice = {}
        
        if self.is_3d:
            for i in range(self.mask_stack.shape[0]):
                unique = np.unique(self.mask_stack[i])
                # Exclude background (0)
                unique = unique[unique > 0]
                masks_per_slice[i] = unique
        else:
            unique = np.unique(self.mask_stack)
            unique = unique[unique > 0]
            masks_per_slice[0] = unique
            
        return masks_per_slice
    
    def process_slice(
        self,
        slice_idx: int,
        mask_value: int,
        processing_fn: Callable[[np.ndarray], Dict[str, float]],
        annotation: Optional[Dict[str, Any]] = None,
    ) -> MaskSliceResult:
        """
        Process a single mask instance in a slice.
        
        Parameters
        ----------
        slice_idx : int
            Index of the slice (for 2D images, always 0).
        mask_value : int
            The specific mask value to process (label in the mask).
        processing_fn : Callable
            Function that takes masked image and returns dict of results.
            Signature: fn(masked_image: np.ndarray) -> Dict[str, float]
        annotation : Dict[str, Any], optional
            Dictionary of annotations for this mask instance.
            
        Returns
        -------
        MaskSliceResult
            Results container with slice info and processing results.
        """
        # Extract the slice
        if self.is_3d:
            query_slice = self.query_image[slice_idx]
            mask_slice = self.mask_stack[slice_idx]
        else:
            query_slice = self.query_image
            mask_slice = self.mask_stack
        
        # Create binary mask for this specific mask value
        binary_mask = (mask_slice > 0).astype(bool)
        
        # Extract masked image
        masked_image = query_slice[binary_mask]
        
        # Apply processing function
        results = processing_fn(masked_image)
        
        # Create result container
        result = MaskSliceResult(
            slice_idx=slice_idx,
            mask_id=mask_value,
            annotations=annotation or {},
            results=results,
        )
        
        return result
    
    def process_all_masks(
        self,
        processing_fn: Callable[[np.ndarray], Dict[str, float]],
        slice_annotations: Optional[Dict[int, Dict[int, Dict[str, Any]]]] = None,
        verbose: bool = True,
    ) -> List[MaskSliceResult]:
        """
        Process all masks in all slices.
        
        Parameters
        ----------
        processing_fn : Callable
            Function to apply to each masked region.
        slice_annotations : Dict[int, Dict[int, Dict[str, Any]]], optional
            Nested dict mapping slice_idx -> mask_id -> annotations dict.
            Example: {0: {1: {'name': 'cell_A'}, 2: {'name': 'cell_B'}}}
        verbose : bool
            If True, log progress information.
            
        Returns
        -------
        List[MaskSliceResult]
            List of results from each mask instance.
        """
        self.results = []
        slice_annotations = slice_annotations or {}
        
        masks_per_slice = self.get_unique_masks_per_slice()
        total_masks = sum(len(m) for m in masks_per_slice.values())
        processed = 0
        
        for slice_idx, mask_values in masks_per_slice.items():
            for mask_value in mask_values:
                # Get annotation if available
                annotation = None
                if slice_idx in slice_annotations:
                    annotation = slice_annotations[slice_idx].get(mask_value, {})
                
                # Process this mask
                result = self.process_slice(
                    slice_idx, mask_value, processing_fn, annotation
                )
                self.results.append(result)
                processed += 1
                
                if verbose and processed % 100 == 0:
                    logger.info(f"Processed {processed}/{total_masks} masks")
        
        if verbose:
            logger.info(f"Processing complete. {total_masks} masks processed.")
        
        return self.results
    
    def results_to_dataframe(self) -> pd.DataFrame:
        """
        Convert results to a pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: slice_idx, mask_id, all annotations,
            and all processing results.
        """
        if not self.results:
            raise ValueError("No results to convert. Run process_all_masks() first.")
        
        rows = []
        for result in self.results:
            row = {
                'slice_idx': result.slice_idx,
                'mask_id': result.mask_id,
                **result.annotations,
                **result.results,
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def save_results(self, output_path: Union[str, Path]) -> None:
        """
        Save results to CSV file.
        
        Parameters
        ----------
        output_path : str or Path
            Path where CSV should be saved.
        """
        df = self.results_to_dataframe()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")


# ============================================================================
# Built-in processing functions
# ============================================================================

def mean_intensity(masked_image: np.ndarray) -> Dict[str, float]:
    """Calculate mean intensity."""
    return {'mean_intensity': float(np.mean(masked_image))}


def intensity_stats(masked_image: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive intensity statistics."""
    return {
        'mean_intensity': float(np.mean(masked_image)),
        'std_intensity': float(np.std(masked_image)),
        'min_intensity': float(np.min(masked_image)),
        'max_intensity': float(np.max(masked_image)),
        'median_intensity': float(np.median(masked_image)),
    }


def region_properties(masked_image: np.ndarray) -> Dict[str, float]:
    """Calculate region and intensity properties."""
    return {
        'pixel_count': float(len(masked_image)),
        'mean_intensity': float(np.mean(masked_image)),
        'total_intensity': float(np.sum(masked_image)),
        'std_intensity': float(np.std(masked_image)),
    }


# ============================================================================
# Convenience function
# ============================================================================

def process_image_with_mask_stack(
    query_image: np.ndarray,
    mask_stack: np.ndarray,
    processing_fn: Callable[[np.ndarray], Dict[str, float]],
    slice_annotations: Optional[Dict[int, Dict[int, Dict[str, Any]]]] = None,
    output_csv: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Convenience function to process image with mask stack in one call.
    
    Parameters
    ----------
    query_image : np.ndarray
        2D or 3D query image.
    mask_stack : np.ndarray
        Stack of 8-bit masks (same shape as query_image).
    processing_fn : Callable
        Function to apply to each masked region.
    slice_annotations : Dict, optional
        Annotations for each mask instance: {slice_idx: {mask_id: {...}}}
    output_csv : str or Path, optional
        If provided, save results to this CSV file.
    verbose : bool
        If True, log progress.
        
    Returns
    -------
    pd.DataFrame
        Results DataFrame.
        
    Examples
    --------
    >>> import numpy as np
    >>> # Create sample data
    >>> query = np.random.rand(512, 512).astype(np.uint8)
    >>> masks = np.random.randint(0, 5, (512, 512)).astype(np.uint8)
    >>> 
    >>> # Process with mean intensity
    >>> df = process_image_with_mask_stack(
    ...     query, masks, mean_intensity, 
    ...     output_csv='results.csv'
    ... )
    >>> print(df)
    
    >>> # Process with annotations
    >>> annotations = {
    ...     0: {
    ...         1: {'cell_type': 'neuron'},
    ...         2: {'cell_type': 'glia'},
    ...     }
    ... }
    >>> df = process_image_with_mask_stack(
    ...     query, masks, intensity_stats,
    ...     slice_annotations=annotations,
    ...     output_csv='results.csv'
    ... )
    """
    processor = ImageMaskProcessor(query_image, mask_stack)
    processor.process_all_masks(
        processing_fn,
        slice_annotations=slice_annotations,
        verbose=verbose,
    )
    df = processor.results_to_dataframe()
    
    if output_csv:
        processor.save_results(output_csv)
    
    return df


if __name__ == '__main__':
    # Example: Create synthetic data and process
    logger.info("Running example...")
    
    # Create 2D synthetic data
    query_2d = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    masks_2d = np.random.randint(0, 5, (512, 512), dtype=np.uint8)
    
    # Process with annotations
    annotations = {
        0: {
            1: {'label': 'region_A', 'batch': 1},
            2: {'label': 'region_B', 'batch': 1},
            3: {'label': 'region_C', 'batch': 1},
        }
    }
    
    df = process_image_with_mask_stack(
        query_2d,
        masks_2d,
        intensity_stats,
        slice_annotations=annotations,
        verbose=True,
    )
    
    print("\nResults:")
    print(df)
    
    # Example: 3D data
    logger.info("\nProcessing 3D data...")
    query_3d = np.random.randint(0, 256, (10, 512, 512), dtype=np.uint8)
    masks_3d = np.random.randint(0, 5, (10, 512, 512), dtype=np.uint8)
    
    # Per-slice annotations
    annotations_3d = {
        slice_idx: {
            mask_id: {'slice': slice_idx, 'mask': mask_id}
            for mask_id in range(1, 5)
        }
        for slice_idx in range(10)
    }
    
    df_3d = process_image_with_mask_stack(
        query_3d,
        masks_3d,
        region_properties,
        slice_annotations=annotations_3d,
        verbose=True,
    )
    
    print("\n3D Results (first 10 rows):")
    print(df_3d.head(10))
