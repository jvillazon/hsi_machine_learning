"""
ROI Coordinate Transformation Module

Handles loading and applying affine transformations to cell coordinates,
enabling mapping from global tissue space to ROI-local coordinate space.

Supports both pixel and micron coordinate systems, with matrices in both
xy and yx formats (from SimpleITK/elastix registration files).
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class TransformationLoader:
    """Load and parse affine transformation matrices from JSON files."""
    
    def __init__(self, json_path: Union[str, Path]):
        """
        Initialize loader with a transformation JSON file.
        
        Parameters
        ----------
        json_path : str or Path
            Path to the transformation JSON file (i2r format from SimpleITK/elastix)
        """
        self.json_path = Path(json_path)
        if not self.json_path.exists():
            raise FileNotFoundError(f"Transformation JSON not found: {json_path}")
        
        with open(self.json_path, 'r') as f:
            self.data = json.load(f)
    
    def get_matrix_xy_px(self) -> np.ndarray:
        """Get affine transformation matrix in xy pixel format (3x3)."""
        return np.array(self.data['matrix_xy_px'])
    
    def get_matrix_yx_px(self) -> np.ndarray:
        """Get affine transformation matrix in yx pixel format (3x3)."""
        return np.array(self.data['matrix_yx_px'])
    
    def get_matrix_xy_um(self) -> np.ndarray:
        """Get affine transformation matrix in xy micron format (3x3)."""
        return np.array(self.data['matrix_xy_um'])
    
    def get_matrix_yx_um(self) -> np.ndarray:
        """Get affine transformation matrix in yx micron format (3x3)."""
        return np.array(self.data['matrix_yx_um'])
    
    def get_inverse_matrix_xy_px(self) -> np.ndarray:
        """Get inverse affine transformation matrix in xy pixel format."""
        return np.array(self.data['matrix_xy_px_inv'])
    
    def get_roi_bounds_px(self) -> Dict[str, float]:
        """
        Get ROI bounding box dimensions from fixed image shape.
        
        Returns
        -------
        dict
            Dictionary with 'height' and 'width' in pixels
        """
        shape = self.data['fixed_paths'][0]['image_shape']
        return {'height': shape[0], 'width': shape[1]}
    
    def get_roi_bounds_um(self) -> Dict[str, float]:
        """
        Get ROI bounding box dimensions from fixed image shape in microns.
        
        Returns
        -------
        dict
            Dictionary with 'height' and 'width' in microns
        """
        bounds_px = self.get_roi_bounds_px()
        pixel_size_um = self.data['fixed_paths'][0]['pixel_size_um']
        return {
            'height': bounds_px['height'] * pixel_size_um,
            'width': bounds_px['width'] * pixel_size_um
        }
    
    def get_pixel_sizes(self) -> Dict[str, float]:
        """
        Get pixel sizes for fixed and moving images.
        
        Returns
        -------
        dict
            Dictionary with 'fixed_um' and 'moving_um' pixel sizes
        """
        return {
            'fixed_um': self.data['fixed_paths'][0]['pixel_size_um'],
            'moving_um': self.data['moving_paths'][0]['pixel_size_um']
        }


class CoordinateTransformer:
    """Apply affine transformations to coordinate arrays."""
    
    def __init__(self, affine_matrix: np.ndarray):
        """
        Initialize with affine transformation matrix.
        
        Parameters
        ----------
        affine_matrix : np.ndarray
            3x3 affine transformation matrix (homogeneous coordinates)
        """
        if affine_matrix.shape != (3, 3):
            raise ValueError(f"Expected 3x3 matrix, got {affine_matrix.shape}")
        self.matrix = affine_matrix
    
    def transform(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply affine transformation to coordinate array.
        
        Parameters
        ----------
        coords : np.ndarray
            Nx2 array of (x, y) coordinates
        
        Returns
        -------
        np.ndarray
            Nx2 array of transformed (x, y) coordinates
        """
        if coords.shape[1] != 2:
            raise ValueError(f"Expected Nx2 array, got shape {coords.shape}")
        
        # Convert to homogeneous coordinates (add column of 1s)
        n_points = coords.shape[0]
        homogeneous = np.hstack([coords, np.ones((n_points, 1))])
        
        # Apply transformation: (Ax)' where A is 3x3 matrix, x is 3x1 vector
        # Matrix multiplication transposes for broadcasting: (3x3) @ (3xN) = (3xN)
        transformed_homogeneous = (self.matrix @ homogeneous.T).T
        
        # Extract xy coordinates (drop homogeneous coordinate)
        return transformed_homogeneous[:, :2]
    
    def transform_and_round(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply transformation and round to nearest integer pixel.
        
        Parameters
        ----------
        coords : np.ndarray
            Nx2 array of (x, y) coordinates
        
        Returns
        -------
        np.ndarray
            Nx2 array of transformed and rounded coordinates
        """
        transformed = self.transform(coords)
        return np.round(transformed).astype(np.int32)


class CellDataMapper:
    """Map cell coordinates from tissue space to ROI-local space."""
    
    def __init__(self, 
                 json_path: Union[str, Path],
                 cell_seg_df: pd.DataFrame,
                 use_microns: bool = False,
                 clip_to_roi: bool = False,
                 flip_y: bool = True):
        """
        Initialize cell data mapper.
        
        Parameters
        ----------
        json_path : str or Path
            Path to transformation JSON file
        cell_seg_df : pd.DataFrame
            Cell segmentation dataframe with columns ['x', 'y', 'cell_id', ...]
        use_microns : bool, optional
            If True, use micrometer transformation; if False, use pixel space.
            Default is False (pixel space).
        clip_to_roi : bool, optional
            If True, clip transformed coordinates to ROI bounds. Default is False.
        flip_y : bool, optional
            If True, invert y-coordinate (y_roi = height - y_roi) to correct for
            coordinate system orientation differences. Default is True.
        """
        self.loader = TransformationLoader(json_path)
        self.cell_seg_df = cell_seg_df.copy()
        self.use_microns = use_microns
        self.clip_to_roi = clip_to_roi
        self.flip_y = flip_y

        # Select appropriate transformation matrix based on coordinate system
        if use_microns:
            matrix = self.loader.get_matrix_xy_um()
        else:
            matrix = self.loader.get_matrix_xy_px()
        
        self.transformer = CoordinateTransformer(matrix)
        self.roi_bounds = self.loader.get_roi_bounds_um() if use_microns else self.loader.get_roi_bounds_px()
    
    def transform_cells(self, 
                       round_coordinates: bool = True,
                       preserve_columns: Optional[list] = None) -> pd.DataFrame:
        """
        Transform cell coordinates to ROI space.
        
        Parameters
        ----------
        round_coordinates : bool, optional
            If True, round transformed coordinates to nearest integer. Default True.
        preserve_columns : list, optional
            List of additional columns to preserve in output. If None, all columns preserved.
        
        Returns
        -------
        pd.DataFrame
            Dataframe with transformed coordinates and preserved metadata.
            New columns: 'x_roi', 'y_roi'
        """
        result_df = self.cell_seg_df.copy()
        
        # Extract coordinate array
        coords = result_df[['x', 'y']].values
        
        # Apply transformation
        if round_coordinates:
            transformed_coords = self.transformer.transform_and_round(coords)
        else:
            transformed_coords = self.transformer.transform(coords)
        
        # Add transformed coordinates
        result_df['x_roi'] = transformed_coords[:, 0]
        result_df['y_roi'] = transformed_coords[:, 1]

        max_x = int(self.roi_bounds['width'])
        max_y = int(self.roi_bounds['height'])

        self.roi_bounds = {'width': max_x, 'height': max_y} # Ensure bounds are integers for clipping
        
        # Apply y-axis inversion if specified (corrects for coordinate system orientation)
        if self.flip_y:
            result_df['y_roi'] = max_y - result_df['y_roi']
        
        
        if self.clip_to_roi:
        # Clip to ROI bounds (optional but recommended for consistency)
            result_df['x_roi_clipped'] = result_df['x_roi'].clip(0, max_x)
            result_df['y_roi_clipped'] = result_df['y_roi'].clip(0, max_y)
            
            # Add flags for cells that were clipped
            result_df['clipped'] = (
                (result_df['x_roi'] != result_df['x_roi_clipped']) |
                (result_df['y_roi'] != result_df['y_roi_clipped'])
            )
        
        return result_df
    
    def filter_cells_in_roi(self, 
                           transformed_df: pd.DataFrame,
                           strict: bool = False) -> pd.DataFrame:
        """
        Filter cells that are within ROI bounds.
        
        Parameters
        ----------
        transformed_df : pd.DataFrame
            Dataframe with x_roi, y_roi columns (output from transform_cells)
        strict : bool, optional
            If True, only include cells entirely within ROI (no clipping).
            If False, include cells with any part in ROI.
            Default is False.
        
        Returns
        -------
        pd.DataFrame
            Filtered dataframe
        """
        if strict:
            # Only cells that didn't require clipping
            return transformed_df[~transformed_df['clipped']].copy()
        else:
            # Cells with transformed coordinates within bounds
            max_x = int(self.roi_bounds['width'])
            max_y = int(self.roi_bounds['height'])
            
            return transformed_df[
                (transformed_df['x_roi_clipped'] >= 0) & 
                (transformed_df['x_roi_clipped'] < max_x) &
                (transformed_df['y_roi_clipped'] >= 0) & 
                (transformed_df['y_roi_clipped'] < max_y)
            ].copy()


class BatchROIProcessor:
    """Batch process multiple ROIs, applying coordinate transformations."""
    
    def __init__(self, 
                 sample_name: str,
                 cell_seg_csv: Union[str, Path],
                 transform_json_dir: Union[str, Path],
                 output_dir: Union[str, Path]):
        """
        Initialize batch processor.
        
        Parameters
        ----------
        sample_name : str
            Sample identifier (e.g., 'ckd', 'aki')
        cell_seg_csv : str or Path
            Path to cell segmentation CSV file (source in moving image space)
        transform_json_dir : str or Path
            Directory containing transformation JSON files
        output_dir : str or Path
            Directory where transformed CSVs will be saved
        """
        self.sample_name = sample_name
        self.cell_seg_csv = Path(cell_seg_csv)
        self.transform_json_dir = Path(transform_json_dir)
        self.output_dir = Path(output_dir)
        
        # Load cell segmentation data once
        self.cell_seg_df = pd.read_csv(self.cell_seg_csv)
        if "Unnamed: 0" in self.cell_seg_df.columns:
            self.cell_seg_df = self.cell_seg_df.drop(columns=["Unnamed: 0"])
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.processing_log = []
    
    def find_transformation_files(self) -> Dict[str, Path]:
        """
        Find all transformation JSON files matching pattern.
        
        Returns
        -------
        dict
            Dictionary mapping ROI id to transformation JSON path
        """
        pattern = f"{self.sample_name}_cell_segmentation-*_transform.i2r.json"
        roi_files = {}
        
        for json_file in self.transform_json_dir.glob(pattern):
            # Extract ROI id from filename: {sample}_cell_segmentation-{roi_id}_transform.i2r.json
            parts = json_file.stem.split('-')
            if len(parts) >= 2:
                roi_id = parts[1]  # roi1, roi2, etc.
                roi_files[roi_id] = json_file
        
        return roi_files
    
    def process_roi(self, roi_id: str, json_path: Path) -> Dict:
        """
        Process a single ROI.
        
        Parameters
        ----------
        roi_id : str
            ROI identifier (e.g., 'roi1')
        json_path : Path
            Path to transformation JSON file
        
        Returns
        -------
        dict
            Processing results with keys: roi_id, status, num_cells, output_path, error
        """
        result = {
            'roi_id': roi_id,
            'json_path': str(json_path),
            'status': 'pending',
            'num_cells_input': len(self.cell_seg_df),
            'num_cells_in_roi': 0,
            'num_cells_clipped': 0,
            'output_path': None,
            'error': None
        }
        
        try:
            # Initialize mapper
            mapper = CellDataMapper(json_path, self.cell_seg_df, use_microns=False)
            
            # Transform cells
            transformed_df = mapper.transform_cells(round_coordinates=True)
            
            # Filter to ROI
            roi_cells_df = mapper.filter_cells_in_roi(transformed_df, strict=False)
            
            # Get statistics
            result['num_cells_in_roi'] = len(roi_cells_df)
            result['num_cells_clipped'] = roi_cells_df['clipped'].sum()
            
            # Save output
            output_path = self.output_dir / f"{self.sample_name}_cells_in_{roi_id}_roi_coordinates.csv"
            roi_cells_df.to_csv(output_path, index=False)
            result['output_path'] = str(output_path)
            result['status'] = 'success'
            
            logger.info(f"Processed {roi_id}: {len(roi_cells_df)} cells in ROI, "
                       f"{roi_cells_df['clipped'].sum()} clipped")
            
        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
            logger.error(f"Error processing {roi_id}: {e}")
        
        self.processing_log.append(result)
        return result
    
    def process_all_rois(self) -> pd.DataFrame:
        """
        Process all ROIs in batch.
        
        Returns
        -------
        pd.DataFrame
            Summary dataframe of processing results
        """
        roi_files = self.find_transformation_files()
        
        if not roi_files:
            logger.warning(f"No transformation files found for sample {self.sample_name}")
            return pd.DataFrame()
        
        logger.info(f"Found {len(roi_files)} transformation files for {self.sample_name}")
        
        for roi_id, json_path in sorted(roi_files.items()):
            self.process_roi(roi_id, json_path)
        
        # Return summary as dataframe
        return pd.DataFrame(self.processing_log)
    
    def get_summary(self) -> pd.DataFrame:
        """Get processing summary."""
        return pd.DataFrame(self.processing_log)


# Backward compatibility with original notebook approach
def apply_roi_transform_from_json(json_path: Union[str, Path],
                                  cell_seg_df: pd.DataFrame,
                                  clip_to_roi: bool = False,
                                  return_roi_bounds: bool = True,
                                  flip_y: bool = True) -> pd.DataFrame:
    """
    Simple interface to transform cell coordinates using transformation JSON.
    
    Parameters
    ----------
    json_path : str or Path
        Path to transformation JSON file
    cell_seg_df : pd.DataFrame
        Cell segmentation dataframe
    clip_to_roi : bool, optional
        If True, clip coordinates to ROI bounds. Default is False.
    return_roi_bounds : bool, optional
        If True, return ROI bounds with dataframe. Default is True.
    flip_y : bool, optional
        If True, invert y-coordinate to correct for coordinate system orientation.
        Default is True.
    
    Returns
    -------
    tuple or pd.DataFrame
        If return_roi_bounds is True: (transformed_df, roi_bounds)
        If return_roi_bounds is False: transformed_df
    """
    mapper = CellDataMapper(json_path, cell_seg_df, use_microns=False, clip_to_roi=clip_to_roi, flip_y=flip_y)

    roi_bounds = mapper.roi_bounds if return_roi_bounds else None

    return mapper.transform_cells(round_coordinates=True), roi_bounds
