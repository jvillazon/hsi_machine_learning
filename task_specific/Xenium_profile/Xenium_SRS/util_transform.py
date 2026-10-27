"""
Image(ROI) Coordinate Transformation Module

Loads and applies affine transformations to cell coordinates,
enabling mapping from image space to coordinate space.

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
    
    def get_moving_bounds_px(self) -> Dict[str, float]:
        """
        Get moving image bounding box dimensions from moving image shape.
        
        Returns
        -------
        dict
            Dictionary with 'height' and 'width' in pixels
        """
        shape = self.data['moving_paths'][0]['image_shape']
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
                 flip_y: bool = True,
                 use_inverse: bool = False):
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
        use_inverse : bool, optional
            If True, apply the inverse transformation matrix and use moving image bounds.
            Default is False.
        """
        self.loader = TransformationLoader(json_path)
        self.cell_seg_df = cell_seg_df.copy()
        self.use_microns = use_microns
        self.clip_to_roi = clip_to_roi
        self.flip_y = flip_y
        self.use_inverse = use_inverse

        # Select appropriate transformation matrix based on coordinate system
        if use_inverse:
            matrix = self.loader.get_inverse_matrix_xy_px()
        elif use_microns:
            matrix = self.loader.get_matrix_xy_um()
        else:
            matrix = self.loader.get_matrix_xy_px()
        
        self.transformer = CoordinateTransformer(matrix)
        
        if use_inverse:
            self.roi_bounds = self.loader.get_moving_bounds_px()
        else:
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

# Function
def apply_roi_transform_from_json(json_path: Union[str, Path],
                                  cell_seg_df: pd.DataFrame,
                                  clip_to_roi: bool = False,
                                  return_roi_bounds: bool = True,
                                  flip_y: bool = False,
                                  use_inverse: bool = False) -> pd.DataFrame:
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
    use_inverse : bool, optional
        If True, apply the inverse transformation matrix and use moving image bounds.
        Default is False.
    
    Returns
    -------
    tuple or pd.DataFrame
        If return_roi_bounds is True: (transformed_df, roi_bounds)
        If return_roi_bounds is False: transformed_df
    """
    mapper = CellDataMapper(json_path, cell_seg_df, use_microns=False, clip_to_roi=clip_to_roi, flip_y=flip_y, use_inverse=use_inverse)

    roi_bounds = mapper.roi_bounds if return_roi_bounds else None

    return mapper.transform_cells(round_coordinates=True), roi_bounds
