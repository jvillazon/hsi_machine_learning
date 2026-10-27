"""
core/roi_transform.py
=====================
Unified master ROI transformation and coordinate mapping module.
Used across hsi_machine_learning and multimodal_gui pipelines.

Features:
- Robust JSON parsing matching image2image registration schema (.i2r.json)
  with support for matrix_xy_px, matrix_yx_px (converted via coordinate swap),
  generic matrix/affine_matrix, parameter dictionaries, and inverted matrices.
- Exact homogeneous 2D projection with projective perspective divide (w):
  [x', y', w]^T = M @ [x, y, 1]^T -> x_cart = x'/w, y_cart = y'/w.
- Standardized direction control: `use_inverse: bool = False` (with `reverse` alias).
- Configurable out-of-canvas polygon cropping (`crop_mode`):
  - 'none': Preserve full polygon geometry without modification or dropping (ideal for multi-step chained transforms).
  - 'intersect': Retain full, unclipped polygon geometry if at least one vertex is inside target canvas bounds.
  - 'clip': Sutherland-Hodgman convex box clipping strictly against [0, W] x [0, H] (prevents display artifacts).
  - 'strict': Retain polygon only if 100% of its vertices are strictly inside target canvas bounds.
- Tile mosaic coordinate composition:
  M_tile = T(-x_tgt, -y_tgt) @ M_canvas @ T(x_tile, y_tile)
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd


class TransformInfo(tuple):
    """Tuple subclass holding (matrix_3x3, target_shape) with extra metadata attributes."""
    def __new__(cls, matrix: np.ndarray, target_shape: Optional[Tuple[int, int]], meta: Optional[Dict[str, Any]] = None):
        obj = super().__new__(cls, (matrix, target_shape))
        obj.matrix = matrix
        obj.target_shape = target_shape
        obj.meta = meta or {}
        return obj


def load_i2r_transform(
    json_path: Union[str, Path],
    use_inverse: bool = False,
    reverse: Optional[bool] = None
) -> TransformInfo:
    """
    Parse an .i2r.json or generic affine/projective transformation JSON file.

    Robustly handles:
    - Standard .i2r.json format: 'matrix_xy_px' and 'matrix_xy_px_inv'
    - 'matrix_yx_px' and 'matrix_yx_px_inv' (converted to XY convention via S @ M @ S)
    - Micron matrices ('matrix_xy_um', 'matrix_yx_um') and pixel resolutions
    - 'fixed_paths' and 'moving_paths' with 'image_shape' and 'pixel_size_um'
    - Generic 'matrix', 'transform_matrix', 'affine_matrix', or 'tform' dictionaries
    - Automatic inverse calculation via np.linalg.inv if inverse requested but not pre-saved.

    Args:
        json_path: Path to transform JSON file.
        use_inverse: If True, compute and return the inverse transform matrix (M^-1, fixed -> moving).
        reverse: Backward-compatible alias for use_inverse.

    Returns:
        TransformInfo tuple: (matrix_3x3, target_shape) with .meta attribute.
    """
    if reverse is not None:
        use_inverse = reverse

    path_obj = Path(json_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Transform JSON file not found: {json_path}")

    with open(path_obj, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta: Dict[str, Any] = {
        "path": str(json_path),
        "schema_version": data.get("schema_version") if isinstance(data, dict) else None,
        "transformation_type": data.get("transformation_type", "affine") if isinstance(data, dict) else "affine",
        "fixed_paths": data.get("fixed_paths", []) if isinstance(data, dict) else [],
        "moving_paths": data.get("moving_paths", []) if isinstance(data, dict) else [],
    }

    # Extract pixel sizes if available
    fixed_um = 1.0
    if meta["fixed_paths"] and isinstance(meta["fixed_paths"][0], dict):
        fixed_um = float(meta["fixed_paths"][0].get("pixel_size_um", 1.0))
    moving_um = 1.0
    if meta["moving_paths"] and isinstance(meta["moving_paths"][0], dict):
        moving_um = float(meta["moving_paths"][0].get("pixel_size_um", 1.0))
    meta["pixel_size_um"] = {"fixed_um": fixed_um, "moving_um": moving_um}

    # 1. Extract Canvas / Image Shapes
    fixed_shape = None
    if meta["fixed_paths"] and isinstance(meta["fixed_paths"][0], dict):
        sh = meta["fixed_paths"][0].get("image_shape")
        if sh:
            fixed_shape = tuple(sh[:2])
    if fixed_shape is None and isinstance(data, dict):
        if "fixed_shape" in data and data["fixed_shape"]:
            fixed_shape = tuple(data["fixed_shape"][:2])
        elif "target_shape" in data and data["target_shape"]:
            fixed_shape = tuple(data["target_shape"][:2])
        elif "roi_bounds" in data:
            fixed_shape = (int(data["roi_bounds"]["height"]), int(data["roi_bounds"]["width"]))
        elif "shape" in data and data["shape"]:
            fixed_shape = tuple(data["shape"][:2])

    moving_shape = None
    if meta["moving_paths"] and isinstance(meta["moving_paths"][0], dict):
        sh = meta["moving_paths"][0].get("image_shape")
        if sh:
            moving_shape = tuple(sh[:2])
    if moving_shape is None and isinstance(data, dict):
        if "moving_shape" in data and data["moving_shape"]:
            moving_shape = tuple(data["moving_shape"][:2])
        elif "moving_bounds" in data:
            moving_shape = (int(data["moving_bounds"]["height"]), int(data["moving_bounds"]["width"]))

    meta["fixed_shape"] = fixed_shape
    meta["moving_shape"] = moving_shape

    # By convention: forward maps moving -> fixed. Inverse maps fixed -> moving.
    target_shape = moving_shape if use_inverse else fixed_shape

    # 2. Extract Affine / Projective Matrix
    mat = None
    if not isinstance(data, dict):
        if isinstance(data, list):
            mat = np.array(data, dtype=np.float64)
            if use_inverse:
                mat = np.linalg.inv(mat)
        else:
            raise ValueError(f"Invalid format in transform file: {json_path}")
    else:
        # Option A: Standard wsireg / image2image .i2r.json matrix_xy_px
        if "matrix_xy_px" in data:
            if use_inverse and "matrix_xy_px_inv" in data:
                mat = np.array(data["matrix_xy_px_inv"], dtype=np.float64)
            else:
                mat_raw = np.array(data["matrix_xy_px"], dtype=np.float64)
                mat = np.linalg.inv(mat_raw) if use_inverse else mat_raw

        # Option B: matrix_yx_px (YX coordinate convention, permute to XY via S @ M @ S)
        elif "matrix_yx_px" in data:
            S = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            if use_inverse and "matrix_yx_px_inv" in data:
                mat_yx = np.array(data["matrix_yx_px_inv"], dtype=np.float64)
            else:
                mat_yx_raw = np.array(data["matrix_yx_px"], dtype=np.float64)
                mat_yx = np.linalg.inv(mat_yx_raw) if use_inverse else mat_yx_raw
            mat = S @ mat_yx @ S

        # Option C: Generic matrix / transform_matrix / affine_matrix
        elif "matrix" in data or "transform_matrix" in data or "affine_matrix" in data:
            raw_mat = data.get("matrix", data.get("transform_matrix", data.get("affine_matrix")))
            mat = np.array(raw_mat, dtype=np.float64)
            if mat.shape == (2, 3):
                mat = np.vstack([mat, [0.0, 0.0, 1.0]])
            if use_inverse:
                mat = np.linalg.inv(mat)

        # Option D: SimpleITK / scikit-image tform dict
        elif "tform" in data and isinstance(data["tform"], dict) and "params" in data["tform"]:
            mat = np.array(data["tform"]["params"], dtype=np.float64)
            if use_inverse:
                mat = np.linalg.inv(mat)

        # Option E: Parameter decomposition (scale, translation, rotation)
        elif "scale" in data or "translation" in data or "offset" in data:
            sx = float(data.get("scale_x", data.get("scale", 1.0)))
            sy = float(data.get("scale_y", data.get("scale", 1.0)))
            tx = float(data.get("tx", data.get("offset_x", data.get("translation_x", 0.0))))
            ty = float(data.get("ty", data.get("offset_y", data.get("translation_y", 0.0))))
            theta = np.deg2rad(float(data.get("rotation_deg", data.get("angle", 0.0))))
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            mat = np.array([
                [sx * cos_t, -sy * sin_t, tx],
                [sx * sin_t,  sy * cos_t, ty],
                [0.0,         0.0,        1.0]
            ], dtype=np.float64)
            if use_inverse:
                mat = np.linalg.inv(mat)

        else:
            raise ValueError(f"Could not locate a valid transformation matrix in: {json_path}")

    # Standardize to 3x3 homogeneous matrix
    if mat.shape == (2, 3):
        mat = np.vstack([mat, [0.0, 0.0, 1.0]])
    elif mat.shape != (3, 3):
        raise ValueError(f"Expected 2x3 or 3x3 matrix, got shape {mat.shape} in {json_path}")

    return TransformInfo(mat, target_shape, meta)


def transform_coordinates(
    coords: np.ndarray,
    matrix_3x3: np.ndarray
) -> np.ndarray:
    """
    Apply 3x3 affine or projective transformation matrix to Nx2 coordinate array.
    Performs homogeneous perspective divide by w:
        [x', y', w]^T = M @ [x, y, 1]^T
        x_cart = x' / w,  y_cart = y' / w

    For affine transformations, w = 1.0 everywhere.
    For projective transformations (homographies), w normalizes perspective tilt.
    """
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected Nx2 coordinate array, got shape {coords.shape}")

    n_pts = len(coords)
    if n_pts == 0:
        return coords.copy()

    pts_homo = np.hstack([coords.astype(np.float64), np.ones((n_pts, 1), dtype=np.float64)])
    # pts_homo is Nx3; matrix is 3x3. (M @ p)^T = p^T @ M^T
    trans_homo = pts_homo @ matrix_3x3.T

    w = trans_homo[:, 2:3]
    # Guard against division by zero
    w_safe = np.where(np.abs(w) > 1e-12, w, 1.0)
    transformed_xy = trans_homo[:, :2] / w_safe
    return transformed_xy


def transform_polygon_vertices(
    vertices: List[List[float]],
    matrix_3x3: np.ndarray
) -> List[List[float]]:
    """
    Apply 3x3 transformation matrix to 2D polygon vertices.
    Input format: [[x0, y0], [x1, y1], ...]
    """
    if len(vertices) < 3:
        return vertices
    pts = np.asarray(vertices, dtype=np.float64)
    trans = transform_coordinates(pts, matrix_3x3)
    return [[float(pt[0]), float(pt[1])] for pt in trans]


def compose_tile_transform_matrix(
    matrix_canvas: np.ndarray,
    tile_x0: float,
    tile_y0: float,
    target_x0: float = 0.0,
    target_y0: float = 0.0
) -> np.ndarray:
    """
    Compute composite transformation matrix mapping local tile coordinates into registered coordinates:
        M_tile = T(-target_x0, -target_y0) @ M_canvas @ T(tile_x0, tile_y0)
    """
    T_in = np.array([
        [1.0, 0.0, float(tile_x0)],
        [0.0, 1.0, float(tile_y0)],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    T_out = np.array([
        [1.0, 0.0, -float(target_x0)],
        [0.0, 1.0, -float(target_y0)],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    return T_out @ matrix_canvas @ T_in


def clip_polygon_to_box(
    vertices: List[List[float]],
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    max_stretch_ratio: float = 4.0
) -> List[List[float]]:
    """
    Robust Sutherland-Hodgman polygon clipping against rectangular canvas bounds.
    """
    if not vertices or len(vertices) < 3:
        return []

    pts = np.asarray(vertices, dtype=np.float64)
    if len(pts) > 2 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]

    if len(pts) < 3:
        return []

    orig_min_x, orig_max_x = float(pts[:, 0].min()), float(pts[:, 0].max())
    orig_min_y, orig_max_y = float(pts[:, 1].min()), float(pts[:, 1].max())
    orig_dx = orig_max_x - orig_min_x
    orig_dy = orig_max_y - orig_min_y

    if orig_min_x >= xmin and orig_max_x <= xmax and orig_min_y >= ymin and orig_max_y <= ymax:
        return [[float(p[0]), float(p[1])] for p in pts]

    if orig_max_x <= xmin or orig_min_x >= xmax or orig_max_y <= ymin or orig_min_y >= ymax:
        return []

    def clip_edge(poly, edge_test, edge_intersect):
        clipped = []
        if not poly:
            return clipped
        for i in range(len(poly)):
            cur_pt = poly[i]
            prev_pt = poly[i - 1]
            cur_inside = edge_test(cur_pt)
            prev_inside = edge_test(prev_pt)

            if cur_inside:
                if not prev_inside:
                    inter = edge_intersect(prev_pt, cur_pt)
                    if inter is not None:
                        clipped.append(inter)
                clipped.append(cur_pt)
            elif prev_inside:
                inter = edge_intersect(prev_pt, cur_pt)
                if inter is not None:
                    clipped.append(inter)
        return clipped

    def intersect_x(p1, p2, x_edge):
        dx = p2[0] - p1[0]
        denom = dx if abs(dx) > 1e-12 else (1e-12 if dx >= 0 else -1e-12)
        y = p1[1] + (p2[1] - p1[1]) * (x_edge - p1[0]) / denom
        return [float(x_edge), float(np.clip(y, ymin, ymax))]

    def intersect_y(p1, p2, y_edge):
        dy = p2[1] - p1[1]
        denom = dy if abs(dy) > 1e-12 else (1e-12 if dy >= 0 else -1e-12)
        x = p1[0] + (p2[0] - p1[0]) * (y_edge - p1[1]) / denom
        return [float(np.clip(x, xmin, xmax)), float(y_edge)]

    poly = [[float(p[0]), float(p[1])] for p in pts]
    poly = clip_edge(poly, lambda p: p[0] >= xmin, lambda p1, p2: intersect_x(p1, p2, xmin))
    poly = clip_edge(poly, lambda p: p[0] <= xmax, lambda p1, p2: intersect_x(p1, p2, xmax))
    poly = clip_edge(poly, lambda p: p[1] >= ymin, lambda p1, p2: intersect_y(p1, p2, ymin))
    poly = clip_edge(poly, lambda p: p[1] <= ymax, lambda p1, p2: intersect_y(p1, p2, ymax))

    if len(poly) < 3:
        return []

    cleaned = [poly[0]]
    for pt in poly[1:]:
        if abs(pt[0] - cleaned[-1][0]) > 1e-6 or abs(pt[1] - cleaned[-1][1]) > 1e-6:
            cleaned.append(pt)
    if len(cleaned) > 2 and abs(cleaned[0][0] - cleaned[-1][0]) < 1e-6 and abs(cleaned[0][1] - cleaned[-1][1]) < 1e-6:
        cleaned.pop()

    if len(cleaned) < 3:
        return []

    c_arr = np.asarray(cleaned)
    clipped_dx = float(c_arr[:, 0].max() - c_arr[:, 0].min())
    clipped_dy = float(c_arr[:, 1].max() - c_arr[:, 1].min())
    if clipped_dx < 0.5 or clipped_dy < 0.5:
        return []

    canvas_w = xmax - xmin
    canvas_h = ymax - ymin
    if canvas_w > 100:
        if orig_dx < 250.0 and clipped_dx > 0.3 * canvas_w and clipped_dx > (orig_dx * max_stretch_ratio):
            return []
    if canvas_h > 100:
        if orig_dy < 250.0 and clipped_dy > 0.3 * canvas_h and clipped_dy > (orig_dy * max_stretch_ratio):
            return []

    return cleaned


def filter_and_crop_polygon(
    vertices: List[List[float]],
    canvas_w: float,
    canvas_h: float,
    crop_mode: str = "intersect"
) -> Optional[List[List[float]]]:
    """
    Apply toggleable boundary filtering and cropping to transformed polygon coordinates.

    Parameters
    ----------
    vertices : List[List[float]]
        Transformed [[x, y], ...] coordinates.
    canvas_w, canvas_h : float
        Target canvas dimensions.
    crop_mode : str
        - 'none'     : Retain full polygon coordinates without modification or dropping.
                       Essential for intermediate stages of chained transformations.
        - 'intersect': Retain full unclipped polygon geometry if at least one vertex is
                       inside [0, W] x [0, H]. No vertices are clipped or modified.
        - 'clip'     : Execute Sutherland-Hodgman polygon box clipping strictly against canvas.
        - 'strict'   : Retain polygon ONLY if 100% of its vertices reside inside canvas bounds.
    """
    if not vertices or len(vertices) < 3:
        return None

    crop_mode_lower = str(crop_mode).lower().strip()
    if crop_mode_lower == "none":
        return vertices

    pts = np.asarray(vertices, dtype=np.float64)
    min_x, max_x = float(pts[:, 0].min()), float(pts[:, 0].max())
    min_y, max_y = float(pts[:, 1].min()), float(pts[:, 1].max())

    if crop_mode_lower == "strict":
        if min_x >= 0.0 and max_x <= canvas_w and min_y >= 0.0 and max_y <= canvas_h:
            return vertices
        return None

    elif crop_mode_lower == "intersect":
        # Keep if any part touches canvas, without chopping coordinates
        if max_x >= 0.0 and min_x <= canvas_w and max_y >= 0.0 and min_y <= canvas_h:
            return vertices
        return None

    elif crop_mode_lower == "clip":
        clipped = clip_polygon_to_box(vertices, 0.0, 0.0, canvas_w, canvas_h)
        return clipped if len(clipped) >= 3 else None

    else:
        # Default fallback to intersect
        if max_x >= 0.0 and min_x <= canvas_w and max_y >= 0.0 and min_y <= canvas_h:
            return vertices
        return None
