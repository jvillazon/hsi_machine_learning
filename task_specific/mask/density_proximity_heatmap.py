"""
Build a distance-propagated density heatmap from a mask image.

Goal
----
Highlight regions that are close to dense clusters of mask instances.

Pipeline
--------
1) Read mask image (2D, or select one slice from 3D stack).
2) Convert to instance labels:
   - If image already has many non-zero labels, treat as labeled instances.
   - Otherwise run connected components on binary mask.
3) Create a centroid map (1 impulse per instance centroid).
4) Smooth centroid map with Gaussian filter -> local density map.
5) Threshold high-density areas as "dense seed" regions.
6) Propagate outward with distance decay from dense seeds.
7) Combine density and propagated proximity into final heatmap.
8) Save:
   - float32 TIFF heatmap
   - optional PNG overlay

Example
-------
python task_specific/mask/density_proximity_heatmap.py \
  --mask-path "C:/path/to/mask.tif" \
  --output-dir "C:/path/to/output" \
  --density-sigma 60 \
  --dense-percentile 92 \
  --decay-distance 180
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import tifffile
import scipy.ndimage as ndi
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a distance-propagated density heatmap from mask instances."
    )
    parser.add_argument("--mask-path", type=str, required=True, help="Path to mask image (tif/tiff/npy).")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save outputs.")

    parser.add_argument(
        "--slice-index",
        type=int,
        default=0,
        help="If mask is 3D, slice index to process (default: 0).",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Optional integer downsample factor for very large images (default: 1 = no downsample).",
    )
    parser.add_argument(
        "--min-instance-area",
        type=int,
        default=20,
        help="Ignore connected components smaller than this many pixels.",
    )

    parser.add_argument(
        "--density-sigma",
        type=float,
        default=50.0,
        help="Gaussian sigma (physical units) to compute local density from centroids.",
    )
    parser.add_argument(
        "--dense-percentile",
        type=float,
        default=90.0,
        help="Percentile (0-100) on density map used to define dense seed regions.",
    )
    parser.add_argument(
        "--decay-scales",
        type=float,
        nargs="+",
        default=[100.0, 200.0, 300.0, 400.0, 500.0],
        help="List of distance decay constants (physical units) for multi-level mask expansion.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Physical resolution (units per pixel, e.g. 0.375). Default: 1.0 (pixels).",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Proximity threshold (0-1) to define binary mask extent (default 0.5 = 1/2 max proximity).",
    )
    parser.add_argument(
        "--density-weight",
        type=float,
        default=0.35,
        help="Weight for density term in final blend [0,1]. Only used if --save-heatmap is set.",
    )

    parser.add_argument(
        "--save-heatmap",
        action="store_true",
        help="Legacy: Save continuous float32 blended heatmap.",
    )
    parser.add_argument(
        "--save-components",
        action="store_true",
        help="Legacy: Save individual density and single-scale proximity TIFFs.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.65,
        help="Alpha for heatmap overlay PNG.",
    )
    parser.add_argument(
        "--overlay-downsample",
        type=int,
        default=1,
        help="Extra downsample factor used only for overlay PNG creation.",
    )
    parser.add_argument(
        "--max-overlay-mpix",
        type=float,
        default=60.0,
        help="Maximum megapixels for overlay rendering before auto-downsampling.",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="If set, skip PNG overlay generation.",
    )
    parser.add_argument(
        "--use-binary-propagation",
        action="store_true",
        help="If set, use hard distance cutoffs (equivalent to dilation) instead of exponential decay.",
    )
    parser.add_argument(
        "--structural-density",
        action="store_true",
        help="If set, compute density from actual instance masks instead of centroids.",
    )

    return parser.parse_args()


def _normalize_robust(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)

    vmin = np.percentile(arr[finite], 1)
    vmax = np.percentile(arr[finite], 99)
    if vmax - vmin < eps:
        return np.zeros_like(arr, dtype=np.float32)

    out = (arr - vmin) / (vmax - vmin)
    return np.clip(out, 0.0, 1.0)


def _load_mask(mask_path: Path, slice_index: int) -> np.ndarray:
    if mask_path.suffix.lower() == ".npy":
        mask = np.load(mask_path)
    else:
        mask = tifffile.imread(mask_path)

    if mask.ndim == 2:
        return mask

    if mask.ndim >= 3:
        if not (0 <= slice_index < mask.shape[0]):
            raise ValueError(
                f"slice_index {slice_index} out of bounds for first dimension size {mask.shape[0]}"
            )
        return mask[slice_index]

    raise ValueError(f"Unsupported mask dimensions: {mask.shape}")


def _to_instance_labels(mask_2d: np.ndarray, min_instance_area: int) -> np.ndarray:
    mask_2d = np.asarray(mask_2d)

    nonzero_vals = np.unique(mask_2d[mask_2d > 0])
    looks_labeled = len(nonzero_vals) > 2

    if looks_labeled:
        labels = mask_2d.astype(np.int32, copy=False)
    else:
        labels, _ = ndi.label(mask_2d > 0)

    if min_instance_area > 1:
        ids, counts = np.unique(labels, return_counts=True)
        valid_ids = ids[(ids != 0) & (counts >= min_instance_area)]
        keep = np.isin(labels, valid_ids)
        labels = np.where(keep, labels, 0).astype(np.int32)

        # Re-label sequentially after filtering to keep indexing compact.
        labels, _ = ndi.label(labels > 0)

    return labels


def build_mask_stack(
    labels: np.ndarray,
    density_sigma: float,
    dense_percentile: float,
    decay_scales: list[float],
    mask_threshold: float,
    use_binary_propagation: bool = False,
    structural_density: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a 3D stack of binary masks at different distance decay scales.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        - mask_stack: shape (N_scales, H, W), uint8
        - density_norm: shape (H, W), float32
        - dense_seed: shape (H, W), uint8
    """
    h, w = labels.shape

    if structural_density:
        # Use full instance shapes for density (more 'structural')
        source_map = (labels > 0).astype(np.float32)
    else:
        # 1 impulse per instance centroid (legacy/sparse)
        source_map = np.zeros((h, w), dtype=np.float32)
        label_ids = np.unique(labels)
        label_ids = label_ids[label_ids > 0]

        if len(label_ids) == 0:
            empty_stack = np.zeros((len(decay_scales), h, w), dtype=np.uint8)
            empty_2d = np.zeros((h, w), dtype=np.float32)
            return empty_stack, empty_2d, empty_2d.astype(np.uint8)

        centers = ndi.center_of_mass(np.ones_like(labels, dtype=np.float32), labels, label_ids)
        for cy, cx in centers:
            y = int(np.clip(round(cy), 0, h - 1))
            x = int(np.clip(round(cx), 0, w - 1))
            source_map[y, x] += 1.0

    density = ndi.gaussian_filter(source_map, sigma=max(0.1, density_sigma)).astype(np.float32)
    density_norm = _normalize_robust(density)

    threshold = np.percentile(density_norm[density_norm > 0], dense_percentile) if np.any(density_norm > 0) else 1.0
    dense_seed = density_norm >= threshold

    ds_labels, _ = ndi.label(dense_seed)

    # overlap_count = 0
    # Remove individual dense seeds with only two labels
    for obj_id in range(1, ds_labels.max() + 1):
        obj_pixels = (ds_labels == obj_id)
        # Check if any pixel of this object overlaps with any object in mask_b
        if np.any(obj_pixels):
            labels_in_obj = np.unique(labels[obj_pixels])
            if len(labels_in_obj) <= 2:  # Only one or two unique labels
                dense_seed[obj_pixels] = False  # Remove this seed
                # overlap_count += 1

    # # Fallback in case percentile is too strict for sparse images
    # if not np.any(dense_seed):
    #     dense_seed = density_norm > 0

    # Distance from dense cluster regions
    dist_to_dense = ndi.distance_transform_edt(~dense_seed).astype(np.float32)

    # Generate stack
    mask_stack = []
    for L in decay_scales:
        if use_binary_propagation:
            # 'Capped distance' / binary dilation equivalent
            # Thresholded at exactly L pixels from the seed
            binary_mask = (dist_to_dense <= L).astype(np.uint8)
        else:
            # Exponential decay proximity (legacy)
            proximity = np.exp(-dist_to_dense / max(1e-6, L)).astype(np.float32)
            # Threshold to create binary mask
            binary_mask = (proximity >= mask_threshold).astype(np.uint8)
        mask_stack.append(binary_mask)

    return np.array(mask_stack, dtype=np.uint8), density_norm, dense_seed.astype(np.uint8)


def build_blended_heatmap(
    density_norm: np.ndarray,
    dense_seed: np.ndarray,
    decay_distance: float,
    density_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Legacy helper to build the blended continuous heatmap."""
    dist_to_dense = ndi.distance_transform_edt(~(dense_seed > 0)).astype(np.float32)
    proximity = np.exp(-dist_to_dense / max(1e-6, decay_distance)).astype(np.float32)
    proximity_norm = _normalize_robust(proximity)

    dw = float(np.clip(density_weight, 0.0, 1.0))
    heatmap = dw * density_norm + (1.0 - dw) * proximity_norm
    heatmap = _normalize_robust(heatmap)
    return heatmap.astype(np.float32), proximity_norm


def _save_overlay_png_memory_safe(
    labels: np.ndarray,
    heatmap: np.ndarray,
    overlay_png: Path,
    alpha: float,
    overlay_downsample: int,
    max_overlay_mpix: float,
) -> tuple[int, int]:
    """Save overlay PNG with bounded memory usage.

    Returns
    -------
    tuple[int, int]
        Effective downsample factor used and output megapixels (rounded int).
    """
    ds = max(1, int(overlay_downsample))
    labels_v = labels[::ds, ::ds]
    heatmap_v = heatmap[::ds, ::ds]

    # Auto-downsample further if the overlay is still too large.
    mpix = heatmap_v.size / 1e6
    if mpix > max_overlay_mpix:
        extra = int(math.ceil(math.sqrt(mpix / max_overlay_mpix)))
        ds *= max(1, extra)
        labels_v = labels[::ds, ::ds]
        heatmap_v = heatmap[::ds, ::ds]

    base = ((labels_v > 0).astype(np.uint8) * 255)
    base_rgb = np.repeat(base[..., None], 3, axis=2).astype(np.float32)

    heat_rgb = (plt.get_cmap("inferno")(heatmap_v)[..., :3] * 255.0).astype(np.float32)
    a = float(np.clip(alpha, 0.0, 1.0))
    overlay = np.clip(base_rgb * (1.0 - a) + heat_rgb * a, 0, 255).astype(np.uint8)

    plt.imsave(overlay_png, overlay)
    out_mpix = int(round((overlay.shape[0] * overlay.shape[1]) / 1e6))
    return ds, out_mpix


def main() -> None:
    # args = parse_args()

    mask_path = r"C:\Users\jvill\Documents\integrated_pipeline\WSI_data\CODEX\cluster_data\K21_198_3\CD45, CD206.tif"
    output_dir = r"C:\Users\jvill\Documents\integrated_pipeline\WSI_data\CODEX\cluster_data\K21_198_3\CD45, CD206-cluster"
    downsample = 2
    density_sigma = 50
    dense_percentile = 90
    decay_scales = [10]
    resolution = 0.65
    mask_threshold = 0.5
    density_weight = 0.35
    use_binary_propagation = True
    structural_density = False
    save_heatmap = False
    save_components = False
    no_overlay = False
    overlay_alpha = 0.65
    overlay_downsample = 1
    max_overlay_mpix = 60.0
    min_instance_area = 20


    mask_path = Path(mask_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_2d = _load_mask(mask_path, slice_index=0)
    if downsample > 1:
        ds = int(downsample)
        mask_2d = mask_2d[::ds, ::ds]

    labels = _to_instance_labels(mask_2d, min_instance_area=min_instance_area)

    # Calculate effective resolution (original res * downsample factor)
    effective_res = resolution * downsample
    pixel_sigma = density_sigma / effective_res
    pixel_decay_scales = [s / effective_res for s in decay_scales]

    print(f"  Resolution: {resolution} units/px, Downsample: {downsample}x")
    print(f"  Effective Res: {effective_res} units/px")
    print(f"  Pixel Sigma: {pixel_sigma:.2f} px")

    mask_stack, density_norm, dense_seed = build_mask_stack(
        labels=labels,
        density_sigma=pixel_sigma,
        dense_percentile=dense_percentile,
        decay_scales=pixel_decay_scales,
        mask_threshold=mask_threshold,
        use_binary_propagation=use_binary_propagation,
        structural_density=structural_density,
    )

    stem = mask_path.stem
    mask_stack_tif = output_dir / f"{stem}_decay_mask_stack.tif"

    # Save 3D stack with metadata labels for ImageJ / users
    channel_names = [f"Scale_{s}" for s in decay_scales]
    tifffile.imwrite(
        mask_stack_tif,
        mask_stack.astype(np.uint8),
        imagej=True,
        metadata={"axes": "ZYX", "Labels": channel_names},
    )

    # Always save the "seed" as a separate file for convenience
    dense_seed_tif = output_dir / f"{stem}_dense_seed_core.tif"
    tifffile.imwrite(dense_seed_tif, (dense_seed*255).astype(np.uint8))

    # Optional heatmap/components for legacy workflows
    if save_heatmap or save_components or not no_overlay:
        # Use the first scale as the 'primary' for legacy outputs if multiple scales are provided
        primary_decay_px = pixel_decay_scales[0]
        heatmap, proximity_norm = build_blended_heatmap(
            density_norm=density_norm,
            dense_seed=dense_seed,
            decay_distance=primary_decay_px,
            density_weight=density_weight,
        )

        if save_heatmap:
            heatmap_tif = output_dir / f"{stem}_dense_proximity_heatmap.tif"
            tifffile.imwrite(heatmap_tif, heatmap.astype(np.float32))
            print(f"  Heatmap TIFF: {heatmap_tif}")

        if save_components:
            density_tif = output_dir / f"{stem}_density_component.tif"
            proximity_tif = output_dir / f"{stem}_proximity_component.tif"
            tifffile.imwrite(density_tif, density_norm.astype(np.float32))
            tifffile.imwrite(proximity_tif, proximity_norm.astype(np.float32))
            print(f"  Components saved: {density_tif}, {proximity_tif}")

        if not no_overlay:
            overlay_png = output_dir / f"{stem}_dense_proximity_overlay.png"
            used_ds, out_mpix = _save_overlay_png_memory_safe(
                labels=labels,
                heatmap=heatmap,
                overlay_png=overlay_png,
                alpha=overlay_alpha,
                overlay_downsample=overlay_downsample,
                max_overlay_mpix=max_overlay_mpix,
            )
            print(f"  Overlay saved (ds={used_ds}, size~{out_mpix} MP): {overlay_png}")

    print("Done.")
    print(f"  Input mask:   {mask_path}")
    print(f"  Output dir:   {output_dir}")
    print(f"  Mask Stack:   {mask_stack_tif} ({len(decay_scales)} levels)")
    print(f"  Seed Core:    {dense_seed_tif}")


if __name__ == "__main__":
    main()
