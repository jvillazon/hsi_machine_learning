#!/usr/bin/env python3
"""
hSRS_embed_util.py
==================
High-Performance, Memory-Efficient Preprocessing, Masking & Feature Extraction
Utility for Hyperspectral Stimulated Raman Scattering (hSRS) Images.

Key Features & Updates:
-----------------------
1. Strict Tissue Mask Filtering:
   - Dynamic non-resonant silent baseline cutoff index: ch_start = int(((2800 - wn1) / (wn2 - wn1)) * n_wn).
   - Foreground tissue mask is strictly what determines the extracted and embedded spectra.
   - Exact (Y, X) pixel coordinates are generated directly from the boolean tissue mask for every valid pixel.
2. Full Population Extraction (No Subsampling):
   - Extracts and embeds 100% of valid tissue pixels identified by the tissue mask.
3. Native `extract_cls_features`:
   - Uses the standard DINO `extract_cls_features` encoder function (calling model.get_features on [B, 1, 62] tensors).
4. Image-Level Custom Metadata Preservation:
   - Preserves all image-level metadata dictionary keys (e.g. sample_id, experimental_condition, class, roi, etc.)
     in the dataset manifest.
5. GPU-Accelerated Vectorized Pipeline:
   - Vectorized water baseline subtraction across all pixels.
   - PyTorch GPU 1D linear interpolation to 62 channels.
   - Single-pass memory mapping (np.memmap) without intermediate per-image .npy disk duplicates.
   - Toggleable debugging visualization (debug=True).
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tqdm import tqdm


# ==============================================================================
# 1. CORE VECTORIZED SPECTRAL OPERATIONS & MASKING
# ==============================================================================

def load_water_baseline(
    water_baseline_path: Optional[Union[str, Path]],
    spectra_start: float = 2700.0,
    spectra_end: float = 3100.0,
) -> Optional[interp1d]:
    """
    Load and return cubic interpolator for water baseline data.
    """
    if not water_baseline_path or not os.path.exists(water_baseline_path):
        return None

    try:
        baseline_data = pd.read_csv(water_baseline_path, header=None).values.flatten()
        x_original = np.linspace(spectra_start, spectra_end, len(baseline_data))
        # Reverse baseline data order to match acquisition sequence (3100 -> 2700 cm^-1)
        baseline_data = baseline_data[::-1]
        interpolator = interp1d(
            x_original,
            baseline_data,
            kind="cubic",
            fill_value=(baseline_data[0], baseline_data[-1]),
            bounds_error=False,
        )
        return interpolator
    except Exception as e:
        print(f"[Warning] Failed to load water baseline from {water_baseline_path}: {e}")
        return None


def vectorized_water_subtraction(
    spectra_2d: np.ndarray,
    water_interpolator: Optional[interp1d],
    wavenumbers: np.ndarray,
    baseline_sample_pts: int = 5,
) -> np.ndarray:
    """
    Performs fully vectorized water baseline subtraction across all pixels in one BLAS pass.
    
    Parameters:
    -----------
    spectra_2d : np.ndarray
        Raw spectra matrix of shape (N_pixels, N_channels).
    water_interpolator : scipy.interpolate.interp1d or None
        Water baseline interpolator.
    wavenumbers : np.ndarray
        1D wavenumber array corresponding to input channels.
    baseline_sample_pts : int
        Number of starting channels to average for per-pixel scaling.
    """
    if water_interpolator is None:
        return spectra_2d

    # Interpolate water baseline to match wavenumber grid
    water_curve = water_interpolator(wavenumbers).astype(np.float32)
    water_curve = np.clip(water_curve, 0.0, 1.0)

    # Compute per-pixel scale factor from the initial baseline channels (N_pixels, 1)
    scale_factors = np.mean(spectra_2d[:, :baseline_sample_pts], axis=1, keepdims=True)
    scale_factors = np.maximum(scale_factors, 0.0)

    # Vectorized subtraction: (N, C) - (N, 1) * (1, C)
    corrected = spectra_2d - (scale_factors * water_curve[np.newaxis, :])
    return np.maximum(corrected, 0.0).astype(np.float32)


def compute_image_mask(
    spectra_2d: np.ndarray,
    shape: Tuple[int, int],
    wavenumbers: Optional[np.ndarray] = None,
    spectra_start: float = 2700.0,
    spectra_end: float = 3100.0,
    baseline_cutoff_wn: float = 2800.0,
    n_sigma: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    Computes the tissue foreground mask using spectral standard deviation against
    the non-resonant silent baseline region (< 2800 cm^-1).
    
    Parameters:
    -----------
    spectra_2d : np.ndarray
        Array of shape (H*W, N_channels) after water subtraction.
    shape : tuple of (height, width)
        Original 2D spatial dimensions.
    wavenumbers : np.ndarray, optional
        Wavenumber grid. If None, linearly spaced from spectra_start to spectra_end.
    spectra_start : float
        Starting wavenumber (default: 2700 cm^-1).
    spectra_end : float
        Ending wavenumber (default: 3100 cm^-1).
    baseline_cutoff_wn : float
        Wavenumber threshold for silent baseline region (default: 2800 cm^-1).
    n_sigma : float
        Multiplier for noise threshold (default: 2.0).

    Returns:
    --------
    tissue_mask : np.ndarray (bool array of shape (H*W,) or (H, W))
    coords : np.ndarray of shape (N_valid, 2) containing exact (Y, X) pixel coordinates.
    threshold : float
    ch_start : int (cutoff channel index)
    """
    h, w = shape
    n_pixels, n_channels = spectra_2d.shape

    # 1. Non-zero pixel filter (active tissue area)
    tissue_active = (spectra_2d != 0).any(axis=1)
    if not tissue_active.any():
        return np.zeros(n_pixels, dtype=bool), np.empty((0, 2), dtype=np.int64), 0.0, 0

    # 2. Dynamic ch_start index calculation based on wavenumber range: (2800 - wn1)/(wn2 - wn1) * n_channels
    if wavenumbers is not None and len(wavenumbers) == n_channels:
        ch_start = int(np.searchsorted(wavenumbers, baseline_cutoff_wn))
    else:
        fraction = (baseline_cutoff_wn - spectra_start) / (spectra_end - spectra_start)
        ch_start = int(round(fraction * n_channels))

    ch_start = max(1, min(ch_start, n_channels - 1))

    # 3. Spectral standard deviations: full-band vs silent baseline region (< 2800 cm^-1)
    spectral_std = spectra_2d.std(axis=1)
    baseline_std = spectra_2d[:, :ch_start].std(axis=1)

    # 4. Statistical threshold from active baseline pixels
    active_baseline_std = baseline_std[tissue_active]
    threshold = float(np.mean(active_baseline_std) + n_sigma * np.std(active_baseline_std))

    # 5. Build boolean tissue mask (True for valid tissue pixels to embed)
    tissue_mask = (spectral_std > threshold) & tissue_active

    # 6. Generate exact (Y, X) pixel coordinates strictly from the tissue mask
    tissue_mask_2d = tissue_mask.reshape(h, w)
    coords = np.column_stack(np.where(tissue_mask_2d)) # (N_valid, 2) array of [y, x]

    return tissue_mask, coords, threshold, ch_start


def gpu_vectorized_interpolation(
    spectra_2d: np.ndarray,
    target_channels: int = 62,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Vectorized 1D linear spectral interpolation using PyTorch GPU acceleration.
    Resamples (N_pixels, C_in) -> (N_pixels, target_channels) in ~1 ms.
    """
    if spectra_2d.shape[1] == target_channels:
        return spectra_2d.astype(np.float32)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        tensor_in = torch.from_numpy(spectra_2d).to(device=device, dtype=torch.float32).unsqueeze(1)
        tensor_out = F.interpolate(
            tensor_in,
            size=target_channels,
            mode="linear",
            align_corners=True,
        ).squeeze(1)

        if device.type == "cuda":
            return tensor_out.cpu().numpy()
        return tensor_out.numpy()


def normalize_spectrum_matrix(
    spectra_2d: np.ndarray,
    method: str = "intensity",
) -> np.ndarray:
    """
    Vectorized normalization across all spectra rows simultaneously.
    """
    if np.all(spectra_2d == 0):
        return spectra_2d

    if method == "intensity":
        max_vals = np.max(np.abs(spectra_2d), axis=1, keepdims=True)
        max_vals[max_vals == 0] = 1.0
        return (spectra_2d / max_vals).astype(np.float32)

    elif method == "minmax":
        min_vals = np.min(spectra_2d, axis=1, keepdims=True)
        max_vals = np.max(spectra_2d, axis=1, keepdims=True)
        diff = max_vals - min_vals
        diff[diff == 0] = 1.0
        return ((spectra_2d - min_vals) / diff).astype(np.float32)

    elif method == "area":
        areas = np.trapz(spectra_2d, axis=1)[:, np.newaxis]
        areas[areas == 0] = 1.0
        return (spectra_2d / areas).astype(np.float32)

    elif method == "snv":
        means = np.mean(spectra_2d, axis=1, keepdims=True)
        stds = np.std(spectra_2d, axis=1, keepdims=True)
        stds[stds == 0] = 1.0
        return ((spectra_2d - means) / stds).astype(np.float32)

    return spectra_2d.astype(np.float32)


# ==============================================================================
# 2. DINO CLS FEATURE EXTRACTION
# ==============================================================================

def extract_cls_features(
    model: nn.Module,
    spectra: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    """
    Run encoder and return CLS features for each spectrum in mini-batches.
    
    Parameters:
    -----------
    model : nn.Module
        Pretrained SRS_DINO model.
    spectra : np.ndarray
        Array of shape (N_pixels, 62).
    device : torch.device
        Computation device (CUDA or CPU).
    batch_size : int
        Batch size for forward pass (default: 4096).
        
    Returns:
    --------
    feats : np.ndarray of shape (N_pixels, feat_dim)
    """
    model.eval()
    num = spectra.shape[0]
    feats = []

    with torch.no_grad():
        for start in tqdm(range(0, num, batch_size), desc="Extracting CLS features", leave=False):
            end = min(start + batch_size, num)
            batch = torch.from_numpy(spectra[start:end]).to(device)  # [B, 62]
            batch = batch.unsqueeze(1)  # [B, 1, 62]

            # Standard DINO feature extraction
            if hasattr(model, "get_features"):
                cls_feat, _ = model.get_features(batch)
            elif hasattr(model, "student") and hasattr(model.student, "backbone"):
                cls_feat = model.student.backbone(batch.squeeze(1))
            elif hasattr(model, "forward_features"):
                cls_feat = model.forward_features(batch)
            else:
                cls_feat = model(batch)

            if isinstance(cls_feat, tuple):
                cls_feat = cls_feat[0]

            feats.append(cls_feat.cpu().numpy())

    return np.concatenate(feats, axis=0) if len(feats) > 0 else np.empty((0, 0), dtype=np.float32)


# ==============================================================================
# 3. DIAGNOSTIC & DEBUGGING VISUALIZATION
# ==============================================================================

def generate_debug_diagnostic_plot(
    raw_cube: np.ndarray,
    water_sub_2d: np.ndarray,
    norm_interp_valid: np.ndarray,
    tissue_mask_2d: np.ndarray,
    coords: np.ndarray,
    threshold: float,
    ch_start: int,
    wavenumbers_in: np.ndarray,
    wavenumbers_target: np.ndarray,
    image_name: str,
    save_path: Union[str, Path],
):
    """
    Generates a 4-panel diagnostic plot validating the tissue masking and spectral pipeline.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=150)
    fig.suptitle(f"hSRS Preprocessing Diagnostic: {image_name}", fontsize=15, fontweight="bold")

    h, w = tissue_mask_2d.shape

    # Panel 1: Peak C-H Intensity Map (around 2930 cm^-1 / peak channel)
    peak_ch_idx = min(len(raw_cube) - 1, int(round(len(raw_cube) * 0.55)))
    im0 = axes[0, 0].imshow(raw_cube[peak_ch_idx], cmap="inferno")
    axes[0, 0].set_title(f"Raw Peak C-H Map (Channel {peak_ch_idx})", fontsize=12)
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # Panel 2: Standard Deviation Map with Thresholding
    std_map = water_sub_2d.std(axis=1).reshape(h, w)
    im1 = axes[0, 1].imshow(std_map, cmap="viridis")
    axes[0, 1].set_title(f"Spectral Std-Dev Map (Threshold: {threshold:.4f}, ch_start: {ch_start})", fontsize=12)
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Panel 3: Foreground Tissue Mask (determines embedded spectra)
    im2 = axes[1, 0].imshow(tissue_mask_2d, cmap="gray")
    valid_count = len(coords)
    pct = (valid_count / (h * w)) * 100.0 if (h * w) > 0 else 0.0
    axes[1, 0].set_title(f"Tissue Mask -> Embedded Spectra ({valid_count:,} px, {pct:.1f}%)", fontsize=12)
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Panel 4: Sample Normalized 62-channel Spectra from Valid Coordinates
    if valid_count > 0:
        sample_indices = np.random.choice(valid_count, size=min(8, valid_count), replace=False)
        for idx in sample_indices:
            axes[1, 1].plot(wavenumbers_target, norm_interp_valid[idx], alpha=0.7, linewidth=1.5)

    axes[1, 1].axvline(2800.0, color="red", linestyle="--", alpha=0.7, label="2800 cm⁻¹ Baseline Cutoff")
    axes[1, 1].set_title("Sample Extracted & Embedded Spectra (62-ch)", fontsize=12)
    axes[1, 1].set_xlabel("Wavenumber (cm⁻¹)", fontsize=10)
    axes[1, 1].set_ylabel("Normalized Intensity", fontsize=10)
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)
    axes[1, 1].legend(loc="upper left")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# 4. HIGH-PERFORMANCE DATASET PROCESSOR & EMBEDDER
# ==============================================================================

def process_and_embed_hSRS_dataset(
    images_dict: Dict[str, Dict[str, Any]],
    model: Optional[nn.Module] = None,
    output_dir: Union[str, Path] = "./hSRS_processed",
    output_prefix: str = "dataset",
    water_baseline_path: Optional[str] = None,
    spectra_start: float = 2700.0,
    spectra_end: float = 3100.0,
    target_channels: int = 62,
    norm_method: str = "intensity",
    baseline_cutoff_wn: float = 2800.0,
    n_sigma: float = 2.0,
    batch_size: int = 4096,
    device: Optional[Union[str, torch.device]] = None,
    debug: bool = False,
    debug_dir: Optional[Union[str, Path]] = None,
    debug_max_images: int = 5,
) -> Dict[str, Any]:
    """
    End-to-End, High-Performance Processor for hSRS Image Collections.
    
    Workflow:
    ---------
    1. Ingests images dictionary with arbitrary image-level metadata:
       images_dict = {
           'image_key': {
               'path': 'path/to/image.tif',
               'class': 'Glomerulus',
               'sample_id': '3811',
               'experimental_condition': 'Ref',
               ... (any custom metadata fields)
           }
       }
    2. Performs vectorized water subtraction and computes dynamic tissue mask (< 2800 cm^-1).
    3. Extracts 100% of the spectra identified by the tissue mask (no subsampling).
    4. Resamples valid spectra to target channels (62) using PyTorch GPU interpolation.
    5. Normalizes spectra (intensity, minmax, area, or snv).
    6. Extracts CLS features using `extract_cls_features(model, spectra, device, batch_size)`.
    7. Directly streams raw spectra and DINO features into unified memory-maps (no intermediate .npy files).
    8. Exports complete structured manifest in Parquet format preserving all image-level metadata.
    """
    t0 = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    print("=" * 70)
    print("hSRS HIGH-PERFORMANCE PREPROCESSING & EMBEDDING PIPELINE")
    print("=" * 70)
    print(f"Total Images    : {len(images_dict)}")
    print(f"Device          : {device}")
    print(f"Target Channels : {target_channels} ({spectra_start} - {spectra_end} cm^-1)")
    print(f"Baseline Cutoff : {baseline_cutoff_wn} cm^-1 (n_sigma = {n_sigma})")
    print(f"Water Baseline  : {water_baseline_path if water_baseline_path else 'None'}")
    print(f"Output Directory: {output_dir}")
    print(f"Debug Plots     : {debug}")
    print("=" * 70)

    # 1. Load water baseline interpolator
    water_interp = load_water_baseline(water_baseline_path, spectra_start, spectra_end)
    target_wavenumbers = np.linspace(spectra_start, spectra_end, target_channels, dtype=np.float32)

    # 2. Phase 1: Ingest, mask, interpolate, and extract all valid tissue spectra
    print("\n[Phase 1/2] Processing images, tissue masks & extracting valid spectra...")
    image_metadata_list = []
    processed_spectra_cache = {}
    total_valid_pixels = 0
    debug_count = 0

    if debug:
        if debug_dir is None:
            debug_dir = output_dir / "debug_plots"
        else:
            debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

    for img_name, img_info in tqdm(images_dict.items(), desc="Image Processing"):
        img_path = img_info.get("path", "")
        if not img_path or not os.path.exists(img_path):
            print(f"[Warning] Skipping missing file for '{img_name}': {img_path}")
            continue

        # Load raw TIFF cube (C, H, W) and flip along axis 0
        raw_stack = tifffile.imread(img_path)
        raw_stack = np.flip(raw_stack, axis=0)  # flip to 2700 -> 3100 cm^-1 order
        c_in, h, w = raw_stack.shape

        in_wavenumbers = np.linspace(spectra_start, spectra_end, c_in, dtype=np.float32)

        # Reshape to (H*W, C_in)
        raw_2d = raw_stack.reshape(c_in, -1).T.astype(np.float32)

        # 2a. Vectorized water subtraction
        water_sub_2d = vectorized_water_subtraction(raw_2d, water_interp, in_wavenumbers)

        # 2b. Compute tissue mask and extract valid pixel coordinates
        tissue_mask, coords, threshold, ch_start = compute_image_mask(
            spectra_2d=water_sub_2d,
            shape=(h, w),
            wavenumbers=in_wavenumbers,
            spectra_start=spectra_start,
            spectra_end=spectra_end,
            baseline_cutoff_wn=baseline_cutoff_wn,
            n_sigma=n_sigma,
        )

        n_valid = len(coords)
        if n_valid == 0:
            print(f"[{img_name}] Warning: 0 valid tissue pixels found. Skipping image.")
            continue

        # 2c. Tissue mask strictly determines the extracted spectra
        valid_raw_spectra = water_sub_2d[tissue_mask]

        # 2d. Vectorized GPU linear interpolation to target channels (62)
        interp_spectra = gpu_vectorized_interpolation(
            spectra_2d=valid_raw_spectra,
            target_channels=target_channels,
            device=device,
        )

        # 2e. Normalization
        norm_spectra = normalize_spectrum_matrix(interp_spectra, method=norm_method)

        # 2f. Debug diagnostic plot
        if debug and debug_count < debug_max_images:
            debug_count += 1
            debug_fig_path = debug_dir / f"{img_name}_diagnostic.png"
            generate_debug_diagnostic_plot(
                raw_cube=raw_stack,
                water_sub_2d=water_sub_2d,
                norm_interp_valid=norm_spectra,
                tissue_mask_2d=tissue_mask.reshape(h, w),
                coords=coords,
                threshold=threshold,
                ch_start=ch_start,
                wavenumbers_in=in_wavenumbers,
                wavenumbers_target=target_wavenumbers,
                image_name=img_name,
                save_path=debug_fig_path,
            )

        start_idx = total_valid_pixels
        end_idx = start_idx + n_valid
        total_valid_pixels = end_idx

        # Cache in memory for Phase 2
        processed_spectra_cache[img_name] = norm_spectra

        # Construct image metadata entry preserving all user-provided metadata keys
        meta_entry = {
            "image_name": str(img_name),
            "height": int(h),
            "width": int(w),
            "shape": (int(h), int(w)),
            "total_pixels": int(h * w),
            "valid_pixels": int(n_valid),
            "valid_idx": (int(start_idx), int(end_idx)),
            "start_idx": int(start_idx),
            "end_idx": int(end_idx),
            "std_threshold": round(float(threshold), 6),
            "ch_start_index": int(ch_start),
            "coords": coords.tolist(),  # Exact (Y, X) pixel coordinates list
        }

        # Merge all additional custom metadata from img_info (e.g. sample_id, experimental_condition, class)
        for key, val in img_info.items():
            if key not in meta_entry:
                meta_entry[key] = val

        image_metadata_list.append(meta_entry)

    print(f"\nTotal valid foreground spectra across dataset: {total_valid_pixels:,}")
    if total_valid_pixels == 0:
        raise ValueError("No valid spectra found across the provided images.")

    # 3. Phase 2: Single-pass Memory Map Allocation & DINO CLS Feature Extraction
    print("\n[Phase 2/2] Allocating single-pass unified memory-maps & extracting CLS features...")
    spec_mmap_path = output_dir / f"{output_prefix}_all_raw_spec.npy"
    spectra_mmap = np.lib.format.open_memmap(
        spec_mmap_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_valid_pixels, target_channels),
    )

    feat_mmap = None
    feat_dim = 0

    # Probe feature dimension if model is provided
    if model is not None:
        model.eval()
        model.to(device)

        dummy_spec = np.zeros((2, target_channels), dtype=np.float32)
        dummy_feat = extract_cls_features(model, dummy_spec, device=device, batch_size=2)
        feat_dim = dummy_feat.shape[1]
        print(f"DINO Embedding Feature Dimension: {feat_dim}")

        feat_mmap_path = output_dir / f"{output_prefix}_all_feats.npy"
        feat_mmap = np.lib.format.open_memmap(
            feat_mmap_path,
            mode="w+",
            dtype=np.float32,
            shape=(total_valid_pixels, feat_dim),
        )

    # Stream write directly into unified memory-maps
    for meta in tqdm(image_metadata_list, desc="Memory-Map Streaming"):
        img_name = meta["image_name"]
        s_idx = meta["start_idx"]
        e_idx = meta["end_idx"]
        specs = processed_spectra_cache.pop(img_name)

        # Write spectra to master memmap
        spectra_mmap[s_idx:e_idx] = specs

        # Extract CLS features using extract_cls_features and write directly to master feats memmap
        if model is not None and feat_mmap is not None:
            feats = extract_cls_features(model, specs, device=device, batch_size=batch_size)
            feat_mmap[s_idx:e_idx] = feats

    spectra_mmap.flush()
    if feat_mmap is not None:
        feat_mmap.flush()

    # 4. Export Structured Parquet Manifest & JSON
    df_manifest = pd.DataFrame(image_metadata_list)
    # Convert coords column to JSON string for efficient Parquet serialization
    df_manifest_export = df_manifest.copy()
    if "coords" in df_manifest_export.columns:
        df_manifest_export["coords_json"] = df_manifest_export["coords"].apply(json.dumps)
        df_manifest_export = df_manifest_export.drop(columns=["coords"])

    manifest_parquet_path = output_dir / f"{output_prefix}_manifest.parquet"
    df_manifest_export.to_parquet(manifest_parquet_path, index=False)

    manifest_json_path = output_dir / f"{output_prefix}_manifest.json"
    with open(manifest_json_path, "w") as f:
        json.dump({
            "total_valid_pixels": total_valid_pixels,
            "target_channels": target_channels,
            "feat_dim": feat_dim,
            "spectra_start": spectra_start,
            "spectra_end": spectra_end,
            "norm_method": norm_method,
            "baseline_cutoff_wn": baseline_cutoff_wn,
            "images": image_metadata_list,
        }, f, indent=2)

    print("\n" + "=" * 70)
    print(f"hSRS Processing Finished Successfully in {time.time() - t0:.2f}s")
    print(f"Unified Spectra Memmap : {spec_mmap_path} ({os.path.getsize(spec_mmap_path)/1024/1024:.2f} MB)")
    if feat_mmap is not None:
        print(f"Unified Feats Memmap   : {feat_mmap_path} ({os.path.getsize(feat_mmap_path)/1024/1024:.2f} MB)")
    print(f"Dataset Manifest       : {manifest_parquet_path}")
    print("=" * 70)

    return {
        "total_spectra": total_valid_pixels,
        "spectra_memmap_path": str(spec_mmap_path),
        "feats_memmap_path": str(feat_mmap_path) if feat_mmap is not None else None,
        "manifest_parquet": str(manifest_parquet_path),
        "manifest_json": str(manifest_json_path),
        "metadata": image_metadata_list,
    }


if __name__ == "__main__":
    print("hSRS Embedding Utility loaded successfully.")
