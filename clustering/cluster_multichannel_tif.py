#!/usr/bin/env python3
"""
Multi-Channel TIFF Segmentation & Visualization Tool

Supports two complementary modes:
1. 'prevalence' (Default / Channel Dominance Split):
   Standardizes each channel (z-score / standard scaling, robust, or minmax),
   then assigns each pixel to the channel where its standardized value is highest (argmax).
   Reveals exactly where each specific channel is most prevalent / elevated relative to normal.

2. 'kmeans':
   Unsupervised multi-channel pixel clustering (e.g. k=7) using MiniBatchKMeans.

Exports:
- Integer Label TIFF (*_labels.tif)
- Colored RGB TIFF (*_rgb.tif)
- Confidence / Dominance Margin TIFF (*_margin.tif) [in prevalence mode]
- Summary Statistics CSV (*_profiles.csv)
- Publication-quality multi-panel visualization (*_summary.png) [Memory-safe downsampled preview]
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import tifffile


# Maximally distinguishable, high-contrast publication-ready colormaps (optimized for both dark tissue background and white figures)
DISTINCT_COLORS_HEX = [
    "#4d1b54",  # 1: Vivid Crimson Red
    "#00FF00",  # 2: Electric Cyan
    "#FFFF00",  # 3: Fluorescent Lime Green
    "#FF0000",  # 4: Bright Electric Yellow
    "#0000FF",  # 5: Vivid Magenta / Hot Pink
    "#00FFFF",  # 6: Vivid Amber / Bright Orange
    "#FF00FF",  # 7: Electric Violet / Bright Purple
    "#4363D8",  # 8: Royal Blue
    "#46F0F0",  # 9: Turquoise
    "#FABED4",  # 10: Pale Rose
    "#A9A9A9",  # 11: Neutral Grey
    "#FFFAC8",  # 12: Pale Lemon
    "#800000",  # 13: Maroon
    "#AAFFC3",  # 14: Mint Green
    "#808000",  # 15: Olive
]


def get_palette(n_items: int) -> np.ndarray:
    """Returns an RGB colormap array of shape (n_items, 3) in range [0, 255]."""
    if n_items <= len(DISTINCT_COLORS_HEX):
        hex_colors = DISTINCT_COLORS_HEX[:n_items]
        rgb = [
            [int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)]
            for h in hex_colors
        ]
        return np.array(rgb, dtype=np.uint8)
    else:
        cmap = plt.get_cmap("tab20", n_items)
        rgb = (cmap(np.arange(n_items))[:, :3] * 255).astype(np.uint8)
        return rgb


def downsample_for_preview(img: np.ndarray, max_dim: int = 1800) -> np.ndarray:
    """
    Subsamples a 2D or 3D image array for memory-safe matplotlib plotting.
    Avoids multi-gigabyte RGBA memory allocations on massive WSIs (e.g. 10k x 20k).
    """
    h, w = img.shape[:2]
    max_side = max(h, w)
    if max_side <= max_dim:
        return img
    step = int(np.ceil(max_side / max_dim))
    if img.ndim == 2:
        return img[::step, ::step]
    elif img.ndim == 3:
        return img[::step, ::step, :]
    return img


def load_multichannel_tif(
    file_path: Union[str, Path],
    use_memmap: bool = False
) -> Tuple[np.ndarray, Tuple[int, int, int], str]:
    """
    Loads a multi-channel TIFF and returns data arranged as (H, W, C) float32,
    along with original shape and detected dimension format.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    print(f"[*] Loading TIFF: {file_path}")
    raw_data = None
    if use_memmap:
        try:
            raw_data = tifffile.memmap(str(file_path), mode="r")
            print("[*] Memory-mapping successful.")
        except Exception as e:
            print(f"[!] Note: TIFF is not directly memory-mappable ({e}). Reading into RAM with tifffile.imread...")
            raw_data = tifffile.imread(str(file_path))
    else:
        raw_data = tifffile.imread(str(file_path))

    orig_shape = raw_data.shape
    print(f"[*] Loaded raw image array with shape: {orig_shape}, dtype: {raw_data.dtype}")

    # Squeeze single-dimension leading axes (e.g. (1, C, H, W) -> (C, H, W))
    while raw_data.ndim > 3 and raw_data.shape[0] == 1:
        raw_data = raw_data[0]

    if raw_data.ndim == 2:
        # Single channel -> reshape to (H, W, 1)
        data = raw_data[:, :, np.newaxis]
        channel_format = "HW -> (H, W, 1)"
    elif raw_data.ndim == 3:
        # Determine whether axis 0 is channel (C, H, W) or axis 2 is channel (H, W, C)
        if raw_data.shape[0] < raw_data.shape[1] and raw_data.shape[0] < raw_data.shape[2]:
            data = np.transpose(raw_data, (1, 2, 0))
            channel_format = "CHW -> HWC"
        else:
            data = raw_data
            channel_format = "HWC"
    else:
        raise ValueError(
            f"Unsupported TIFF dimensions: {raw_data.shape}. Expected 2D or 3D multi-channel stack."
        )

    h, w, c = data.shape
    print(f"[*] Standardized image shape (H, W, C): {h} x {w} pixels across {c} channel(s) [{channel_format}]")
    return data, (h, w, c), channel_format


def get_scaler(normalization: str):
    """Returns the requested scikit-learn feature scaler."""
    if normalization == "standard":
        return StandardScaler()
    elif normalization == "robust":
        return RobustScaler()
    elif normalization == "minmax":
        return MinMaxScaler()
    elif normalization == "none":
        return None
    else:
        raise ValueError(f"Unknown normalization mode: {normalization}")


# ==============================================================================
# METHOD 1: Channel Prevalence / Dominance Segmentation (Argmax of Standardized)
# ==============================================================================
def channel_prevalence_segmentation(
    data: np.ndarray,
    normalization: str = "standard",
    mask_background: bool = False,
    bg_threshold: float = 0.0,
    sample_size: int = 1_000_000,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Standardizes each channel across foreground pixels, then assigns each pixel
    to the channel where its standardized value is highest:
        Prevalent_Channel(x, y) = argmax_c ( Z_c(x, y) )

    Also calculates a dominance margin:
        Margin(x, y) = Top_Z_score - 2nd_Top_Z_score

    Returns:
    --------
    prevalence_map : np.ndarray (H, W)
        Integer label indicating dominant channel index (0 to C-1). -1 for background.
    margin_map : np.ndarray (H, W)
        Float32 margin / confidence score.
    stats_df : pd.DataFrame
        Summary metrics for each channel's territory.
    """
    h, w, c = data.shape
    n_pixels = h * w
    print(f"[*] Reshaping {n_pixels:,} total pixels for Channel Prevalence Segmentation...")

    flat_data = data.reshape(-1, c)

    # Background filtering
    if mask_background:
        pixel_intensities = np.sum(np.abs(flat_data), axis=1)
        valid_mask = pixel_intensities > bg_threshold
        valid_pixels = flat_data[valid_mask]
        n_valid = len(valid_pixels)
        print(f"[*] Background masking active: {n_valid:,} / {n_pixels:,} ({100*n_valid/n_pixels:.1f}%) non-background pixels.")
        if n_valid == 0:
            raise ValueError("All pixels fell below the background threshold. Lower bg_threshold.")
    else:
        valid_mask = None
        valid_pixels = flat_data
        n_valid = n_pixels

    # Fit Scaler on sample of valid pixels to compute global channel means & standard deviations
    scaler = get_scaler(normalization if normalization != "none" else "standard")
    actual_sample_size = min(sample_size, n_valid)
    if actual_sample_size < n_valid:
        rng = np.random.default_rng(random_state)
        sample_indices = rng.choice(n_valid, size=actual_sample_size, replace=False)
        train_sample = valid_pixels[sample_indices]
    else:
        train_sample = valid_pixels

    print(f"[*] Fitting {scaler.__class__.__name__} across channels...")
    scaler.fit(train_sample)

    if hasattr(scaler, "mean_"):
        for ch_idx in range(c):
            print(f"    - Channel {ch_idx+1}: Mean = {scaler.mean_[ch_idx]:.2f}, Std = {scaler.scale_[ch_idx]:.2f}")

    # Standardize & compute argmax in memory-safe chunks
    print(f"[*] Standardizing pixels and calculating dominant channel per pixel...")
    labels_flat = np.full(n_pixels, -1, dtype=np.int32)
    margin_flat = np.zeros(n_pixels, dtype=np.float32)

    if valid_mask is not None:
        indices = np.flatnonzero(valid_mask)
        data_to_process = valid_pixels
    else:
        indices = np.arange(n_pixels)
        data_to_process = flat_data

    chunk_size = 500_000
    n_chunks = int(np.ceil(len(data_to_process) / chunk_size))
    t0 = time.time()

    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(data_to_process))
        chunk = data_to_process[start_idx:end_idx]
        chunk_scaled = scaler.transform(chunk)  # (N_chunk, C)

        # Find top 1 and top 2 channels
        # For small C (e.g. 2 to 30), np.partition or sort is very fast
        if c > 1:
            top_channel = np.argmax(chunk_scaled, axis=1)
            # Sort descending along channel axis to find top 2
            sorted_scaled = np.sort(chunk_scaled, axis=1)[:, ::-1]
            margin = sorted_scaled[:, 0] - sorted_scaled[:, 1]
        else:
            top_channel = np.zeros(len(chunk), dtype=np.int32)
            margin = chunk_scaled[:, 0]

        target_idx = indices[start_idx:end_idx]
        labels_flat[target_idx] = top_channel
        margin_flat[target_idx] = margin

    print(f"[+] Prevalence calculation completed in {time.time() - t0:.2f} seconds.")

    prevalence_map = labels_flat.reshape(h, w)
    margin_map = margin_flat.reshape(h, w)

    # Compute territory profiles
    print("[*] Computing channel dominance statistics...")
    stats_list = []
    for ch_idx in range(c):
        mask = (labels_flat == ch_idx)
        count = int(np.sum(mask))
        pct = 100.0 * count / n_valid if n_valid > 0 else 0.0

        if count > 0:
            dom_pixels_raw = flat_data[mask]
            mean_raw = np.mean(dom_pixels_raw, axis=0)
            std_raw = np.std(dom_pixels_raw, axis=0)
            mean_margin = float(np.mean(margin_flat[mask]))
        else:
            mean_raw = np.zeros(c)
            std_raw = np.zeros(c)
            mean_margin = 0.0

        row = {
            "channel_id": ch_idx,
            "channel_name": f"Channel_{ch_idx + 1}",
            "dominant_pixel_count": count,
            "percentage_of_image": pct,
            "mean_dominance_margin": mean_margin,
            "own_channel_raw_mean": float(mean_raw[ch_idx]),
        }
        for sub_ch in range(c):
            row[f"ch_{sub_ch + 1}_raw_mean"] = float(mean_raw[sub_ch])
            row[f"ch_{sub_ch + 1}_raw_std"] = float(std_raw[sub_ch])

        stats_list.append(row)

    stats_df = pd.DataFrame(stats_list)
    return prevalence_map, margin_map, stats_df


# ==============================================================================
# METHOD 2: KMeans Clustering
# ==============================================================================
def fast_kmeans_clustering(
    data: np.ndarray,
    n_clusters: int = 7,
    sample_size: int = 500_000,
    batch_size: int = 10_000,
    normalization: str = "none",
    mask_background: bool = False,
    bg_threshold: float = 0.0,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Performs fast KMeans clustering on large multi-channel pixel data."""
    h, w, c = data.shape
    n_pixels = h * w
    print(f"[*] Reshaping {n_pixels:,} total pixels for KMeans (k={n_clusters})...")

    flat_data = data.reshape(-1, c)

    if mask_background:
        pixel_intensities = np.sum(np.abs(flat_data), axis=1)
        valid_mask = pixel_intensities > bg_threshold
        valid_pixels = flat_data[valid_mask]
        n_valid = len(valid_pixels)
        print(f"[*] Background masking active: {n_valid:,} / {n_pixels:,} ({100*n_valid/n_pixels:.1f}%) non-background pixels.")
        if n_valid == 0:
            raise ValueError("All pixels fell below the background threshold. Lower bg_threshold.")
        train_pool = valid_pixels
    else:
        valid_mask = None
        train_pool = flat_data
        n_valid = n_pixels

    scaler = get_scaler(normalization)
    actual_sample_size = min(sample_size, n_valid)
    if actual_sample_size < n_valid:
        print(f"[*] Sampling {actual_sample_size:,} random pixels for fast KMeans centroid fitting...")
        rng = np.random.default_rng(random_state)
        sample_indices = rng.choice(n_valid, size=actual_sample_size, replace=False)
        train_samples = train_pool[sample_indices]
    else:
        train_samples = train_pool

    if scaler is not None:
        train_samples_scaled = scaler.fit_transform(train_samples)
    else:
        train_samples_scaled = train_samples

    t0 = time.time()
    print(f"[*] Fitting KMeans (k={n_clusters}) on training sample...")
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        random_state=random_state,
        n_init=3,
        max_iter=300,
    )
    kmeans.fit(train_samples_scaled)
    print(f"[+] KMeans converged in {time.time() - t0:.2f} seconds.")

    print(f"[*] Predicting labels for all {n_pixels:,} pixels...")
    labels_flat = np.full(n_pixels, -1, dtype=np.int32)

    if valid_mask is not None:
        indices_to_predict = np.flatnonzero(valid_mask)
        data_to_predict = flat_data[valid_mask]
    else:
        indices_to_predict = np.arange(n_pixels)
        data_to_predict = flat_data

    chunk_size = 500_000
    n_chunks = int(np.ceil(len(data_to_predict) / chunk_size))
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(data_to_predict))
        chunk = data_to_predict[start_idx:end_idx]
        if scaler is not None:
            chunk = scaler.transform(chunk)
        pred = kmeans.predict(chunk)
        labels_flat[indices_to_predict[start_idx:end_idx]] = pred

    cluster_map = labels_flat.reshape(h, w)

    print("[*] Computing cluster profiles & channel dominance metrics...")
    cluster_stats = []
    cluster_centers_raw = np.zeros((n_clusters, c), dtype=np.float32)

    for k in range(n_clusters):
        k_mask = (labels_flat == k)
        count = int(np.sum(k_mask))
        pct = 100.0 * count / (n_valid if valid_mask is not None else n_pixels)

        if count > 0:
            k_pixels = flat_data[k_mask]
            mean_c = np.mean(k_pixels, axis=0)
            std_c = np.std(k_pixels, axis=0)
        else:
            mean_c = np.zeros(c)
            std_c = np.zeros(c)

        cluster_centers_raw[k] = mean_c
        dominant_ch = int(np.argmax(mean_c))

        row = {
            "cluster_id": k,
            "pixel_count": count,
            "percentage": pct,
            "dominant_channel": f"Ch_{dominant_ch + 1}",
            "dominant_channel_mean": float(mean_c[dominant_ch]),
        }
        for ch_idx in range(c):
            row[f"ch_{ch_idx + 1}_mean"] = float(mean_c[ch_idx])
            row[f"ch_{ch_idx + 1}_std"] = float(std_c[ch_idx])
        cluster_stats.append(row)

    stats_df = pd.DataFrame(cluster_stats)
    return cluster_map, cluster_centers_raw, stats_df


# ==============================================================================
# OUTPUT EXPORT & MEMORY-SAFE VISUALIZATION
# ==============================================================================
def save_outputs(
    segmentation_map: np.ndarray,
    raw_data: np.ndarray,
    stats_df: pd.DataFrame,
    output_dir: Path,
    base_name: str,
    method: str = "prevalence",
    margin_map: Optional[np.ndarray] = None,
    channel_names: Optional[List[str]] = None,
    save_summary_plot: bool = True,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    h, w, c = raw_data.shape
    n_categories = len(stats_df)
    palette = get_palette(n_categories)

    if channel_names is None or len(channel_names) != c:
        channel_names = [f"Channel {i+1}" for i in range(c)]

    prefix = f"{base_name}_{method}"

    # 1. Export 2D Label TIFF (uint8/uint16)
    label_tif_path = output_dir / f"{prefix}_labels.tif"
    if n_categories < 255:
        out_labels = np.where(segmentation_map == -1, 255, segmentation_map).astype(np.uint8)
    else:
        out_labels = segmentation_map.astype(np.uint16)

    print(f"[*] Writing full-resolution label TIFF ({w}x{h}) to: {label_tif_path}")
    tifffile.imwrite(
        str(label_tif_path),
        out_labels,
        compression="zlib",
        metadata={"axes": "YX", "Description": f"{method} segmentation map ({n_categories} classes)"},
    )

    # 2. Export Colormapped RGB TIFF
    rgb_tif_path = output_dir / f"{prefix}_rgb.tif"
    print(f"[*] Writing colored RGB TIFF ({w}x{h}x3) to: {rgb_tif_path}")
    rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
    for k in range(n_categories):
        rgb_img[segmentation_map == k] = palette[k]

    tifffile.imwrite(
        str(rgb_tif_path),
        rgb_img,
        compression="zlib",
        metadata={"axes": "YXS", "Description": f"{method} colored segmentation map"},
    )

    # 3. Export Margin TIFF if available
    if margin_map is not None:
        margin_tif_path = output_dir / f"{prefix}_margin.tif"
        print(f"[*] Writing dominance margin / confidence TIFF to: {margin_tif_path}")
        tifffile.imwrite(
            str(margin_tif_path),
            margin_map.astype(np.float32),
            compression="zlib",
            metadata={"axes": "YX", "Description": "Dominance margin (Top Z - Second Top Z)"},
        )

    # 4. Export CSV Statistics
    csv_path = output_dir / f"{prefix}_profiles.csv"
    print(f"[*] Writing statistics CSV to: {csv_path}")
    stats_df.to_csv(csv_path, index=False)

    # 5. Publication Summary Figure (with downsampled previews for memory safety)
    if save_summary_plot:
        plot_path = output_dir / f"{prefix}_summary.png"
        print(f"[*] Generating memory-safe publication summary figure to: {plot_path}")
        generate_summary_figure(
            raw_data=raw_data,
            segmentation_map=segmentation_map,
            stats_df=stats_df,
            palette=palette,
            channel_names=channel_names,
            output_path=plot_path,
            base_name=base_name,
            method=method,
            margin_map=margin_map,
        )


def generate_summary_figure(
    raw_data: np.ndarray,
    segmentation_map: np.ndarray,
    stats_df: pd.DataFrame,
    palette: np.ndarray,
    channel_names: List[str],
    output_path: Path,
    base_name: str,
    method: str = "prevalence",
    margin_map: Optional[np.ndarray] = None,
):
    """
    Creates a multi-panel visualization. Downsamples large arrays during display to prevent OOM.
    """
    h, w, c = raw_data.shape
    n_categories = len(stats_df)

    # Downsample thumbnails for matplotlib
    max_preview_dim = 1600
    sub_raw = downsample_for_preview(raw_data, max_dim=max_preview_dim)
    sub_seg = downsample_for_preview(segmentation_map, max_dim=max_preview_dim)
    sub_h, sub_w = sub_seg.shape

    # Build RGB preview thumbnail
    sub_rgb = np.zeros((sub_h, sub_w, 3), dtype=np.uint8)
    for k in range(n_categories):
        sub_rgb[sub_seg == k] = palette[k]

    n_ch_to_show = min(c, 4)
    total_cols = max(3, n_ch_to_show)

    fig = plt.figure(figsize=(6 * total_cols, 14), dpi=150)
    gs = fig.add_gridspec(3, total_cols, height_ratios=[1.1, 1.2, 1.1], hspace=0.32, wspace=0.25)

    # --- ROW 1: Raw Channel Previews ---
    for ch_idx in range(n_ch_to_show):
        ax = fig.add_subplot(gs[0, ch_idx])
        ch_img = sub_raw[:, :, ch_idx]
        p1, p99 = np.percentile(ch_img, (1, 99))
        vmax = p99 if p99 > p1 else p1 + 1
        im = ax.imshow(ch_img, cmap="magma", vmin=p1, vmax=vmax)
        ax.set_title(f"{channel_names[ch_idx]}\n(1st-99th pct: [{p1:.0f}, {vmax:.0f}])", fontsize=10, fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Fill remaining row 1 slots if any
    for extra_col in range(n_ch_to_show, total_cols):
        ax = fig.add_subplot(gs[0, extra_col])
        ax.axis("off")

    # --- ROW 2: Segmentation Map & Proportions ---
    ax_map = fig.add_subplot(gs[1, :total_cols - 1])
    ax_map.imshow(sub_rgb)
    if method == "prevalence":
        map_title = f"Channel Prevalence Map: Which Channel Dominates (Highest Z-Score)"
    else:
        map_title = f"KMeans Segmentation Map (k={n_categories})"
    ax_map.set_title(map_title, fontsize=12, fontweight="bold")
    ax_map.axis("off")

    # Pie breakdown
    ax_pie = fig.add_subplot(gs[1, total_cols - 1])
    pie_colors = [palette[k] / 255.0 for k in range(n_categories)]
    pct_key = "percentage_of_image" if "percentage_of_image" in stats_df.columns else "percentage"
    labels = [
        f"{channel_names[k] if method == 'prevalence' else f'C{k}'}"
        for k in range(n_categories)
    ]
    ax_pie.pie(
        stats_df[pct_key],
        labels=labels,
        autopct="%1.1f%%",
        colors=pie_colors,
        startangle=140,
        textprops={"fontsize": 8},
    )
    ax_pie.set_title("Territory Breakdown (% Area)", fontsize=10, fontweight="bold")

    # --- ROW 3: Channel Profiles ---
    # 3A: Absolute Mean Raw Intensities in each territory
    ax_bar = fig.add_subplot(gs[2, :max(1, total_cols // 2)])
    x = np.arange(n_categories)
    width = 0.8 / max(1, c)

    for ch_idx in range(c):
        mean_col = f"ch_{ch_idx + 1}_raw_mean" if f"ch_{ch_idx + 1}_raw_mean" in stats_df.columns else f"ch_{ch_idx + 1}_mean"
        std_col = f"ch_{ch_idx + 1}_raw_std" if f"ch_{ch_idx + 1}_raw_std" in stats_df.columns else f"ch_{ch_idx + 1}_std"
        ch_means = stats_df[mean_col].values
        ch_stds = stats_df[std_col].values if std_col in stats_df.columns else np.zeros_like(ch_means)
        offset = (ch_idx - (c - 1) / 2) * width
        ax_bar.bar(
            x + offset,
            ch_means,
            width=width,
            yerr=ch_stds,
            capsize=2,
            label=channel_names[ch_idx],
            alpha=0.85,
        )

    ax_bar.set_xticks(x)
    x_labels = [channel_names[k] if method == "prevalence" else f"C{k}" for k in range(n_categories)]
    ax_bar.set_xticklabels(x_labels, fontsize=8, fontweight="bold", rotation=25)
    ax_bar.set_ylabel("Mean Raw Intensity ± SD", fontsize=9)
    ax_bar.set_title("Mean Channel Intensities in Each Territory", fontsize=10, fontweight="bold")
    ax_bar.legend(fontsize=7, loc="upper right")
    ax_bar.grid(axis="y", linestyle="--", alpha=0.4)

    # 3B: Relative Normalized / Margin Plot
    ax_right = fig.add_subplot(gs[2, max(1, total_cols // 2):])
    if method == "prevalence" and "mean_dominance_margin" in stats_df.columns:
        # Plot dominance margin (confidence in Z-score units)
        bar_colors = [palette[k] / 255.0 for k in range(n_categories)]
        bars = ax_right.bar(x, stats_df["mean_dominance_margin"], color=bar_colors, edgecolor="black", alpha=0.85)
        ax_right.set_xticks(x)
        ax_right.set_xticklabels(x_labels, fontsize=8, fontweight="bold", rotation=25)
        ax_right.set_ylabel("Mean Dominance Margin (Z-score Δ)", fontsize=9)
        ax_right.set_title("Dominance Strength (Top Z - 2nd Top Z)", fontsize=10, fontweight="bold")
        ax_right.grid(axis="y", linestyle="--", alpha=0.4)
    else:
        # Relative % share
        mean_cols = [f"ch_{i + 1}_mean" for i in range(c)]
        mean_matrix = np.array([stats_df[col].values for col in mean_cols]).T
        row_sums = np.sum(mean_matrix, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        rel_pct = 100.0 * mean_matrix / row_sums

        bottom = np.zeros(n_categories)
        for ch_idx in range(c):
            ax_right.bar(x, rel_pct[:, ch_idx], bottom=bottom, label=f"{channel_names[ch_idx]} (%)", alpha=0.9)
            bottom += rel_pct[:, ch_idx]
        ax_right.set_xticks(x)
        ax_right.set_xticklabels(x_labels, fontsize=8, fontweight="bold", rotation=25)
        ax_right.set_ylabel("Relative Share (%)", fontsize=9)
        ax_right.set_ylim(0, 100)
        ax_right.set_title("Relative Channel Proportions", fontsize=10, fontweight="bold")
        ax_right.legend(fontsize=7, loc="lower right")

    fig.suptitle(f"Multi-Channel Analysis ({method.upper()}): {base_name}", fontsize=14, fontweight="bold", y=0.98)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[+] Figure successfully generated: {output_path}")


# ==============================================================================
# MAIN CLI
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Multi-channel TIFF Segmentation: Channel Prevalence (Argmax of Z-score) or KMeans Clustering."
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default=r"C:\Users\jvill\OneDrive - UC San Diego\integrated_pipeline\WSI_data\SRS\channels\K21_273_3.tif",
        help="Path to input multi-channel TIFF image.",
    )
    parser.add_argument(
        "-m", "--method",
        type=str,
        choices=["prevalence", "kmeans"],
        default="prevalence",
        help="Segmentation method: 'prevalence' (split by highest standardized channel, default) or 'kmeans'.",
    )
    parser.add_argument(
        "-k", "--clusters",
        type=int,
        default=7,
        help="Number of clusters if using --method kmeans (default: 7).",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Directory to save output TIFFs and plots.",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        choices=["standard", "robust", "minmax", "none"],
        default="standard",
        help="Feature scaling before computing prevalence / clustering (default: 'standard' z-score).",
    )
    parser.add_argument(
        "--mask-bg",
        action="store_true",
        help="Ignore dark/empty background pixels during segmentation.",
    )
    parser.add_argument(
        "--bg-thresh",
        type=float,
        default=0.0,
        help="Background threshold intensity when --mask-bg is set.",
    )
    parser.add_argument(
        "--channel-names",
        nargs="+",
        default=None,
        help="Optional custom channel names (e.g. --channel-names Lipids Proteins CH2).",
    )
    parser.add_argument(
        "--use-memmap",
        action="store_true",
        help="Attempt memory mapping for reading large TIFFs.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[!] Error: File does not exist at {input_path}")
        sys.exit(1)

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = input_path.parent / f"{input_path.stem}_{args.method}_output"

    base_name = input_path.stem

    print("=" * 72)
    print(f" Multi-Channel TIFF Segmentation Tool")
    print(f" Method:        {args.method.upper()}")
    print(f" Normalization: {args.normalization}")
    print(f" Input File:    {input_path}")
    print(f" Output Folder: {out_dir}")
    print("=" * 72)

    # 1. Load Data
    data, (h, w, c), fmt = load_multichannel_tif(input_path, use_memmap=args.use_memmap)

    # 2. Segment
    t_start = time.time()
    margin_map = None

    if args.method == "prevalence":
        print(f"[*] Splitting image by channel prevalence (argmax of {args.normalization} standardized channels across {c} channels)...")
        seg_map, margin_map, stats_df = channel_prevalence_segmentation(
            data=data,
            normalization=args.normalization,
            mask_background=args.mask_bg,
            bg_threshold=args.bg_thresh,
        )
    else:
        print(f"[*] Clustering image using KMeans (k={args.clusters})...")
        seg_map, cluster_centers, stats_df = fast_kmeans_clustering(
            data=data,
            n_clusters=args.clusters,
            normalization=args.normalization,
            mask_background=args.mask_bg,
            bg_threshold=args.bg_thresh,
        )

    # 3. Export
    save_outputs(
        segmentation_map=seg_map,
        raw_data=data,
        stats_df=stats_df,
        output_dir=out_dir,
        base_name=base_name,
        method=args.method,
        margin_map=margin_map,
        channel_names=args.channel_names,
        save_summary_plot=True,
    )

    total_time = time.time() - t_start
    print("=" * 72)
    print(f"[+] COMPLETE in {total_time:.2f}s!")
    print(f"[+] Output Directory: {out_dir}")
    print(f"    - Label TIFF:       {base_name}_{args.method}_labels.tif")
    print(f"    - Colored RGB TIFF: {base_name}_{args.method}_rgb.tif")
    if margin_map is not None:
        print(f"    - Margin TIFF:      {base_name}_{args.method}_margin.tif")
    print(f"    - Statistics CSV:   {base_name}_{args.method}_profiles.csv")
    print(f"    - Summary Plot:     {base_name}_{args.method}_summary.png")
    print("=" * 72)
    print("\nTerritory Summary Statistics:")
    print(stats_df.to_string(index=False))


if __name__ == "__main__":
    main()
