"""
========================================================================================
Group Statistics & Publication-Quality Visualization Pipeline
========================================================================================
Overhauled Non-Parametric Engine (Kruskal-Wallis, Dunn's + BH FDR, Cliff's Delta, Van Elteren)
Specifically architected for non-Gaussian distributions, large observation counts, and low biological replicates.

Features:
- Flexible Multi-Column Metric selection (e.g. % Lipids, Turnover, Ratios, Intensity, etc.)
- Multi-Level Grouping (1, 2, or more grouping columns, combine/facet/hue modes)
- Non-Parametric Statistical Testing:
    * Kruskal-Wallis H-Test (variation in group medians, η²_H effect size)
    * Dunn's Post-Hoc Test (exact Kruskal-Wallis pooled rank variance with tie adjustments)
    * Benjamini-Hochberg (BH) False Discovery Rate (FDR) multiplicity correction (q-values)
    * Cliff's Delta (δ) Non-Parametric Effect Size & Romano et al. (2006) 95% CIs
    * Van Elteren Stratified Wilcoxon across biological replicates (Sample ID / tissue_id)
    * Biological Concordance Validation (identifying potential pseudoreplication)
    * Dual Significance Filtering (q_BH < 0.05 AND |δ| >= 0.147)
- Rich Visualizations:
    * Violin Plots (with inner quartile/box, overlayed strip plot, median point + glow labels)
    * Bar Plots (with group medians or means +/- IQR / SEM error bars, strip points)
    * Automatic Significance Brackets with multi-level collision avoidance
    * Standalone Legend generation for sample IDs / hue categories
- Data Formats: Parquet (.parquet), Feather (.feather), CSV (.csv), Excel (.xlsx)
========================================================================================
"""

import os
import re
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as PathEffects
import seaborn as sns

# Add parent directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.nonparametric_stats import (
    compute_cliffs_delta,
    run_kruskal_wallis,
    run_dunns_posthoc_fdr,
    run_stratified_sample_wilcoxon,
    compute_nonparametric_descriptives,
    p_to_stars,
    format_p_value,
    interpret_cliffs_delta
)

# Optional pingouin import for legacy parametric fallback
try:
    import pingouin as pg
except ImportError:
    pg = None

# -----------------------------------------------------------------------------
# PROJECT-AGNOSTIC DEFAULT CONFIGURATION
# -----------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = "group_stats_output"

# Statistical Mode: "nonparametric" (default) or "parametric"
STATS_MODE = "nonparametric"

# Plotting parameters
PLOT_TYPE = "both"             # "violin", "bar", or "both"
COMBINE_TWO_GROUPS = False     # True: merge 2 group cols into one combined category (e.g. "GroupA | GroupB")
                               # False: facet by 2nd group column with aligned shared axes
ALPHA = 0.05                   # Significance threshold for post-hoc tests and brackets
MAX_BRACKETS = 6               # Cap on brackets per subplot to prevent clutter (None = show all)
MIN_EFFECT_SIZE_BRACKET = 0.147  # Only draw brackets if |Cliff's δ| >= 0.147 (filters out negligible shifts on massive N)
ERROR_BAR_TYPE = "iqr"         # For bar plots: "iqr" (median +/- IQR), "sem", or "sd"
CENTRAL_TENDENCY = "median"    # "median" (recommended for non-parametric) or "mean"

PALETTE = ["#e85b1e", "#f19d78", "#37498b", "#a7b4e3"]        # Base palette; dynamically scales if groups > len(palette)
STRIP_HUE_COL = None           # Replicate column (auto-detected via canonical aliases if None)
STRIP_PALETTE = "husl"        # Palette for strip plot dots
SAVE_STRIP_LEGEND = True       # Save separate clean PNG legend for strip dots
FONT_SCALE = 1.5               # Global font scaling factor for publication figures
DPI = 300                      # Output resolution for saved figures
TRANSPARENT_BG = True          # Save figures with transparent background

# Canonical column aliases for zero-touch schema normalization
SAMPLE_ID_ALIASES = [
    "Sample_ID", "Sample ID", "sample_id", "sample", "Sample", 
    "Animal_ID", "Animal ID", "animal_id", "Subject_ID", "Subject ID"
]
IMAGE_NAME_ALIASES = [
    "Image_Name", "Image Name", "image_name", "image", "Image", 
    "Tile_Name", "tile_name", "tile"
]
ROI_NUMBER_ALIASES = [
    "ROI_Number", "ROI #", "ROI_Num", "roi_number", "roi_num", "ROI"
]
ROI_INSTANCE_ALIASES = [
    "ROI_instance", "ROI Inst", "roi_instance", "roi_inst"
]
AREA_ALIASES = [
    "Area_px", "Area (px)", "area_px", "area", "Area"
]
NON_METRIC_COLS = {
    "Centroid_X", "Centroid_Y", "ROI_Number", "ROI #", "ROI_Num", "roi_number",
    "ROI_instance", "ROI Inst", "roi_instance", "slice_index", 
    "Image_Width", "Image_Height", "Pixel_Size_um", "Area_px", "Area (px)",
    "UID", "Image_Path", "Vertices_JSON"
}


# -----------------------------------------------------------------------------
# LEGACY PARAMETRIC SUPPORT (Welch / Games-Howell / Cohen's d)
# -----------------------------------------------------------------------------

def compute_cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> Tuple[float, float]:
    """Computes Cohen's d and Hedges' g for parametric mode."""
    a = group_a[~np.isnan(group_a)]
    b = group_b[~np.isnan(group_b)]
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan
    
    mean1, mean2 = np.mean(a), np.mean(b)
    var1, var2 = np.var(a, ddof=1), np.var(b, ddof=1)
    
    s_pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / max(1, (n1 + n2 - 2)))
    if s_pooled == 0 or np.isnan(s_pooled):
        return 0.0, 0.0
    
    d = (mean1 - mean2) / s_pooled
    j = 1.0 - (3.0 / max(1.0, (4.0 * (n1 + n2) - 9.0)))
    g = d * j
    return float(d), float(g)


def run_welch_games_howell(
    data: pd.DataFrame, 
    value_col: str, 
    group_col: str, 
    alpha: float = ALPHA,
    ignore_zeros: bool = False
) -> Tuple[Optional[Dict[str, Any]], Optional[pd.DataFrame]]:
    """Legacy Welch's ANOVA and Games-Howell post-hoc test."""
    if pg is None:
        raise ImportError("pingouin is required for parametric Welch/Games-Howell mode.")
        
    sub = data[[value_col, group_col]].dropna()
    sub = sub[np.isfinite(sub[value_col])]
    if ignore_zeros:
        sub = sub[sub[value_col] != 0]
    counts = sub[group_col].value_counts()
    valid_groups = counts[counts >= 2].index
    sub = sub[sub[group_col].isin(valid_groups)]
    
    k = sub[group_col].nunique()
    if k < 2:
        return None, None
    
    try:
        aov = pg.welch_anova(dv=value_col, between=group_col, data=sub)
        row = aov.iloc[0]
        anova_dict = {
            "F": float(row["F"]),
            "df1": float(row["ddof1"]),
            "df2": float(row["ddof2"]),
            "p_unc": float(row["p_unc"]),
            "k_groups": k,
            "N_total": len(sub),
            "np2": float(row.get("np2", np.nan))
        }
    except Exception as e:
        print(f"  [Warning] Welch's ANOVA failed for '{value_col}' on '{group_col}': {e}")
        return None, None
    
    posthoc_df = None
    try:
        posthoc_df = pg.pairwise_gameshowell(dv=value_col, between=group_col, data=sub)
        if posthoc_df is not None and not posthoc_df.empty:
            cohens_d_list = []
            hedges_g_list = []
            pct_diff_list = []
            effect_interp_list = []
            n_a_list = []
            n_b_list = []
            
            for _, r in posthoc_df.iterrows():
                val_a = sub.loc[sub[group_col] == r['A'], value_col].values
                val_b = sub.loc[sub[group_col] == r['B'], value_col].values
                d, g = compute_cohens_d(val_a, val_b)
                
                mean_a = float(r.get('mean(A)', np.mean(val_a)))
                mean_b = float(r.get('mean(B)', np.mean(val_b)))
                pct_diff = ((mean_a - mean_b) / mean_b * 100.0) if mean_b != 0 else np.nan
                
                abs_d = abs(d) if not np.isnan(d) else 0
                interp = "Negligible" if abs_d < 0.2 else ("Small" if abs_d < 0.5 else ("Medium" if abs_d < 0.8 else ("Large" if abs_d < 1.2 else "Very Large")))
                
                cohens_d_list.append(d)
                hedges_g_list.append(g)
                pct_diff_list.append(pct_diff)
                effect_interp_list.append(interp)
                n_a_list.append(len(val_a))
                n_b_list.append(len(val_b))
                
            posthoc_df['N(A)'] = n_a_list
            posthoc_df['N(B)'] = n_b_list
            posthoc_df['cohen_d'] = cohens_d_list
            posthoc_df['hedges_g'] = hedges_g_list
            posthoc_df['pct_diff'] = pct_diff_list
            posthoc_df['effect_size_magnitude'] = effect_interp_list
            posthoc_df['significance'] = posthoc_df['pval'].apply(p_to_stars)
            posthoc_df['is_significant'] = posthoc_df['pval'] < alpha
    except Exception as e:
        print(f"  [Warning] Games-Howell post-hoc failed for '{value_col}': {e}")
        posthoc_df = None
        
    return anova_dict, posthoc_df


# -----------------------------------------------------------------------------
# PLOTTING UTILITIES
# -----------------------------------------------------------------------------

def build_palette_dict(order: List[Any], palette: Union[str, List, Dict]) -> Dict[Any, Any]:
    """Constructs a categorical color mapping dictionary for groups."""
    if isinstance(palette, dict):
        return {cat: palette.get(cat, "#4C72B0") for cat in order}
    elif isinstance(palette, (list, tuple)):
        if len(order) > len(palette):
            # Scale up palette when category count exceeds list length to avoid repeated colors
            colors = sns.color_palette("tab20" if len(order) <= 20 else "husl", len(order))
            return {cat: colors[i] for i, cat in enumerate(order)}
        return {cat: palette[i % len(palette)] for i, cat in enumerate(order)}
    elif isinstance(palette, str):
        colors = sns.color_palette(palette, len(order))
        return {cat: colors[i] for i, cat in enumerate(order)}
    return {cat: "#4C72B0" for cat in order}


def save_strip_legend(
    order: List[Any], 
    palette_dict: Dict[Any, Any], 
    title: str, 
    output_dir: str
) -> str:
    """Generates and saves a clean, separate, publication-ready legend figure."""
    n_items = len(order)
    ncols = min(8, max(1, int(np.ceil(np.sqrt(n_items)))))
    nrows = int(np.ceil(n_items / ncols))
    
    fig, ax = plt.subplots(figsize=(max(3.5, 1.3 * ncols), max(1.2, 0.45 * nrows + 0.6)))
    handles = [
        plt.Line2D(
            [0], [0], marker='o', color='w', 
            markerfacecolor=palette_dict.get(val, '#555555'),
            markeredgecolor='black', markeredgewidth=0.8, markersize=10, 
            label=f"Sample {val}" if isinstance(val, (int, np.integer)) and "sample" not in str(title).lower() else str(val)
        )
        for val in order
    ]
    ax.legend(
        handles=handles, title=title, loc='center', ncol=ncols,
        frameon=True, fontsize=10, title_fontsize=11, framealpha=0.9,
        edgecolor='#cccccc'
    )
    ax.axis('off')
    fig.tight_layout()
    
    clean_title = re.sub(r"[^\w\s]", "", str(title)).strip().replace(" ", "_")
    legend_filename = f"legend_{clean_title}.png"
    os.makedirs(output_dir, exist_ok=True)
    legend_path = os.path.normpath(os.path.join(output_dir, legend_filename))
    try:
        fig.savefig(legend_path, dpi=DPI, bbox_inches='tight', transparent=TRANSPARENT_BG)
        print(f"Saved strip hue legend to: {legend_path}")
    except Exception as e:
        print(f"  [Warning] Could not save legend image to '{legend_path}': {e}")
    finally:
        plt.close(fig)
    return legend_path


def add_significance_brackets(
    ax: plt.Axes, 
    sig_pairs: List[Tuple[Any, Any, float, float]], 
    order: List[Any], 
    initial_ymin: Optional[float] = None, 
    initial_ymax: Optional[float] = None,
    max_brackets: Optional[int] = MAX_BRACKETS,
    font_scale: float = FONT_SCALE
) -> int:
    """
    Renders non-overlapping significance brackets above categorical groups.
    sig_pairs: list of tuples (group_a, group_b, p_or_q_value, effect_size)
    """
    if not sig_pairs:
        return 0

    ymin = initial_ymin if initial_ymin is not None else ax.get_ylim()[0]
    ymax = initial_ymax if initial_ymax is not None else ax.get_ylim()[1]
    y_range = max(ymax - ymin, 1e-6)
    y_start = ymax + 0.04 * y_range
    h = 0.07 * y_range

    valid_pairs = []
    for item in sig_pairs:
        g1, g2 = item[0], item[1]
        if g1 in order and g2 in order and g1 != g2:
            valid_pairs.append(item)
            
    valid_pairs = sorted(valid_pairs, key=lambda t: abs(order.index(t[0]) - order.index(t[1])))
    if max_brackets is not None and max_brackets > 0:
        valid_pairs = valid_pairs[:max_brackets]

    levels_occupied = []
    max_level_used = 0
    star_fs = max(8, int(10 * font_scale))

    for item in valid_pairs:
        g1, g2, p = item[0], item[1], item[2]
        x1, x2 = order.index(g1), order.index(g2)
        left, right = min(x1, x2), max(x1, x2)

        assigned_level = 0
        while True:
            if assigned_level >= len(levels_occupied):
                levels_occupied.append([])
            overlap = False
            for existing_left, existing_right in levels_occupied[assigned_level]:
                if existing_left <= right and left <= existing_right:
                    overlap = True
                    break
            if not overlap:
                levels_occupied[assigned_level].append((left, right))
                break
            assigned_level += 1

        y = y_start + assigned_level * (h * 1.45)
        ax.plot([x1, x1, x2, x2], [y, y + h * 0.3, y + h * 0.3, y], lw=1.2, c="black", clip_on=False)
        ax.text(
            (x1 + x2) / 2.0, y + h * 0.32, p_to_stars(p), 
            ha="center", va="bottom", fontsize=star_fs, fontweight='bold', c="black"
        )
        max_level_used = max(max_level_used, assigned_level + 1)

    if initial_ymin is None and initial_ymax is None and max_level_used > 0:
        ax.set_ylim(ymin, y_start + max_level_used * (h * 1.45) + h * 1.5)

    return max_level_used


# -----------------------------------------------------------------------------
# CORE SUBPLOT PLOTTING (VIOLIN & BAR)
# -----------------------------------------------------------------------------

def plot_single_group_axes(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    group_order: List[Any],
    plot_type: str = "violin",
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    alpha: float = ALPHA,
    min_effect_size_bracket: float = MIN_EFFECT_SIZE_BRACKET,
    stats_mode: str = STATS_MODE,
    central_tendency: str = CENTRAL_TENDENCY,
    palette: Union[str, List, Dict] = PALETTE,
    add_brackets: bool = True,
    strip_hue_col: Optional[str] = STRIP_HUE_COL,
    strip_order: Optional[List[Any]] = None,
    strip_palette_dict: Optional[Dict[Any, Any]] = None,
    error_bar_type: str = ERROR_BAR_TYPE,
    font_scale: float = FONT_SCALE,
    ignore_zeros: bool = False,
    show_points: bool = True,
    max_points: Optional[int] = None
) -> Tuple[Optional[Dict[str, Any]], Optional[pd.DataFrame]]:
    """
    Renders either a Violin plot or a Bar plot with median points, strip dots,
    and non-parametric statistical testing on a given matplotlib axes.
    """
    cols_to_keep = [value_col, group_col]
    if strip_hue_col and strip_hue_col in df.columns and strip_hue_col not in cols_to_keep:
        cols_to_keep.append(strip_hue_col)
        
    sub = df[cols_to_keep].dropna(subset=[value_col, group_col])
    sub = sub[np.isfinite(sub[value_col])]
    if ignore_zeros:
        sub = sub[sub[value_col] != 0]

    order = [cat for cat in group_order if cat in sub[group_col].unique()]
    if not order:
        order = list(sub[group_col].dropna().unique())
        
    title_fs = max(10, int(13 * font_scale))
    label_fs = max(9, int(11 * font_scale))
    tick_fs  = max(7, int(9 * font_scale))
    text_fs  = max(8, int(9.5 * font_scale))
    
    palette_dict = build_palette_dict(order, palette)
    
    if ax is None:
        fig_w = max(4.5, 1.4 * len(order))
        fig_h = 5.5
        _, ax = plt.subplots(figsize=(fig_w, fig_h))

    if len(order) == 0 or len(sub) == 0:
        ax.set_title(f"{title or value_col}\n(No data)", fontsize=title_fs)
        return None, None

    # 1. Base Plot (Violin or Bar)
    if plot_type.lower() == "bar":
        if central_tendency.lower() == "median":
            heights = [float(sub.loc[sub[group_col] == cat, value_col].median()) for cat in order]
            if error_bar_type.lower() == "iqr":
                # Asymmetric error bars: [median - Q25, Q75 - median]
                lower_err = [
                    max(0, float(sub.loc[sub[group_col] == cat, value_col].median() - np.percentile(sub.loc[sub[group_col] == cat, value_col].dropna(), 25)))
                    for cat in order
                ]
                upper_err = [
                    max(0, float(np.percentile(sub.loc[sub[group_col] == cat, value_col].dropna(), 75) - sub.loc[sub[group_col] == cat, value_col].median()))
                    for cat in order
                ]
                errors = [lower_err, upper_err]
            elif error_bar_type.lower() == "sem":
                errors = [
                    sub.loc[sub[group_col] == cat, value_col].std() / np.sqrt(max(1, len(sub.loc[sub[group_col] == cat, value_col])))
                    for cat in order
                ]
            else: # SD
                errors = [sub.loc[sub[group_col] == cat, value_col].std() for cat in order]
        else: # Mean
            heights = [sub.loc[sub[group_col] == cat, value_col].mean() for cat in order]
            if error_bar_type.lower() == "sem":
                errors = [
                    sub.loc[sub[group_col] == cat, value_col].std() / np.sqrt(max(1, len(sub.loc[sub[group_col] == cat, value_col])))
                    for cat in order
                ]
            elif error_bar_type.lower() == "sd":
                errors = [sub.loc[sub[group_col] == cat, value_col].std() for cat in order]
            else: # 95% CI
                errors = [
                    1.96 * sub.loc[sub[group_col] == cat, value_col].std() / np.sqrt(max(1, len(sub.loc[sub[group_col] == cat, value_col])))
                    for cat in order
                ]
            
        bar_colors = [palette_dict[cat] for cat in order]
        x_indices = np.arange(len(order))
        ax.bar(
            x_indices, heights, yerr=errors, capsize=4, 
            color=bar_colors, edgecolor='black', linewidth=1.1, alpha=0.82, zorder=1
        )
        ax.set_xticks(x_indices)
        ax.set_xticklabels(order)
    else:
        # Violin plot with inner box/quartiles
        sns.violinplot(
            data=sub, x=group_col, y=value_col, order=order,
            hue=group_col, hue_order=order, palette=palette_dict,
            cut=0, inner="box", ax=ax, dodge=False, legend=False,
            linewidth=1.2, zorder=1
        )

    # 2. Overlay Strip Plot (points)
    if show_points and (max_points is None or max_points > 0):
        strip_sub = sub
        if max_points is not None and max_points > 0 and len(strip_sub) > max_points:
            per_cat_limit = max(1, max_points // max(1, len(order)))
            sample_chunks = []
            for cat in order:
                cat_df = strip_sub[strip_sub[group_col] == cat]
                if len(cat_df) > per_cat_limit:
                    sample_chunks.append(cat_df.sample(n=per_cat_limit, random_state=42))
                else:
                    sample_chunks.append(cat_df)
            strip_sub = pd.concat(sample_chunks, ignore_index=True) if sample_chunks else strip_sub

        if strip_hue_col and strip_hue_col in strip_sub.columns:
            s_order = strip_order if strip_order is not None else sorted(strip_sub[strip_hue_col].dropna().unique())
            s_pal = strip_palette_dict if strip_palette_dict is not None else build_palette_dict(s_order, STRIP_PALETTE)
            sns.stripplot(
                data=strip_sub, x=group_col, y=value_col, order=order,
                hue=strip_hue_col, hue_order=s_order, palette=s_pal,
                dodge=False, jitter=0.22, size=4, alpha=0.75,
                edgecolor="black", linewidth=0.5, ax=ax, legend=False, zorder=3
            )
        else:
            sns.stripplot(
                data=strip_sub, x=group_col, y=value_col, order=order,
                hue=group_col, hue_order=order, palette=palette_dict,
                dodge=False, jitter=0.22, size=4, alpha=0.6,
                edgecolor="black", linewidth=0.5, ax=ax, legend=False, zorder=3
            )

    # 3. Central Tendency Marker & Text Overlay (Median by default)
    ymin, ymax = ax.get_ylim()
    y_span = max(ymax - ymin, 1e-6)
    y_offset = 0.025 * y_span
    
    for idx, category in enumerate(order):
        cat_data = sub.loc[sub[group_col] == category, value_col]
        if len(cat_data) > 0:
            rep_val = cat_data.median() if central_tendency.lower() == "median" else cat_data.mean()
            # White marker point with crisp outline
            ax.plot(
                idx, rep_val, marker='o', color='white', markeredgecolor='black',
                markeredgewidth=1.2, markersize=7, zorder=4
            )
            # Text label with glowing outline for high contrast
            txt = ax.text(
                idx, rep_val + y_offset, f"{rep_val:.2f}",
                ha='center', va='bottom', fontsize=text_fs, fontweight='bold', zorder=5
            )
            txt.set_path_effects([
                PathEffects.withStroke(linewidth=3.5, foreground='white')
            ])

    # 4. Styling & Grid
    ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.65, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("")
    ax.set_ylabel(value_col, fontsize=label_fs, fontweight='bold')
    ax.tick_params(axis="y", labelsize=tick_fs)
    ax.tick_params(axis="x", labelsize=tick_fs, rotation=25 if len(order) > 4 else 0)

    # 5. Statistical Testing & Significance Brackets
    if stats_mode.lower() == "nonparametric":
        omnibus = run_kruskal_wallis(sub, value_col, group_col, ignore_zeros=ignore_zeros)
        posthoc = run_dunns_posthoc_fdr(sub, value_col, group_col, alpha=alpha, min_cliffs_delta=min_effect_size_bracket, ignore_zeros=ignore_zeros)
        
        if omnibus is not None:
            p_val_str = format_p_value(omnibus['p_unc'])
            subtitle = f"{title or value_col}\nKruskal-Wallis p = {p_val_str}"
            ax.set_title(subtitle, fontsize=title_fs, fontweight='bold', pad=10)
            
            if posthoc is not None and add_brackets:
                sig_pairs = [
                    (row["A"], row["B"], row["pval"], row.get("cliffs_delta", np.nan))
                    for _, row in posthoc.iterrows() 
                    if row["pval"] < alpha and abs(row.get("cliffs_delta", 999)) >= min_effect_size_bracket
                ]
                add_significance_brackets(ax, sig_pairs, order, max_brackets=MAX_BRACKETS, font_scale=font_scale)
        else:
            ax.set_title(f"{title or value_col}\n(Insufficient data for Kruskal-Wallis)", fontsize=title_fs)
    else:
        # Legacy parametric mode
        omnibus, posthoc = run_welch_games_howell(sub, value_col, group_col, alpha, ignore_zeros=ignore_zeros)
        if omnibus is not None:
            p_val_str = format_p_value(omnibus['p_unc'])
            subtitle = f"{title or value_col}\nWelch's ANOVA p = {p_val_str}"
            ax.set_title(subtitle, fontsize=title_fs, fontweight='bold', pad=10)
            
            if posthoc is not None and add_brackets:
                sig_pairs = [
                    (row["A"], row["B"], row["pval"], row.get("cohen_d", np.nan))
                    for _, row in posthoc.iterrows() 
                    if row["pval"] < alpha and abs(row.get("cohen_d", 999)) >= min_effect_size_bracket
                ]
                add_significance_brackets(ax, sig_pairs, order, max_brackets=MAX_BRACKETS, font_scale=font_scale)
        else:
            ax.set_title(f"{title or value_col}\n(Insufficient data for ANOVA)", fontsize=title_fs)

    return omnibus, posthoc


# -----------------------------------------------------------------------------
# MAIN PIPELINE EXECUTION
# -----------------------------------------------------------------------------

def _safe_savefig(fig, fpath, dpi=DPI, transparent=TRANSPARENT_BG, bbox_inches='tight'):
    """Save figure with robust retry handling for locked or synced files (e.g. OneDrive)."""
    import time
    for attempt in range(4):
        try:
            fig.savefig(fpath, dpi=dpi, transparent=transparent, bbox_inches=bbox_inches)
            return
        except OSError:
            if attempt < 3:
                time.sleep(0.3)
            else:
                base, ext = os.path.splitext(fpath)
                fallback = f"{base}_new{ext}"
                try:
                    fig.savefig(fallback, dpi=dpi, transparent=transparent, bbox_inches=bbox_inches)
                except Exception as e:
                    print(f"[Warning] Failed to save {fpath}: {e}")


# -----------------------------------------------------------------------------
# SCHEMA ADAPTER & LOOKUP MAPPING UTILITIES
# -----------------------------------------------------------------------------

def resolve_canonical_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Finds the first existing column in df that matches any of the candidate alias names."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def apply_lookup_mapping(
    df: pd.DataFrame,
    source_col: Optional[str] = None,
    target_col: str = "Condition",
    mapping: Optional[Union[Dict[Any, Any], str, Path]] = None
) -> pd.DataFrame:
    """
    Applies a key-value lookup mapping from source_col to create target_col.
    Supports in-memory dicts, JSON files, or 2-column CSV files.
    Handles type mismatch (e.g. str keys vs int column values) transparently.
    """
    if mapping is None:
        return df

    # If mapping is a path to a file
    if isinstance(mapping, (str, Path)):
        map_path = str(mapping)
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Lookup mapping file not found: {map_path}")
        if map_path.endswith(".json"):
            with open(map_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict) and "mapping" in loaded:
                source_col = loaded.get("source_col", source_col)
                target_col = loaded.get("target_col", target_col)
                mapping = loaded["mapping"]
            else:
                mapping = loaded
        elif map_path.endswith(".csv"):
            map_df = pd.read_csv(map_path)
            if map_df.shape[1] >= 2:
                source_col = source_col or map_df.columns[0]
                target_col = target_col or map_df.columns[1]
                mapping = dict(zip(map_df.iloc[:, 0], map_df.iloc[:, 1]))
            else:
                raise ValueError(f"Lookup CSV must have at least 2 columns: {map_path}")

    if isinstance(mapping, dict):
        if "mapping" in mapping and ("source_col" in mapping or "target_col" in mapping):
            source_col = mapping.get("source_col", source_col)
            target_col = mapping.get("target_col", target_col)
            mapping = mapping["mapping"]

    if source_col is None:
        source_col = resolve_canonical_column(df, SAMPLE_ID_ALIASES)
        if source_col is None:
            raise ValueError("source_col for lookup mapping could not be determined automatically.")

    if source_col not in df.columns:
        resolved = resolve_canonical_column(df, [source_col] + SAMPLE_ID_ALIASES)
        if resolved:
            source_col = resolved
        else:
            raise KeyError(f"Source column '{source_col}' not found in DataFrame columns.")

    # Create dual-keyed lookup (matching both str(k) and native k) to handle int/str mismatches
    dual_map = {}
    for k, v in mapping.items():
        dual_map[k] = v
        dual_map[str(k)] = v
        try:
            dual_map[int(k)] = v
        except (ValueError, TypeError):
            pass

    df[target_col] = df[source_col].map(dual_map).fillna(df[source_col].astype(str))
    print(f"Applied lookup mapping from '{source_col}' to '{target_col}'. Category counts: {df[target_col].value_counts().to_dict()}")
    return df


def discover_metric_columns(df: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> List[str]:
    """Auto-discovers continuous numeric columns suitable for statistical testing."""
    excludes = set(NON_METRIC_COLS)
    if exclude_cols:
        excludes.update(exclude_cols)
    
    numeric_cols = []
    for col in df.select_dtypes(include=np.number).columns:
        if col in excludes:
            continue
        if col.endswith("_JSON") or col.endswith("_px") or col.endswith("_idx") or col.endswith("_id"):
            continue
        if df[col].nunique(dropna=True) <= 1:
            continue
        numeric_cols.append(col)
    return numeric_cols


def load_table(input_path: Union[str, Path]) -> pd.DataFrame:
    """Loads tabular data from Parquet, Feather, CSV, or Excel formats."""
    p = str(input_path)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Input file not found: {p}")
    if p.endswith(".parquet"):
        return pd.read_parquet(p)
    elif p.endswith(".feather"):
        return pd.read_feather(p)
    elif p.endswith(".csv"):
        return pd.read_csv(p)
    elif p.endswith(".xlsx") or p.endswith(".xls"):
        return pd.read_excel(p)
    else:
        try:
            return pd.read_parquet(p)
        except Exception:
            return pd.read_csv(p)


def load_json_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Loads a JSON configuration file for pipeline execution."""
    p = str(config_path)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Configuration file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"JSON configuration must be an object/dict, got {type(config)}")
    return config


def run_group_stats_pipeline(
    df: pd.DataFrame,
    group_cols: Optional[List[str]] = None,
    value_cols: Optional[List[str]] = None,
    col_order: Optional[Dict[str, List[Any]]] = None,
    plot_types: Union[str, List[str]] = "both",
    combine_two_groups: bool = COMBINE_TWO_GROUPS,
    filter_values: Optional[Dict[str, List[Any]]] = None,
    palette: Union[str, List, Dict] = PALETTE,
    add_brackets: bool = True,
    output_dir: Optional[str] = None,
    strip_hue_col: Optional[str] = None,
    strip_palette: str = STRIP_PALETTE,
    save_legend: bool = SAVE_STRIP_LEGEND,
    font_scale: float = FONT_SCALE,
    ignore_zeros: bool = False,
    stats_mode: str = STATS_MODE,
    min_effect_size_bracket: float = MIN_EFFECT_SIZE_BRACKET,
    central_tendency: str = CENTRAL_TENDENCY,
    error_bar_type: str = ERROR_BAR_TYPE,
    lookup_table: Optional[Union[Dict[str, Any], str, Path]] = None,
    lookup_source_col: Optional[str] = None,
    lookup_target_col: str = "Condition",
    show_points: bool = True,
    max_points: Optional[int] = None,
    export_format: str = "both",
    export_csv: Optional[bool] = None,
    export_parquet: Optional[bool] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executes the end-to-end visualization and statistical testing pipeline.
    
    Returns:
        (omnibus_results_df, posthoc_results_df, descriptive_stats_df)
    """
    df_proc = df.copy()

    # 1. Apply Lookup Mapping (e.g. mapping sample IDs to experimental conditions)
    if lookup_table is not None:
        df_proc = apply_lookup_mapping(
            df=df_proc, 
            source_col=lookup_source_col, 
            target_col=lookup_target_col, 
            mapping=lookup_table
        )

    # 2. Resolve Group Columns
    if not group_cols:
        if "Condition" in df_proc.columns and "Label" in df_proc.columns:
            group_cols = ["Condition", "Label"]
        elif "group" in df_proc.columns and "Region" in df_proc.columns:
            group_cols = ["group", "Region"]
        elif "group" in df_proc.columns:
            group_cols = ["group"]
        elif "Label" in df_proc.columns:
            group_cols = ["Label"]
        else:
            raise ValueError(
                "group_cols must be specified (e.g. group_cols=['Condition', 'Label'] or ['group', 'Region'])."
            )
        print(f"Defaulting group_cols to: {group_cols}")

    # 3. Resolve Value Metric Columns
    if not value_cols:
        value_cols = discover_metric_columns(df_proc)
        if not value_cols:
            raise ValueError("No numeric metric columns found for statistical testing.")
        print(f"Auto-discovered {len(value_cols)} numeric metric columns: {value_cols[:8]}...")

    # 4. Resolve Output Directory
    if not output_dir:
        output_dir = DEFAULT_OUTPUT_DIR
    group_subfolder = "_".join([re.sub(r"[^\w\s]", "", str(g)).strip().replace(" ", "_") for g in group_cols])
    effective_output_dir = os.path.normpath(os.path.join(output_dir, group_subfolder)) if group_subfolder else output_dir
    os.makedirs(effective_output_dir, exist_ok=True)

    # 5. Resolve Strip Hue Column (Sample Replicates)
    if strip_hue_col is None:
        strip_hue_col = resolve_canonical_column(df_proc, SAMPLE_ID_ALIASES)
        if strip_hue_col:
            print(f"Auto-detected sample replicate column: '{strip_hue_col}'")

    # 6. Apply Filtering
    if filter_values:
        for col, keep_vals in filter_values.items():
            if col in df_proc.columns and keep_vals:
                df_proc = df_proc[df_proc[col].isin(keep_vals)]
                print(f"Filtered '{col}' to {len(keep_vals)} categories. Remaining rows: {len(df_proc)}")

    if len(df_proc) == 0:
        raise ValueError("DataFrame is empty after applying filter_values! Please check your filter values.")

    # 7. Setup Strip Plot Palette & Separate Legend
    strip_order = None
    strip_palette_dict = None
    if strip_hue_col and strip_hue_col in df_proc.columns:
        try:
            strip_order = sorted(df_proc[strip_hue_col].dropna().unique())
        except TypeError:
            strip_order = list(df_proc[strip_hue_col].dropna().unique())
        strip_palette_dict = build_palette_dict(strip_order, strip_palette)
        if save_legend:
            save_strip_legend(strip_order, strip_palette_dict, title=strip_hue_col, output_dir=effective_output_dir)

    # 3. Standardize Col Order mapping
    if col_order is None:
        col_order = {}
    for gcol in group_cols:
        if gcol in df_proc.columns and gcol not in col_order:
            col_order[gcol] = list(df_proc[gcol].dropna().unique())

    # 4. Resolve Plot Types
    if isinstance(plot_types, str):
        if plot_types.lower() == "both":
            active_plot_types = ["violin", "bar"]
        else:
            active_plot_types = [plot_types.lower()]
    else:
        active_plot_types = [p.lower() for p in plot_types]

    omnibus_records = []
    posthoc_records = []
    descriptive_records = []
    stratified_records = []

    # 5. Iterate through Value Metrics and Plot Types
    for ptype in active_plot_types:
        ptype_dir = os.path.join(effective_output_dir, f"{ptype}_plots")
        os.makedirs(ptype_dir, exist_ok=True)

        for item in value_cols:
            if item not in df_proc.columns:
                print(f"[Skip] Metric '{item}' not found in dataframe columns.")
                continue

            if not pd.api.types.is_numeric_dtype(df_proc[item]):
                try:
                    converted = pd.to_numeric(df_proc[item], errors='coerce')
                    if converted.dropna().empty:
                        print(f"[Skip] Metric '{item}' contains non-numeric/categorical data and cannot be plotted quantitatively.")
                        continue
                    df_proc[item] = converted
                except Exception:
                    print(f"[Skip] Metric '{item}' is non-numeric and cannot be processed.")
                    continue

            clean_name = re.sub(r"[^\w\s]", "", item).strip().replace(" ", "_")
            if not clean_name:
                clean_name = f"metric_{hash(item)}"

            # ----------------------------------------------------
            # Scenario A: Single Grouping Column
            # ----------------------------------------------------
            if len(group_cols) == 1:
                g_col = group_cols[0]
                order = col_order.get(g_col, list(df_proc[g_col].dropna().unique()))
                
                fig_width = max(4.8, 1.35 * len(order))
                fig, ax = plt.subplots(figsize=(fig_width, 6))
                
                omni, ph = plot_single_group_axes(
                    df=df_proc, value_col=item, group_col=g_col, group_order=order,
                    plot_type=ptype, ax=ax, title=f"{item} by {g_col}",
                    alpha=ALPHA, min_effect_size_bracket=min_effect_size_bracket,
                    stats_mode=stats_mode, central_tendency=central_tendency,
                    palette=palette, add_brackets=add_brackets,
                    strip_hue_col=strip_hue_col, strip_order=strip_order,
                    strip_palette_dict=strip_palette_dict, error_bar_type=error_bar_type,
                    font_scale=font_scale, ignore_zeros=ignore_zeros,
                    show_points=show_points, max_points=max_points
                )
                fig.tight_layout()
                fig_path = os.path.join(ptype_dir, f"{clean_name}.png")
                _safe_savefig(fig, fig_path, dpi=DPI, transparent=TRANSPARENT_BG, bbox_inches='tight')
                plt.close(fig)

                if ptype == active_plot_types[0]:
                    if omni is not None:
                        omnibus_records.append({**omni, "Metric": item, "Group_By": g_col, "Facet": "None"})
                    if ph is not None:
                        ph["Metric"] = item
                        ph["Group_By"] = g_col
                        ph["Facet"] = "None"
                        posthoc_records.append(ph)
                    desc = compute_nonparametric_descriptives(df_proc, item, g_col, sample_col=strip_hue_col, ignore_zeros=ignore_zeros)
                    desc["Facet"] = "None"
                    descriptive_records.append(desc)
                    
                    if strip_hue_col and strip_hue_col in df_proc.columns:
                        strat = run_stratified_sample_wilcoxon(df_proc, item, g_col, strip_hue_col, alpha=ALPHA, ignore_zeros=ignore_zeros)
                        if strat is not None and not strat.empty:
                            strat["Metric"] = item
                            strat["Group_By"] = g_col
                            strat["Facet"] = "None"
                            stratified_records.append(strat)

            # ----------------------------------------------------
            # Scenario B: Two Grouping Columns
            # ----------------------------------------------------
            elif len(group_cols) == 2:
                g1, g2 = group_cols[0], group_cols[1]
                order1 = col_order.get(g1, list(df_proc[g1].dropna().unique()))
                order2 = col_order.get(g2, list(df_proc[g2].dropna().unique()))

                if combine_two_groups:
                    # Combined Categories Mode
                    combo_col = f"{g1} | {g2}"
                    d_combo = df_proc.copy()
                    d_combo[combo_col] = d_combo[g1].astype(str) + " | " + d_combo[g2].astype(str)
                    
                    combined_order = [
                        f"{v1} | {v2}" for v1 in order1 for v2 in order2 
                        if f"{v1} | {v2}" in d_combo[combo_col].unique()
                    ]
                    if not combined_order:
                        combined_order = list(d_combo[combo_col].unique())
                    
                    fig_width = max(5.5, 1.35 * len(combined_order))
                    fig, ax = plt.subplots(figsize=(fig_width, 6.5))
                    
                    omni, ph = plot_single_group_axes(
                        df=d_combo, value_col=item, group_col=combo_col, group_order=combined_order,
                        plot_type=ptype, ax=ax, title=f"{item} ({g1} & {g2})",
                        alpha=ALPHA, min_effect_size_bracket=min_effect_size_bracket,
                        stats_mode=stats_mode, central_tendency=central_tendency,
                        palette=palette, add_brackets=add_brackets,
                        strip_hue_col=strip_hue_col, strip_order=strip_order,
                        strip_palette_dict=strip_palette_dict, error_bar_type=error_bar_type,
                        font_scale=font_scale, ignore_zeros=ignore_zeros,
                        show_points=show_points, max_points=max_points
                    )
                    fig.tight_layout()
                    fig_path = os.path.join(ptype_dir, f"{clean_name}.png")
                    _safe_savefig(fig, fig_path, dpi=DPI, transparent=TRANSPARENT_BG, bbox_inches='tight')
                    plt.close(fig)

                    if ptype == active_plot_types[0]:
                        if omni is not None:
                            omnibus_records.append({**omni, "Metric": item, "Group_By": f"{g1}+{g2} (Combined)", "Facet": "None"})
                        if ph is not None:
                            ph["Metric"] = item
                            ph["Group_By"] = f"{g1}+{g2} (Combined)"
                            ph["Facet"] = "None"
                            posthoc_records.append(ph)
                        desc = compute_nonparametric_descriptives(d_combo, item, combo_col, sample_col=strip_hue_col, ignore_zeros=ignore_zeros)
                        desc["Facet"] = "None"
                        descriptive_records.append(desc)
                        
                        if strip_hue_col and strip_hue_col in d_combo.columns:
                            strat = run_stratified_sample_wilcoxon(d_combo, item, combo_col, strip_hue_col, alpha=ALPHA, ignore_zeros=ignore_zeros)
                            if strat is not None and not strat.empty:
                                strat["Metric"] = item
                                strat["Group_By"] = f"{g1}+{g2} (Combined)"
                                strat["Facet"] = "None"
                                stratified_records.append(strat)

                else:
                    # Faceted Subplots Mode
                    facets = [fval for fval in order2 if not df_proc[df_proc[g2] == fval].empty]
                    if not facets:
                        facets = list(df_proc[g2].dropna().unique())

                    n_facets = max(1, len(facets))
                    fig_width = max(6.0, 1.4 * len(order1))
                    fig_height = max(5.0, 4.2 * n_facets)
                    
                    fig, axes = plt.subplots(n_facets, 1, figsize=(fig_width, fig_height), sharey=True, sharex=True)
                    if n_facets == 1:
                        axes = [axes]
                    else:
                        axes = list(axes)

                    temp_results = []
                    for ax, fval in zip(axes, facets):
                        sub_facet = df_proc.loc[df_proc[g2] == fval]
                        omni, ph = plot_single_group_axes(
                            df=sub_facet, value_col=item, group_col=g1, group_order=order1,
                            plot_type=ptype, ax=ax, title=f"{item} | {g2} = {fval}",
                            alpha=ALPHA, min_effect_size_bracket=min_effect_size_bracket,
                            stats_mode=stats_mode, central_tendency=central_tendency,
                            palette=palette, add_brackets=False,
                            strip_hue_col=strip_hue_col, strip_order=strip_order,
                            strip_palette_dict=strip_palette_dict, error_bar_type=error_bar_type,
                            font_scale=font_scale, ignore_zeros=ignore_zeros,
                            show_points=show_points, max_points=max_points
                        )
                        temp_results.append((ax, omni, ph, sub_facet, fval))
                        
                        if ptype == active_plot_types[0]:
                            if omni is not None:
                                omnibus_records.append({**omni, "Metric": item, "Group_By": g1, "Facet": f"{g2}={fval}"})
                            if ph is not None:
                                ph["Metric"] = item
                                ph["Group_By"] = g1
                                ph["Facet"] = f"{g2}={fval}"
                                posthoc_records.append(ph)
                            desc = compute_nonparametric_descriptives(sub_facet, item, g1, sample_col=strip_hue_col, ignore_zeros=ignore_zeros)
                            desc["Facet"] = f"{g2}={fval}"
                            descriptive_records.append(desc)
                            
                            if strip_hue_col and strip_hue_col in sub_facet.columns:
                                strat = run_stratified_sample_wilcoxon(sub_facet, item, g1, strip_hue_col, alpha=ALPHA, ignore_zeros=ignore_zeros)
                                if strat is not None and not strat.empty:
                                    strat["Metric"] = item
                                    strat["Group_By"] = g1
                                    strat["Facet"] = f"{g2}={fval}"
                                    stratified_records.append(strat)

                    # Pass 2: Add brackets using shared y-limits
                    initial_ymin, initial_ymax = axes[0].get_ylim()
                    max_level_overall = 0
                    if add_brackets:
                        for ax, omni, ph, sub_facet, fval in temp_results:
                            if ph is not None:
                                es_col = "cliffs_delta" if stats_mode.lower() == "nonparametric" else "cohen_d"
                                sig_pairs = [
                                    (row["A"], row["B"], row["pval"], row.get(es_col, np.nan))
                                    for _, row in ph.iterrows()
                                    if row["pval"] < ALPHA and abs(row.get(es_col, 999)) >= min_effect_size_bracket
                                ]
                                levels = add_significance_brackets(
                                    ax, sig_pairs, order1, 
                                    initial_ymin=initial_ymin, initial_ymax=initial_ymax,
                                    max_brackets=MAX_BRACKETS, font_scale=font_scale
                                )
                                max_level_overall = max(max_level_overall, levels)

                    # Pass 3: Adjust final shared y-limit if brackets were drawn
                    if max_level_overall > 0:
                        y_range = max(initial_ymax - initial_ymin, 1e-6)
                        y_start = initial_ymax + 0.04 * y_range
                        h = 0.07 * y_range
                        final_ymax = y_start + max_level_overall * (h * 1.45) + h * 1.5
                        axes[0].set_ylim(initial_ymin, final_ymax)

                    fig.tight_layout()
                    fig_path = os.path.join(ptype_dir, f"{clean_name}.png")
                    _safe_savefig(fig, fig_path, dpi=DPI, transparent=TRANSPARENT_BG, bbox_inches='tight')
                    plt.close(fig)

            else:
                raise ValueError(f"group_cols currently supports 1 or 2 columns (received {len(group_cols)}).")

    # 6. Format and Export Statistical Summary CSV Tables
    omnibus_df = pd.DataFrame(omnibus_records) if omnibus_records else pd.DataFrame()
    posthoc_df = pd.concat(posthoc_records, ignore_index=True) if posthoc_records else pd.DataFrame()
    desc_df = pd.concat(descriptive_records, ignore_index=True) if descriptive_records else pd.DataFrame()
    stratified_df = pd.concat(stratified_records, ignore_index=True) if stratified_records else pd.DataFrame()

    # Resolve output formats
    fmt = str(export_format or "both").lower()
    should_export_parquet = (export_parquet is True) or (fmt in ["parquet", "both"] and export_parquet is not False)
    should_export_csv = (export_csv is True) or (fmt in ["csv", "both"] and export_csv is not False)
    if export_csv is False and export_parquet is None:
        should_export_csv = False
        should_export_parquet = True
    if export_parquet is False and export_csv is None:
        should_export_parquet = False
        should_export_csv = True

    def _export_table(df_table: pd.DataFrame, file_stem: str):
        if df_table is None or df_table.empty:
            return
        out_base = os.path.join(effective_output_dir, file_stem)
        saved = []
        if should_export_parquet:
            p_path = f"{out_base}.parquet"
            df_table.to_parquet(p_path, index=False)
            saved.append(p_path)
        if should_export_csv:
            c_path = f"{out_base}.csv"
            df_table.to_csv(c_path, index=False)
            saved.append(c_path)
        if saved:
            print(f"Exported {file_stem} to: {', '.join(saved)}")

    if not omnibus_df.empty:
        if "p_unc" in omnibus_df.columns:
            omnibus_df["p_unc_formatted"] = omnibus_df["p_unc"].apply(format_p_value)
            omnibus_df["significance"] = omnibus_df["p_unc"].apply(p_to_stars)
        omni_stem = "kruskal_wallis_results" if stats_mode.lower() == "nonparametric" else "welch_anova_results"
        _export_table(omnibus_df, omni_stem)

    if not posthoc_df.empty:
        if "pval" in posthoc_df.columns:
            posthoc_df["pval_formatted"] = posthoc_df["pval"].apply(format_p_value)
        lead_cols = [c for c in [
            "Metric", "Group_By", "Facet", "A", "B", "N(A)", "N(B)", 
            "median(A)", "median(B)", "diff", "pct_diff", "se", "Z", "pval_raw", 
            "pval", "pval_formatted", "significance", "cliffs_delta", "delta_95_ci", 
            "effect_size_magnitude", "is_biologically_significant", "mean(A)", "mean(B)", 
            "cohen_d", "hedges_g"
        ] if c in posthoc_df.columns]
        other_cols = [c for c in posthoc_df.columns if c not in lead_cols]
        posthoc_df = posthoc_df[lead_cols + other_cols]
        ph_stem = "dunns_pairwise_fdr_cliffs_delta" if stats_mode.lower() == "nonparametric" else "games_howell_pairwise_results"
        _export_table(posthoc_df, ph_stem)

    if not desc_df.empty:
        _export_table(desc_df, "descriptive_statistics_summary")

    if not stratified_df.empty:
        _export_table(stratified_df, "stratified_sample_wilcoxon")

    print(f"\n=======================================================")
    print(f" Pipeline Complete! Visualizations & Stats saved to:")
    print(f" {effective_output_dir}")
    print(f"=======================================================\n")
    return omnibus_df, posthoc_df, desc_df


# -----------------------------------------------------------------------------
# CLI & DEMO SCRIPT ENTRYPOINT
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Group Statistical Testing & Publication-Quality Visualization Pipeline"
    )
    parser.add_argument("--config", "-c", type=str, default=None, help="Path to JSON configuration file")
    parser.add_argument("--input", "-i", type=str, default=None, help="Path to input table (.parquet, .feather, .csv, .xlsx)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output directory for plots and statistical tables")
    parser.add_argument("--group-cols", "-g", nargs="+", default=None, help="1 or 2 categorical column names to group by")
    parser.add_argument("--value-cols", "-v", nargs="+", default=None, help="Numeric metric column names to analyze (auto-discovered if omitted)")
    parser.add_argument("--sample-col", "-s", type=str, default=None, help="Sample/replicate column for strip dots & stratified Wilcoxon")
    parser.add_argument("--lookup-file", type=str, default=None, help="Path to JSON or CSV lookup table to assign condition metadata")
    parser.add_argument("--lookup-source", type=str, default=None, help="Source column for lookup mapping (e.g. Sample_ID)")
    parser.add_argument("--lookup-target", type=str, default="Condition", help="Target column name to create from lookup (default: Condition)")
    parser.add_argument("--plot-type", type=str, default=None, choices=["violin", "bar", "both"], help="Plot style to render")
    parser.add_argument("--combine", action="store_true", default=None, help="Combine two group columns into single unified category x-axis")
    parser.add_argument("--mode", type=str, default=None, choices=["nonparametric", "parametric"], help="Statistical mode (nonparametric or parametric)")
    parser.add_argument("--min-delta", type=float, default=None, help="Minimum Cliff's delta threshold for bracket rendering")
    parser.add_argument("--ignore-zeros", action="store_true", default=None, help="Exclude 0, NaN, and Inf values from analysis")
    args, unknown = parser.parse_known_args()

    # Load configuration from JSON if provided
    config = {}
    if args.config:
        print(f"Loading configuration from JSON: {args.config}")
        config = load_json_config(args.config)

    # CLI arguments override JSON config
    input_path = args.input or config.get("input_path")
    if not input_path:
        raise ValueError("Input data path must be specified via --input or inside a --config JSON file.")

    output_dir = args.output or config.get("output_dir", DEFAULT_OUTPUT_DIR)
    group_cols = args.group_cols or config.get("group_cols")
    value_cols = args.value_cols or config.get("value_cols")
    sample_col = args.sample_col or config.get("sample_col")
    plot_type = args.plot_type or config.get("plot_types", PLOT_TYPE)
    combine = args.combine if args.combine is not None else config.get("combine_two_groups", COMBINE_TWO_GROUPS)
    stats_mode = args.mode or config.get("stats_mode", STATS_MODE)
    min_delta = args.min_delta if args.min_delta is not None else config.get("min_effect_size_bracket", MIN_EFFECT_SIZE_BRACKET)
    ignore_zeros = args.ignore_zeros if args.ignore_zeros is not None else config.get("ignore_zeros", False)
    filter_values = config.get("filter_values")
    col_order = config.get("col_order")

    # Resolve lookup table
    lookup_table = args.lookup_file or config.get("lookup_table")
    lookup_source = args.lookup_source
    lookup_target = args.lookup_target or "Condition"

    print(f"Loading data from: {input_path}")
    df = load_table(input_path)
    print(f"Loaded dataset: {df.shape[0]:,} rows x {df.shape[1]} columns.")

    run_group_stats_pipeline(
        df=df,
        group_cols=group_cols,
        value_cols=value_cols,
        col_order=col_order,
        plot_types=plot_type,
        combine_two_groups=combine,
        filter_values=filter_values,
        palette=config.get("palette", PALETTE),
        add_brackets=config.get("add_brackets", True),
        output_dir=output_dir,
        strip_hue_col=sample_col,
        strip_palette=config.get("strip_palette", STRIP_PALETTE),
        save_legend=config.get("save_strip_legend", SAVE_STRIP_LEGEND),
        font_scale=config.get("font_scale", FONT_SCALE),
        ignore_zeros=ignore_zeros,
        stats_mode=stats_mode,
        min_effect_size_bracket=min_delta,
        lookup_table=lookup_table,
        lookup_source_col=lookup_source,
        lookup_target_col=lookup_target
    )


if __name__ == "__main__":
    main()
