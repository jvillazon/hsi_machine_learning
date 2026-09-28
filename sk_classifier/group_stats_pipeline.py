"""
========================================================================================
Group Statistics & Publication-Quality Visualization Pipeline
========================================================================================
Features:
- Flexible Multi-Column Metric selection (e.g. % Lipids, Turnover, Ratios, Intensity, etc.)
- Multi-Level Grouping (1, 2, or more grouping columns, combine/facet/hue modes)
- Robust Statistical Testing:
    * Welch's ANOVA (robust to heteroscedasticity and unequal sample sizes)
    * Games-Howell Post-Hoc Test (pairwise comparisons with Welch t-test & adjusted p-values)
    * Effect Size Calculations:
        - Cohen's d (sample standard deviation pooled)
        - Hedges' g (sample-size corrected unbiased Cohen's d)
        - Percentage Difference between group means
        - Effect size qualitative interpretation (Negligible, Small, Medium, Large, Very Large)
- Rich Visualizations:
    * Violin Plots (with boxplot inner, overlayed strip plot, mean point + labels)
    * Bar Plots (with mean +/- SEM/SD error bars, overlayed strip points, mean labels)
    * Automatic Significance Brackets with multi-level collision avoidance
    * Standalone Legend generation for sample IDs / hue categories
- Data Formats: Parquet (.parquet), Feather (.feather), CSV (.csv), Excel (.xlsx)
========================================================================================
"""

import os
import re
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as PathEffects
import seaborn as sns
import pingouin as pg

# -----------------------------------------------------------------------------
# DEFAULT CONFIGURATION
# -----------------------------------------------------------------------------
DEFAULT_DATA_PATH = r"C:\Users\jvill\OneDrive - UC San Diego\Vallon Kidney\headless_arithmetic_outputs\all_images_roi_metrics.parquet"
DEFAULT_OUTPUT_DIR = r"C:\Users\jvill\OneDrive - UC San Diego\Vallon Kidney\headless_arithmetic_outputs\plots\group_stats"

# Plotting parameters
PLOT_TYPE = "both"             # "violin", "bar", or "both"
COMBINE_TWO_GROUPS = False     # True: merge 2 group cols into one combined category (e.g. "Ctrl/Veh | PT-S1/S2")
                               # False: facet by 2nd group column with aligned shared axes
ALPHA = 0.05                   # Significance threshold for post-hoc tests and brackets
MAX_BRACKETS = 6               # Cap on brackets per subplot to prevent clutter (None = show all)
MIN_EFFECT_SIZE_BRACKET = 0.0  # Only draw brackets if |Cohen's d| or |Hedges' g| >= this threshold (0.0 = all sig)
ERROR_BAR_TYPE = "sem"         # For bar plots: "sem", "sd", or "ci" (95% confidence interval)

PALETTE = "colorblind"         # Seaborn or custom palette name / list / dict
STRIP_HUE_COL = "Sample ID"    # Color individual strip dots by Sample ID / Animal / Replicate (or None)
STRIP_PALETTE = "tab20"        # Palette for strip plot dots
SAVE_STRIP_LEGEND = True       # Save separate clean PNG legend for strip dots
FONT_SCALE = 1.5               # Global font scaling factor for publication figures
DPI = 300                      # Output resolution for saved figures
TRANSPARENT_BG = True          # Save figures with transparent background


# -----------------------------------------------------------------------------
# STATISTICAL UTILITIES & EFFECT SIZES
# -----------------------------------------------------------------------------

def p_to_stars(p: float) -> str:
    """Converts a p-value to standard scientific asterisk notation."""
    if pd.isna(p):
        return ""
    if p < 0.0001:
        return "****"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def format_p_value(p: float) -> str:
    """Formats p-values cleanly for publication tables."""
    if pd.isna(p):
        return "N/A"
    if p == 0 or p < 1e-15:
        return "< 1e-15"
    if p < 0.001:
        return f"{p:.4e}"
    return f"{p:.4f}"


def interpret_effect_size(d: float) -> str:
    """Provides Cohen's standard qualitative interpretation of effect size."""
    if pd.isna(d):
        return "N/A"
    abs_d = abs(d)
    if abs_d < 0.2:
        return "Negligible"
    elif abs_d < 0.5:
        return "Small"
    elif abs_d < 0.8:
        return "Medium"
    elif abs_d < 1.2:
        return "Large"
    else:
        return "Very Large"


def compute_cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> Tuple[float, float]:
    """
    Computes Cohen's d and Hedges' g (sample-size corrected Cohen's d).
    Returns (cohens_d, hedges_g).
    """
    a = group_a[~np.isnan(group_a)]
    b = group_b[~np.isnan(group_b)]
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan
    
    mean1, mean2 = np.mean(a), np.mean(b)
    var1, var2 = np.var(a, ddof=1), np.var(b, ddof=1)
    
    # Pooled standard deviation
    s_pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / max(1, (n1 + n2 - 2)))
    if s_pooled == 0 or np.isnan(s_pooled):
        return 0.0, 0.0
    
    d = (mean1 - mean2) / s_pooled
    # Hedges' g correction factor
    j = 1.0 - (3.0 / max(1.0, (4.0 * (n1 + n2) - 9.0)))
    g = d * j
    return float(d), float(g)


def compute_descriptive_stats(df: pd.DataFrame, value_col: str, group_col: str) -> pd.DataFrame:
    """Computes comprehensive summary statistics per group."""
    grouped = df.groupby(group_col, observed=True)[value_col]
    stats = grouped.agg(
        N='count',
        Mean='mean',
        Std='std',
        Median='median',
        Min='min',
        Max='max',
        Q25=lambda x: np.percentile(x.dropna(), 25) if len(x.dropna()) > 0 else np.nan,
        Q75=lambda x: np.percentile(x.dropna(), 75) if len(x.dropna()) > 0 else np.nan,
    ).reset_index()
    stats['SEM'] = stats['Std'] / np.sqrt(stats['N'].clip(lower=1))
    stats['IQR'] = stats['Q75'] - stats['Q25']
    stats['Metric'] = value_col
    return stats


def run_welch_games_howell(
    data: pd.DataFrame, 
    value_col: str, 
    group_col: str, 
    alpha: float = ALPHA
) -> Tuple[Optional[Dict[str, Any]], Optional[pd.DataFrame]]:
    """
    Performs Welch's ANOVA and Games-Howell post-hoc testing with Cohen's d & Hedges' g.
    
    Returns:
        anova_dict: Dictionary of ANOVA results (F, df1, df2, p_unc, np2/eta-sq) or None
        posthoc_df: DataFrame containing pairwise Games-Howell results with effect sizes or None
    """
    sub = data[[value_col, group_col]].dropna()
    counts = sub[group_col].value_counts()
    valid_groups = counts[counts >= 2].index
    sub = sub[sub[group_col].isin(valid_groups)]
    
    k = sub[group_col].nunique()
    if k < 2:
        return None, None
    
    # 1. Welch's ANOVA
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
    
    # 2. Games-Howell Post-Hoc (if groups >= 2)
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
                
                cohens_d_list.append(d)
                hedges_g_list.append(g)
                pct_diff_list.append(pct_diff)
                effect_interp_list.append(interpret_effect_size(d))
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
    legend_path = os.path.join(output_dir, legend_filename)
    fig.savefig(legend_path, dpi=DPI, bbox_inches='tight', transparent=TRANSPARENT_BG)
    plt.close(fig)
    print(f"Saved strip hue legend to: {legend_path}")
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
    sig_pairs: list of tuples (group_a, group_b, p_value, effect_size)
    """
    if not sig_pairs:
        return 0

    ymin = initial_ymin if initial_ymin is not None else ax.get_ylim()[0]
    ymax = initial_ymax if initial_ymax is not None else ax.get_ylim()[1]
    y_range = max(ymax - ymin, 1e-6)
    y_start = ymax + 0.04 * y_range
    h = 0.07 * y_range

    # Sort shortest (adjacent) comparisons first so they sit lower
    valid_pairs = []
    for item in sig_pairs:
        g1, g2 = item[0], item[1]
        if g1 in order and g2 in order and g1 != g2:
            valid_pairs.append(item)
            
    valid_pairs = sorted(valid_pairs, key=lambda t: abs(order.index(t[0]) - order.index(t[1])))
    if max_brackets is not None and max_brackets > 0:
        valid_pairs = valid_pairs[:max_brackets]

    levels_occupied = []  # list of lists of (left, right) x intervals
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
    palette: Union[str, List, Dict] = PALETTE,
    add_brackets: bool = True,
    strip_hue_col: Optional[str] = STRIP_HUE_COL,
    strip_order: Optional[List[Any]] = None,
    strip_palette_dict: Optional[Dict[Any, Any]] = None,
    error_bar_type: str = ERROR_BAR_TYPE,
    font_scale: float = FONT_SCALE
) -> Tuple[Optional[Dict[str, Any]], Optional[pd.DataFrame]]:
    """
    Renders either a Violin plot or a Bar plot with mean points, strip dots,
    and statistical testing on a given matplotlib axes.
    """
    cols_to_keep = [value_col, group_col]
    if strip_hue_col and strip_hue_col in df.columns and strip_hue_col not in cols_to_keep:
        cols_to_keep.append(strip_hue_col)
        
    sub = df[cols_to_keep].dropna(subset=[value_col, group_col])
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
        # Calculate means and errors for bar plot
        means = [sub.loc[sub[group_col] == cat, value_col].mean() for cat in order]
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
            x_indices, means, yerr=errors, capsize=4, 
            color=bar_colors, edgecolor='black', linewidth=1.1, alpha=0.82, zorder=1
        )
        ax.set_xticks(x_indices)
        ax.set_xticklabels(order)
    else:
        # Violin plot
        sns.violinplot(
            data=sub, x=group_col, y=value_col, order=order,
            hue=group_col, hue_order=order, palette=palette_dict,
            cut=0, inner="box", ax=ax, dodge=False, legend=False,
            linewidth=1.2, zorder=1
        )

    # 2. Overlay Strip Plot
    if strip_hue_col and strip_hue_col in sub.columns:
        s_order = strip_order if strip_order is not None else sorted(sub[strip_hue_col].dropna().unique())
        s_pal = strip_palette_dict if strip_palette_dict is not None else build_palette_dict(s_order, STRIP_PALETTE)
        sns.stripplot(
            data=sub, x=group_col, y=value_col, order=order,
            hue=strip_hue_col, hue_order=s_order, palette=s_pal,
            dodge=False, jitter=0.22, size=5, alpha=0.75,
            edgecolor="black", linewidth=0.5, ax=ax, legend=False, zorder=3
        )
    else:
        sns.stripplot(
            data=sub, x=group_col, y=value_col, order=order,
            hue=group_col, hue_order=order, palette=palette_dict,
            dodge=False, jitter=0.22, size=5, alpha=0.6,
            edgecolor="black", linewidth=0.5, ax=ax, legend=False, zorder=3
        )

    # 3. Mean Marker & Text Overlay
    ymin, ymax = ax.get_ylim()
    y_span = max(ymax - ymin, 1e-6)
    y_offset = 0.025 * y_span
    
    for idx, category in enumerate(order):
        cat_data = sub.loc[sub[group_col] == category, value_col]
        if len(cat_data) > 0:
            mean_val = cat_data.mean()
            # Mean point (white with black outline)
            ax.plot(
                idx, mean_val, marker='o', color='white', markeredgecolor='black',
                markeredgewidth=1.2, markersize=7, zorder=4
            )
            # Mean text label with white outline glow for readability
            txt = ax.text(
                idx, mean_val + y_offset, f"{mean_val:.2f}",
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
    anova, posthoc = run_welch_games_howell(sub, value_col, group_col, alpha)
    if anova is not None:
        p_val_str = format_p_value(anova['p_unc'])
        subtitle = f"{title or value_col}\nWelch's ANOVA p = {p_val_str}"
        ax.set_title(subtitle, fontsize=title_fs, fontweight='bold', pad=10)
        
        if posthoc is not None and add_brackets:
            sig_pairs = [
                (row["A"], row["B"], row["pval"], row.get("cohen_d", np.nan))
                for _, row in posthoc.iterrows() 
                if row["pval"] < alpha and abs(row.get("cohen_d", 999)) >= MIN_EFFECT_SIZE_BRACKET
            ]
            add_significance_brackets(ax, sig_pairs, order, max_brackets=MAX_BRACKETS, font_scale=font_scale)
    else:
        ax.set_title(f"{title or value_col}\n(Insufficient data for ANOVA)", fontsize=title_fs)

    return anova, posthoc


# -----------------------------------------------------------------------------
# MAIN PIPELINE EXECUTION
# -----------------------------------------------------------------------------

def run_group_stats_pipeline(
    df: pd.DataFrame,
    group_cols: List[str],
    value_cols: List[str],
    col_order: Optional[Dict[str, List[Any]]] = None,
    plot_types: Union[str, List[str]] = "both",
    combine_two_groups: bool = COMBINE_TWO_GROUPS,
    filter_values: Optional[Dict[str, List[Any]]] = None,
    palette: Union[str, List, Dict] = PALETTE,
    add_brackets: bool = True,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    strip_hue_col: Optional[str] = STRIP_HUE_COL,
    strip_palette: str = STRIP_PALETTE,
    save_legend: bool = SAVE_STRIP_LEGEND,
    font_scale: float = FONT_SCALE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executes the end-to-end visualization and statistical testing pipeline.
    
    Returns:
        (anova_results_df, posthoc_results_df, descriptive_stats_df)
    """
    os.makedirs(output_dir, exist_ok=True)
    df_proc = df.copy()

    # 1. Apply Filtering
    if filter_values:
        for col, keep_vals in filter_values.items():
            if col in df_proc.columns and keep_vals:
                df_proc = df_proc[df_proc[col].isin(keep_vals)]
                print(f"Filtered '{col}' to {len(keep_vals)} categories. Remaining rows: {len(df_proc)}")

    if len(df_proc) == 0:
        raise ValueError("DataFrame is empty after applying filter_values! Please check your filter values.")

    # 2. Setup Strip Plot Palette & Separate Legend
    strip_order = None
    strip_palette_dict = None
    if strip_hue_col and strip_hue_col in df_proc.columns:
        try:
            strip_order = sorted(df_proc[strip_hue_col].dropna().unique())
        except TypeError:
            strip_order = list(df_proc[strip_hue_col].dropna().unique())
        strip_palette_dict = build_palette_dict(strip_order, strip_palette)
        if save_legend:
            save_strip_legend(strip_order, strip_palette_dict, title=strip_hue_col, output_dir=output_dir)

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

    anova_records = []
    posthoc_records = []
    descriptive_records = []

    # 5. Iterate through Value Metrics and Plot Types
    for ptype in active_plot_types:
        ptype_dir = os.path.join(output_dir, f"{ptype}_plots")
        os.makedirs(ptype_dir, exist_ok=True)

        for item in value_cols:
            if item not in df_proc.columns:
                print(f"[Skip] Metric '{item}' not found in dataframe columns.")
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
                
                aov, ph = plot_single_group_axes(
                    df=df_proc, value_col=item, group_col=g_col, group_order=order,
                    plot_type=ptype, ax=ax, title=f"{item} by {g_col}",
                    alpha=ALPHA, palette=palette, add_brackets=add_brackets,
                    strip_hue_col=strip_hue_col, strip_order=strip_order,
                    strip_palette_dict=strip_palette_dict, font_scale=font_scale
                )
                fig.tight_layout()
                fig_path = os.path.join(ptype_dir, f"{clean_name}.png")
                fig.savefig(fig_path, dpi=DPI, transparent=TRANSPARENT_BG, bbox_inches='tight')
                plt.close(fig)

                if ptype == active_plot_types[0]:
                    if aov is not None:
                        anova_records.append({**aov, "Metric": item, "Group_By": g_col, "Facet": "None"})
                    if ph is not None:
                        ph["Metric"] = item
                        ph["Group_By"] = g_col
                        ph["Facet"] = "None"
                        posthoc_records.append(ph)
                    desc = compute_descriptive_stats(df_proc, item, g_col)
                    desc["Facet"] = "None"
                    descriptive_records.append(desc)

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
                    
                    aov, ph = plot_single_group_axes(
                        df=d_combo, value_col=item, group_col=combo_col, group_order=combined_order,
                        plot_type=ptype, ax=ax, title=f"{item} ({g1} & {g2})",
                        alpha=ALPHA, palette=palette, add_brackets=add_brackets,
                        strip_hue_col=strip_hue_col, strip_order=strip_order,
                        strip_palette_dict=strip_palette_dict, font_scale=font_scale
                    )
                    fig.tight_layout()
                    fig_path = os.path.join(ptype_dir, f"{clean_name}.png")
                    fig.savefig(fig_path, dpi=DPI, transparent=TRANSPARENT_BG, bbox_inches='tight')
                    plt.close(fig)

                    if ptype == active_plot_types[0]:
                        if aov is not None:
                            anova_records.append({**aov, "Metric": item, "Group_By": f"{g1}+{g2} (Combined)", "Facet": "None"})
                        if ph is not None:
                            ph["Metric"] = item
                            ph["Group_By"] = f"{g1}+{g2} (Combined)"
                            ph["Facet"] = "None"
                            posthoc_records.append(ph)
                        desc = compute_descriptive_stats(d_combo, item, combo_col)
                        desc["Facet"] = "None"
                        descriptive_records.append(desc)

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

                    # Pass 1: Render subplots without brackets to establish unified shared y-limits
                    temp_results = []
                    for ax, fval in zip(axes, facets):
                        sub_facet = df_proc.loc[df_proc[g2] == fval]
                        aov, ph = plot_single_group_axes(
                            df=sub_facet, value_col=item, group_col=g1, group_order=order1,
                            plot_type=ptype, ax=ax, title=f"{item} | {g2} = {fval}",
                            alpha=ALPHA, palette=palette, add_brackets=False,
                            strip_hue_col=strip_hue_col, strip_order=strip_order,
                            strip_palette_dict=strip_palette_dict, font_scale=font_scale
                        )
                        temp_results.append((ax, aov, ph, sub_facet, fval))
                        
                        if ptype == active_plot_types[0]:
                            if aov is not None:
                                anova_records.append({**aov, "Metric": item, "Group_By": g1, "Facet": f"{g2}={fval}"})
                            if ph is not None:
                                ph["Metric"] = item
                                ph["Group_By"] = g1
                                ph["Facet"] = f"{g2}={fval}"
                                posthoc_records.append(ph)
                            desc = compute_descriptive_stats(sub_facet, item, g1)
                            desc["Facet"] = f"{g2}={fval}"
                            descriptive_records.append(desc)

                    # Pass 2: Add brackets using shared y-limits
                    initial_ymin, initial_ymax = axes[0].get_ylim()
                    max_level_overall = 0
                    if add_brackets:
                        for ax, aov, ph, sub_facet, fval in temp_results:
                            if ph is not None:
                                sig_pairs = [
                                    (row["A"], row["B"], row["pval"], row.get("cohen_d", np.nan))
                                    for _, row in ph.iterrows()
                                    if row["pval"] < ALPHA and abs(row.get("cohen_d", 999)) >= MIN_EFFECT_SIZE_BRACKET
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
                    fig.savefig(fig_path, dpi=DPI, transparent=TRANSPARENT_BG, bbox_inches='tight')
                    plt.close(fig)

            else:
                raise ValueError(f"group_cols currently supports 1 or 2 columns (received {len(group_cols)}).")

    # 6. Format and Export Statistical Summary CSV Tables
    anova_df = pd.DataFrame(anova_records)
    posthoc_df = pd.concat(posthoc_records, ignore_index=True) if posthoc_records else pd.DataFrame()
    desc_df = pd.concat(descriptive_records, ignore_index=True) if descriptive_records else pd.DataFrame()

    if not anova_df.empty:
        if "p_unc" in anova_df.columns:
            anova_df["p_unc_formatted"] = anova_df["p_unc"].apply(format_p_value)
            anova_df["significance"] = anova_df["p_unc"].apply(p_to_stars)
        anova_path = os.path.join(output_dir, "welch_anova_results.csv")
        anova_df.to_csv(anova_path, index=False)
        print(f"Exported Welch's ANOVA results to: {anova_path}")

    if not posthoc_df.empty:
        if "pval" in posthoc_df.columns:
            posthoc_df["pval_formatted"] = posthoc_df["pval"].apply(format_p_value)
        # Reorder columns logically
        lead_cols = [c for c in ["Metric", "Group_By", "Facet", "A", "B", "N(A)", "N(B)", "mean(A)", "mean(B)", "diff", "pct_diff", "se", "T", "df", "pval", "pval_formatted", "significance", "cohen_d", "hedges_g", "effect_size_magnitude"] if c in posthoc_df.columns]
        other_cols = [c for c in posthoc_df.columns if c not in lead_cols]
        posthoc_df = posthoc_df[lead_cols + other_cols]
        posthoc_path = os.path.join(output_dir, "games_howell_pairwise_results.csv")
        posthoc_df.to_csv(posthoc_path, index=False)
        print(f"Exported Games-Howell pairwise results with Cohen's d & Hedges' g to: {posthoc_path}")

    if not desc_df.empty:
        desc_path = os.path.join(output_dir, "descriptive_statistics_summary.csv")
        desc_df.to_csv(desc_path, index=False)
        print(f"Exported descriptive statistics summary to: {desc_path}")

    print(f"\n=======================================================")
    print(f" Pipeline Complete! Visualizations & Stats saved to:")
    print(f" {output_dir}")
    print(f"=======================================================\n")
    return anova_df, posthoc_df, desc_df


# -----------------------------------------------------------------------------
# CLI & DEMO SCRIPT ENTRYPOINT
# -----------------------------------------------------------------------------

def map_vallon_sample_conditions(sample_id: Any) -> str:
    """Helper to map Vallon kidney sample IDs (string or numeric) to conditions."""
    try:
        s_num = int(sample_id)
        if s_num < 5:
            return "Ctrl/Veh"
        elif s_num < 9:
            return "Ctrl/SGLT2i"
        elif s_num < 16:
            return "WD/Veh"
        else:
            return "WD/SGLT2i"
    except (ValueError, TypeError):
        return str(sample_id)


def main():
    parser = argparse.ArgumentParser(description="Multi-Group Statistical Testing & Publication-Quality Visualization Pipeline")
    parser.add_argument("--input", "-i", type=str, default=DEFAULT_DATA_PATH, help="Path to input table (.parquet, .feather, .csv, .xlsx)")
    parser.add_argument("--output", "-o", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory for plots and statistical tables")
    parser.add_argument("--plot-type", type=str, default=PLOT_TYPE, choices=["violin", "bar", "both"], help="Plot style to render")
    parser.add_argument("--combine", action="store_true", help="Combine two group columns into single unified category x-axis")
    args, unknown = parser.parse_known_args()

    input_path = args.input
    output_dir = args.output
    plot_type = args.plot_type
    combine = args.combine or COMBINE_TWO_GROUPS

    print(f"Loading data from: {input_path}")
    if input_path.endswith(".parquet"):
        df = pd.read_parquet(input_path)
    elif input_path.endswith(".feather"):
        df = pd.read_feather(input_path)
    elif input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    elif input_path.endswith(".xlsx") or input_path.endswith(".xls"):
        df = pd.read_excel(input_path)
    else:
        try:
            df = pd.read_parquet(input_path)
        except Exception:
            df = pd.read_csv(input_path)

    print(f"Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns.")

    # -------------------------------------------------------------------------
    # DATA PRE-PROCESSING & CONDITION MAPPING
    # -------------------------------------------------------------------------
    if "Sample ID" in df.columns and "Condition" not in df.columns:
        df["Condition"] = df["Sample ID"].apply(map_vallon_sample_conditions)
        print(f"Mapped Sample IDs to Conditions: {df['Condition'].value_counts().to_dict()}")

    # -------------------------------------------------------------------------
    # USER CONFIGURABLE GROUPS & VALUES
    # (Easily customize these lists for your experiment)
    # -------------------------------------------------------------------------
    # Grouping columns: 1 column (e.g. ["Condition"]) or 2 columns (e.g. ["Condition", "Label"])
    group_cols = ["Condition", "Label"]
    
    # Selected metric columns of interest
    value_cols = [
        '% Sterols', '% Cardiolipins', '% Phospholipids', 
        '% Lysophospholipids', '% Glycerols', '% Sphingolipids', 
        '% SFA', '% PUFA', 
        'Protein_Turnover Mean', 'Lipid_Turnover Mean', 
        'Lipid_to_Protein Mean', 'Lipid_Unsaturation Mean'
    ]
    # Filter to only columns that actually exist in df
    value_cols = [c for c in value_cols if c in df.columns]
    if not value_cols:
        value_cols = df.select_dtypes(include=np.number).columns.tolist()[:10]
        print(f"Defaulting to first numeric columns: {value_cols}")

    group_order = ['Ctrl/Veh', 'Ctrl/SGLT2i', 'WD/Veh', 'WD/SGLT2i']
    unit_order = ['PT-S1/S2', 'PT-S3', 'cTAL', 'mTAL']
    
    col_order = {
        'Condition': group_order,
        'Label': unit_order
    }
    
    filter_values = {
        'Label': unit_order,
        'Condition': group_order
    }

    # Set up specific output subdirectory based on selected grouping
    run_output_dir = os.path.join(
        output_dir, 
        f"{group_cols[0]}_{group_cols[1]}" if len(group_cols) == 2 else group_cols[0]
    )

    run_group_stats_pipeline(
        df=df,
        group_cols=group_cols,
        value_cols=value_cols,
        col_order=col_order,
        plot_types=plot_type,
        combine_two_groups=combine,
        filter_values=filter_values,
        palette=PALETTE,
        add_brackets=True,
        output_dir=run_output_dir,
        strip_hue_col=STRIP_HUE_COL if STRIP_HUE_COL in df.columns else None,
        strip_palette=STRIP_PALETTE,
        save_legend=SAVE_STRIP_LEGEND,
        font_scale=FONT_SCALE
    )


if __name__ == "__main__":
    main()
