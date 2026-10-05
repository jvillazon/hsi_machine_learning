"""
Integrated Pipeline Plotting Toolkit & Statistical Analysis Suite
=================================================================
Consolidated visualization tools for multi-modal spatial biology, mass spectrometry,
and hyperspectral imaging datasets.

Features:
- Dataset Registry & Audit Logger (CSV, Parquet, TIF manifest)
- Statistical Group Comparisons (Violin / Box + Strip + Welch's ANOVA + Games-Howell + Significance Brackets)
- Paired & Multi-Group Feature Bar Plots (with automated error bars, significance brackets & compact insets)
- Metabolite & Marker Correlation Heatmaps & Feature Ranking Bar Charts
- Spatial Coordinate & Instance Distribution Mapping
- High-Dimensional UMAP Projections (Categorical & Continuous)
"""

import os
import re
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as PathEffects
import seaborn as sns
import pingouin as pg
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler

# Add parent directory to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent))

# ==============================================================================
# 1. DATASET REGISTRY & AUDIT LOGGER
# ==============================================================================

DATASET_REGISTRY = {
    "CODEX_WSI": {
        "description": "CODEX Single-Cell Clusters & WSI Kidney Multi-Modal Regions",
        "files": {
            "main_output": r"/Users/jorgevillazon/Library/CloudStorage/OneDrive-UCSanDiego/integrated_pipeline/WSI_data/WSI_data/CODEX_clusters-regions_masks_output_csv-test/output.csv",
            "umap_embeddings": r"/Users/jorgevillazon/Library/CloudStorage/OneDrive-UCSanDiego/integrated_pipeline/WSI_data/WSI_data/CODEX_clusters-regions_masks_output_csv-test/umap-nomid-output.csv",
            "fibrosis_data": r"/Users/jorgevillazon/Library/CloudStorage/OneDrive-UCSanDiego/integrated_pipeline/WSI_data/WSI_data/fib-output.csv",
            "percentages_csv": r"/Users/jorgevillazon/Library/CloudStorage/OneDrive-UCSanDiego/integrated_pipeline_backup/HSI_data/CODEX_clusters_masks_output_csv/percentages-output.csv",
        },
        "default_group_col": "group",
        "default_unit_col": "mask_name"
    },
    "VALLON_D2O_SGLT2I": {
        "description": "Vallon D2O SGLT2i Kidney Turnover & Headless SRS Metrics",
        "files": {
            "srs_metrics_csv": r"D:\VallonD2O\jorge_kidney_data\headless_srs_outputs\all_images_roi_metrics.csv",
            "turnover_parquet": r"D:\VallonD2O\jorge_kidney_data\headless_arithmetic_outputs\roi_metrics.parquet",
        },
        "default_group_col": "Condition",
        "default_unit_col": "Label"
    },
    "COLLAGEN_PROXIMITY": {
        "description": "Collagen Fiber Structure & Proximity Distance Maps",
        "files": {
            "coherence_csv": r"/Users/jorgevillazon/Documents/integrated_pipeline/journal_figures/proximity/S21-K21_198_3-proximity/collagen/masked_coherence.csv",
            "thickness_csv": r"/Users/jorgevillazon/Documents/integrated_pipeline/journal_figures/proximity/S21-K21_198_3-proximity/collagen/masked_thickness.csv",
            "distance_map_tif": r"/Users/jorgevillazon/Documents/integrated_pipeline/journal_figures/proximity/S21-K21_198_3-proximity/collagen/distance_to_background_um.tif",
        }
    },
    "SNRNA_INTEGRATION": {
        "description": "snRNA-seq UMAP Coordinates Integration",
        "files": {
            "umap_3712": r"/Users/jorgevillazon/OneDrive - UC San Diego/integrated_pipeline/snRNA/umap_coordinates_3712.csv",
            "umap_3723": r"/Users/jorgevillazon/OneDrive - UC San Diego/integrated_pipeline/snRNA/umap_coordinates_3723.csv",
        }
    }
}

DATASET_LOAD_LOG = []

def load_project_dataset(project_key, file_key, fallback_path=None, **read_kwargs):
    """
    Loads a dataset from the central registry, logging metadata for reproducibility.
    """
    if project_key not in DATASET_REGISTRY:
        raise KeyError(f"Project '{project_key}' not in DATASET_REGISTRY. Available: {list(DATASET_REGISTRY.keys())}")
    
    files = DATASET_REGISTRY[project_key]["files"]
    path = files.get(file_key, fallback_path)
    if path is None:
        raise KeyError(f"File key '{file_key}' not found for project '{project_key}' and no fallback provided.")
    
    if not os.path.exists(path) and fallback_path and os.path.exists(fallback_path):
        path = fallback_path

    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        df = pd.read_parquet(path, **read_kwargs)
    elif ext in [".csv", ".txt"]:
        df = pd.read_csv(path, **read_kwargs)
    else:
        raise ValueError(f"Unsupported file extension: {ext} for path {path}")

    log_entry = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Project": project_key,
        "Dataset Key": file_key,
        "File Path": path,
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Memory (MB)": round(df.memory_usage(deep=True).sum() / 1024**2, 2)
    }
    DATASET_LOAD_LOG.append(log_entry)
    print(f" Loaded [{project_key} :: {file_key}] -> {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df

def display_dataset_log():
    """Returns or displays a summary dataframe of all datasets loaded in the current session."""
    if not DATASET_LOAD_LOG:
        print("No datasets loaded in the current session log.")
        return pd.DataFrame()
    return pd.DataFrame(DATASET_LOAD_LOG)


# ==============================================================================
# 2. PRESET DICTIONARIES & TAXONOMY MAPPINGS
# ==============================================================================

UNIT_CLUSTERS = {
    "Cluster1": "Glomeruli | Podo, CD31",
    "Cluster2": "PT | LRP2",
    "Cluster3": "PT | LRP2, AQP1, CD90",
    "Cluster4": "PT | THY1",
    "Cluster5": "PT | PROM1/VCAM1",
    "Cluster6": "TDL | AQP1, Cyto8",
    "Cluster7": "TDL | AQP1, Cyto8",
    "Cluster8": "TAL | NaK/UMOD",
    "Cluster9": "TAL | UMOD",
    "Cluster10": "TAL | UMOD",
    "Cluster11": "DCT | SLC12a3",
    "Cluster12": "DN | Cyto8",
    "Cluster13": "DN | Cyto8",
    "Cluster14": "DN | Cyto8",
    "Cluster15": "DN | Cyto8, Bcat",
    "Cluster16": "DN | Cyto8, Bcat, Ecad",
    "Cluster17": "DN | Cyto8, Ecad",
    "Cluster18": "Endothelium | CD31",
    "Cluster19": "Endothelium | CD31, Podo",
    "Cluster20": "Pericytes | CD90, CD31(-)",
    "Cluster21": "VSMCs | aSMA",
    "Cluster22": "Immune | CD20",
    "Cluster23": "Immune - MP | CD45, CD206",
    "Cluster24": "Immune - T | CD3, CD4",
    "Cluster25": "Immune - T | CD3, CD8",
    "Cluster26": "Interstitial cells | IGFBP7"
}

CLUSTER_COLORS = {
    "Cluster1": "#eb2d13",
    "Cluster2": "#0a8416",
    "Cluster3": "#10c922",
    "Cluster4": "#13eb28",
    "Cluster5": "#7af487",
    "Cluster6": "#84530a",
    "Cluster7": "#f4c37a",
    "Cluster8": "#840a62",
    "Cluster9": "#f47ad2",
    "Cluster10": "#f47ad2",
    "Cluster11": "#ebd613",
    "Cluster12": "#0a3b84",
    "Cluster13": "#0e52b7",
    "Cluster14": "#0e52b7",
    "Cluster15": "#1369eb",
    "Cluster16": "#478af0",
    "Cluster17": "#7aabf4",
    "Cluster18": "#0a8484",
    "Cluster19": "#10c9c9",
    "Cluster20": "#35eeee",
    "Cluster21": "#7af4f4",
    "Cluster22": "#8b3103",
    "Cluster23": "#d44a05",
    "Cluster24": "#f9702a",
    "Cluster25": "#fba173",
    "Cluster26": "#9913eb"
}

FTU_MAP = {
    "Cluster1": "GL", "Cluster2": "PT", "Cluster3": "PT", "Cluster4": "PT", "Cluster5": "PT",
    "Cluster6": "TDL", "Cluster7": "TDL", "Cluster8": "TAL", "Cluster9": "TAL", "Cluster10": "TAL",
    "Cluster11": "DCT", "Cluster12": "DN", "Cluster13": "DN", "Cluster14": "DN", "Cluster15": "DN",
    "Cluster16": "DN", "Cluster17": "DN", "Cluster18": "Endo", "Cluster19": "Endo",
    "Cluster20": "Pericytes", "Cluster21": "VSMCs", "Cluster22": "Immune", "Cluster23": "MP",
    "Cluster24": "T", "Cluster25": "T", "Cluster26": "Interstitial"
}

FIB_MAP = {
    "CFIB": "FIB", "CMFIB": "FIB", "IMFIB": "FIB", "OMFIB": "FIB", "pvFIB": "FIB",
    "dFIB": "dFIB", "dIMFIB": "dFIB", "dOMFIB": "dFIB",
    "CMYOF": "MYOF", "pvMYOF": "MYOF", "IMpvMYOF": "MYOF"
}

GROUP_ORDER = ['Outer Cortex', 'Inner Cortex', 'Outer Stripe', 'Inner Stripe', 'Inner Medulla']
UNIT_ORDER = ['PT-S1/S2', 'PT-S3', 'cTAL', 'mTAL']

METRIC_GROUPS = {
    "ORR": ["Redox"],
    "CAR": ["CAR 18:2 (+H)", "CAR 18:1 (+H)", "CAR 18:0 (+H)"],
    "CL": ["CL 72:8 (-H)", "CL 74:11 (-H)"],
    "LPL": ["LPC 16:0 (+H)", "LPE 18:0 (-H)", "LPI 18:0 (-H)"],
    "PL (Sat)": ["PC 32:0 (+H)", "PE 16:0_18:0 (-H)", "PI 16:0_18:0 (-H)"],
    "PL (MUFA)": ["PC 16:0_18:1 (+H)", "PC 18:0_18:1 (+H)", "PA 36:1 (-H)", "PS 18:0_18:1 (-H)", "PI 34:1 (-H)", "PI 18:0_18:1 (-H)"],
    "PL (AA)": ["PE 16:0_20:4 (-H)", "PE 18:1_20:4 (-H)", "PE 18:1_20:3 (-H)", "PS 16:0_20:4 (-H)", "PS 18:0_20:4 (-H)", "PI 16:0_20:4 (-H)", "PI 18:1_20:3 (-H)", "PC 35:4 (+H)", "PC 36:5 (+H)", "PC 18:0_20:4 (+H)"],
    "PL (DHA)": ["PE 16:0_22:6 (-H)", "PG 18:1_22:6 (-H)"],
    "PL (PUFA)": ["PC O-38:5 (+H)", "PC 38:5 (+H)", "PC 46:9 (-CH3)", "PE 49:9 (-H)", "PC 48:9 (-CH3)"],
    "SM": ["SM 34:1;2O (+H)", "SM 38:1;20 (+H)", "SM 42:2;2O (+H)", "SM 42:1;2O (+H)", "SM 41:0;3O (+H)", "SM 42:0;3O (+H)", "SM 18:1;2O/16:0 (-CH3)"],
    "SHexCer": ["SHexCer 40:1;3O (-H)", "SHexCer 41:1;3O (-H)", "SHexCer 42:2;3O (-H)", "SHexCer 42:1;3O (-H)", "SHexCer 42:0;3O (-H)", "SHexCer 50:0;2O (-H)"],
    "Lipid Unsat.": ["Lipid_Unsaturation"],
    "Lipid/Protein": ["Lipid_to_Protein"],
    "Fib. Collagen": ["Collagen_Thickness", "Collagen_Coherence"]
}


# ==============================================================================
# 3. STATISTICAL ENGINE & HELPERS
# ==============================================================================

ALPHA = 0.05
FONT_SCALE = 1.5
STRIP_HUE_COL = "Sample ID"
STRIP_PALETTE = "tab20"
SAVE_STRIP_LEGEND = True
MAX_BRACKETS = None

def stars(p):
    """Converts a p-value float into a standard star annotation string."""
    if p is None or pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"

def format_p_value(p):
    """Formats numeric p-value into publication-ready string."""
    if pd.isna(p) or p is None:
        return ""
    if p == 0 or p < 1e-15:
        return "< 1e-15"
    if p < 0.001:
        return f"{p:.4e}"
    return f"{p:.4f}"

def run_welch_games_howell(data, value_col, group_col, alpha=ALPHA):
    """
    Runs Welch's ANOVA then Games-Howell post-hoc test (via pingouin) if significant.
    Returns (anova_dict_or_None, posthoc_df_or_None).
    """
    sub = data[[value_col, group_col]].dropna()
    counts = sub[group_col].value_counts()
    valid_groups = counts[counts >= 2].index
    sub = sub[sub[group_col].isin(valid_groups)]
    if sub[group_col].nunique() < 2:
        return None, None

    aov = pg.welch_anova(dv=value_col, between=group_col, data=sub)
    row = aov.iloc[0]
    anova = {
        "F": row["F"], "df1": row["ddof1"], "df2": row["ddof2"],
        "p_unc": row["p_unc"], "k": sub[group_col].nunique()
    }

    posthoc = None
    if anova["p_unc"] < alpha:
        posthoc = pg.pairwise_gameshowell(dv=value_col, between=group_col, data=sub)
    return anova, posthoc

def add_significance_brackets(ax, sig_pairs, order, initial_ymin=None, initial_ymax=None, max_brackets=MAX_BRACKETS):
    """
    Draws tiered significance brackets with star strings over categorical x-axes.
    """
    if not sig_pairs:
        return 0

    ymin = initial_ymin if initial_ymin is not None else ax.get_ylim()[0]
    ymax = initial_ymax if initial_ymax is not None else ax.get_ylim()[1]
    y_range = ymax - ymin
    y_start = ymax + 0.03 * y_range
    h = 0.08 * y_range

    sig_pairs = sorted(sig_pairs, key=lambda t: abs(order.index(t[0]) - order.index(t[1])))
    if max_brackets is not None and max_brackets > 0:
        sig_pairs = sig_pairs[:max_brackets]

    levels_occupied = []
    max_level_used = 0
    for g1, g2, p in sig_pairs:
        x1, x2 = order.index(g1), order.index(g2)
        left, right = min(x1, x2), max(x1, x2)
        
        assigned_level = 0
        while True:
            if assigned_level >= len(levels_occupied):
                levels_occupied.append([])
            overlap = any(ex_l <= right and left <= ex_r for ex_l, ex_r in levels_occupied[assigned_level])
            if not overlap:
                levels_occupied[assigned_level].append((left, right))
                break
            assigned_level += 1
            
        y = y_start + assigned_level * (h * 1.4)
        star_fs = max(8, int(10 * FONT_SCALE))
        ax.plot([x1, x1, x2, x2], [y, y + h * 0.3, y + h * 0.3, y], lw=1.2, c="black")
        ax.text((x1 + x2) / 2, y + h * 0.3, stars(p), ha="center", va="bottom", fontsize=star_fs)
        max_level_used = max(max_level_used, assigned_level + 1)

    if initial_ymin is None and initial_ymax is None:
        ax.set_ylim(ymin, y_start + max_level_used * (h * 1.4) + h * 2)
    return max_level_used

def save_strip_legend(order, palette_dict, title, output_dir):
    """Generates and saves a clean standalone image containing only the legend."""
    n_items = len(order)
    ncols = min(8, max(1, int(np.ceil(np.sqrt(n_items)))))
    nrows = int(np.ceil(n_items / ncols))
    
    fig, ax = plt.subplots(figsize=(max(3.5, 1.3 * ncols), max(1.2, 0.45 * nrows + 0.6)))
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette_dict.get(val, '#555555'),
                   markeredgecolor='black', markeredgewidth=0.8, markersize=10,
                   label=f"Sample {val}" if isinstance(val, (int, np.integer)) and "sample" not in str(title).lower() else str(val))
        for val in order
    ]
    ax.legend(handles=handles, title=title, loc='center', ncol=ncols,
              frameon=True, fontsize=10, title_fontsize=11, framealpha=0.9)
    ax.axis('off')
    fig.tight_layout()
    clean_title = re.sub(r"[^\w\s]", "", str(title)).strip().replace(" ", "_")
    legend_path = os.path.join(output_dir, f"legend_{clean_title}.png")
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(legend_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig)


# ==============================================================================
# 4. PLOTTING TOOL A: STATISTICAL GROUP COMPARISONS (BOX/VIOLIN + STRIP)
# ==============================================================================

def plot_single_grouping(df, value_col, group_order, group_col, ax=None, title=None, alpha=ALPHA,
                         palette="colorblind", add_brackets=True, strip_hue_col=None,
                         strip_order=None, strip_palette_dict=None):
    """
    Plots a single violin + strip plot with mean overlay and Welch ANOVA / Games-Howell brackets.
    """
    cols_to_keep = [value_col, group_col]
    if strip_hue_col and strip_hue_col in df.columns and strip_hue_col not in cols_to_keep:
        cols_to_keep.append(strip_hue_col)
    sub = df[cols_to_keep].dropna(subset=[value_col, group_col])
    order = [g for g in group_order if g in sub[group_col].unique()]
    if not order:
        order = list(sub[group_col].unique())

    title_fs = max(9, int(12 * FONT_SCALE))
    tick_fs  = max(6, int(8 * FONT_SCALE))
    text_fs  = max(7, int(9 * FONT_SCALE))

    if isinstance(palette, str):
        palette_dict = {cat: color for cat, color in zip(order, sns.color_palette(palette, len(order)))}
    elif isinstance(palette, (list, tuple)):
        palette_dict = {cat: palette[i % len(palette)] for i, cat in enumerate(order)}
    elif isinstance(palette, dict):
        palette_dict = palette
    else:
        palette_dict = palette

    if ax is None:
        _, ax = plt.subplots(figsize=(max(4, 1.4 * len(order)), 5))

    sns.violinplot(data=sub, x=group_col, y=value_col, order=order,
                   hue=group_col, hue_order=order, palette=palette_dict,
                   cut=0, inner="box", ax=ax, dodge=False, legend=False)

    if strip_hue_col and strip_hue_col in sub.columns:
        s_order = strip_order if strip_order is not None else sorted(sub[strip_hue_col].dropna().unique())
        s_pal = strip_palette_dict if strip_palette_dict is not None else {
            val: col for val, col in zip(s_order, sns.color_palette(STRIP_PALETTE, len(s_order)))
        }
        sns.stripplot(data=sub, x=group_col, y=value_col, order=order,
                      hue=strip_hue_col, hue_order=s_order, palette=s_pal,
                      dodge=False, jitter=0.2, size=4, alpha=0.75,
                      edgecolor="black", linewidth=0.5, ax=ax, legend=False, zorder=2)
    else:
        sns.stripplot(data=sub, x=group_col, y=value_col, order=order,
                      hue=group_col, hue_order=order, palette=palette_dict,
                      dodge=False, jitter=0.2, size=4, alpha=0.6,
                      edgecolor="black", linewidth=0.5, ax=ax, legend=False, zorder=2)

    # Mean overlay
    for idx, category in enumerate(order):
        cat_data = sub.loc[sub[group_col] == category, value_col]
        if len(cat_data) > 0:
            mean_val = cat_data.mean()
            ax.plot(idx, mean_val, marker='o', color='white', markeredgecolor='black', 
                    markersize=6, zorder=4, label='mean' if idx == 0 else "")
            ymin, ymax = ax.get_ylim()
            y_offset = 0.02 * (ymax - ymin)
            text = ax.text(idx, mean_val + y_offset, f'{mean_val:.2f}', ha='center', va='bottom', fontsize=text_fs, zorder=4)
            text.set_path_effects([PathEffects.withStroke(linewidth=4, foreground='white')])

    ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.7, zorder=0)

    anova, posthoc = run_welch_games_howell(sub, value_col, group_col, alpha)
    if anova is not None:
        ax.set_title(f"{title or value_col}\nWelch's ANOVA p = {anova['p_unc']:.4g}", fontsize=title_fs)
        if posthoc is not None and add_brackets:
            sig_pairs = [(row["A"], row["B"], row["pval"]) for _, row in posthoc.iterrows() if row["pval"] < alpha]
            add_significance_brackets(ax, sig_pairs, order)
    else:
        ax.set_title(f"{title or value_col}\n(n/a)", fontsize=title_fs)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=tick_fs)
    return anova, posthoc

def plot_group_comparisons(df, value_cols, group_cols, col_order=None, combine=False,
                           filter_values=None, palette="colorblind", add_brackets=True,
                           output_dir="./plots", strip_hue_col=None, strip_palette="tab20",
                           save_legend=True):
    """
    Main driver for generating multi-feature statistical violin + strip plots.
    Saves welch_anova_results.csv and games_howell_results.csv.
    """
    os.makedirs(output_dir, exist_ok=True)
    d = df.copy()
    if filter_values:
        for col, keep_values in filter_values.items():
            if col in d.columns:
                d = d[d[col].isin(keep_values)]

    if col_order is None:
        col_order = {g: list(d[g].dropna().unique()) for g in group_cols if g in d.columns}

    strip_order = None
    strip_palette_dict = None
    if strip_hue_col and strip_hue_col in d.columns:
        try:
            strip_order = sorted(d[strip_hue_col].dropna().unique())
        except TypeError:
            strip_order = list(d[strip_hue_col].dropna().unique())
        colors = sns.color_palette(strip_palette, len(strip_order))
        strip_palette_dict = {val: col for val, col in zip(strip_order, colors)}
        if save_legend:
            save_strip_legend(strip_order, strip_palette_dict, title=strip_hue_col, output_dir=output_dir)

    anova_records = []
    posthoc_records = []

    for item in value_cols:
        if item not in d.columns:
            continue
        item_name = re.sub(r"[^\w\s]", "", item).strip().replace(" ", "_")
        
        if len(group_cols) == 1:
            group_col = group_cols[0]
            fig, ax = plt.subplots(figsize=(max(4, 1.2 * len(col_order.get(group_col, []))), 6))
            anova, posthoc = plot_single_grouping(
                d, item, col_order.get(group_col, []), group_col, ax=ax, palette=palette,
                add_brackets=add_brackets, strip_hue_col=strip_hue_col, strip_order=strip_order,
                strip_palette_dict=strip_palette_dict
            )
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f"{item_name}.png"), dpi=300, transparent=True)
            plt.close(fig)

            if anova is not None:
                anova_records.append({**anova, "item": item, "group_by": group_col, "facet": None})
            if posthoc is not None:
                posthoc["item"] = item
                posthoc["group_by"] = group_col
                posthoc["facet"] = None
                posthoc_records.append(posthoc)

        elif len(group_cols) == 2:
            g1, g2 = group_cols
            if combine:
                combo_col = f"{g1}__{g2}"
                d[combo_col] = d[g1].astype(str) + " | " + d[g2].astype(str)
                order = [f"{g1_val} | {g2_val}" for g1_val in col_order.get(g1, []) for g2_val in col_order.get(g2, [])]
                fig, ax = plt.subplots(figsize=(max(4, 1.4 * len(order)), 7))
                anova, posthoc = plot_single_grouping(
                    d, item, order, combo_col, ax=ax, title=f"{item} by {g1} & {g2}", palette=palette,
                    add_brackets=add_brackets, strip_hue_col=strip_hue_col, strip_order=strip_order,
                    strip_palette_dict=strip_palette_dict
                )
                fig.tight_layout()
                fig.savefig(os.path.join(output_dir, f"{item_name}.png"), dpi=300, transparent=True)
                plt.close(fig)

                if anova is not None:
                    anova_records.append({**anova, "item": item, "group_by": f"{g1}+{g2} (combined)", "facet": None})
                if posthoc is not None:
                    posthoc["item"] = item
                    posthoc["group_by"] = f"{g1}+{g2} (combined)"
                    posthoc["facet"] = None
                    posthoc_records.append(posthoc)
            else:
                facets = [fval for fval in col_order.get(g2, []) if not d[d[g2] == fval].empty]
                if not facets:
                    continue
                fig, axes = plt.subplots(len(facets), 1, figsize=(max(6, 1.6 * len(col_order.get(g1, []))), 5.5 * len(facets)),
                                         sharey=True, sharex=True)
                axes = [axes] if len(facets) == 1 else list(axes)
                temp_results = []
                for ax, fval in zip(axes, facets):
                    sub = d.loc[d[g2] == fval]
                    a, p = plot_single_grouping(sub, item, col_order.get(g1, []), g1, ax=ax,
                                                 title=f"{item} | {g2}={fval}", palette=palette,
                                                 add_brackets=False, strip_hue_col=strip_hue_col,
                                                 strip_order=strip_order, strip_palette_dict=strip_palette_dict)
                    temp_results.append((ax, a, p, sub))
                    if a is not None:
                        anova_records.append({**a, "item": item, "group_by": g1, "facet": f"{g2}={fval}"})
                    if p is not None:
                        p["item"] = item
                        p["group_by"] = g1
                        p["facet"] = f"{g2}={fval}"
                        posthoc_records.append(p)

                initial_ymin, initial_ymax = axes[0].get_ylim()
                max_level_overall = 0
                if add_brackets:
                    for ax, a, p, sub in temp_results:
                        if p is not None:
                            sig_pairs = [(row["A"], row["B"], row["pval"]) for _, row in p.iterrows() if row["pval"] < ALPHA]
                            levels = add_significance_brackets(ax, sig_pairs, col_order.get(g1, []),
                                                               initial_ymin=initial_ymin, initial_ymax=initial_ymax)
                            max_level_overall = max(max_level_overall, levels)

                if max_level_overall > 0:
                    y_range = initial_ymax - initial_ymin
                    y_start = initial_ymax + 0.03 * y_range
                    h = 0.08 * y_range
                    final_ymax = y_start + max_level_overall * (h * 1.4) + h * 2
                    axes[0].set_ylim(initial_ymin, final_ymax)

                fig.tight_layout()
                fig.savefig(os.path.join(output_dir, f"{item_name}.png"), dpi=300, transparent=True)
                plt.close(fig)

    anova_df = pd.DataFrame([{k: v for k, v in rec.items() if k != "labels"} for rec in anova_records])
    posthoc_df = pd.concat(posthoc_records, ignore_index=True) if posthoc_records else pd.DataFrame()

    if not anova_df.empty and "p_unc" in anova_df.columns:
        anova_df["p_unc_formatted"] = anova_df["p_unc"].apply(format_p_value)
    if not posthoc_df.empty and "pval" in posthoc_df.columns:
        posthoc_df["pval_formatted"] = posthoc_df["pval"].apply(format_p_value)

    anova_df.to_csv(os.path.join(output_dir, "welch_anova_results.csv"), index=False)
    posthoc_df.to_csv(os.path.join(output_dir, "games_howell_results.csv"), index=False)
    print(f"✅ Group comparison plots and statistics saved to: {output_dir}")
    return anova_df, posthoc_df


# ==============================================================================
# 5. PLOTTING TOOL B: PAIRED FEATURE BARS & COMPACT INSETS
# ==============================================================================

def plot_feature_bars(df, features, group_col, groups, pvals=None, feature_display_names=None,
                      colors=None, yerr='std', figsize=None, save_path=None):
    """
    Plots paired or multi-group bar charts across multiple features with error bars & significance.
    """
    valid_feats = [f for f in features if f in df.columns]
    if not valid_feats:
        raise ValueError("None of the specified features exist in the dataframe.")

    sub = df[df[group_col].isin(groups)].copy()
    means_df = sub.groupby(group_col)[valid_feats].mean().loc[groups]
    if yerr == 'sem':
        errs_df = sub.groupby(group_col)[valid_feats].sem().loc[groups]
    else:
        errs_df = sub.groupby(group_col)[valid_feats].std().loc[groups]

    n_feats = len(valid_feats)
    n_groups = len(groups)
    if colors is None:
        colors = sns.color_palette("Set2", n_groups)

    if figsize is None:
        figsize = (max(8, 0.9 * n_feats), 4.2)
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    x = np.arange(n_feats)
    total_width = 0.7
    bar_width = total_width / n_groups

    for i, group in enumerate(groups):
        offset = (i - (n_groups - 1) / 2) * bar_width
        ax.bar(
            x + offset, means_df.loc[group], yerr=errs_df.loc[group],
            width=bar_width, label=group, color=colors[i % len(colors)],
            edgecolor='black', linewidth=1.2, capsize=3.5,
            error_kw=dict(lw=1.2, capthick=1.2), zorder=3
        )

    # Add Significance Stars/Brackets
    if pvals is not None and n_groups == 2:
        y_min, y_max = ax.get_ylim()
        y_span = y_max - y_min
        bracket_h = 0.03 * y_span
        for idx, feat in enumerate(valid_feats):
            pval = pvals.get(feat, None)
            star_str = stars(pval)
            if star_str != "":
                top_y = max(means_df.loc[g, feat] + (errs_df.loc[g, feat] if pd.notna(errs_df.loc[g, feat]) else 0) for g in groups)
                y_bracket = top_y + 0.04 * y_span
                x1 = idx - bar_width / 2
                x2 = idx + bar_width / 2
                ax.plot([x1, x1, x2, x2], [y_bracket, y_bracket + bracket_h, y_bracket + bracket_h, y_bracket],
                        color='black', lw=1.2, zorder=4)
                ax.text((x1 + x2) / 2, y_bracket + bracket_h, star_str,
                        ha='center', va='bottom', fontsize=11, fontweight='bold', zorder=5)

    display_labels = [feature_display_names.get(f, f) if feature_display_names else f for f in valid_feats]
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=0 if n_feats <= 8 else 30, ha='center', fontsize=11, fontweight='bold')
    ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.7)

    curr_ymin, curr_ymax = ax.get_ylim()
    ax.set_ylim(curr_ymin, curr_ymax + 0.12 * (curr_ymax - curr_ymin))
    ax.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=10, loc='upper right')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, transparent=True, dpi=300)
        print(f"Saved feature bars plot to: {save_path}")
    return fig, ax

def plot_compact_inset(df, features, group_name, group_col='group', colors=None,
                       ylim=(-2, 5), yticks=[-2, 0, 2, 4], save_filename=None):
    """
    Plots a compact, borderless square bar chart inset with error bars for overlay in figures.
    """
    sub = df[df[group_col] == group_name]
    means = sub[features].mean()
    stds = sub[features].std()

    fig, ax = plt.subplots(figsize=(3.4, 3.4), dpi=300)
    if colors is None:
        colors = sns.husl_palette(n_colors=len(features))

    x_pos = np.arange(len(features))
    ax.bar(
        x_pos, means, yerr=stds, width=0.75, color=colors,
        edgecolor='black', linewidth=1.3, capsize=3.5,
        error_kw=dict(lw=1.3, capthick=1.3, ecolor='black'), zorder=3
    )

    if ylim:
        ax.set_ylim(ylim)
    ax.axhline(0, color='black', linewidth=1.0, zorder=2)
    if yticks:
        ax.set_yticks(yticks)

    ax.tick_params(axis='y', which='major', labelsize=13, width=1.5, length=5, labelcolor='black')
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(1.5)
        ax.spines[spine].set_color('black')

    ax.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.8, zorder=0)
    ax.set_title(group_name, fontsize=14, fontweight='bold', pad=8, color='black')
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    plt.tight_layout(pad=0.2)

    if save_filename:
        os.makedirs(os.path.dirname(save_filename) if os.path.dirname(save_filename) else ".", exist_ok=True)
        plt.savefig(save_filename, dpi=300, transparent=True, bbox_inches='tight', pad_inches=0.02)
        print(f"Saved compact inset to: {save_filename}")
    return fig, ax


# ==============================================================================
# 6. PLOTTING TOOL C: CORRELATION MATRICES & HEATMAPS
# ==============================================================================

def plot_correlation_matrix(df, features=None, method='spearman', cmap='RdBu_r',
                            vmin=-1.0, vmax=1.0, annot=False, title="Correlation Matrix",
                            figsize=(14, 12), save_path=None):
    """
    Computes and plots a publication-ready correlation matrix heatmap.
    """
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
        features = [f for f in features if 'centroid' not in f.lower() and f not in ['Instance_Label', 'Sample ID']]

    corr = df[features].corr(method=method)
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    sns.heatmap(corr, cmap=cmap, vmin=vmin, vmax=vmax, center=0, annot=annot,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title(f"{title} ({method.capitalize()})", fontsize=16, fontweight='bold', pad=15)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, transparent=True, dpi=300)
        print(f"Saved correlation matrix to: {save_path}")
    return corr, fig, ax

def plot_target_correlations(df, target_metric, feature_cols=None, method='spearman',
                             top_n=20, title=None, save_path=None):
    """
    Calculates and ranks feature correlations against a specific target metric (e.g. Collagen_Thickness).
    """
    if target_metric not in df.columns:
        raise ValueError(f"Target metric '{target_metric}' not found in dataframe.")

    if feature_cols is None:
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_metric]

    corrs = []
    for col in feature_cols:
        sub = df[[target_metric, col]].dropna()
        if len(sub) > 5:
            if method == 'spearman':
                r, p = spearmanr(sub[target_metric], sub[col])
            else:
                r, p = pearsonr(sub[target_metric], sub[col])
            corrs.append({"feature": col, "correlation": r, "pval": p, "stars": stars(p)})

    corr_df = pd.DataFrame(corrs).sort_values(by="correlation", ascending=False)
    plot_subset = pd.concat([corr_df.head(top_n // 2), corr_df.tail(top_n // 2)]) if len(corr_df) > top_n else corr_df

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(plot_subset))), dpi=300)
    colors = ['#c75b5b' if r >= 0 else '#5b8ac7' for r in plot_subset['correlation']]
    ax.barh(plot_subset['feature'], plot_subset['correlation'], color=colors, edgecolor='black', linewidth=0.8)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel(f"{method.capitalize()} Correlation (r)", fontsize=11, fontweight='bold')
    ax.set_title(title or f"Correlations with {target_metric}", fontsize=13, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, transparent=True, dpi=300)
        print(f"Saved target correlations plot to: {save_path}")
    return corr_df, fig, ax


# ==============================================================================
# 7. PLOTTING TOOL D: SPATIAL & CATEGORY DISTRIBUTIONS
# ==============================================================================

def plot_spatial_coordinates(df, x_col='centroid_x', y_col='centroid_y', hue_col='mask_name',
                             palette=None, title="Spatial Distribution", s=15, alpha=0.85,
                             figsize=(9, 8), save_path=None):
    """
    Plots single-cell or tissue instance centroid coordinates colored by cluster/unit.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, palette=palette,
                    s=s, alpha=alpha, edgecolor='none', ax=ax)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("X coordinate (px / um)", fontsize=11)
    ax.set_ylabel("Y coordinate (px / um)", fontsize=11)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, fontsize=9)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, transparent=True, dpi=300)
        print(f"Saved spatial coordinates plot to: {save_path}")
    return fig, ax

def plot_category_distribution(df, cat_col, group_col=None, normalize=False,
                               figsize=(10, 5), save_path=None):
    """
    Plots stacked or grouped bar charts of cell/instance counts across anatomical groups.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    if group_col:
        counts = pd.crosstab(df[cat_col], df[group_col], normalize='columns' if normalize else False)
        counts.plot(kind='bar', stacked=True, ax=ax, edgecolor='black', linewidth=0.8)
        ax.set_ylabel("Proportion" if normalize else "Instance Count", fontsize=11, fontweight='bold')
    else:
        df[cat_col].value_counts().plot(kind='bar', ax=ax, color='skyblue', edgecolor='black', linewidth=0.8)
        ax.set_ylabel("Count", fontsize=11, fontweight='bold')

    ax.set_xlabel(cat_col, fontsize=11, fontweight='bold')
    ax.set_title(f"Distribution of {cat_col}" + (f" across {group_col}" if group_col else ""), fontsize=13, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, transparent=True, dpi=300)
        print(f"Saved category distribution plot to: {save_path}")
    return fig, ax


# ==============================================================================
# 8. PLOTTING TOOL E: UMAP & DIMENSIONALITY REDUCTION
# ==============================================================================

def plot_umap_embedding(df, umap_1_col='umap_1', umap_2_col='umap_2', hue_col=None,
                        palette=None, is_continuous=False, cmap='viridis',
                        title="UMAP Projection", s=18, alpha=0.85, figsize=(8, 7),
                        save_path=None):
    """
    Publication-quality UMAP visualization with categorical palettes or continuous feature heatmaps.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    if is_continuous and hue_col:
        sc = ax.scatter(df[umap_1_col], df[umap_2_col], c=df[hue_col], cmap=cmap,
                        s=s, alpha=alpha, edgecolor='none')
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(hue_col, fontsize=11, fontweight='bold')
    else:
        sns.scatterplot(data=df, x=umap_1_col, y=umap_2_col, hue=hue_col, palette=palette,
                        s=s, alpha=alpha, edgecolor='none', ax=ax)
        if hue_col:
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, fontsize=9)

    ax.set_xlabel("UMAP 1", fontsize=12, fontweight='bold')
    ax.set_ylabel("UMAP 2", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    sns.despine(ax=ax)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, transparent=True, dpi=300)
        print(f"Saved UMAP plot to: {save_path}")
    return fig, ax
