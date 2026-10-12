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
from matplotlib.lines import Line2D
import seaborn as sns
import pingouin as pg
from scipy.stats import spearmanr, pearsonr, gaussian_kde, ttest_ind
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

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
    "Cluster3": "PT | THY1",
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

UNIT_COLORS = {UNIT_CLUSTERS.get(c, c): color for c, color in CLUSTER_COLORS.items()}
for c, color in CLUSTER_COLORS.items():
    UNIT_COLORS[c] = color

REGION_COLORS = {
    'Outer Cortex': '#4e79a7',
    'Inner Cortex': '#f28e2b',
    'Outer Stripe': '#59a14f',
    'Inner Stripe': '#edc948',
    'Inner Medulla': '#b07aa1'
}
GROUP_COLORS = REGION_COLORS

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

def compute_cohens_d(group_a, group_b, weighted=True):
    """Computes Cohen's d and Hedges' g effect sizes."""
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan
    var1, var2 = np.var(a, ddof=1), np.var(b, ddof=1)
    if weighted:
        s_pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / max(1, (n1 + n2 - 2)))
    else:
        s_pooled = np.sqrt((var1 + var2) / 2.0)
    if s_pooled == 0 or np.isnan(s_pooled):
        return 0.0, 0.0
    d = (np.mean(a) - np.mean(b)) / s_pooled
    g = d * (1.0 - (3.0 / max(1.0, (4.0 * (n1 + n2) - 9.0))))
    return float(d), float(g)

def interpret_effect_size(d):
    """Cohen's qualitative effect size categories."""
    if pd.isna(d) or d is None:
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

def run_welch_games_howell(data, value_col, group_col, alpha=ALPHA, weighted_pooled_sd=True):
    """
    Runs Welch's ANOVA then Games-Howell post-hoc test with Cohen's d effect sizes.
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
        "F": float(row["F"]), "df1": float(row["ddof1"]), "df2": float(row["ddof2"]),
        "p_unc": float(row["p_unc"]), "k": sub[group_col].nunique()
    }

    posthoc = None
    if anova["p_unc"] < alpha:
        posthoc = pg.pairwise_gameshowell(dv=value_col, between=group_col, data=sub)
        if posthoc is not None and not posthoc.empty:
            cohens_d_list = []
            hedges_g_list = []
            mean_a_list = []
            mean_b_list = []
            pct_diff_list = []
            effect_interp_list = []
            n_a_list = []
            n_b_list = []
            for _, r in posthoc.iterrows():
                val_a = sub.loc[sub[group_col] == r['A'], value_col].values
                val_b = sub.loc[sub[group_col] == r['B'], value_col].values
                d, g = compute_cohens_d(val_a, val_b, weighted=weighted_pooled_sd)
                mean_a = float(r.get('mean(A)', np.mean(val_a))) if 'mean(A)' in r else float(np.mean(val_a))
                mean_b = float(r.get('mean(B)', np.mean(val_b))) if 'mean(B)' in r else float(np.mean(val_b))
                pct_diff = ((mean_a - mean_b) / mean_b * 100.0) if mean_b != 0 else np.nan
                mean_a_list.append(mean_a)
                mean_b_list.append(mean_b)
                cohens_d_list.append(d)
                hedges_g_list.append(g)
                pct_diff_list.append(pct_diff)
                effect_interp_list.append(interpret_effect_size(d))
                n_a_list.append(len(val_a))
                n_b_list.append(len(val_b))
            posthoc['mean(A)'] = mean_a_list
            posthoc['mean(B)'] = mean_b_list
            posthoc['N(A)'] = n_a_list
            posthoc['N(B)'] = n_b_list
            posthoc['cohen_d'] = cohens_d_list
            posthoc['hedges_g'] = hedges_g_list
            posthoc['pct_diff'] = pct_diff_list
            posthoc['effect_size_magnitude'] = effect_interp_list
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

def plot_feature_dumbbells(df, features, group_col, groups, pvals=None,
                           feature_display_names=None, colors=None,
                           orient='h', show_pvals=True, pval_format='star',
                           dot_size=110, line_width=2.5, line_color='#7f7f7f',
                           show_error=False, yerr='sem', show_values=False,
                           figsize=None, title=None, xlabel=None, ylabel=None,
                           save_path=None):
    """
    Plots a publication-grade dumbbell (connected dot) chart comparing two groups 
    across multiple features/metrics, displaying group means and statistical significance.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset containing features and group_col.
    features : list of str
        List of numeric feature column names to compare.
    group_col : str
        Column name specifying group assignments (e.g., 'group').
    groups : list of str
        The two group labels to compare (e.g. ['Inner Cortex', 'Outer Cortex']).
    pvals : dict, optional
        Dictionary mapping feature names to precalculated p-values. If None, p-values are 
        automatically calculated using Welch's t-test (scipy.stats.ttest_ind with equal_var=False).
    feature_display_names : dict, optional
        Dictionary mapping feature column names to clean publication display names.
    colors : list or dict, optional
        Colors for the two groups. E.g., ['#c75b5b', '#911c1c'] or {'Inner Cortex': '#c75b5b', ...}.
    orient : {'h', 'v'}, default 'h'
        Orientation: 'h' (horizontal, features on Y-axis) or 'v' (vertical, features on X-axis).
    show_pvals : bool, default True
        Whether to display significance annotations (stars or formatted numbers).
    pval_format : {'star', 'numeric', 'both'}, default 'star'
        Format for significance labels: 'star' (***, **, *, ns), 'numeric' (p = 0.0012), or 'both'.
    dot_size : int, default 110
        Size (area) of the dumbbell endpoint markers.
    line_width : float, default 2.5
        Width of the connecting line segment between the two group means.
    line_color : str, default '#7f7f7f'
        Color of the connecting line segment.
    show_error : bool, default False
        Whether to draw error bars (SEM or STD) around group means.
    yerr : {'sem', 'std'}, default 'sem'
        Type of error bars to plot if show_error=True.
    show_values : bool, default False
        Whether to print numerical mean values directly next to each dot.
    figsize : tuple, optional
        Matplotlib figure size.
    title : str, optional
        Main plot title.
    xlabel / ylabel : str, optional
        Custom axis labels.
    save_path : str, optional
        File path to save the high-res figure (png/pdf).

    Returns:
    --------
    fig, ax, stats_df : tuple
        Matplotlib Figure, Axes, and a summary DataFrame containing means, delta, and p-values.
    """
    valid_feats = [f for f in features if f in df.columns]
    if not valid_feats:
        raise ValueError("None of the specified features exist in the dataframe.")
    if len(groups) != 2:
        raise ValueError(f"Dumbbell plot requires exactly 2 groups to compare, got {len(groups)}: {groups}")

    g1, g2 = groups[0], groups[1]
    sub = df[df[group_col].isin(groups)].copy()

    # Determine group colors
    if colors is None:
        c1, c2 = '#c75b5b', '#911c1c'
    elif isinstance(colors, dict):
        c1 = colors.get(g1, '#c75b5b')
        c2 = colors.get(g2, '#911c1c')
    elif isinstance(colors, (list, tuple)) and len(colors) >= 2:
        c1, c2 = colors[0], colors[1]
    else:
        c1, c2 = '#c75b5b', '#911c1c'

    g1_sub = sub[sub[group_col] == g1]
    g2_sub = sub[sub[group_col] == g2]

    # Calculate statistics
    stats_records = []
    for feat in valid_feats:
        s1_data = g1_sub[feat].dropna()
        s2_data = g2_sub[feat].dropna()
        m1 = s1_data.mean() if len(s1_data) > 0 else np.nan
        m2 = s2_data.mean() if len(s2_data) > 0 else np.nan

        if yerr == 'sem':
            e1 = s1_data.sem() if len(s1_data) > 1 else 0
            e2 = s2_data.sem() if len(s2_data) > 1 else 0
        else:
            e1 = s1_data.std() if len(s1_data) > 1 else 0
            e2 = s2_data.std() if len(s2_data) > 1 else 0

        # P-value calculation or retrieval
        if pvals is not None and feat in pvals:
            pval = pvals[feat]
        elif len(s1_data) >= 2 and len(s2_data) >= 2:
            try:
                _, pval = ttest_ind(s1_data, s2_data, equal_var=False)
            except Exception:
                pval = np.nan
        else:
            pval = np.nan

        star_str = stars(pval)
        if pval_format == 'numeric':
            sig_label = f"p={format_p_value(pval)}" if pd.notna(pval) else ""
        elif pval_format == 'both':
            sig_label = f"{star_str} (p={format_p_value(pval)})" if pd.notna(pval) else ""
        else:
            sig_label = star_str

        disp_name = feature_display_names.get(feat, feat) if feature_display_names else feat
        stats_records.append({
            'Feature': feat,
            'Display_Name': disp_name,
            f'{g1}_Mean': m1,
            f'{g2}_Mean': m2,
            f'{g1}_Error': e1,
            f'{g2}_Error': e2,
            'Delta (G1-G2)': m1 - m2 if pd.notna(m1) and pd.notna(m2) else np.nan,
            'pval': pval,
            'pval_formatted': format_p_value(pval),
            'Significance': star_str,
            'sig_label': sig_label
        })

    stats_df = pd.DataFrame(stats_records)
    n_feats = len(valid_feats)

    if orient == 'h':
        if figsize is None:
            figsize = (9, max(4.5, 0.55 * n_feats + 1.2))
        fig, ax = plt.subplots(figsize=figsize, dpi=300)
        y_pos = np.arange(n_feats)

        all_vals = []
        for idx, row in stats_df.iterrows():
            y = y_pos[idx]
            m1, m2 = row[f'{g1}_Mean'], row[f'{g2}_Mean']
            e1, e2 = row[f'{g1}_Error'], row[f'{g2}_Error']
            all_vals.extend([m1, m2])

            # Draw connecting dumbbell bar
            if pd.notna(m1) and pd.notna(m2):
                ax.plot([m1, m2], [y, y], color=line_color, lw=line_width, zorder=2, solid_capstyle='round')

            # Draw optional error bars
            if show_error:
                if pd.notna(m1):
                    ax.errorbar(m1, y, xerr=e1, fmt='none', ecolor=c1, elinewidth=1.2, capsize=3, zorder=3)
                if pd.notna(m2):
                    ax.errorbar(m2, y, xerr=e2, fmt='none', ecolor=c2, elinewidth=1.2, capsize=3, zorder=3)

            # Draw endpoint markers
            if pd.notna(m1):
                ax.scatter(m1, y, color=c1, s=dot_size, edgecolor='black', linewidth=1.2, zorder=4,
                           label=g1 if idx == 0 else "")
            if pd.notna(m2):
                ax.scatter(m2, y, color=c2, s=dot_size, edgecolor='black', linewidth=1.2, zorder=4,
                           label=g2 if idx == 0 else "")

            # Show numeric values near dots
            if show_values:
                if pd.notna(m1):
                    ax.text(m1, y + 0.18, f"{m1:.2f}", ha='center', va='bottom', fontsize=8.5, color=c1, fontweight='bold')
                if pd.notna(m2):
                    ax.text(m2, y + 0.18, f"{m2:.2f}", ha='center', va='bottom', fontsize=8.5, color=c2, fontweight='bold')

        # Formatting & Labels
        display_labels = stats_df['Display_Name'].tolist()
        ax.set_yticks(y_pos)
        ax.set_yticklabels(display_labels, fontsize=11, fontweight='bold')
        ax.invert_yaxis()  # Top-down feature ordering

        # Set X-axis range and add significance labels on the right
        valid_vals = [v for v in all_vals if pd.notna(v)]
        if valid_vals:
            min_val, max_val = min(valid_vals), max(valid_vals)
            val_span = max_val - min_val if max_val != min_val else 1.0
            ax.set_xlim(min_val - 0.08 * val_span, max_val + (0.22 if show_pvals else 0.08) * val_span)
        else:
            val_span = 1.0

        if show_pvals:
            for idx, row in stats_df.iterrows():
                y = y_pos[idx]
                m1, m2 = row[f'{g1}_Mean'], row[f'{g2}_Mean']
                sig_text = row['sig_label']
                if sig_text:
                    rightmost = max(m for m in [m1, m2] if pd.notna(m)) if pd.notna(m1) or pd.notna(m2) else 0
                    ax.text(rightmost + 0.04 * val_span, y, sig_text,
                            va='center', ha='left', fontsize=11, fontweight='bold',
                            color='#222222', zorder=5)

        ax.grid(True, which='major', axis='x', linestyle='--', alpha=0.5, zorder=0)
        ax.axvline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.6, zorder=1)
        ax.set_xlabel(xlabel or "Mean Quantification / Relative Intensity", fontsize=11, fontweight='bold', labelpad=8)
        ax.tick_params(axis='both', which='major', labelsize=10.5)

    else:  # Vertical orientation ('v')
        if figsize is None:
            figsize = (max(8, 0.85 * n_feats + 1.2), 5.5)
        fig, ax = plt.subplots(figsize=figsize, dpi=300)
        x_pos = np.arange(n_feats)

        all_vals = []
        for idx, row in stats_df.iterrows():
            x = x_pos[idx]
            m1, m2 = row[f'{g1}_Mean'], row[f'{g2}_Mean']
            e1, e2 = row[f'{g1}_Error'], row[f'{g2}_Error']
            all_vals.extend([m1, m2])

            if pd.notna(m1) and pd.notna(m2):
                ax.plot([x, x], [m1, m2], color=line_color, lw=line_width, zorder=2, solid_capstyle='round')

            if show_error:
                if pd.notna(m1):
                    ax.errorbar(x, m1, yerr=e1, fmt='none', ecolor=c1, elinewidth=1.2, capsize=3, zorder=3)
                if pd.notna(m2):
                    ax.errorbar(x, m2, yerr=e2, fmt='none', ecolor=c2, elinewidth=1.2, capsize=3, zorder=3)

            if pd.notna(m1):
                ax.scatter(x, m1, color=c1, s=dot_size, edgecolor='black', linewidth=1.2, zorder=4,
                           label=g1 if idx == 0 else "")
            if pd.notna(m2):
                ax.scatter(x, m2, color=c2, s=dot_size, edgecolor='black', linewidth=1.2, zorder=4,
                           label=g2 if idx == 0 else "")

            if show_values:
                if pd.notna(m1):
                    ax.text(x - 0.15, m1, f"{m1:.2f}", ha='right', va='center', fontsize=8.5, color=c1, fontweight='bold')
                if pd.notna(m2):
                    ax.text(x + 0.15, m2, f"{m2:.2f}", ha='left', va='center', fontsize=8.5, color=c2, fontweight='bold')

        display_labels = stats_df['Display_Name'].tolist()
        ax.set_xticks(x_pos)
        ax.set_xticklabels(display_labels, rotation=0 if n_feats <= 8 else 30, ha='center', fontsize=11, fontweight='bold')

        valid_vals = [v for v in all_vals if pd.notna(v)]
        if valid_vals:
            min_val, max_val = min(valid_vals), max(valid_vals)
            val_span = max_val - min_val if max_val != min_val else 1.0
            ax.set_ylim(min_val - 0.08 * val_span, max_val + (0.20 if show_pvals else 0.08) * val_span)
        else:
            val_span = 1.0

        if show_pvals:
            for idx, row in stats_df.iterrows():
                x = x_pos[idx]
                m1, m2 = row[f'{g1}_Mean'], row[f'{g2}_Mean']
                sig_text = row['sig_label']
                if sig_text:
                    top_val = max(m for m in [m1, m2] if pd.notna(m)) if pd.notna(m1) or pd.notna(m2) else 0
                    ax.text(x, top_val + 0.04 * val_span, sig_text,
                            ha='center', va='bottom', fontsize=11, fontweight='bold',
                            color='#222222', zorder=5)

        ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.5, zorder=0)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.6, zorder=1)
        ax.set_ylabel(ylabel or "Mean Quantification / Relative Intensity", fontsize=11, fontweight='bold', labelpad=8)
        ax.tick_params(axis='both', which='major', labelsize=10.5)

    if title:
        ax.set_title(title, fontsize=13, fontweight='bold', pad=12)

    ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=10.5, loc='upper right')
    sns.despine(ax=ax)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, transparent=True, dpi=300, bbox_inches='tight')
        base, _ = os.path.splitext(save_path)
        plt.savefig(f"{base}.svg", transparent=True, bbox_inches='tight')
        print(f"Saved dumbbell plot to: {save_path}")

    return fig, ax, stats_df

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
# 7b. SPATIAL GLOMERULI CLUSTERING & IDENTIFICATION (CLUSTER 1 / PODOCYTES)
# ==============================================================================

def identify_glomeruli_clusters(df, cluster_name='Cluster1', cluster_col='mask_name',
                                eps=80, min_samples=20, x_col='centroid_x', y_col='centroid_y',
                                tissue_col='tissue_id', glom_col='glom'):
    """
    Identifies individual glomeruli instances by spatial density clustering (DBSCAN)
    on centroid coordinates per tissue section for a specified cluster (e.g. Cluster1).
    
    Assigns unique integer glomerulus IDs (glom >= 0) per tissue section.
    Returns the dataframe with the new glom_col attached.
    """
    if cluster_col not in df.columns or x_col not in df.columns or y_col not in df.columns:
        print(f"Warning: Required columns missing for glomeruli clustering: {cluster_col}, {x_col}, {y_col}")
        return df

    df_out = df.copy()
    tissue_ids = df_out[tissue_col].unique() if tissue_col in df_out.columns else [None]
    
    glom_series = pd.Series(np.nan, index=df_out.index)
    total_gloms = 0

    for tid in tissue_ids:
        if tid is not None:
            mask = (df_out[tissue_col] == tid) & (df_out[cluster_col] == cluster_name)
        else:
            mask = (df_out[cluster_col] == cluster_name)
        
        sub = df_out[mask].dropna(subset=[x_col, y_col])
        if len(sub) < min_samples:
            continue
        
        coords = sub[[x_col, y_col]].values
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        labels = dbscan.fit_predict(coords)
        
        # Shift non-negative cluster labels so each tissue has globally unique glom IDs
        valid_mask = labels >= 0
        assigned_labels = labels.astype(float)
        assigned_labels[valid_mask] += total_gloms
        assigned_labels[~valid_mask] = np.nan
        
        glom_series.loc[sub.index] = assigned_labels
        if np.any(valid_mask):
            total_gloms += int(np.max(labels[valid_mask])) + 1

    df_out[glom_col] = glom_series
    print(f" Identified {total_gloms} individual glomeruli across {len(tissue_ids)} tissue sections (Cluster '{cluster_name}').")
    return df_out


def plot_glomeruli_spatial_clusters(df, tissue_id=None, x_col='centroid_x', y_col='centroid_y',
                                    glom_col='glom', tissue_col='tissue_id', palette='turbo',
                                    s=80, alpha=0.9, figsize=(10, 8), save_path=None):
    """
    Plots spatial centroid scatter map of clustered glomeruli with numbered centroid text callouts.
    """
    sub = df[df[glom_col].notna() & (df[glom_col] >= 0)].copy()
    if tissue_id is not None and tissue_col in sub.columns:
        sub = sub[sub[tissue_col] == tissue_id]

    if sub.empty:
        print("No valid clustered glomeruli found to plot.")
        return None, None

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    sns.scatterplot(
        data=sub, x=x_col, y=y_col, hue=glom_col, palette=palette,
        s=s, alpha=alpha, edgecolor='black', linewidth=0.5, legend=False, ax=ax
    )

    # Compute geometric cluster centers to place bold numbered text labels
    centers = sub.groupby(glom_col)[[x_col, y_col]].mean().reset_index()
    for _, row in centers.iterrows():
        t = ax.text(row[x_col], row[y_col], str(int(row[glom_col])),
                    fontsize=11, fontweight='bold', color='black',
                    ha='center', va='center', zorder=10)
        t.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])

    title = f"Individual Glomeruli Spatial Map ({tissue_id})" if tissue_id else "Individual Glomeruli Spatial Map"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Centroid X (px / um)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Centroid Y (px / um)", fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, transparent=True, dpi=300)
        print(f"Saved glomeruli spatial plot to: {save_path}")
    return fig, ax


# ==============================================================================
# 8. PLOTTING TOOL E: UMAP & DIMENSIONALITY REDUCTION

# ==============================================================================

def save_standalone_legend(labels, palette, title=None, save_path=None, ncol=1,
                           fontsize=11, title_fontsize=12, markersize=10, marker='o'):
    """
    Renders and saves a clean, standalone, highly legible legend image.
    
    Args:
        labels: List of label strings in the desired order.
        palette: Dictionary mapping label -> hex/rgb color, or seaborn palette name / list.
        title: Optional title header for the legend box.
        save_path: Destination file path (.png, .svg).
        ncol: Number of columns in the legend layout.
        fontsize: Font size for label text.
        title_fontsize: Font size for title text.
        markersize: Size of legend marker points.
    """
    if isinstance(palette, dict):
        palette_dict = palette
    elif isinstance(palette, (list, tuple)):
        palette_dict = {cat: palette[i % len(palette)] for i, cat in enumerate(labels)}
    elif isinstance(palette, str):
        colors = sns.color_palette(palette, len(labels))
        palette_dict = {cat: c for cat, c in zip(labels, colors)}
    else:
        palette_dict = {cat: '#808080' for cat in labels}

    handles = [
        Line2D([0], [0], marker=marker, color='w', label=str(lbl),
               markerfacecolor=palette_dict.get(lbl, '#808080'),
               markeredgecolor='black', markeredgewidth=0.8,
               markersize=markersize)
        for lbl in labels
    ]

    nrows = int(np.ceil(len(labels) / ncol))
    fig_w = max(3.2, ncol * 2.8)
    fig_h = max(1.2, nrows * 0.42 + (0.6 if title else 0.2))

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    leg = ax.legend(
        handles=handles,
        title=title,
        loc='center',
        frameon=True,
        facecolor='white',
        edgecolor='#bbbbbb',
        fontsize=fontsize,
        title_fontsize=title_fontsize,
        ncol=ncol,
        columnspacing=1.4,
        handletextpad=0.6,
        borderpad=0.8
    )
    if title:
        leg.get_title().set_fontweight('bold')
    for text in leg.get_texts():
        text.set_fontweight('bold')
    
    fig.patch.set_alpha(0.0)
    plt.tight_layout(pad=0.2)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=300, transparent=True, bbox_inches='tight')
        base, _ = os.path.splitext(save_path)
        fig.savefig(f"{base}.svg", transparent=True, bbox_inches='tight')
        print(f"Saved standalone legend to: {save_path}")
    return fig


def plot_umap_embedding(df, umap_1_col='umap_1', umap_2_col='umap_2', hue_col=None,
                        palette=None, is_continuous=False, cmap='viridis',
                        title="UMAP Projection", s=1.5, alpha=0.8, figsize=(8, 8),
                        hue_order=None, label_map=None, show_legend=False,
                        save_separate_legend=True, legend_ncol=1,
                        legend_save_path=None, save_path=None):
    """
    Publication-quality UMAP visualization with consistent square dimensions,
    optimized point size for high-density instances, and standalone legible legends.
    """
    plot_df = df.copy()
    if hue_col and label_map and hue_col in plot_df.columns:
        plot_df[hue_col] = plot_df[hue_col].map(label_map).fillna(plot_df[hue_col])
        
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    
    if is_continuous and hue_col:
        sc = ax.scatter(plot_df[umap_1_col], plot_df[umap_2_col], c=plot_df[hue_col], cmap=cmap,
                        s=s, alpha=alpha, edgecolor='none', rasterized=True)
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(hue_col, fontsize=12, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)
    else:
        if hue_order is None and hue_col and hue_col in plot_df.columns:
            hue_order = list(plot_df[hue_col].dropna().unique())

        if isinstance(palette, dict):
            pal_dict = palette
        elif isinstance(palette, str) and hue_order:
            pal_dict = {cat: color for cat, color in zip(hue_order, sns.color_palette(palette, len(hue_order)))}
        elif isinstance(palette, (list, tuple)) and hue_order:
            pal_dict = {cat: palette[i % len(palette)] for i, cat in enumerate(hue_order)}
        else:
            pal_dict = palette

        sns.scatterplot(
            data=plot_df, x=umap_1_col, y=umap_2_col, hue=hue_col,
            hue_order=hue_order, palette=pal_dict,
            s=s, alpha=alpha, edgecolor='none', legend=show_legend,
            rasterized=True, ax=ax
        )
        if show_legend and hue_col:
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, fontsize=10)

    ax.set_xlabel("UMAP 1", fontsize=13, fontweight='bold', labelpad=8)
    ax.set_ylabel("UMAP 2", fontsize=13, fontweight='bold', labelpad=8)
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, length=4)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight('bold')
        
    ax.set_title(title, fontsize=15, fontweight='bold', pad=14)
    sns.despine(ax=ax)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, transparent=True, dpi=300, bbox_inches='tight')
        base, _ = os.path.splitext(save_path)
        plt.savefig(f"{base}.svg", transparent=True, bbox_inches='tight')
        print(f"Saved UMAP plot to: {save_path}")

    # Generate standalone separate legend if requested
    if not is_continuous and hue_col and save_separate_legend and hue_order:
        if legend_save_path is None and save_path:
            base, ext = os.path.splitext(save_path)
            legend_save_path = f"{base}_legend{ext}"
        if legend_save_path:
            save_standalone_legend(
                labels=hue_order,
                palette=pal_dict if pal_dict else (palette or 'tab20'),
                title=hue_col,
                save_path=legend_save_path,
                ncol=legend_ncol,
                fontsize=11,
                title_fontsize=12,
                markersize=10
            )
            
    return fig, ax


def plot_umap_with_region_density_overlay(
    df,
    umap_1_col='umap_1',
    umap_2_col='umap_2',
    cluster_col='cluster_display',
    cluster_palette=None,
    cluster_order=None,
    region_col='group',
    region_palette=None,
    region_order=None,
    density_levels=[0.35, 0.75],
    contour_alpha=0.9,
    contour_lw=1.8,
    contour_fill=True,
    fill_alpha=0.07,
    annotate_region_labels=False,
    show_region_legend=True,
    region_legend_title="Regions",
    region_legend_loc="upper right",
    label_fontsize=11,
    s=1.5,
    alpha=0.65,
    figsize=(8, 8),
    title="UMAP: Functional Units & Regional Density Boundaries",
    save_path=None
):
    """
    Renders high-density cluster instance scatter points overlaid with 2D Gaussian KDE
    density boundary contours and a clean on-plot legend for anatomical regions.
    """
    plot_df = df.dropna(subset=[umap_1_col, umap_2_col]).copy()
    if cluster_col in plot_df.columns:
        plot_df = plot_df.dropna(subset=[cluster_col])
        
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    
    # 1. Base Scatter Plot of Clusters
    if cluster_order is None and cluster_col and cluster_col in plot_df.columns:
        cluster_order = list(plot_df[cluster_col].dropna().unique())
        
    if isinstance(cluster_palette, dict):
        c_pal_dict = cluster_palette
    elif isinstance(cluster_palette, str) and cluster_order:
        c_pal_dict = {cat: color for cat, color in zip(cluster_order, sns.color_palette(cluster_palette, len(cluster_order)))}
    else:
        c_pal_dict = cluster_palette

    sns.scatterplot(
        data=plot_df, x=umap_1_col, y=umap_2_col, hue=cluster_col,
        hue_order=cluster_order, palette=c_pal_dict,
        s=s, alpha=alpha, edgecolor='none', legend=False,
        rasterized=True, ax=ax, zorder=2
    )

    # 2. 2D Gaussian KDE Density Contours per Region
    if region_col in plot_df.columns:
        if region_order is None:
            region_order = list(plot_df[region_col].dropna().unique())
            
        if isinstance(region_palette, dict):
            r_pal_dict = region_palette
        elif isinstance(region_palette, str) and region_order:
            r_pal_dict = {g: c for g, c in zip(region_order, sns.color_palette(region_palette, len(region_order)))}
        else:
            r_pal_dict = {g: '#000000' for g in region_order}

        # Shared coordinate grid for density estimation
        x_min, x_max = plot_df[umap_1_col].min(), plot_df[umap_1_col].max()
        y_min, y_max = plot_df[umap_2_col].min(), plot_df[umap_2_col].max()
        x_pad = (x_max - x_min) * 0.05
        y_pad = (y_max - y_min) * 0.05
        
        gx = np.linspace(x_min - x_pad, x_max + x_pad, 150)
        gy = np.linspace(y_min - y_pad, y_max + y_pad, 150)
        grid_x, grid_y = np.meshgrid(gx, gy)
        grid_coords = np.vstack([grid_x.ravel(), grid_y.ravel()])

        for reg in region_order:
            reg_sub = plot_df[plot_df[region_col] == reg]
            if len(reg_sub) < 30:
                continue

            rx = reg_sub[umap_1_col].values
            ry = reg_sub[umap_2_col].values
            
            try:
                kde = gaussian_kde(np.vstack([rx, ry]), bw_method='scott')
                z = kde(grid_coords).reshape(grid_x.shape)
                z_norm = (z - z.min()) / (z.max() - z.min() + 1e-12)
                
                reg_color = r_pal_dict.get(reg, '#333333')

                # Translucent fill for greatest density zone
                if contour_fill:
                    ax.contourf(
                        grid_x, grid_y, z_norm,
                        levels=[density_levels[0], 1.0],
                        colors=[reg_color],
                        alpha=fill_alpha,
                        zorder=3
                    )

                # Boundary contour lines
                ax.contour(
                    grid_x, grid_y, z_norm,
                    levels=density_levels,
                    colors=[reg_color],
                    linewidths=contour_lw,
                    alpha=contour_alpha,
                    linestyles=['dashed', 'solid'] if len(density_levels) == 2 else ['solid'],
                    zorder=4
                )

                # Optional Text Annotation at Density Peak (disabled by default)
                if annotate_region_labels:
                    peak_idx = np.unravel_index(np.argmax(z), z.shape)
                    peak_x, peak_y = grid_x[peak_idx], grid_y[peak_idx]

                    txt = ax.text(
                        peak_x, peak_y, str(reg),
                        fontsize=label_fontsize, fontweight='bold',
                        color=reg_color, ha='center', va='center',
                        zorder=6,
                        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.85, edgecolor=reg_color, lw=1.2)
                    )
                    txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
            except Exception as e:
                print(f"Warning: Could not compute KDE for region '{reg}': {e}")

        # 3. Add Region Legend on UMAP
        if show_region_legend and region_order:
            leg_handles = [
                Line2D([0], [0], color=r_pal_dict.get(reg, '#333333'), lw=2.2,
                       marker='o', markerfacecolor=r_pal_dict.get(reg, '#333333'),
                       markeredgecolor='black', markeredgewidth=0.8,
                       markersize=7.5, label=str(reg))
                for reg in region_order
            ]
            reg_leg = ax.legend(
                handles=leg_handles,
                title=region_legend_title,
                loc=region_legend_loc,
                frameon=True,
                facecolor='white',
                edgecolor='#cccccc',
                fontsize=10,
                title_fontsize=11,
                borderpad=0.6,
                handletextpad=0.5
            )
            if region_legend_title:
                reg_leg.get_title().set_fontweight('bold')
            for text in reg_leg.get_texts():
                text.set_fontweight('bold')

    ax.set_xlabel("UMAP 1", fontsize=13, fontweight='bold', labelpad=8)
    ax.set_ylabel("UMAP 2", fontsize=13, fontweight='bold', labelpad=8)
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, length=4)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight('bold')
        
    ax.set_title(title, fontsize=15, fontweight='bold', pad=14)
    sns.despine(ax=ax)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, transparent=True, dpi=300, bbox_inches='tight')
        base, _ = os.path.splitext(save_path)
        plt.savefig(f"{base}.svg", transparent=True, bbox_inches='tight')
        print(f"Saved region density overlay UMAP plot to: {save_path}")

    return fig, ax
