#!/usr/bin/env python3
"""
========================================================================================
Group Statistics & Effect Size Determination Suite
========================================================================================
Overhauled Non-Parametric Engine (Kruskal-Wallis, Dunn's + BH FDR, Cliff's Delta, Van Elteren)
Specifically architected for non-Gaussian distributions, large observation counts, and low biological replicates.

Features:
- Automated / Dynamic Feature Column Detection (with exclusions for centroids/IDs)
- Multi-Level Grouping (1 or 2 grouping columns, combined or faceted modes)
- Robust Statistical Testing:
    * Kruskal-Wallis H-Test (omnibus variation in group medians & η²_H effect size)
    * Dunn's Post-Hoc Test (pairwise rank-sum tests with exact tie adjustments)
    * Benjamini-Hochberg (BH) False Discovery Rate (FDR) multiplicity correction
    * Cliff's Delta (δ) Non-Parametric Effect Size with Romano et al. (2006) 95% CIs
    * Van Elteren Stratified Wilcoxon across biological replicates (Sample ID / tissue_id)
    * Biological Concordance Validation (Micro-observation vs. Macro-replicate agreement)
- Comprehensive Non-Parametric Summary Tables:
    * descriptive_statistics.csv (Median, IQR, Q25, Q75, MAD, Replicate Counts)
    * kruskal_wallis_results.csv (H-stat, df, p-val, η²_H)
    * dunns_posthoc_fdr_cliffs_delta.csv (Pairwise medians, q_BH, Cliff's δ, CI, Magnitude)
    * stratified_replicate_wilcoxon.csv (Van Elteren Z & p, Replicate Concordance)
- Legacy Parametric Option:
    * Supports mode="parametric" for Welch's ANOVA, Games-Howell, and Cohen's d / Hedges' g
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


# ==============================================================================
# 1. CONFIGURATION (Edit here for default runs)
# ==============================================================================
DEFAULT_DATA_PATH = r"C:\Users\jvill\OneDrive - UC San Diego\integrated_pipeline\WSI_data\WSI_data\CODEX_clusters-regions_masks_output_csv-test\output.csv"
DEFAULT_OUTPUT_DIR = r"./plots/group_statistics_output"

# Grouping settings
GROUP_COLS = ["group", "mask_name"]                 # 1 or 2 columns, e.g. ["group"] or ["group", "mask_name"]
COMBINE_TWO_GROUPS = False             # False = faceted per 2nd column; True = combined "group | mask_name"
ALPHA = 0.05                           # Significance threshold
MIN_CLIFFS_DELTA = 0.147               # Biological significance cutoff: Negligible (<0.147), Small (0.147), Medium (0.33), Large (0.474)
IGNORE_ZEROS = False                   # True = filter 0 values; False = keep 0s
STATS_MODE = "nonparametric"           # "nonparametric" (default) or "parametric"

# Sample / Biological Replicate Column (e.g. "Sample ID", "tissue_id", "sample_id", "Animal")
SAMPLE_ID_CANDIDATES = ["Sample ID", "tissue_id", "sample_id"]

# Dynamic vs. Custom Metric Selection:
VALUE_COLS: Optional[List[str]] = None

# Columns to exclude from automated numeric column detection
EXCLUDE_COLS = [
    "centroid_x", "centroid_y", "glom", "cluster_group_pct", 
    "Unnamed: 0", "ROI Inst", "Label", "Sample ID", "tissue_id",
    "umap_1", "umap_2", "UMAP_1", "UMAP_2"
]


# ==============================================================================
# 2. LEGACY PARAMETRIC SUPPORT (Welch's ANOVA / Games-Howell / Cohen's d)
# ==============================================================================

def compute_cohens_d(group_a: np.ndarray, group_b: np.ndarray, weighted: bool = True) -> Tuple[float, float]:
    """Computes Cohen's d and Hedges' g for parametric mode."""
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan
    
    var1 = np.var(a, ddof=1)
    var2 = np.var(b, ddof=1)
    
    if weighted:
        denom = n1 + n2 - 2
        s_pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / max(1, denom))
    else:
        s_pooled = np.sqrt((var1 + var2) / 2.0)
        
    if s_pooled == 0 or np.isnan(s_pooled):
        return 0.0, 0.0
    
    d = (np.mean(a) - np.mean(b)) / s_pooled
    j = 1.0 - (3.0 / max(1.0, (4.0 * (n1 + n2) - 9.0)))
    g = d * j
    return float(d), float(g)


def run_welch_games_howell(
    data: pd.DataFrame, 
    value_col: str, 
    group_col: str, 
    alpha: float = ALPHA,
    ignore_zeros: bool = False,
    weighted_pooled_sd: bool = True
) -> Tuple[Optional[Dict[str, Any]], Optional[pd.DataFrame]]:
    """Legacy parametric Welch's ANOVA and Games-Howell post-hoc test."""
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
    
    # 1. Welch's ANOVA
    anova_dict = None
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
    
    # 2. Games-Howell Post-Hoc Pairwise Comparisons
    posthoc_df = None
    try:
        posthoc_df = pg.pairwise_gameshowell(dv=value_col, between=group_col, data=sub)
        if posthoc_df is not None and not posthoc_df.empty:
            cohens_d_list = []
            hedges_g_list = []
            mean_a_list = []
            mean_b_list = []
            pct_diff_list = []
            effect_interp_list = []
            n_a_list = []
            n_b_list = []
            
            for _, r in posthoc_df.iterrows():
                val_a = sub.loc[sub[group_col] == r['A'], value_col].values
                val_b = sub.loc[sub[group_col] == r['B'], value_col].values
                d, g = compute_cohens_d(val_a, val_b, weighted=weighted_pooled_sd)
                
                mean_a = float(r.get('mean(A)', np.mean(val_a))) if 'mean(A)' in r else float(np.mean(val_a))
                mean_b = float(r.get('mean(B)', np.mean(val_b))) if 'mean(B)' in r else float(np.mean(val_b))
                pct_diff = ((mean_a - mean_b) / mean_b * 100.0) if mean_b != 0 else np.nan
                
                abs_d = abs(d) if not np.isnan(d) else 0
                interp = "Negligible" if abs_d < 0.2 else ("Small" if abs_d < 0.5 else ("Medium" if abs_d < 0.8 else ("Large" if abs_d < 1.2 else "Very Large")))
                
                mean_a_list.append(mean_a)
                mean_b_list.append(mean_b)
                cohens_d_list.append(d)
                hedges_g_list.append(g)
                pct_diff_list.append(pct_diff)
                effect_interp_list.append(interp)
                n_a_list.append(len(val_a))
                n_b_list.append(len(val_b))
                
            posthoc_df['mean(A)'] = mean_a_list
            posthoc_df['mean(B)'] = mean_b_list
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


# ==============================================================================
# 3. PIPELINE ORCHESTRATOR
# ==============================================================================

def get_quantitative_columns(df: pd.DataFrame, value_cols: Optional[List[str]] = None, 
                             exclude_cols: Optional[List[str]] = None) -> List[str]:
    """Dynamically resolves quantitative columns from dataframe if not explicitly specified."""
    if value_cols is not None and len(value_cols) > 0:
        return [col for col in value_cols if col in df.columns]
    
    exclusions = set(exclude_cols or EXCLUDE_COLS)
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    valid_cols = [c for c in numeric_cols if c not in exclusions]
    return valid_cols


def detect_sample_column(df: pd.DataFrame) -> Optional[str]:
    """Detects biological replicate column (e.g. 'Sample ID', 'tissue_id', 'Animal') in dataframe."""
    for col in SAMPLE_ID_CANDIDATES:
        if col in df.columns:
            return col
    return None


def run_group_statistics_pipeline(
    df: pd.DataFrame,
    group_cols: List[str] = GROUP_COLS,
    value_cols: Optional[List[str]] = VALUE_COLS,
    combine_two_groups: bool = COMBINE_TWO_GROUPS,
    alpha: float = ALPHA,
    min_cliffs_delta: float = MIN_CLIFFS_DELTA,
    mode: str = STATS_MODE,
    sample_col: Optional[str] = None,
    output_dir: Optional[str] = DEFAULT_OUTPUT_DIR,
    ignore_zeros: bool = IGNORE_ZEROS,
    exclude_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Main statistical pipeline driver. Iterates through all metrics across 1 or 2 groups.
    
    In Non-Parametric Mode (default):
    - Kruskal-Wallis H-test & η²_H
    - Dunn's Post-Hoc Test with Benjamini-Hochberg (BH) FDR correction
    - Cliff's Delta (δ) effect size & Romano 95% CIs
    - Van Elteren Stratified Wilcoxon test across biological replicates
    - Non-parametric descriptive statistics (Median, IQR, Q25, Q75, MAD, Replicate Counts)
    
    Returns:
    --------
    (omnibus_df, posthoc_df, descriptive_df, stratified_df) : Tuple of DataFrames
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Resolve value columns
    valid_metrics = get_quantitative_columns(df, value_cols=value_cols, exclude_cols=exclude_cols)
    if not valid_metrics:
        raise ValueError("No valid numeric quantitative metric columns found to analyze.")
    
    for g in group_cols:
        if g not in df.columns:
            raise KeyError(f"Grouping column '{g}' not found in dataframe.")
            
    # Resolve biological sample column
    resolved_sample_col = sample_col if sample_col and sample_col in df.columns else detect_sample_column(df)
    
    print("=" * 80)
    print("Starting Group Statistical Analysis Pipeline")
    print(f"Mode:               {mode.upper()}")
    print(f"Data instances:     {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Metrics ({len(valid_metrics)}):       {valid_metrics[:5]}{'...' if len(valid_metrics) > 5 else ''}")
    print(f"Grouping:           {group_cols} | Combine: {combine_two_groups} | Alpha: {alpha}")
    if mode == "nonparametric":
        print(f"FDR Correction:     Benjamini-Hochberg (BH)")
        print(f"Effect Size:        Cliff's Delta (delta) [Threshold: |delta| >= {min_cliffs_delta}]")
        print(f"Sample Stratification: {resolved_sample_col if resolved_sample_col else 'None detected'}")
    print("=" * 80)

    d = df.copy()
    omnibus_records = []
    posthoc_records = []
    desc_records = []
    stratified_records = []

    for i, item in enumerate(valid_metrics, 1):
        if len(group_cols) == 1:
            g_col = group_cols[0]
            
            if mode == "nonparametric":
                # 1. Non-parametric descriptives
                desc = compute_nonparametric_descriptives(d, item, g_col, sample_col=resolved_sample_col, ignore_zeros=ignore_zeros)
                desc['group_by'] = g_col
                desc['facet'] = None
                desc_records.append(desc)
                
                # 2. Kruskal-Wallis omnibus
                kw = run_kruskal_wallis(d, item, g_col, ignore_zeros=ignore_zeros)
                if kw is not None:
                    omnibus_records.append({**kw, "item": item, "group_by": g_col, "facet": None})
                    
                # 3. Dunn's post-hoc with BH-FDR & Cliff's Delta
                ph = run_dunns_posthoc_fdr(d, item, g_col, alpha=alpha, min_cliffs_delta=min_cliffs_delta, ignore_zeros=ignore_zeros)
                if ph is not None and not ph.empty:
                    ph["item"] = item
                    ph["group_by"] = g_col
                    ph["facet"] = None
                    posthoc_records.append(ph)
                    
                # 4. Stratified Wilcoxon across Sample ID
                if resolved_sample_col:
                    strat = run_stratified_sample_wilcoxon(d, item, g_col, resolved_sample_col, alpha=alpha, ignore_zeros=ignore_zeros)
                    if strat is not None and not strat.empty:
                        strat["item"] = item
                        strat["facet"] = None
                        stratified_records.append(strat)
            else:
                # Legacy parametric mode
                desc = compute_nonparametric_descriptives(d, item, g_col, sample_col=resolved_sample_col, ignore_zeros=ignore_zeros)
                desc['group_by'] = g_col
                desc['facet'] = None
                desc_records.append(desc)
                
                aov, ph = run_welch_games_howell(d, item, g_col, alpha=alpha, ignore_zeros=ignore_zeros)
                if aov is not None:
                    omnibus_records.append({**aov, "item": item, "group_by": g_col, "facet": None})
                if ph is not None and not ph.empty:
                    ph["item"] = item
                    ph["group_by"] = g_col
                    ph["facet"] = None
                    posthoc_records.append(ph)

        elif len(group_cols) == 2:
            g1, g2 = group_cols
            if combine_two_groups:
                combo_col = f"{g1}__{g2}"
                d[combo_col] = d[g1].astype(str) + " | " + d[g2].astype(str)
                
                if mode == "nonparametric":
                    desc = compute_nonparametric_descriptives(d, item, combo_col, sample_col=resolved_sample_col, ignore_zeros=ignore_zeros)
                    desc['group_by'] = f"{g1}+{g2} (combined)"
                    desc['facet'] = None
                    desc_records.append(desc)
                    
                    kw = run_kruskal_wallis(d, item, combo_col, ignore_zeros=ignore_zeros)
                    if kw is not None:
                        omnibus_records.append({**kw, "item": item, "group_by": f"{g1}+{g2} (combined)", "facet": None})
                        
                    ph = run_dunns_posthoc_fdr(d, item, combo_col, alpha=alpha, min_cliffs_delta=min_cliffs_delta, ignore_zeros=ignore_zeros)
                    if ph is not None and not ph.empty:
                        ph["item"] = item
                        ph["group_by"] = f"{g1}+{g2} (combined)"
                        ph["facet"] = None
                        posthoc_records.append(ph)
                        
                    if resolved_sample_col:
                        strat = run_stratified_sample_wilcoxon(d, item, combo_col, resolved_sample_col, alpha=alpha, ignore_zeros=ignore_zeros)
                        if strat is not None and not strat.empty:
                            strat["item"] = item
                            strat["facet"] = None
                            stratified_records.append(strat)
                else:
                    desc = compute_nonparametric_descriptives(d, item, combo_col, sample_col=resolved_sample_col, ignore_zeros=ignore_zeros)
                    desc['group_by'] = f"{g1}+{g2} (combined)"
                    desc['facet'] = None
                    desc_records.append(desc)
                    
                    aov, ph = run_welch_games_howell(d, item, combo_col, alpha=alpha, ignore_zeros=ignore_zeros)
                    if aov is not None:
                        omnibus_records.append({**aov, "item": item, "group_by": f"{g1}+{g2} (combined)", "facet": None})
                    if ph is not None and not ph.empty:
                        ph["item"] = item
                        ph["group_by"] = f"{g1}+{g2} (combined)"
                        ph["facet"] = None
                        posthoc_records.append(ph)
            else:
                facets = [fval for fval in d[g2].dropna().unique()]
                for fval in facets:
                    sub = d[d[g2] == fval]
                    if sub[g1].nunique() < 2:
                        continue
                    
                    if mode == "nonparametric":
                        desc = compute_nonparametric_descriptives(sub, item, g1, sample_col=resolved_sample_col, ignore_zeros=ignore_zeros)
                        desc['group_by'] = g1
                        desc['facet'] = f"{g2}={fval}"
                        desc_records.append(desc)
                        
                        kw = run_kruskal_wallis(sub, item, g1, ignore_zeros=ignore_zeros)
                        if kw is not None:
                            omnibus_records.append({**kw, "item": item, "group_by": g1, "facet": f"{g2}={fval}"})
                            
                        ph = run_dunns_posthoc_fdr(sub, item, g1, alpha=alpha, min_cliffs_delta=min_cliffs_delta, ignore_zeros=ignore_zeros)
                        if ph is not None and not ph.empty:
                            ph["item"] = item
                            ph["group_by"] = g1
                            ph["facet"] = f"{g2}={fval}"
                            posthoc_records.append(ph)
                            
                        if resolved_sample_col:
                            strat = run_stratified_sample_wilcoxon(sub, item, g1, resolved_sample_col, alpha=alpha, ignore_zeros=ignore_zeros)
                            if strat is not None and not strat.empty:
                                strat["item"] = item
                                strat["facet"] = f"{g2}={fval}"
                                stratified_records.append(strat)
                    else:
                        desc = compute_nonparametric_descriptives(sub, item, g1, sample_col=resolved_sample_col, ignore_zeros=ignore_zeros)
                        desc['group_by'] = g1
                        desc['facet'] = f"{g2}={fval}"
                        desc_records.append(desc)
                        
                        aov, ph = run_welch_games_howell(sub, item, g1, alpha=alpha, ignore_zeros=ignore_zeros)
                        if aov is not None:
                            omnibus_records.append({**aov, "item": item, "group_by": g1, "facet": f"{g2}={fval}"})
                        if ph is not None and not ph.empty:
                            ph["item"] = item
                            ph["group_by"] = g1
                            ph["facet"] = f"{g2}={fval}"
                            posthoc_records.append(ph)

        if i % 10 == 0 or i == len(valid_metrics):
            print(f"  Processed {i}/{len(valid_metrics)} features...")

    # Consolidate & Format Result DataFrames
    omnibus_df = pd.DataFrame(omnibus_records) if omnibus_records else pd.DataFrame()
    posthoc_df = pd.concat(posthoc_records, ignore_index=True) if posthoc_records else pd.DataFrame()
    desc_df = pd.concat(desc_records, ignore_index=True) if desc_records else pd.DataFrame()
    stratified_df = pd.concat(stratified_records, ignore_index=True) if stratified_records else pd.DataFrame()

    if not omnibus_df.empty and "p_unc" in omnibus_df.columns:
        omnibus_df["p_unc_formatted"] = omnibus_df["p_unc"].apply(format_p_value)
        omnibus_df["significance"] = omnibus_df["p_unc"].apply(p_to_stars)
        if "H_stat" in omnibus_df.columns:
            lead_cols = ["item", "group_by", "facet", "H_stat", "df", "p_unc", "p_unc_formatted", "significance", "eta_sq_h", "k_groups", "N_total"]
        else:
            lead_cols = ["item", "group_by", "facet", "F", "df1", "df2", "p_unc", "p_unc_formatted", "significance", "np2", "k_groups", "N_total"]
        omnibus_df = omnibus_df[[c for c in lead_cols if c in omnibus_df.columns]]

    if not posthoc_df.empty:
        if "pval" in posthoc_df.columns:
            posthoc_df["pval_formatted"] = posthoc_df["pval"].apply(format_p_value)
        if "q_fdr_bh" in posthoc_df.columns:
            posthoc_df["q_fdr_bh_formatted"] = posthoc_df["q_fdr_bh"].apply(format_p_value)
            
        lead_cols = [
            "item", "group_by", "facet", "A", "B", "N(A)", "N(B)", 
            "median(A)", "median(B)", "iqr(A)", "iqr(B)", "diff", "pct_diff",
            "Z", "pval_raw", "q_fdr_bh", "q_fdr_bh_formatted", "significance", 
            "is_significant", "cliffs_delta", "delta_95_ci", "effect_size_magnitude", 
            "is_biologically_significant", "mean(A)", "mean(B)", "cohen_d", "hedges_g"
        ]
        posthoc_df = posthoc_df[[c for c in lead_cols if c in posthoc_df.columns]]

    # Save CSV Reports
    if output_dir:
        if mode == "nonparametric":
            omnibus_path = os.path.join(output_dir, "kruskal_wallis_results.csv")
            posthoc_path = os.path.join(output_dir, "dunns_posthoc_fdr_cliffs_delta.csv")
        else:
            omnibus_path = os.path.join(output_dir, "welch_anova_results.csv")
            posthoc_path = os.path.join(output_dir, "games_howell_results.csv")
            
        desc_path = os.path.join(output_dir, "descriptive_statistics.csv")
        strat_path = os.path.join(output_dir, "stratified_replicate_wilcoxon.csv")
        
        omnibus_df.to_csv(omnibus_path, index=False)
        posthoc_df.to_csv(posthoc_path, index=False)
        desc_df.to_csv(desc_path, index=False)
        if not stratified_df.empty:
            stratified_df.to_csv(strat_path, index=False)
        
        print("\n" + "=" * 80)
        print("Statistical Analysis Complete!")
        print(f"  [1] Omnibus Test:            {omnibus_path}")
        print(f"  [2] Pairwise Post-Hoc:       {posthoc_path}")
        print(f"  [3] Descriptive Statistics:  {desc_path}")
        if not stratified_df.empty:
            print(f"  [4] Stratified Concordance:  {strat_path}")
        print("=" * 80)

    return omnibus_df, posthoc_df, desc_df, (stratified_df if not stratified_df.empty else None)


# ==============================================================================
# 4. CLI INTERFACE & MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Calculate Kruskal-Wallis, Dunn's post-hoc with BH FDR, Cliff's Delta, and Van Elteren stratified Wilcoxon.")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_PATH, help="Path to input dataset (.csv or .parquet)")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory for CSV reports")
    parser.add_argument("--groups", nargs="+", default=GROUP_COLS, help="1 or 2 grouping column names")
    parser.add_argument("--combine", action="store_true", default=COMBINE_TWO_GROUPS, help="Combine 2 grouping columns into single category")
    parser.add_argument("--alpha", type=float, default=ALPHA, help="Significance alpha threshold")
    parser.add_argument("--min-delta", type=float, default=MIN_CLIFFS_DELTA, help="Minimum Cliff's delta threshold for biological significance")
    parser.add_argument("--mode", type=str, default=STATS_MODE, choices=["nonparametric", "parametric"], help="Statistical mode (nonparametric or parametric)")
    parser.add_argument("--sample-col", type=str, default=None, help="Biological sample/replicate column for stratified Wilcoxon")
    parser.add_argument("--ignore-zeros", action="store_true", default=IGNORE_ZEROS, help="Ignore 0, NaN, and Inf values")
    args = parser.parse_args()

    data_path = args.data
    if not os.path.exists(data_path):
        print(f"Error: Input data file '{data_path}' not found.")
        sys.exit(1)

    print(f"Loading dataset from: {data_path}")
    ext = os.path.splitext(data_path)[1].lower()
    if ext == ".parquet":
        df = pd.read_parquet(data_path)
    elif ext in [".csv", ".txt"]:
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file format '{ext}'.")

    run_group_statistics_pipeline(
        df=df,
        group_cols=args.groups,
        value_cols=VALUE_COLS,
        combine_two_groups=args.combine,
        alpha=args.alpha,
        min_cliffs_delta=args.min_delta,
        mode=args.mode,
        sample_col=args.sample_col,
        output_dir=args.output,
        ignore_zeros=args.ignore_zeros
    )


if __name__ == "__main__":
    main()
