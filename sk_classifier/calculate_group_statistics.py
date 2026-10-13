#!/usr/bin/env python3
"""
========================================================================================
Group Statistics & Effect Size Determination Suite (Welch's ANOVA, Games-Howell, Cohen's d)
========================================================================================
Features:
- Automated / Dynamic Feature Column Detection (with exclusions for centroids/IDs)
- Multi-Level Grouping (1 or 2 grouping columns, combined or faceted modes)
- Robust Statistical Testing:
    * Welch's ANOVA (robust to unequal variances and group sample sizes)
    * Games-Howell Post-Hoc Test (pairwise Welch t-tests with adjusted p-values)
    * Effect Size Calculations:
        - Cohen's d (sample-size weighted or unweighted pooled SD)
        - Hedges' g (sample-size corrected unbiased Cohen's d)
        - Percentage Difference between group means
        - Qualitative Magnitude Interpretation (Negligible, Small, Medium, Large, Very Large)
- Summary Tables:
    * descriptive_statistics.csv
    * welch_anova_results.csv
    * games_howell_results.csv
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
import pingouin as pg

# Add parent directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ==============================================================================
# 1. CONFIGURATION (Edit here for default runs)
# ==============================================================================
DEFAULT_DATA_PATH = r"C:\Users\jvill\OneDrive - UC San Diego\integrated_pipeline\WSI_data\WSI_data\CODEX_clusters-regions_masks_output_csv-test\output.csv"
DEFAULT_OUTPUT_DIR = r"./plots/group_statistics_output"

# Grouping settings
GROUP_COLS = ["group", "mask_name"]                 # 1 or 2 columns, e.g. ["group"] or ["group", "mask_name"]
COMBINE_TWO_GROUPS = False             # False = faceted per 2nd column; True = combined "group | mask_name"
ALPHA = 0.05                           # Significance threshold
IGNORE_ZEROS = False                   # True = filter 0 values; False = keep 0s
WEIGHTED_POOLED_SD = True              # True = (n-1) weighted pooled SD; False = sqrt((var1+var2)/2)

# Dynamic vs. Custom Metric Selection:
# Set VALUE_COLS = None to automatically select all numeric columns in the dataframe
VALUE_COLS: Optional[List[str]] = None

# Columns to exclude from automated numeric column detection
EXCLUDE_COLS = [
    "centroid_x", "centroid_y", "glom", "cluster_group_pct", 
    "Unnamed: 0", "ROI Inst", "Label", "Sample ID", "tissue_id",
    "umap_1", "umap_2", "UMAP_1", "UMAP_2"
]


# ==============================================================================
# 2. STATISTICAL ENGINE & EFFECT SIZES
# ==============================================================================

def p_to_stars(p: float) -> str:
    """Converts a p-value float into standard asterisk notation."""
    if pd.isna(p) or p is None:
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
    """Formats numeric p-value into publication-ready string."""
    if pd.isna(p) or p is None:
        return "N/A"
    if p == 0 or p < 1e-15:
        return "< 1e-15"
    if p < 0.001:
        return f"{p:.4e}"
    return f"{p:.4f}"


def interpret_effect_size(d: float) -> str:
    """Provides Cohen's standard qualitative interpretation of effect size magnitude."""
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


def compute_cohens_d(group_a: np.ndarray, group_b: np.ndarray, weighted: bool = True) -> Tuple[float, float]:
    """
    Computes Cohen's d and Hedges' g (sample-size corrected Cohen's d).
    
    Parameters:
    -----------
    group_a, group_b : array-like
        Numeric arrays of values for group A and group B.
    weighted : bool, default True
        If True, uses the standard sample-size weighted pooled standard deviation:
            s_pooled = sqrt( ((n1 - 1)*var1 + (n2 - 1)*var2) / (n1 + n2 - 2) )
        If False, uses the unweighted average of group variances:
            s_pooled = sqrt( (var1 + var2) / 2 )
    
    Returns:
    --------
    (cohen_d, hedges_g) : Tuple[float, float]
    """
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
    
    # Hedges' g correction factor for small-sample bias
    j = 1.0 - (3.0 / max(1.0, (4.0 * (n1 + n2) - 9.0)))
    g = d * j
    return float(d), float(g)


def compute_descriptive_stats(
    df: pd.DataFrame, 
    value_col: str, 
    group_col: str,
    ignore_zeros: bool = False
) -> pd.DataFrame:
    """Computes comprehensive descriptive summary statistics per group."""
    sub = df[[group_col, value_col]].dropna()
    sub = sub[np.isfinite(sub[value_col])]
    if ignore_zeros:
        sub = sub[sub[value_col] != 0]

    grouped = sub.groupby(group_col, observed=True)[value_col]
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
    alpha: float = ALPHA,
    ignore_zeros: bool = False,
    weighted_pooled_sd: bool = True
) -> Tuple[Optional[Dict[str, Any]], Optional[pd.DataFrame]]:
    """
    Performs Welch's ANOVA and pairwise Games-Howell post-hoc testing with Cohen's d & Hedges' g.
    
    Returns:
    --------
    anova_dict : dict or None
        Dictionary containing omnibus ANOVA results (F, df1, df2, p_unc, np2).
    posthoc_df : pd.DataFrame or None
        DataFrame containing pairwise post-hoc tests with effect size metrics.
    """
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
                
                mean_a_list.append(mean_a)
                mean_b_list.append(mean_b)
                cohens_d_list.append(d)
                hedges_g_list.append(g)
                pct_diff_list.append(pct_diff)
                effect_interp_list.append(interpret_effect_size(d))
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


def run_group_statistics_pipeline(
    df: pd.DataFrame,
    group_cols: List[str] = GROUP_COLS,
    value_cols: Optional[List[str]] = VALUE_COLS,
    combine_two_groups: bool = COMBINE_TWO_GROUPS,
    alpha: float = ALPHA,
    output_dir: Optional[str] = DEFAULT_OUTPUT_DIR,
    ignore_zeros: bool = IGNORE_ZEROS,
    weighted_pooled_sd: bool = WEIGHTED_POOLED_SD,
    exclude_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Main statistical pipeline driver. Iterates through all metrics across 1 or 2 groups,
    calculates Welch's ANOVA, Games-Howell post-hoc, Cohen's d & Hedges' g effect sizes,
    and exports summary CSVs.
    
    Returns:
    --------
    (anova_df, posthoc_df, descriptive_df) : Tuple of DataFrames
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 1. Resolve Value Columns
    valid_metrics = get_quantitative_columns(df, value_cols=value_cols, exclude_cols=exclude_cols)
    if not valid_metrics:
        raise ValueError("No valid numeric quantitative metric columns found to analyze.")
    
    for g in group_cols:
        if g not in df.columns:
            raise KeyError(f"Grouping column '{g}' not found in dataframe.")
    
    print("=" * 80)
    print("Starting Group Statistical Analysis Pipeline")
    print(f"Data instances: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Metrics ({len(valid_metrics)}): {valid_metrics[:5]}{'...' if len(valid_metrics) > 5 else ''}")
    print(f"Grouping: {group_cols} | Combine: {combine_two_groups} | Alpha: {alpha}")
    print(f"Cohen's d Pooled SD Mode: {'Sample-size Weighted' if weighted_pooled_sd else 'Unweighted Average'}")
    print("=" * 80)

    d = df.copy()
    anova_records = []
    posthoc_records = []
    desc_records = []

    # 2. Iterate Metrics
    for i, item in enumerate(valid_metrics, 1):
        if len(group_cols) == 1:
            g_col = group_cols[0]
            
            # Descriptive stats
            desc = compute_descriptive_stats(d, item, g_col, ignore_zeros=ignore_zeros)
            desc['group_by'] = g_col
            desc['facet'] = None
            desc_records.append(desc)
            
            # Hypothesis testing
            anova, posthoc = run_welch_games_howell(
                d, item, g_col, alpha=alpha, 
                ignore_zeros=ignore_zeros, weighted_pooled_sd=weighted_pooled_sd
            )
            if anova is not None:
                anova_records.append({**anova, "item": item, "group_by": g_col, "facet": None})
            if posthoc is not None:
                posthoc["item"] = item
                posthoc["group_by"] = g_col
                posthoc["facet"] = None
                posthoc_records.append(posthoc)

        elif len(group_cols) == 2:
            g1, g2 = group_cols
            if combine_two_groups:
                combo_col = f"{g1}__{g2}"
                d[combo_col] = d[g1].astype(str) + " | " + d[g2].astype(str)
                
                desc = compute_descriptive_stats(d, item, combo_col, ignore_zeros=ignore_zeros)
                desc['group_by'] = f"{g1}+{g2} (combined)"
                desc['facet'] = None
                desc_records.append(desc)
                
                anova, posthoc = run_welch_games_howell(
                    d, item, combo_col, alpha=alpha, 
                    ignore_zeros=ignore_zeros, weighted_pooled_sd=weighted_pooled_sd
                )
                if anova is not None:
                    anova_records.append({**anova, "item": item, "group_by": f"{g1}+{g2} (combined)", "facet": None})
                if posthoc is not None:
                    posthoc["item"] = item
                    posthoc["group_by"] = f"{g1}+{g2} (combined)"
                    posthoc["facet"] = None
                    posthoc_records.append(posthoc)
            else:
                facets = [fval for fval in d[g2].dropna().unique()]
                for fval in facets:
                    sub = d[d[g2] == fval]
                    if sub[g1].nunique() < 2:
                        continue
                    
                    desc = compute_descriptive_stats(sub, item, g1, ignore_zeros=ignore_zeros)
                    desc['group_by'] = g1
                    desc['facet'] = f"{g2}={fval}"
                    desc_records.append(desc)
                    
                    anova, posthoc = run_welch_games_howell(
                        sub, item, g1, alpha=alpha, 
                        ignore_zeros=ignore_zeros, weighted_pooled_sd=weighted_pooled_sd
                    )
                    if anova is not None:
                        anova_records.append({**anova, "item": item, "group_by": g1, "facet": f"{g2}={fval}"})
                    if posthoc is not None:
                        posthoc["item"] = item
                        posthoc["group_by"] = g1
                        posthoc["facet"] = f"{g2}={fval}"
                        posthoc_records.append(posthoc)

        if i % 10 == 0 or i == len(valid_metrics):
            print(f"  Processed {i}/{len(valid_metrics)} features...")

    # 3. Consolidate & Format Result DataFrames
    anova_df = pd.DataFrame([{k: v for k, v in rec.items() if k != "labels"} for rec in anova_records])
    posthoc_df = pd.concat(posthoc_records, ignore_index=True) if posthoc_records else pd.DataFrame()
    desc_df = pd.concat(desc_records, ignore_index=True) if desc_records else pd.DataFrame()

    if not anova_df.empty and "p_unc" in anova_df.columns:
        anova_df["p_unc_formatted"] = anova_df["p_unc"].apply(format_p_value)
        # Reorder columns
        cols = ["item", "group_by", "facet", "F", "df1", "df2", "p_unc", "p_unc_formatted", "np2", "k_groups", "N_total"]
        anova_df = anova_df[[c for c in cols if c in anova_df.columns]]

    if not posthoc_df.empty and "pval" in posthoc_df.columns:
        posthoc_df["pval_formatted"] = posthoc_df["pval"].apply(format_p_value)
        cols = [
            "item", "group_by", "facet", "A", "B", "N(A)", "N(B)", 
            "mean(A)", "mean(B)", "diff", "pct_diff", "T", "df", 
            "pval", "pval_formatted", "significance", "is_significant", 
            "cohen_d", "hedges_g", "effect_size_magnitude"
        ]
        posthoc_df = posthoc_df[[c for c in cols if c in posthoc_df.columns]]

    # 4. Save CSV Reports
    if output_dir:
        anova_path = os.path.join(output_dir, "welch_anova_results.csv")
        posthoc_path = os.path.join(output_dir, "games_howell_results.csv")
        desc_path = os.path.join(output_dir, "descriptive_statistics.csv")
        
        anova_df.to_csv(anova_path, index=False)
        posthoc_df.to_csv(posthoc_path, index=False)
        desc_df.to_csv(desc_path, index=False)
        
        print("\n" + "=" * 80)
        print("Statistical Analysis Complete!")
        print(f"  [1] Welch's ANOVA:          {anova_path}")
        print(f"  [2] Games-Howell + Cohen d: {posthoc_path}")
        print(f"  [3] Descriptive Statistics: {desc_path}")
        print("=" * 80)

    return anova_df, posthoc_df, desc_df


# ==============================================================================
# 4. CLI INTERFACE & MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Calculate Welch's ANOVA, Games-Howell post-hoc, and Cohen's d effect sizes.")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_PATH, help="Path to input dataset (.csv or .parquet)")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory for CSV reports")
    parser.add_argument("--groups", nargs="+", default=GROUP_COLS, help="1 or 2 grouping column names")
    parser.add_argument("--combine", action="store_true", default=COMBINE_TWO_GROUPS, help="Combine 2 grouping columns into single category")
    parser.add_argument("--alpha", type=float, default=ALPHA, help="Significance alpha threshold")
    parser.add_argument("--ignore-zeros", action="store_true", default=IGNORE_ZEROS, help="Ignore 0, NaN, and Inf values")
    parser.add_argument("--unweighted-sd", action="store_true", help="Use unweighted average of variances for Cohen's d instead of (n-1) weighted")
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
        output_dir=args.output,
        ignore_zeros=args.ignore_zeros,
        weighted_pooled_sd=not args.unweighted_sd
    )


if __name__ == "__main__":
    main()
