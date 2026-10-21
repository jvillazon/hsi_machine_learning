"""
========================================================================================
Core Non-Parametric Statistics & Effect Size Module
========================================================================================
Designed for high-throughput, non-Gaussian, multi-replicate biological and imaging data.

Key Components:
1. Fast O(N log N) Cliff's Delta (δ) with analytical Romano et al. (2006) 95% CIs.
2. Kruskal-Wallis H-test with η²_H effect size for omnibus group median variation.
3. Dunn's Post-Hoc Test with Benjamini-Hochberg (BH) FDR correction.
4. Stratified Wilcoxon (Van Elteren test) across biological replicates (Sample/Tissue IDs).
5. Replicate-level pseudo-bulk concordance validation (preventing pseudoreplication).
6. Non-parametric descriptive statistics (Median, IQR, MAD, Replicate Counts).
========================================================================================
"""

import os
import re
from typing import List, Dict, Tuple, Optional, Union, Any
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def p_to_stars(p: float) -> str:
    """Converts a p-value or FDR q-value float into publication-standard asterisk notation."""
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


def interpret_cliffs_delta(delta: float) -> str:
    """
    Standard Romano et al. (2006) / Hess & Kromrey (2004) qualitative interpretation:
      - |δ| < 0.147: Negligible
      - 0.147 <= |δ| < 0.330: Small
      - 0.330 <= |δ| < 0.474: Medium
      - |δ| >= 0.474: Large
    """
    if pd.isna(delta) or delta is None:
        return "N/A"
    abs_d = abs(delta)
    if abs_d < 0.147:
        return "Negligible"
    elif abs_d < 0.330:
        return "Small"
    elif abs_d < 0.474:
        return "Medium"
    else:
        return "Large"


def compute_cliffs_delta(
    group_a: Union[np.ndarray, List[float], pd.Series], 
    group_b: Union[np.ndarray, List[float], pd.Series]
) -> Tuple[float, str, Tuple[float, float]]:
    """
    Computes Cliff's Delta (δ) and Romano et al. (2006) 95% asymptotic confidence interval.
    
    Optimized for massive datasets: Uses Mann-Whitney U relationship and binary search (np.searchsorted)
    to achieve O(N log N) time and O(N) memory complexity, completely avoiding O(N^2) memory bottlenecks.

    Returns:
    --------
    (delta, magnitude, (ci_lower, ci_upper))
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    n1, n2 = len(a), len(b)
    
    if n1 == 0 or n2 == 0:
        return np.nan, "N/A", (np.nan, np.nan)
    
    # 1. High-performance delta via Mann-Whitney U
    u_stat, _ = stats.mannwhitneyu(a, b, alternative='two-sided')
    delta = (2.0 * u_stat) / (n1 * n2) - 1.0
    # Numerical safeguard to strictly bound in [-1.0, 1.0]
    delta = float(np.clip(delta, -1.0, 1.0))
    magnitude = interpret_cliffs_delta(delta)
    
    # 2. Fast O(N log N) Romano Asymptotic Variance for 95% CI
    if n1 < 2 or n2 < 2:
        return delta, magnitude, (np.nan, np.nan)
        
    b_sorted = np.sort(b)
    a_sorted = np.sort(a)
    
    # For each a_i: sum(a_i > b) - sum(a_i < b)
    # np.searchsorted(b_sorted, a, side='left') count elements in b strictly < a
    # n2 - np.searchsorted(b_sorted, a, side='right') count elements in b strictly > a
    d_i_dot = (
        np.searchsorted(b_sorted, a, side='left') - 
        (n2 - np.searchsorted(b_sorted, a, side='right'))
    ) / float(n2)
    
    # For each b_j: sum(b_j < a) - sum(b_j > a) = sum(a > b_j) - sum(a < b_j)
    d_dot_j = (
        (n1 - np.searchsorted(a_sorted, b, side='right')) - 
        np.searchsorted(a_sorted, b, side='left')
    ) / float(n1)
    
    s_d1 = np.var(d_i_dot, ddof=1)
    s_d2 = np.var(d_dot_j, ddof=1)
    
    # Romano et al. (2006) asymptotic variance
    var_delta = (
        (n2 - 1.0) * s_d1 + 
        (n1 - 1.0) * s_d2 + 
        (1.0 - delta**2)
    ) / (float(n1) * float(n2))
    
    se_delta = np.sqrt(max(0.0, var_delta))
    ci_margin = 1.96 * se_delta
    ci_lower = float(np.clip(delta - ci_margin, -1.0, 1.0))
    ci_upper = float(np.clip(delta + ci_margin, -1.0, 1.0))
    
    return delta, magnitude, (ci_lower, ci_upper)


def compute_nonparametric_descriptives(
    df: pd.DataFrame, 
    value_col: str, 
    group_col: str,
    sample_col: Optional[str] = None,
    ignore_zeros: bool = False
) -> pd.DataFrame:
    """
    Computes non-parametric summary statistics per group:
    N_observations, N_replicates (if sample_col provided), Median, IQR, Q25, Q75, MAD, Min, Max.
    """
    cols = [group_col, value_col]
    if sample_col and sample_col in df.columns:
        cols.append(sample_col)
        
    sub = df[cols].dropna(subset=[group_col, value_col])
    sub = sub[np.isfinite(sub[value_col])]
    if ignore_zeros:
        sub = sub[sub[value_col] != 0]

    records = []
    for grp, g_df in sub.groupby(group_col, observed=True):
        vals = g_df[value_col].values
        n_obs = len(vals)
        if n_obs == 0:
            continue
        
        med = float(np.median(vals))
        q25 = float(np.percentile(vals, 25))
        q75 = float(np.percentile(vals, 75))
        iqr = q75 - q25
        mad = float(stats.median_abs_deviation(vals, scale='normal'))
        n_reps = g_df[sample_col].nunique() if (sample_col and sample_col in g_df.columns) else np.nan
        
        records.append({
            group_col: grp,
            'Metric': value_col,
            'N_Observations': n_obs,
            'N_Replicates': n_reps,
            'Median': med,
            'IQR': iqr,
            'Q25': q25,
            'Q75': q75,
            'MAD': mad,
            'Min': float(np.min(vals)),
            'Max': float(np.max(vals))
        })
        
    return pd.DataFrame(records)


def run_kruskal_wallis(
    data: pd.DataFrame, 
    value_col: str, 
    group_col: str, 
    ignore_zeros: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Performs Kruskal-Wallis H-test across groups with Kruskal-Wallis η²_H effect size.
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
        return None
        
    group_data = [sub.loc[sub[group_col] == g, value_col].values for g in valid_groups]
    try:
        h_stat, p_val = stats.kruskal(*group_data)
        N_total = len(sub)
        # Kruskal-Wallis effect size: eta-squared H = (H - k + 1) / (N - k)
        eta_sq_h = (h_stat - k + 1.0) / max(1.0, (N_total - k))
        eta_sq_h = float(np.clip(eta_sq_h, 0.0, 1.0))
        
        return {
            "H_stat": float(h_stat),
            "df": int(k - 1),
            "p_unc": float(p_val),
            "k_groups": int(k),
            "N_total": int(N_total),
            "eta_sq_h": eta_sq_h
        }
    except Exception as e:
        print(f"  [Warning] Kruskal-Wallis H-test failed for '{value_col}' on '{group_col}': {e}")
        return None


def run_dunns_posthoc_fdr(
    data: pd.DataFrame, 
    value_col: str, 
    group_col: str, 
    alpha: float = 0.05,
    min_cliffs_delta: float = 0.147,
    ignore_zeros: bool = False
) -> Optional[pd.DataFrame]:
    """
    Performs Dunn's post-hoc rank-sum test with Benjamini-Hochberg (BH) FDR correction
    and Cliff's Delta effect size.
    
    Features:
    - Uses exact pooled Kruskal-Wallis rank sums with tie adjustments (Dunn, 1964).
    - Benjamini-Hochberg FDR control (less conservative, higher statistical power than Bonferroni).
    - High-performance Cliff's Delta (δ) and Romano 95% CIs.
    - Dual significance check: is_significant_fdr AND is_biologically_significant (|δ| >= min_cliffs_delta).
    """
    sub = data[[value_col, group_col]].dropna()
    sub = sub[np.isfinite(sub[value_col])]
    if ignore_zeros:
        sub = sub[sub[value_col] != 0]
        
    counts = sub[group_col].value_counts()
    valid_groups = list(counts[counts >= 2].index)
    sub = sub[sub[group_col].isin(valid_groups)].copy()
    
    k = len(valid_groups)
    if k < 2:
        return None

    N = len(sub)
    # Assign ranks to all pooled observations
    sub['rank'] = stats.rankdata(sub[value_col])
    
    # Tie correction term
    _, tie_counts = np.unique(sub['rank'], return_counts=True)
    ties = np.sum(tie_counts**3 - tie_counts)
    denom_var = (N * (N + 1.0)) / 12.0 - (ties / (12.0 * max(1.0, N - 1.0)))
    if denom_var <= 0:
        denom_var = 1e-12

    rank_means = sub.groupby(group_col, observed=True)['rank'].mean()
    group_sizes = sub.groupby(group_col, observed=True)['rank'].count()
    
    pairs = list(combinations(valid_groups, 2))
    records = []
    
    for g1, g2 in pairs:
        r1, r2 = rank_means[g1], rank_means[g2]
        n1, n2 = group_sizes[g1], group_sizes[g2]
        
        # Standard error under H0 pooled ranks
        se = np.sqrt(denom_var * (1.0 / n1 + 1.0 / n2))
        z_stat = (r1 - r2) / se if se > 0 else 0.0
        p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))
        
        v1 = sub.loc[sub[group_col] == g1, value_col].values
        v2 = sub.loc[sub[group_col] == g2, value_col].values
        
        med1, med2 = float(np.median(v1)), float(np.median(v2))
        iqr1 = float(np.percentile(v1, 75) - np.percentile(v1, 25))
        iqr2 = float(np.percentile(v2, 75) - np.percentile(v2, 25))
        med_diff = med1 - med2
        pct_med_diff = ((med1 - med2) / med2 * 100.0) if med2 != 0 else np.nan
        
        delta, magnitude, ci = compute_cliffs_delta(v1, v2)
        
        records.append({
            'A': g1,
            'B': g2,
            'N(A)': int(n1),
            'N(B)': int(n2),
            'median(A)': med1,
            'median(B)': med2,
            'iqr(A)': iqr1,
            'iqr(B)': iqr2,
            'diff': med_diff,
            'pct_diff': pct_med_diff,
            'mean_rank(A)': float(r1),
            'mean_rank(B)': float(r2),
            'se': float(se),
            'Z': float(z_stat),
            'pval_raw': float(p_val),
            'cliffs_delta': delta,
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'delta_95_ci': f"[{ci[0]:.3f}, {ci[1]:.3f}]",
            'effect_size_magnitude': magnitude
        })
        
    if not records:
        return None
        
    res_df = pd.DataFrame(records)
    
    # Benjamini-Hochberg (BH) False Discovery Rate (FDR)
    _, q_vals, _, _ = multipletests(res_df['pval_raw'], alpha=alpha, method='fdr_bh')
    res_df['pval'] = q_vals   # Standardized to 'pval' for compatibility with visualization routines
    res_df['q_fdr_bh'] = q_vals
    res_df['significance'] = res_df['pval'].apply(p_to_stars)
    res_df['is_significant'] = res_df['pval'] < alpha
    res_df['is_biologically_significant'] = (res_df['pval'] < alpha) & (res_df['cliffs_delta'].abs() >= min_cliffs_delta)
    
    return res_df


def run_stratified_sample_wilcoxon(
    df: pd.DataFrame, 
    value_col: str, 
    group_col: str, 
    sample_col: str,
    alpha: float = 0.05,
    ignore_zeros: bool = False
) -> pd.DataFrame:
    """
    Evaluates treatment difference controlling for inter-sample heterogeneity across
    biological replicates (Sample ID / Tissue ID).
    
    Computes:
    1. Van Elteren Test (Stratified Wilcoxon-Mann-Whitney rank-sum test):
       Combines within-replicate treatment effects using inverse-variance weights w_s = 1 / (n_1s + n_2s + 1).
    2. Replicate-level Pseudo-bulk Median test:
       Aggregates observations into sample medians (N = N_replicates) and performs paired or unpaired Wilcoxon.
    3. Concordance Flag:
       Validates whether observational micro-level effects hold true at the biological replicate macro-level.
    """
    cols = [group_col, value_col, sample_col]
    sub = df[cols].dropna()
    sub = sub[np.isfinite(sub[value_col])]
    if ignore_zeros:
        sub = sub[sub[value_col] != 0]
        
    groups = list(sub[group_col].dropna().unique())
    pairs = list(combinations(groups, 2))
    records = []
    
    for g1, g2 in pairs:
        pair_data = sub[sub[group_col].isin([g1, g2])].copy()
        
        # Van Elteren Stratified Rank Sum Test
        strat_stats = []
        for sample_id, s_df in pair_data.groupby(sample_col):
            v1 = s_df.loc[s_df[group_col] == g1, value_col].values
            v2 = s_df.loc[s_df[group_col] == g2, value_col].values
            n1, n2 = len(v1), len(v2)
            if n1 >= 1 and n2 >= 1:
                u, _ = stats.mannwhitneyu(v1, v2, alternative='two-sided')
                w = 1.0 / (n1 + n2 + 1.0)
                e_u = (n1 * n2) / 2.0
                var_u = (n1 * n2 * (n1 + n2 + 1.0)) / 12.0
                strat_stats.append({
                    'sample_id': sample_id,
                    'w': w,
                    'diff': u - e_u,
                    'var': var_u,
                    'n1': n1,
                    'n2': n2
                })
                
        if strat_stats:
            s_df = pd.DataFrame(strat_stats)
            denom = np.sqrt(np.sum((s_df['w']**2) * s_df['var']))
            if denom > 0:
                z_van_elteren = float(np.sum(s_df['w'] * s_df['diff']) / denom)
                p_van_elteren = float(2.0 * (1.0 - stats.norm.cdf(abs(z_van_elteren))))
            else:
                z_van_elteren, p_van_elteren = np.nan, np.nan
            n_strata_used = len(strat_stats)
        else:
            z_van_elteren, p_van_elteren, n_strata_used = np.nan, np.nan, 0
            
        # Replicate-level Pseudo-bulk Median Concordance Check
        sample_medians = pair_data.groupby([sample_col, group_col], observed=True)[value_col].median().unstack()
        p_replicate = np.nan
        replicate_test_type = "None"
        n_replicates_tested = 0
        
        if g1 in sample_medians.columns and g2 in sample_medians.columns:
            clean_pairs = sample_medians[[g1, g2]].dropna()
            n_replicates_tested = len(clean_pairs)
            if n_replicates_tested >= 3:
                try:
                    # Paired Wilcoxon signed-rank test across shared replicates
                    _, p_replicate = stats.wilcoxon(clean_pairs[g1], clean_pairs[g2])
                    p_replicate = float(p_replicate)
                    replicate_test_type = "Paired Wilcoxon"
                except Exception:
                    p_replicate = np.nan
        else:
            # Unpaired comparison across distinct biological replicate cohorts
            m1 = pair_data.loc[pair_data[group_col] == g1].groupby(sample_col, observed=True)[value_col].median().dropna().values
            m2 = pair_data.loc[pair_data[group_col] == g2].groupby(sample_col, observed=True)[value_col].median().dropna().values
            n_replicates_tested = len(m1) + len(m2)
            if len(m1) >= 2 and len(m2) >= 2:
                try:
                    _, p_replicate = stats.mannwhitneyu(m1, m2, alternative='two-sided')
                    p_replicate = float(p_replicate)
                    replicate_test_type = "Unpaired Mann-Whitney"
                except Exception:
                    p_replicate = np.nan
                    
        # Concordance evaluation
        is_strat_sig = (p_van_elteren < alpha) if not pd.isna(p_van_elteren) else False
        is_rep_sig = (p_replicate < alpha) if not pd.isna(p_replicate) else False
        concordance = "Robust Concordance" if (is_strat_sig and is_rep_sig) else (
            "Observation-Only (Potential Pseudoreplication)" if (is_strat_sig and not is_rep_sig) else (
                "Replicate-Only" if (not is_strat_sig and is_rep_sig) else "Non-Significant"
            )
        )
        
        records.append({
            'Metric': value_col,
            'Group_A': g1,
            'Group_B': g2,
            'Strata_Column': sample_col,
            'N_Strata_Analyzed': n_strata_used,
            'Van_Elteren_Z': z_van_elteren,
            'Van_Elteren_p': p_van_elteren,
            'Van_Elteren_Sig': p_to_stars(p_van_elteren),
            'N_Replicates_Tested': n_replicates_tested,
            'Replicate_Test_Type': replicate_test_type,
            'Replicate_Median_p': p_replicate,
            'Replicate_Median_Sig': p_to_stars(p_replicate),
            'Biological_Concordance': concordance
        })
        
    return pd.DataFrame(records)
