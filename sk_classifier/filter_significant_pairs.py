#!/usr/bin/env python3
"""
Identify variables/groups (A) that are statistically significantly HIGHER or LOWER
than >= 80% (or configurable threshold) of their paired comparison groups (B) for
each item and/or (item, facet) pair in a Games-Howell results CSV.

Also lists the specific variables each qualifying group is NOT statistically significant with.
"""

import os
import argparse
import pandas as pd
import numpy as np


def load_games_howell_results(csv_path: str) -> pd.DataFrame:
    """Loads Games-Howell post-hoc results CSV into a pandas DataFrame."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    df["pval"] = pd.to_numeric(df["pval"], errors="coerce")
    if "diff" in df.columns:
        df["diff"] = pd.to_numeric(df["diff"], errors="coerce")
    elif "mean(A)" in df.columns and "mean(B)" in df.columns:
        df["diff"] = pd.to_numeric(df["mean(A)"], errors="coerce") - pd.to_numeric(df["mean(B)"], errors="coerce")
    return df


def analyze_directional_significance(
    df: pd.DataFrame,
    alpha: float = 0.05,
    majority_thresh: float = 0.80,
    item_filter: str = None,
    facet_filter: str = None
) -> pd.DataFrame:
    """
    For every unique item (and item-facet pair if faceted), analyzes each variable X
    against all paired variables Y to determine:
    1. If X is significantly HIGHER than >= majority_thresh of other variables (diff > 0, pval < alpha)
    2. If X is significantly LOWER than >= majority_thresh of other variables (diff < 0, pval < alpha)
    3. The exact list of variables X is NOT statistically significant with (pval >= alpha)
    """
    data = df.copy()
    if item_filter is not None:
        data = data[data["item"] == item_filter]
    if facet_filter is not None:
        data = data[data["facet"] == facet_filter]

    if data.empty:
        print("Warning: No records found matching the given filters.")
        return pd.DataFrame()

    has_facets = "facet" in data.columns and data["facet"].notna().any()

    # Expand every pairwise row into directed comparisons: X -> Y
    directed_records = []
    for _, row in data.iterrows():
        item = row["item"]
        facet = row.get("facet", "All") if has_facets else "All"
        A = str(row["A"])
        B = str(row["B"])
        pval = row["pval"]
        diff = row["diff"]
        
        is_sig = (pval < alpha) if pd.notna(pval) else False

        # Perspective of A
        directed_records.append({
            "item": item,
            "facet": facet,
            "variable": A,
            "counterpart": B,
            "diff": diff,
            "pval": pval,
            "is_sig": is_sig,
            "is_higher": is_sig and (diff > 0),
            "is_lower": is_sig and (diff < 0),
            "is_ns": not is_sig
        })

        # Perspective of B
        directed_records.append({
            "item": item,
            "facet": facet,
            "variable": B,
            "counterpart": A,
            "diff": -diff if pd.notna(diff) else np.nan,
            "pval": pval,
            "is_sig": is_sig,
            "is_higher": is_sig and (diff < 0),
            "is_lower": is_sig and (diff > 0),
            "is_ns": not is_sig
        })

    long_df = pd.DataFrame(directed_records)

    group_cols = ["item", "facet", "variable"] if has_facets else ["item", "variable"]
    results = []

    for keys, group in long_df.groupby(group_cols):
        if has_facets:
            item_val, facet_val, var_val = keys
        else:
            item_val, var_val = keys
            facet_val = "All"

        total_pairs = group["counterpart"].nunique()
        if total_pairs == 0:
            continue

        higher_rows = group[group["is_higher"]]
        lower_rows = group[group["is_lower"]]
        ns_rows = group[group["is_ns"]]

        n_higher = len(higher_rows)
        n_lower = len(lower_rows)
        n_ns = len(ns_rows)

        frac_higher = n_higher / total_pairs
        frac_lower = n_lower / total_pairs
        frac_ns = n_ns / total_pairs

        is_majority_higher = frac_higher >= majority_thresh
        is_majority_lower = frac_lower >= majority_thresh
        is_majority_sig = (n_higher + n_lower) / total_pairs >= majority_thresh

        status = "None"
        if is_majority_higher:
            status = "Significantly Higher"
        elif is_majority_lower:
            status = "Significantly Lower"
        elif is_majority_sig:
            status = "Significantly Mixed"

        sig_higher_list = higher_rows["counterpart"].tolist()
        sig_lower_list = lower_rows["counterpart"].tolist()
        ns_list = ns_rows["counterpart"].tolist()

        results.append({
            "item": item_val,
            "facet": facet_val,
            "variable": var_val,
            "status": status,
            "is_majority_higher": is_majority_higher,
            "is_majority_lower": is_majority_lower,
            "is_majority_any": (is_majority_higher or is_majority_lower),
            "total_pairs": total_pairs,
            "n_higher": n_higher,
            "frac_higher": frac_higher,
            "n_lower": n_lower,
            "frac_lower": frac_lower,
            "n_not_sig": n_ns,
            "frac_not_sig": frac_ns,
            "sig_higher_than": ", ".join(sig_higher_list) if sig_higher_list else "(None)",
            "sig_lower_than": ", ".join(sig_lower_list) if sig_lower_list else "(None)",
            "not_significant_with": ", ".join(ns_list) if ns_list else "(None)"
        })

    out_df = pd.DataFrame(results)
    return out_df


def main():
    default_csv = "/Users/jorgevillazon/Library/CloudStorage/OneDrive-UCSanDiego/integrated_pipeline/final_plots/FIB/unit_group/all_nobrackets/games_howell_results.csv"

    default_output_csv = os.path.dirname(default_csv) + "/directional_significance_" + os.path.basename(default_csv)

    parser = argparse.ArgumentParser(description="Find variables that are significantly higher or lower than 80% of paired variables in Games-Howell results.")
    parser.add_argument("--csv", type=str, default=default_csv, help="Path to games_howell_results.csv")
    parser.add_argument("--item", type=str, default=None, help="Filter for a specific item (e.g. 'Redox', 'CAR 18:1 (+H)')")
    parser.add_argument("--facet", type=str, default=None, help="Filter for a specific facet (e.g. 'group=Outer Cortex')")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance threshold alpha (default: 0.05)")
    parser.add_argument("--thresh", type=float, default=0.80, help="Majority threshold fraction (default: 0.80 for 80%%)")
    parser.add_argument("--output_dir", type=str, default=default_output_csv, help="Directory to save summary CSV files")
    parser.add_argument("--show_all", action="store_true", help="Show all variables regardless of whether they meet the 80%% threshold")

    args = parser.parse_args()

    print(f"Loading data from: {args.csv}")
    try:
        df = load_games_howell_results(args.csv)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    print(f"Total pairwise records: {len(df)}")
    print(f"Unique items in dataset: {df['item'].nunique() if 'item' in df.columns else 1}")

    full_results = analyze_directional_significance(
        df,
        alpha=args.alpha,
        majority_thresh=args.thresh,
        item_filter=args.item,
        facet_filter=args.facet
    )

    if full_results.empty:
        print("No results found.")
        return

    # Filter to 80% majority variables unless show_all is specified
    if args.show_all:
        display_df = full_results
    else:
        display_df = full_results[full_results["is_majority_any"]]

    thresh_pct = int(args.thresh * 100)
    print("\n" + "="*110)
    print(f"VARIABLES STATISTICALLY SIGNIFICANT ACROSS >= {thresh_pct}% OF PAIRS (HIGHER OR LOWER)")
    print("="*110)

    if display_df.empty:
        print(f"No variables met the >= {thresh_pct}% majority threshold.")
    else:
        for (item_val, facet_val), group_df in display_df.groupby(["item", "facet"]):
            facet_str = f" | Facet: {facet_val}" if facet_val != "All" else ""
            print(f"\n>>> Item: {item_val}{facet_str}")
            print("-" * 110)
            for _, r in group_df.iterrows():
                direction_label = f"[{r['status'].upper()}]"
                higher_pct = f"{r['frac_higher']*100:.1f}% ({r['n_higher']}/{r['total_pairs']})"
                lower_pct = f"{r['frac_lower']*100:.1f}% ({r['n_lower']}/{r['total_pairs']})"
                ns_pct = f"{r['frac_not_sig']*100:.1f}% ({r['n_not_sig']}/{r['total_pairs']})"
                
                print(f"  * Variable: {r['variable']:<25} {direction_label}")
                print(f"    - Sig Higher Than: {higher_pct:<18} -> {r['sig_higher_than']}")
                print(f"    - Sig Lower Than:  {lower_pct:<18} -> {r['sig_lower_than']}")
                print(f"    - Not Sig With:    {ns_pct:<18} -> {r['not_significant_with']}")
                print()

    # Save to output directory if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        all_out_path = os.path.join(args.output_dir, "all_variable_significance_summary.csv")
        full_results.to_csv(all_out_path, index=False)
        print(f"\n[Saved full summary to]: {all_out_path}")

        majority_out_path = os.path.join(args.output_dir, f"majority_{thresh_pct}pct_significant_variables.csv")
        full_results[full_results["is_majority_any"]].to_csv(majority_out_path, index=False)
        print(f"[Saved >= {thresh_pct}% majority variables to]: {majority_out_path}")


if __name__ == "__main__":
    main()
