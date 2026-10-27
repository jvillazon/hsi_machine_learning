"""
========================================================================================
Run Group Statistics & Publication-Quality Visualization
========================================================================================
Flexible runner script supporting:
1. JSON Configuration files (e.g. configs/xenium_stats_config.json, wsi_stats_config.json)
2. Direct programmatic Python execution with custom DataFrames and parameter dictionaries
========================================================================================
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

# Add parent directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from group_stats_pipeline import (
        run_group_stats_pipeline,
        load_table,
        load_json_config
    )
except ImportError:
    from sk_classifier.group_stats_pipeline import (
        run_group_stats_pipeline,
        load_table,
        load_json_config
    )


def run_from_config(config_path: str):
    """Executes the pipeline directly from a JSON configuration file."""
    print(f"Executing pipeline from config: {config_path}")
    cfg = load_json_config(config_path)
    
    input_path = cfg.get("input_path")
    if not input_path:
        raise ValueError(f"Config '{config_path}' must specify 'input_path'.")
    
    df = load_table(input_path)
    print(f"Loaded {len(df):,} rows x {len(df.columns)} columns from {input_path}")

    # Optional user-defined feature engineering hooks can be applied here
    if "LD Total Area (um2)" in df.columns and "LD Count" in df.columns and "LD Area / Count" not in df.columns:
        df["LD Area / Count"] = df["LD Total Area (um2)"] / df["LD Count"]
    if "Area (px)" in df.columns and "Pixel_Size_um" in df.columns and "LD Count" in df.columns and "LD Count Density" not in df.columns:
        df["LD Count Density"] = df["LD Count"] / (df["Area (px)"] * df["Pixel_Size_um"] ** 2)

    value_cols = cfg.get("value_cols")
    filter_significant_only = cfg.get("filter_significant_only", False)
    
    if filter_significant_only:
        print("\n[Significance Filter] Running headless statistical pre-screening across all metrics...")
        try:
            from calculate_group_statistics import run_group_statistics_pipeline
            from group_stats_pipeline import apply_lookup_mapping
        except ImportError:
            from integrated_pipeline.calculate_group_statistics import run_group_statistics_pipeline
            from integrated_pipeline.group_stats_pipeline import apply_lookup_mapping
        
        df_screen = df.copy()
        lookup_cfg = cfg.get("lookup_table")
        if lookup_cfg:
            df_screen = apply_lookup_mapping(
                df=df_screen,
                source_col=lookup_cfg.get("source_col") if isinstance(lookup_cfg, dict) else None,
                target_col=lookup_cfg.get("target_col", "Condition") if isinstance(lookup_cfg, dict) else "Condition",
                mapping=lookup_cfg
            )
        
        filter_vals = cfg.get("filter_values")
        if filter_vals:
            for f_col, f_vals in filter_vals.items():
                if f_col in df_screen.columns and f_vals:
                    df_screen = df_screen[df_screen[f_col].isin(f_vals)]
                    
        alpha_thresh = cfg.get("alpha", 0.05)
        effect_thresh = cfg.get("min_effect_size_filter", cfg.get("min_effect_size_bracket", 0.147))
        
        omni_df, ph_df, _, _ = run_group_statistics_pipeline(
            df=df_screen,
            group_cols=cfg.get("group_cols"),
            value_cols=value_cols,
            sample_col=cfg.get("sample_col"),
            combine_two_groups=cfg.get("combine_two_groups", False),
            alpha=alpha_thresh,
            min_cliffs_delta=effect_thresh,
            mode=cfg.get("stats_mode", "nonparametric"),
            output_dir=None,
            ignore_zeros=cfg.get("ignore_zeros", True)
        )
        
        sig_posthoc = ph_df[(ph_df["q_fdr_bh"] < alpha_thresh) & (ph_df["cliffs_delta"].abs() >= effect_thresh)]
        
        if sig_posthoc.empty:
            print(f"[Significance Filter] No metrics met significance threshold (q < {alpha_thresh}, |delta| >= {effect_thresh}). No plots generated.")
            return
            
        ranked_metrics = (
            sig_posthoc.groupby("item")["cliffs_delta"]
            .apply(lambda s: s.abs().max())
            .sort_values(ascending=False)
            .index.tolist()
        )
        
        top_n = cfg.get("top_n_metrics")
        if top_n and isinstance(top_n, int):
            ranked_metrics = ranked_metrics[:top_n]
            
        print(f"[Significance Filter] Screened {len(value_cols) if value_cols else 'all'} metrics -> narrowed to {len(ranked_metrics)} most significant: {ranked_metrics}")
        value_cols = ranked_metrics

    return run_group_stats_pipeline(
        df=df,
        group_cols=cfg.get("group_cols"),
        value_cols=value_cols,
        col_order=cfg.get("col_order"),
        plot_types=cfg.get("plot_types", "both"),
        combine_two_groups=cfg.get("combine_two_groups", False),
        filter_values=cfg.get("filter_values"),
        palette=cfg.get("palette", ["#e85b1e", "#f19d78", "#37498b", "#a7b4e3"]),
        add_brackets=cfg.get("add_brackets", True),
        output_dir=cfg.get("output_dir"),
        strip_hue_col=cfg.get("sample_col"),
        strip_palette=cfg.get("strip_palette", "husl"),
        save_legend=cfg.get("save_strip_legend", True),
        font_scale=cfg.get("font_scale", 1.5),
        ignore_zeros=cfg.get("ignore_zeros", True),
        stats_mode=cfg.get("stats_mode", "nonparametric"),
        min_effect_size_bracket=cfg.get("min_effect_size_bracket", 0.147),
        lookup_table=cfg.get("lookup_table"),
        show_points=cfg.get("show_points", cfg.get("show_strip_points", True)),
        max_points=cfg.get("max_points", cfg.get("max_strip_points", cfg.get("max_points_per_group", None))),
        export_format=cfg.get("export_format", cfg.get("stats_format", "both")),
        export_csv=cfg.get("export_csv", None),
        export_parquet=cfg.get("export_parquet", None)
    )


def main():
    parser = argparse.ArgumentParser(description="Run Group Stats & Publication Plotting Pipeline")
    parser.add_argument(
        "--config", "-c", 
        type=str, 
        default=str(Path(__file__).parent / "configs" / "vallon_stats_config.json"),
        help="Path to JSON configuration file (e.g. configs/xenium_stats_config.json)"
    )
    args = parser.parse_args()

    run_from_config(args.config)


if __name__ == "__main__":
    main()
