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

    return run_group_stats_pipeline(
        df=df,
        group_cols=cfg.get("group_cols"),
        value_cols=cfg.get("value_cols"),
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
        lookup_table=cfg.get("lookup_table")
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
