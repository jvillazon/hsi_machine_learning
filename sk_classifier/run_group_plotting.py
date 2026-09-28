import pandas as pd
import sys 


from pathlib import Path

import re
sys.path.insert(0, str(Path(__file__).parent.parent))
from sk_classifier.group_stats_pipeline import run_group_stats_pipeline

df = pd.read_parquet(r"C:\Users\jvill\OneDrive - UC San Diego\Vallon Kidney\headless_arithmetic_outputs\all_images_roi_metrics-final.parquet")

anova_df, posthoc_df, desc_df = run_group_stats_pipeline(
    df=df,
    group_cols=["Condition", "Label"],
    value_cols=["% Sterols", "Protein_Turnover Mean", "Lipid_to_Protein Mean"],
    plot_types="both",              # "violin", "bar", or "both"
    combine_two_groups=False,       # False = Faceted subplots; True = "Condition | Label" combined x-axis
    output_dir=r"C:\Users\jvill\OneDrive - UC San Diego\Vallon Kidney\headless_arithmetic_outputs\plots\group_stats"
)
