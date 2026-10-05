import pandas as pd
import sys 


from pathlib import Path

import re
sys.path.insert(0, str(Path(__file__).parent.parent))
from sk_classifier.group_stats_pipeline import run_group_stats_pipeline

df = pd.read_parquet(r"C:\Users\jvill\OneDrive - UC San Diego\Vallon Kidney\10-04-26_arithmetic_outputs\all_images_roi_metrics-final.parquet")

# df= df[df['Sample ID']<16]

df=df[df['Label'] != 'PT']
df['LD Area / Count'] = df['LD Total Area (um2)'] / df['LD Count']
df['LD Count Density'] = df['LD Count'] / (df['Area (px)'] * df['Pixel_Size_um']**2)

anova_df, posthoc_df, desc_df = run_group_stats_pipeline(
    df=df,
    group_cols=["Condition", "Label"],
    value_cols=[
        "% Sterols", 
        "% Cardiolipins",
        "% Phospholipids", 
        "% Lysophospholipids", 
        "% Glycerols", 
        "% Sphingolipids", 
        "% SFA", 
        "% PUFA", 
        "% Sterol Esters", 
        "LD Count", 
        "LD Total Area (um2)", 
        "LD Mean Area (um2)", 
        "LD Area Fraction (%)", 
        "LD Area / Count",
        "LD Count Density",
        "LD % Sterols", 
        "LD % Cardiolipins", 
        "LD % Phospholipids", 
        "LD % Lysophospholipids", 
        "LD % Glycerols", 
        "LD % Sphingolipids", 
        "LD % SFA", 
        "LD % PUFA", 
        "LD % Background", 
        "LD % Sterol Esters", 
        "Protein_Turnover Mean",
        "Lipid_Turnover Mean", 
        "Lipid_to_Protein Mean", 
        "Lipid_Unsaturation Mean", 
        "LD_Protein_Turnover Mean",  
        "LD_Lipid_Turnover Mean", 
        "LD_Lipid_to_Protein Mean", 
        "LD_Lipid_Unsaturation Mean", 
        ],
    plot_types="both",              # "violin", "bar", or "both"
    combine_two_groups=False,       # False = Faceted subplots; True = "Condition | Label" combined x-axis
    output_dir=r"C:\Users\jvill\OneDrive - UC San Diego\Vallon Kidney\10-04-26_arithmetic_outputs\plots\group_stats",
    ignore_zeros=True               # Automatically ignores 0, NaN, Inf values
)
