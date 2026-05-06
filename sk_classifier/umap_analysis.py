import matplotlib
matplotlib.use('Agg')
import umap
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_ion_umaps(
    df_wide, 
    unit_colors, 
    unit_names, 
    plots_directory, 
    output_dir=None,
    selected_ratios=None,
    selected_groups=None,
    selected_units=None,
    display_groups=None,
    display_units=None
):
    """
    Splits wide_df into positive and negative modes and generates UMAP plots.
    Allows filtering by specific ratios, groups, and units for both embedding and display.
    
    Args:
        df_wide: The wide-format DataFrame containing ratio means.
        unit_colors: Dictionary mapping unit names to hex colors.
        unit_names: Dictionary mapping unit names to display names.
        plots_directory: Base directory for saving plots.
        output_dir: Specific directory for UMAP plots (optional).
        selected_ratios: Ratios to use as features for embedding.
        selected_groups: Groups to include in the UMAP embedding calculation.
        selected_units: Units to include in the UMAP embedding calculation.
        display_groups: Groups to actually show on the plot (must be a subset of selected_groups).
        display_units: Units to actually show on the plot (must be a subset of selected_units).
    """
    if df_wide is None or df_wide.empty:
        print("Error: wide_df is empty or None. Please run the masking and quantification section first.")
        return

    # 1. Apply Row Filtering for EMBEDDING (Selection)
    df = df_wide.copy()
    if selected_units:
        df = df[df['Unit'].isin(selected_units)]
        print(f"Embedding calculation includes units: {selected_units} ({len(df)} instances)")
    
    if selected_groups:
        if 'Group_ID' in df.columns:
            df = df[df['Group_ID'].isin(selected_groups)]
            print(f"Embedding calculation includes groups: {selected_groups} ({len(df)} instances)")
        else:
            print("Warning: Group_ID column not found. Skipping group filter.")

    if df.empty:
        print("Error: No data remaining for embedding after applying filters.")
        return

    if output_dir is None:
        output_dir = os.path.join(plots_directory, 'umaps_ratios')

    # Set aesthetic parameters for premium look
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

    # 2. Identify metadata and feature columns
    metadata_cols = ['Image_Name', 'Unit', 'Instance_Label', 'Source_ID', 'centroid_x', 'centroid_y', 'Group_ID']
    metadata_cols = [c for c in metadata_cols if c in df.columns]
    
    # Identify available ratio columns
    all_ratio_cols = [c for c in df.columns if c not in metadata_cols]
    
    # Apply Column Filtering (Ratios)
    if selected_ratios:
        valid_ratios = [r for r in selected_ratios if r in all_ratio_cols]
        if not valid_ratios:
            print("Error: None of the selected ratios were found in the dataframe.")
            return
        all_ratio_cols = valid_ratios
        print(f"Using {len(all_ratio_cols)} selected ratios as features.")

    # Identify Ion Modes by Ratio_Type prefixes
    pos_ion_cols = [c for c in all_ratio_cols if 'Positive' in str(c)]
    neg_ion_cols = [c for c in all_ratio_cols if 'Negative' in str(c)]
    standard_cols = [c for c in all_ratio_cols if 'Positive' not in str(c) and 'Negative' not in str(c)]

    # 3. Define Mode Subsets
    if pos_ion_cols:
        df_pos = df[df[pos_ion_cols].notna().any(axis=1)].copy()
    else:
        df_pos = pd.DataFrame()

    if neg_ion_cols:
        df_neg = df[df[neg_ion_cols].notna().any(axis=1)].copy()
    else:
        df_neg = pd.DataFrame()

    if not pos_ion_cols and not neg_ion_cols:
        modes = [('Combined Ratio Analysis', df, standard_cols)]
    else:
        modes = [
            ('Positive Ion Mode', df_pos, standard_cols + pos_ion_cols),
            ('Negative Ion Mode', df_neg, standard_cols + neg_ion_cols)
        ]

    for mode_name, subset_df, feature_cols in modes:
        if subset_df.empty or len(subset_df) < 5:
            continue
            
        active_features = [f for f in feature_cols if subset_df[f].notna().any()]
        if not active_features:
            continue

        print(f"\n--- Processing {mode_name} ({len(subset_df)} instances embedded) ---")
        
        # Prepare Features
        X = subset_df[active_features].fillna(0)
        X_scaled = StandardScaler().fit_transform(X)
        
        # Run UMAP
        print(f"  Computing UMAP embedding...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        embedding = reducer.fit_transform(X_scaled)
        
        subset_df['umap_1'] = embedding[:, 0]
        subset_df['umap_2'] = embedding[:, 1]
        
        # Apply display names for plotting
        if 'Unit' in subset_df.columns:
            subset_df['Unit_Display'] = subset_df['Unit'].map(lambda x: unit_names.get(x, x))

        # Generate plots for each label type
        for color_by in ['Unit', 'Group_ID']:
            if color_by not in subset_df.columns: continue
            
            # APPLY DISPLAY FILTERING (Show/Hide logic)
            plot_df = subset_df.copy()
            if display_units:
                plot_df = plot_df[plot_df['Unit'].isin(display_units)]
            if display_groups:
                plot_df = plot_df[plot_df['Group_ID'].isin(display_groups)]
            
            if plot_df.empty:
                print(f"  Skipping plot for {color_by}: No instances remaining after display filter.")
                continue

            print(f"  Generating plot for {color_by} ({len(plot_df)} instances shown)...")
            plt.figure(figsize=(12, 12), dpi=100)
            
            # Determine palette and order
            if color_by == 'Unit':
                current_units = plot_df['Unit'].unique()
                # Use display_units for order if provided, otherwise selected_units, otherwise sorted
                order_source = display_units if display_units else (selected_units if selected_units else sorted(current_units))
                hue_order_internal = [u for u in order_source if u in current_units]
                hue_order_internal += [u for u in current_units if u not in hue_order_internal]
                
                palette = {unit_names.get(u, u): unit_colors.get(u, '#808080') for u in hue_order_internal}
                hue_order_display = [unit_names.get(u, u) for u in hue_order_internal]
                plot_hue = 'Unit_Display'
                title_label = "Unit Type"
            else:
                current_groups = plot_df['Group_ID'].unique()
                # Use display_groups for order if provided
                order_source = display_groups if display_groups else (selected_groups if selected_groups else sorted(current_groups))
                hue_order_display = [g for g in order_source if g in current_groups]
                hue_order_display += [g for g in current_groups if g not in hue_order_display]
                
                palette = "husl"
                plot_hue = 'Group_ID'
                title_label = "Subgroup (Group ID)"

            sns.scatterplot(
                data=plot_df,
                x='umap_1', y='umap_2',
                hue=plot_hue,
                palette=palette,
                hue_order=hue_order_display,
                alpha=0.7,
                edgecolor='w',
                linewidth=0.5,
                s=25
            )
            
            plt.title(f"UMAP Visualization: {mode_name}\nColored by {title_label}", fontsize=15, fontweight='bold', pad=20)
            plt.xlabel("UMAP Dimension 1", fontsize=12)
            plt.ylabel("UMAP Dimension 2", fontsize=12)
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title=title_label, frameon=True, shadow=True)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                clean_mode = mode_name.lower().replace(' ', '_')
                plt.savefig(os.path.join(output_dir, f"umap_{clean_mode}_{color_by.lower()}.png"), bbox_inches='tight', dpi=300)
            
            plt.show()
