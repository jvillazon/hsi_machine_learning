
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob

import sys 
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.hsi_unlabeled_dataset import HSI_Unlabeled_Dataset
from core.hsi_visualizer import HSI_Visualizer
from core.hsi_sk_classifier import HSI_Classifier
from core.hsi_labeled_dataset import HSI_Labeled_Dataset
from config_loader import load_config, apply_config, list_available_profiles, print_config_info


def main():
    # ========================================================================
    # CONFIGURATION PROFILE SELECTION
    # ========================================================================
    # Available profiles: 'kidney_clusters' (default), 'kidney_nephron'
    # Change this to switch between different configurations
    PROFILE = 'kidney_clusters'  # Update with desired profile name (without .json extension)
    
    # Print available profiles on first run
    print(f"Available configuration profiles:")
    for profile in list_available_profiles():
        print(f"  - {profile}")
    
    # Load configuration from selected profile
    print(f"\nLoading configuration from profile: {PROFILE}")
    config = load_config(PROFILE)
    print_config_info(PROFILE)
    
    # Apply configuration - unpack all settings
    (perform_RF_classification, perform_masking, display_plots,
     mask_type, subgroups, display_units_maps, molecules_to_display, ratios_to_display,
     display_names, unit_mappings, unit_colors, unit_display_names, heatmap_unit_order,
     selected_molecules, selected_ratios, display_units_boxplots,
     individual_plot_config, sample_analysis_config) = apply_config(config)
    
    display_units_boxplots = ["Cluster2", "Cluster3", "Cluster4", "Cluster5"]

    if display_units_boxplots:
        display_units_boxplots_type = f"{display_units_boxplots[0]}-{display_units_boxplots[-1]}"
    

    if display_units_boxplots:
        print("Currently selected boxplot units:")
        for i, unit in enumerate(display_units_boxplots):
            dname = unit_display_names.get(unit, unit)
            print(f"  {i+1}. {unit}: {dname}")
    else:
        print("No specific units designated - will default to all available units.")

    # Extract analysis granularity settings
    granularity_levels = sample_analysis_config.get('granularity_levels', [])
    selected_values = sample_analysis_config.get('selected_values', None) # Renamed from selected_samples to be agnostic
    filter_col = sample_analysis_config.get('filter_col', 'Source_ID') # Allow specifying the filter column

    # Derive bubble_unit_order as reverse of heatmap order
    bubble_unit_order = heatmap_unit_order[::-1]
    
    # ========================================================================
    # DATASET INITIALIZATION (independent of profile)
    # ========================================================================

    # Define dataset parameters (these could also be moved to the config if desired)
    base_directory = r"D:\integrated_pipeline\registered_data\CODEX\data"
    wn_1 = 2700  # Starting wavenumber (inclusive)
    wn_2 = 3100  # Ending wavenumber (inclusive)
    num_samp = 61
    ch_start = int(((wn_2 - 2800)/(wn_2 - wn_1)) * num_samp)  # Calculate channel index corresponding to end of cell silent region (2800 cm^-1)

    # Define model name
    chosen_model_name = 'rf_n150_md40_mss2_msl9_a9_best_model_platt'  # Update with correct model name
    model_path = f'sk_classifier/models/{chosen_model_name}.joblib'

    # Create visualizer
    visualizer = HSI_Visualizer(
        mol_path='molecule_dataset/lipid_subtype_wn_61_test.npz',
        wavenumber_start=wn_1,
        wavenumber_end=wn_2,
        num_samples=num_samp
    )
    plots_directory = os.path.join(os.path.dirname(base_directory), f'{chosen_model_name}_{mask_type}_S21_plots')

    # Create dataset with specified wavenumber range
    dataset = HSI_Unlabeled_Dataset(
        base_directory,
        ch_start,
        transform=None,
        image_normalization=True,
        min_max_normalization=False,
        num_samples=num_samp,
        wavenumber_start=wn_1,
        wavenumber_end=wn_2,
        compute_stats=perform_RF_classification
    )
    
    labeled_dataset = HSI_Labeled_Dataset(
        molecule_dataset_path='molecule_dataset/lipid_subtype_wn_61_test',
        srs_params_path='params_dataset/srs_params_61',
        num_samples_per_class=10000,
        normalize_per_molecule=False,
        compute_min_max=True,
        noise_multiplier=0.5
    )
    # Initialize Random Forest classifier
    output_base = os.path.join(os.path.dirname(base_directory), f"{chosen_model_name}_outputs")
    classifier = HSI_Classifier(dataset, model_path=model_path, output_base=output_base, visualizer=visualizer, labeled_dataset=labeled_dataset)

    # Initialize variables that may be used across conditional blocks
    percentages_df = None
    ratios_df = None

    if perform_RF_classification:    
        if os.path.exists(model_path):
            # Run inference
            print("\nRunning RandomForest inference on dataset...")
            timing_stats = classifier.predict(alpha=10, generate_shap=True)  # Adjust alpha as needed for spectral weighting
        else:
            print(f"No model found at {model_path}")
            # Just show some sample spectra without predictions
            sample_indices = [100001, 200001, 300001]
            for idx in sample_indices:
                spectra, img_idx = dataset[idx]
                visualizer.visualize_spectrum(spectra, img_idx=img_idx)

    # Make mask directory to save .csv outputs from masked quantification
    masked_output_dir = os.path.join(os.path.dirname(base_directory), f"{mask_type}_output_csv")
    os.makedirs(masked_output_dir, exist_ok=True)

    output_csv = os.path.join(masked_output_dir, 'masked_predictions_percentages.csv')
    output_ratio_csv = os.path.join(masked_output_dir, 'masked_ratio_means.csv')

    if perform_masking:
        # Apply masking if enabled
        mask_prefix = mask_type.split("_")[0]  
        print("\nProcessing masks and quantifying results...")
        
        predictions_per_unit = {}
        ratios_per_unit = {}
        
        # mask_type = 'Xenium_cells_masks'  # Update with actual mask type if different
        
        with tqdm(enumerate(dataset.img_list), total=len(dataset.img_list), desc="Processing masks") as pbar:
            for img_idx, img_path in pbar:
                stats = dataset.image_stats[img_path]
                img_name = os.path.basename(img_path)
                img_name_no_ext = os.path.splitext(img_name)[0]
                img_folder = os.path.join(classifier.output_base, img_name_no_ext)
                mask_folder = os.path.join(os.path.dirname(base_directory), mask_type, img_name_no_ext)

                # Get original prediction CSV and ratio folder
                csv_path = os.path.join(img_folder, f"{img_name_no_ext}_predictions.csv")
                ratio_folder = os.path.join(os.path.dirname(base_directory), 'Ratio', img_name_no_ext)
                
                                

                
                # Check if mask folder exists
                if not os.path.exists(mask_folder):
                    print(f"Mask folder not found for {img_name}: {mask_folder}")
                    continue

                # Process predictions with masks
                try:
                    # Read original predictions CSV once
                    if not os.path.exists(csv_path):
                        print(f"\nPrediction CSV not found for {img_name}: {csv_path}")
                    else:
                        pred_results = visualizer.apply_rf_masking(
                            prediction_csv_path=csv_path,
                            mask_list_path=mask_folder,
                            subgroups=subgroups,
                            img_name=img_name_no_ext,
                            prefix=mask_prefix,
                            group_subclasses=True
                        )
                    
                        # Merge results into predictions_per_unit
                        for unit_name, entries in pred_results.items():
                            if unit_name not in predictions_per_unit:
                                predictions_per_unit[unit_name] = entries
                            else:
                                predictions_per_unit[unit_name].extend(entries)
                except Exception as e:
                    print(f"Failed to process predictions for {img_name}: {e}")
                
                # Process ratio TIFFs with masks
                if os.path.exists(ratio_folder):
                    ratio_tiff_paths = glob(os.path.join(ratio_folder, '*.tif'))
                    ratio_tiff_paths = [f for f in ratio_tiff_paths if '_masked' not in os.path.basename(f)]
                    
                    for ratio_tiff_path in ratio_tiff_paths:
                        try:
                            ratio_results = visualizer.apply_rf_masking(
                                ratio_tiff_path=ratio_tiff_path,
                                mask_list_path=mask_folder,
                                subgroups=subgroups,
                                img_name=img_name_no_ext,
                                prefix=mask_prefix
                            )
                            
                            # Merge results into ratios_per_unit
                            for unit_name, entries in ratio_results.items():
                                if unit_name not in ratios_per_unit:
                                    ratios_per_unit[unit_name] = entries
                                else:
                                    ratios_per_unit[unit_name].extend(entries)
                        except Exception as e:
                            print(f"Failed to process ratio {os.path.basename(ratio_tiff_path)} for {img_name}: {e}")
        
        # Calculate percentages
        if predictions_per_unit:
            percentages_df = visualizer.quantify_unit_class_percentages_nested(predictions_per_unit, unit_mappings=unit_mappings)
        
            # Display summary
            print("\nPercentages DataFrame (first 10 rows):")
            print(percentages_df.head(10))
            
            # Save percentages to CSV
            percentages_df.to_csv(output_csv, index=False)
            print(f"\nSaved percentages DataFrame to {os.path.basename(output_csv)}")
        else:
            print("\nNo masked predictions found to quantify.")
        
        # Calculate mean ratios
        if ratios_per_unit:
            ratios_df = visualizer.quantify_unit_ratio_means_nested(ratios_per_unit, unit_mappings=unit_mappings)
            
            # Display summary
            print("\nMean Ratios DataFrame (first 10 rows):")
            print(ratios_df.head(10))
            
            # Save ratios to CSV
            ratios_df.to_csv(output_ratio_csv, index=False)
            print(f"\nSaved mean ratios DataFrame to {os.path.basename(output_ratio_csv)}")
        else:
            print("\nNo masked ratio TIFFs found to quantify.")
    
    # Load existing DataFrames for plotting if not generated above
    if display_plots and not perform_masking:
        try:
            percentages_df = pd.read_csv(output_csv)
            print(f"\nLoaded percentages DataFrame from {os.path.basename(output_csv)}")
        except Exception as e:
            print(f"Failed to load percentages DataFrame: {e}")
        
        try:
            if os.path.exists(output_ratio_csv):
                ratios_df = pd.read_csv(output_ratio_csv)
                print(f"\nLoaded mean ratios DataFrame from {os.path.basename(output_ratio_csv)}")
        except Exception as e:
            print(f"Failed to load mean ratios DataFrame: {e}")
    # --- Optional Data Filtering ---
    # Apply global filters defined in config
    if selected_values is not None and filter_col:
        for df in [percentages_df, ratios_df]:
            if df is not None and filter_col in df.columns:
                df = df[df[filter_col].isin(selected_values)].copy()
                print(f"  Applied global filter: {filter_col} in {selected_values}")
    
    # --- S21-Anchored Unit Filtering ---
    # Filter both dataframes to only include units that have presence in S21
    anchor_source = 'S21'
    if ratios_df is not None and 'Source_ID' in ratios_df.columns:
        unit_counts = ratios_df[ratios_df['Source_ID'] == anchor_source]['Unit'].value_counts() 
        s21_units = unit_counts[unit_counts > 10].index.tolist()
        
        if len(s21_units) > 0:
            # Anchor units (keep units from S21, include their S24 data if it exists)
            if percentages_df is not None:
                percentages_df = percentages_df[percentages_df['Unit'].isin(s21_units)].copy()
                print(f"  Unit-level filter: Kept {len(s21_units)} units that exist in {anchor_source} (percentages)")
            
            if ratios_df is not None:
                # 1. Prioritize S21 for Redox (Remove non-S21 Redox entries)
                # redox_mask = (ratios_df['Ratio_Type'] == 'Redox') #& (ratios_df['Source_ID'] != anchor_source)
                # ratios_df = ratios_df[~redox_mask].copy()
                
                # 2. Re-anchor Ratio units (keep units from S21)
                ratios_df = ratios_df[ratios_df['Unit'].isin(s21_units)].copy()
                print(f"  Unit-level filter: Kept units that exist in {anchor_source} for ratios")
                print(f"  Item-level filter: Redox analysis restricted to {anchor_source} only")
        else:
            print(f"  Warning: No data found for anchor source {anchor_source}. Skipping unit-anchored filtering.")

    # Display statistical significance plots if enabled
    if display_plots:
        # Extract individual plot settings early to avoid UnboundLocalError in granularity loop
        create_individual_plots = individual_plot_config.get('create_individual_plots', True)
        individual_molecules = individual_plot_config.get('selected_molecules', "All")
        individual_ratios = individual_plot_config.get('selected_ratios', "All")

        if  percentages_df is None:
            try:
                percentages_df = pd.read_csv(output_csv)
                print(f"\nLoaded percentages DataFrame from {os.path.basename(output_csv)}")
            except Exception as e:
                print(f"Failed to load data DataFrame: {e}")

        if percentages_df is not None and not percentages_df.empty:
            print("\n================================================================================")
            print("GENERATING HEATMAPS")
            print("================================================================================ ")

            # Generate unit-aggregated heatmaps for percentages
            percentages_unit_heatmap_dir = os.path.join(plots_directory, 'heatmaps_unit_percentages')
            visualizer.generate_unit_heatmaps(
                df=percentages_df,
                value_col='Percentage',
                grouping_col='Molecule',
                output_dir=percentages_unit_heatmap_dir,
                data_type='percentage',
                show_plots=False,
                display_name_map=display_names,
                units_to_display=display_units_maps,
                unit_display_map=unit_display_names,
                unit_order=heatmap_unit_order,
                cmap= 'RdBu_r'  # Red-Blue colormap for percentages
            )
        if ratios_df is not None and not ratios_df.empty:
            visualizer.generate_unit_heatmaps(
                df=ratios_df,
                value_col='Mean_Ratio',
                grouping_col='Ratio_Type',
                output_dir=os.path.join(plots_directory, 'heatmaps_unit_ratios'),
                data_type='ratio',
                show_plots=False,
                display_name_map=display_names,
                units_to_display=display_units_maps,
                unit_display_map=unit_display_names,
                unit_order=heatmap_unit_order,
                cmap = 'RdBu_r'  # Red-Blue colormap for ratios (adjust as needed)
            )

        print("\n================================================================================")
        print("GENERATING BUBBLE PLOTS")
        print("================================================================================ ")
        if percentages_df is not None and not percentages_df.empty:
            percentages_unit_buble_dir = os.path.join(plots_directory, 'bubble_plots_unit_percentages')
            visualizer.generate_unit_bubble_charts(
                df=percentages_df,
                value_col='Percentage',
                grouping_col='Molecule',
            output_dir=percentages_unit_buble_dir,
            data_type='percentage',
            show_plots=False,
            display_name_map=display_names,
            units_to_display=display_units_maps,
            unit_display_map=unit_display_names,
            unit_order=bubble_unit_order,
            groups_to_display=molecules_to_display,
            cmap = 'RdBu_r'
        )

        if ratios_df is not None and not ratios_df.empty:
            ratios_unit_buble_dir = os.path.join(plots_directory, 'bubble_plots_unit_ratios')
            visualizer.generate_unit_bubble_charts(
                df=ratios_df,
                value_col='Mean_Ratio',
                grouping_col='Ratio_Type',
                output_dir=ratios_unit_buble_dir,
                data_type='ratio',
                show_plots=False,
                display_name_map=display_names,
                units_to_display=display_units_maps,
                unit_display_map=unit_display_names,
                unit_order=bubble_unit_order,
                groups_to_display=ratios_to_display,
                cmap = 'RdBu_r'
            )


            ratios_raw_bubble_dir = os.path.join(plots_directory, 'bubble_chart_raw_ratios')
            raw_ratio_bubble_data = visualizer.generate_raw_ratio_bubble_chart(
                df=ratios_df,
                value_col='Mean_Ratio',
                grouping_col='Ratio_Type',
                output_dir=ratios_raw_bubble_dir,
                show_plots=False,
                display_name_map=display_names,
                units_to_display=display_units_maps,
                unit_display_map=unit_display_names,
                unit_order=bubble_unit_order
            )
        else:
            print("\nSkipping bubble chart generation for ratios: No data available")

        # ----------------------------------------------------------------
        # Multi-Granularity Analysis
        # Generates plots based on configurations like ["Sample_Name"], ["Group_ID"], etc.
        # ----------------------------------------------------------------
        if granularity_levels:
            print("\n================================================================================")
            print("GENERATING MULTI-GRANULARITY HEATMAPS AND BUBBLE CHARTS")
            print("================================================================================ ")  

            for df_label, df_iter, val_col, grp_col, dtype in [
                ('percentages', percentages_df, 'Percentage',  'Molecule',   'percentage'),
                ('ratios',      ratios_df,      'Mean_Ratio',  'Ratio_Type', 'ratio'),
            ]:
                if df_iter is None or df_iter.empty:
                    continue
                
                # Global filter if provided (redundant check if already applied above, but keeps loop robust)
                if selected_values is not None and filter_col in df_iter.columns:
                    df_iter = df_iter[df_iter[filter_col].isin(selected_values)].copy()

                for group_cols in granularity_levels:
                    # Check if all columns exist
                    cols_to_use = [c for c in group_cols if c in df_iter.columns]
                    if not cols_to_use:
                        print(f"  Skipping granularity level {group_cols} (columns not found)")
                        continue
                    
                    print(f"\n  Analyzing granularity: {cols_to_use}")
                    
                    # Find unique combinations of the specified columns
                    combinations = df_iter[cols_to_use].drop_duplicates().values
                    
                    for combo in combinations:
                        # Create mask for this combination
                        mask = True
                        combo_parts = []
                        for i, col in enumerate(cols_to_use):
                            val = combo[i]
                            mask &= (df_iter[col] == val) if pd.notna(val) else df_iter[col].isna()
                            combo_parts.append(str(val))
                        
                        subset_df = df_iter[mask].copy()
                        if subset_df.empty:
                            continue
                            
                        # Flat name for folder and titles
                        combo_str = "_".join(combo_parts)
                        print(f"    Processing grouping: {combo_str} ({len(subset_df)} records)")

                        level_folder = "_".join(cols_to_use)
                        output_dir_base = os.path.join(plots_directory, f'granularity_{df_label}', level_folder, combo_str)
                        
                        _groups = molecules_to_display if grp_col == 'Molecule' else ratios_to_display

                        # Heatmap
                        visualizer.generate_unit_heatmaps(
                            df=subset_df,
                            value_col=val_col,
                            grouping_col=grp_col,
                            output_dir=os.path.join(output_dir_base, 'heatmaps'),
                            data_type=dtype,
                            top_n=10,
                            groups_to_display=_groups,
                            show_plots=False,
                            display_name_map=display_names,
                            units_to_display=display_units_maps,
                            unit_display_map=unit_display_names,
                            unit_order=heatmap_unit_order,
                            cmap='RdBu_r'
                        )

                        # Bubble chart
                        visualizer.generate_unit_bubble_charts(
                            df=subset_df,
                            value_col=val_col,
                            grouping_col=grp_col,
                            output_dir=os.path.join(output_dir_base, 'bubble_plots'),
                            data_type=dtype,
                            top_n=10,
                            groups_to_display=_groups,
                            show_plots=False,
                            display_name_map=display_names,
                            units_to_display=display_units_maps,
                            unit_display_map=unit_display_names,
                            unit_order=bubble_unit_order,
                            cmap='RdBu_r'
                        )

                        # Individual Bar Plots
                        if create_individual_plots and len(cols_to_use) == 1:
                            # Determine items to plot based on the config selection
                            if df_label == 'percentages':
                                items_to_plot = individual_plot_config.get('selected_molecules', 'All')
                            else:
                                items_to_plot = individual_plot_config.get('selected_ratios', 'All')

                            print(f"    Generating individual bar plots for {combo_str} (Unit-level comparison)")
                            visualizer.create_individual_barplots(
                                df=subset_df,
                                value_col=val_col,
                                grouping_col=grp_col,
                                items_list=items_to_plot,
                                output_dir=os.path.join(output_dir_base, f'individual_bar_{display_units_boxplots_type}_plots'),
                                data_type=dtype,
                                show_plots=False,
                                display_name_map=display_names,
                                unit_color_map=unit_colors,
                                unit_display_map=unit_display_names,
                                units_to_display=display_units_boxplots,
                                compare_by=None,  # Consolidate: No compare_by for granularity subfolders
                                compare_order=None,
                                consolidate_Group_IDs=individual_plot_config.get('consolidate_Group_IDs', True)
                            )
        else:
            print("\nSkipping granularity analysis (no levels defined in config)")
    
    # Generate individual box plots for selected molecules/ratios
    print("\n================================================================================")
    print("GENERATING INDIVIDUAL BOX PLOTS")
    print("================================================================================")
    
    if create_individual_plots:

        if percentages_df is not None and not percentages_df.empty and individual_molecules:
            percentages_output_dir = os.path.join(plots_directory, f'individual_bar_plots_{display_units_boxplots_type}_percentages')

            # percentages_individual_dir = os.path.join(plots_directory, 'individual_percentages')
            # print("\nCreating individual plots for percentages...")
            # visualizer.create_individual_boxplots(
            #     df=percentages_df,
            #     value_col='Percentage',
            #     grouping_col='Molecule',
            #     items_list=individual_molecules,
            #     output_dir=percentages_individual_dir,
            #     data_type='percentage',
            #     figure_width=individual_figure_width,
            #     figure_height=individual_figure_height,
            #     show_plots=False,
            #     display_name_map=display_names,
            #     unit_color_map=unit_colors,
            #     unit_display_map=unit_display_names,
            #     units_to_display=display_units_boxplots,
            #     compare_by='Sample_Type' if sample_compare_order else None,
            #     compare_order=sample_compare_order if sample_compare_order else None,
            #     consolidate_Group_IDs=True
            # )
            # print(f"\nGenerated individual box plots for percentages in {percentages_individual_dir}")

            print("\nCreating individual bar plots for percentages...")
            visualizer.create_individual_barplots(
                df=percentages_df,
                value_col='Percentage',
                grouping_col='Molecule',
                items_list=individual_molecules,
                output_dir=percentages_output_dir,
                data_type='percentage',
                show_plots=False,
                display_name_map=display_names,
                unit_color_map=unit_colors,
                unit_display_map=unit_display_names,
                units_to_display=display_units_boxplots,
                compare_by=individual_plot_config.get('compare_by', None),
                compare_order=individual_plot_config.get('compare_order', None),
                consolidate_Group_IDs=individual_plot_config.get('consolidate_Group_IDs', True)
            )
            visualizer.create_individual_barplots(
                df=percentages_df,
                value_col='Percentage',
                grouping_col='Molecule',
                items_list=individual_molecules,
                output_dir=percentages_output_dir,
                data_type='percentage',
                show_plots=False,
                display_name_map=display_names,
                unit_color_map=unit_colors,
                unit_display_map=unit_display_names,
                units_to_display=display_units_boxplots,
                compare_by=None,
                compare_order=None,
                consolidate_Group_IDs=individual_plot_config.get('consolidate_Group_IDs', True)
            )
            print(f"Generated individual bar plots for percentages")
        
        if ratios_df is not None and not ratios_df.empty and individual_ratios:
            ratios_output_dir = os.path.join(plots_directory, f'individual_bar_plots_{display_units_boxplots_type}_ratios')
            # ratios_individual_dir = os.path.join(plots_directory, 'individual_ratios')
            # print("\nCreating individual plots for ratios...")
            # visualizer.create_individual_boxplots(
            #     df=ratios_df,
            #     value_col='Mean_Ratio',
            #     grouping_col='Ratio_Type',
            #     items_list=individual_ratios,
            #     output_dir=ratios_individual_dir,
            #     data_type='ratio',
            #     figure_width=individual_figure_width,
            #     figure_height=individual_figure_height,
            #     show_plots=False,
            #     display_name_map=display_names,
            #     unit_color_map=unit_colors,
            #     unit_display_map=unit_display_names,
            #     units_to_display=display_units_boxplots,
            #     compare_by='Sample_Type' if sample_compare_order else None,
            #     compare_order=sample_compare_order if sample_compare_order else None,
            #     consolidate_Group_IDs=True
            # )
            # print(f"\nGenerated individual box plots for ratios in {ratios_individual_dir}")
            print("\nCreating individual bar plots for ratios...")
            visualizer.create_individual_barplots(
                df=ratios_df,
                value_col='Mean_Ratio',
                grouping_col='Ratio_Type',
                items_list=individual_ratios,
                output_dir=ratios_output_dir,
                data_type='ratio',
                show_plots=False,
                display_name_map=display_names,
                unit_color_map=unit_colors,
                unit_display_map=unit_display_names,
                units_to_display=display_units_boxplots,
                compare_by=individual_plot_config.get('compare_by', None),
                compare_order=individual_plot_config.get('compare_order', None),
                consolidate_Group_IDs=individual_plot_config.get('consolidate_Group_IDs', True),
                
            )
            visualizer.create_individual_barplots(
                df=ratios_df,
                value_col='Mean_Ratio',
                grouping_col='Ratio_Type',
                items_list=individual_ratios,
                output_dir=ratios_output_dir,
                data_type='ratio',
                show_plots=False,
                display_name_map=display_names,
                unit_color_map=unit_colors,
                unit_display_map=unit_display_names,
                units_to_display=display_units_boxplots,
                compare_by=None,
                compare_order=None,
                consolidate_Group_IDs=individual_plot_config.get('consolidate_Group_IDs', True),
                
            )
            print(f"Generated individual bar plots for ratios")
    else:
        print("Individual plot configuration disabled")

if __name__ == "__main__":
    main()