
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import tifffile
import warnings
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
    PROFILE = 'kidney_ftu'  # Update with desired profile name (without .json extension)
    
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
     mask_type, regions, display_units, molecules_to_display, ratios_to_display,
     display_names, unit_colors, unit_display_names, heatmap_unit_order,
     selected_molecules, selected_ratios) = apply_config(config)
    
    # Derive bubble_unit_order as reverse of heatmap order
    bubble_unit_order = heatmap_unit_order[::-1]
    
    # ========================================================================
    # DATASET INITIALIZATION (independent of profile)
    # ========================================================================

    # Define dataset parameters (these could also be moved to the config if desired)
    base_directory = r"D:\integrated_pipeline\HSI_data\data"
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
    plots_directory = os.path.join(os.path.dirname(base_directory), f'{chosen_model_name}_{mask_type}_plots')

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
                
                                
                # Read original predictions CSV once
                if not os.path.exists(csv_path):
                    print(f"Prediction CSV not found for {img_name}: {csv_path}")
                    continue
                
                # Check if mask folder exists
                if not os.path.exists(mask_folder):
                    print(f"Mask folder not found for {img_name}: {mask_folder}")
                    continue

                
                # Extract sample name
                sample_name = img_name.split('-')[0] if '-' in img_name else img_name_no_ext
                
                # Process predictions with masks
                try:
                    pred_results = visualizer.apply_rf_masking(
                        prediction_csv_path=csv_path,
                        mask_list_path=mask_folder,
                        regions=regions,
                        img_name=img_name_no_ext,
                        sample_name=sample_name,
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
                                regions=regions,
                                img_name=img_name_no_ext,
                                sample_name=sample_name,
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
            # Note: unit_mappings is not used in the current config system
            # To implement unit mappings, add them to your config JSON file
            percentages_df = visualizer.quantify_unit_class_percentages_nested(predictions_per_unit, unit_mappings=None)
        
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
            # Note: unit_mappings is not used in the current config system
            # To implement unit mappings, add them to your config JSON file
            ratios_df = visualizer.quantify_unit_ratio_means_nested(ratios_per_unit, unit_mappings=None)
            
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

    # Display statistical significance plots if enabled
    if display_plots:
        if  percentages_df is None:
            try:
                percentages_df = pd.read_csv(output_csv)
                print(f"\nLoaded percentages DataFrame from {os.path.basename(output_csv)}")
            except Exception as e:
                print(f"Failed to load percentages DataFrame: {e}")
        
        if ratios_df is None:
            if os.path.exists(output_ratio_csv):
                try:
                    if os.path.exists(output_ratio_csv):
                        ratios_df = pd.read_csv(output_ratio_csv)
                        print(f"\nLoaded mean ratios DataFrame from {os.path.basename(output_ratio_csv)}")
                except Exception as e:
                    print(f"Failed to load mean ratios DataFrame: {e}")
                    ratios_df = None

            
        # Perform one-way ANOVA comparing units for percentages_df
        if percentages_df is not None and not percentages_df.empty:
        #     print("\n" + "="*80)
        #     print("GENERATING BOX PLOTS AND PERFORMING ONE-WAY ANOVA")
        #     print("="*80)
        #     percentages_unit_dir = os.path.join(plots_directory, 'anova_unit_percentages')
        #     percentages_unit_results = visualizer.one_way_anova_unit_comparison(
        #         df=percentages_df,
        #         value_col='Percentage',
        #         grouping_col='Molecule',
        #         output_dir=percentages_unit_dir,
        #         data_type='percentage',
        #         show_plots=False,  # Set to True to display plots interactively
        #         display_name_map=display_names,
        #         unit_color_map=unit_colors,
        #         unit_display_map=unit_display_names
        #     )

        # if ratios_df is not None and not ratios_df.empty:
        #     ratios_unit_dir = os.path.join(plots_directory, 'anova_unit_ratios')
        #     ratios_unit_results = visualizer.one_way_anova_unit_comparison(
        #         df=ratios_df,
        #         value_col='Mean_Ratio',
        #         grouping_col='Ratio_Type',
        #         output_dir=ratios_unit_dir,
        #         data_type='ratio',
        #         show_plots=False,  # Set to True to display plots interactively
        #         display_name_map=display_names,
        #         unit_color_map=unit_colors,
        #         unit_display_map=unit_display_names
        #     )


            print("\nGENERATING HEATMAPS")
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
                units_to_display=display_units,
                unit_display_map=unit_display_names,
                unit_order=heatmap_unit_order,
                cmap= 'RdBu_r'  # Red-Blue colormap for percentages
            )
            visualizer.generate_unit_heatmaps(
                df=ratios_df,
                value_col='Mean_Ratio',
                grouping_col='Ratio_Type',
                output_dir=os.path.join(plots_directory, 'heatmaps_unit_ratios'),
                data_type='ratio',
                show_plots=False,
                display_name_map=display_names,
                units_to_display=display_units,
                unit_display_map=unit_display_names,
                unit_order=heatmap_unit_order,
                cmap = 'RdBu_r'  # Red-Blue colormap for ratios (adjust as needed)
            )

            print("\nGENERATING BUBBLE PLOTS")

            percentages_unit_buble_dir = os.path.join(plots_directory, 'bubble_plots_unit_percentages')
            visualizer.generate_unit_bubble_charts(
                df=percentages_df,
                value_col='Percentage',
                grouping_col='Molecule',
                output_dir=percentages_unit_buble_dir,
                data_type='percentage',
                show_plots=False,
                display_name_map=display_names,
                units_to_display=display_units,
                unit_display_map=unit_display_names,
                unit_order=bubble_unit_order,
                groups_to_display=molecules_to_display,
                cmap = 'RdBu_r'
            )

            ratios_unit_buble_dir = os.path.join(plots_directory, 'bubble_plots_unit_ratios')
            visualizer.generate_unit_bubble_charts(
                df=ratios_df,
                value_col='Mean_Ratio',
                grouping_col='Ratio_Type',
                output_dir=ratios_unit_buble_dir,
                data_type='ratio',
                show_plots=False,
                display_name_map=display_names,
                units_to_display=display_units,
                unit_display_map=unit_display_names,
                unit_order=bubble_unit_order,
                groups_to_display=ratios_to_display,
                cmap = 'RdBu_r'
            )

            # ----------------------------------------------------------------
            # Per-sample heatmaps and bubble charts
            # Groups rows by the Sample_Name column (e.g. 'aki', 'ckd', 'ref')
            # and generates a top-10 heatmap + bubble chart for each group.
            # ----------------------------------------------------------------
            print("\nGENERATING PER-SAMPLE HEATMAPS AND BUBBLE CHARTS")

            for df_label, df_iter, val_col, grp_col, dtype in [
                ('percentages', percentages_df, 'Percentage',  'Molecule',   'percentage'),
                ('ratios',      ratios_df,      'Mean_Ratio',  'Ratio_Type', 'ratio'),
            ]:
                if df_iter is None or df_iter.empty:
                    continue
                if 'Sample_Name' not in df_iter.columns:
                    print(f"  Sample_Name column not found in {df_label} DataFrame, skipping.")
                    continue

                unique_samples = sorted(df_iter['Sample_Name'].dropna().unique())
                print(f"\nFound {len(unique_samples)} sample group(s) in {df_label}: {unique_samples}")

                for sname in unique_samples:
                    sample_df = df_iter[df_iter['Sample_Name'] == sname].copy()
                    if sample_df.empty:
                        continue
                    print(f"  Sample '{sname}': {len(sample_df)} records")

                    _groups = molecules_to_display if grp_col == 'Molecule' else ratios_to_display

                    # Heatmap
                    visualizer.generate_unit_heatmaps(
                        df=sample_df,
                        value_col=val_col,
                        grouping_col=grp_col,
                        output_dir=os.path.join(plots_directory, f'per_sample_heatmaps_{df_label}', sname),
                        data_type=dtype,
                        top_n=10,
                        groups_to_display=_groups,
                        show_plots=False,
                        display_name_map=display_names,
                        units_to_display=display_units,
                        unit_display_map=unit_display_names,
                        unit_order=heatmap_unit_order,
                        cmap='RdBu_r'
                    )

                    # Bubble chart
                    visualizer.generate_unit_bubble_charts(
                        df=sample_df,
                        value_col=val_col,
                        grouping_col=grp_col,
                        output_dir=os.path.join(plots_directory, f'per_sample_bubble_plots_{df_label}', sname),
                        data_type=dtype,
                        top_n=10,
                        groups_to_display=_groups,
                        show_plots=False,
                        display_name_map=display_names,
                        units_to_display=display_units,
                        unit_display_map=unit_display_names,
                        unit_order=bubble_unit_order,
                        cmap = 'RdBu_r'
                    )

        else:
            print("\nSkipping unit comparison ANOVA for percentages: No data available")
        
        # Generate raw (non-normalized) ratio heatmap
        if ratios_df is not None and not ratios_df.empty:
            ratios_raw_heatmap_dir = os.path.join(plots_directory, 'heatmap_raw_ratios')
            raw_ratio_heatmap_data = visualizer.generate_raw_ratio_heatmap(
                df=ratios_df,
                value_col='Mean_Ratio',
                grouping_col='Ratio_Type',
                output_dir=ratios_raw_heatmap_dir,
                show_plots=False,
                display_name_map=display_names,
                units_to_display=display_units,
                unit_display_map=unit_display_names,
                unit_order=heatmap_unit_order
            )
        else:
            print("\nSkipping heatmap generation for ratios: No data available")

        # Generate multi-panel box plots for selected molecules/ratios
        print("\nGENERATING MULTI-PANEL BOX PLOTS")
        
        if percentages_df is not None and not percentages_df.empty:
            # Define molecules to include in multi-panel figure
            selected_molecules = ['18:1 Cholesterol ester', 'AG 16:0',
                                'PC Mix', '16:0 Cardiolipin']
            
            multi_panel_percentages_dir = os.path.join(plots_directory, 'multi_panel_percentages')
            visualizer.create_multi_panel_boxplots(
                df=percentages_df,
                value_col='Percentage',
                grouping_col='Molecule',
                molecules_list=selected_molecules,
                nrows=2,
                ncols=2,
                output_dir=multi_panel_percentages_dir,
                data_type='percentage',
                show_plots=False,
                display_name_map=display_names,
                unit_color_map=unit_colors,
                # unit_display_map=unit_display_names
            )
            print(f"\nGenerated multi-panel box plots for percentages in {multi_panel_percentages_dir}")
        
        if ratios_df is not None and not ratios_df.empty:
            # Define ratio types to include in multi-panel figure
            selected_ratios = ['Redox', 'Lipid_Unsaturation']
            
            multi_panel_ratios_dir = os.path.join(plots_directory, 'multi_panel_ratios')
            visualizer.create_multi_panel_boxplots(
                df=ratios_df,
                value_col='Mean_Ratio',
                grouping_col='Ratio_Type',
                molecules_list=selected_ratios,
                nrows=1,
                ncols=2,
                output_dir=multi_panel_ratios_dir,
                data_type='ratio',
                show_plots=False,
                display_name_map=display_names,
                unit_color_map=unit_colors,
                # unit_display_map=unit_display_names
            )
            print(f"\nGenerated multi-panel box plots for ratios in {multi_panel_ratios_dir}")


if __name__ == "__main__":
    main()