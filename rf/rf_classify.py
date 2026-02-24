from glob import glob
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import tifffile
import warnings

import sys 
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.hsi_unlabeled_dataset import HSI_Unlabeled_Dataset
from core.hsi_visualizer import HSI_Visualizer
from rf.rf_random_forest import HSI_RandomForest


def main():
    # Define processing options
    perform_RF_classification = False # Set to True to run RandomForest classification   

    # Define dataset parameters
    base_directory = "D:\\ADATA Backup\\HuBMAP\\HuBMAP CODEX\\data"
    num_samp = 61
    wn_1 = 2700
    wn_2 = 3100
    ch_start = int((2800 - wn_1) / (wn_2 - wn_1) * num_samp)

    # Define model name
    chosen_model_name = 'best_model'  # Update with correct model name
    
    # Create visualizer
    visualizer = HSI_Visualizer(
        mol_path='molecule_dataset/lipid_subtype_wn_61_test.npz',
        wavenumber_start=wn_1,
        wavenumber_end=wn_2,
        num_samples=num_samp
    )

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
    

    # Initialize Random Forest classifier
    classifier = HSI_RandomForest(dataset, output_base=None, visualizer=visualizer)


    # Initialize variables that may be used across conditional blocks
    percentages_df = None
    ratios_df = None

    if perform_RF_classification:    
        # Load and run inference with Random Forest model
        rf_model_path = f'rf/models/{chosen_model_name}.joblib'  # Update with correct model name  

        if os.path.exists(rf_model_path):
            # Run inference
            print("\nRunning RandomForest inference on dataset...")
            predictions, probabilities, _ = classifier.predict(
                model_path=rf_model_path
            )
            
            print("\nRandomForest Inference Results:")
            print(f"Total predictions: {len(predictions)}")
            print(f"Number of classes: {probabilities.shape[1]}")
            
            # Show example predictions
            print("\nShowing sample predictions...")
            visualizer.show_random_predictions(dataset, predictions, probabilities,
                                            num_images=1, spectra_per_image=3, exclude_classes=['No Match'])
        else:
            print(f"No model found at {rf_model_path}")
            # Just show some sample spectra without predictions
            sample_indices = [100001, 200001, 300001]
            for idx in sample_indices:
                spectra, img_idx = dataset[idx]
                visualizer.visualize_spectrum(spectra, img_idx=img_idx)



    # (Optional) Define masking and quantification options
    perform_masking = True# Set to True to enable both FTU masking and ratio masking
    display_plots = False # Set to True to display statistical significance plots
   
    # Define regions to identify in masked units
    regions = ['Cortex', 'Medulla']
   
    # Define unit name mappings (abbreviated -> full name)
    unit_mappings = {
        'glom': 'Glomeruli',
        'proxtub': 'Proximal Tubule',
        'disttub': 'Distal Tubule',
        'tal': 'Thick Ascending Limb',
        'distneph': 'Distal Nephron',
        'tdl': 'Thin Descending Limb',
        'vasc': 'Vasculature'
    }


    if perform_masking:
        # Apply masking if enabled
        output_csv = os.path.join(classifier.output_base, 'masked_predictions_percentages.csv')
        output_ratio_csv = os.path.join(classifier.output_base, 'masked_ratio_means.csv')
        print("\nProcessing masks and quantifying results...")
        
        predictions_per_unit = {}
        ratios_per_unit = {}
        
        mask_type = 'Processed_Masks'
        
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

                # Plot class spectra for predictions if available
                if os.path.exists(csv_path):
                    try:
                        for class_name in visualizer.molecule_names:
                            if class_name != 'No Match':
                                visualizer.visualize_class_spectra(
                                    predictions=pd.read_csv(csv_path, header=None),
                                    img_path=img_path,
                                    img_name=f"{img_name_no_ext}_{class_name}",
                                    max_spectra=10,
                                    class_filter=class_name
                                )
                    except Exception as e:
                        print(f"Failed to plot class spectra for {img_name}: {e}")
                
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
                        stats=stats,
                        regions=regions,
                        img_name=img_name_no_ext,
                        sample_name=sample_name
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
                                stats=stats,
                                regions=regions,
                                img_name=img_name_no_ext,
                                sample_name=sample_name
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
    percentages_df = None
    ratios_df = None
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
        # Define display names for plots (optional - maps original names to display names)
        display_names = {
            # Glycerophospholipids 
            '16:0 Cardiolipin': 'CL',
            'PC Mix': 'PC 18:1',
            'PE Mix': 'PE 18:1',
            'PS Mix': 'PS 18:0',
            'PI Mix': 'PI XX:X',
            'PG Mix': 'PG XX:X',
            '18:0 LPA': 'PA(18:0/0:0)',
            '18:1 LPA': 'PA(18:1/0:0)',
            'DOPC': 'PC(18:1/18:1)',
            'DOPE': 'PE(18:1/18:1)',
            'DSPC': 'PC(18:0/18:0)',
            '16:0 CDP DG' : 'CDP-DG(16:0)',


            
            # Sterols
            'Cholesterol (ovine)': 'Chol',
            '18:1 Cholesterol ester': 'CE(18:1)',
            
            # Glycerolipids
            'TAG 16:0': 'TG(16:0)',
            'TAG 18:1': 'TG(18:1)',
            'DAG 16:0': 'DG(16:0)',
            'DAG 18:0 and 24:0 ': 'DG(18:0/24:0)',
            
            # Sphingolipids
            'Sphingosine': 'SM',

            # Ceramides
            'Cer 18:1-12:0': 'Cer(d18:1/12:0)',
            'Cer 18:1-18:0': 'Cer(d18:1/18:0)',
            'Cer 18:1-18:1': 'Cer(d18:1/18:1)',
            'Cer 18:1-22:0': 'Cer(d18:1/22:0)',
            'Cer 18:1-24:0': 'Cer(d18:1/24:0)',
            'Cer 18:1-24:1': 'Cer(d18:1/24:1)',
            'Cer 18:0-14:0': 'dhCer(d18:0/14:0)',
            'Cer 18:1-24:1(15Z)': 'dhCer(d18:0/24:1)',
            'Cer m18:1-16:0': 'doxCer(m18:1/16:0)',
            'Cer m18:1-24:1': 'doxCer(m18:1/24:1)',

            # Carnitine
            'C8 L-carnitine': 'CAR 8:0',
            'C12 Carnitine': 'CAR 12:0',
            'C16 Carnitine': 'CAR 16:0',
            'C18 Carnitine (d9-cis)': 'CAR 18:0',
            
            # Fatty Acids
            'Stearic Acid': 'SA',
            'Palmitic Acid': 'PA',
            'Docosahexaenoic Acid': 'DHA',
            'Tetracosapentaenoic Acid': 'TPA',

            # Add more mappings as needed for your specific molecules

            # Non-lipids
            "Glucosylceramide (Gaucher's Spleen)" : 'GlcCer',
            "Glucose": 'Glc',
            "Lactate": 'Lac',


            # Ratio Types
            'Lipid_to_Protein': 'Lipid/Protein',
            'Lipid_Unsaturation': 'Lipid Unsat',
            'Redox': 'ORR',
        }
        
        # Optional: Define custom colors for specific units (hex color codes)
        # These colors will be used for box plots - professional journal-quality palette
        # Adjusted Paul Tol's vibrant scheme with requested color shifts
        unit_colors = {
            'Glomeruli': '#EC1411',        # Red (shifted from orange)
            'Distal Nephron': '#CCBB22',   # Yellow
            'Distal Tubule': '#EE3377',    # Magenta
            'Proximal Tubule': '#00A43D',  # Green (shifted from teal)
            'Thick Ascending Limb': '#339FEE',  # Cyan
            'Thin Descending Limb': '#002BAA',   # Blue
            'Vasculature': '#EE7733'            # Orange (shifted from purple)
        }
        
        # Map full unit names (in data) to abbreviated display names (for plots)
        unit_display_names = {
            'Glomeruli': 'GL',
            'Proximal Tubule': 'PT',
            'Distal Tubule': 'DT',
            'Thick Ascending Limb': 'TAL',
            'Distal Nephron': 'DN',
            'Thin Descending Limb': 'TDL',
            'Vasculature': 'V'
        }
        
        # Define custom order for units in heatmaps (x-axis)
        # Modify this list to change the order units appear in heatmaps
        custom_unit_order = [
            'Glomeruli',
            'Proximal Tubule',
            'Distal Tubule',
            'Distal Nephron',
            'Thick Ascending Limb',
            'Vasculature',
            'Thin Descending Limb',
        ]

        # Create plots directory
        plots_directory = os.path.join(os.path.dirname(base_directory), 'rf_plots')
        os.makedirs(plots_directory, exist_ok=True)
        print(f"\nPlots will be saved to: {plots_directory}")
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
        # if percentages_df is not None and not percentages_df.empty:
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


            print("\n" + "="*80)
            print("GENERATING HEATMAPS")
            print("="*80)
            # Generate unit-aggregated heatmaps for percentages
            percentages_unit_heatmap_dir = os.path.join(plots_directory, 'heatmaps_unit_percentages')
            percentages_unit_heatmap_data = visualizer.generate_unit_heatmaps(
                df=percentages_df,
                value_col='Percentage',
                grouping_col='Molecule',
                output_dir=percentages_unit_heatmap_dir,
                data_type='percentage',
                show_plots=False,
                display_name_map=display_names,
                unit_display_map=unit_display_names,
                unit_order=custom_unit_order
            )
        else:
            print("\nSkipping unit comparison ANOVA for percentages: No data available")
        
        # Generate raw (non-normalized) ratio heatmap
        if ratios_df is not None and not ratios_df.empty:
            ratios_raw_heatmap_dir = os.path.join(plots_directory, 'heatmaps_raw_ratios')
            raw_ratio_heatmap_data = visualizer.generate_raw_ratio_heatmap(
                df=ratios_df,
                value_col='Mean_Ratio',
                grouping_col='Ratio_Type',
                output_dir=ratios_raw_heatmap_dir,
                show_plots=False,
                display_name_map=display_names,
                unit_display_map=unit_display_names,
                unit_order=custom_unit_order
            )
        else:
            print("\nSkipping heatmap generation for ratios: No data available")

        # Generate multi-panel box plots for selected molecules/ratios
        print("\n" + "="*80)
        print("GENERATING MULTI-PANEL BOX PLOTS")
        print("="*80)
        
        if percentages_df is not None and not percentages_df.empty:
            # Define molecules to include in multi-panel figure
            selected_molecules = ['18:1 Cholesterol ester', 'Triglyceride 16:0',
                                'Phosphatidylethanolamine Mix', 'Cardiolipin']
            
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
                unit_display_map=unit_display_names
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
                unit_display_map=unit_display_names
            )
            print(f"\nGenerated multi-panel box plots for ratios in {multi_panel_ratios_dir}")


if __name__ == "__main__":
    main()