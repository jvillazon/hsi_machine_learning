
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


def main():
    # Define processing options
    perform_RF_classification = True # Set to True to run classifier inference   

    # Define dataset parameters
    base_directory = r"D:\ADATA Backup\HuBMAP\HuBMAP Xenium\Xenium HSI\data"
    num_samp = 61
    wn_1 = 2700
    wn_2 = 3100
    ch_start = int((2800 - wn_1) / (wn_2 - wn_1) * num_samp)

    # Define model name
    chosen_model_name = 'rf_best_model'  # Update with correct model name
    model_path = f'sk_classifier/models/{chosen_model_name}.joblib'

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
    
    labeled_dataset = HSI_Labeled_Dataset(
        molecule_dataset_path='molecule_dataset/lipid_subtype_wn_61_test',
        srs_params_path='params_dataset/srs_params_61',
        num_samples_per_class=10000,
        normalize_per_molecule=False,
        compute_min_max=True,
        noise_multiplier=0.5
    )
    # Initialize Random Forest classifier
    output_base = r"D:\integrated_pipeline\HSI_data\CODEX_FTUcells_outputs\rf_best_model_outputs"
    classifier = HSI_Classifier(dataset, model_path=model_path, output_base=output_base, visualizer=visualizer, labeled_dataset=labeled_dataset)


    # Initialize variables that may be used across conditional blocks
    percentages_df = None
    ratios_df = None

    if perform_RF_classification:    
        if os.path.exists(model_path):
            # Run inference
            print("\nRunning RandomForest inference on dataset...")
            timing_stats = classifier.predict(alpha=2.77)  # Adjust alpha as needed for spectral weighting
            
            # print("\nRandomForest Inference Results:")
            # print(f"Total predictions: {len(predictions)}")
            # print(f"Number of classes: {probabilities.shape[1]}")
            
            # # Show example predictions
            # print("\nShowing sample predictions...")
            # visualizer.show_random_predictions(dataset, predictions, probabilities,
            #                                 num_images=1, spectra_per_image=3, exclude_classes=['No Match'])
        else:
            print(f"No model found at {model_path}")
            # Just show some sample spectra without predictions
            sample_indices = [100001, 200001, 300001]
            for idx in sample_indices:
                spectra, img_idx = dataset[idx]
                visualizer.visualize_spectrum(spectra, img_idx=img_idx)

    
        # Generate spectral comparison
        spectra_comparison_dir = os.path.join(os.path.dirname(base_directory), 'spectra_comparison')
        os.makedirs(spectra_comparison_dir, exist_ok=True)
        visualizer.generate_class_spectra_comparison_plots(
            dataset=dataset,
            prediction_dir=classifier.output_base,
            output_dir=spectra_comparison_dir,
            srs_params_path='params_dataset/srs_params_61.npz',
            classes_to_exclude=['No Match']
        )

    # (Optional) Define masking and quantification options
    perform_masking = True # Set to True to enable both FTU masking and ratio masking
    display_plots = True # Set to True to display statistical significance plots
   
    # Define regions to identify in masked units
    regions = ['Cortex', 'Medulla']
   
    # Define unit name mappings (abbreviated -> full name)
    # unit_mappings = {
    #     'glom': 'Glomeruli',
    #     'proxtub': 'Proximal Tubule',
    #     'disttub': 'Distal Tubule',
    #     'tal': 'Thick Ascending Limb',
    #     'distneph': 'Distal Nephron',
    #     'tdl': 'Thin Descending Limb',
    #     'vasc': 'Vasculature'
    # }

    unit_mappings = {
        "altpt": "altPT",
        "altdtl": "altDTL",
        "pt": "PT",
        "dtl": "DTL",
        "ec": "EC",
        "tal": "TAL",
        "unassigned": "Unassigned",
        "alttal": "altTAL",
        "low_quality": "Low Quality",
        "pc": "PC",
        "ic": "IC",
        "pod": "POD",
        "pec": "PEC",
        "lymphoid": "Lymphoid",
        "myeloid": "Myeloid",
        "t": "T",
        "atl": "ATL",
        "dct": "DCT",
        "altatl": "altATL",
        "vsm,p": "VSM/P",
        "fib": "FIB",
        "cnt": "CNT",
        "altcnt": "altCNT",
        "altic": "altIC",
        "mon": "MON",
        "dc": "DC",
        "inffib": "infFIB",
        "sc,neu": "SC/NEU",
        "momac": "moMAC",
        "altdct": "altDCT",
        "myof": "MYOF",
        "pvfib": "pvFIB",
        "resmac": "resMAC",
        "afib": "aFIB",
        "momac,inf": "moMAC-INF",
        "md": "MD",
        "nk": "NK",
        "n": "N",
        "mast": "MAST",
        "ren": "REN",
        "ec,gc": "EC-GC",
        "neu": "NEU",
        "altpod": "altPOD",
        "b": "B",
        "mc": "MC",
    }

    # display_units= ['POD', 'PT', 'TAL', 'PC', 'DCT']
    display_units = None
    # molecules_to_display = [
    #     '16:0 Cardiolipin',
    #     "DOPC", 
    #     "DOPE", 
    #     "PC Mix", 
    #     "PE Mix", 
    #     'Sphingosine', 
    #     'Cholesterol (ovine)', 
    #     '18:1 Cholesterol Ester',
    #     'TAG 16:0',
    #     'TAG Mix',

    #     'Cer 18:1-18:0',
    #     'Cer 18:1-18:1',
    #     'Cer 18:1-24:0',
    #     'Cer 18:1-24:1',

    #     # Carnitine
    #     'C8 L-Carnitine',

        
    #     # Fatty Acids
    #     'Stearic Acid',
    #     'Palmitic Acid',
    #     'Docosahexaenoic Acid',
    #     'Tetracosapentaenoic Acid',
          
    #       ]  # e.g. ['CH2 sym stretch', 'CH3 sym stretch']; None falls back to top_n]
    molecules_to_display = None
    ratios_to_display = None     # e.g. ['CH2/CH3']; None shows all ratio types

    output_csv = os.path.join(classifier.output_base, 'masked_predictions_percentages.csv')
    output_ratio_csv = os.path.join(classifier.output_base, 'masked_ratio_means.csv')

    mask_type = 'CODEX_FTUcells_masks'
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
                        prefix=mask_prefix
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

    # Display statistical significance plots if enabled
    if display_plots:
        # Define display names for plots (optional - maps original names to display names)
        display_names = {
            # Glycerophospholipids 
            '16:0 Cardiolipin': 'CL(16:0)',
            'PC Mix': 'PC Mix',
            'PE Mix': 'PE Mix',
            'PS Mix': 'PS Mix',
            'PI Mix': 'PI Mix',
            'PG Mix': 'PG Mix',
            '18:0 Lyso PA': 'PA(18:0/0:0)',
            '18:1 LPA': 'PA(18:1/0:0)',
            'DOPC': 'PC(18:1/18:1)',
            'DOPE': 'PE(18:1/18:1)',
            'DSPC': 'PC(18:0/18:0)',
            '16:0 CDP DG' : 'CDP-DG(16:0)',


            
            # Sterols
            'Cholesterol (ovine)': 'Chol',
            '18:1 Cholesterol Ester': 'CE(18:1)',
            
            # Glycerolipids
            'TAG Mix': 'TG Mix',
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
            "Glucosylceramide (Gaucher's Spleen)": 'GlcCer',

            # Carnitine
            'C8 L-Carnitine': 'CAR(8:0)',
            'C12 Carnitine': 'CAR(12:0)',
            'C16 Carnitine': 'CAR(16:0)',
            'C18 Carnitine (d9-cis)': 'CAR(18:0)',
            
            # Fatty Acids
            'Stearic Acid': 'SA',
            'Palmitic Acid': 'PA',
            'Docosahexaenoic Acid': 'DHA',
            'Tetracosapentaenoic Acid': 'TPA',

            # Add more mappings as needed for your specific molecules

            # Non-lipids
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
        # unit_colors = {
        #     'Glomeruli': '#EC1411',        # Red (shifted from orange)
        #     'Distal Nephron': '#CCBB22',   # Yellow
        #     'Distal Tubule': '#EE3377',    # Magenta
        #     'Proximal Tubule': '#00A43D',  # Green (shifted from teal)
        #     'Thick Ascending Limb': '#339FEE',  # Cyan
        #     'Thin Descending Limb': '#002BAA',   # Blue
        #     'Vasculature': '#EE7733'            # Orange (shifted from purple)
        # }



        unit_colors = {
            "Cluster 1":	"#e8260c",
            "Cluster 2":	"#0a8416",
            "Cluster 3":	"#10c922",
            "Cluster 4":	"#35ee48",
            "Cluster 5":	"#7af487",
            "Cluster 6":	"#8b5403",
            "Cluster 7":	"#fbc573",
            "Cluster 8":	"#870763",
            "Cluster 9":	"#f20cb1",
            "Cluster 10":	"#f777d3",
            "Cluster 11":	"#ebd409",
            "Cluster 12":	"#073a87",
            "Cluster 13":	"#094cb2",
            "Cluster 14":	"#0b5fdc",
            "Cluster 15":	"#2275f3",
            "Cluster 16":	"#4c90f5",
            "Cluster 17":	"#77aaf7",
            "Cluster 18":	"#088686",
            "Cluster 19":	"#0dcccc",
            "Cluster 20":	"#32f1f1",
            "Cluster 21":	"#78f6f6",
            "Cluster 22":	"#5c0b8d",
            "Cluster 23":	"#8210c9",
            "Cluster 24":	"#a42ced",
            "Cluster 25":	"#be68f2",
            "Cluster 26":	"#7a711e",
        }

        # unit_colors = {
        #     "POD": "#e8260c",
        #     "PT": "#0a8416",
        #     "TAL": "#EE2DB7",
        #     "DCT": "#ebd409",
        #     "PC": "#245bad",
        # }
        
        # Map full unit names (in data) to abbreviated display names (for plots)
        unit_display_names = {
            'Glomeruli': 'Glomeruli',
            'Proximal Tubule': 'Proximal Tubule',
            'Distal Tubule': 'Distal Tubule',
            'Thick Ascending Limb': 'Thick Ascending Limb',
            'Distal Nephron': 'Collecting Duct',
            'Thin Descending Limb': 'Thin Descending Limb',
            'Vasculature': 'Vasculature'
        }

        unit_display_names = {
            'Cluster 1': "Glomeruli | Podo, CD31",
            'Cluster 2':	"PT | LRP2",
            'Cluster 3':	"PT | LRP2, AQP1, CD90",
            'Cluster 4':	"PT | LRP2, CD90",
            'Cluster 5':	"PT | LRP2, PROM1/VCAM1+",
            'Cluster 6':	"TDL | AQP1, Cyto8",
            'Cluster 7':	"TDL | AQP1, Cyto8",
            'Cluster 8':	"TAL | NaK/UMOD+",
            'Cluster 9':	"TAL | UMOD+",
            'Cluster 10':	"TAL | UMOD+",
            'Cluster 11':	"DCT | SLC12a3",
            'Cluster 12':	"CD | Cyto8",
            'Cluster 13':	"CD | Cyto8+",
            'Cluster 14':	"CD | Cyto8+",
            'Cluster 15':	"CD | Cyto8, Bcat",
            'Cluster 16':	"CD | Cyto8, Bcat, Ecad",
            'Cluster 17':	"CD | Cyto8, Ecad",
            'Cluster 18':	"Endothelium | CD31, Podocalyxin",
            'Cluster 19':	"Endothelium | CD31, Podocalyxin ",
            'Cluster 20':	"Pericytes | CD90+, CD31 low",
            'Cluster 21':	"VSMCs | aSMA",
            'Cluster 22':	"Immune  | CD20 +",
            'Cluster 23':	"Immune - MP | CD45, CD206",
            'Cluster 24':	"Immune - T | CD3, CD4",
            'Cluster 25':	"Immune - T | CD3, CD8",
            'Cluster 26':	"Interstitial cells | IGFBP7",
        }

        # unit_display_names = {
        #     "POD": "POD",
        #     "PT": "PT",
        #     "TAL": "TAL",
        #     "DCT": "DCT",
        #     "PC": "CD",
        # }

        
        # Define custom order for units in heatmaps (x-axis)
        # Modify this list to change the order units appear in heatmaps
        bubble_unit_order = [
            'Vasculature',
            'Thin Descending Limb',
            'Thick Ascending Limb',
            'Proximal Tubule',
            'Glomeruli',
            'Distal Tubule',
            'Distal Nephron',
        ]


        bubble_unit_order = [
            'Cluster 1',
            'Cluster 2', 
            'Cluster 3',
            'Cluster 4',
            'Cluster 5',
            'Cluster 6',
            'Cluster 7',
            'Cluster 8',
            'Cluster 9',
            'Cluster 10',
            'Cluster 11',
            'Cluster 12',
            'Cluster 13',
            'Cluster 14',
            'Cluster 15',
            'Cluster 16',
            'Cluster 17',
            'Cluster 18',
            'Cluster 19',
            'Cluster 20',
            'Cluster 21',
            'Cluster 22',
            'Cluster 23',
            'Cluster 24',
            'Cluster 25',
            'Cluster 26'
        ]
        heatmap_unit_order = bubble_unit_order.reverse() # Use same order for heatmaps and bubble plots

        # Create plots directory
        plots_directory = os.path.join(os.path.dirname(base_directory), f'rf_{mask_type}_plots')
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


            print("\n" + "="*80)
            print("GENERATING HEATMAPS")
            print("="*80)
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

            print("\n" + "="*80)
            print("GENERATING BUBBLE PLOTS")
            print("="*80)
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
                cmap = 'RdBu_r'
            )

            # ----------------------------------------------------------------
            # Per-sample heatmaps and bubble charts
            # Groups rows by the Sample_Name column (e.g. 'aki', 'ckd', 'ref')
            # and generates a top-10 heatmap + bubble chart for each group.
            # ----------------------------------------------------------------
            print("\n" + "="*80)
            print("GENERATING PER-SAMPLE HEATMAPS AND BUBBLE CHARTS")
            print("="*80)

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
        print("\n" + "="*80)
        print("GENERATING MULTI-PANEL BOX PLOTS")
        print("="*80)
        
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