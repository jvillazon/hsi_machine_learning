import matplotlib
matplotlib.use('Agg')
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import tifffile
from glob import glob
import json
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime

import sys 
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.hsi_unlabeled_dataset import HSI_Unlabeled_Dataset
from core.hsi_visualizer import HSI_Visualizer
from core.hsi_sk_classifier import HSI_Classifier
from core.hsi_labeled_dataset import HSI_Labeled_Dataset
from config_loader import load_config, apply_config, list_available_profiles, print_config_info
from umap_analysis import generate_ion_umaps

# CONFIGURATION PROFILE SELECTION
PROFILE = 'kidney_clusters-regions'  # Default profile

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

# Extract individual plot settings early
create_individual_plots = individual_plot_config.get('create_individual_plots', True)
individual_molecules = individual_plot_config.get('selected_molecules', "All")
individual_ratios = individual_plot_config.get('selected_ratios', "All")
consolidate_Group_IDs = individual_plot_config.get('consolidate_Group_IDs', True)

# Extract granularity and sample analysis settings
granularity_levels = sample_analysis_config.get('granularity_levels', [])
selected_values = sample_analysis_config.get('selected_values', None)
filter_col = sample_analysis_config.get('filter_col', 'Source_ID') 

# Derive bubble_unit_order as reverse of heatmap order
bubble_unit_order = heatmap_unit_order[::-1]

print("\n✓ Configuration loaded successfully")

workspace_dir = str(Path.cwd())
base_directory = r"C:\Users\jvill\OneDrive - UC San Diego\integrated_pipeline\WSI_data\WSI_data\data"

# SRS parameters
wn_1, wn_2, num_samp = 2700, 3100, 61
ch_start = int(((2800 - wn_1)/(wn_2 - wn_1)) * num_samp)

chosen_model_name = 'rf_n150_md40_mss2_msl9_a9_best_model_platt'
model_path = os.path.join(workspace_dir, 'sk_classifier', 'models', f'{chosen_model_name}.joblib')


# Create visualizer
visualizer = HSI_Visualizer(
    mol_path=os.path.join(workspace_dir, 'molecule_dataset', 'lipid_subtype_wn_61_test.npz'),
    wavenumber_start=wn_1, wavenumber_end=wn_2, num_samples=num_samp
)
plots_directory = os.path.join(os.path.dirname(base_directory), f'plots_{mask_type}')


# Create dataset
dataset = HSI_Unlabeled_Dataset(
    base_directory, ch_start, image_normalization=True, min_max_normalization=False,
    num_samples=num_samp, wavenumber_start=wn_1, wavenumber_end=wn_2, compute_stats=perform_RF_classification
)
labeled_dataset = HSI_Labeled_Dataset(
    molecule_dataset_path=os.path.join(workspace_dir, 'molecule_dataset', 'lipid_subtype_wn_61_test'),
    srs_params_path=os.path.join(workspace_dir, 'params_dataset', 'srs_params_61'),
    num_samples_per_class=10000, compute_min_max=True
)

# Initialize classifier
output_base = os.path.join(os.path.dirname(base_directory), f"{chosen_model_name}_outputs")
classifier = HSI_Classifier(dataset, model_path=model_path, output_base=output_base, 
                    visualizer=visualizer, labeled_dataset=labeled_dataset)





percentages_df, ratios_df = None, None
print("✓ Dataset and classifier initialized successfully")

if perform_RF_classification:
    if os.path.exists(model_path):


        print("Running RandomForest inference...")
        classifier.predict(alpha=10, generate_shap=True)
    else:
        print(f"No model found at {model_path}")
else:
    print("RF classification disabled")

masked_output_dir = os.path.join(os.path.dirname(base_directory), f"{mask_type}_output_csv-test")
os.makedirs(masked_output_dir, exist_ok=True)
output_csv = os.path.join(masked_output_dir, 'masked_predictions_percentages.csv')
output_ratio_csv = os.path.join(masked_output_dir, 'masked_ratio_means.csv')
output_ratio_wide_csv = os.path.join(masked_output_dir, 'masked_ratio_means-wide.csv')

if perform_masking:
    mask_prefix = mask_type.split("_")[0]  
    print("Processing masks and quantifying results...")
    
    predictions_per_unit, ratios_per_unit = {}, {}
    
    for img_idx, img_path in enumerate(dataset.img_list):
        img_name_no_ext = os.path.splitext(os.path.basename(img_path))[0]
        mask_folder = os.path.join(os.path.dirname(base_directory), mask_type, img_name_no_ext)
        csv_path = os.path.join(classifier.output_base, img_name_no_ext, f"{img_name_no_ext}_predictions.csv")
        ratio_folder = os.path.join(os.path.dirname(base_directory), 'Ratio', img_name_no_ext)
        
        if not os.path.exists(mask_folder): continue

        if os.path.exists(csv_path):
            print(f"Processing prediction CSV for image {img_name_no_ext}")
            res = visualizer.apply_rf_masking(
                prediction_csv_path=csv_path, 
                mask_list_path=mask_folder, 
                subgroups=subgroups, 
                img_name=img_name_no_ext, 
                prefix=mask_prefix,
                min_instance_size=100,
                group_subclasses=True)
            for u, e in res.items(): predictions_per_unit.setdefault(u, []).extend(e)
        
        if os.path.exists(ratio_folder):
            file_exts = ['*.tif', "*.tiff"]
            for ext in file_exts:
                for r_tif in glob(os.path.join(ratio_folder, ext)):
                    print(f"Processing ratio image: {r_tif}")
                    if 'Collagen' in r_tif:
                        dilations = 10
                    else:
                        dilations = 2
                    res = visualizer.apply_rf_masking(ratio_tiff_path=r_tif, mask_list_path=mask_folder, subgroups=subgroups, img_name=img_name_no_ext, prefix=mask_prefix, min_instance_size=100,dilations=dilations)
                    for u, e in res.items(): ratios_per_unit.setdefault(u, []).extend(e)
    
    index_cols = ['Image_Name', 'Unit', 'Instance_Label', 'Source_ID']
    if predictions_per_unit:
        percentages_df = visualizer.quantify_unit_class_percentages_nested(predictions_per_unit, unit_mappings=unit_mappings)
        percentages_df.to_csv(output_csv, index=False)


        # 1. Pivot the dataframe
        if 'Group_ID' in percentages_df.columns:
            index_cols.append('Group_ID')

        percentages_wide_df = percentages_df.pivot(index=index_cols, columns='Molecule', values='Percentage').reset_index()
        percentages_wide_df.to_csv(output_ratio_wide_csv)
    
    if ratios_per_unit:
        ratios_df = visualizer.quantify_unit_ratio_means_nested(ratios_per_unit, unit_mappings=unit_mappings)
        ratios_df.to_csv(output_ratio_csv, index=False)

        group_cols = ['Image_Name', 'Unit', 'Instance_Label', 'Group_ID', 'Source_ID','centroid_x', 'centroid_y']
        has_lipid_protein = ratios_df[ratios_df['Ratio_Type'] == 'Lipid_to_Protein'][group_cols].drop_duplicates()
        has_unsaturation = ratios_df[ratios_df['Ratio_Type'] == 'Lipid_Unsaturation'][group_cols].drop_duplicates()
        valid_instances = pd.merge(has_lipid_protein, has_unsaturation, on=group_cols)
        df = pd.merge(ratios_df, valid_instances, on=group_cols, how='inner')

        # 1. Pivot the dataframe
        if 'Group_ID' in df.columns:
            index_cols.append('Group_ID')

        ratios_wide_df = df.pivot(index=index_cols, columns='Ratio_Type', values='Mean_Ratio').reset_index()
        ratios_wide_df.to_csv(output_ratio_wide_csv)
        
        if percentages_df is not None:
            wide_df = pd.merge(percentages_wide_df, ratios_wide_df, on=index_cols)
        else:
            wide_df = ratios_wide_df.copy()



else:
    print("Masking disabled")


# masked_output_dir = os.path.join(os.path.dirname(base_directory), f"{mask_type}_output_csv")
# os.makedirs(masked_output_dir, exist_ok=True)
# output_csv = os.path.join(masked_output_dir, 'masked_predictions_percentages.csv')
# output_ratio_csv = os.path.join(masked_output_dir, 'masked_ratio_means.csv')
# output_ratio_wide_csv = os.path.join(masked_output_dir, 'masked_ratio_means-wide.csv')


if display_units_boxplots:
    display_units_boxplots_type = f"{display_units_boxplots[0]}-{display_units_boxplots[-1]}"


if display_units_boxplots:
    print("Currently selected boxplot units:")
    for i, unit in enumerate(display_units_boxplots):
        
        dname = unit_display_names.get(unit, unit)
        print(f"  {i+1}. {unit}: {dname}")
else:
    print("No specific units designated - will default to all available units.")

if display_plots:
    if percentages_df is None and os.path.exists(output_csv): 
        percentages_df = pd.read_csv(output_csv)
    if ratios_df is None and os.path.exists(output_ratio_csv): 
        ratios_df = pd.read_csv(output_ratio_csv)
    
    index_cols = ['Image_Name', 'Unit', 'Instance_Label', 'Source_ID', 'centroid_x', 'centroid_y']
    
    if percentages_df is not None:

        # 1. Pivot the dataframe
        if 'Group_ID' in percentages_df.columns:
            if isinstance(subgroups, dict):
                percentages_df['Group_ID'] = percentages_df['Group_ID'].replace(subgroups)
            index_cols.append('Group_ID')
        percentages_wide_df = percentages_df.pivot(index=index_cols, columns='Molecule', values='Percentage').reset_index()
        # percentages_wide_df.to_csv(output_ratio_wide_csv)

    if ratios_df is not None:
        if 'Group_ID' in ratios_df.columns:
            if isinstance(subgroups, dict):
                ratios_df['Group_ID'] = ratios_df['Group_ID'].replace(subgroups)
        if 'Group_ID' in ratios_df.columns and 'Group_ID' not in index_cols:

            index_cols.append('Group_ID')
        
        ratios_wide_df = ratios_df.pivot(index=index_cols, columns='Ratio_Type', values='Mean_Ratio').reset_index()
        ratios_wide_df.to_csv(output_ratio_wide_csv, index=False)

        if percentages_df is not None:
            wide_df = pd.merge(percentages_wide_df, ratios_wide_df, on=index_cols)
        else:
            wide_df = ratios_wide_df.copy()
else:  
    # Apply global filters
    if selected_values is not None and filter_col:
        for df in [percentages_df, ratios_df]:
            if df is not None and filter_col in df.columns:
                df = df[df[filter_col].isin(selected_values)].copy()






if 'Unnamed: 0' in wide_df.columns:
    wide_df.drop(columns="Unnamed: 0")
# Filter by size
if 'Group_ID' in wide_df.columns and wide_df['Group_ID'].notna().any():
    unit_sizes = wide_df.groupby('Unit').size()
    unit_group_sizes = wide_df.groupby(['Unit', 'Group_ID']).size()

    df = wide_df.copy()
    df = df.merge(unit_group_sizes.rename('unit_group_size'), on=['Unit', 'Group_ID'])
    df = df.merge(unit_sizes.rename('unit_size'), on='Unit')
    if "clusters" in mask_type.lower():
        df_filtered = df[df['unit_group_size'] / df['unit_size'] >= 0.01] # Keep groups that represent at least 5% of the unit's instances
    else:
        df_filtered = df

    df = df_filtered.copy()
    has_groups = True
else:
    has_groups = False
    df = wide_df.copy()

df_dict = {}
# Separate by source
for source in df_filtered['Source_ID'].unique():
    if 'Unnamed: 0' in wide_df.columns:
        df_filtered.drop(columns="Unnamed: 0", inplace=True)
    df_dict[source] = df_filtered[df_filtered['Source_ID']==source]
    output_path = os.path.join(masked_output_dir, f'{source}_filtered-ratios.csv')
    df_dict[source].to_csv(output_path, index=False)
df = df_dict['S21']



# df = pd.read_csv(os.path.join(masked_output_dir, 'S21_filtered-ratios.csv'))
# print(df.head())



plot_type = 'violin'


if len(display_units_boxplots) != len(df['Unit'].unique()):
    name_change = f"{display_units_boxplots[0]}_{display_units_boxplots[-1]}"
else:
    name_change = "all_units"

values_to_plot = []
if ratios_df is not None and 'Ratio_Type' in ratios_df.columns:
    values_to_plot.append('Ratio_Type')
if percentages_df is not None and 'Molecule' in percentages_df.columns: 
    values_to_plot.append('Molecule')

# if "Group_ID" in df.columns:
#     df = df[df['Group_ID'].isin(subgroups)] if subgroups and has_groups else df

for value in values_to_plot:
    if value=="Ratio_Type":
        ratio_types = list(ratios_df['Ratio_Type'].unique())
        items_list = ratio_types
        data_type = 'Mean_Ratio'
        tmp_df = df[df['Source_ID']=='S21'].copy()
    elif value=="Molecule":
        molecules = list(percentages_df['Molecule'].unique())
        items_list = molecules
        data_type = 'Percentage'
        tmp_df = df.copy()
    
#     inter_unit_dir = os.path.join(plots_directory, f'{name_change}', f'{data_type}_inter-unit_{plot_type}-plots')
#     visualizer.create_individual_barplots(
#         df=tmp_df[tmp_df['Group_ID'].isin(subgroups)] if subgroups and has_groups else tmp_df,
#         value_col=None,        # Not needed for wide format
#         grouping_col=None,     # Signal that we are using Wide format
#         items_list=items_list, # The column names
#         output_dir=inter_unit_dir,
#         data_type=data_type,
#         units_to_display=display_units_boxplots,
#         consolidate_Group_IDs=True,
#         plot_type=plot_type,
#         sig_style='stars_only' if len(display_units_boxplots) > 6 else 'brackets',
#         show_plots=False,
#         display_name_map=display_names,
#         unit_color_map=unit_colors,
#         unit_display_map=unit_display_names,
#         subgroup_list = subgroups,
#         remove_outliers=True,
#         show_scatter=True,
#         show_empty_subgroups=True,
        # orientation='horizontal'
    # )

    # if subgroups:
    #     intra_unit_dir = os.path.join(plots_directory, f'{name_change}', f'{data_type}_intra-unit_{plot_type}-plots')
    #     visualizer.create_individual_barplots(
    #         df=tmp_df[tmp_df['Group_ID'].isin(subgroups)] if subgroups and has_groups else tmp_df,
    #         value_col=None,        # Not needed for wide format
    #         grouping_col=None,     # Signal that we are using Wide format
    #         items_list=items_list, # The column names
    #         output_dir=intra_unit_dir,
    #         data_type=data_type,
    #         units_to_display=display_units_boxplots,
    #         consolidate_Group_IDs=False,
    #         plot_type=plot_type,
    #         sig_style='stars_only' if len(display_units_boxplots) > 6  else 'brackets',
    #         show_plots=False,
    #         display_name_map=display_names,
    #         unit_color_map=unit_colors,
    #         unit_display_map=unit_display_names,
    #         remove_outliers=True,
    #         show_empty_subgroups=True,
    #         subgroup_list=subgroups,
    #         show_scatter=False,
    #         # orientation='horizontal'
    #     )
    #     for group in tqdm(subgroups if subgroups else df['Group_ID'].unique()):
    #         inter_unit_dir = os.path.join(plots_directory, f'{name_change}', f'{data_type}_inter-unit_{group}_{plot_type}-plots')
    #         tmp_group_df = tmp_df[tmp_df['Group_ID']==group]
    #         visualizer.create_individual_barplots(
    #             df=tmp_group_df,
    #             value_col=None,        # Not needed for wide format
    #             grouping_col=None,     # Signal that we are using Wide format
    #             items_list=items_list, # The column names
    #             output_dir=inter_unit_dir,
    #             data_type=data_type,
    #             units_to_display=display_units_boxplots,
    #             consolidate_Group_IDs=True,
    #             plot_type=plot_type,
    #             sig_style='stars_only' if len(display_units_boxplots) > 6  else 'brackets',
    #             show_plots=False,
    #             display_name_map=display_names,
    #             unit_color_map=unit_colors,
    #             subgroup_list=[group],
    #             unit_display_map=unit_display_names,
    #             remove_outliers=True,
    #             show_scatter=False,
    #             show_empty_subgroups=True,
    #             # orientation='horizontal'
    #         )



# output_dir = os.path.join(plots_directory, 'regions_gradient_heatmaps')

# visualizer.plot_subgroup_gradient_heatmap_wide(df, ratio_types , subgroups.values(),
#                                             output_dir=output_dir, display_name_map=display_names,
#                                             unit_display_map=unit_display_names, unit_order=display_units_boxplots,
#                                             units_to_display=display_units_boxplots, cmap='RdBu_r',
#                                             min_size_to_plot=5, remove_outliers=True,
#                                             show_values=False, 
#                                             cmap_scale={
#                                                 # 'Lipid_to_Protein': [0.2975, 0.3175],
#                                                 # 'Lipid_Unsaturation' : [0.425, 0.465],
#                                                 # 'Redox' : [0.12, 0.24]
#                                             },
#                                             cbar_label='Mean Value', display_significance=True
#                                             )
# visualizer.plot_subgroup_gradient_heatmap_wide_transposed(tmp_df[tmp_df['Group_ID'].isin(subgroups.values())] if subgroups and has_groups else tmp_df, ratio_types , subgroups.values(),  
#                                             output_dir=output_dir, display_name_map=display_names,unit_order=display_units_boxplots,
#                                             unit_display_map=unit_display_names, units_to_display=display_units_boxplots, cmap='RdBu_r',
#                                             show_values=False, cbar_label='Mean Value', remove_outliers=False)



# plot_type = 'violin'
# unit_group = [
#     "Cluster2",
#     "Cluster4",
#     "Cluster5"
# ]
# ratio_types = list(ratios_df['Ratio_Type'].unique())

# intra_unit_dir = os.path.join(plots_directory, f'intra-unit_{plot_type}-plots', f"{unit_group[0]}-{unit_group[-1]}")
# os.makedirs(intra_unit_dir, exist_ok=True)
# visualizer.create_individual_barplots(
#     df=df,
#     value_col=None,        # Not needed for wide format
#     grouping_col=None,     # Signal that we are using Wide format
#     items_list=ratio_types, # The column names
#     output_dir=intra_unit_dir,
#     show_empty_subgroups=True,
#     data_type='Mean_Ratio',
#     units_to_display=unit_group,
#     consolidate_Group_IDs=False,
#     plot_type=plot_type,
#     sig_style='stars_only',
#     show_plots=False,
#     display_name_map=display_names,
#     unit_color_map=unit_colors,
#     unit_display_map=unit_display_names,
#     remove_outliers=True,
#     subgroup_list=subgroups,
#     show_scatter=False,
# )


# unit_group = [
#     "Cluster6",
#     "Cluster7",
# ]


# intra_unit_dir = os.path.join(plots_directory, f'intra-unit_{plot_type}-plots', f"{unit_group[0]}-{unit_group[-1]}")
# os.makedirs(intra_unit_dir, exist_ok=True)
# visualizer.create_individual_barplots(
#     df=df,
#     value_col=None,        # Not needed for wide format
#     grouping_col=None,     # Signal that we are using Wide format
#     items_list=ratio_types, # The column names
#     output_dir=intra_unit_dir,
#     show_empty_subgroups=True,
#     data_type='Mean_Ratio',
#     units_to_display=unit_group,
#     consolidate_Group_IDs=False,
#     plot_type=plot_type,
#     sig_style='stars_only',
#     show_plots=False,
#     display_name_map=display_names,
#     unit_color_map=unit_colors,
#     unit_display_map=unit_display_names,
#     remove_outliers=True,
#     subgroup_list=subgroups,
#     show_scatter=False,
# )


# unit_group = [
#     "Cluster8",
#     "Cluster9",
# ]


# intra_unit_dir = os.path.join(plots_directory, f'intra-unit_{plot_type}-plots', f"{unit_group[0]}-{unit_group[-1]}")
# os.makedirs(intra_unit_dir, exist_ok=True)
# visualizer.create_individual_barplots(
#     df=df,
#     value_col=None,        # Not needed for wide format
#     grouping_col=None,     # Signal that we are using Wide format
#     items_list=ratio_types, # The column names
#     output_dir=intra_unit_dir,
#     show_empty_subgroups=True,
#     data_type='Mean_Ratio',
#     units_to_display=unit_group,
#     consolidate_Group_IDs=False,
#     plot_type=plot_type,
#     sig_style='stars_only',
#     show_plots=False,
#     display_name_map=display_names,
#     unit_color_map=unit_colors,
#     unit_display_map=unit_display_names,
#     remove_outliers=True,
#     subgroup_list=subgroups,
#     show_scatter=False,
# )


# unit_group = [
#     "Cluster13",
#     "Cluster15",
#     "Cluster16",
#     "Cluster17",
# ]


# intra_unit_dir = os.path.join(plots_directory, f'intra-unit_{plot_type}-plots', f"{unit_group[0]}-{unit_group[-1]}")
# os.makedirs(intra_unit_dir, exist_ok=True)
# visualizer.create_individual_barplots(
#     df=df,
#     value_col=None,        # Not needed for wide format
#     grouping_col=None,     # Signal that we are using Wide format
#     items_list=ratio_types, # The column names
#     output_dir=intra_unit_dir,
#     show_empty_subgroups=True,
#     data_type='Mean_Ratio',
#     units_to_display=unit_group,
#     consolidate_Group_IDs=False,
#     plot_type=plot_type,
#     sig_style='stars_only',
#     show_plots=False,
#     display_name_map=display_names,
#     unit_color_map=unit_colors,
#     unit_display_map=unit_display_names,
#     remove_outliers=True,
#     subgroup_list=subgroups,
#     show_scatter=False,
# )


# unit_group = [
#     "Cluster18",
#     "Cluster19",
#     "Cluster20",
#     "Cluster21",
# ]


# intra_unit_dir = os.path.join(plots_directory, f'intra-unit_{plot_type}-plots', f"{unit_group[0]}-{unit_group[-1]}")
# os.makedirs(intra_unit_dir, exist_ok=True)
# visualizer.create_individual_barplots(
#     df=df,
#     value_col=None,        # Not needed for wide format
#     grouping_col=None,     # Signal that we are using Wide format
#     items_list=ratio_types, # The column names
#     output_dir=intra_unit_dir,
#     show_empty_subgroups=True,
#     data_type='Mean_Ratio',
#     units_to_display=unit_group,
#     consolidate_Group_IDs=False,
#     plot_type=plot_type,
#     sig_style='stars_only',
#     show_plots=False,
#     display_name_map=display_names,
#     unit_color_map=unit_colors,
#     unit_display_map=unit_display_names,
#     remove_outliers=True,
#     subgroup_list=subgroups,
#     show_scatter=False,
# )





# Define your filters (None = use all)
# display_units = [
#     "Glomeruli",
#     "Proximal Tubule",
#     "Thin Descending Limb",
#     "Thick Ascending Limb",
#     "Distal Tubule",
#     "Distal Nephron",
#     "Vasculature"
# ]
display_units = display_units_boxplots if display_units_boxplots else None
selected_units = display_units_boxplots

selected_groups = ['IS','OS', 'CM', 'IC', 'OC']
selected_ratios = list(ratios_df['Ratio_Type'].unique())


umap_df = df.drop(['Collagen', "Collagen_Coherence", "Collagen_Thickness"], axis=1)
# Define output directory
umap_output_dir = os.path.join(plots_directory, 'S21-regions_umaps')

# Execute the filtered UMAP analysis
generate_ion_umaps(
    df_wide=umap_df, 
    unit_colors=unit_colors, 
    unit_names=unit_display_names, 
    plots_directory=plots_directory,
    output_dir=umap_output_dir,
    selected_units=selected_units,   
    selected_groups=selected_groups, 
    selected_ratios=selected_ratios,
    display_units=display_units  
)

# unit_group = [
#     "Cluster2",
#     "Cluster4",
#     "Cluster5"
# ]

# for group in df['Group_ID'].unique():
#     inter_unit_dir = os.path.join(plots_directory, f'inter-unit_{group}_{plot_type}-plots, f"{unit_group[0]}-{unit_group[-1]}")')
#     os.makedirs(inter_unit_dir, exist_ok=True)
#     s21_group_df = df[df['Group_ID']==group]
#     visualizer.create_individual_barplots(
#         df=s21_group_df,
#         value_col=None,        # Not needed for wide format
#         grouping_col=None,     # Signal that we are using Wide format
#         items_list=ratio_types, # The column names
#         output_dir=inter_unit_dir,
#         data_type='Mean_Ratio',
#         units_to_display=display_units_boxplots,
#         consolidate_Group_IDs=True,
#         plot_type=plot_type,
#         sig_style='stars_only',
#         show_plots=False,
#         display_name_map=display_names,
#         unit_color_map=unit_colors,
#         subgroup_list=[group],
#         unit_display_map=unit_display_names,
#     )
# intra_unit_dir = os.path.join(plots_directory, f'intra-unit_{plot_type}-plots', f"{unit_group[0]}-{unit_group[-1]}")
# os.makedirs(intra_unit_dir, exist_ok=True)
# visualizer.create_individual_barplots(
#     df=df,
#     value_col=None,        # Not needed for wide format
#     grouping_col=None,     # Signal that we are using Wide format
#     items_list=ratio_types, # The column names
#     output_dir=intra_unit_dir,
#     data_type='Mean_Ratio',
#     units_to_display=display_units_boxplots,
#     consolidate_Group_IDs=False,
#     plot_type=plot_type,
#     sig_style='stars_only',
#     show_plots=False,
#     display_name_map=display_names,
#     unit_color_map=unit_colors,
#     unit_display_map=unit_display_names,
#     remove_outliers=False

# )


# # subgroups = list(df['Group_ID'].unique())
# subgroups = [
#     'CD3,CD4',
#     'CD3,CD8',
#     'CD45,CD206'
# ]

# N = len(subgroups)
# # df = df[(df['Group_ID']!="M2")]
# group_rank = {group: idx for idx, group in enumerate(subgroups)}
# # Ensure you only process rows that belong to your known subgroups
# valid_mask = df['Group_ID'].isin(subgroups)
# # 2. Scale coordinates to [0, 1] strictly within each Group per Image
# df.loc[valid_mask, 'x_in_group'] = df[valid_mask].groupby(['Image_Name', 'Group_ID'])['centroid_x'].transform(
#     lambda x: (x - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) != 0 else 0.5
# )
# df.loc[valid_mask, 'y_in_group'] = df[valid_mask].groupby(['Image_Name', 'Group_ID'])['centroid_y'].transform(
#     lambda x: (x - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) != 0 else 0.5
# )
# # 3. Map Group_ID to its relative index (e.g., 'P' -> 0, 'C1' -> 5)
# df.loc[valid_mask, 'group_rank'] = df.loc[valid_mask, 'Group_ID'].map(group_rank)
# # 4. Project into sequential bins
# # Global Scaled Coordinate = (Base Group Position) + (Within-Group Scaled Offset)
# df.loc[valid_mask, 'centroid_x_global'] = (df.loc[valid_mask, 'group_rank'] / N) + (df.loc[valid_mask, 'x_in_group'] / N)
# df.loc[valid_mask, 'centroid_y_global'] = (df.loc[valid_mask, 'group_rank'] / N) + (df.loc[valid_mask, 'y_in_group'] / N)


# # 2. Choose the column you want to use to color the points (e.g., 'Image_Name' or 'Group_ID')
# color_by_col = 'Group_ID' 

# x_label = 'Collagen'
# y_label = 'Redox'

# stats_per_group = True
# df = df[df['Group_ID'].isin(subgroups)]
# # Set seaborn style for nicer looking plots (optional but recommended!)
# # sns.set_theme(style="whitegrid")
# unique_groups = df[color_by_col].dropna().unique()
# palette = sns.color_palette("tab10", len(unique_groups))
# color_map = dict(zip(unique_groups, palette))


# # Scale centroid_x
# output_dir = os.path.join(plots_directory, f"{x_label}_by_{y_label}-scatter")
# os.makedirs(output_dir, exist_ok=True)

# for unit in df['Unit'].unique():
#     print(f"\nUnit: {unit}")
#     subset = df[df['Unit'] == unit].dropna(subset=['Lipid_Unsaturation', 'Lipid_to_Protein', 'Redox', 'Collagen'])  # Ensure no NaNs in key columns
    
#     min_group_size = 1  # Adjust threshold here (e.g., 3, 5, or 10)
#     group_counts = subset[color_by_col].value_counts()
#     print(group_counts)
#     valid_groups = group_counts[group_counts >= min_group_size].index
    
#     subset = subset[subset[color_by_col].isin(valid_groups)]


#     if not subset.empty:
#         groups = subset.groupby(color_by_col, observed=True)[y_label]
#         q1 = groups.transform(lambda x: x.quantile(0.25))
#         q3 = groups.transform(lambda x: x.quantile(0.75))
#         iqr = q3 - q1
        
#         lower_bound = q1 - 1.5 * iqr
#         upper_bound = q3 + 1.5 * iqr
        
#         # Restrict to clean dataset
#         mask = (subset[y_label] >= lower_bound) & (subset[y_label] <= upper_bound)
#         subset_filtered = subset[mask]
#     else:
#         continue

#     x = subset[x_label]
#     y = subset[y_label]
    
#     # unique_groups = subset[color_by_col].dropna().unique()
#     # palette = sns.color_palette("tab10", len(unique_groups))
#     # color_map = dict(zip(unique_groups, palette))


#     # Plotting
#     plt.figure(figsize=(8, 6))
    
#     # === SEABORN SCATTERPLOT ===
#     sc = sns.scatterplot(
#         data=subset_filtered, 
#         x=x_label, 
#         y=y_label, 
#         hue=color_by_col,   # This automatically colors points based on your column!,
#         hue_order=subgroups if color_by_col == 'Group_ID' else None,
#         alpha=0.7,
#         s=25, # marker size,
#         marker='o',
#         # facecolors='none',
#         edgecolors='none',
#         palette=color_map
#     )

#     group_means = subset.groupby(color_by_col)[[x_label, y_label]].mean()
#     for group_val, mean_row in group_means.iterrows():
#         if pd.notna(mean_row[x_label]) and pd.notna(mean_row[y_label]):
#             plt.scatter(
#                 mean_row[x_label], 
#                 mean_row[y_label], 
#                 marker='X', 
#                 s=50,             # Distinct, larger size
#                 color=color_map[group_val], 
#                 edgecolors='black', # High contrast outline
#                 linewidths=1.5,
#                 zorder=5            # Forces plotting on top of scatter dots
#             )

#     x_name = display_names.get(x_label, x_label.capitalize().replace('_',' '))
#     y_name = display_names.get(y_label, y_label.capitalize().replace('_',''))

#     if stats_per_group and color_by_col in subset.columns:
#         count = 0
#         for group_val, group_df in subset.groupby(color_by_col):
#             group_clean = group_df.dropna(subset=[x_label, y_label])
#             gx = group_clean[x_label]
#             gy = group_clean[y_label]
            
#             if len(gx) > 2: 
#                 g_pearson_corr, g_pearson_p = stats.pearsonr(gx, gy)
#                 g_spearman_corr, g_spearman_p = stats.spearmanr(gx, gy)
                
#                 is_significant = g_pearson_p < 0.05
#                 line_style = "-" if is_significant else "--"
#                 line_width = 3.5 if is_significant else 1.5
#                 line_alpha = 1.0 if is_significant else 0.3
                
#                 gm, gb = np.polyfit(gx, gy, 1)
#                 gx_range = np.linspace(gx.min(), gx.max(), 100)
#                 plt.plot(gx_range, gm * gx_range + gb, color=color_map[group_val], 
#                         linestyle=line_style, linewidth=line_width, alpha=line_alpha)
                
#                 # SINGLE LINE COMPACT TEXT
#                 sc.text(
#                     1.02, 0.5 - (count * 0.05),  # 0.05 tight spacing
#                     f"{group_val}: r={g_pearson_corr:.2f} (p={g_pearson_p:.3f}) | ρ={g_spearman_corr:.2f}",
#                     transform=sc.transAxes,
#                     fontsize=10,
#                     fontweight='bold' if is_significant else 'normal',
#                     color=color_map[group_val]
#                 )
#                 count += 1
   

#     else:
#             # Statistics
#         pearson_corr, pearson_p = stats.pearsonr(x, y)
#         spearman_corr, spearman_p = stats.spearmanr(x, y)
        
#         print("=== Correlation Test Results ===")
#         print(f"Pearson correlation: {pearson_corr:.3f}, p-value: {pearson_p:.4f}")
#         print(f"Spearman correlation: {spearman_corr:.3f}, p-value: {spearman_p:.4f}")
#         # Fit and plot regression line
#         if len(x) > 1:
#             m, b = np.polyfit(x, y, 1)
#             plt.plot(x, m * x + b, color="black", linestyle="--", label="Linear Fit", alpha=0.5)

#             # Move legend outside if there are many labels
#         # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')


#         # Plot correlation as text under legend
#         sc.text(
#             1.02, 0.5,
#             f"Pearson Correlation: {pearson_corr:.3f}, \np-value: {pearson_p:.4f} \n\nSpearman Correlation.: {spearman_corr:.3f}, \np-value: {spearman_p:.4f}",
#             transform=sc.transAxes,  # Keep position relative to axes
#             fontsize=12,
#             fontweight='bold'

#         )
#         plt.legend(loc='upper left', bbox_to_anchor=(0, -0.1))

#         # sc.set_xlabel(f"{x_name}")
#     sc.set_ylabel(f"{y_name}",fontsize=14, fontweight='bold', labelpad=12)
#     if "centroid_x" in x_label or "centroid_y" in x_label:
#             # 1. Remove the standard numeric tick marks and labels
#         plt.xticks([]) 
#         # 2. Set the custom range label. Swap order if necessary!
#         plt.xlabel("Kidney Regions", fontsize=14, fontweight='bold', labelpad=12)
#     else:
#         sc.set_xlabel(f"{x_name}",fontsize=14, fontweight='bold', labelpad=12)
#     # plt.title(f"{unit}: {x_name} vs {y_name}", fontsize=16, fontweight='bold')
#     # Final layout tweaks
#     # plt.legend(loc='lower left')
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, f"{unit}.png"))

#     # plt.show()

# img_path = r"C:\Users\jvill\Documents\integrated_pipeline\WSI_data\WSI_data\Ratio\S21-K21_198_3-Collagen-ratio.tif"
# img = tifffile.imread(img_path)

# # dataframe_path = r"C:\Users\jvill\Documents\integrated_pipeline\WSI_data\WSI_data\CODEX_clusters-regions_masks_output_csv\S21_filtered-ratios.csv"
# # df = pd.read_csv(dataframe_path)
# visualizer.plot_topographical_intensity_map(
#     img,  
#     sigma=5,              # Smooths the terrain
#     cmap='jet',           # Vibrant intensity map
#     elevation=40, 
#     azimuth=90, 
#     output_dir=output_path
# )
