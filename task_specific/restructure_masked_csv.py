"""
Process molecule percentages and masked ratio means CSV files
Each file is processed separately to create structured output with:
- data_type column
- Region×Unit columns (e.g., "GL (C)", "DN (M)")
- Subcolumns for each instance×replicate
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob
import re

def process_all_csv_files(input_dir, output_dir=None):
    """
    Process all CSV files matching patterns in the input directory
    Each file is processed separately and saved as a new CSV
    
    Parameters:
    -----------
    input_dir : str
        Path to directory containing the CSV files
    output_dir : str, optional
        Directory for output CSV files. If None, saves to same directory as input files
    """
    input_path = Path(input_dir)
    
    if output_dir is None:
        output_path = None  # Will save to same directory as each input file
    else:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all CSV files matching patterns
    masked_files = glob.glob(str(input_path / "masked_*.csv"))
    molecule_files = glob.glob(str(input_path / "molecule*.csv"))
    
    # Exclude already restructured files
    all_files = [f for f in (masked_files + molecule_files) if '_restructured' not in f]
    
    if not all_files:
        print(f"No CSV files found in {input_path}")
        print(f"Searched for: masked_*.csv and molecule*.csv")
        return
    
    print(f"Found {len(all_files)} file(s) to process:\n")
    
    for input_file in all_files:
        process_single_file(input_file, output_path)
    
    print("\n" + "="*80)
    print("All files processed successfully!")


def process_single_file(input_file, output_path):
    """
    Process a single CSV file and create separate restructured outputs for each data type
    
    Parameters:
    -----------
    input_file : str
        Path to input CSV file
    output_path : Path
        Directory to save output files (if None, saves to same directory as input)
    """
    input_file = Path(input_file)
    print(f"\n{'='*80}")
    print(f"Processing: {input_file.name}")
    print('='*80)
    
    # Load the CSV file
    df = pd.read_csv(input_file)
    print(f"Shape: {df.shape}")
    print(f"\nColumns ({len(df.columns)}):")
    print(df.columns.tolist())
    print(f"\nFirst few rows:")
    print(df.head())
    
    sample_selection = '24'  # Change this value to select different samples if needed

    # Filter: Keep only instances from sample 24
    if sample_selection != '':
        print(f"\nFiltering to Sample {sample_selection} only...")
    else:
        print(f"\nProcessing all samples...")

    if 'Sample_Name' in df.columns:
        original_count = len(df)
        # Keep only rows where Sample_Name contains the selected sample
        df = df[df['Sample_Name'].str.contains(sample_selection, na=False)].copy()
        removed_count = original_count - len(df)
        
        if len(df) > 0:
            print(f"\nFiltered to Sample {sample_selection} only: kept {len(df)} rows, removed {removed_count} rows from other samples")
            print(f"New shape: {df.shape}")
        else:
            print(f"\nWarning: No data found for Sample {sample_selection}!")
    
    # Reshape the data - creates multiple CSV files
    reshape_data_by_type(df, input_file, output_path, sample_selection)


def reshape_data_by_type(df, input_file, output_path, sample_selection):
    """
    Reshape the dataframe and create separate CSV files for each data type
    Each CSV has:
    - Rows: Individual instances (each column may have different number of rows)
    - Columns: Region×Unit combinations (abbreviated as "UnitAbbrev (R)" where R is first letter of region)
    - Values: measurements for that specific data type
    - Note: Columns may have different lengths, padded with NaN where needed
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with columns: Unit, Ratio_Type/Molecule, Region, Instance_Label, Replicate, Sample_Name, etc.
    input_file : Path
        Original input file path
    output_path : Path
        Directory to save output files (if None, saves to same directory as input)
    sample_selection : str
        Sample number to filter and process
    """
    
    # Define unit abbreviations (matching hsi_classify_rf.py)
    unit_display_names = {
        'Glomeruli': 'GL',
        'Proximal Tubule': 'PT',
        'Distal Tubule': 'DT',
        'Thick Ascending Limb': 'TAL',
        'Distal Nephron': 'DN',
        'Thin Ascending Limb': 'ATL',
        'Vasculature': 'V'
    }
    
    # Determine if this is ratio or molecule data
    if 'Ratio_Type' in df.columns:
        data_type_col = 'Ratio_Type'
        value_col = 'Mean_Ratio'
    elif 'Molecule' in df.columns:
        data_type_col = 'Molecule'
        value_col = 'Percentage'
    else:
        raise ValueError("Could not find Ratio_Type or Molecule column")
    
    print(f"\nReshaping data by {data_type_col}...")
    print(f"Value column: {value_col}")
    
    # Create abbreviated Unit×Region column
    # Format: "UnitAbbrev (R)" where R is first letter of region
    df['Unit_Abbrev'] = df['Unit'].map(unit_display_names)
    df['Region_Abbrev'] = df['Region'].str[0]  # First letter of region
    df['Region_Unit'] = df['Unit_Abbrev'] + ' (' + df['Region_Abbrev'] + ')'
    
    # Create Sample×Instance identifier for tracking unique instances
    df['Sample_Instance'] = df['Sample_Name'] + '_Instance_' + df['Instance_Label'].astype(str)
    
    print(f"Number of unique {data_type_col}s: {df[data_type_col].nunique()}")
    print(f"Number of unique Region×Unit combinations: {df['Region_Unit'].nunique()}")
    
    # Get unique data types
    data_types = sorted(df[data_type_col].unique())
    
    # Determine output directory
    if sample_selection == '':
            sample_selection = 'all'

    output_dir_name = f"restructured_by_type_sample_{sample_selection}"
    if output_path is None:
        output_dir = input_file.parent / output_dir_name
        summary_output_dir = input_file.parent / f"restructured_by_type_summary_sample_{sample_selection}"
    else:
        output_dir = output_path / output_dir_name
        summary_output_dir = output_path / f"restructured_by_type_summary_sample_{sample_selection}"
    
    # Create output directories if they don't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_output_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = input_file.stem
    
    # Process each data type separately
    for data_type in data_types:
        # Filter data for this specific data type
        df_type = df[df[data_type_col] == data_type].copy()
        
        # Get all unique Region×Unit combinations
        region_units = sorted(df_type['Region_Unit'].unique())
        
        # Create a dictionary to store data for each Region×Unit
        data_dict = {}
        max_length = 0
        
        for region_unit in region_units:
            # Get all instances for this Region×Unit and data type
            region_data = df_type[df_type['Region_Unit'] == region_unit][value_col].values
            # Exclude zeros and 100s
            region_data = region_data[(region_data != 0) & (region_data != 100)]
            data_dict[region_unit] = region_data.tolist()
            max_length = max(max_length, len(region_data))
        
        # Pad columns with NaN to make them equal length for CSV format
        for region_unit in region_units:
            current_length = len(data_dict[region_unit])
            if current_length < max_length:
                data_dict[region_unit].extend([np.nan] * (max_length - current_length))
        
        # Create DataFrame from dictionary
        result_df = pd.DataFrame(data_dict)
        
        # Add index column
        result_df.insert(0, 'Index', range(1, len(result_df) + 1))
        
        # Create safe filename from data type name
        safe_data_type = data_type.replace('/', '_').replace(':', '_').replace(' ', '_')
        output_file = output_dir / f"{base_name}_{safe_data_type}.csv"
        
        # Save to CSV
        result_df.to_csv(output_file, index=False)
        
        # Report actual counts per column (excluding NaN)
        counts = {col: result_df[col].notna().sum() for col in result_df.columns if col != 'Index'}
        print(f"\n  {data_type}: {len(region_units)} Region×Units, max {max_length} instances")
        print(f"    Instance counts per Region×Unit:")
        for col in sorted(counts.keys()):
            print(f"      {col}: {counts[col]} instances")
        print(f"    Saved to: {output_file.name}")
        
        # Create summary CSV with averages, standard deviation, and sample sizes
        summary_data = {}
        for region_unit in region_units:
            col_data = result_df[region_unit].dropna()
            summary_data[region_unit] = {
                'Mean': col_data.mean() if len(col_data) > 0 else np.nan,
                'Std_Dev': col_data.std() if len(col_data) > 0 else np.nan,
                'Sample_Size': len(col_data)
            }
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_data).T
        summary_df.index.name = 'Region_Unit'
        summary_df = summary_df.reset_index()
        
        # Save summary CSV
        summary_output_file = summary_output_dir / f"sample_{sample_selection}_{base_name}_{safe_data_type}_summary.csv"
        summary_df.to_csv(summary_output_file, index=False)
        print(f"    Summary saved to: {summary_output_file.name}")
    
    print(f"\nCreated {len(data_types)} separate CSV files")




if __name__ == "__main__":
    # Set the input directory
    input_dir = "/Users/jorgevillazon/Documents/files/codex-srs/HuBMAP .tif files for Jorge Part 1/rf_outputs/"
    
    # Set output directory to None to save in same directory as input files
    output_dir = None
    
    # Process all files
    process_all_csv_files(input_dir, output_dir)
