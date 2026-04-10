"""
Configuration loader for classify.py
Handles loading profile-based JSON configurations
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List


def load_config(profile_name: str = "kidney_clusters") -> Dict[str, Any]:
    """
    Load configuration from JSON profile file.
    
    Args:
        profile_name: Name of the profile (without .json extension)
                     e.g., "kidney_clusters", "kidney_nephron"
    
    Returns:
        Dictionary containing the configuration
        
    Raises:
        FileNotFoundError: If the profile file doesn't exist
        json.JSONDecodeError: If the JSON is invalid
    """
    config_dir = Path(__file__).parent / "configs"
    config_file = config_dir / f"{profile_name}.json"
    
    if not config_file.exists():
        available = list(config_dir.glob("*.json"))
        profiles = [f.stem for f in available]
        raise FileNotFoundError(
            f"Profile '{profile_name}' not found.\n"
            f"Available profiles: {', '.join(profiles)}\n"
            f"Expected: {config_file}"
        )
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    return config


def list_available_profiles() -> List[str]:
    """
    List all available configuration profiles.
    
    Returns:
        List of profile names (without .json extension)
    """
    config_dir = Path(__file__).parent / "configs"
    profiles = [f.stem for f in config_dir.glob("*.json")]
    return sorted(profiles)


def apply_config(config: Dict[str, Any]) -> tuple:
    """
    Extract configuration values into variables used in classify.py
    
    Args:
        config: Configuration dictionary from load_config()
    
    Returns:
        Tuple of all configuration variables in order:
        (perform_RF_classification, perform_masking, display_plots,
         mask_type, subgroups, display_units_maps, molecules_to_display, ratios_to_display,
         display_names, unit_mappings, unit_colors, unit_display_names, heatmap_unit_order,
         selected_molecules, selected_ratios, display_units_boxplots, 
         individual_plot_config, sample_analysis_config)
    """
    # Processing options
    proc_opts = config.get("processing_options", {})
    perform_RF_classification = proc_opts.get("perform_RF_classification", False)
    perform_masking = proc_opts.get("perform_masking", False)
    display_plots = proc_opts.get("display_plots", True)
    
    # Dataset parameters
    dataset_params = config.get("dataset_parameters", {})
    mask_type = dataset_params.get("mask_type", "Mask")
    subgroups = dataset_params.get("subgroups",  None)
    display_units_maps = dataset_params.get("display_units_maps", None)
    molecules_to_display = dataset_params.get("molecules_to_display", None)
    ratios_to_display = dataset_params.get("ratios_to_display", None)
    display_units_boxplots = dataset_params.get("display_units_boxplots", None)
    
    # Display configuration
    display_names = config.get("display_names", {})
    unit_colors = config.get("unit_colors", {})
    unit_display_names = config.get("unit_display_names", {})
    heatmap_unit_order = config.get("heatmap_unit_order", [])
    
    # Individual plot config
    individual_plot_config = config.get("individual_plot_config", {})
    selected_molecules = individual_plot_config.get("selected_molecules", [])
    selected_ratios = individual_plot_config.get("selected_ratios", [])

    # Unit mappings config
    unit_mappings = config.get("unit_mappings", {})
    
    # Sample analysis config
    sample_analysis_config = config.get("sample_analysis_config", {})

    return (
        perform_RF_classification, perform_masking, display_plots,
        mask_type, subgroups, display_units_maps, molecules_to_display, ratios_to_display,
        display_names, unit_mappings, unit_colors, unit_display_names, heatmap_unit_order,
        selected_molecules, selected_ratios, display_units_boxplots,
        individual_plot_config, sample_analysis_config
    )


def print_config_info(profile_name: str = "kidney_clusters") -> None:
    """
    Print information about a configuration profile.
    
    Args:
        profile_name: Name of the profile to display info for
    """
    config = load_config(profile_name)
    print(f"\n{'='*80}")
    print(f"Configuration Profile: {config.get('profile_name', profile_name)}")
    print(f"{'='*80}")
    print(f"Description: {config.get('description', 'No description')}")
    print(f"\nProcessing Options:")
    for key, val in config.get("processing_options", {}).items():
        print(f"  {key}: {val}")
    print(f"\nDataset Parameters:")
    for key, val in config.get("dataset_parameters", {}).items():
        if key == "molecules_to_display" and isinstance(val, list):
            print(f"  {key}: {len(val)} molecules")
        else:
            print(f"  {key}: {val}")
    # print(f"\nIndividual Plot Configuration:")
    # individual_config = config.get("individual_plot_config", {})
    # if individual_config:
    #     for key, val in individual_config.items():
    #         if isinstance(val, list):
    #             print(f"  {key}: {val}")
    #         else:
    #             print(f"  {key}: {val}")
    # else:
    #     print(f"  (not configured)")
    print(f"\nUnits: {len(config.get('unit_colors', {}))}")
