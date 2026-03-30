# Configuration System Guide

This project uses a JSON-based configuration system to manage analysis parameters, making it easy to switch between different experimental setups without modifying the Python code.

## Quick Start

### Switching Profiles

To change your configuration profile, edit [classify.py](classify.py) line ~30:

```python
def main():
    # ========================================================================
    # CONFIGURATION PROFILE SELECTION
    # ========================================================================
    # Available profiles: 'kidney_clusters' (default), 'kidney_nephron'
    # Change this to switch between different configurations
    PROFILE = 'kidney_clusters'  # ← Change this value
```

### Available Profiles

- **`kidney_clusters`**: Kidney tissue analysis with 26 clusters (default)
  - 26 cell type clusters with detailed annotations
  - Cluster-specific colors and display names
  
- **`kidney_nephron`**: Kidney tissue analysis with major nephron segments
  - 7 major anatomical units (Glomeruli, PT, Distal Tubule, etc.)
  - Simpler configuration with fewer units

## Configuration Structure

Each profile is a JSON file in the `configs/` directory with the following sections:

### 1. Profile Metadata
```json
{
  "profile_name": "Kidney Clusters (26 cell types)",
  "description": "Configuration for kidney tissue analysis with 26 clusters"
}
```

### 2. Processing Options
```json
{
  "processing_options": {
    "perform_RF_classification": false,
    "perform_masking": false,
    "display_plots": true
  }
}
```

### 3. Dataset Parameters
```json
{
  "dataset_parameters": {
    "mask_type": "CODEX_clusters_masks",
    "regions": ["Cortex", "Medulla"],
    "display_units": null,
    "molecules_to_display": [...],
    "ratios_to_display": null
  }
}
```

### 4. Display Names
Maps original molecule/ratio names to display names for plots:
```json
{
  "display_names": {
    "16:0 Cardiolipin": "CL 16:0",
    "PC Mix": "PC",
    ...
  }
}
```

### 5. Unit Colors (Hex Codes)
```json
{
  "unit_colors": {
    "Cluster 1": "#e8260c",
    "Cluster 2": "#0a8416",
    ...
  }
}
```

### 6. Unit Display Names
```json
{
  "unit_display_names": {
    "Cluster1": "Glomeruli | Podo, CD31",
    "Cluster2": "PT | LRP2",
    ...
  }
}
```

### 7. Heatmap Unit Order
Controls the order units appear on x-axis in heatmaps:
```json
{
  "heatmap_unit_order": [
    "Cluster1", "Cluster2", "Cluster3", ...
  ]
}
```

### 8. Multi-Panel Configuration
```json
{
  "multi_panel_config": {
    "selected_molecules": [...],
    "selected_ratios": [...]
  }
}
```

## Creating a New Profile

1. Create a new JSON file in `configs/` directory:
   ```bash
   cp configs/kidney_clusters.json configs/my_new_profile.json
   ```

2. Edit the new file with your custom settings

3. Update `classify.py` to use your profile:
   ```python
   PROFILE = 'my_new_profile'
   ```

## Command-Line Utilities

You can check available profiles programmatically:

```python
from config_loader import list_available_profiles, print_config_info

# List all profiles
profiles = list_available_profiles()
print(profiles)  # ['kidney_clusters', 'kidney_nephron']

# Print info about a profile
print_config_info('kidney_clusters')
```

## Configuration Precedence

The configuration system applies settings in this order:
1. Load JSON profile from `configs/` directory
2. Extract all settings into variables
3. Derive bubble_unit_order as reverse of heatmap_unit_order
4. Use throughout the analysis script

## Tips for Configuration

- **Colors**: Use standard hex color codes (#RRGGBB)
- **Unit Order**: Bubble plots use reverse of heatmap order by default
- **Display Names**: Keep short for readability in plots
- **Molecules**: Use exact names as they appear in your data

## Troubleshooting

### Profile Not Found
Error: `FileNotFoundError: Profile 'xyz' not found`

Solution: Check that your JSON file exists in `configs/` directory and the filename matches exactly (case-sensitive).

### Configuration Mismatch
If plots don't display correctly:
1. Verify unit names in config match your data
2. Check colors are valid hex codes
3. Ensure all required sections are present in JSON

## File Locations

```
sk_classifier/
├── classify.py           # Main analysis script (edit PROFILE here)
├── config_loader.py      # Configuration loading utilities
└── configs/
    ├── kidney_clusters.json     # Default cluster-based profile
    ├── kidney_nephron.json      # Nephron segments profile
    └── README.md                # This file
```

## Version History

- **v1.0**: Initial configuration system with two profiles
