"""
Create UMAP plots with cluster positions from CODEX tissue data,
overlaid with ratio values from masked_ratio_means.csv.

This script:
1. Loads UMAP embedding positions from cluster_info.csv (CODEX tissue clusters)
2. Loads ratio data from masked_ratio_means.csv (RF predictions)
3. For each cluster and each ratio type:
   - Maps replicates to UMAP positions from that cluster
   - Colors points by raw ratio values
   - Creates a separate plot per ratio type

Usage:
    python ratio_umap_overlay.py
"""

import os
import sys
from pathlib import Path
import io

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def select_clusters_by_label_names(label_names_map, selected_label_names):
    """Return original cluster indices whose names match any selected term."""
    if not selected_label_names:
        return sorted(label_names_map.keys())

    normalized_terms = [str(term).strip().lower() for term in selected_label_names if str(term).strip()]
    if not normalized_terms:
        return sorted(label_names_map.keys())

    selected = []
    for cluster_idx, cluster_name in label_names_map.items():
        cluster_name_lower = str(cluster_name).lower()
        if any(term in cluster_name_lower for term in normalized_terms):
            selected.append(cluster_idx)
    return sorted(selected)


def apply_z_normalization(values):
    """
    Apply z-score normalization to values.
    
    Parameters
    ----------
    values : np.ndarray
        Array of values to normalize
    
    Returns
    -------
    normalized : np.ndarray
        Z-normalized values (zero mean, unit variance)
    """
    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        return np.zeros_like(values)
    return (values - mean) / std


def load_codex_umap_data(codex_csv_path):
    """
    Load UMAP embedding positions from CODEX cluster_info.csv.
    
    Parameters
    ----------
    codex_csv_path : str
        Path to cluster_info.csv with UMAP_1, UMAP_2, FastPG columns
    
    Returns
    -------
    umap_positions : dict
        Mapping of cluster_idx -> list of (umap_1, umap_2) positions
    cluster_labels : dict
        Mapping of cluster_idx -> cluster name/label
    """
    df = pd.read_csv(codex_csv_path)
    
    # Extract columns
    umap_1 = df['UMAP_1'].values
    umap_2 = df['UMAP_2'].values
    fastpg = df['FastPG'].values
    
    print(f"Loaded UMAP data: {len(df)} points from {len(np.unique(fastpg))} clusters")
    
    # Group positions by cluster
    umap_positions = {}
    for cluster_idx in np.unique(fastpg):
        if np.isnan(cluster_idx):
            continue
        cluster_idx = int(cluster_idx)
        mask = fastpg == cluster_idx
        positions = list(zip(umap_1[mask], umap_2[mask]))
        umap_positions[cluster_idx] = positions
    
    print(f"Cluster position groups: {sorted(umap_positions.keys())}")
    print(f"  Min points per cluster: {min(len(v) for v in umap_positions.values())}")
    print(f"  Max points per cluster: {max(len(v) for v in umap_positions.values())}")
    
    return umap_positions, df


def load_ratio_data(ratio_csv_path):
    """
    Load ratio data from masked_ratio_means.csv.
    
    Parameters
    ----------
    ratio_csv_path : str
        Path to masked_ratio_means.csv
    
    Returns
    -------
    df : pd.DataFrame
        DataFrame with Unit, Ratio_Type, Mean_Ratio columns
    """
    df = pd.read_csv(ratio_csv_path)
    print(f"\nLoaded ratio data: {len(df)} records")
    print(f"Columns: {df.columns.tolist()}")
    
    # Extract key columns
    print(f"Unique units: {sorted(df['Unit'].unique())}")
    print(f"Unique ratio types: {sorted(df['Ratio_Type'].unique())}")
    
    return df


def create_ratio_umap_plots(codex_df, ratio_df, output_dir, codex_dir,
                           cmap_ranges=None, generate_both_versions=True,
                           selected_label_names=None, selected_clusters=None):
    """
    Create UMAP plots with cluster positions, colored by ratio values.
    
    Parameters
    ----------
    codex_df : pd.DataFrame
        DataFrame from CODEX cluster_info.csv
    ratio_df : pd.DataFrame
        DataFrame from masked_ratio_means.csv
    output_dir : str
        Directory to save plots
    codex_dir : str
        Directory containing CODEX data and ClusterKey.xlsx
    cmap_ranges : dict, optional
        Dictionary mapping ratio_type (lowercase, underscore format) to [min, max] ranges.
        Example: {"lipid_unsaturation": [0, 1], "lipid_to_protein": None, "redox": None}
        None means use automatic range from data. Default None uses auto range for all.
    generate_both_versions : bool, optional
        If True, generate both raw and z-normalized plots. Default True.
    selected_label_names : list[str] | None, optional
        List of cluster-name search terms used to subset clusters. Matching is
        case-insensitive and uses substring search. None or [] includes all.
    selected_clusters : list[int] | None, optional
        Specific FastPG indices to plot. If provided, overrides selected_label_names.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize cmap_ranges if not provided
    if cmap_ranges is None:
        cmap_ranges = {}
    
    # Normalize cmap_ranges keys to lowercase with underscores
    cmap_ranges_normalized = {}
    for key, val in cmap_ranges.items():
        normalized_key = key.lower().replace(' ', '_').replace('-', '_')
        cmap_ranges_normalized[normalized_key] = val
    cmap_ranges = cmap_ranges_normalized
    
    print(f"\nCustom colormap ranges: {cmap_ranges if cmap_ranges else 'None (auto)'}")
    
    # Load ClusterKey for optional name-based cluster filtering
    cluster_key_path = os.path.join(codex_dir, 'ClusterKey.xlsx')
    
    if not os.path.exists(cluster_key_path):
        print(f"WARNING: ClusterKey.xlsx not found at {cluster_key_path}")
        print("Using FastPG indices as-is")
        label_names_orig = {}
        # If no key, we assume all clusters in df are selected unless indices were provided
        available_clusters = sorted(int(x) for x in codex_df['FastPG'].dropna().unique())
        if selected_clusters is not None:
            selected_clusters = [c for c in selected_clusters if c in available_clusters]
        else:
            selected_clusters = available_clusters
    else:
        cluster_labels_df = pd.read_excel(cluster_key_path)
        label_names_orig = {
            int(idx): str(cluster_id) + ' | ' + str(reason) 
            for idx, (cluster_id, reason) in enumerate(
                zip(cluster_labels_df['Cluster ID'], cluster_labels_df['Reason for Call'])
            )
        }

        if selected_clusters is not None:
            selected_clusters = [c for c in selected_clusters if c in label_names_orig]
            print(f"Selected specific FastPG clusters: {selected_clusters}")
        else:
            selected_clusters = select_clusters_by_label_names(label_names_orig, selected_label_names)
            if selected_label_names:
                print(f"Selected label name filters: {selected_label_names}")
                print(f"Matched {len(selected_clusters)} clusters: {selected_clusters}")
        
        if not selected_clusters:
            print("No clusters matched filters. Nothing to plot.")
            return
    
    # Create inverse mapping from unit name to FastPG index
    # index_to_unit maps FastPG index (int) -> unit_name (str)
    # We need unit_name -> FastPG index for UMAP lookup
    index_to_unit = {
        "20": "Cluster1",
        "15": "Cluster2",
        "19": "Cluster3",
        "16": "Cluster4",
        "9": "Cluster5",
        "18": "Cluster6",
        "26": "Cluster7",
        "1": "Cluster8",
        "0": "Cluster9",
        "23": "Cluster10",
        "25": "Cluster11",
        "2": "Cluster12",
        "3": "Cluster13",
        "24": "Cluster14",
        "12": "Cluster15",
        "21": "Cluster16",
        "17": "Cluster17",
        "10": "Cluster18",
        "4": "Cluster19",
        "5": "Cluster20",
        "14": "Cluster21",
        "22": "Cluster22",
        "8": "Cluster23",
        "11": "Cluster24",
        "13": "Cluster25",
        "6": "Cluster26",
        "7": "Cluster27",
    }
    unit_to_fastpg = {v: int(k) for k, v in index_to_unit.items()}
    
    # Create mapping from FastPG index to cluster number (for display labels)
    # Extract "N" from "ClusterN" to use as display label on plots
    fastpg_to_cluster_number = {}
    for fastpg_idx_str, unit_name in index_to_unit.items():
        fastpg_idx = int(fastpg_idx_str)
        cluster_num = int(unit_name.replace("Cluster", ""))
        fastpg_to_cluster_number[fastpg_idx] = cluster_num

    # To avoid skipped numbers on the plot (e.g., labels 1, 3, 5), if a subset is selected,
    # we remap the original cluster numbers to be contiguous (1, 2, 3...).
    all_fastpg_indices = sorted(unit_to_fastpg.values())
    if len(selected_clusters) < len(all_fastpg_indices):
        print("Adjusting cluster numbers to be contiguous (no skipping)...")
        # Get original numbers for the selected clusters
        selected_orig_nums = []
        for idx in selected_clusters:
            if idx in fastpg_to_cluster_number:
                selected_orig_nums.append((idx, fastpg_to_cluster_number[idx]))
        
        # Sort by their original sequence provided in index_to_unit
        selected_orig_nums.sort(key=lambda x: x[1])
        
        # Create contiguous mapping
        display_label_map = {idx: i+1 for i, (idx, _) in enumerate(selected_orig_nums)}
    else:
        display_label_map = fastpg_to_cluster_number
    
    # Extract cluster from ratio data Unit column and map to FastPG index
    # Unit values should be "ClusterN" format
    ratio_df['Unit_Name'] = ratio_df['Unit'].astype(str).str.strip()
    ratio_df['Cluster_Idx'] = ratio_df['Unit_Name'].map(unit_to_fastpg)
    ratio_df = ratio_df.dropna(subset=['Cluster_Idx']).copy()
    ratio_df['Cluster_Idx'] = ratio_df['Cluster_Idx'].astype(int)
    
    # Restrict CODEX and ratio data to selected clusters before plotting.
    codex_df = codex_df[codex_df['FastPG'].isin(selected_clusters)].copy()
    ratio_df = ratio_df[ratio_df['Cluster_Idx'].isin(selected_clusters)].copy()

    if codex_df.empty:
        print("No CODEX rows remain after selected label filtering. Nothing to plot.")
        return
    if ratio_df.empty:
        print("No ratio rows remain after selected label filtering. Nothing to plot.")
        return

    # Extract UMAP positions grouped by ORIGINAL cluster index
    umap_1 = codex_df['UMAP_1'].values
    umap_2 = codex_df['UMAP_2'].values
    fastpg = codex_df['FastPG'].values
    
    # Extract cluster positions by ORIGINAL FastPG index
    umap_positions = {}
    cluster_centroids = {}  # Store centroids for label placement
    for cluster_idx in np.unique(fastpg):
        if np.isnan(cluster_idx):
            continue
        cluster_idx = int(cluster_idx)
        mask = fastpg == cluster_idx
        positions = np.array(list(zip(umap_1[mask], umap_2[mask])))
        umap_positions[cluster_idx] = positions
        # Calculate centroid
        cluster_centroids[cluster_idx] = (
            np.mean(positions[:, 0]),
            np.mean(positions[:, 1])
        )
    
    print("\nUMAP position groups by original FastPG index:")
    for idx in sorted(umap_positions.keys()):
        print(f"  Cluster {idx}: {len(umap_positions[idx])} positions")
    
    print("\nCluster centroids (for label placement):")
    for idx in sorted(cluster_centroids.keys()):
        cx, cy = cluster_centroids[idx]
        print(f"  Cluster {idx}: ({cx:.2f}, {cy:.2f})")
    
    # Get unique ratio types
    ratio_types = sorted(ratio_df['Ratio_Type'].unique())
    
    print(f"\nCreating plots for {len(ratio_types)} ratio types:")
    for ratio_type in ratio_types:
        print(f"  - {ratio_type}")
    
    # Determine versions to generate
    versions = []
    if generate_both_versions:
        versions = ['raw', 'normalized']
    else:
        versions = ['raw']
    
    print(f"\nGenerating {len(versions)} version(s) per ratio type: {versions}")
    
    # Pre-compute selected positions for each cluster (use consistent selection across ratio types)
    # Determine the maximum number of replicates per cluster across all ratio types
    max_replicates_per_cluster = {}
    for cluster_idx in ratio_df['Cluster_Idx'].unique():
        cluster_data = ratio_df[ratio_df['Cluster_Idx'] == cluster_idx]
        n_replicates = len(cluster_data)
        max_replicates_per_cluster[cluster_idx] = n_replicates
    
    # Pre-select positions for each cluster (fixed seed for reproducibility)
    np.random.seed(42)
    cluster_position_mapping = {}  # Maps cluster_idx -> list of selected (x, y) positions
    
    print("\nPre-computing cluster position selection for consistency:")
    for cluster_idx in sorted(max_replicates_per_cluster.keys()):
        if cluster_idx not in umap_positions:
            print(f"  Warning: Cluster {cluster_idx} not in CODEX data, skipping")
            continue
        
        n_replicates = max_replicates_per_cluster[cluster_idx]
        positions = umap_positions[cluster_idx]
        n_positions = len(positions)
        
        print(f"  Cluster {cluster_idx}: {n_replicates} replicates, {n_positions} positions available")
        
        # If more positions than replicates, randomly sample (with fixed seed for consistency)
        if n_positions > n_replicates:
            selected_indices = np.random.choice(n_positions, n_replicates, replace=False)
            selected_positions = positions[selected_indices]
            print(f"    Selected {n_replicates} positions from {n_positions}")
        else:
            selected_positions = positions[:n_replicates]
            if n_replicates > n_positions:
                print(f"    Warning: Only {n_positions} positions available for {n_replicates} replicates")
        
        cluster_position_mapping[cluster_idx] = selected_positions
    
    # For each version (raw, normalized) and each ratio type, create a plot
    for version in versions:
        print(f"\n{'='*70}")
        print(f"Processing {version.upper()} values")
        print(f"{'='*70}")
        
        # Create version-specific output directory
        version_output_dir = os.path.join(output_dir, version)
        os.makedirs(version_output_dir, exist_ok=True)
        
        for ratio_type in ratio_types:
            print(f"\nProcessing {ratio_type}...")
            
            # Filter for this ratio type
            ratio_subset = ratio_df[ratio_df['Ratio_Type'] == ratio_type].copy()

            ratio_subset = ratio_subset[ratio_subset['Source_ID']=='S21'].copy()
            print(f"  Found {len(ratio_subset)} records for {ratio_type}")
            
            # Initialize arrays for embedded points
            embedded_x = []
            embedded_y = []
            embedded_values = []
        
            # For each cluster, map replicates to UMAP positions
            clusters_in_data = sorted(ratio_subset['Cluster_Idx'].unique())
        
            for cluster_idx in clusters_in_data:
                if cluster_idx not in umap_positions:
                    print(f"  Warning: Cluster {cluster_idx} not in CODEX data, skipping")
                    continue
            
                # Get all replicates for this cluster and ratio type
                cluster_data = ratio_subset[ratio_subset['Cluster_Idx'] == cluster_idx]
                n_replicates = len(cluster_data)
            
                # Use pre-computed selected positions for this cluster
                if cluster_idx not in cluster_position_mapping:
                    print(f"    Warning: Cluster {cluster_idx} not in position mapping, skipping")
                    continue
                
                selected_positions = cluster_position_mapping[cluster_idx]
                print(f"    Cluster {cluster_idx}: {n_replicates} replicates using pre-selected positions")
            
                # Map replicates to positions
                ratio_values = cluster_data['Mean_Ratio'].values
            
                for pos, val in zip(selected_positions, ratio_values):
                    embedded_x.append(pos[0])
                    embedded_y.append(pos[1])
                    embedded_values.append(val)
        
            embedded_x = np.array(embedded_x)
            embedded_y = np.array(embedded_y)
            embedded_values = np.array(embedded_values)
        
            print(f"  Total embedded points: {len(embedded_x)}")
            print(f"  Value range: {embedded_values.min():.4f} to {embedded_values.max():.4f}")
        
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6.5))
        
            # Normalize values for colormap - check for custom range (only for raw version)
            ratio_key_normalized = ratio_type.lower().replace(' ', '_').replace('-', '_')
            if version == 'raw' and ratio_key_normalized in cmap_ranges and cmap_ranges[ratio_key_normalized] is not None:
                vmin, vmax = cmap_ranges[ratio_key_normalized]
                print(f"  Using custom colormap range: [{vmin}, {vmax}]")
            else:
                vmin = embedded_values.min()
                vmax = embedded_values.max()
                if version == 'normalized':
                    print(f"  Using automatic colormap range (z-normalized): [{vmin:.4f}, {vmax:.4f}]")
                else:
                    print(f"  Using automatic colormap range: [{vmin:.4f}, {vmax:.4f}]")
        
            norm = Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.cm.RdYlBu_r  # Red (high) to Blue (low)
        
            # Scatter plot
            scatter = ax.scatter(
                embedded_x, embedded_y,
                c=embedded_values, cmap=cmap, norm=norm,
                s=50, alpha=0.7, edgecolors='black', linewidth=0.5,
                rasterized=False
            )
        
            # Colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.ax.tick_params(labelsize=11)
            cbar.set_label('Mean Ratio Value', fontsize=12, fontweight='bold', color='black')
        
            # Add cluster center labels
            import matplotlib.patheffects as pe
            for cluster_idx in sorted(cluster_centroids.keys()):
                cx, cy = cluster_centroids[cluster_idx]
                # Only label clusters that have data in this ratio type
                if len(embedded_x) > 0:  # Safety check
                    # Use the remapped display number to avoid skips
                    cluster_num = display_label_map.get(cluster_idx, cluster_idx)
                    ax.text(
                        cx, cy, str(int(cluster_num)),
                        fontsize=9, fontweight='bold', ha='center', va='center',
                        color='black',
                        path_effects=[pe.withStroke(linewidth=2.5, foreground='white')],
                        zorder=100  # Ensure labels are on top
                    )

            # Labels and title
            ax.set_xlabel('UMAP 1', fontsize=12, fontweight='bold')
            ax.set_ylabel('UMAP 2', fontsize=12, fontweight='bold')
            ax.set_title(f'{ratio_type.replace("_", " ")}', 
                        fontsize=13, fontweight='bold', pad=15, color='black')
            ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.5)
            ax.set_axisbelow(True)
            ax.set_facecolor('white')
            ax.tick_params(axis='both', which='major', labelsize=11, colors='black')
            # Make all spines visible and styled
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1)

            ax.set_aspect('equal', adjustable='box')
        
            fig.tight_layout()
        
            # Save
            selection = "all"
            if selected_label_names:
                selection = "_".join([name.replace(" ", "_") for name in selected_label_names])

            # Sanitize filename: remove/replace invalid Windows characters
            sanitized_ratio_type = ratio_type.replace(" ", "_").replace("-", "_").replace("/", "_").replace("\\", "_")
            
            # Ensure output directory exists
            os.makedirs(version_output_dir, exist_ok=True)
            
            output_filename = f'ratio_umap_{sanitized_ratio_type}_{selection}.png'
            output_path = os.path.join(version_output_dir, output_filename)
            
            # Save using buffer to avoid PIL path issues on Windows
            print(f"  Saving to: {output_path}")
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            with open(output_path, 'wb') as f:
                f.write(buf.getvalue())
            print(f"  Saved to: {output_path}")
        
            plt.close(fig)
        
            # Also save CSV with coordinates and values
            csv_filename = f'ratio_umap_{sanitized_ratio_type}_{selection}.csv'
            csv_path = os.path.join(version_output_dir, csv_filename)
            export_df = pd.DataFrame({
                'UMAP_1': embedded_x,
                'UMAP_2': embedded_y,
                'Ratio_Value': embedded_values
            })
            export_df.to_csv(csv_path, index=False)
            print(f"  Data saved to: {csv_path}")
    
    print(f"\n{'='*70}")
    print("Ratio UMAP overlay plots complete!")
    print(f"{'='*70}")


def main():
    """Main workflow."""
    
    # CODEX data paths
    codex_dir = r'D:\integrated_pipeline\registered_data\CODEX\Clusters-selected'
    codex_csv = os.path.join(codex_dir, 'cluster_info.csv')
    
    # RF predictions data paths
    rf_dir = r"D:\integrated_pipeline\HSI_data\CODEX_clusters_masks_output_csv"
    ratio_csv = os.path.join(rf_dir, 'masked_ratio_means.csv')
    
    # Output directory
    output_dir = os.path.join(rf_dir, 'ratio_umap_overlay')
    
    print("=" * 70)
    print("Ratio UMAP Overlay Creator")
    print("=" * 70)
    print(f"CODEX cluster data: {codex_csv}")
    print(f"RF ratio data: {ratio_csv}")
    print(f"Output directory: {output_dir}")
    
    # Verify files exist
    if not os.path.exists(codex_csv):
        print(f"ERROR: CODEX CSV not found at {codex_csv}")
        return
    
    if not os.path.exists(ratio_csv):
        print(f"ERROR: Ratio CSV not found at {ratio_csv}")
        return
    
    # Load data
    print("\n" + "=" * 70)
    print("Loading data...")
    print("=" * 70)
    
    codex_df = pd.read_csv(codex_csv)
    ratio_df = pd.read_csv(ratio_csv)
    
    print(f"CODEX data shape: {codex_df.shape}")
    print(f"Ratio data shape: {ratio_df.shape}")
    
    # Optional: Define custom colormap ranges per ratio type
    # Example: {"lipid_unsaturation": [0, 1], "lipid_to_protein": None, "redox": None}
    cmap_ranges = {
        "lipid_unsaturation": [0, 0.6],  # Custom range: 0 to 1
        "lipid_to_protein": [0, 0.75],  # Custom range: 0 to 1
        "redox": [0.1, 0.5],  # Custom range: 0.15 to 0.55
    }

    # Selection filters:
    # 1. Provide specific FastPG indices (0-indexed integers) in selected_fastpg_values.
    #    Example: [0, 5, 12] or None to use name-based filtering.
    selected_fastpg_values = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        20,
        21,
        22,
        25,
        26
    ]
    
    
    # 2. Provide name terms in selected_label_names.
    #    Example: ["Glomeruli", "Tubule"] or None for all.
    selected_label_names = None

    # Create plots
    create_ratio_umap_plots(
        codex_df,
        ratio_df,
        output_dir,
        codex_dir,
        cmap_ranges=cmap_ranges,
        selected_label_names=selected_label_names,
        selected_clusters=selected_fastpg_values
    )


if __name__ == "__main__":
    main()
