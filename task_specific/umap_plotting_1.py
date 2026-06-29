import os 
import sys
from pathlib import Path

# Add parent directory to path so we can import from core/
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np 
import pandas as pd
from tqdm import tqdm

import umap 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import seaborn as sns
import colorsys
from difflib import SequenceMatcher
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


 

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





def create_ftu_based_cluster_colors(df, labels, label_names, ftu_label_names):
    """
    Assigns colors to FastPG clusters based on their dominant FTU group.
    Each FTU group gets a distinct base hue. Clusters belonging to the same
    FTU group share that hue but vary in lightness.

    Args:
        df (pd.DataFrame): DataFrame with 'FastPG' and 'FTUgroup' columns.
        labels (array-like): Remapped FastPG labels (after sorting).
        label_names (dict): {remapped_idx: name_string} for FastPG clusters.
        ftu_label_names (dict): {ftu_idx: ftu_name} for FTU groups.
    Returns:
        cluster_label_to_color (dict): Mapping of remapped cluster idx -> RGB color.
        ftu_label_to_color (dict): Mapping of FTU group idx -> RGB base color.
        cluster_to_ftu (dict): Mapping of remapped cluster idx -> dominant FTU group idx.
    """
    # 1) Assign distinct base hues to FTU groups (exclude -1/Unknown)
    known_ftu = {k: v for k, v in ftu_label_names.items() if k != -1}

    # Fixed hues derived from old_color_mappings.txt FTU base colors
    fixed_hues = {
        1: 0.020,   # Glomeruli        → red-orange   (#e8260c)
        2: 0.350,   # Proximal Tubule   → green        (#0fbc20)
        3: 0.100,   # Thin Desc. Limbs  → orange       (#f89706)
        4: 0.880,   # Thick Asc. Limbs  → magenta-pink (#f20cb1)
        5: 0.150,   # Distal Tubule     → yellow       (#ebd409)
        6: 0.600,   # Collecting Duct   → cornfl. blue (#0b5eda)
        7: 0.500,   # Vasculature       → cyan         (#0cc9c9)
    }

    # Remaining FTU groups get golden-ratio-spaced hues seeded from the
    # largest gap among the fixed hues
    golden_ratio = 0.618033988749895
    assigned_hues = sorted(fixed_hues.values())
    # Find largest gap on the circular hue wheel
    gaps = [(assigned_hues[(i + 1) % len(assigned_hues)] - assigned_hues[i]) % 1.0
            for i in range(len(assigned_hues))]
    best_gap_idx = int(np.argmax(gaps))
    gap_start = assigned_hues[best_gap_idx]
    gap_size = gaps[best_gap_idx]

    unassigned = sorted(k for k in known_ftu.keys() if k not in fixed_hues)
    ftu_hues = dict(fixed_hues)
    for i, ftu_idx in enumerate(unassigned):
        ftu_hues[ftu_idx] = (gap_start + gap_size * (i + 1) / (len(unassigned) + 1)) % 1.0

    ftu_label_to_color = {-1: "#808080"}
    for ftu_idx, hue in ftu_hues.items():
        ftu_label_to_color[ftu_idx] = colorsys.hls_to_rgb(hue, 0.5, 0.85)

    # 2) Map each remapped FastPG cluster to its dominant FTU group
    labels_arr = np.array(labels)
    ftu_values = df['FTUgroup'].values
    ftu_values = np.where(np.isnan(ftu_values), -1, ftu_values)

    unique_clusters = sorted(set(labels_arr))
    cluster_to_ftu = {}
    for cluster_idx in unique_clusters:
        mask = labels_arr == cluster_idx
        ftu_for_cluster = ftu_values[mask]
        # Most common FTU group for this cluster
        cluster_to_ftu[cluster_idx] = int(pd.Series(ftu_for_cluster).mode()[0])

    # 3) Group clusters by FTU group, assign colors with varied lightness
    ftu_to_clusters = {}
    for cluster_idx, ftu_idx in cluster_to_ftu.items():
        ftu_to_clusters.setdefault(ftu_idx, []).append(cluster_idx)

    # Keyword → (hue, saturation) for named 'Other' sub-groups
    OTHER_SUBGROUP_HUES = [
        ("immune",        0.056, 0.95),  # orange  
        ("interstitial",  0.770, 0.85),  # purple          
    ]

    cluster_label_to_color = {}
    for ftu_idx, cluster_list in ftu_to_clusters.items():
        if ftu_idx == -1:
            # 1) Imaging artifacts → gray
            artifact_clusters = [
                cidx for cidx in cluster_list
                if "artifact" in label_names.get(cidx, "").lower()
            ]
            for cidx in artifact_clusters:
                cluster_label_to_color[cidx] = (0.5, 0.5, 0.5)
            cluster_list = [c for c in cluster_list if c not in artifact_clusters]
            if not cluster_list:
                continue

            # 2) Keyword-based sub-groups
            subgroups = {kw: [] for kw, _, _ in OTHER_SUBGROUP_HUES}
            subgroups["_misc"] = []
            for cidx in sorted(cluster_list):
                name = label_names.get(cidx, "").lower()
                matched = False
                for kw, _, _ in OTHER_SUBGROUP_HUES:
                    if kw in name:
                        subgroups[kw].append(cidx)
                        matched = True
                        break
                if not matched:
                    subgroups["_misc"].append(cidx)

            for kw, hue, sat in OTHER_SUBGROUP_HUES:
                members = subgroups[kw]
                n_m = len(members)
                for m_i, cidx in enumerate(sorted(members)):
                    lightness = 0.5 if n_m == 1 else 0.28 + 0.44 * (m_i / (n_m - 1))
                    cluster_label_to_color[cidx] = colorsys.hls_to_rgb(hue, lightness, sat)

            # Misc clusters get evenly spaced hues in the largest wheel gap
            misc = subgroups["_misc"]
            for m_i, cidx in enumerate(sorted(misc)):
                hue = (0.42 + m_i * 0.618033988749895) % 1.0
                cluster_label_to_color[cidx] = colorsys.hls_to_rgb(hue, 0.5, 0.80)
            continue
        hue = ftu_hues[ftu_idx]
        n_members = len(cluster_list)
        for i, cluster_idx in enumerate(sorted(cluster_list)):
            lightness = 0.5 if n_members == 1 else 0.28 + 0.44 * (i / (n_members - 1))
            rgb = colorsys.hls_to_rgb(hue, lightness, 0.85)
            cluster_label_to_color[cluster_idx] = rgb

    return cluster_label_to_color, ftu_label_to_color, cluster_to_ftu

def plot_umap(embeddings, labels, colors, label_names=None, save_path=None,
             cluster_to_ftu=None, ftu_label_names=None):
    """
    Plots UMAP embeddings with color-coded labels.

    Args:
        embeddings (np.ndarray): 2D array of shape (n_samples, n_features) containing the UMAP embeddings.
        labels (np.ndarray): 1D array of shape (n_samples,) containing the class labels for each sample.
        colors (dict): Dictionary mapping label values to colors for plotting.
        label_names (dict, optional): Dictionary mapping label values to human-readable names. If None, labels will be used as-is.
        save_path (str, optional): Path to save the plot. If None, the plot will be displayed instead.
        cluster_to_ftu (dict, optional): Mapping of cluster idx -> FTU group idx for legend grouping.
        ftu_label_names (dict, optional): Mapping of FTU group idx -> FTU name for section headers.

    Returns:
        None
    """
    # Assign name to the labels for better visualization
    unique_labels = np.unique(labels)
    if label_names is None:
        label_names = {label: f"Class {label}" for label in unique_labels}

    # Create a list of colors for each label in the embeddings
    colors_list = [colors[label] for label in tqdm(labels)]
    
    # --- Journal-quality figure setup (compact, readable, matches ratio_umap_overlay style) ---
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.scatter(embeddings[:, 0], embeddings[:, 1], c=colors_list, alpha=0.7, s=20, 
               edgecolors='black', linewidth=0.5, rasterized=True)
    
    # Add cluster index labels at the centroid of each cluster
    import matplotlib.patheffects as pe
    for label in unique_labels:
        mask = labels == label
        centroid_x = np.median(embeddings[mask, 0])
        centroid_y = np.median(embeddings[mask, 1])
        ax.text(centroid_x, centroid_y, str(int(label)+1), fontsize=9, fontweight='bold',
                ha='center', va='center', color='black',
                path_effects=[pe.withStroke(linewidth=2.5, foreground='white')],
                zorder=100)

    # Build legend handles, grouped by FTU association if available
    if cluster_to_ftu is not None and ftu_label_names is not None:
        # Group clusters by their FTU group
        ftu_to_cluster_labels = {}
        for label in unique_labels:
            ftu_idx = cluster_to_ftu.get(label, -1)
            ftu_to_cluster_labels.setdefault(ftu_idx, []).append(label)

        handles = []
        # Sort FTU groups by their lowest cluster number, Unknown (-1) last
        sorted_ftu_keys = sorted(
            ftu_to_cluster_labels.keys(),
            key=lambda k: (k == -1, min(int(l) + 1 for l in ftu_to_cluster_labels[k]))
        )
        for ftu_idx in sorted_ftu_keys:
            ftu_name = ftu_label_names.get(ftu_idx, f"FTU {ftu_idx}")
            # Add a bold section header (invisible patch with bold label)
            ftu_name_math = ftu_name.replace(' ', r'\ ')
            header = mpatches.Patch(color='none', label=f'$\\bf{{{ftu_name_math}}}$')
            handles.append(header)
            # Add cluster handles under this FTU group, sorted numerically
            sorted_members = sorted(ftu_to_cluster_labels[ftu_idx], key=lambda l: int(l) + 1)
            for label in sorted_members:
                label_num = int(label) + 1
                patch = mpatches.Patch(
                    color=colors[label],
                    label=f"  {label_num}: {label_names[label]}"
                )
                handles.append(patch)
    else:
        handles = [
            mpatches.Patch(color=colors[label], label=f"{label_idx+1}: {label_names[label]}")
            for label_idx, label in enumerate(unique_labels)
        ]

    # Ensure no FTU group is split across columns by inserting blank spacers
    ncol = 2
    if cluster_to_ftu is not None and ftu_label_names is not None:
        # Collect handles into group blocks (header + members)
        groups_blocks = []
        current_block = []
        for h in handles:
            # Detect header patches (invisible color='none' patches)
            if h.get_facecolor() == (0.0, 0.0, 0.0, 0.0) and current_block:
                groups_blocks.append(current_block)
                current_block = [h]
            else:
                current_block.append(h)
        if current_block:
            groups_blocks.append(current_block)

        # Greedy assignment: fill column 1 with complete groups,
        # stop when adding the next group would exceed half the total entries
        total_entries = sum(len(b) for b in groups_blocks)
        rows_per_col = -(-total_entries // ncol)  # ceiling division
        col1 = []
        col2_start = 0
        for idx, block in enumerate(groups_blocks):
            if len(col1) + len(block) <= rows_per_col:
                col1.extend(block)
                col2_start = idx + 1
            else:
                break

        # Remaining groups go to column 2
        col2 = []
        for block in groups_blocks[col2_start:]:
            col2.extend(block)

        # Pad column 1 with invisible spacers so both columns have equal rows
        max_rows = max(len(col1), len(col2))
        while len(col1) < max_rows:
            col1.append(mpatches.Patch(color='none', label=' '))
        while len(col2) < max_rows:
            col2.append(mpatches.Patch(color='none', label=' '))

        # Matplotlib ncol=2 fills top-to-bottom, left-to-right
        handles = col1 + col2

    # Journal-quality legend: 2 columns, compact spacing (will be saved separately)
    font_props = fm.FontProperties(weight='bold', size=12)
    legend = ax.legend(
        handles=handles, title="Clusters", loc='center left',
        bbox_to_anchor=(1.01, 0.5), fontsize=11,
        title_fontproperties=font_props,
        ncol=ncol, handlelength=1.0, handletextpad=0.6,
        columnspacing=1.2, labelspacing=0.5,
        frameon=True, edgecolor='0.8', fancybox=False,
    )
    
    # Remove legend from main plot (will save separately)
    legend.remove()

    # ax.set_title('UMAP Plot of Embeddings', fontsize=13, fontweight='bold', pad=15, color='black')
    ax.set_xlabel('UMAP 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('UMAP 2', fontsize=12, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=11, colors='black')
    ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    
    # Save the main plot (without legend)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
        print(f"UMAP plot saved to {save_path}")
        
        # Save legend as a separate PNG file with 2-column compact format
        legend_output_dir = os.path.dirname(save_path)
        legend_output_path = os.path.join(legend_output_dir, "UMAP_legend.png")
        
        # Create a figure with just the legend
        fig_legend = plt.figure(figsize=(4, 4))
        ax_legend = fig_legend.add_subplot(111)
        ax_legend.axis('off')
        
        # Create the legend with 2-column compact style
        legend_handles = list(handles)  # Use the same handles from the main plot
        legend_fig = ax_legend.legend(
            handles=legend_handles, title="Clusters", fontsize=11,
            title_fontproperties=font_props,
            ncol=ncol, handlelength=1.0, handletextpad=0.6,
            columnspacing=1.2, labelspacing=0.5,
            frameon=True, edgecolor='0.8', fancybox=False,
        )
        
        fig_legend.tight_layout()
        fig_legend.savefig(legend_output_path, dpi=300, bbox_inches='tight', transparent=True)
        print(f"Legend saved to {legend_output_path}")
        plt.close(fig_legend)
    else:
        plt.show()
    plt.close(fig)

def main():
    # .csv file with embedding and label columns
    data_dir = r"C:\Users\jvill\Documents\integrated_pipeline\WSI_data\CODEX\clusters-metadata"
    csv_file = "cluster_info.csv"  # Update with your actual file path

    # Load the data
    df = pd.read_csv(os.path.join(data_dir, csv_file))

    # Plot with FastPg labels
    cluster_labels_df = pd.read_excel(os.path.join(data_dir, "ClusterKey.xlsx"))
    label_names_orig = {int(idx): str(cluster_id) + ' | ' + str(reason) for idx, (cluster_id, reason) in enumerate(zip(cluster_labels_df['Cluster ID'], cluster_labels_df['Reason for Call']))}

    # Selection filters:
    # 1. Provide specific FastPG indices (0-indexed integers) in selected_fastpg_values.
    #    Example: [0, 5, 12] or None to use name-based filtering.
    selected_fastpg_values = [
        0,
        1,
        # 2,
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
        # 22,
        25,
        26
    ]
    
    # 2. Provide name terms in selected_label_names.
    #    Example: ["Glomeruli", "Tubule"] or None for all.
    selected_label_names = None

    # Logic: priority to selected_fastpg_values if provided.
    if selected_fastpg_values is not None:
        selected_clusters = [c for c in selected_fastpg_values if c in label_names_orig]
        print(f"Plotting specific FastPG clusters: {selected_clusters}")
    else:
        selected_clusters = select_clusters_by_label_names(label_names_orig, selected_label_names)
        if selected_label_names:
            print(f"Plotting clusters filtered by names: {selected_label_names}")
    if not selected_clusters:
        print(f"No clusters matched selected_label_names={selected_label_names}.")
        return

    df = df[df['FastPG'].isin(selected_clusters)].copy()
    labels_raw = df['FastPG'].values.astype(int)

    # Extract UMAP embeddings and labels after optional filtering
    umap_1 = df['UMAP_1'].values
    umap_2 = df['UMAP_2'].values
    embeddings = np.vstack((umap_1, umap_2))
    print(f"UMAP embeddings shape: {embeddings.shape}")

    # Define FTU group names (needed here for sort order)
    ftu_label_names = {
        -1: "Other",
        1: "Glomeruli",
        2: "Proximal Tubule",
        3: "Thin Descending Limbs",
        4: "Thick Ascending Limbs",
        5: "Distal Tubule",
        6: "Distal Nephron",
        7: "Vasculature",
    }

    # Determine dominant FTU group for each original cluster index
    ftu_values = df['FTUgroup'].values
    ftu_values = np.where(np.isnan(ftu_values), -1, ftu_values)

    orig_cluster_to_ftu = {}
    for orig_idx in selected_clusters:
        mask = labels_raw == orig_idx
        if mask.any():
            orig_cluster_to_ftu[orig_idx] = int(pd.Series(ftu_values[mask]).mode()[0])
        else:
            orig_cluster_to_ftu[orig_idx] = -1

    # Sort clusters by: FTU group order (known first, "Other"/-1 last),
    # then alphabetically by name within each FTU group
    sorted_keys = sorted(
        selected_clusters,
        key=lambda k: (
            orig_cluster_to_ftu[k] == -1,   # Other goes last
            orig_cluster_to_ftu[k],          # then by FTU group index
            label_names_orig[k]              # then alphabetically within group
        )
    )

    # Remap labels: new index is the position in the FTU-sorted order
    old_to_new = {old_key: new_idx for new_idx, old_key in enumerate(sorted_keys)}
    labels = np.array([old_to_new[label] for label in labels_raw], dtype=int)

    label_names = {new_idx: label_names_orig[old_key] for new_idx, old_key in enumerate(sorted_keys)}
    
    # Remove imaging artifacts after label remapping
    # Find the new index for the imaging artifacts cluster
    artifact_new_idx = None
    for new_idx, label_text in label_names.items():
        if "artifact" in label_text.lower():
            artifact_new_idx = new_idx
            print(f"Found imaging artifacts cluster at new index {new_idx}: {label_text}")
            break
    
    # Exclude artifact cluster from plotting only (keep it in mappings/reporting)
    if artifact_new_idx is not None:
        artifact_orig_idx = sorted_keys[artifact_new_idx]
        removed_count = int(np.sum(df['FastPG'].values.astype(int) == artifact_orig_idx))
        print(
            f"Excluding {removed_count} points from imaging artifacts cluster "
            f"(original index {artifact_orig_idx}) in plot only"
        )

    # Create color mappings: FTU groups get distinct hues,
    # FastPG clusters inherit hue from their dominant FTU group
    cluster_label_to_color, ftu_label_to_color, cluster_to_ftu = create_ftu_based_cluster_colors(
        df, labels, label_names, ftu_label_names
    )

    # Plot FastPG UMAP (legend grouped by FTU association)
    # Keep artifact in mappings but exclude its points/legend from the plot.
    labels_plot = labels
    embeddings_plot = embeddings
    if artifact_new_idx is not None:
        plot_mask = labels != artifact_new_idx
        labels_plot = labels[plot_mask]
        embeddings_plot = embeddings[:, plot_mask]

    selection = "all"
    if selected_label_names:
        selection = "_".join([name.replace(" ", "_") for name in selected_label_names])
    plot_umap(embeddings_plot.T, labels_plot, colors=cluster_label_to_color, label_names=label_names,
             save_path=os.path.join(data_dir, f"UMAP_FastPG_{selection}-1.png"),
             cluster_to_ftu=cluster_to_ftu, ftu_label_names=ftu_label_names)

    # --- FTUgroup plot ---
    # ftu_labels = df['FTUgroup'].values
    # if np.isnan(ftu_labels).any():
    #     ftu_labels = np.where(np.isnan(ftu_labels), -1, ftu_labels)

    # print(f"Unique FTU labels found: {np.unique(ftu_labels)}")

    # # Plot FTUgroup UMAP with the base FTU colors
    # plot_umap(embeddings.T, ftu_labels, colors=ftu_label_to_color, label_names=ftu_label_names, save_path=os.path.join(data_dir, "UMAP_FTUgroup.png"))

    # --- Export color mappings to .txt as hex codes ---
    def to_hex(color):
        """Convert an RGB tuple (0-1 floats) or hex string to a hex code."""
        if isinstance(color, str):
            return color if color.startswith('#') else f"#{color}"
        return '#{:02x}{:02x}{:02x}'.format(int(color[0]*255), int(color[1]*255), int(color[2]*255))

    color_file = os.path.join(data_dir, "color_mappings-1.txt")
    with open(color_file, 'w') as f:
        f.write("=== FastPG Cluster Colors ===\n")
        for cluster_idx in sorted(cluster_label_to_color.keys()):
            orig_cluster_num = int(sorted_keys[cluster_idx]+1)
            hex_color = to_hex(cluster_label_to_color[cluster_idx])
            name = label_names.get(cluster_idx, f"Cluster {cluster_idx}")
            ftu_idx = cluster_to_ftu.get(cluster_idx, -1)
            ftu_name = ftu_label_names.get(ftu_idx, "Other")
            f.write(f"Cluster {orig_cluster_num}\t{hex_color}\t{name}\t{ftu_name}\n")

        f.write("\n=== FTU Group Colors ===\n")
        for ftu_idx in sorted(ftu_label_to_color.keys()):
            hex_color = to_hex(ftu_label_to_color[ftu_idx])
            ftu_name = ftu_label_names.get(ftu_idx, f"FTU {ftu_idx}")
            f.write(f"FTU {ftu_idx}\t{hex_color}\t{ftu_name}\n")

    print(f"Color mappings saved to {color_file}")
    print("UMAP plotting completed.")

if __name__ == "__main__":
    main()