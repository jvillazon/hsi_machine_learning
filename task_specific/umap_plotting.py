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

    # Fixed hues for specific FTU groups (HLS hue, 0-1 scale)
    fixed_hues = {
        1: 0.0,     # Glomeruli → red
        2: 0.33,    # Proximal Tubule → light green
        4: 0.833,   # Thick Ascending Limb → magenta
        5: 0.167,   # Distal Tubule → yellow
        6: 0.667,   # Distal Nephron → blue
        7: 0.5,     # Vasculature → cyan
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
        ftu_label_to_color[ftu_idx] = colorsys.hls_to_rgb(hue, 0.5, 0.8)

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

    cluster_label_to_color = {}
    for ftu_idx, cluster_list in ftu_to_clusters.items():
        if ftu_idx == -1:
            # --- Group unknown clusters by label name similarity ---
            known_hue_list = sorted(ftu_hues.values())
            n_unknown = len(cluster_list)
            sorted_unknown = sorted(cluster_list)

            if n_unknown == 1:
                # Single unknown cluster — just pick a gap hue
                hue = ((known_hue_list[0] + known_hue_list[-1]) / 2 + 0.5) % 1 if known_hue_list else 0.0
                cluster_label_to_color[sorted_unknown[0]] = colorsys.hls_to_rgb(hue, 0.5, 0.9)
                continue

            # Extract the cluster ID prefix (before ' | ') for name comparison
            cluster_name_keys = {}
            for cidx in sorted_unknown:
                full_name = label_names.get(cidx, str(cidx))
                # Use text before ' | ' as the comparable cluster ID 
                cluster_name_keys[cidx] = full_name.split(' | ')[0].strip()

            # Build pairwise distance matrix using SequenceMatcher
            names_list = [cluster_name_keys[c] for c in sorted_unknown]
            n = len(names_list)
            dist_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    sim = SequenceMatcher(None, names_list[i], names_list[j]).ratio()
                    dist_matrix[i, j] = 1.0 - sim
                    dist_matrix[j, i] = 1.0 - sim

            # Hierarchical clustering on the distance matrix
            condensed = squareform(dist_matrix)
            Z = linkage(condensed, method='average')
            group_ids = fcluster(Z, t=0.5, criterion='distance')

            # Group unknown clusters by their similarity group
            groups = {}
            for idx, gid in enumerate(group_ids):
                groups.setdefault(gid, []).append(sorted_unknown[idx])

            # Place unknown subgroup hues in the largest gaps between known FTU hues
            known_hue_sorted = sorted(ftu_hues.values())
            n_groups = len(groups)
            # Find all gaps on the circular hue wheel
            all_gaps = []
            for gi in range(len(known_hue_sorted)):
                h1 = known_hue_sorted[gi]
                h2 = known_hue_sorted[(gi + 1) % len(known_hue_sorted)]
                gap_size = (h2 - h1) % 1.0
                all_gaps.append((gap_size, h1))
            # Sort gaps largest-first and distribute unknown groups across them
            all_gaps.sort(key=lambda g: -g[0])
            group_hues = []
            for i in range(n_groups):
                gap_size, gap_start = all_gaps[i % len(all_gaps)]
                slot = (i // len(all_gaps)) + 1
                total_slots = (n_groups // len(all_gaps)) + 2
                group_hues.append((gap_start + gap_size * slot / total_slots) % 1.0)

            # Assign each similarity group a shared hue, vary lightness within
            for g_idx, (gid, members) in enumerate(sorted(groups.items())):
                hue = group_hues[g_idx % len(group_hues)]
                n_members = len(members)
                for m_idx, cluster_idx in enumerate(sorted(members)):
                    lightness = 0.5 if n_members == 1 else 0.35 + 0.3 * (m_idx / (n_members - 1))
                    rgb = colorsys.hls_to_rgb(hue, lightness, 0.85)
                    cluster_label_to_color[cluster_idx] = rgb
            continue
        hue = ftu_hues[ftu_idx]
        n_members = len(cluster_list)
        for i, cluster_idx in enumerate(sorted(cluster_list)):
            lightness = 0.5 if n_members == 1 else 0.35 + 0.3 * (i / (n_members - 1))
            rgb = colorsys.hls_to_rgb(hue, lightness, 0.8)
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
    
    # --- Journal-quality figure setup ---
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.scatter(embeddings[:, 0], embeddings[:, 1], c=colors_list, alpha=0.6, s=0.5, rasterized=True)
    
    # Add cluster index labels at the centroid of each cluster
    import matplotlib.patheffects as pe
    for label in unique_labels:
        mask = labels == label
        centroid_x = np.median(embeddings[mask, 0])
        centroid_y = np.median(embeddings[mask, 1])
        ax.text(centroid_x, centroid_y, str(int(label)+1), fontsize=11, fontweight='bold',
                ha='center', va='center', color='black',
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])

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

    # Journal-quality legend: 2 columns, compact spacing
    font_props = fm.FontProperties(weight='bold', size=18)
    legend = ax.legend(
        handles=handles, title="Clusters", loc='center left',
        bbox_to_anchor=(1.01, 0.5), fontsize=16,
        title_fontproperties=font_props,
        ncol=ncol, handlelength=1.0, handletextpad=0.4,
        columnspacing=1.0, labelspacing=0.3,
        frameon=True, edgecolor='0.8', fancybox=False,
    )


    ax.set_title('UMAP Plot of Embeddings', fontsize=20, fontweight='bold', pad=10)
    ax.set_xlabel('UMAP 1', fontsize=18)
    ax.set_ylabel('UMAP 2', fontsize=18)
    ax.tick_params(axis='both', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"UMAP plot saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)

def main():
    # .csv file with embedding and label columns
    data_dir = r"/Users/jorgevillazon/Documents/files/integrated_pipeline/registered_data/CODEX/Clusters-selected"
    csv_file = "cluster_info.csv"  # Update with your actual file path

    # Load the data
    df = pd.read_csv(os.path.join(data_dir, csv_file))

    # Extract UMAP embeddings and labels
    umap_1 = df['UMAP_1'].values
    umap_2 = df['UMAP_2'].values
    embeddings = np.vstack((umap_1, umap_2))
    
    print(f"UMAP embeddings shape: {embeddings.shape}")

    # Plot with FastPg labels
    cluster_labels_df = pd.read_excel(os.path.join(data_dir, "ClusterKey.xlsx"))
    label_names_orig = {int(idx): str(cluster_id) + ' | ' + str(reason) for idx, (cluster_id, reason) in enumerate(zip(cluster_labels_df['Cluster ID'], cluster_labels_df['Reason for Call']))}

    labels_raw = df['FastPG'].values  # Original FastPG cluster indices

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
    for orig_idx in label_names_orig.keys():
        mask = labels_raw == orig_idx
        if mask.any():
            orig_cluster_to_ftu[orig_idx] = int(pd.Series(ftu_values[mask]).mode()[0])
        else:
            orig_cluster_to_ftu[orig_idx] = -1

    # Sort clusters by: FTU group order (known first, "Other"/-1 last),
    # then alphabetically by name within each FTU group
    sorted_keys = sorted(
        label_names_orig.keys(),
        key=lambda k: (
            orig_cluster_to_ftu[k] == -1,   # Other goes last
            orig_cluster_to_ftu[k],          # then by FTU group index
            label_names_orig[k]              # then alphabetically within group
        )
    )

    # Remap labels: new index is the position in the FTU-sorted order
    old_to_new = {old_key: new_idx for new_idx, old_key in enumerate(sorted_keys)}
    labels = [old_to_new[label] for label in labels_raw]

    label_names = {new_idx: label_names_orig[old_key] for new_idx, old_key in enumerate(sorted_keys)}
    
    if np.isnan(labels).any():
        labels = np.where(np.isnan(labels), -1, labels)

    # Create color mappings: FTU groups get distinct hues,
    # FastPG clusters inherit hue from their dominant FTU group
    cluster_label_to_color, ftu_label_to_color, cluster_to_ftu = create_ftu_based_cluster_colors(
        df, labels, label_names, ftu_label_names
    )

    # Plot FastPG UMAP (legend grouped by FTU association)
    plot_umap(embeddings.T, labels, colors=cluster_label_to_color, label_names=label_names,
             save_path=os.path.join(data_dir, "UMAP_FastPG.png"),
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

    color_file = os.path.join(data_dir, "color_mappings.txt")
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