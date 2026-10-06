
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as PathEffects
import seaborn as sns
import pingouin as pg
from sklearn.preprocessing import StandardScaler
import sys 


from pathlib import Path

import re
sys.path.insert(0, str(Path(__file__).parent.parent))

# ------------------------- CONFIG ---------------------------------------
    # <- change me (or set df directly in main())
GROUP_COLS  = ["unit"]            # 1 or 2 column names, e.g. ["treatment", "timepoint"]
ALPHA       = 0.05
COMBINE_TWO_GROUPS = False   # True: merge 2 group cols into one combined x-axis category
                            # False: facet by 2nd group col, separate stats per facet
PALETTE     = "colorblind"
STRIP_HUE_COL = "Sample ID" # Column to color individual strip plot dots by (e.g. "Sample ID", None)
STRIP_PALETTE = "tab20"     # Palette for strip plot dots when STRIP_HUE_COL is used
SAVE_STRIP_LEGEND = True    # Save a separate image containing the legend for STRIP_HUE_COL
MAX_BRACKETS = None         # cap on significance brackets drawn per plot (None = show all).
                            # Useful when COMBINE_TWO_GROUPS creates many categories and
                            # pairwise comparisons get crowded. Shortest (nearest-neighbor)
                            # comparisons are kept first.
FILTER_VALUES = {
    # "treatment": ["Ctrl", "DrugA"],   # keep only these values of a GROUP_COLS column
}

FONT_SCALE=2

def stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def format_p_value(p):
    if pd.isna(p):
        return ""
    if p == 0 or p < 1e-15:
        return "< 1e-15"
    if p < 0.001:
        return f"{p:.4e}"
    return f"{p:.4f}"


def get_legend_handles(order, palette):
    """Generates patch handles for a custom legend matching the palette colors."""
    handles = []
    for i, cat in enumerate(order):
        if isinstance(palette, dict):
            color = palette.get(cat, "#333333")
        elif isinstance(palette, (list, tuple)):
            color = palette[i % len(palette)]
        else:
            color = sns.color_palette(palette, len(order))[i]
        handles.append(mpatches.Patch(facecolor=color, edgecolor='black', linewidth=0.8, label=str(cat)))
    return handles


def save_strip_legend(order, palette_dict, title, output_dir):
    """Generates and saves a clean separate image containing only the legend."""
    n_items = len(order)
    ncols = min(8, max(1, int(np.ceil(np.sqrt(n_items)))))
    nrows = int(np.ceil(n_items / ncols))
    
    fig, ax = plt.subplots(figsize=(max(3.5, 1.3 * ncols), max(1.2, 0.45 * nrows + 0.6)))
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette_dict.get(val, '#555555'),
                   markeredgecolor='black', markeredgewidth=0.8, markersize=10, label=f"Sample {val}" if isinstance(val, (int, np.integer)) and "sample" not in str(title).lower() else str(val))
        for val in order
    ]
    ax.legend(handles=handles, title=title, loc='center', ncol=ncols,
              frameon=True, fontsize=10, title_fontsize=11, framealpha=0.9)
    ax.axis('off')
    fig.tight_layout()
    clean_title = re.sub(r"[^\w\s]", "", str(title)).strip().replace(" ", "_")
    legend_filename = f"legend_{clean_title}.png"
    legend_path = os.path.join(output_dir, legend_filename)
    fig.savefig(legend_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig)
    print(f"Saved separate legend to '{legend_path}'")


# ----------------------------- STATS -------------------------------------

def run_welch_games_howell(data, value_col, group_col, alpha=ALPHA):
    """Runs Welch's ANOVA (pingouin) then Games-Howell post-hoc if significant.
    Returns (anova_dict_or_None, posthoc_df_or_None). posthoc_df has pingouin's
    native columns: A, B, mean(A), mean(B), diff, se, T, df, pval, hedges."""
    sub = data[[value_col, group_col]].dropna()
    counts = sub[group_col].value_counts()
    valid_groups = counts[counts >= 2].index
    sub = sub[sub[group_col].isin(valid_groups)]
    if sub[group_col].nunique() < 2:
        return None, None

    aov = pg.welch_anova(dv=value_col, between=group_col, data=sub)
    row = aov.iloc[0]
    anova = {"F": row["F"], "df1": row["ddof1"], "df2": row["ddof2"],
             "p_unc": row["p_unc"], "k": sub[group_col].nunique()}

    posthoc = None
    if anova["p_unc"] < alpha:
        posthoc = pg.pairwise_gameshowell(dv=value_col, between=group_col, data=sub)
    return anova, posthoc


# ----------------------------- PLOTTING -----------------------------------

def add_significance_brackets(ax, sig_pairs, order, initial_ymin=None, initial_ymax=None):
    """sig_pairs: list of (group_a, group_b, p_value), already filtered to p < ALPHA.
    Returns the maximum number of levels used."""
    if not sig_pairs:
        return 0

    ymin = initial_ymin if initial_ymin is not None else ax.get_ylim()[0]
    ymax = initial_ymax if initial_ymax is not None else ax.get_ylim()[1]
    y_range = ymax - ymin
    y_start = ymax + 0.03 * y_range
    h = 0.08 * y_range

    # shorter (adjacent-group) brackets stack first, at lower heights
    sig_pairs = sorted(sig_pairs, key=lambda t: abs(order.index(t[0]) - order.index(t[1])))
    if MAX_BRACKETS is not None:
        if MAX_BRACKETS <= 0:
            return 0
        sig_pairs = sig_pairs[:MAX_BRACKETS]

    levels_occupied = []  # list of lists of (left, right) intervals at each level
    
    max_level_used = 0
    for g1, g2, p in sig_pairs:
        x1, x2 = order.index(g1), order.index(g2)
        left, right = min(x1, x2), max(x1, x2)
        
        assigned_level = 0
        while True:
            if assigned_level >= len(levels_occupied):
                levels_occupied.append([])
            
            overlap = False
            for existing_left, existing_right in levels_occupied[assigned_level]:
                if existing_left <= right and left <= existing_right:
                    overlap = True
                    break
            
            if not overlap:
                levels_occupied[assigned_level].append((left, right))
                break
            assigned_level += 1
            
        y = y_start + assigned_level * (h * 1.4)
        star_fs = max(8, int(10 * FONT_SCALE))
        ax.plot([x1, x1, x2, x2], [y, y + h * 0.3, y + h * 0.3, y], lw=1.2, c="black")
        ax.text((x1 + x2) / 2, y + h * 0.3, stars(p), ha="center", va="bottom", fontsize=star_fs)
        max_level_used = max(max_level_used, assigned_level + 1)

    if initial_ymin is None and initial_ymax is None:
        ax.set_ylim(ymin, y_start + max_level_used * (h * 1.4) + h * 2)
        
    return max_level_used



def plot_single_grouping(df, value_col, group_order, group_col, ax=None, title=None, alpha=ALPHA, palette=PALETTE, add_brackets=True, strip_hue_col=None, strip_order=None, strip_palette_dict=None):
    cols_to_keep = [value_col, group_col]
    if strip_hue_col and strip_hue_col in df.columns and strip_hue_col not in cols_to_keep:
        cols_to_keep.append(strip_hue_col)
    sub = df[cols_to_keep].dropna(subset=[value_col, group_col])
    order = group_order

    title_fs = max(9, int(12 * FONT_SCALE))
    label_fs = max(9, int(11 * FONT_SCALE))
    tick_fs  = max(6, int(8 * FONT_SCALE))
    text_fs  = max(7, int(9 * FONT_SCALE))

    if isinstance(palette, str):
        palette_dict = {cat: color for cat, color in zip(order, sns.color_palette(palette, len(order)))}
    elif isinstance(palette, (list, tuple)):
        palette_dict = {cat: palette[i % len(palette)] for i, cat in enumerate(order)}
    elif isinstance(palette, dict):
        palette_dict = palette
    else:
        palette_dict = palette

    if ax is None:
        _, ax = plt.subplots(figsize=(max(4, 1.4 * len(order)), 5))

    sns.violinplot(data=sub, x=group_col, y=value_col, order=order,
                    hue=group_col, hue_order=order, palette=palette_dict,
                    cut=0, inner="box", ax=ax, dodge=False, legend=False)

    if strip_hue_col and strip_hue_col in sub.columns:
        s_order = strip_order if strip_order is not None else sorted(sub[strip_hue_col].dropna().unique())
        s_pal = strip_palette_dict if strip_palette_dict is not None else {
            val: col for val, col in zip(s_order, sns.color_palette(STRIP_PALETTE if "STRIP_PALETTE" in globals() else "tab20", len(s_order)))
        }
        sns.stripplot(data=sub, x=group_col, y=value_col, order=order,
                      hue=strip_hue_col, hue_order=s_order, palette=s_pal,
                      dodge=False, jitter=0.2, size=4, alpha=0.75,
                      edgecolor="black", linewidth=0.5, ax=ax, legend=False, zorder=2)
    else:
        sns.stripplot(data=sub, x=group_col, y=value_col, order=order,
                      hue=group_col, hue_order=order, palette=palette_dict,
                      dodge=False, jitter=0.2, size=4, alpha=0.6,
                      edgecolor="black", linewidth=0.5, ax=ax, legend=False, zorder=2)

    # Overlay the mean value as a white point with black border
    for idx, category in enumerate(order):
        cat_data = sub.loc[sub[group_col] == category, value_col]
        if len(cat_data) > 0:
            mean_val = cat_data.mean()
            ax.plot(idx, mean_val, marker='o', color='white', markeredgecolor='black', 
                    markersize=6, zorder=4, label='mean' if idx == 0 else "")
            
            # Use a constant offset relative to the y-axis range instead of a percentage of the mean value
            ymin, ymax = ax.get_ylim()
            y_offset = 0.02 * (ymax - ymin)
            text = ax.text(idx, mean_val + y_offset, f'{mean_val:.2f}', ha='center', va='bottom', fontsize=text_fs, zorder=4)
            text.set_path_effects([
                PathEffects.withStroke(linewidth=5, foreground='white')
            ])
    ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.7, zorder=0)
    # ax.set_axisbelow(True)

    anova, posthoc = run_welch_games_howell(sub, value_col, group_col, alpha)

    if anova is not None:
        ax.set_title(f"{title or value_col}\nWelch's ANOVA p = {anova['p_unc']:.4g}",
                     fontsize=title_fs)
        if posthoc is not None:
            sig_pairs = [(row["A"], row["B"], row["pval"])
                         for _, row in posthoc.iterrows() if row["pval"] < alpha]
            if add_brackets:
                add_significance_brackets(ax, sig_pairs, order)
    else:
        ax.set_title(f"{title or value_col}\n(not enough groups/data for test)", fontsize=title_fs)

    ax.set_xlabel("")
    # ax.set_ylabel(value_col, fontsize=label_fs)
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=tick_fs)
    # ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    return anova, posthoc


# ------------------------------- MAIN --------------------------------------

def run(df, group_cols, value_cols, col_order, combine, filter_values, palette, add_brackets, output_dir, strip_hue_col=STRIP_HUE_COL, strip_palette=STRIP_PALETTE, save_legend=SAVE_STRIP_LEGEND):
    """Callable directly with an in-memory dataframe (skips DATA_PATH read)."""
    for col, keep_values in filter_values.items():
        df = df[df[col].isin(keep_values)]

    if strip_hue_col and strip_hue_col in df.columns:
        try:
            strip_order = sorted(df[strip_hue_col].dropna().unique())
        except TypeError:
            strip_order = list(df[strip_hue_col].dropna().unique())
        pal_name = strip_palette if strip_palette is not None else STRIP_PALETTE
        n_colors = max(1, len(strip_order))
        if isinstance(pal_name, dict):
            strip_palette_dict = pal_name
        else:
            colors = sns.color_palette(pal_name, n_colors)
            strip_palette_dict = {val: col for val, col in zip(strip_order, colors)}
        if save_legend:
            save_strip_legend(strip_order, strip_palette_dict, title=strip_hue_col, output_dir=output_dir)
    else:
        strip_order = None
        strip_palette_dict = None

    anova_records = []
    posthoc_records = []

    for item in value_cols:
        item_name = re.sub(r"[^\w\s]", "", item) 
        if len(group_cols) == 1:
            group_col = group_cols[0]
            fig, ax = plt.subplots(figsize=(max(4, 1.2 * len(col_order[group_col])), 6))
            anova, posthoc = plot_single_grouping(df, item, col_order[group_col], group_col, ax=ax, palette=palette, add_brackets=add_brackets, strip_hue_col=strip_hue_col, strip_order=strip_order, strip_palette_dict=strip_palette_dict)
            fig.tight_layout()
            
            fig.savefig(os.path.join(output_dir, f"{item_name}.png"), dpi=300, transparent=True)
            plt.close(fig)
            if anova is not None:
                anova_records.append({**anova, "item": item, "group_by": group_col,
                                       "facet": None})
            if posthoc is not None:
                posthoc["item"] = item
                posthoc["group_by"] = group_col
                posthoc["facet"] = None
                posthoc_records.append(posthoc)

        elif len(group_cols) == 2:
            g1, g2 = group_cols

            if combine:
                combo_col = f"{g1}__{g2}"
                d = df.copy()
                d[combo_col] = d[g1].astype(str) + " | " + d[g2].astype(str)
                
                order = [f"{g1_val} | {g2_val}" for g1_val in col_order[g1] for g2_val in col_order[g2]]
                fig, ax = plt.subplots(figsize=(max(4, 1.4 * len(order)), 7))
                anova, posthoc = plot_single_grouping(d, item, order, combo_col, ax=ax,
                                                       title=f"{item} by {g1} & {g2}", palette=palette,
                                                       add_brackets=add_brackets, strip_hue_col=strip_hue_col, strip_order=strip_order, strip_palette_dict=strip_palette_dict)
                fig.tight_layout()
                fig.savefig(os.path.join(output_dir, f"{item_name}.png"), dpi=300, transparent=True)
                plt.close(fig)
                if anova is not None:
                    anova_records.append({**anova, "item": item,
                                           "group_by": f"{g1}+{g2} (combined)", "facet": None})
                if posthoc is not None:
                    posthoc["item"] = item
                    posthoc["group_by"] = f"{g1}+{g2} (combined)"
                    posthoc["facet"] = None
                    posthoc_records.append(posthoc)

            else:
                facets = [fval for fval in col_order.get(g2, []) if not df[df[g2] == fval].empty]
                fig, axes = plt.subplots(len(facets), 1,
                                          figsize=(max(6, 1.6 * len(col_order.get(g1, []))), 6 * len(facets)),
                                          sharey=True, sharex=True)
                axes = [axes] if len(facets) == 1 else list(axes)
                
                # 1. Plot all subplots first without brackets to set correct shared y-limits
                temp_results = []
                for ax, fval in zip(axes, facets):
                    sub = df.loc[df[g2] == fval]
                    a, p = plot_single_grouping(sub, item, col_order.get(g1, []), g1, ax=ax,
                                                 title=f"{item} | {g2}={fval}", palette=palette,
                                                 add_brackets=False, strip_hue_col=strip_hue_col, strip_order=strip_order, strip_palette_dict=strip_palette_dict)
                    temp_results.append((ax, a, p, sub))
                    
                    if a is not None:
                        anova_records.append({**a, "item": item, "group_by": g1,
                                               "facet": f"{g2}={fval}"})
                    if p is not None:
                        p["item"] = item
                        p["group_by"] = g1
                        p["facet"] = f"{g2}={fval}"
                        posthoc_records.append(p)
                
                # 2. Get initial shared y-limits
                initial_ymin, initial_ymax = axes[0].get_ylim()
                max_level_overall = 0
                
                # 3. Add brackets to each subplot based on the initial shared y-limits
                if add_brackets:
                    for ax, a, p, sub in temp_results:
                        if p is not None:
                            sig_pairs = [(row["A"], row["B"], row["pval"])
                                         for _, row in p.iterrows() if row["pval"] < ALPHA]
                            levels = add_significance_brackets(ax, sig_pairs, col_order.get(g1, []), 
                                                               initial_ymin=initial_ymin, initial_ymax=initial_ymax)
                            max_level_overall = max(max_level_overall, levels)
                
                # 4. Set final unified y-limits for the shared axis if brackets were drawn
                if max_level_overall > 0:
                    y_range = initial_ymax - initial_ymin
                    y_start = initial_ymax + 0.03 * y_range
                    h = 0.08 * y_range
                    final_ymax = y_start + max_level_overall * (h * 1.4) + h * 2
                    axes[0].set_ylim(initial_ymin, final_ymax)
                
                fig.tight_layout()
                fig.savefig(os.path.join(output_dir, f"{item_name}.png"), dpi=300, transparent=True)
                plt.close(fig)

        else:
            raise ValueError("group_cols must have length 1 or 2")

    anova_df = pd.DataFrame(
        [{k: v for k, v in rec.items() if k != "labels"} for rec in anova_records]
    )
    posthoc_df = pd.concat(posthoc_records, ignore_index=True) if posthoc_records else pd.DataFrame()

    if not anova_df.empty and "p_unc" in anova_df.columns:
        anova_df["p_unc_formatted"] = anova_df["p_unc"].apply(format_p_value)

    if not posthoc_df.empty and "pval" in posthoc_df.columns:
        posthoc_df["pval_formatted"] = posthoc_df["pval"].apply(format_p_value)

    anova_df.to_csv(os.path.join(output_dir, "welch_anova_results.csv"), index=False)
    posthoc_df.to_csv(os.path.join(output_dir, "games_howell_results.csv"), index=False)
    print(f"Done. Plots and stats saved to '{output_dir}/'")
    return anova_df, posthoc_df


def main():
    # path = r"D:\VallonD2O\jorge_kidney_data\headless_srs_outputs\all_images_roi_metrics.csv"
    # path = r"D:\VallonD2O\jorge_kidney_data\headless_srs_outputs\all_images_roi_metrics.csv"
    # df_1 = pd.read_csv(path)
    # second_path = r"D:\VallonD2O\jorge_kidney_data\headless_arithmetic_outputs\roi_metrics.parquet"
    # sglt2i_df = pd.read_parquet(second_path)
    # sglt2i_subset = sglt2i_df[['Image Name', 'ROI Inst', 'Label', 'Protein_Turnover Mean', 'Lipid_Turnover Mean', 'Lipid_to_Protein Mean', 'Lipid_Unsaturation Mean']]
    # df = pd.merge(df_1, sglt2i_subset, on=['Image Name', 'ROI Inst', 'Label'], how='left')


    path = r"C:\Users\jvill\OneDrive - UC San Diego\integrated_pipeline\WSI_data\WSI_data\CODEX_clusters-regions_masks_output_csv-test\output.csv"
    df = pd.read_csv(path)







    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    count = 0
    map = {}
    
    # while count < 17:
    #     if count < 5:
    #         map[count] = "Ctrl/Veh"
    #     elif count < 9:
    #         map[count] = "Ctrl/SGLT2i"
    #     elif count < 13:
    #         map[count] = "WD/Veh"
    #     else:
    #         map[count] = "WD/SGLT2i"
    #     count += 1
    # df['Condition'] = df['Sample ID'].map(map)
    print(df['Condition'].unique())
    # For isolating a specific tissue
    # df.loc[df['tissue_id']== 'S24-K21_273_3', 'Redox'] = np.nan


    # col_groups = {
    #     "ORR": ["Redox"],
    #     "CAR" :[
    #         "CAR 18:2 (+H)",
    #         "CAR 18:1 (+H)",
    #         "CAR 18:0 (+H)"
    #     ],
    #     "CL" : [
    #         "CL 72:8 (-H)", #1447.965 CL 72:8 (-H),
    #         "CL 74:11 (-H)",# 1469.9493 CL 74:11 (-H)
    #     ],
    #     "LPL" : [
    #         "LPC 16:0 (+H)",
    #         "LPE 18:0 (-H)",
    #         "LPI 18:0 (-H)",
    #     ],
    #     "PL (Sat)" :[
    #         "PC 32:0 (+H)",
    #         "PE 16:0_18:0 (-H)",
    #         "PI 16:0_18:0 (-H)",
    #     ],
    #     "PL (MUFA)":[
    #         "PC 16:0_18:1 (+H)",
    #         "PC 18:0_18:1 (+H)",
    #         "PA 36:1 (-H)",
    #         "PS 18:0_18:1 (-H)",
    #         "PI 34:1 (-H)",
    #         "PI 18:0_18:1 (-H)",   
    #     ] ,
    #     "PL (AA)":[
    #         "PE 16:0_20:4 (-H)",
    #         "PE 18:1_20:4 (-H)",
    #         "PE 18:1_20:3 (-H)",
    #         "PS 16:0_20:4 (-H)",
    #         "PS 18:0_20:4 (-H)",
    #         "PI 16:0_20:4 (-H)",
    #         "PI 18:1_20:3 (-H)",
    #         "PC 35:4 (+H)",
    #         "PC 36:5 (+H)",
    #         "PC 18:0_20:4 (+H)",
    #     ],
    #     "PL (DHA)":[
    #         "PE 16:0_22:6 (-H)",
    #         "PG 18:1_22:6 (-H)",
    #     ],
    #     "PL (PUFA)":[
    #         "PC O-38:5 (+H)",
    #         "PC 38:5 (+H)",
    #         "PC 46:9 (-CH3)",
    #         "PE 49:9 (-H)",
    #         "PC 48:9 (-CH3)",
    #     ],

    #     "SM":[
    #         "SM 34:1;2O (+H)",
    #         "SM 38:1;20 (+H)",
    #         "SM 42:2;2O (+H)",
    #         "SM 42:1;2O (+H)",
    #         "SM 41:0;3O (+H)",
    #         "SM 42:0;3O (+H)",
    #         "SM 18:1;2O/16:0 (-CH3)"
    #     ],
    #     "SHexCer" : [
    #         "SHexCer 40:1;3O (-H)",
    #         "SHexCer 41:1;3O (-H)",
    #         "SHexCer 42:2;3O (-H)",
    #         "SHexCer 42:1;3O (-H)",
    #         "SHexCer 42:0;3O (-H)",
    #         "SHexCer 50:0;2O (-H)"
    #     ],
    #         "Lipid Unsat.": ["Lipid_Unsaturation"],
    #         "Lipid/Protein" : ["Lipid_to_Protein"],

    #     "Fib. Collagen": [
    #         # 'Collagen',
    #         "Collagen_Thickness",
    #         "Collagen_Coherence"
    #     ],

    # }
    # for new_col, source_cols in col_groups.items():
    #     df[new_col] = df[source_cols].mean(axis=1)


    # filter_cluster = [
    #     # "Cluster1",
    #     "Cluster2",
    #     #   "Cluster3",
    #     "Cluster4",
    #     "Cluster5",
    #     # "Cluster6",
    #     # "Cluster7",
    #     # "Cluster8",
    #     # "Cluster9",
    #     #   "Cluster10",
    #     # "Cluster11",
    #     # "Cluster12",
    #     #   "Cluster13",
    #     #   "Cluster14",
    #     # "Cluster15",
    #     # "Cluster16",
    #     # "Cluster17",
    #     # "Cluster18",
    #     # "Cluster19",
    #     # "Cluster20",
    #     # "Cluster21",
    #     # "Cluster22",
    #     # "Cluster23",
    #     # "Cluster24",
    #     # "Cluster25",
    #     # "Cluster26"
    #     ]


    # group_order = [
    #     "CFIB",
    #     "CMFIB",
    #     "IMFIB",
    #     "OMFIB",
    #     "pvFIB",
    #     "dFIB",
    #     "dIMFIB",
    #     "dOMFIB",
    #     "CMYOF",
    #     "pvMYOF",
    #     "IMpvMYOF",
    # ]
    # group_order = [
    #     'Outer Cortex', 
    #     'Inner Cortex', 
    #     'Outer Stripe', 
    #     'Inner Stripe', 
    #     'Inner Medulla'
    #     ]

    # Replace mistaken group names
    # df = df.replace({"CMMYOF": "CMYOF", "IMpvFIB": "IMpvMYOF"})

#     unit_clusters = {
#     "Cluster1": "Glomeruli | Podo, CD31",
#     "Cluster2": "PT | LRP2",
#     "Cluster3": "PT | LRP2, AQP1, CD90",
#     "Cluster4": "PT | THY1",
#     "Cluster5": "PT | PROM1/VCAM1",
#     "Cluster6": "TDL | AQP1, Cyto8 (N1)",
#     "Cluster7": "TDL | AQP1, Cyto8 (N2)",
#     "Cluster8": "TAL | NaK/UMOD",
#     "Cluster9": "TAL | UMOD",
#     "Cluster10": "TAL | UMOD",
#     "Cluster11": "DCT | SLC12a3",
#     "Cluster12": "DN | Cyto8",
#     "Cluster13": "DN | Cyto8",
#     "Cluster14": "DN | Cyto8",
#     "Cluster15": "DN | Cyto8, Bcat",
#     "Cluster16": "DN | Cyto8, Bcat, Ecad",
#     "Cluster17": "DN | Cyto8, Ecad",
#     "Cluster18": "Endothelium | CD31",
#     "Cluster19": "Endothelium | CD31, Podo",
#     "Cluster20": "Pericytes | CD90, CD31(-)",
#     "Cluster21": "VSMCs | aSMA",
#     "Cluster22": "Immune | CD20",
#     "Cluster23": "Immune - MP | CD45, CD206",
#     "Cluster24": "Immune - T | CD3, CD4",
#     "Cluster25": "Immune - T | CD3, CD8",
#     "Cluster26": "Interstitial cells | IGFBP7"
# }

    # df = df.replace({
    #     'Middle Cortex': 'Outer Cortex', 
    #     'Juxtamedullary Cortex': 'Inner Cortex', 
    #     'Cluster3':'Cluster4',
    #     unit_clusters.get('Cluster3') : unit_clusters.get('Cluster4'),
    #     'Cluster13': 'Cluster12', 
    #     unit_clusters.get('Cluster13') : unit_clusters.get('Cluster12'),
    #     'Cluster14': 'Cluster12', 
    #     unit_clusters.get('Cluster14') : unit_clusters.get('Cluster12'),
    #     'Cluster10': 'Cluster9', 
    #     unit_clusters.get('Cluster10') : unit_clusters.get('Cluster9'),
    #     'Cluster7':'Cluster6',
    #     unit_clusters.get('Cluster7') : 'TDL | AQP1, Cyto8',
    #     unit_clusters.get('Cluster6') : 'TDL | AQP1, Cyto8',
    #     } 
    # )
    # if 'group' in df.columns and not df['group'].isna().all():
    #     df['group'] = pd.Categorical(df['group'], categories=group_order, ordered=True)
    #     if "Inner Cortex" in df['group'].unique():
    #         unit_size = df.groupby('mask_name')['mask_name'].transform('count')
    #         unit_group_size = df.groupby(['group', 'mask_name'])['mask_name'].transform('count')
    #         df['Percentage'] = (unit_group_size / unit_size) * 100

    #         df = df[df['Percentage'] >= 7].copy()
    #         df = df.drop(columns=['Percentage'])

    # value_cols = df.select_dtypes(include='number').columns.tolist()
    # value_cols = [val_col for val_col in value_cols if 'centroid' not in val_col]


    # df = df[(df[value_cols] != 0).all(axis=1)].copy()
    # df[value_cols] = df.groupby(['Image'])[value_cols].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))
    # df[value_cols] = df[value_cols] - df[value_cols].


            
    # for i, val_col in enumerate(value_cols):
    #     df[val_col] = df[val_col].fillna(df.groupby(['group', 'mask_name', 'tissue_id'])[val_col].transform('median'))
    # if 'group' in df.columns:
    #     if set(['Outer Cortex', 'Inner Cortex', 'Outer Stripe', 'Inner Stripe', 'Inner Medulla']) & set(list(df['group'].unique())):    
    #         condition_tdl = (df['mask_name'].isin(['Cluster6', 'Cluster7']) & df['group'].isin(['Outer Cortex', 'Inner Cortex']))
    #         df.loc[condition_tdl, 'group'] = 'Outer Stripe'

    #         condition_pt = (df['mask_name'].isin(['Cluster2', 'Cluster4', 'Cluster5']) & df['group'].isin(['Inner Stripe']))
    #         df.loc[condition_pt, 'group'] = 'Outer Stripe'

    #         condition_tal = (df['mask_name'].isin(['Cluster8', 'Cluster9']) & df['group'].isin(['Inner Medulla']))
    #         df.loc[condition_tal, 'group'] = 'Inner Stripe'

    #         condition_glom = (df['mask_name'].isin(['Cluster1']) & df['group'].isin(['Outer Stripe']))
    #         df.loc[condition_glom, 'group'] = 'Inner Cortex'


    # cluster_colors= {
    #     "Cluster1": "#eb2d13",
    #     "Cluster2": "#0a8416",
    #     "Cluster3": "#10c922",
    #     "Cluster4": "#13eb28",
    #     "Cluster5": "#7af487",
    #     "Cluster6": "#84530a",
    #     "Cluster7": "#f4c37a",
    #     "Cluster8": "#840a62",
    #     "Cluster9": "#f47ad2",
    #     "Cluster10": "#f47ad2",
    #     "Cluster11": "#ebd613",
    #     "Cluster12": "#0a3b84",
    #     "Cluster13": "#0e52b7",
    #     "Cluster14": "#0e52b7",
    #     "Cluster15": "#1369eb",
    #     "Cluster16": "#478af0",
    #     "Cluster17": "#7aabf4",
    #     "Cluster18": "#0a8484",
    #     "Cluster19": "#10c9c9",
    #     "Cluster20": "#35eeee",
    #     "Cluster21": "#7af4f4",
    #     "Cluster22": "#8b3103",
    #     "Cluster23": "#d44a05",
    #     "Cluster24": "#f9702a",
    #     "Cluster25": "#fba173",
    #     "Cluster26": "#9913eb"
    # }


    # unit_clusters = {
    #     "Cluster1": "Glomeruli | Podo, CD31",
    #     "Cluster2": "PT | LRP2",
    #     "Cluster3": "PT | LRP2, AQP1, CD90",
    #     "Cluster4": "PT | THY1",
    #     "Cluster5": "PT | PROM1/VCAM1",
    #     "Cluster6": "TDL | AQP1, Cyto8",
    #     "Cluster7": "TDL | AQP1, Cyto8",
    #     "Cluster8": "TAL | NaK/UMOD",
    #     "Cluster9": "TAL | UMOD",
    #     "Cluster10": "TAL | UMOD",
    #     "Cluster11": "DCT | SLC12a3",
    #     "Cluster12": "DN | Cyto8",
    #     "Cluster13": "DN | Cyto8",
    #     "Cluster14": "DN | Cyto8",
    #     "Cluster15": "DN | Cyto8, Bcat",
    #     "Cluster16": "DN | Cyto8, Bcat, Ecad",
    #     "Cluster17": "DN | Cyto8, Ecad",
    #     "Cluster18": "Endothelium | CD31",
    #     "Cluster19": "Endothelium | CD31, Podo",
    #     "Cluster20": "Pericytes | CD90, CD31(-)",
    #     "Cluster21": "VSMCs | aSMA",
    #     "Cluster22": "Immune | CD20",
    #     "Cluster23": "Immune - MP | CD45, CD206",
    #     "Cluster24": "Immune - T | CD3, CD4",
    #     "Cluster25": "Immune - T | CD3, CD8",
    #     "Cluster26": "Interstitial cells | IGFBP7"
    # }

    # ftu_map = {
    #     "Cluster1": "GL",
    #     "Cluster2": "PT",
    #     "Cluster3": "PT",
    #     "Cluster4": "PT",
    #     "Cluster5": "PT",
    #     "Cluster6": "TDL",
    #     "Cluster7": "TDL",
    #     "Cluster8": "TAL",
    #     "Cluster9": "TAL",
    #     "Cluster10": "TAL",
    #     "Cluster11": "DCT",
    #     "Cluster12": "DN",
    #     "Cluster13": "DN",
    #     "Cluster14": "DN",
    #     "Cluster15": "DN",
    #     "Cluster16": "DN",
    #     "Cluster17": "DN",
    #     "Cluster18": "Endo",
    #     "Cluster19": "Endo",
    #     "Cluster20": "Pericytes",
    #     "Cluster21": "VSMCs",
    #     "Cluster22": "Immune",
    #     "Cluster23": "MP",
    #     "Cluster24": "T",
    #     "Cluster25": "T",
    #     "Cluster26": "Interstitial"
    # }
    
    # df['FTU'] = df['mask_name'].map(ftu_map)

    # fib_map = {
    #     "CFIB": 'FIB',
    #     "CMFIB": 'FIB',
    #     "IMFIB": 'FIB',
    #     "OMFIB": 'FIB',
    #     "pvFIB": 'FIB',
    #     "dFIB": 'dFIB',
    #     "dIMFIB": 'dFIB',
    #     "dOMFIB": 'dFIB',
    #     "CMYOF": 'MYOF',
    #     "pvMYOF": 'MYOF',
    #     "IMpvMYOF": 'MYOF'
    
    # }
    # df['fib'] = df['group'].map(fib_map)



    # filter_units = [unit_clusters.get(c, c) for c in filter_cluster]
    # cluster_colors = [cluster_colors.get(c, c) for c in filter_cluster]
    # unit_colors = dict(zip(filter_units, cluster_colors))

    
    order = {
        'Condition': group_order,
        # 'mask_name': filter_cluster,
        'Label': unit_order,
        # 'FTU': ['GL', 'PT', 'TDL', 'TAL', 'DCT', 'DN', 'Endo', 'Pericytes', 'VSMCs', 'Immune', 'MP', 'T', 'Interstitial'],  # preserve original order
        # 'fib': ['FIB', 'dFIB', 'MYOF']
    }
    filter_values={
        # 'unit':filter_units, 
        # 'group': ['CFIB', 'CMFIB', 'IMFIB', 'OMFIB', 'pvFIB', 'dFIB', 'dIMFIB', 'dOMFIB']
        'Label':unit_order,
        'Condition': group_order
        }

    output_dir = os.path.join(r'D:\VallonD2O\jorge_kidney_data\headless_srs_outputs\plots', f"{group_cols[0]}_{group_cols[1]}" if len(group_cols)==2 else group_cols[0])
    os.makedirs(output_dir,exist_ok=True)


    add_brackets = True
    run(df, 
    group_cols=group_cols,
    value_cols=value_cols, 
    col_order=order, 
    combine=COMBINE_TWO_GROUPS, 
    filter_values=filter_values,
    palette=PALETTE, 
    add_brackets=add_brackets,
    output_dir=output_dir,
    strip_hue_col=STRIP_HUE_COL,
    strip_palette=STRIP_PALETTE,
    save_legend=SAVE_STRIP_LEGEND)

if __name__ == "__main__":
    main()

