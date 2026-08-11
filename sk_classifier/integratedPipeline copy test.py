
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import sys 


from pathlib import Path

import re
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.hsi_visualizer import HSI_Visualizer

# ------------------------- CONFIG ---------------------------------------
    # <- change me (or set df directly in main())
GROUP_COLS  = ["mask_name"]            # 1 or 2 column names, e.g. ["treatment", "timepoint"]
ITEMS_LIST  = ["gene_A", "gene_B"]     # value columns to test/plot, one per figure
OUTPUT_DIR  = "violin_plots"
ALPHA       = 0.05
COMBINE_TWO_GROUPS = True   # True: merge 2 group cols into one combined x-axis category
                            # False: facet by 2nd group col, separate stats per facet
PALETTE     = "Set2"
MAX_BRACKETS = None         # cap on significance brackets drawn per plot (None = show all).
                            # Useful when COMBINE_TWO_GROUPS creates many categories and
                            # pairwise comparisons get crowded. Shortest (nearest-neighbor)
                            # comparisons are kept first.
FILTER_VALUES = {
    # "treatment": ["Ctrl", "DrugA"],   # keep only these values of a GROUP_COLS column
}


def stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


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

def add_significance_brackets(ax, sig_pairs, order):
    """sig_pairs: list of (group_a, group_b, p_value), already filtered to p < ALPHA."""
    if not sig_pairs:
        return

    ymin, ymax = ax.get_ylim()
    y_range = ymax - ymin
    y_start = ymax + 0.03 * y_range
    h = 0.08 * y_range

    # shorter (adjacent-group) brackets stack first, at lower heights
    sig_pairs = sorted(sig_pairs, key=lambda t: abs(order.index(t[0]) - order.index(t[1])))
    if MAX_BRACKETS is not None:
        sig_pairs = sig_pairs[:MAX_BRACKETS]

    level = 0
    for g1, g2, p in sig_pairs:
        x1, x2 = order.index(g1), order.index(g2)
        y = y_start + level * (h * 1.4)
        ax.plot([x1, x1, x2, x2], [y, y + h * 0.3, y + h * 0.3, y], lw=1.2, c="black")
        ax.text((x1 + x2) / 2, y + h * 0.3, stars(p), ha="center", va="bottom", fontsize=11)
        level += 1

    ax.set_ylim(ymin, y_start + level * (h * 1.4) + h * 2)



def plot_single_grouping(df, value_col, group_order, group_col, ax=None, title=None, alpha=ALPHA, palette=PALETTE):
    sub = df[[value_col, group_col]].dropna()
    # order = sorted(sub[group_col].unique().tolist(), key=str)
    # order = list(dict.fromkeys(sub[group_col].dropna().tolist(), None).keys())  # preserve original order
    order = [g for g in group_order if g in sub[group_col].unique()]

    if ax is None:
        _, ax = plt.subplots(figsize=(max(4, 1.2 * len(order)), 5))

    sns.violinplot(data=sub, x=group_col, y=value_col, order=order,
                    palette=palette, cut=0, inner="box", ax=ax, hue=group_col,
                    legend=False)

    ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    # sns.stripplot(data=sub, x=group_col, y=value_col, order=order,
    #                color="black", alpha=0.4, size=3, jitter=True, ax=ax)

    anova, posthoc = run_welch_games_howell(sub, value_col, group_col, alpha)

    if anova is not None:
        ax.set_title(f"{title or value_col}\nWelch's ANOVA p = {anova['p_unc']:.4g}",
                     fontsize=11)
        if posthoc is not None:
            sig_pairs = [(row["A"], row["B"], row["pval"])
                         for _, row in posthoc.iterrows() if row["pval"] < alpha]
            add_significance_brackets(ax, sig_pairs, order)
    else:
        ax.set_title(f"{title or value_col}\n(not enough groups/data for test)", fontsize=11)

    ax.set_xlabel(group_col)
    ax.set_ylabel(value_col)
    if len(order) > 4:
        ax.tick_params(axis="x", rotation=30)
    return anova, posthoc


# ------------------------------- MAIN --------------------------------------

def run(df, group_cols, items_list, col_order, combine, filter_values, palette, output_dir):
    """Callable directly with an in-memory dataframe (skips DATA_PATH read)."""
    for col, keep_values in filter_values.items():
        df = df[df[col].isin(keep_values)]

    anova_records = []
    posthoc_records = []

    for item in items_list:
        item_name = re.sub(r"[^\w\s]", "", item) 
        if len(group_cols) == 1:
            group_col = group_cols[0]
            fig, ax = plt.subplots(figsize=(max(4, 1.2 * len(col_order)), 5))
            anova, posthoc = plot_single_grouping(df, item, next(iter(col_order)), group_col, ax=ax, palette=palette)
            fig.tight_layout()
            
            fig.savefig(os.path.join(output_dir, f"{item_name}.png"), dpi=200)
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
                fig, ax = plt.subplots(figsize=(max(4, 1.2 * len(order)), 5))
                anova, posthoc = plot_single_grouping(d, item, order, combo_col, ax=ax,
                                                       title=f"{item} by {g1} & {g2}", palette=palette)
                fig.tight_layout()
                fig.savefig(os.path.join(output_dir, f"{item_name}.png"), dpi=200)
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
                # facets = sorted(df[g2].dropna().unique().tolist(), key=str)
            
                facets = col_order.get(g2, [])  # preserve original order
                for fval in facets:
                    if df[df[g2] == fval].empty:
                        print(f"Warning: no data for {g2}={fval}, skipping facet.")
                        facets.remove(fval)
                fig, axes = plt.subplots(1, len(facets),
                                          figsize=(5 * len(facets), 5), sharey=True)
                axes = [axes] if len(facets) == 1 else list(axes)
                for ax, fval in zip(axes, facets):
                    sub= df.loc[df[g2] == fval]
                    a, p = plot_single_grouping(sub, item, col_order.get(g1, []), g1, ax=ax,
                                                 title=f"{item} | {g2}={fval}", palette=palette)
                    if a is not None:
                        anova_records.append({**a, "item": item, "group_by": g1,
                                               "facet": f"{g2}={fval}"})
                    if p is not None:
                        p["item"] = item
                        p["group_by"] = g1
                        p["facet"] = f"{g2}={fval}"
                        posthoc_records.append(p)
                fig.tight_layout()
                
                fig.savefig(os.path.join(output_dir, f"{item_name}.png"), dpi=200)
                plt.close(fig)

        else:
            raise ValueError("group_cols must have length 1 or 2")

    anova_df = pd.DataFrame(
        [{k: v for k, v in rec.items() if k != "labels"} for rec in anova_records]
    )
    posthoc_df = pd.concat(posthoc_records, ignore_index=True) if posthoc_records else pd.DataFrame()

    anova_df.to_csv(os.path.join(output_dir, "welch_anova_results.csv"), index=False)
    posthoc_df.to_csv(os.path.join(output_dir, "games_howell_results.csv"), index=False)
    print(f"Done. Plots and stats saved to '{output_dir}/'")
    return anova_df, posthoc_df


def main():
    path = r"/Users/jorgevillazon/Library/CloudStorage/OneDrive-UCSanDiego/integrated_pipeline/WSI_data/WSI_data/CODEX_clusters-regions_masks_output_csv-test/output.csv"
    df = pd.read_csv(path)
    col_groups = {
        "ORR": ["Redox"],
        "CAR" :[
            "CAR 18:2 (+H)",
            "CAR 18:1 (+H)",
            "CAR 18:0 (+H)"
        ],
        "CL" : [
            "CL 72:8 (-H)", #1447.965 CL 72:8 (-H),
            "CL 74:11 (-H)",# 1469.9493 CL 74:11 (-H)
        ],
        "LPL" : [
            "LPC 16:0 (+H)",
            "LPE 18:0 (-H)",
            "LPI 18:0 (-H)",
        ],
        "PL (Sat)" :[
            "PC 32:0 (+H)",
            "PE 16:0_18:0 (-H)",
            "PI 16:0_18:0 (-H)",
        ],
        "PL (MUFA)":[
            "PC 16:0_18:1 (+H)",
            "PC 18:0_18:1 (+H)",
            "PA 36:1 (-H)",
            "PS 18:0_18:1 (-H)",
            "PI 34:1 (-H)",
            "PI 18:0_18:1 (-H)",   
        ] ,
        "PL (AA)":[
            "PE 16:0_20:4 (-H)",
            "PE 18:1_20:4 (-H)",
            "PE 18:1_20:3 (-H)",
            "PS 16:0_20:4 (-H)",
            "PS 18:0_20:4 (-H)",
            "PI 16:0_20:4 (-H)",
            "PI 18:1_20:3 (-H)",
            "PC 35:4 (+H)",
            "PC 36:5 (+H)",
            "PC 18:0_20:4 (+H)",
        ],
        "PL (DHA)":[
            "PE 16:0_22:6 (-H)",
            "PG 18:1_22:6 (-H)",
        ],
        "PL (PUFA)":[
            "PC O-38:5 (+H)",
            "PC 38:5 (+H)",
            "PC 46:9 (-CH3)",
            "PE 49:9 (-H)",
            "PC 48:9 (-CH3)",
        ],

        "SM":[
            "SM 34:1;2O (+H)",
            "SM 38:1;20 (+H)",
            "SM 42:2;2O (+H)",
            "SM 42:1;2O (+H)",
            "SM 41:0;3O (+H)",
            "SM 42:0;3O (+H)",
            "SM 18:1;2O/16:0 (-CH3)"
        ],
        "SHexCer" : [
            "SHexCer 40:1;3O (-H)",
            "SHexCer 41:1;3O (-H)",
            "SHexCer 42:2;3O (-H)",
            "SHexCer 42:1;3O (-H)",
            "SHexCer 42:0;3O (-H)",
            "SHexCer 50:0;2O (-H)"
        ],
            "Lipid Unsat.": ["Lipid_Unsaturation"],
            "Lipid/Protein" : ["Lipid_to_Protein"],

        "Fib. Collagen": [
            # 'Collagen',
            "Collagen_Thickness",
            "Collagen_Coherence"
        ],

    }
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # df = df[df['Source_ID']=='S21']

    # df = df[df['tissue_id']=='S21-K21_198_3']

    for new_col, source_cols in col_groups.items():
        df[new_col] = df[source_cols].mean(axis=1)

    unit_order = [
        "Cluster1",
        "Cluster2",
        #   "Cluster3",
        "Cluster4",
        "Cluster5",
        "Cluster6",
        "Cluster7",
        "Cluster8",
        "Cluster9",
        #   "Cluster10",
        "Cluster11",
        "Cluster12",
        #   "Cluster13",
        #   "Cluster14",
        "Cluster15",
        "Cluster16",
        "Cluster17",
        "Cluster18",
        "Cluster19",
        "Cluster20",
        "Cluster21",
        "Cluster22",
        "Cluster23",
        "Cluster24",
        "Cluster25",
        "Cluster26"
        ]
    # group_order = ['OC', 'IC', 'CM', 'OS', "IS"]

    group_order = ['Outer Cortex', 'Inner Cortex', 'Outer Stripe', 'Inner Stripe', 'Inner Medulla']
    df = df.replace({'Middle Cortex': 'Outer Cortex', 'Juxtamedullary Cortex': 'Inner Cortex', 'Cluster3':'Cluster4', 'Cluster13': 'Cluster12', 'Cluster14': 'Cluster12', 'Cluster10': 'Cluster9'} )

    # group_map ={
    #     'C1': 'Outer Cortex',
    #     'C2': 'Inner Cortex',
    #     'C3': 'CM Boundary',
    #     'CM': 'Outer Stripe',
    #     'M': 'Inner Stripe'
    # }


    df['mask_name'] = pd.Categorical(df['mask_name'], categories=unit_order, ordered=True)
    df['group'] = pd.Categorical(df['group'], categories=group_order, ordered=True)
    for i, val_col in enumerate(df.columns[7:]):
        df[val_col] = df[val_col].fillna(df.groupby(['group', 'mask_name'])[val_col].transform('median'))

    # df[df.columns[7:]] = df[df.columns[7:]].fillna(0)

    # df['Group_ID'] = df['Group_ID'].map(group_map)
    # df.drop(columns=['Redox', 'Collagen_Coherence', 'Collagen_Thickness'],inplace=True)

    condition_tdl = (df['mask_name'].isin(['Cluster6', 'Cluster7']) & df['group'].isin(['Outer Cortex', 'Inner Cortex']))
    df.loc[condition_tdl, 'group'] = 'Outer Stripe'

    condition_pt = (df['mask_name'].isin(['Cluster2', 'Cluster4', 'Cluster5']) & df['group'].isin(['Inner Stripe']))
    df.loc[condition_pt, 'group'] = 'Outer Stripe'

    condition_tal = (df['mask_name'].isin(['Cluster8', 'Cluster9']) & df['group'].isin(['Inner Medulla']))
    df.loc[condition_tal, 'group'] = 'Inner Stripe'

    condition_glom = (df['mask_name'].isin(['Cluster1']) & df['group'].isin(['Outer Stripe']))
    df.loc[condition_glom, 'group'] = 'Inner Cortex'

    workspace_dir = str(Path.cwd())
    wn_1 =2700
    wn_2 = 3100
    num_samp = 61

    visualizer = HSI_Visualizer(
        mol_path=os.path.join(workspace_dir, 'molecule_dataset', 'lipid_subtype_wn_61_test.npz'),
        wavenumber_start=wn_1, wavenumber_end=wn_2, num_samples=num_samp
    )

    df = df[(df.groupby(['mask_name', 'group'])['mask_name'].transform('size') / df.groupby('mask_name')['mask_name'].transform('size')) * 100 >= 7]
    # df.rename(columns={'mask_name':'Unit', 'group': 'Group_ID'}, inplace=True)

    items_list = list(df.columns[7:])

    unit_colors= {
        "Cluster1": "#eb2d13",
        "Cluster2": "#0a8416",
        "Cluster3": "#10c922",
        "Cluster4": "#13eb28",
        "Cluster5": "#7af487",
        "Cluster6": "#84530a",
        "Cluster7": "#f4c37a",
        "Cluster8": "#840a62",
        "Cluster9": "#f47ad2",
        "Cluster10": "#f47ad2",
        "Cluster11": "#ebd613",
        "Cluster12": "#0a3b84",
        "Cluster13": "#0e52b7",
        "Cluster14": "#0e52b7",
        "Cluster15": "#1369eb",
        "Cluster16": "#478af0",
        "Cluster17": "#7aabf4",
        "Cluster18": "#0a8484",
        "Cluster19": "#10c9c9",
        "Cluster20": "#35eeee",
        "Cluster21": "#7af4f4",
        "Cluster22": "#8b3103",
        "Cluster23": "#d44a05",
        "Cluster24": "#f9702a",
        "Cluster25": "#fba173",
        "Cluster26": "#9913eb"
    }
    display_units_boxplots = [
        "Cluster8",
        "Cluster9",
    ]

    unit_display_names = {
        "Cluster1": "Glomeruli | Podo, CD31",
        "Cluster2": "PT | LRP2",
        "Cluster3": "PT | LRP2, AQP1, CD90",
        "Cluster4": "PT | THY1",
        "Cluster5": "PT | PROM1/VCAM1",
        "Cluster6": "TDL | AQP1, Cyto8 (N1)",
        "Cluster7": "TDL | AQP1, Cyto8 (N2)",
        "Cluster8": "TAL | NaK/UMOD",
        "Cluster9": "TAL | UMOD",
        "Cluster10": "TAL | UMOD",
        "Cluster11": "DCT | SLC12a3",
        "Cluster12": "DN | Cyto8 (N1)",
        "Cluster13": "DN | Cyto8",
        "Cluster14": "DN | Cyto8 (N3)",
        "Cluster15": "DN | Cyto8, Bcat",
        "Cluster16": "DN | Cyto8, Bcat, Ecad",
        "Cluster17": "DN | Cyto8, Ecad",
        "Cluster18": "Endothelium | CD31",
        "Cluster19": "Endothelium | CD31, Podo",
        "Cluster20": "Pericytes | CD90, CD31(-)",
        "Cluster21": "VSMCs | aSMA",
        "Cluster22": "Immune | CD20",
        "Cluster23": "Immune - MP | CD45, CD206",
        "Cluster24": "Immune - T | CD3, CD4",
        "Cluster25": "Immune - T | CD3, CD8",
        "Cluster26": "Interstitial cells | IGFBP7"
    }

    plot_df = df[(df[items_list] != 0).all(axis=1)].copy()

    scaler = StandardScaler()
    plot_df[items_list]= scaler.fit_transform(plot_df[items_list])

    # visualizer.create_individual_barplots(
    #     df=plot_df, 
    #     value_col=None,        # Not needed for wide format
    #     grouping_col=None,     # Signal that we are using Wide format
    #     items_list=items_list, # The column names
    #     output_dir=output_dir,
    #     data_type='Ratio',
    #     units_to_display=display_units_boxplots,
    #     consolidate_Group_IDs=False,
    #     plot_type='violin',
    #     sig_style='stars_only' if len(display_units_boxplots) > 6 else 'brackets',
    #     show_plots=False,
    #     display_name_map=unit_display_names,
    #     unit_color_map=unit_colors,
    #     unit_display_map=unit_display_names,
    #     subgroup_list = group_order,
    #     remove_outliers=True,
    #     show_scatter=True,
    #     show_empty_subgroups=True,
    #     orientation='vertical'
    # )

    # items_list = ['ORR']
    output_dir = os.path.join(workspace_dir, 'group_plots', 'dn')
    os.makedirs(output_dir,exist_ok=True)
    group_cols = ['group', 'mask_name']
    filter_cluster = [
        # "Cluster1",
        # "Cluster2",
        # #   "Cluster3",
        # "Cluster4",
        # "Cluster5",
        # "Cluster6",
        # "Cluster7",
        # "Cluster8",
        # "Cluster9",
        # #   "Cluster10",
        # "Cluster11",
        # "Cluster12",
        #   "Cluster13",
        #   "Cluster14",
        # "Cluster15",
        # "Cluster16",
        # "Cluster17",
        "Cluster18",
        "Cluster19",
        "Cluster20",
        "Cluster21",
        # "Cluster22",
        # "Cluster23",
        # "Cluster24",
        # "Cluster25",
        # "Cluster26"
        ]
    filter_units = [unit_display_names.get(c, c) for c in filter_cluster]

    order = {
        'group': group_order,
        'mask_name': filter_cluster
    }

    df = plot_df
    run(df, group_cols=group_cols, items_list=items_list, col_order=order, combine=False, filter_values={'mask_name':filter_cluster},palette=PALETTE, output_dir=output_dir)

if __name__ == "__main__":
    main()

