import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tifffile import tifffile
import itertools
from skimage import measure, io

from statannotations.Annotator import Annotator

data_directory = "/Users/jorgevillazon/Documents/files/codex-srs/HuBMAP .tif files for Jorge Part 1/10-modality_tissue/K21_198_3_CODEX_registration/3712 ROI/"
mol_names = [

]
def load_mol_data(data_path, mask_path):

    mask = tifffile.imread(mask_path)
    labeled_data = tifffile.imread(data_path)

    mask = (mask > 0)#.astype(np.uint8)

    masked_data = labeled_data[mask]

    return masked_data

def main():

    data_folders = glob.glob(os.path.join(data_directory, "*/"))
    data_folder = data_folders[0]

    mol_names = pd.read_csv(os.path.join(data_folder, 'Correlation_Metrics/Metrics.csv'))['Molecule'].to_numpy()[:-1]
    mol_names = np.insert(mol_names, 0, 'Unknown')


    data_dict = {}


    for data_folder in data_folders:

        mask_dir = os.path.join(data_folder, 'CODEX/')
        mask_files = glob.glob(os.path.join(mask_dir, '*-mask.tif'))
        if not mask_files:
            print(f"Mask directory is empty.")
            continue
        data_path = os.path.join(data_folder, 'Correlation_Metrics/Correlation-Image.tif')

        for mask_file in mask_files:
            mask_name = os.path.basename(mask_file).split("-")[1]
            if mask_name not in data_dict:
                data_dict[mask_name] = {}

            mask_path = os.path.join(mask_dir, os.path.basename(mask_file))

            labeled_mask = load_mol_data(data_path, mask_path)

            unique, counts = np.unique(labeled_mask, return_counts=True)
            percents = ((counts/counts.sum())*100)



            for mol_name, percent in zip(mol_names[unique.astype(int)], percents):

                if mol_name not in data_dict[mask_name]:
                    data_dict[mask_name][mol_name] = []
                data_dict[mask_name][mol_name].append(percent)

    data = []
    for mask_name, mol_names in data_dict.items():
        for mol_name, values in mol_names.items():
            for value in values:
                data.append({"FTU": mask_name, "Molecule": mol_name, "Value": value})

    df = pd.DataFrame(data=data)

    plt.figure(figsize=(80, 4))
    ax = sns.boxplot(data=df, x="Molecule", y="Value", hue="FTU")
    pairs = []

    pairs = []
    groups = df['FTU'].unique()
    conditions = df['Molecule'].unique()

    for cond in conditions:
        groups_in_cond = df[df['Molecule'] == cond]['FTU'].unique()
        group_pairs = list(itertools.combinations(groups_in_cond, 2))
        # Add only valid pairs
        pairs += [((cond, g1), (cond, g2)) for g1, g2 in group_pairs]

    annotator = Annotator(ax, pairs, data=df, x="Molecule", y="Value", hue="FTU")
    annotator.configure(test='t-test_ind', text_format='star', loc='inside')
    annotator.apply_and_annotate()
    plt.show()

    print(df)

if __name__ == '__main__':
    main()

    print("done")

