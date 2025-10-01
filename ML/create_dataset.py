import os

# Artificial Data
import numpy as np
import pandas as pd
import unsupervised_scripts

## Initialize start and end of hyperspectral sweep; num_samp is calculated from image stack
wavenum_1 = int(input('Enter first wavenumber/higher wavelength (Default=2700)): ').strip() or '2700')
wavenum_2 = int(input('Enter last wavenumber/lower wavelength (Default=3100): ').strip() or '3100')
num_samp = int(input('Enter HSI step size: ').strip())
ch_start = int(np.floor(num_samp/((wavenum_2-wavenum_1)/(2800-wavenum_1))))
background_df = pd.read_csv('water_HSI_76.csv')
molecule_df = pd.read_excel('lipid_subtype.xlsx')
shift = 20

## Training Directory (different for everyone)
data_dir = os.getcwd()+"/"
sample_dir = data_dir + 'training_data/'+str(num_samp)+'-'+str(wavenum_1)+'-'+str(wavenum_2)+'/'

artificial_dataset = unsupervised_scripts.artificial_dataset(wavenum_1, wavenum_2, num_samp, ch_start, background_df, shift)
[_, noise_scale_vec, bg_scale_vec, ratio_scale_vec] = artificial_dataset.save_srs_params(sample_dir)
[mol_norm, mol_names] = artificial_dataset.molecule_dataset(molecule_df)
artificial_dataset.save_artificial_dataset(mol_norm, noise_scale_vec, bg_scale_vec, ratio_scale_vec, noise_param=np.median(noise_scale_vec), ch_param=1, num_values=1000, num_repeats=10, name='artificial_training_data-')