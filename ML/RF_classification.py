#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io
from scipy.signal import savgol_filter
import random
from joblib import load
import helper_scripts
import unsupervised_scripts


## Sample Directory (different for everyone)

data_dir = (input('Enter the path to your .tif hyperstacks: ').strip() or os.getcwd())
save_dir = (input('Enter the path to your save directory: ').strip() or os.getcwd())

## Load Image
from tkinter import*
root = Tk()
root.withdraw()
root.call('wm', 'attributes', '.', '-topmost', True)

from tkinter import filedialog
sample = filedialog.askopenfilename(initialdir=data_dir, multiple=True)
image = io.imread(sample[0])

## Vectorize Image
image_vec = np.reshape(image, (image.shape[0], image.shape[1]*image.shape[2]))
image_vec = image_vec.T

## Remove NaNs and inf
image[np.isinf(image)] = 0
image[np.isnan(image)] = 0

## Initialize start and end of hyperspectral sweep; num_samp is calculated from image stack
wavenum_1 = int(input('Enter first wavenumber/higher wavelength (Default=2750)): ').strip() or '2750')
wavenum_2 = int(input('Enter last wavenumber/lower wavelength (Default=3100): ').strip() or '3100')
num_samp = int(image_vec.shape[1]) 
ch_start = int(input('Enter stack index for 2800 cm^-1 (Default="auto"): ').strip() or str(int(np.floor(num_samp/((wavenum_2-wavenum_1)/(2800-wavenum_1))))))
background_df = pd.read_csv('water_HSI_76.csv')
molecule_df = pd.read_excel('lipid_subtype.xlsx')

### Semi-Supervised Learning

## Load artificial dataset
artificial_data = unsupervised_scripts.artificial_dataset(wavenum_1, wavenum_2, num_samp, ch_start, background_df)
[mol_norm, mol_names] = artificial_data.molecule_dataset(molecule_df)

X_data = np.load('artificial_training_data-28.npy')
Y_data = np.tile(np.array(range(mol_names.shape[0])), (X_data.shape[0], 1))
X = np.reshape(X_data, (X_data.shape[0] * X_data.shape[1], X_data.shape[2]))
Y = np.reshape(Y_data, (Y_data.shape[0] * Y_data.shape[1]))

## Process both datasets
preprocessing = unsupervised_scripts.preprocessing(wavenum_1, wavenum_2, num_samp, ch_start, background_df)
X_standard = preprocessing.spectral_standardization(np.flip(X,axis=1))
image_standard = preprocessing.spectral_standardization(np.flip(image_vec,axis=1))
image_max = np.max(np.percentile(image_standard,99,axis=0))
image_norm = helper_scripts.normalize_manual(image_standard, max_val=image_max, min_val=np.median(image_standard[:,:ch_start]))
image_norm = image_norm-np.median(image_norm[:,:ch_start],axis=1)[0]
X_norm = helper_scripts.normalize(X_standard, max=np.max(image_norm))
X_norm = X_norm-np.median(X_norm[:,:ch_start],axis=1)[0]

## Visualize random spectra for validation of preprocessing
wavenumbers = np.linspace(wavenum_1, wavenum_2, num_samp)
indeces = [random.randint(0,image_norm.shape[0]-1), random.randint(0,image_norm.shape[0]-1), random.randint(0,image_norm.shape[0]-1)]
plt.plot(wavenumbers, image_norm[indeces].T)
plt.title('Baseline Corrected + Normalized Spectra')
plt.xlabel('Wavenumbers (cm$^{-1}$)')
plt.ylabel('Normalized Intensity (A.U.)')


## Optimize Smoothing of HSI and artificial spectra
w = int(input('Enter window size for Savitzky-Golay smoothing (Default=7): ').strip() or '7')
p = int(input('Enter polynomial for Savitzky-Golay smoothing (Default=3): ').strip() or '3')
preprocessing.sav_gol_optimization(image_norm, w, p)
preprocessing.sav_gol_optimization(X_norm, w, p)
x = savgol_filter(image_norm,w,p, axis=1, mode='mirror')
X = savgol_filter(X_norm,w,p, axis=1, mode='mirror')


## Random Forest Classification
rf_classifier = unsupervised_scripts.RF_classify(x, X, Y, .25)
rfc = load('rfc.joblib')
print('Training accuracy: ' + str(rfc.score(rf_classifier.X_train, rf_classifier.y_train)))
rf_classifier.confusion_matrix(mol_names, rfc)
print('Classifiying Hyperspectral Image...')
outputs = unsupervised_scripts.semi_supervised_outputs(x, mol_names, rfc)

print('Save spectral graphs?')
save_input = helper_scripts.save_input()
outputs.spectral_graphs(mol_norm, wavenumbers, save_input, save_dir)

print('Save similarity metric .csv?')
save_input = helper_scripts.save_input()
outputs.similarity_metrics(mol_norm, save_input, save_dir)

print('Save probability images?')
save_input = helper_scripts.save_input()
outputs.probability_images(image, save_input, save_dir)




