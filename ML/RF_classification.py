import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io
import random
from joblib import load
import time

from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import unsupervised_scripts
import helper_scripts
from tkinter import Tk, filedialog
import processing



def macro_idx(wavenumber, num_samp, wavenum_1=2700, wavenum_2=3100):
    idx = int(((wavenumber - wavenum_1) / (wavenum_2 - wavenum_1)) * num_samp)
    return idx



class RandomForestDetect():
    def __init__(self, wavenum_1=2700, wavenum_2=3100):
        self.wavenum_1 = wavenum_1
        self.wavenum_2 = wavenum_2

    def RF_classify(self, num_estimators=250, save_input = True):
        for sample in sample_list:
            if sample.startswith('.') or not sample.endswith('.tif'):
                continue
            print(f'\nProcessing {sample}...')
            image = io.imread(sample_dir+os.sep+sample)

            indeces = [index for index, char in enumerate(sample) if char == '/']
            sample_name = os.path.splitext(sample)[0]

            data_dir = sample_dir

            save_dir = data_dir + '/'+ sample_name + '-output/'
            if os.path.exists(save_dir) is False:
                os.mkdir(save_dir)

            ## Remove NaNs and inf
            image[np.isinf(image)] = 0
            image[np.isnan(image)] = 0

            ## Vectorize Image
            image_vec = np.reshape(image, (image.shape[0], image.shape[1] * image.shape[2]))
            image_vec = image_vec.T

            ## Initialize start and end of hyperspectral sweep; num_samp is calculated from image stack
            num_samp = int(image_vec.shape[1])
            ch_start = int(np.floor(num_samp / ((self.wavenum_2 - self.wavenum_1) / (2800 - wavenum_1))))
            background_df = pd.read_csv('water_HSI_76.csv')
            molecule_df = pd.read_excel('lipid_subtype.xlsx')

            ### Semi-Supervised Learning

            shift = 20
            ## Load artificial dataset
            artificial_data = unsupervised_scripts.artificial_dataset(wavenum_1, wavenum_2, num_samp, ch_start, background_df,
                                                                      shift=shift)
            [mol_norm, mol_names] = artificial_data.molecule_dataset(molecule_df)
            X_data = np.load('artificial_data/artificial_training_data-' + str(wavenum_1) + '_' + str(wavenum_2) + '_' + str(
                num_samp) + '.npy')
            Y = list(range(len(mol_names)))*len(X_data)
            X = np.reshape(np.transpose(X_data, (1, 0, 2)), (X_data.shape[0] * X_data.shape[1],
                                                             X_data.shape[2]), order='F')

            ## Process both datasets
            background = unsupervised_scripts.create_background(wavenum_1, wavenum_2, num_samp, background_df, br_shift=shift)
            preprocessing = unsupervised_scripts.preprocessing(wavenum_1, wavenum_2, num_samp, ch_start, background_df)
            image_norm = preprocessing.spectral_standardization(np.flip(image_vec, axis=1), br_shift=shift)

            ## Save Normalized Image and channels (ONLY FOR 2700-3100)
            print('Saving macromolecule channels...')
            unsat_idx = macro_idx(3010 + shift, num_samp, wavenum_1, wavenum_2)
            protein_idx = macro_idx(2938 + shift, num_samp, wavenum_1, wavenum_2)
            sat_idx = macro_idx(2885 + shift, num_samp, wavenum_1, wavenum_2)
            lipid_idx = macro_idx(2850 + shift, num_samp, wavenum_1, wavenum_2)


            norm_image = np.reshape(image_norm, (image.shape[1], image.shape[2], image.shape[0]))
            norm_image = np.moveaxis(norm_image, 2, 0)
            io.imsave(save_dir + 'normalized-' + sample, norm_image.astype('float32'))
            io.imsave(save_dir + 'normalized-unsat-' + sample,
                      np.max(norm_image[list(range(unsat_idx - 2, unsat_idx + 2))], axis=0).astype('float32'))
            io.imsave(save_dir + 'normalized-protein-' + sample,
                      np.max(norm_image[list(range(protein_idx - 2, protein_idx + 2))], axis=0).astype('float32'))
            io.imsave(save_dir + 'normalized-sat-' + sample,
                      np.max(norm_image[list(range(sat_idx - 2, sat_idx + 2))], axis=0).astype('float32'))
            io.imsave(save_dir + 'normalized-lipid-' + sample,
                      np.max(norm_image[list(range(lipid_idx - 2, lipid_idx + 2))], axis=0).astype('float32'))
            print('done.')


            image_norm = processing.normalize(np.flip(image_vec, axis=1))
            image_norm = (image_norm.T - np.median(image_norm[:, :ch_start])).T
            X_norm = processing.normalize(np.flip(X, axis=1), max=np.max(image_norm))
            X_norm = (X_norm.T - np.median(X_norm[:, :ch_start])).T

            ## Visualize distribution of CH scale
            # plt.hist(np.max(image_norm, axis=1) - image_norm[:, 0], label="Image Spectra")
            # plt.hist(np.max(X_norm, axis=1) - X_norm[:, 0], alpha=0.5, label="Artificial Spectra")
            # plt.legend()
            # plt.title("Spectra Peak Intensity")
            # plt.show()

            ## Visualize random spectra for validation of preprocessing

            # wavenumbers = np.linspace(wavenum_1, wavenum_2, num_samp)
            # indeces = [random.randint(0, image_norm.shape[0] - 1), random.randint(0, image_norm.shape[0] - 1),
            #            random.randint(0, image_norm.shape[0] - 1)]
            # plt.plot(wavenumbers, image_norm[indeces].T, label='Training Spectra')
            # rand_idx = np.random.randint(0, X_norm.shape[0])
            # plt.plot(wavenumbers, X_norm[rand_idx].T, label=f'{mol_names[Y[rand_idx]]} Spectra')
            # plt.legend()
            # plt.title('Baseline Corrected + Normalized Spectra')
            # plt.xlabel('Wavenumbers (cm$^{-1}$)')
            # plt.ylabel('Normalized Intensity (A.U.)')
            # plt.show()

            x = image_norm
            X = X_norm

            smooth = 'No Correction'

            ## Random Forest Classification
            print('Classifying  validation data...')
            rf_classifier = unsupervised_scripts.RF_classify(x[~np.all(x == 0, axis=1)], X, Y, .25)
            rfc = load(
                'rf_classifiers/rfc-' +
                smooth + '_' + str(wavenum_1) + '_' + str(wavenum_2) + '_' + str(num_samp) + '_' + str(num_estimators)
                + '.joblib')
            print('Accuracy Score: ' + str(rfc.score(rf_classifier.X_train, rf_classifier.y_train)))
            rf_classifier.confusion_matrix(mol_names, rfc)
            print('done.')
            # plt.show()

            ## Use Random Forest Classifier on Unlabeled HSI
            print('Classifying...')
            start_time = time.time()
            outputs = unsupervised_scripts.semi_supervised_outputs(x[~np.all(x == 0, axis=1)], mol_names, rfc)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Execution time: {execution_time} seconds")
            print('done')

            ## Spectral Graphs
            print('Generating outputs...')
            wavenumbers = np.linspace(wavenum_1, wavenum_2, num_samp)
            outputs.spectral_graphs(mol_norm, background, wavenumbers, save_input, save_dir)
            plt.show()
            outputs.probability_images(image, save_input, save_dir)
            outputs.similarity_metrics(mol_norm, save_input, save_dir=save_dir)
            print('done.')

if __name__ == '__main__':
    ## Sample Directory (different for everyone)
    data_dir = os.getcwd()
    print('Choose the data directory.')
    ## Load Images
    root = Tk()
    root.withdraw()
    root.call('wm', 'attributes', '.', '-topmost', True)
    sample_dir = filedialog.askdirectory(initialdir=data_dir)
    sample_list = os.listdir(sample_dir)
    wavenum_1 = int(input('Enter wavenumber 1 (default=2700): ') or '2700')
    wavenum_2 = int(input('Enter wavenumber 2 (default=3100): ') or '3100')
    detect = RandomForestDetect(wavenum_1, wavenum_2)
    num_estimators = int(input('Enter number of estimators (default=250): ') or '250')
    detect.RF_classify(num_estimators, save_input=True)

