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
    def __init__(self, wavenum_1=2700, wavenum_2=3100, num_samp=61, background_path='water_HSI_76.csv', molecule_path='lipid_subtype.xlsx', shift=20):
        ## Initialize spectral information
        self.wavenum_1 = wavenum_1
        self.wavenum_2 = wavenum_2
        self.num_samp = num_samp
        self.ch_start = int(np.floor(num_samp / ((self.wavenum_2 - self.wavenum_1) / (2800 - self.wavenum_1))))

        background_df = pd.read_csv(background_path)
        molecule_df = pd.read_excel(molecule_path)
        self.shift = shift

        ## Load artificial dataset
        artificial_data = unsupervised_scripts.artificial_dataset(
            self.wavenum_1,
            self.wavenum_2,
            self.num_samp,
            self.ch_start,
            background_df,
            self.shift,
        )

        [self.mol_norm, self.mol_names] = artificial_data.molecule_dataset(molecule_df)
        X_data = np.load(
            'artificial_data/artificial_training_data-'
            + str(self.wavenum_1) + '_'
            + str(self.wavenum_2) + '_'
            + str(num_samp) + '.npy'
        )
        self.Y = list(range(len(self.mol_names)))*len(X_data)
        self.X_data = np.reshape(
            np.transpose(X_data, (1, 0, 2)),
            (X_data.shape[0] * X_data.shape[1], X_data.shape[2]),
            order='F'
        )

        ## Process both datasets
        self.background = unsupervised_scripts.create_background(
            self.wavenum_1,
            self.wavenum_2,
            self.num_samp,
            background_df,
            br_shift=self.shift
        )
        self.preprocessing = unsupervised_scripts.preprocessing(
            self.wavenum_1,
            self.wavenum_2,
            self.num_samp,
            self.ch_start,
            background_df
        )

        self.unsat_idx = macro_idx(3010 + self.shift, self.num_samp, self.wavenum_1, self.wavenum_2)
        self.protein_idx = macro_idx(2938 + self.shift, self.num_samp, self.wavenum_1, self.wavenum_2)
        self.sat_idx = macro_idx(2885 + self.shift, self.num_samp, self.wavenum_1, self.wavenum_2)
        self.lipid_idx = macro_idx(2850 + self.shift, self.num_samp, self.wavenum_1, self.wavenum_2)

    def RF_classify(self, sample_dir, sample_list, num_estimators=250, save_input = True):

        for sample in sample_list:
            if sample.startswith('.') or not sample.endswith('.tif'):
                continue

            print(f'\nProcessing {sample}...')

            # Load image and process
            image = io.imread(sample_dir+os.sep+sample)
            image_shape = image.shape
            if image_shape[0] != self.num_samp:
                print(f"Wavenumber count is not equal to {self.num_samp} for sample {sample}. Continuing to next image...")
                continue
            ## Flip and correct gradient
            image = np.flip(image, axis=0)
            image = image - np.mean(image[:self.ch_start], axis=0)
            ## Remove NaNs and inf
            image[np.isinf(image)] = 0
            image[np.isnan(image)] = 0
            ## Vectorize Image
            image_vec = np.reshape(image, (image.shape[0], image.shape[1] * image.shape[2]))
            image_vec = image_vec.T


            sample_name = os.path.splitext(sample)[0]
            data_dir = sample_dir
            save_dir = data_dir + '/'+ sample_name + '-output/'
            if os.path.exists(save_dir) is False:
                os.mkdir(save_dir)

            output_macromolecule = True
            if output_macromolecule:

                image_norm = self.preprocessing.spectral_standardization(image_vec, br_shift=self.shift)

                ## Save Normalized Image and channels (ONLY FOR 2700-3100)
                print('Saving macromolecule channels...')

                norm_image = np.reshape(image_norm, (image.shape[1], image.shape[2], image.shape[0]))
                norm_image = np.moveaxis(norm_image, 2, 0)
                output_dir = os.path.join(os.path.dirname(save_dir), 'Normalized_Images/')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                io.imsave(output_dir + 'normalized-' + sample, norm_image.astype('float32'))
                io.imsave(output_dir + 'Unsat-' + sample,
                          np.max(norm_image[list(range(self.unsat_idx - 2, self.unsat_idx + 2))], axis=0).astype('float32'))
                io.imsave(output_dir + 'Protein-' + sample,
                          np.max(norm_image[list(range(self.protein_idx - 2, self.protein_idx + 2))], axis=0).astype('float32'))
                io.imsave(output_dir + 'Sat-' + sample,
                          np.max(norm_image[list(range(self.sat_idx - 2, self.sat_idx + 2))], axis=0).astype('float32'))
                io.imsave(output_dir + 'Lipid-' + sample,
                          np.max(norm_image[list(range(self.lipid_idx - 2, self.lipid_idx + 2))], axis=0).astype('float32'))
                print('done.')


            image_norm = processing.normalize(image_vec)
            image_norm = (image_norm.T - np.median(image_norm[:, :self.ch_start])).T
            X_norm = processing.normalize(np.flip(self.X_data, axis=1), max=np.max(image_norm))
            X_norm = (X_norm.T - np.median(X_norm[:, :self.ch_start])).T


            # Visualize random spectra for validation of preprocessing
            wavenumbers = np.linspace(self.wavenum_1, self.wavenum_2, self.num_samp)
            indeces = [random.randint(0, image_norm.shape[0] - 1), random.randint(0, image_norm.shape[0] - 1),
                       random.randint(0, image_norm.shape[0] - 1)]
            plt.plot(wavenumbers, image_norm[indeces].T, label='Training Spectra')
            rand_idx = np.random.randint(0, X_norm.shape[0])
            plt.plot(wavenumbers, X_norm[rand_idx].T, label=f'{self.Y[rand_idx]} Spectra')
            plt.legend()
            plt.title('Baseline Corrected + Normalized Spectra')
            plt.xlabel('Wavenumbers (cm$^{-1}$)')
            plt.ylabel('Normalized Intensity (A.U.)')
            plt.show()

            x = image_norm
            X = X_norm

            smooth = 'No Correction'

            ## Random Forest Classification
            print('Classifying  validation data...')
            rf_classifier = unsupervised_scripts.RF_classify(
                x,
                X,
                self.Y,
                .25
            )

            rfc = load(
                'rf_classifiers/rfc-'
                + smooth + '_'
                + str(self.wavenum_1) + '_'
                + str(self.wavenum_2) + '_'
                + str(self.num_samp) + '_'
                + str(num_estimators) + '.joblib'
            )
            print('Accuracy Score: ' + str(rfc.score(rf_classifier.X_train, rf_classifier.y_train)))
            rf_classifier.confusion_matrix(self.mol_names, rfc)
            print('done.')
            # plt.show()

            ## Use Random Forest Classifier on Unlabeled HSI
            print('Classifying...')
            start_time = time.time()
            outputs = unsupervised_scripts.semi_supervised_outputs(
                x,
                self.mol_names,
                rfc
            )
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Execution time: {execution_time} seconds")
            print('done')

            ## Spectral Graphs
            print('Generating outputs...')
            wavenumbers = np.linspace(self.wavenum_1, self.wavenum_2, self.num_samp)
            outputs.spectral_graphs(self.mol_norm, self.background, wavenumbers, save_input, save_dir)
            plt.show()
            outputs.probability_images(image_shape, save_input, save_dir)
            outputs.similarity_metrics(self.mol_norm, image_shape, self.unsat_idx, self.background, save_input, save_dir=save_dir)


def main():
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
    detect.RF_classify(sample_dir, sample_list, num_estimators, save_input=True)


if __name__ == '__main__':
    main()

    print('Done.')

