import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from skimage import io
from sklearn import cluster, decomposition, model_selection, ensemble, metrics, pipeline
from scipy import signal, interpolate
from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import helper_scripts
import imageio
from tkinter import filedialog, Tk


def spectra_correction(spectrum, ch_start):
    """Correct HSI Spectra"""

    temp_spectra = spectrum
    temp_spectra[np.isinf(temp_spectra)] = 0
    temp_spectra[np.isnan(temp_spectra)] = 0

    # temp_spectra = normalize(temp_spectra)
    spectra_max_idx = np.argmax(np.mean(temp_spectra, axis=1))
    temp_spectra = normalizebyvalue(temp_spectra, max_val=np.mean(temp_spectra[spectra_max_idx]) + 3 * np.std(
        temp_spectra[spectra_max_idx]), min_val=0)
    temp_spectra = np.flip(temp_spectra, axis=0)
    temp_spectra = temp_spectra - np.median(temp_spectra[:ch_start:, :], axis=0)

    # temp_spectra, arr_mean = snv(temp_spectra)

    return temp_spectra

class load_data():

    def __init__(self, wn_1=2700, wn_2=3100):
        """Choose Directory"""
        root = Tk()
        root.withdraw()
        root.call('wm', 'attributes', '.', '-topmost', True)
        self.data_dir = filedialog.askdirectory()
        path_name = [index for index, char in enumerate(self.data_dir) if char=='/']
        self.sample_name = self.data_dir[path_name[-1]+1:]
        self.data_dir = os.path.abspath(self.data_dir)

        self.save_dir = os.path.join(os.path.dirname(self.data_dir)+os.sep+self.sample_name+"_output")
        if os.path.exists(self.save_dir) is False:
            os.mkdir(self.save_dir)

        """Designate Raman Region"""
        self.wn_1 = wn_1
        self.wn_2 = wn_2


    def load_spectra(self):
        """Load HSI spectra"""

        list_dir = [name for name in os.listdir(self.data_dir)if not name.startswith(".") and name.endswith('.tif')]
        img_dict = {key: {} for key in list_dir if key.endswith('.tif')}

        for idx, image in tqdm(enumerate(list_dir)):
            img_path = os.path.join(self.data_dir+os.sep+image)
            img = io.imread(img_path)

            img_dict[image]['shape'] = img.shape

            self.num_samp = img_dict[image]['shape'][0]
            if self.wn_1 < 2850:
                self.ch_start = int(np.floor(self.num_samp/((self.wn_2-self.wn_1)/(2850-self.wn_1))))
            else:
                self.ch_start = 1

            old_spectra = img.reshape(img.shape[0], img.shape[1]*img.shape[2])
            temp_spectra = spectra_correction(old_spectra, self.ch_start)

            if idx == 0:
                spectra = temp_spectra
                # orig_spectra = old_spectra
            else:
                spectra = np.concatenate((spectra, temp_spectra), axis=1)
                print(f"Spectra size is: {spectra.shape}")
                # orig_spectra = np.concatenate((orig_spectra, old_spectra), axis=1)

        return spectra, img_dict #, orig_spectra

    def create_background(self, background_path):
        background_df = pd.read_csv(background_path)
        Y = background_df.values[:, 0]
        Y = normalize(Y)
        X = np.linspace(0, 1, Y.shape[0])
        spline = interpolate.CubicSpline(X, Y)
        x = np.linspace(0, 1, self.num_samp)
        background = spline(x)

        return np.flip(background)

def snv(array):
    snv_arr = np.zeros_like(array)
    for i in range(array.shape[0]):
        snv_arr[i,:] = (array[i,:] - np.mean(array[i,:])) / np.std(array[i,:])
    return snv_arr


def normalize(array, max=1, min=0,axis=None):

    
    min_val = np.min(array, axis=axis)
    max_val = np.max(array, axis=axis)
    
    diff = max_val-min_val
    if axis is None:
        if  np.all(diff==0):
            norm = array
        else:
            norm = ((array-min_val)/(diff))*(max-min)+min
    else:
        norm = array.copy()

        norm = (((array.T-min_val)/(diff))*(max-min)+min).T

    return norm

def normalizebyidx(array, max_idx=0, min_idx=-1, max=1, min=0, axis=None):

    max_val = array[max_idx]
    min_val = array[min_idx]

    diff = max_val-min_val
    if axis is None:
        if np.all(diff==0):
            norm = array
        else:
            norm = ((array-min_val)/(diff))*(max-min)+min
    else:
        norm = array.copy()
        idx = np.where(diff!=0)[0]
        norm[idx] = (((array[idx].T-min_val[idx])/(diff[idx]))*(max-min)+min).T

    return norm

def normalizebyvalue(array, max_val, min_val, max=1, min=0, axis=None):
    
    diff = max_val-min_val
    if axis is None:
        if np.all(diff==0):
            norm = array
        else:
            norm = ((array-min_val)/(diff))*(max-min)+min
    else:
        norm = array.copy()
        idx = np.where(diff!=0)[0]
        norm[idx] = (((array[idx].T-min_val[idx])/(diff[idx]))*(max-min)+min).T

    return norm




# def spectra_correction(spectra, background):
#     spectra_corr = normalize(spectra)
#     spectra_mean = np.mean(spectra_corr,axis=1)
#     background = normalize(background, spectra_mean[0], spectra_mean[-1])
#     spectra_corr = spectra_corr.T - background
#     spectra_max_idx = np.argmax(np.mean(spectra_corr,axis=0))
#     spectra_corr = normalizebyvalue(spectra_corr, max_val=np.mean(spectra_corr[spectra_max_idx])+3*np.std(spectra_corr[spectra_max_idx]),min_val=0)
#     spectra_corr = np.flip(spectra_corr.T,axis=0)

#     return spectra_corr

