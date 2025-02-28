import os
import numpy as np
import pandas as pd
from skimage import io
import processing
from tkinter import filedialog, Tk

from scipy.interpolate import CubicSpline


## Sample Directory (different for everyone)
data_dir = (input('Enter the path to your .tif folder: ').strip() or os.getcwd())


## Load Image

root = Tk()
root.withdraw()
root.call('wm', 'attributes', '.', '-topmost', True)

directory = filedialog.askdirectory()

indeces = [index for index, char in enumerate(directory) if char=='/']
sample_name = directory[indeces[-1]+1:]

## Initialize start and end of hyperspectral sweep; num_samp is calculated from image stack
wavenum_1 = int(input('Enter first wavenumber/higher wavelength (Default=2700)):').strip() or '2700')
wavenum_2 = int(input('Enter last wavenumber/lower wavelength (Default=3100):').strip() or '3100')

# background_df = pd.read_csv('no_background_HSI_76.csv')

## Load Image
data = processing.load_data(directory)
spectra, img_dict = data.load_spectra()#background_df)

new_wn = int(input('Enter the new wavenumber to extrapolate to (Default=62): ').strip() or '62')
x  = np.linspace(0, 1, spectra.shape[0])
new_x = np.linspace(0, 1, new_wn)

## Extrapolate
spline = CubicSpline(x, spectra, axis=0, extrapolate=True)
new_wn_spectra = spline(new_x)

## Save
for images in img_dict.keys():
    new_image = np.reshape(new_wn_spectra, (img_dict[images]['spectra_dim'], img_dict[images]['height'], img_dict[images]['width']))
    io.imsave(directory + '/extrapolated_' + sample_name + '_' + images, new_image)