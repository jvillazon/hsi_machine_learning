from processing import load_data, normalize
import matplotlib.pyplot as plt
import numpy as np
import os
import skimage.io as io

data = load_data()
spectra, img_dict, og_spectra = data.load_spectra()
background = data.create_background('water_HSI_76.csv')
mean_spectra = np.mean(spectra, axis=1)
background = normalize(background, mean_spectra[-1], np.median(mean_spectra[:data.ch_start]))
corrected_spectra = (spectra.T-background).T

idx = 2932
fig, ax = plt.subplots(1, 3)
ax[0].plot(og_spectra[:,idx])
ax[0].plot(og_spectra[:,idx+500])
ax[0].plot(og_spectra[:,idx+1000])
ax[1].plot(spectra[:,idx])
ax[1].plot(spectra[:,idx+1000])
ax[1].plot(spectra[:,idx+500])
ax[2].plot(corrected_spectra[:,idx])
ax[2].plot(corrected_spectra[:,idx+1000])
ax[2].plot(corrected_spectra[:,idx+500])

counter = 0
for img in img_dict.keys():
    size = img_dict[img]['shape'][1]*img_dict[img]['shape'][2]
    corrected_img = np.reshape(corrected_spectra[:,counter:counter+size], img_dict[img]['shape'])
    io.imsave(data.save_dir+os.sep+"normalized-"+img, corrected_img)
    counter += size

