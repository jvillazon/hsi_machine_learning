import numpy as np
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt


core_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'core'))
if core_path not in sys.path:
    sys.path.insert(0, core_path)
    
from hsi_unlabeled_dataset import HSI_Unlabeled_Dataset
from hsi_load_data import normalize
import tifffile


def macro_idx(wavenumber, num_samp, wavenum_1=2700, wavenum_2=3100):
    idx = int(((wavenumber - wavenum_1) / (wavenum_2 - wavenum_1)) * num_samp)
    return idx

def spectral_standardization(data, wavenum_1=2700, wavenum_2=3100, num_samp=61, background=None, plot_example=False):
    ch_start = int((2800 - wavenum_1) / (wavenum_2 - wavenum_1) * num_samp)
    if background is None:
        raise ValueError("background cannot be None. Load from srs_params_path.")
    temp_norm = normalize(data)
    temp_end = temp_norm[:,-1:-4:-1]
    temp_start = temp_norm[:,:ch_start]
    temp = temp_norm-np.mean(temp_start,axis=1)[:, np.newaxis]
    spectra_magnitude = np.mean(temp_end, axis=1)-np.mean(temp_start, axis=1)
    background_arr = np.outer(spectra_magnitude, background)

    spectra_standard = temp-background_arr
    if plot_example:
        index = np.random.randint(0, spectra_standard.shape[0], size=1)
        plt.plot(temp[index[0]])
        plt.plot(background_arr[index[0]])
        plt.plot(spectra_standard[index[0]])
        plt.title('Spectral Standardization Example')
        plt.legend(['Original Spectrum', 'Background Estimate', 'Standardized Spectrum'])
        plt.show()

    spectra_max_idx = np.argmax(np.mean(spectra_standard,axis=0))
    spectra_norm = normalize(spectra_standard, max_val=np.mean(spectra_standard[:,spectra_max_idx])+3*np.std(spectra_standard[:,spectra_max_idx]),min_val=0)
    spectra = spectra_norm-np.median(spectra_norm[:,:ch_start],axis=1)[:,np.newaxis]

    return spectra


def main():
    # Setup parameters
    base_directory = "D:/ADATA Backup/HuBMAP/HuBMAP CosMx/CosMx HSI/HSI/"
    num_samp = 61
    wn_1 = 2700
    wn_2 = 3100
    ch_start = int((2800 - wn_1) / (wn_2 - wn_1) * num_samp)
    shift = 20  
    
    srs_params_path = 'params_dataset/srs_params_61.npz'

    # Load SRS parameters to get background dataframe
    srs_data = np.load(srs_params_path)
    background = srs_data['background']

    dataset = HSI_Unlabeled_Dataset(
        base_directory,
        ch_start,
        transform=None,
        image_normalization=True,
        min_max_normalization=False,
        num_samples=num_samp,
        wavenumber_start=wn_1,
        wavenumber_end=wn_2,
    )
    print('Saving macromolecule channels...')

    unsat_idx = macro_idx(3015 + shift, num_samp, wn_1, wn_2)
    protein_idx = macro_idx(2930 + shift, num_samp, wn_1, wn_2)
    sat_idx = macro_idx(2885 + shift, num_samp, wn_1, wn_2)
    lipid_idx = macro_idx(2850 + shift, num_samp, wn_1, wn_2)
    print(f"unsat idx: {unsat_idx}")
    print(f"protein idx: {protein_idx}")
    print(f"sat idx: {sat_idx}")
    print(f"lipid idx: {lipid_idx}")

    for img_path in tqdm(dataset.img_list, desc="Processing images"):
        sample = os.path.basename(img_path)
        img_spectra = dataset.load_and_process_image(img_path)
        stats = dataset.image_stats[img_path]
        height = stats['height']
        width = stats['width']

        pixel_size_x = stats['pixel_size_x']
        pixel_size_y = stats['pixel_size_y']
        resolution  = (1/pixel_size_x, 1/pixel_size_y)
        
        # Normalize image spectra
        img_norm = spectral_standardization(img_spectra, wavenum_1=wn_1, wavenum_2=wn_2, num_samp=num_samp, background=background)
        norm_image = np.reshape(img_norm, (height, width, img_norm.shape[1]))
        norm_image = np.moveaxis(norm_image, 2, 0)  # Move channels to first axis: (channels, height, width)
        output_dir = os.path.join(os.path.dirname(os.path.dirname(img_path)), os.path.join('Normalized_Images', f"{sample.replace('.tif','')}"))
        os.makedirs(output_dir, exist_ok=True)

        # Save normalized image
        tifffile.imwrite(
            os.path.join(output_dir, f"normalized-{sample}"), 
            norm_image.astype('float32'),
            resolution=resolution,  # X and Y resolution
            imagej=True,
            metadata={
                'axes': 'CYX',  # Channels, Y, X
                'unit': 'um',
                'spacing': 1.0,
                'PhysicalSizeX': pixel_size_x,
                'PhysicalSizeXUnit': 'um',
                'PhysicalSizeY': pixel_size_y,
                'PhysicalSizeYUnit': 'um',
            }
        )

        
        # Save macromolecule images
        protein_image = np.max(norm_image[list(range(protein_idx - 2, protein_idx + 2))], axis=0).astype('float32')
        tifffile.imwrite(
            os.path.join(output_dir, f"Protein-{sample}"), 
            protein_image,
            resolution=resolution,  # X and Y resolution
            imagej=True,
            metadata={
                'axes': 'YX',  # Y, X for 2D image
                'unit': 'um',
                'PhysicalSizeX': pixel_size_x,
                'PhysicalSizeXUnit': 'um',
                'PhysicalSizeY': pixel_size_y,
                'PhysicalSizeYUnit': 'um',
            }
        )
        sat_image = np.max(norm_image[list(range(sat_idx - 2, sat_idx + 2))], axis=0).astype('float32')
        tifffile.imwrite(
            os.path.join(output_dir, f"Sat-{sample}"), 
            sat_image,
            resolution=resolution,  # X and Y resolution
            imagej=True,
            metadata={
                'axes': 'YX',  # Y, X for 2D image
                'unit': 'um',
                'PhysicalSizeX': pixel_size_x,
                'PhysicalSizeXUnit': 'um',
                'PhysicalSizeY': pixel_size_y,
                'PhysicalSizeYUnit': 'um',
            }
        )
        lipid_image = np.max(norm_image[list(range(lipid_idx - 2, lipid_idx + 2))], axis=0).astype('float32')
        tifffile.imwrite(
            os.path.join(output_dir, f"Lipid-{sample}"), 
            lipid_image,
            resolution=resolution,  # X and Y resolution
            imagej=True,
            metadata={
                'axes': 'YX',  # Y, X for 2D image
                'unit': 'um',  # Unit for all dimensions
                'PhysicalSizeX': pixel_size_x,
                'PhysicalSizeXUnit': 'um',
                'PhysicalSizeY': pixel_size_y,
                'PhysicalSizeYUnit': 'um',
            }
        )
        unsat_image = np.max(norm_image[list(range(unsat_idx - 2, unsat_idx + 2))], axis=0).astype('float32')
        tifffile.imwrite(
            os.path.join(output_dir, f"Unsat-{sample}"), 
            unsat_image,
            resolution=resolution,  # X and Y resolution
            imagej=True,
            metadata={
                'axes': 'YX',  # Y, X for 2D image
                'unit': 'um',
                'PhysicalSizeX': pixel_size_x,
                'PhysicalSizeXUnit': 'um',
                'PhysicalSizeY': pixel_size_y,
                'PhysicalSizeYUnit': 'um',
            }
        )
        unsat_ratio = np.divide(unsat_image, (sat_image + 1e-6))
        unsat_ratio = np.maximum(unsat_ratio, 0)
        unsat_ratio = np.where(unsat_ratio > 50, 0, unsat_ratio)
        lipid_ratio = np.divide(lipid_image, (protein_image + 1e-6))
        lipid_ratio = np.maximum(lipid_ratio, 0)
        lipid_ratio = np.where(lipid_ratio > 50, 0, lipid_ratio)

        # Save ratio images
        ratio_output_dir = os.path.join(os.path.dirname(os.path.dirname(img_path)), os.path.join('Ratio', f"{sample.replace('.tif','')}"))
        os.makedirs(ratio_output_dir, exist_ok=True)
        # unsat_ratio = np.expand_dims(unsat_ratio, axis=(0,1))  # Add channel dimension for ZCXY format
        # lipid_ratio = np.expand_dims(lipid_ratio, axis=(0,1))  # Add channel dimension for ZCXY format

        tifffile.imwrite(
            os.path.join(ratio_output_dir, f"{sample.replace('.tif','')}-Lipid_Unsaturation-ratio.tif"), 
            unsat_ratio.astype('float32'),
            resolution=resolution,  # X and Y resolution
            imagej=True,
            metadata={
                'axes': 'YX',  # Y, X for 2D image
                'unit': 'um',
                'PhysicalSizeX': pixel_size_x,
                'PhysicalSizeXUnit': 'um',
                'PhysicalSizeY': pixel_size_y,
                'PhysicalSizeYUnit': 'um',
            }
        )
        tifffile.imwrite(
            os.path.join(ratio_output_dir, f"{sample.replace('.tif','')}-Lipid_to_Protein-ratio.tif"), 
            lipid_ratio.astype('float32'),
            resolution=resolution,  # X and Y resolution
            imagej=True,
            metadata={
                'axes': 'YX',  # Y, X for 2D image
                'unit': 'um',
                'PhysicalSizeX': pixel_size_x,
                'PhysicalSizeXUnit': 'um',
                'PhysicalSizeY': pixel_size_y,
                'PhysicalSizeYUnit': 'um',
            }
        )


if __name__ == "__main__":
    main()
    print("Processing complete.")