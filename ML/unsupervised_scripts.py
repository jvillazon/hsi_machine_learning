import os
import numpy as np
import pandas as pd
from skimage import io
import imageio
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, silhouette_score, calinski_harabasz_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from scipy.signal import savgol_filter, correlate
from scipy.interpolate import CubicSpline
from time import time
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
import helper_scripts

def save_input():
    while True:
        try:
            y_n_input = input('Save into save directory (y/n)?: ').lower().strip() or 'y'
            if y_n_input in ['y', 'n']:
                if y_n_input == 'y':
                    return True
                if y_n_input == 'n':
                    return False
            else:
                raise Exception('Invalid input, please press y/n.')
        except Exception('Invalid input, please press y/n.'):
            continue
        else:
            break


def create_background(wavenumber_1, wavenumber_2, num_samp, background_df, br_shift):

    """Generate background (PBS) baseline from background .csv file (WARNING: Only for CH-stretching region)
    
    Parameters
    ----------
    wavenumber_1 : int
        Initial Raman wavenumber (lower wavenumber) captured by hyperspectral image (measured in cm^-1).
    wavenumber_2 : int
        Final Raman wavenumber (higher wavenumber) captured by hyperspectral image (measured in cm^-1).
    num_samp : int
        Number of hyperspectral slices
    background_df : Pandas DataFrame
        Dataframe generated from background .csv file corresponding to PBS baseline

    Outputs
    ----------
    background : ndarray of shape (num_samp,)
        1-dimensional array corresponding to the PBS background between the two wavenumbers
    """

    CH_wavenumber = np.linspace(wavenumber_1,wavenumber_2,num_samp-br_shift)
    temp = background_df[:].to_numpy() 
    temp = temp[:,0]
    x = np.linspace(wavenumber_1, wavenumber_2,len(temp))
    y = np.array(temp)
    spline = CubicSpline(x, y)
    background = spline(CH_wavenumber)
    background = helper_scripts.normalize(background)
    background = np.flip(background)
    back_temp = np.zeros(num_samp)
    back_temp[br_shift:]=background
    background = back_temp

    return background

def bench_k_means(kmeans, name, data):
    'Adapted from https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html'

    """Benchmark to evaluate the KMeans initialization methods.

    Parameters
    ----------
    kmeans : KMeans instance
        A :class:`~sklearn.cluster.KMeans` instance with the initialization
        already set.
    name : str
        Name given to the strategy. It will be used to show the results in a
        table.
    data : ndarray of shape (n_samples, n_features)
        The data to cluster.
    labels : ndarray of shape (n_samples,)
        The labels used to compute the clustering metrics which requires some
        supervision.
    """
    t0 = time()
    estimator = make_pipeline(kmeans).fit(data)
    fit_time = time() - t0
    results = [name, fit_time, estimator[-1].inertia_]

    # The silhouette score requires the full dataset
    results += [
        silhouette_score(
            data,
            estimator[-1].labels_,
            metric="euclidean",
            sample_size=300,
        )
    ]

    # Show the results
    formatter_result = (
        "{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}"
    )
    print(formatter_result.format(*results))




class artificial_dataset:

    def __init__(self, wavenumber_1, wavenumber_2, num_samp, ch_start, background_df, shift=15):
        self.shift = shift
        self.wavenumber_1 = wavenumber_1-self.shift
        self.wavenumber_2 = wavenumber_2-self.shift
        self.num_samp = num_samp
        self.ch_start = ch_start
        self.background = create_background(self.wavenumber_1, self.wavenumber_2, self.num_samp, background_df,br_shift=0)
    

    def molecule_dataset(self, molecule_df):
        molecule_df = molecule_df.dropna(axis='columns',how='all')
        molecule_df_size = molecule_df.columns.shape[0]
        molecule_arr = molecule_df.to_numpy()
        num_mol = int(molecule_df_size / 2)
        temp_names = np.array(molecule_df.columns[0:molecule_df_size:2])
        wavenumber = np.linspace(self.wavenumber_1, self.wavenumber_2, self.num_samp)
        temp = np.empty((num_mol, self.num_samp), dtype='float32')
        remove_nan = []
        for i in list(range(0, molecule_df_size, 2)):
            molx = molecule_arr[:, i]
            moly = molecule_arr[:, i + 1]
            molx = molx[~np.isnan(molx)]
            moly = moly[~np.isnan(moly)]
            if molx.min() <= self.wavenumber_1 and molx.max() >= self.wavenumber_2:
                temp_spline = CubicSpline(molx, moly, extrapolate=True)
                temp[int(i / 2)] = temp_spline(wavenumber)
            else:
                remove_nan.append(int(i / 2))
        keep = [i for i in range(temp.shape[0]) if i not in remove_nan]
        molecules = temp[keep, :]
        mol_norm = helper_scripts.normalize(molecules)
        mol_norm = np.vstack((mol_norm, np.zeros(self.num_samp,dtype='float32')))
        mol_names = temp_names[keep]
        mol_names = np.hstack((mol_names, 'Background'))
    
        return mol_norm, mol_names
    
    def DAE_dataset(self, mol_norm):
        dataset_size = len(mol_norm)
        total_classes_size = int(((dataset_size-1) * (dataset_size)) / 2) + 1
        mol_add = np.zeros((total_classes_size - (dataset_size), self.num_samp), dtype='float32')
        count = 0
        for j in range(dataset_size - 2):
            for i in tqdm(range(j + 1, dataset_size - 1)):
                temp_spectra = mol_norm[j, :] + mol_norm[i, :]
                mol_add[count, :] = helper_scripts.normalize(temp_spectra)
                count += 1
        dataset_norm = np.vstack((mol_norm, mol_add))

        return dataset_norm
    
    def save_srs_params(self, data_dir):

        boolean = True
        data_list = os.listdir(data_dir)
        for name in tqdm(data_list):
            if not name.startswith("."):
                image = imageio.imread(data_dir + name)
            else:
                continue
            image_vector = np.reshape(image, (image.shape[0], image.shape[1] * image.shape[2]))
            image_vector = np.flip(image_vector, axis=0)
            first_val = np.median(image_vector[:self.ch_start, :])
            image_vector = image_vector - first_val.T
            if boolean:
                 temp = image_vector
                 boolean = False
            else:
                temp = np.concatenate((temp, image_vector), axis=1)
        image_spec = helper_scripts.normalize(temp)
        image_spec = image_spec[:, np.logical_not(image_spec[0, :] > image_spec[-1, :])]
        spec_start = image_spec[:self.ch_start]
        image_vec = image_spec[:, np.all(
            np.logical_and(spec_start < np.mean(spec_start) + 3 * np.std(spec_start),
                    spec_start > np.mean(spec_start) - 3 * np.std(spec_start)), axis=0)] 
        
        noise_scale_vec = np.std(spec_start, axis=0)
        start_vec = image_vec[0]
        bg_scale_vec = image_vec[-1] - image_vec[0]
        max_vec = np.max(image_vec, axis=0)
        ratio_scale_vec = max_vec-start_vec

        return image_vec.T, noise_scale_vec, bg_scale_vec, ratio_scale_vec

        # Generate artificial SRS spectra from the molecular spectra
    def save_artificial_dataset(self, mol_norm, noise_scale_vec, bg_scale_vec, ratio_scale_vec, num_values=200, num_repeats=10, bg_param=1, ch_param=1, noise_param=1, name='artificial_training_data-'):
        mol_norm = np.flip(mol_norm,axis=1)

        random_integers = np.random.randint(0, int(bg_scale_vec.shape[0]), size=num_values)
        background_scale = bg_param*bg_scale_vec[random_integers]
        ch_scale = ch_param*ratio_scale_vec[random_integers]
        noise_scale = np.maximum(noise_param*noise_scale_vec[random_integers], 0.25)
        background = np.flip(np.outer(background_scale, self.background),axis=1)
        noise = np.random.normal(0, noise_scale, (self.num_samp, num_values)).T
        artificial_mol = np.zeros((mol_norm.shape[0], num_values, mol_norm.shape[1]), dtype='float32')
        for mol_index in tqdm(range(len(mol_norm))):
            mol_temp = mol_norm[mol_index]
            mol_vec = np.outer(ch_scale, mol_temp)
            artificial_mol[mol_index] = (mol_vec + noise + background)
        artificial_mol= helper_scripts.normalize(artificial_mol)
        artificial_mol = np.tile(artificial_mol, (num_repeats, 1, 1, 1))
        artificial_mol = np.moveaxis(artificial_mol, 2, 1)
        artificial_mol = np.reshape(artificial_mol, (artificial_mol.shape[0] * artificial_mol.shape[1], artificial_mol.shape[2], artificial_mol.shape[3]))

        np.save('artificial_data/'+name+str(self.wavenumber_1+self.shift)+'_'+str(self.wavenumber_2+self.shift)+'_'+str(self.num_samp)+'.npy', artificial_mol)



class preprocessing():
    def __init__(self, wavenumber_1, wavenumber_2, num_samp, ch_start, background_df):
        self.wavenumber_1 = wavenumber_1
        self.wavenumber_2 = wavenumber_2
        self.num_samp = num_samp
        self.ch_start = ch_start
        self.background_df = background_df


    def spectral_standardization(self, data, br_shift=0):
        self.background = create_background(self.wavenumber_1, self.wavenumber_2, self.num_samp, self.background_df, br_shift)
        temp_norm = helper_scripts.normalize(data)
        temp_end = temp_norm[:,-1:-4:-1]
        temp_start = temp_norm[:,:self.ch_start]
        temp = temp_norm-np.mean(temp_start,axis=1)[0]
        spectra_magnitude = np.mean(temp_end, axis=1)-np.mean(temp_start, axis=1)
        background_arr = np.outer(spectra_magnitude, self.background)
        spectra_standard = temp-background_arr
        spectra_max_idx = np.argmax(np.mean(spectra_standard,axis=0))
        spectra_norm = helper_scripts.normalize_manual(spectra_standard, max_val=np.mean(spectra_standard[:,spectra_max_idx])+3*np.std(spectra_standard[:,spectra_max_idx]),min_val=0)
        spectra = spectra_norm.T-np.median(spectra_norm[:,:self.ch_start],axis=1)
        spectra = spectra.T

        return spectra

    def sav_gol_optimization(self, standardized_data, w, p):
        wavenumbers = np.linspace(self.wavenumber_1, self.wavenumber_2, self.num_samp)

        idx = np.random.randint(0,int(standardized_data.shape[1]))
        b_idx = np.argmin(standardized_data,axis=0)[[np.argmax(np.mean(standardized_data,axis=0))]]
        b_idx = b_idx[0]

        ps = np.abs(np.fft.fftshift(np.fft.fft(standardized_data[b_idx,:])))**2
        fpix = np.arange(ps.shape[0]) - ps.shape[0]//2
        filt_image_norm = savgol_filter(standardized_data[b_idx,:], w, p)
        ps_1 = np.abs(np.fft.fftshift(np.fft.fft(filt_image_norm)))**2

        fig, ax = plt.subplots(1, 3, figsize=(24,6))
        ax[0].plot(wavenumbers,standardized_data[b_idx,:], label = 'No smoothing')
        ax[0].plot(wavenumbers, savgol_filter(standardized_data[b_idx,:], w, p,mode='mirror'), label = f'Smoothing - w = {w}, p = {p}')
        ax[0].legend()
        ax[0].set_xlabel('Wavenumber (cm$^{-1}$)')
        ax[0].set_ylabel('Normalized Intensity (A.U.)')
        ax[0].set_title('Background Spectra')

        ax[1].semilogy(fpix, ps, label = 'No smoothing')
        ax[1].semilogy(fpix, ps_1, label = f'Smoothing - w = {w}, p = {p}')
        ax[1].legend()
        ax[1].set_xlabel('Fourier Space')
        ax[1].set_ylabel('Power Spectrum')
        ax[1].set_title('PSD')

        ax[2].plot(wavenumbers, standardized_data[idx,:], label = 'No smoothing')
        ax[2].plot(wavenumbers, savgol_filter(standardized_data[idx,:], w, p,mode='mirror'), label = f'Smoothing - w = {w}, p = {p}')
        ax[2].legend()
        ax[2].set_xlabel('Wavenumber (cm$^{-1}$)')
        ax[2].set_ylabel('Normalized Intensity (A.U.)')
        ax[2].set_title('Pixel Spectra')

        plt.suptitle('Savitzky-Golay Optimization')

def hex_2_rgb (hexcodes):
    hex_color = hexcodes.lstrip('#')
    return tuple(int(hex_color[i:i+2],16) for i in (0, 2, 4))


class K_means_cluster():

    def __init__ (self, standardized_data):
        self.data = standardized_data

    def kmeans_silhoutette_score (self, n_clusters_range):
        print(82 * "_")
        print("init\t\ttime\tinertia\tsilhouette")
        for n_clusters in list(range(2,n_clusters_range)):
            pca = PCA(n_components=n_clusters).fit(self.data)
            kmeans = KMeans(init=pca.components_, n_clusters=n_clusters, n_init=1)
            print(f'Clusters: {n_clusters}') 
            bench_k_means(kmeans=kmeans, name="PCA-based K-means", data=self.data)
        print(82 * "_")

    def kmeans (self, n_clusters):
        pca = PCA(n_components=n_clusters).fit(self.data)
        kmeans = KMeans(init=pca.components_, n_clusters=n_clusters, n_init=1)
        kmeans.fit(self.data)

        return kmeans
    
    def kmeans_graph (self, kmeans, wavenumbers, save_input, color_list=['#000000', '#FF00FF', '#FFFF00', '#00FFFF', '#00FF00','#0000FF','#FF0000'], save_dir=None):
        centers = kmeans.cluster_centers_
        cent_range = list(range(len(centers)))
        centers = (centers.T-np.min(centers,axis=1)).T
        sorted_centers = centers[np.argsort(centers.mean(axis=1))]
        plt.figure()
        plt.tight_layout()
        for idx in cent_range:
            cent = sorted_centers[idx,:]
            plt.plot(wavenumbers, cent, label='Cluster '+str(idx+1), color=color_list[idx])
        plt.grid()
        plt.tick_params(right = False , labelleft = False) 
        plt.xlabel('Wavenumber (cm$^-$$^1$)',fontsize=14,fontweight='bold')
        plt.ylabel('Raman Intensity (A.U.)',fontsize=14,fontweight='bold')
        plt.legend(loc='upper left')
        plt.title('K-means Cluster',fontsize=14,fontweight='bold')
        if save_input:
            fig1 = plt.gcf()
            fig1.savefig(save_dir + 'Graph.tif', bbox_inches='tight')    
        plt.show()
        return sorted_centers
   

    def kmeans_image (self, kmeans, image_list, img_dict, save_input, color_list=['#FFFFFF', '#FF00FF', '#FFFF00', '#00FFFF', '#00FF00','#0000FF','#FF0000'], save_dir=None):
        rgb_list = [hex_2_rgb(hexcode) for hexcode in color_list]
        centers = kmeans.cluster_centers_
        cent_range = list(range(len(kmeans.cluster_centers_)))
        centers = (centers.T-np.min(centers,axis=1)).T
        row_idx = np.argsort(centers.mean(axis=1))
        orig_labels = kmeans.labels_
        temp = np.empty((orig_labels.shape[0],3))
        label_idx = np.empty(orig_labels.shape[0])
        for idx in cent_range:
            new_idx = row_idx[idx]
            temp[np.where(orig_labels==new_idx)[0]] = rgb_list[idx]
            label_idx[np.where(orig_labels==new_idx)[0]] = idx

        labels = temp
        counter = 0
        label_list = []
        for i, image in enumerate(image_list):
            size = image.shape[0]*image.shape[1]
            label_img = np.reshape(labels[counter:counter+(size)], (image.shape[0], image.shape[1], 3))
            label_list.append(label_img)
            counter += size

            cluster_stack = np.empty((len(centers), image.shape[0], image.shape[1], 3))
            for idx, color in tqdm(enumerate(rgb_list)):
                mask = np.all(label_img == color, axis=-1)
                cluster_stack[idx] = label_img * (mask[:, :, None])
            if save_input:
                io.imsave(save_dir + [*img_dict.keys()][i], label_img)
                io.imsave(save_dir + 'cluster_stack-'+[*img_dict.keys()][i], cluster_stack)
    
        fig, ax = plt.subplots(len(image_list), 2, figsize=(8,int(len(image_list)*4)))
    
        if len(image_list)==1:
            ax[0].imshow(image_list[0])
            ax[0].set_yticks([])
            ax[0].set_xticks([])
            ax[1].imshow(label_list[0])
            ax[1].set_yticks([])
            ax[1].set_xticks([])
        else:
            for i in range(len(image_list)):
                ax[i,0].imshow(image_list[i])
                ax[i,0].set_yticks([])
                ax[i,0].set_xticks([])
                ax[i,1].imshow(label_list[i])
                ax[i,1].set_yticks([])
                ax[i,1].set_xticks([])
        plt.show()
        return label_idx


    
class RF_classify():

    def __init__ (self, x, X, y, test_percent):
        self.x = x
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_percent,shuffle=True)



    def train(self, num_estimators=100, criterion='entropy', max_features='sqrt'):
        rf_classifier = RandomForestClassifier(n_estimators=num_estimators, criterion=criterion , max_features=max_features)
        rf_classifier.fit(self.X_train, self.y_train)

        return rf_classifier

    def confusion_matrix(self,  mol_names, rf_classifier):
        y_test_pred = rf_classifier.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_test_pred)

        disp = ConfusionMatrixDisplay(cm, display_labels=mol_names)
        fig, ax = plt.subplots(figsize=(20,20))
        disp.plot(ax=ax)
        plt.show()
    
def calculate_mse(spectrum1, spectrum2):
    return mean_squared_error(spectrum1, spectrum2)

def compute_cosine_similarity(spectrum1, spectrum2):
    return cosine_similarity([spectrum1], [spectrum2])[0, 0]

def compute_crosscor(spectrum1, spectrum2):
    crosscorrelation = correlate(spectrum1, spectrum2, mode='full')
    norm_signal1 = np.sqrt(np.sum(spectrum1**2))
    norm_signal2 = np.sqrt(np.sum(spectrum2**2))
    normalized_crosscorrelation = crosscorrelation / (norm_signal1 * norm_signal2)

    return normalized_crosscorrelation


class semi_supervised_outputs():
    
    def __init__ (self, x, label, classifier):
        self.x = x
        self.label = label
        self.classifier = classifier
        self.y_pred = self.classifier.predict(x)
        self.y_prob = self.classifier.predict_proba(x)
        self.car_hab_score = calinski_harabasz_score(self.x, self.y_pred)
    
    def spectral_graphs (self, mol_norm, wavenumbers, save_input, save_dir=None):
        for idx in tqdm(range(len(self.label)-1)):
            cur_label = self.x[self.y_pred==idx]
            if cur_label.shape[0] > 0:
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111)
                ax.set(yticklabels=[])
                ymean = cur_label.mean(axis=0)
                yerror = cur_label.std(axis=0)/2
                plt.plot(wavenumbers, ymean,color='green', label='sample signal mean' )
                plt.fill_between(wavenumbers, ymean-yerror, ymean+yerror, alpha=0.5, color='green')
                plt.plot(wavenumbers,helper_scripts.normalize(mol_norm[idx], np.max(ymean),0), alpha=0.75, color='black', label=self.label[idx]+' signal',linewidth = 3)
                plt.legend(loc='center left',fontsize='16')
                plt.title('Predicted SRS Spectra for \n'+str(self.label[idx]),fontsize='28', weight ='bold')
                plt.xlabel('Wavenumber (cm$^-$$^1$)',fontsize='24', weight ='bold')
                plt.ylabel('Normalized Intensity',fontsize='24', weight ='bold')
                plt.xticks(fontsize=16)
                if save_input:
                    plt.savefig(save_dir + 'Predicted '+str(self.label[idx])+' Match'+'.tif', bbox_inches = "tight")

    def probability_images (self, original_image, save_input, save_dir=None):
        temp_image = self.y_prob.T
        prob_image  = temp_image.reshape(self.y_prob.shape[1],original_image.shape[1],original_image.shape[2])
        background_image = prob_image[-1,:,:]
        background_mask = np.zeros(background_image.shape)
        background_mask[background_image<0.01]=1
        if save_input:
            for idx in tqdm(range(len(temp_image)-1)):
                image = prob_image[idx]*background_mask
                io.imsave(save_dir +str(self.label[idx])+'-Probability_Figure.tif', (image*100).astype('float32'))
            io.imsave(save_dir +'Bacgkground-Probability_Figure.tif', (background_image*100).astype('float32'))


    def similarity_metrics (self, mol_norm, save_input, save_dir=None):
        MSE_arr = np.zeros(len(self.label))
        similarity_arr = np.zeros(len(self.label))
        correlation_arr = np.zeros(len(self.label))

        for idx in range(len(self.label)-1):
            cur_label = self.x[self.y_pred==idx]
            if cur_label.shape[0] > 0:
                mse_values = []
                similarities = []
                correlations = []
                ymean = cur_label.mean(axis=0)
                for spectrum_idx in tqdm(range(len(cur_label))):
                    mse = calculate_mse(helper_scripts.normalize(mol_norm[idx], np.max(ymean),0), cur_label[spectrum_idx])
                    mse_values.append(mse)
                    similarity = compute_cosine_similarity(helper_scripts.normalize(mol_norm[idx], np.max(ymean),0), cur_label[spectrum_idx])
                    similarities.append(similarity)
                    crosscorrelation = compute_crosscor(helper_scripts.normalize(mol_norm[idx], np.max(ymean),0),  cur_label[spectrum_idx])
                    max_correlation = np.max(crosscorrelation)
                    correlations.append(max_correlation)
                MSE_arr[idx] = np.mean(mse)
                similarity_arr[idx] = np.mean(similarity)
                correlation_arr[idx] = np.mean(max_correlation)
                print('The MSE for ' + self.label[idx] + ' is: ' + str(np.mean(mse)))
                print('The Cosine Similarity for ' + self.label[idx] + ' is: ' + str(np.mean(similarity)))
                print('The Cross-Correlation for ' + self.label[idx] + ' is: ' + str(np.mean(max_correlation)))
        metric_dic = {
            'Molecule': self.label,
            'Average MSE': MSE_arr,
            'Average Cosine': similarity_arr,
            'Average Correlation': correlation_arr
        }
        metric_df = pd.DataFrame(metric_dic)
        if save_input:
            metric_df.to_csv(save_dir+'Metrics.csv', index=False)
            



