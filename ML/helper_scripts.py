import numpy as np
from tqdm import tqdm
from scipy.signal import savgol_filter

def normalize(array, max=1, min=0):
    min_val = np.min(array)
    max_val = np.max(array)
    
    if min > max:
        min = max
    
    norm = ((array-min_val)/(max_val-min_val))*(max-min)+min

    return norm

def normalize_manual(array, max_val, min_val, max=1, min=0):
    
    if min > max:
        min = max
    
    norm = ((array-min_val)/(max_val-min_val))*(max-min)+min

    return norm

def SRS_CH_background_removal(image_vector, pbs,w=2,p=2):

    img = np.zeros(image_vector.shape)
    mean = np.array(np.median(image_vector, axis=0))

    xdata = np.asarray(list(range(pbs.shape[0])))
    
    for i in tqdm(range(image_vector.shape[0])):
        img_array = image_vector[i,:]
        new_array = savgol_filter(img_array,3.5*w+1, polyorder = 2*p,mode = 'nearest')
        pbs_diff1 = new_array[0] 
        pbs_diff2 = np.mean(new_array[-1:-17:-1])
        background = normalize(pbs, pbs_diff1,pbs_diff2)
        img_sub = new_array-background
        
        img_final = img_sub
        img_flip = np.flip(img_final)
        img[i,:] = img_flip
        
    return img


def SRS_CH_background_removal_normalize(image_vector, pbs,w=2,p=2):

    img_norm = np.zeros(image_vector.shape)
    mean = np.array(np.median(image_vector, axis=0))

    xdata = np.asarray(list(range(pbs.shape[0])))
    
    for i in tqdm(range(image_vector.shape[0])):
        img_array = image_vector[i,:]
        img_temp = normalize(img_array)
        new_array = savgol_filter(img_temp ,3.5*w+1, polyorder = 2*p,mode = 'nearest')
        pbs_diff1 = new_array[0] 
        pbs_diff2 = np.mean(new_array[-1])
        background = normalize(pbs, pbs_diff1,pbs_diff2)
        img_sub = new_array-background
        
        img_final = img_sub
        img_flip = np.flip(img_final)
        #img_new = (1/np.max(np.abs(img_flip)))*img_flip # FOR NORMALIZING TO 0,1
        img_norm[i,:] = normalize_manual(img_flip, np.max(img_flip), np.mean(img_flip[0:17]))
        

    return img_norm

def lip_normalize(lipids):
    
    lip_norm   = np.zeros(lipids.shape) 
    
    for i in range(lipids.shape[0]):
        lip_array = lipids[i,:]
        lip_norm[i,:] = normalize(lip_array)
        
    lip_norm = lip_norm

    return lip_norm

def save_input():
    y_n_input = input('Save into save directory (y/n)?: ').lower()
    if y_n_input == 'y':
        return True
    elif y_n_input == 'n':
        return False
    else:
        raise Exception('Invalid input, please press y/n.')
            