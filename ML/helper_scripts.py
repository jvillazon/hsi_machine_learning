import numpy as np

def normalize(array, max=1, min=0):
    min_val = np.min(array)
    max_val = np.max(array)


    if min_val == max_val:
        norm = array
    else:
        norm = ((array-min_val)/(max_val-min_val))*(max-min)+min

    return norm

def normalize_manual(array, max_val, min_val, max=1, min=0):

    if np.all(array==0):
        norm = array
    else:
        if min > max:
            min = max
    
        norm = ((array-min_val)/(max_val-min_val))*(max-min)+min


    return norm

def normalize_idx(array, max_idx=-1, min_idx=0, max=1, min=0):

    max_val = array[max_idx]
    min_val = array[min_idx]

    diff = max_val-min_val
    diff[diff<=0] = np.min(diff[diff>0])   

    if np.all(array==0):
        norm = array
    else:
        if min > max:
            min = max
    
        norm = ((array-min_val)/(max_val-min_val))*(max-min)+min

    return norm


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
            