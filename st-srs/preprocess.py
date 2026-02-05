import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import binary_dilation, label
from skimage.measure import regionprops

def extract_instance_spectra(mask_tif_path, hsi_tif_path, csv_path, class_col='class_label', slice_col='slice_idx', dilation_iters=2):
	"""
	For each slice in mask_tif, label instances, dilate, and extract spectra from hsi_tif.
	CSV maps slice index to class label. Column names are arguments.
	
	Parameters:
	-----------
	- mask_tif_path: str, path to mask TIFF file
    - hsi_tif_path: str, path to hyperspectral image TIFF file
    - csv_path: str, path to CSV file mapping slice indices to class labels
    - class_col: str, column name in CSV for class labels
    - slice_col: str, column name in CSV for slice indices
    - dilation_iters: int, number of iterations for binary dilation
	
	Returns:
    --------
	- results: list of dicts, each containing:
        - 'spectra': np.ndarray of shape (N, num_wavenumber), spectra for the instance
        - 'class': class label for the instance
        - 'instance_id': int, instance identifier
        - 'slice_idx': int, slice index
	
	Returns list of dicts: {'spectra': np.ndarray, 'class': class_label, 'instance_id': int, 'slice_idx': int}
    """
    # Load mask stack and hyperspectral image
    mask_stack = tifffile.imread(mask_tif_path)
    hsi_img = tifffile.imread(hsi_tif_path)
    # Load CSV mapping slice index to class
    df = pd.read_csv(csv_path)
    results = []
    bbox_shape = None
    # Find bounding box from first instance of first valid slice only
    found_bbox = False
    for slice_idx in range(mask_stack.shape[0]):
        mask = mask_stack[slice_idx]
        labeled_mask, num_instances = label(mask)
        if num_instances > 0:
            instance_mask = (labeled_mask == 1)
            dilated_mask = binary_dilation(instance_mask, iterations=dilation_iters)
            ys, xs = np.where(dilated_mask)
            if len(ys) == 0 or len(xs) == 0:
                continue
            min_y, max_y = ys.min(), ys.max()
            min_x, max_x = xs.min(), xs.max()
            bbox_shape = (max_y - min_y + 1, max_x - min_x + 1)
            found_bbox = True
            break
    if bbox_shape is None:
        return results  # No valid instance found

    box_height, box_width = bbox_shape
    n_wavenumbers = hsi_img.shape[2]

    for slice_idx in range(mask_stack.shape[0]):
        # Get class label for this slice
        class_label = df.loc[df[slice_col] == slice_idx, class_col].values
        if len(class_label) == 0:
            continue  # skip if not mapped
        class_label = class_label[0]
        mask = mask_stack[slice_idx]
        labeled_mask, num_instances = label(mask)
        for instance_id in range(1, num_instances + 1):
            instance_mask = (labeled_mask == instance_id)
            # Find centroid of instance
            props = regionprops(instance_mask.astype(int))
            if not props:
                continue
            cy, cx = np.round(props[0].centroid).astype(int)
            # Center bounding box on centroid
            min_y = max(cy - box_height // 2, 0)
            max_y = min(min_y + box_height, hsi_img.shape[0])
            min_x = max(cx - box_width // 2, 0)
            max_x = min(min_x + box_width, hsi_img.shape[1])
            # Adjust if box goes out of bounds
            if max_y - min_y < box_height:
                min_y = max(max_y - box_height, 0)
            if max_x - min_x < box_width:
                min_x = max(max_x - box_width, 0)
            # Extract spectra
            spectra_box = np.zeros((box_height, box_width, n_wavenumbers), dtype=hsi_img.dtype)
            y1, y2 = min_y, max_y
            x1, x2 = min_x, max_x
            spectra_box[:y2-y1, :x2-x1, :] = hsi_img[slice_idx, y1:y2, x1:x2, :] if hsi_img.ndim == 4 else hsi_img[y1:y2, x1:x2, :]
            results.append({
                'spectra': spectra_box,
                'class': class_label,
                'instance_id': instance_id,
                'slice_idx': slice_idx,
                'bbox': (min_y, max_y, min_x, max_x)
            })
    return results