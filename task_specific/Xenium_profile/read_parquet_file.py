import os
from shapely.geometry import Polygon
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.draw import polygon
import tifffile
import seaborn as sns


def main():

    file_path = r"C:\Users\jvill\Downloads\ref_cell_boundaries.parquet"
    metadata_path = r"D:\ADATA Backup\HuBMAP\HuBMAP Xenium\Xenium_data\Xen24_KPMP_038_ref-selected\_metadata.csv"

    metadata_list = [
    'low_quality',
    "POD",
    "PT",
    "CNT",
    "DCT",
    "TAL",
    "DTL",
    "ATL",
    "IC",
    "altPOD",
    "altPT",
    "altCNT",
    "altDCT",
    "altTAL",
    "altDTL",
    "altATL",
    "altIC",
    "PC",
    "PEC",
    "MD",
    "EC",
    "EC-GC",
    "FIB",
    "infFIB",
    "pvFIB",
    "aFIB",
    "B",
    "DC",
    "MON",
    "N",
    "NK",
    "PL",
    "T",
    "MAC",
    "resMAC",
    "moMAC",
    "moMAC-INF",
    "MAST",
    "Lymphoid",
    "Myeloid",
    "NEU",
    "SC/NEU",
    "MC",
    "MYOF",
    "REN",
    "VSM/P",
    "unassigned"
    ]

    df = pd.read_parquet(file_path)
    df_x = df['vertex_x']
    df_y = df['vertex_y']

    df_metadata = pd.read_csv(metadata_path)

    diff_x = np.diff(df_x)
    diff_y = np.diff(df_y)

    min_x = np.min(diff_x[diff_x>0])
    min_y = np.min(diff_y[diff_y>0])

    pad_x = 10
    pad_y = 10

    diff_x_rounded = np.round(diff_x / min_x)
    first_x = np.round(df['vertex_x'].iloc[0])
    restored_x = np.concatenate(([first_x], first_x + np.cumsum(diff_x_rounded)))
    restored_x = restored_x - restored_x.min() + pad_x

    diff_y_rounded = np.round(diff_y / min_y)
    first_y = np.round(df['vertex_y'].iloc[0])
    restored_y = np.concatenate(([first_y], first_y + np.cumsum(diff_y_rounded)))
    restored_y = restored_y - restored_y.min() + pad_y

    max_x = int(restored_x.max()) + 1
    max_y = int(restored_y.max()) + 1

    df['x'] = restored_x
    df['y'] = restored_y

    # num_slices = len(metadata_list)
    mask = np.zeros((max_y+(2*pad_y), max_x+(2*pad_x)), dtype=np.uint8)
    # Pre-calculate metadata mapping for fast O(1) lookup
    label_to_idx = {label: i for i, label in enumerate(metadata_list)}
    # Map cell_ID to label index directly
    cell_to_idx = df_metadata.set_index('cell_ID')['levelKID_refined'].map(label_to_idx).to_dict()
    # Default index for "unassigned" if missing (assuming it's at the end or handle accordingly)
    unassigned_idx = label_to_idx.get('unassigned', 0)

#     # Final formatting
    for label_id, group in df.groupby('label_id'):
        # Get cell_id once
        cell_id = group['cell_id'].iloc[0]
        
        # Fast lookup
        cell_label_idx = cell_to_idx.get(cell_id, unassigned_idx)

        # Draw polygon
        r = group['y'].values
        c = group['x'].values

        rr, cc = polygon(r, c, shape=mask.shape)
        mask[rr, cc] = cell_label_idx + 1

    res_x = 1 / float(min_x)
    res_y = 1 / float(min_y)


    name = os.path.basename(file_path).split("_")[0]
    output_dir = os.path.dirname(metadata_path)

#     tif_name = os.path.join(output_dir, f'{name}_cell_segmentation.tif')
#     tifffile.imwrite(
#     tif_name,
#     mask,
#     imagej=True,
#     resolution=(res_x, res_y),
#     resolutionunit=1,
#     metadata={
#         'axes': 'YX',       # Stack as Z slices
#         'unit':  'micron',     # Physical unit string (e.g. 'um', 'micron')
#         'spacing': 1,  # Z spacing reuses Y pixel size
#     },
# )
    
    # Save Dataframe
    df.to_csv(os.path.join(output_dir, f'{name}_cell_segmentation.csv'))


    # Downsample to make a large image
#     downsample_scale = 4
#     ds_restored_x = restored_x / downsample_scale
#     ds_restored_y = restored_y / downsample_scale

#     df['ds_x'] = ds_restored_x
#     df['ds_y'] = ds_restored_y

#     ds_max_x = int(ds_restored_x.max()) + 1
#     ds_max_y = int(ds_restored_y.max()) + 1

#     mask_stack = np.zeros((num_slices, ds_max_y+(2*pad_y), ds_max_x+(2*pad_x)), dtype=np.uint8)
#     # Final formatting
#     for label_id, group in df.groupby('label_id'):
#         # Get cell_id once
#         cell_id = group['cell_id'].iloc[0]
        
#         # Fast lookup
#         cell_label_idx = cell_to_idx.get(cell_id, unassigned_idx)

#         # Draw polygon
#         r = group['ds_y'].values
#         c = group['ds_x'].values

#         rr, cc = polygon(r, c, shape=mask_stack.shape[1:])
#         mask_stack[cell_label_idx, rr, cc] = 255
        
#     stack_name = F'{os.path.basename(file_path).split("_")[0]}_cell_segmentation_stack.tif'

#     res_x = float(min_x * downsample_scale)
#     res_y = float(min_y * downsample_scale)

#     tifffile.imwrite(
#     stack_name,
#     mask_stack,
#     imagej=True,
#     resolution=(res_x, res_y),
#     resolutionunit=1,
#     metadata={
#         'axes': 'ZYX',       # Stack as Z slices
#         'Labels': metadata_list,
#         'unit':  'micron',     # Physical unit string (e.g. 'um', 'micron')
#         'spacing': 1,  # Z spacing reuses Y pixel size
#     },
# )


    # # pt.2 read label mask image and convert to stack of binary masks

    # tif_name = os.path.join(output_dir, 'aki_cell_segmentation-roi2-transformed.ome.tiff')

    # label_mask = tifffile.imread(tif_name)
    # unique_labels = np.unique(label_mask[label_mask > 0])  # Exclude background (0)

    # # create palette of 40+ colors for visualization
    # palette = sns.color_palette("husl", len(unique_labels))

    # mask_stack = np.zeros((len(unique_labels), label_mask.shape[0], label_mask.shape[1]), dtype=np.uint8)
    # for i, label in enumerate(unique_labels):
    #     mask_stack[i] = (label_mask == label).astype(np.uint8) * 255

    # # create RGB composite for visualization
    # rgb_stack = np.zeros((label_mask.shape[0], label_mask.shape[1], 3), dtype=np.uint8)
    # for i, label in enumerate(unique_labels):
    #     rgb_stack[mask_stack[i] > 0] = palette[i]

    # # save images
    # res = 0.4971847
    # res_x = res
    # res_y = res
    # tifffile.imwrite(
    #     os.path.join(output_dir, f'aki_cell_segmentation-roi2n_stack.tif'),
    #     mask_stack,
    #     imagej=True,
    #     resolution=(res_x, res_y),
    #     resolutionunit=1,
    #     metadata={
    #         'axes': 'ZYX',       # Stack as Z slices
    #         'Labels': metadata_list,
    #         'unit':  'micron',     # Physical unit string (e.g. 'um', 'micron')
    #         'spacing': 1,  # Z spacing reuses Y pixel size
    #     },
    # )
    # tifffile.imwrite(
    #     os.path.join(output_dir, f'aki_cell_segmentation-roi2_rgb.tif'),
    #     rgb_stack,
    #     imagej=True,
    #     resolution=(res_x, res_y),
    #     resolutionunit=1,
    #     metadata={
    #         'axes': 'YXC',       # RGB image
    #         'Labels': metadata_list,
    #         'unit':  'micron',     # Physical unit string (e.g. 'um', 'micron')
    #     },
    # )



    
    print("done!")

if __name__ == "__main__":
    main()