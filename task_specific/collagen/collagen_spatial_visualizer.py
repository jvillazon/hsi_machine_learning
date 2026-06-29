"""
Collagen Spatial Visualization Pipeline

This script computes collagen thickness and coherence maps and generates 
binned spatial distribution maps (heatmaps) similar to the HSI visualizer.

Usage:
    python task_specific/collagen/collagen_spatial_visualizer.py --image <path> --masks <dir> --output <dir>
"""

from IPython.core import display_functions
import os
import sys
import argparse
import pandas as pd
import tifffile
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D  # Register 3D projection

# Add parent directories to path to allow imports from core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.hsi_visualizer import HSI_Visualizer
from task_specific.collagen.collagen_structure_tensor import (
    load_image, compute_structure_tensor, compute_thickness
)

def generate_collagen_spatial_pipeline(image_path, mask_dir, output_dir, 
                                      pixel_size_um=0.65, grid_size=(1000, 1000),
                                      prefix='CODEX', subgroups=None):
    """
    Runs the full collagen spatial visualization pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    
    print(f"--- Starting Collagen Spatial Pipeline for {img_name} ---")
    start_time = time.time()
    
    # 1. Compute Collagen Metrics
    print("\n[Step 1/4] Computing thickness and coherence maps...")
    image = load_image(image_path, threshold_method='none')
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
        
    # Compute structure tensor (coherence)
    _, _, coherence = compute_structure_tensor(
        image, 
        sigma_gradient=0.5,
        sigma_tensor=1.0,
        bin_size=None
    )
    
    # Compute thickness
    thickness_map = compute_thickness(
        image,
        pixel_size_um=pixel_size_um,
        map_type='distance_um',
        threshold_sigma=2.0
    )
    
    # Save temporary maps for quantification
    coherence_tif = os.path.join(output_dir, f"{img_name}-coherence-ratio.tif")
    thickness_tif = os.path.join(output_dir, f"{img_name}-thickness-ratio.tif")
    
    tifffile.imwrite(coherence_tif, coherence.astype(np.float32))
    tifffile.imwrite(thickness_tif, thickness_map.astype(np.float32))
    print(f"  Saved temporary metric maps to {output_dir}")
    
    # 2. Quantify Metrics per Mask Instance
    print("\n[Step 2/4] Quantifying metrics per mask instance...")
    visualizer = HSI_Visualizer(mol_path=None)
    
    # Quantify coherence
    print("  Quantifying coherence...")
    coherence_results = visualizer.apply_rf_masking(
        ratio_tiff_path=coherence_tif,
        mask_list_path=mask_dir,
        prefix=prefix,
        subgroups=subgroups,
        img_name=img_name,
        dilations=10,
        min_instance_size=250
    )
    
    # Quantify thickness
    print("  Quantifying thickness...")
    thickness_results = visualizer.apply_rf_masking(
        ratio_tiff_path=thickness_tif,
        mask_list_path=mask_dir,
        prefix=prefix,
        subgroups=subgroups,
        img_name=img_name,
        dilations=10,
        min_instance_size=250
    )
    
    # 3. Convert to DataFrames
    print("\n[Step 3/4] Converting results to DataFrames...")
    df_coherence = visualizer.quantify_unit_ratio_means_nested(coherence_results)
    df_thickness = visualizer.quantify_unit_ratio_means_nested(thickness_results)

    df_coherence.to_csv(os.path.join(output_dir, "masked_coherence.csv"))
    df_thickness.to_csv(os.path.join(output_dir, "masked_thickness.csv"))

    # 4. Generate Spatial Value Maps
    print("\n[Step 4/4] Generating spatial value maps...")
    
    # Generate map for Coherence
    if not df_coherence.empty:
        print("  Generating coherence spatial maps...")
        visualizer.generate_spatial_value_maps(
            df=df_coherence,
            value_col='Mean_Ratio',
            grouping_col='Ratio_Type',
            output_dir=os.path.join(output_dir, "spatial_maps_coherence"),
            grid_size=grid_size,
            cmap='viridis',
            vmin=0,
            vmax=1.0
        )
    
    # Generate map for Thickness
    if not df_thickness.empty:
        print("  Generating thickness spatial maps...")
        # Get max thickness for scaling
        max_thick = df_thickness['Mean_Ratio'].max() if not df_thickness.empty else 10
        visualizer.generate_spatial_value_maps(
            df=df_thickness,
            value_col='Mean_Ratio',
            grouping_col='Ratio_Type',
            output_dir=os.path.join(output_dir, "spatial_maps_thickness"),
            grid_size=grid_size,
            cmap='magma',
            vmin=0,
            vmax=max_thick
        )
    
    # 5. Generate Directional Profiles
    print("\n[Step 5/4] Generating directional profile plots...")
    profile_coherence_dir = os.path.join(output_dir, "directional_profiles_coherence")
    profile_thickness_dir = os.path.join(output_dir, "directional_profiles_thickness")
    # Combine dataframes for shared profile plots to allow direct comparison
    # combined_df = pd.concat([df_coherence, df_thickness], ignore_index=True)
    # if not combined_df.empty:
    visualizer.generate_directional_profiles(
        df=df_thickness,
        value_col='Mean_Ratio',
        grouping_col='Ratio_Type',
        output_dir=profile_coherence_dir,
        bin_ratio=10,
        normalize_auc=True,
        directions=['y']
    )    
    visualizer.generate_directional_profiles(
        df=df_thickness,
        value_col='Mean_Ratio',
        grouping_col='Ratio_Type',
        output_dir=profile_coherence_dir,
        bin_ratio=10,
        normalize_auc=True,
        directions=['x']
    )

    visualizer.generate_directional_profiles(
        df=df_coherence,
        value_col='Mean_Ratio',
        grouping_col='Ratio_Type',
        output_dir=profile_thickness_dir,
        bin_ratio=10,
        normalize_auc=True,
        directions=['x']
    )
    visualizer.generate_directional_profiles(
        df=df_coherence,
        value_col='Mean_Ratio',
        grouping_col='Ratio_Type',
        output_dir=profile_thickness_dir,
        bin_ratio=10,
        normalize_auc=True,
        directions=['y']
    )

    # Cleanup temporary files if desired (optional)

    # Cleanup temporary files if desired (optional)
    # os.remove(coherence_tif)
    # os.remove(thickness_tif)
    
    elapsed = time.time() - start_time
    print(f"\n--- Pipeline completed in {elapsed:.2f} seconds ---")
    print(f"Results saved to: {output_dir}")

def crop_to_content(image):
    """
    Crops the image to remove all-zero rows and columns.
    """
    print("  Cropping image to remove all-zero background padding...")
    # Find indices where rows/cols have non-zero values
    non_zero_rows = np.where(np.any(image != 0, axis=1))[0]
    non_zero_cols = np.where(np.any(image != 0, axis=0))[0]
    
    if len(non_zero_rows) == 0 or len(non_zero_cols) == 0:
        print("    Warning: Image is entirely zero. No cropping performed.")
        return image
        
    row_start, row_end = non_zero_rows[0], non_zero_rows[-1]
    col_start, col_end = non_zero_cols[0], non_zero_cols[-1]
    
    cropped = image[row_start:row_end+1, col_start:col_end+1]
    print(f"    Cropped from {image.shape} to {cropped.shape}")
    return cropped

def visualize_3d_topography(data, output_path, title="Topography Visualization", 
                            cmap='magma', max_pixels=1_000_000, mode='both', 
                            auto_crop=True):
    """
    Generate 3D topography visualizations from a 2D map (e.g., thickness or ratio).
    
    Args:
        data (ndarray or str): 2D array of values, or path to a TIFF file.
        output_path (str): Base path/filename for saving the images.
        title (str): Title for the plots.
        cmap (str): Colormap to use.
        max_pixels (int): Maximum pixels for the 3D surface plot.
        mode (str): 'surface', 'hillshade', or 'both'.
        auto_crop (bool): If True, removes all-zero background rows/columns.
    """
    # Load data if path provided
    if isinstance(data, str):
        print(f"Loading image for 3D visualization: {data}")
        data = tifffile.imread(data)
    
    if data.ndim != 2:
        print(f"Error: 3D visualization requires a 2D image, but got shape {data.shape}")
        return

    # Auto-crop background
    if auto_crop:
        data = crop_to_content(data)

    h, w = data.shape
    
    # 1. Perspective 3D Surface Plot
    if mode in ['surface', 'both']:
        # Downsample for 3D surface plotting
        stride = int(np.ceil(np.sqrt((h * w) / max_pixels)))
        if stride > 1:
            print(f"  Downsampling map by {stride}x for 3D surface visualization...")
            z = data[::stride, ::stride]
        else:
            z = data
            
        ny, nx = z.shape
        x = np.linspace(0, w, nx)
        y = np.linspace(0, h, ny)
        xv, yv = np.meshgrid(x, y)
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the surface
        surf = ax.plot_surface(xv, yv, z, cmap=cmap, linewidth=0, antialiased=False, shade=True)
        
        ax.set_title(f"{title} (3D Surface)", fontsize=16, fontweight='bold')
        ax.set_xlabel('X [pixels]')
        ax.set_ylabel('Y [pixels]')
        ax.set_zlabel('Value')
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        
        # Set a nice initial view angle
        ax.view_init(elev=45, azim=-45)
        
        surface_out = output_path.replace('.png', '_3d_surface.png')
        plt.savefig(surface_out, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved 3D surface plot to: {surface_out}")

    # 2. Hillshaded Topography (2D but looks 3D)
    if mode in ['hillshade', 'both']:
        # Higher resolution for hillshading
        stride_hs = int(np.ceil(np.sqrt((h * w) / 10_000_000)))
        if stride_hs > 1:
            z_hs = data[::stride_hs, ::stride_hs]
        else:
            z_hs = data

        ls = LightSource(azdeg=315, altdeg=45)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        rgb = ls.shade(z_hs, cmap=plt.get_cmap(cmap), blend_mode='soft', 
                       vert_exag=1.0, dx=1, dy=1)
        
        ax.imshow(rgb, extent=[0, w, h, 0])
        ax.set_title(f"{title} (Hillshaded)", fontsize=16, fontweight='bold')
        ax.axis('off')
        
        hillshade_out = output_path.replace('.png', '_hillshade.png')
        plt.savefig(hillshade_out, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved hillshaded topography to: {hillshade_out}")

def main():
    # --- STANDALONE 3D VISUALIZATION EXAMPLE ---
    # This can be used for any 2D TIFF map (thickness, ratio, etc.)
    
    # Example for a thickness map
    image_path = r"C:\Users\jvill\Documents\integrated_pipeline\journal_figures\proximity\S21-K21_198_3-proximity\collagen\S21-K21_198_3-collagen-thickness-ratio.tif"
    output_dir = r"C:\Users\jvill\Documents\integrated_pipeline\journal_figures\proximity\S21-K21_198_3-proximity\collagen\topography"
    os.makedirs(output_dir, exist_ok=True)
    
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    topo_output = os.path.join(output_dir, f"{img_name}_topography.png")
    
    print(f"--- Starting Standalone 3D Topography for {img_name} ---")
    visualize_3d_topography(
        data=image_path,
        output_path=topo_output,
        title=f"Collagen Thickness Topography - {img_name}",
        cmap='magma',
        auto_crop=True
    )
    
    # --- PIPELINE EXAMPLE (Uncomment to run full mask-based pipeline) ---
    r"""
    image_path = r"C:\Users\jvill\Documents\integrated_pipeline\WSI_data\SRS\S21-K21_198_3-collagen.tif"
    mask_dir = r"C:\Users\jvill\Documents\integrated_pipeline\WSI_data\WSI_data\CODEX_FTUcells_masks\S21-K21_198_3"
    output_dir = r"C:\Users\jvill\Documents\integrated_pipeline\journal_figures\proximity\S21-K21_198_3-proximity\collagen"
    pixel_size = 0.65
    grid_size = (1000,100)
    mask_prefix = "CODEX"
    subgroups = None
    
    generate_collagen_spatial_pipeline(
        image_path=image_path,
        mask_dir=mask_dir,
        output_dir=output_dir,
        pixel_size_um=pixel_size,
        grid_size=grid_size,
        prefix=mask_prefix,
        subgroups=subgroups
    )
    """

if __name__ == "__main__":
    main()
