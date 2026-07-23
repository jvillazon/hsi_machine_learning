import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import tifffile
from glob import glob
import os
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.ndimage as ndi
from scipy.stats import ttest_ind, spearmanr
from tqdm import tqdm

# Colormap extracted from bubble plot SVG (blue → grey → red diverging)
_SVG_BUBBLE_CMAP_COLORS = [
    '#273478', '#3d50b7', '#7483d5', '#b2b8de',
    '#d3d3d3', '#d3d3d3',
    '#e4a9b7', '#e36381', '#c52b50', '#811a33',
]
SVG_BUBBLE_CMAP = LinearSegmentedColormap.from_list(
    'svg_bubble', _SVG_BUBBLE_CMAP_COLORS, N=256
)

class HSI_Visualizer:
    # Default color palette - colorblind-friendly
    DEFAULT_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def __init__(self, mol_path=None, wavenumber_start=2700, wavenumber_end=3100, num_samples=61):
        """
        Initialize HSI Visualizer
        Args:
            mol_path: Optional path to molecule dataset (.npz)
            wavenumber_start: Starting wavenumber for molecule dataset (default 2700)
            wavenumber_end: Ending wavenumber for molecule dataset (default 3100)
            num_samples: Number of samples in wavenumber range (default 61)
        """
        self.wavenumber_start = wavenumber_start
        self.wavenumber_end = wavenumber_end
        self.num_samples = num_samples

        if mol_path:
            print(f"Loading molecules from: {mol_path}")
            mol_data = np.load(mol_path, allow_pickle=True)
            self.normalized_molecules = mol_data['normalized_molecules']
            self.molecule_names = mol_data['molecule_names']
            print(f"HSI Visualizer initialized with {len(self.molecule_names)} molecules.")
        else:
            self.normalized_molecules = None
            self.molecule_names = None
            print("HSI Visualizer initialized without molecule reference.")
    
    @staticmethod
    def _hex_to_rgb_normalized(hex_color):
        """Convert hex color to normalized RGB tuple (0-1 range)."""
        r = int(hex_color[1:3], 16) / 255.0
        g = int(hex_color[3:5], 16) / 255.0
        b = int(hex_color[5:7], 16) / 255.0
        return (r, g, b)
    
    @staticmethod
    def _create_lighter_shade(hex_color, blend_factor=0.5):
        """Create a lighter shade by blending with white."""
        r, g, b = HSI_Visualizer._hex_to_rgb_normalized(hex_color)
        lighter_color = (r * blend_factor + (1 - blend_factor), 
                        g * blend_factor + (1 - blend_factor), 
                        b * blend_factor + (1 - blend_factor))
        return lighter_color
    
    @staticmethod
    def _create_color_map(units, unit_color_map=None):
        """
        Create color mapping for units.
        
        Args:
            units: List of unit names
            unit_color_map: Optional dict mapping unit names to hex colors
            
        Returns:
            Dictionary mapping unit names to hex colors
        """
        if unit_color_map:
            return {unit: unit_color_map.get(unit, HSI_Visualizer.DEFAULT_PALETTE[i % len(HSI_Visualizer.DEFAULT_PALETTE)]) 
                   for i, unit in enumerate(units)}
        else:
            colors = [HSI_Visualizer.DEFAULT_PALETTE[i % len(HSI_Visualizer.DEFAULT_PALETTE)] 
                     for i in range(len(units))]
            return {unit: color for unit, color in zip(units, colors)}
    
    @staticmethod
    def _apply_display_name(name, display_name_map):
        """Apply display name mapping if provided."""
        if display_name_map:
            return display_name_map.get(name, name)
        return name
    
    @staticmethod
    def _calculate_upper_whisker_values(plot_data):
        """
        Calculate upper whisker values for box plots.
        Upper whisker = Q3 + 1.5*IQR or max value within range.
        
        Args:
            plot_data: List of arrays containing data for each box
            
        Returns:
            List of upper whisker values
        """
        upper_whisker_values = []
        for d in plot_data:
            if len(d) > 0:
                q1 = np.percentile(d, 25)
                q3 = np.percentile(d, 75)
                iqr = q3 - q1
                upper_whisker = q3 + 1.5 * iqr
                non_outliers = d[d <= upper_whisker]
                if len(non_outliers) > 0:
                    upper_whisker_values.append(np.max(non_outliers))
                else:
                    upper_whisker_values.append(q3)
            else:
                upper_whisker_values.append(0)
        return upper_whisker_values

    @staticmethod
    def _save_standalone_legend(patches, title, output_path):
        """
        Creates and saves a standalone legend image.
        """
        import matplotlib.pyplot as plt
        import os
        
        # Create parent directory if it doesn't exist
        parent_dir = os.path.dirname(output_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
            
        fig = plt.figure(figsize=(3, 2))
        fig.legend(handles=patches, title=title, loc='center', frameon=True)
        plt.axis('off')
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.close(fig)
        # print(f"    Saved standalone legend to {output_path}")

    @staticmethod
    def _apply_outlier_filtration(df, value_col, group_cols):
        """
        Apply 1.5 * IQR outlier filtration per group.
        
        Args:
            df: DataFrame to filter
            value_col: Column containing values to filter
            group_cols: Columns to group by for filtration logic
            
        Returns:
            Filtered DataFrame
        """
        if df.empty:
            return df
            
        # Group by specified columns and calculate IQR boundaries for each group
        groups = df.groupby(group_cols, observed=True)[value_col]
        
        # Calculate Q1, Q3 per group and broadcast back to original shape
        q1 = groups.transform(lambda x: x.quantile(0.25))
        q3 = groups.transform(lambda x: x.quantile(0.75))
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Use .values for comparison to avoid "ValueError: Can only compare identically-labeled Series objects"
        # which occurs when the DataFrame has a non-unique index (common when concatenating images).
        mask = (df[value_col].values >= lower_bound.values) & (df[value_col].values <= upper_bound.values)
        
        return df[mask]

    def _adjust_color_for_group(self, hex_color, group_id, subgroup_list=None):
        """
        Adjust a hex color slightly based on group ID to distinguish subgroups visually.
        
        If subgroup_list is provided, it uses a dynamic gradient based on index.
        Otherwise, it falls back to keyword matching ('Cortex' -> lighter, 'Medulla' -> darker).
        
        Args:
            hex_color: Hex color string (e.g., '#1f77b4')
            group_id: Group identifier (e.g., 'CortexInner', 'M')
            subgroup_list: Optional list of all subgroups to calculate dynamic scaling.
            
        Returns:
            Adjusted hex color string
        """
        import matplotlib.colors as mcolors
        adjust_limit = 0.5  # Max adjustment (50% lighter/darker)

        # Check if we can use dynamic scaling based on index in list
        if subgroup_list and group_id in subgroup_list:
            n = len(subgroup_list)
            if n > 1:
                idx = subgroup_list.index(group_id)
                # Linear gradient: first item is +0.5 (lighter), last item is -0.5 (darker)
                adjustment = adjust_limit - (idx / (n - 1))
            else:
                adjustment = 0.0
        else:
            # Fallback to keyword matching logic (supports initials or full names)
            if not group_id:
                adjustment = 0.0
            elif 'Cortex' in str(group_id) or str(group_id) == 'C':
                adjustment = adjust_limit
            elif 'Medulla' in str(group_id) or str(group_id) == 'M':
                adjustment = -1 * adjust_limit
            elif str(group_id) == 'A':
                adjustment = adjust_limit
            elif str(group_id) == 'B':
                adjustment = -1 * adjust_limit
            else:
                adjustment = 0.0
        
        # Convert hex to RGB
        rgb = mcolors.hex2color(hex_color)
        
        # Adjust brightness
        if adjustment > 0:
            # Lighten: move towards white
            adjusted_rgb = tuple(min(1.0, c + (1 - c) * adjustment) for c in rgb)
        else:
            # Darken: move towards black
            adjusted_rgb = tuple(max(0.0, c + c * adjustment) for c in rgb)
        
        # Convert back to hex
        return mcolors.rgb2hex(adjusted_rgb)

    def _get_hatch_for_group(self, group_id, subgroup_list=None):
        """
        Get a hatch pattern based on group ID to distinguish subgroups visually.
        
        Args:
            group_id: Group identifier (e.g., 'CortexInner', 'M')
            subgroup_list: Optional list of all subgroups to ensure consistent indexing.
            
        Returns:
            Hatch string (e.g., '//', '///', '////')
        """
        # Define available patterns with increasing density
        patterns = ['/', '//', '///', '////', '/////', '//////', '///////']
        
        if not group_id:
            return None
            
        if subgroup_list and group_id in subgroup_list:
            n = len(subgroup_list)
            if n > 0:
                idx = subgroup_list.index(group_id)
                # Ensure we don't exceed patterns list; wrap if necessary but usually we have enough
                return patterns[idx % len(patterns)]
        
        # Fallback keyword matching with density
        if 'Cortex' in str(group_id) or str(group_id) == 'C':
            return '//'
        elif 'Medulla' in str(group_id) or str(group_id) == 'M':
            return '////'
        
        return '/'


    def visualize_spectrum(self, spectrum, prediction=None, probabilities=None, img_idx=None):
        """
        Visualize a single spectrum with optional molecule reference and probabilities
        
        Args:
            spectrum: The spectrum to visualize (numpy array or torch tensor)
            prediction: Optional class prediction for this spectrum (class index)
            probabilities: Optional class probabilities for this spectrum
            img_idx: Optional image index for title
        """
        
        # Convert to numpy if it's a tensor
        if hasattr(spectrum, 'numpy'):
            spectrum = spectrum.numpy()
        if hasattr(probabilities, 'numpy'):
            probabilities = probabilities.numpy()
            
        wavenumber = np.linspace(self.wavenumber_start, self.wavenumber_end, self.num_samples)
        
        if probabilities is not None and prediction is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
            
            # Get sample spectrum maximum
            sample_max = np.max(spectrum)
            
            # Plot sample spectrum
            ax1.plot(wavenumber, spectrum, label='Sample Spectrum', color='blue')
        
            # Get molecule spectrum and scale to match sample maximum
            molecule_spectrum = self.normalized_molecules[prediction]
            molecule_max = np.max(molecule_spectrum)
            scaled_molecule = molecule_spectrum * (sample_max / (molecule_max + 1e-8))
            
            ax1.plot(wavenumber, scaled_molecule, '--', 
                    label=f'Reference: {self.molecule_names[prediction]}',
                    color='red', alpha=0.7)
            ax1.legend()
            
            title = "Sample Spectrum"
            if img_idx is not None:
                title += f" from Image {img_idx+1}"
            if prediction is not None and self.molecule_names is not None:
                title += f"\nPredicted: {self.molecule_names[prediction]}"
            ax1.set_title(title)
            ax1.set_xlabel("Wavenumber (cm⁻¹)")
            ax1.set_ylabel("Normalized Intensity")
            
            # Plot probabilities with molecule names
            if self.molecule_names is not None:
                # Ensure we only use molecule names for classes present in probabilities
                valid_names = self.molecule_names[:len(probabilities)]
                x_ticks = range(len(probabilities))
                ax2.bar(x_ticks, probabilities)
                ax2.set_xticks(x_ticks)
                ax2.set_xticklabels(valid_names, rotation=45, ha='center')
            else:
                ax2.bar(range(len(probabilities)), probabilities)
                ax2.set_xlabel("Class")
            ax2.set_title("Class Probabilities")
            ax2.set_ylabel("Probability")
        else:
            plt.figure(figsize=(10, 4))
            plt.plot(wavenumber, spectrum, label='Sample Spectrum')
            title = "Sample Spectrum"
            if img_idx is not None:
                title += f" from Image {img_idx+1}"
            plt.title(title)
            plt.xlabel("Wavenumber (cm⁻¹)")
            plt.ylabel("Normalized Intensity")
        
        plt.tight_layout()
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
    
    def visualize_spectrum_in_axes(self, ax_spectrum, ax_prob, spectrum, prediction=None, probabilities=None, img_idx=None):
        """
        Visualize a single spectrum in provided axes objects (for subplot composition)
        
        Args:
            ax_spectrum: Matplotlib axes object for spectrum plot
            ax_prob: Matplotlib axes object for probability bar chart
            spectrum: The spectrum to visualize (numpy array or torch tensor)
            prediction: Optional class prediction for this spectrum (class index)
            probabilities: Optional class probabilities for this spectrum
            img_idx: Optional image index for title
        """
        
        # Convert to numpy if it's a tensor
        if hasattr(spectrum, 'numpy'):
            spectrum = spectrum.numpy()
        if hasattr(probabilities, 'numpy'):
            probabilities = probabilities.numpy()
            
        wavenumber = np.linspace(self.wavenumber_start, self.wavenumber_end, self.num_samples)
        
        # Get sample spectrum maximum
        sample_max = np.max(spectrum)
        
        # Plot sample spectrum
        ax_spectrum.plot(wavenumber, spectrum, label='Sample Spectrum', color='blue', linewidth=2)
    
        if prediction is not None:
            # Get molecule spectrum and scale to match sample maximum
            molecule_spectrum = self.normalized_molecules[prediction]
            molecule_max = np.max(molecule_spectrum)
            scaled_molecule = molecule_spectrum * (sample_max / (molecule_max + 1e-8))
            
            ax_spectrum.plot(wavenumber, scaled_molecule, '--', 
                    label=f'Reference: {self.molecule_names[prediction]}',
                    color='red', alpha=0.7, linewidth=2)
        
        ax_spectrum.legend(fontsize=9)
        
        title = "Sample Spectrum"
        if img_idx is not None:
            title += f" from Image {img_idx+1}"
        if prediction is not None and self.molecule_names is not None:
            prob_str = f" (p={probabilities[prediction]:.3f})" if probabilities is not None else ""
            title += f" ΓåÆ {self.molecule_names[prediction]}{prob_str}"
        ax_spectrum.set_title(title, fontsize=11, fontweight='bold')
        ax_spectrum.set_xlabel("Wavenumber (cmΓü╗┬╣)", fontsize=10)
        ax_spectrum.set_ylabel("Normalized Intensity", fontsize=10)
        ax_spectrum.grid(True, alpha=0.3)
        
        # Plot probabilities if provided
        if probabilities is not None and prediction is not None:
            sorted_indices = np.argsort(probabilities)[::-1][:5]
            sorted_probs = probabilities[sorted_indices]
            sorted_names = [self.molecule_names[i] for i in sorted_indices]
            
            colors = ['green' if i == prediction else 'gray' for i in sorted_indices]
            ax_prob.barh(range(len(sorted_probs)), sorted_probs, color=colors)
        if img_idx is not None:
            title += f" from Image {img_idx+1}"
        if prediction is not None and self.molecule_names is not None:
            prob_str = f" (p={probabilities[prediction]:.3f})" if probabilities is not None else ""
            title += f" → {self.molecule_names[prediction]}{prob_str}"
        ax_spectrum.set_title(title, fontsize=11, fontweight='bold')
        ax_spectrum.set_xlabel("Wavenumber (cm⁻¹)", fontsize=10)
        ax_spectrum.set_ylabel("Normalized Intensity", fontsize=10)
        ax_spectrum.grid(True, alpha=0.3)
        
        # Plot probabilities if provided
        if probabilities is not None and prediction is not None:
            sorted_indices = np.argsort(probabilities)[::-1][:5]
            sorted_probs = probabilities[sorted_indices]
            sorted_names = [self.molecule_names[i] for i in sorted_indices]
            
            colors = ['green' if i == prediction else 'gray' for i in sorted_indices]
            ax_prob.barh(range(len(sorted_probs)), sorted_probs, color=colors)

    def plot_spectral_grid(self, spectra_list, class_idx, titles=None, output_path=None, figsize=(18, 12)):
        """
        Plot a grid of spectra overlaid with a reference spectrum.
        
        Args:
            spectra_list: List of numpy arrays (n_wavenumbers,)
            class_idx: Index of the class for the reference overlay
            titles: Optional list of titles for each subplot
            output_path: Optional path to save the plot
            figsize: Tuple for figure size
        """
        import matplotlib.pyplot as plt
        n_samples = len(spectra_list)
        if n_samples == 0:
            print("No spectra to plot.")
            return None
            
        n_cols = int(np.ceil(np.sqrt(n_samples)))
        n_rows = int(np.ceil(n_samples / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True)
        if n_samples == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        wavenumber = np.linspace(self.wavenumber_start, self.wavenumber_end, self.num_samples)
        
        if self.normalized_molecules is not None and class_idx < len(self.normalized_molecules):
            reference_spectrum = self.normalized_molecules[class_idx]
            class_name = self.molecule_names[class_idx]
            ref_max = np.max(reference_spectrum)
        else:
            reference_spectrum = None
            class_name = f"Class {class_idx}"
        
        for i in range(n_samples):
            ax = axes[i]
            spec = spectra_list[i]
            
            # Scale to [0, 1] for visual comparison if possible
            spec_min = np.min(spec)
            spec_max = np.max(spec)
            spec_norm = (spec - spec_min) / (spec_max - spec_min + 1e-8)
            
            ax.plot(wavenumber, spec_norm, 'b-', label='Predicted Pixel', alpha=0.8, linewidth=1.5)
            
            if reference_spectrum is not None:
                # Reference is already normalized in Labeled_Dataset usually
                ref_min = np.min(reference_spectrum)
                ref_max_val = np.max(reference_spectrum)
                ref_norm = (reference_spectrum - ref_min) / (ref_max_val - ref_min + 1e-8)
                ax.plot(wavenumber, ref_norm, 'r--', label=f'Ref: {class_name}', alpha=0.6, linewidth=1.5)
            
            if titles and i < len(titles):
                ax.set_title(titles[i], fontsize=10)
            
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(fontsize=8)
        
        # Hide unused axes
        for i in range(n_samples, len(axes)):
            axes[i].axis('off')
            
        fig.suptitle(f"Representative Predictions for '{class_name}'", fontsize=20, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if output_path:
            plt.savefig(output_path, dpi=600, bbox_inches='tight')
            print(f"Saved spectral grid plot to {output_path}")
            
        plt.show()
        return fig

    def plot_spectral_averages_grid(self, class_stats_list, output_path=None, figsize=(18, 12)):
        """
        Plot a grid of class average spectra with standard deviation shading.
        
        Args:
            class_stats_list: List of dicts, each with {name, mean, std, reference, n_pixels}
            output_path: Optional path to save the plot
            figsize: Tuple for figure size
        """
        import matplotlib.pyplot as plt
        n_samples = len(class_stats_list)
        if n_samples == 0:
            print("No class stats to plot.")
            return None
            
        n_cols = int(np.ceil(np.sqrt(n_samples)))
        n_rows = int(np.ceil(n_samples / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True)
        if n_samples == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        wavenumber = np.linspace(self.wavenumber_start, self.wavenumber_end, self.num_samples)
        
        for i in range(n_samples):
            ax = axes[i]
            stats = class_stats_list[i]
            
            name = stats['name']
            mean = stats['mean']
            std = stats['std']
            ref = stats['reference']
            n_pixels = stats['n_pixels']
            
            # Normalize for visualization [0, 1]
            v_min = np.min(mean)
            v_max = np.max(mean)
            norm_mean = (mean - v_min) / (v_max - v_min + 1e-8)
            norm_std = std / (v_max - v_min + 1e-8)
            
            ax.plot(wavenumber, norm_mean, 'b-', label='Mean Predicted', linewidth=2)
            ax.fill_between(wavenumber, norm_mean - norm_std, norm_mean + norm_std, color='b', alpha=0.2, label='±1 Std')
            
            if ref is not None:
                r_min = np.min(ref)
                r_max = np.max(ref)
                norm_ref = (ref - r_min) / (r_max - r_min + 1e-8)
                ax.plot(wavenumber, norm_ref, 'r--', label='Reference', alpha=0.7, linewidth=1.5)
            
            ax.set_title(f"{name}\n(n={n_pixels} pixels)", fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(fontsize=9)
        
        # Hide unused axes
        for i in range(n_samples, len(axes)):
            axes[i].axis('off')
            
        fig.suptitle("Predicted Class Averages (Mean ± Std)", fontsize=22, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if output_path:
            plt.savefig(output_path, dpi=600, bbox_inches='tight')
            print(f"Saved class averages grid plot to {output_path}")
            
        plt.show()
        return fig

    def create_prediction_csv(self, img_predictions, img_shape, img_path, output_path):
        """
        Reconstruct predictions for a specific image as a pandas DataFrame
        
        Args:
            img_predictions: Array of string predictions for a single image
            img_path: Path of the image to reconstruct
            
        Returns:
            df: pandas DataFrame with predictions arranged in the original image shape
        """
      
        # Get image shape
        height = img_shape[0]
        width = img_shape[1]

        # Reshape predictions to match image dimensions
        pred_matrix = img_predictions.reshape(height, width)
        
        # Create DataFrame
        df = pd.DataFrame(pred_matrix)
        
        # Map numerical predictions to molecule names if available
        if self.molecule_names is not None:
            df = df.replace(dict(enumerate(self.molecule_names)))
        
        try:
            df.to_csv(output_path, index=False, header=False)
        except Exception as e:
            print(f"Warning: failed to save CSV for {img_path}: {e}")
        

    def create_probability_stack(self, img_probabilities, img_shape, img_path, output_path, stats):
        """
        Reconstruct probability stack for a specific image and save as TIFF.

        The output stack inherits the spatial resolution (pixel size) and unit
        from the source image so that it opens in ImageJ with the same physical
        calibration as the original HSI image.

        Args:
            img_probabilities: Array of probabilities for a single image (n_pixels, n_classes)
            img_path: Path of the source image (used only for error messages)
            output_path: Path to save the probability stack TIFF
            stats: Dictionary of image statistics from HSI_Unlabeled_Dataset,
                   expected to contain 'pixel_size_x', 'pixel_size_y',
                   'resolution_unit', and 'ij_unit'.
        """

        # Get image shape
        height = img_shape[0]
        width = img_shape[1]
        
        # Reshape probabilities to (n_classes, height, width)
        n_classes = img_probabilities.shape[1]
        prob_stack = img_probabilities.reshape((height, width, n_classes))
        prob_stack = np.transpose(prob_stack, (2, 0, 1))  # Move classes to first dimension
        
        # Use molecule names for channel labels if available from visualizer
        if self.molecule_names is not None:
            # Convert numpy array to list of strings
            channel_names = [str(name) for name in self.molecule_names]
        else:
            channel_names = [str(c) for c in range(n_classes)]

        pixel_size_x = stats.get('pixel_size_x', 1)
        pixel_size_y = stats.get('pixel_size_y', 1)
        resolution_unit = stats.get('resolution_unit', 1)  # TIFF ResolutionUnit tag int
        ij_unit = stats.get('ij_unit', 'micron')           # ImageJ unit string

        res_x = (1.0 / pixel_size_x) if pixel_size_x and pixel_size_x != 1 else 1.0
        res_y = (1.0 / pixel_size_y) if pixel_size_y and pixel_size_y != 1 else 1.0

        # Save as TIFF with individual page labels
        try:
            # ImageJ format requires writing entire stack at once.
            # Pass resolution=(pixels/unit in X, pixels/unit in Y) and the
            # matching resolutionunit tag so the calibration survives round-trips.
            tifffile.imwrite(
                output_path,
                prob_stack.astype(np.float32),
                imagej=True,
                resolution=(res_x, res_y),
                resolutionunit=resolution_unit,
                metadata={
                    'axes': 'ZYX',       # Stack as Z slices
                    'Labels': channel_names,
                    'unit': ij_unit,     # Physical unit string (e.g. 'um', 'micron')
                    'spacing': pixel_size_y,  # Z spacing reuses Y pixel size
                },
            )
            
            print(f"Saved probability stack with {n_classes} channels to: {output_path}")
            
        except Exception as e:
            print(f"Warning: failed to save probability stack for {img_path}: {e}")



    def _instance_masking(self, mask, dilations=2, min_instance_size=250):
        """
        Apply dilation and size filtering to an instance mask.
        
        Args:
            mask: 2D numpy array of the instance mask (binary or labeled)
            dilations: Number of dilation iterations to apply
            min_instance_size: Minimum size (in pixels) for an instance to be kept

        Returns:
            List of dicts with instance metadata (label, padded_slice, instance_mask, centroid_x, centroid_y)
        """
        # Label mask for instance segmentation
        labeled_mask, num_features = ndi.label(mask > 0)
        if num_features == 0:
            return None
        
        # Identify valid instances and pre-calculate their metadata
        object_slices = ndi.find_objects(labeled_mask)
        instance_metadata = []
        
        for label, obj_slice in enumerate(object_slices, start=1):
            if obj_slice is None:
                continue

            y_slice, x_slice = obj_slice
            y_start = max(0, y_slice.start - dilations)
            y_end = min(labeled_mask.shape[0], y_slice.stop + dilations)
            x_start = max(0, x_slice.start - dilations)
            x_end = min(labeled_mask.shape[1], x_slice.stop + dilations)
            
            padded_slice = (slice(y_start, y_end), slice(x_start, x_end))
            local_labels = labeled_mask[padded_slice]
            instance_mask = local_labels == label

            if int(np.sum(instance_mask)) < min_instance_size:
                continue

            # Dilate mask if requested (using original logic: dilations-1)
            if dilations > 0:
                instance_mask = ndi.binary_dilation(instance_mask, iterations=dilations-1)

            # Centroids
            rows, cols = np.nonzero(local_labels == label)
            rows += padded_slice[0].start
            cols += padded_slice[1].start
            centroid_y = int(np.mean(rows))
            centroid_x = int(np.mean(cols))

            instance_metadata.append({
                'label': label,
                'padded_slice': padded_slice,
                'instance_mask': instance_mask,
                'centroid_x': centroid_x,
                'centroid_y': centroid_y
            })

        return instance_metadata

    def apply_rf_masking(self, prediction_csv_path=None, ratio_tiff_path=None, mask_list_path=None, prefix='CODEX', results_per_unit=None, 
                        subgroups=None, img_name=None, Source_ID=None, classes_to_ignore=None, group_subclasses=False,dilations=2, min_instance_size=250,
                        mask_dict=None):
        """
        Apply masks and quantify predictions or ratios for each instance.
        
        Args:
            prediction_csv_path: Path to prediction CSV file (for prediction quantification)
            ratio_tiff_path: Path to ratio TIFF file (for ratio quantification)
            mask_list_path: Path to folder containing mask TIFF files
            prefix: Prefix for mask filenames (default 'CODEX')
            results_per_unit: Dictionary to store results per unit (optional, will be created if None)
            stats: Dictionary of image statistics
            subgroups: List of subgroup names (formerly regions) for fallback extraction from filenames
            img_name: Image name (without extension)
            Source_ID: Explicit source name (formerly sample_name)
            classes_to_ignore: List of class names to ignore in quantification (default: ['Masked', 'Kidney Background', 'No Match'])
            group_subclasses: If True, consolidates counts of subclasses into their overarching ' Mix' superclass
            mask_dict: Optional dictionary mapping unit names to mask stack slices (keys are indeces in mask_img)


        Returns:
            Dictionary with quantified data per unit and instance
        """
        if prediction_csv_path is None and ratio_tiff_path is None:
            raise ValueError("Must provide either prediction_csv_path or ratio_tiff_path")
        
        if prediction_csv_path is not None and ratio_tiff_path is not None:
            raise ValueError("Provide only one of prediction_csv_path or ratio_tiff_path, not both")
    


        # Determine mode
        is_prediction_mode = prediction_csv_path is not None

        # Set default classes to ignore if not provided
        if classes_to_ignore is None and is_prediction_mode:
            classes_to_ignore = ['Masked', 'Kidney Background', 'No Match']
            classes_to_ignore_set = {cls.strip() for cls in classes_to_ignore}
        
        # Load data matrix or prepare TIFF handle
        if is_prediction_mode:
            # Load CSV as matrix (no header, no index) - each cell is a predicted class name
            data_df = pd.read_csv(prediction_csv_path, header=None, index_col=False)
            data_matrix = data_df.values  # Convert to numpy array for easier indexing
            target_shape = data_matrix.shape
            if img_name is None:
                img_name = os.path.basename(prediction_csv_path).replace('_predictions.csv', '')
        else: 
            # In ratio mode, we use TiffFile for memory efficiency
            tif = tifffile.TiffFile(ratio_tiff_path)
            num_pages = len(tif.pages)
            target_shape = tif.pages[0].shape
            
            if img_name is None:
                img_name = os.path.splitext(os.path.basename(ratio_tiff_path))[0]
                # Remove ratio type suffix (e.g., "image-Lipid_to_Protein-ratio" -> "image")
                if '-ratio' in img_name:
                    img_name = img_name.split('-')[0]
            
            # Extract ratio type from filename
            ratio_basename = os.path.splitext(os.path.basename(ratio_tiff_path))[0]
            ratio_type = ratio_basename.replace(f"{img_name}-", "").replace("-ratio", "")
        
        # Load mask files supporting both -mask.tif and -Mask.tif
        for ext in ["*.tiff", "*.tiff"]:
            mask_files = glob(os.path.join(mask_list_path, f'*-mask{ext}'))
        if not mask_files:
            raise ValueError(f"No mask TIFF files found in {mask_list_path}")
        
        if results_per_unit is None:
            results_per_unit = {}
        
        # Process each mask file
        for mask_file in tqdm(mask_files, desc="Processing masks"):

            mask_basename = os.path.basename(mask_file)

            # Load mask
            mask_img = tifffile.imread(mask_file)
            if mask_img.ndim > 2:
                mask_img = np.squeeze(mask_img)
            
            if mask_img.ndim == 3 and mask_img.shape[0] == 1:
                mask_img = mask_img[0]
            
            if mask_img.ndim == 3 and mask_img.shape[0] > 1:
                if mask_dict is None:
                    raise Exception(f"Mask image {mask_basename} has multiple channels but no mask_dict provided for indexing.")
                else:
                    ## Extract names from dictionary keys and match to filename
                    for idx, unit in mask_dict.items():
                        mask_slice = mask_img[idx]
                        if mask_slice.ndim != 2: 
                            print(f"Skipping {mask_basename} channel {idx}: expected 2D mask, got shape {mask_slice.shape}")
                            continue

                        unit_name, subgroup = self._parse_mask_basename(mask_basename, prefix, subgroups)
                        if subgroups is not None and subgroup is None:
                            continue
                        
                        # Convert mask to boolean
                        mask_slice = mask_slice > 0
                        if not np.any(mask_slice) or np.all(mask_slice == 0):
                            print(f"Skipping {mask_basename} channel {idx} | {unit_name}")
                            continue
                        
                        instance_metadata = self._instance_masking(
                            mask_slice, 
                            dilations=dilations, 
                            min_instance_size=min_instance_size
                            )
                    

            elif mask_img.ndim == 2:
                # --- EXTRACT NAMES FROM FILENAME FIRST ---
                unit_name = mask_basename
 
                unit_name, subgroup = self._parse_mask_basename(mask_basename, prefix, subgroups)
                if subgroups is not None and subgroup is None:
                    continue
                
                # # Resize mask if needed
                # if mask_img.shape != target_shape:
                #     print(f"  Resizing mask {mask_basename} from {mask_img.shape} to {target_shape} due to downsampling...")
                #     zoom_factors = (target_shape[0] / mask_img.shape[0], target_shape[1] / mask_img.shape[1])
                #     mask_img = ndi.zoom(mask_img, zoom_factors, order=0)
                
                # Label mask for instance segmentation
                instance_metadata = self._instance_masking(
                    mask_img, 
                    dilations=dilations, 
                    min_instance_size=min_instance_size
                    )

            if not instance_metadata:
                continue

            # Quantify based on mode
            if is_prediction_mode:
                for inst in instance_metadata:
                    predictions = data_matrix[inst['padded_slice']][inst['instance_mask']]
                    if len(predictions) == 0:
                        continue
                    
                    unique_classes, counts = np.unique(predictions, return_counts=True)
                    class_counts = {key.strip(): value for key, value in dict(zip(unique_classes, counts)).items()}
                    
                    for cls in classes_to_ignore_set:
                        if cls in class_counts:
                            del class_counts[cls]

                    if group_subclasses:
                        all_classes = self.molecule_names if hasattr(self, 'molecule_names') and self.molecule_names else list(class_counts.keys())
                        mix_classes = [c.strip() for c in all_classes if isinstance(c, str) and "Mix" in c]
                        for mix_class in mix_classes:
                            superclass = mix_class.replace(" Mix", "")
                            subclasses = [c for c in class_counts.keys() if superclass in c and c != mix_class]
                            if subclasses:
                                combined_count = class_counts.get(mix_class, 0)
                                for sub in subclasses:
                                    combined_count += class_counts[sub]
                                    del class_counts[sub]
                                class_counts[mix_class] = combined_count

                    total = sum(class_counts.values())
                    pct = {k: (v / total * 100.0) if total > 0 else 0.0 for k, v in class_counts.items()}
                    
                    data_entry = {
                        'percentages': pct,
                        'image_name': img_name,
                        'Source_ID': Source_ID if Source_ID else img_name.split('-')[0],
                        'instance_label': f"{int(inst['label'])}",
                        'centroid_x': inst['centroid_x'],
                        'centroid_y': inst['centroid_y']
                    }
                    if subgroup:
                        data_entry['Group_ID'] = subgroup
                    results_per_unit.setdefault(unit_name, []).append(data_entry)
            
            else:
                # Ratio mode: Process each slice memory-efficiently
                for slice_idx in range(num_pages):
                    data_slice = tif.pages[slice_idx].asarray()
                    r_type_slice = f"{ratio_type}_{slice_idx + 1}" if num_pages > 1 else ratio_type
                    
                    for inst in instance_metadata:
                        masked_ratio = data_slice[inst['padded_slice']][inst['instance_mask']]
                        non_zero_ratios = masked_ratio[(masked_ratio > 0) & np.isfinite(masked_ratio)]
                        
                        if non_zero_ratios.size > 0:
                            mean_ratio = np.nanmean(non_zero_ratios)
                            ratio_entry = {
                                'ratios': {r_type_slice: mean_ratio},
                                'image_name': img_name,
                                'Source_ID': Source_ID if Source_ID else img_name.split('-')[0],
                                'instance_label': f"{int(inst['label'])}",
                                'centroid_x': inst['centroid_x'],
                                'centroid_y': inst['centroid_y']
                            }
                            if subgroup:
                                ratio_entry['Group_ID'] = subgroup
                            results_per_unit.setdefault(unit_name, []).append(ratio_entry)
        
        if not is_prediction_mode:
            tif.close()
            
        return results_per_unit


    def _parse_mask_basename(self, basename, prefix, subgroups):
        """Return cleaned unit_name and optional subgroup from a mask basename."""
        unit_name = basename
        if prefix and unit_name.startswith(f"{prefix}-"):
            unit_name = unit_name[len(prefix) + 1:]

        for suffix in ['-mask.tif', '-Mask.tif', '-mask.ome.tiff', "-mask.tiff"]:
            if unit_name.endswith(suffix):
                unit_name = unit_name[:-len(suffix)]

        # Convert all subgroups keys to lower case to be case insensitive
        subgroups_lower = {k.lower(): v for k, v in subgroups.items()}

        subgroup = None
        if "_" in unit_name and subgroups_lower:
            
            unit_parts = unit_name.split("_")
            
            # Handle the specific 2-part logic
            if len(unit_parts) == 2:
                match0 = subgroups_lower.get(unit_parts[0].lower())
                match1 = subgroups_lower.get(unit_parts[1].lower())
                
                if match0:
                    subgroup = match0  # Maps to the dictionary value
                    unit_name = unit_parts[1]
                elif match1:
                    subgroup = match1  # Maps to the dictionary value
                    unit_name = unit_parts[0]

            # Handle the fallback multi-part logic if no match was found yet
            if subgroup is None:
                for i, part in enumerate(unit_parts):
                    match = subgroups_lower.get(part.lower())
                    if match:
                        subgroup = match  # Maps to the dictionary value
                        remaining = [p for j, p in enumerate(unit_parts) if i != j]
                        unit_name = "_".join(remaining)
                        break

        return unit_name, subgroup


    def unified_applied_masking(self, data_configs, mask_list_path, target_names=None, mol_groups=None, prefix=None, mask_dict=None,
                                subgroups=None, img_name=None, classes_to_ignore=None, min_instance_size=100, output_flat_csv=None):
        """
        Unified masking quantification for all data types in data_configs in a single pass.
        Avoids repeatedly loading and segmenting masks by generating instances once.
        
        Returns a wide pandas DataFrame with one row per mask instance and columns for all
        quantified values.
        """
        for ext in ["*.tiff", "*.tiff"]:
            mask_files = glob(os.path.join(mask_list_path, f'*-mask{ext}'))
        if not mask_files:
            raise ValueError(f'No mask TIFF files found in {mask_list_path}')

        if mask_list_path is None:
            raise ValueError('mask_list_path is required')

        # Defaults
        if classes_to_ignore is None:
            classes_to_ignore = ['Masked', 'Kidney Background', 'No Match']
        classes_to_ignore_set = {c.strip() for c in classes_to_ignore}

        local_img_name = img_name
        
        # Pre-load/open data sources to avoid redundant file IO
        loaded_sources = []
        try:
            for config in data_configs:
                mode = config['mode']
                files = config['files']
                dilations = config.get('dilations', 2)
                col_prefix = config.get('column_prefix', '')
                
                for f in files:
                    if not os.path.exists(f):
                        print(f"Warning: File not found {f}, skipping from masking analysis.")
                        continue
                        
                    if mode == 'ratios':
                        # Use TiffFile for robust, lazy page access
                        tif_handle = tifffile.TiffFile(f)
                        num_pages = len(tif_handle.pages)
                        ratio_basename = os.path.splitext(os.path.basename(f))[0]
                        if '-ratio' in ratio_basename:
                            ratio_basename = ratio_basename.split('-')[-2]
                        
                        if local_img_name is None:
                            local_img_name = os.path.splitext(os.path.basename(f))[0]
                            if '-ratio' in local_img_name:
                                local_img_name = local_img_name.split('-')[0]
                                
                        loaded_sources.append({
                            'type': 'ratios',
                            'file_path': f,
                            'handle': tif_handle,
                            'num_pages': num_pages,
                            'ratio_basename': ratio_basename,
                            'dilations': dilations,
                            'column_prefix': col_prefix
                        })

                    elif mode == 'collagen':
                        tif_handle  = tifffile.TiffFile(f)
                        num_pages   = len(tif_handle.pages)
                        base_no_ext = os.path.splitext(os.path.basename(f))[0]
                        if base_no_ext.endswith('-collagen'):
                            base_no_ext = base_no_ext[:-len('-collagen')]
                        ratio_basename = base_no_ext.split('-')[-1]

                        if local_img_name is None:
                            local_img_name = os.path.splitext(os.path.basename(f))[0].split('-')[0]

                        loaded_sources.append({
                            'type': 'ratios',
                            'file_path': f,
                            'handle': tif_handle,
                            'num_pages': num_pages,
                            'ratio_basename': ratio_basename,
                            'dilations': dilations,
                            'column_prefix': 'Collagen_'
                        })

                    elif mode == 'sam_classmap':
                        sam_class_map = tifffile.imread(f)
                        if sam_class_map.ndim > 2:
                            sam_class_map = np.squeeze(sam_class_map)

                        if local_img_name is None:
                            local_img_name = os.path.splitext(os.path.basename(f))[0]

                        class_names = []
                        if target_names is not None:
                            class_names = [target.strip() for target in target_names if target.strip() not in classes_to_ignore_set]

                        loaded_sources.append({
                            'type': 'sam_classmap',
                            'file_path': f,
                            'matrix': sam_class_map,
                            'class_names': class_names,
                            'dilations': dilations,
                            'column_prefix': col_prefix
                        })

                    elif mode == 'predictions':
                        data_df = pd.read_csv(f, header=None, index_col=False)
                        rf_class_map = data_df.values
                        if rf_class_map.ndim > 2:
                            rf_class_map = np.squeeze(rf_class_map)

                        if local_img_name is None:
                            local_img_name = os.path.basename(f).replace('_predictions.csv', '')

                        loaded_sources.append({
                            'type': 'predictions',
                            'file_path': f,
                            'matrix': rf_class_map,
                            'dilations': dilations,
                            'column_prefix': col_prefix
                        })

                    elif mode == 'hsi':
                        hsi_data = tifffile.imread(f)
                        if hsi_data.ndim > 2:
                            if hsi_data.shape[0] > hsi_data.shape[2]:
                                hsi_data = np.transpose(hsi_data, (2, 0, 1))
                        else:
                            hsi_data = np.expand_dims(hsi_data, axis=0)
                            
                        if local_img_name is None:
                            local_img_name = os.path.splitext(os.path.basename(f))[0]
                            
                        loaded_sources.append({
                            'type': 'hsi',
                            'file_path': f,
                            'matrix': hsi_data,
                            'dilations': dilations,
                            'column_prefix': col_prefix
                        })

            # Process each mask file
            records = []
            for mask_file in tqdm(mask_files, desc='Processing masks...'):
                mask_basename = os.path.basename(mask_file)
                mask_img = tifffile.imread(mask_file)
                if mask_img.ndim > 2:
                    mask_img = np.squeeze(mask_img)

                # Determine mask slices & units
                slices_to_process = []
                if mask_img.ndim == 3 and mask_img.shape[0] > 1:
                    if mask_dict is None:
                        raise Exception(f"Mask image {mask_basename} has multiple channels but no mask_dict provided for indexing.")
                    for idx, unit in mask_dict.items():
                        mask_slice = mask_img[int(idx)]
                        unit_name = unit
                        subgroup = None
                        if subgroups is not None:
                            for sg in subgroups:
                                if sg.lower() in unit_name.lower():
                                    unit_name = unit_name.replace(sg, '')[:-1]
                                    subgroup = sg
                                    break
                        slices_to_process.append((mask_slice, unit_name, subgroup))
                else:
                    unit_name, subgroup = self._parse_mask_basename(mask_basename, prefix, subgroups)
                    if subgroups is not None and subgroup is None:
                        continue
                    slices_to_process.append((mask_img, unit_name, subgroup))

                # Segment and process each slice
                for mask_slice, unit_name, subgroup in slices_to_process:
                    labeled_mask, num_features = ndi.label(mask_slice > 0)
                    if num_features == 0:
                        continue

                    object_slices = ndi.find_objects(labeled_mask)
                    for label, obj_slice in enumerate(object_slices, start=1):
                        if obj_slice is None:
                            continue

                        # Calculate original centroid coords
                        rows, cols = np.nonzero(labeled_mask[obj_slice] == label)
                        rows += obj_slice[0].start
                        cols += obj_slice[1].start
                        centroid_y = int(np.mean(rows))
                        centroid_x = int(np.mean(cols))

                        # Base metadata record
                        inst_record = {
                            'Image_Name': local_img_name,
                            'Unit': unit_name,
                            'Instance_Label': f"{int(label)}",
                            'Source_ID': local_img_name.split('-')[0] if local_img_name else None,
                            'Group_ID': subgroup,
                            'centroid_x': centroid_x,
                            'centroid_y': centroid_y,
                        }
                        
                        has_any_data = False

                        # Extract data from each loaded source
                        for source in loaded_sources:
                            dilations = source['dilations']
                            y_slice, x_slice = obj_slice
                            y_start = max(0, y_slice.start - dilations)
                            y_end = min(labeled_mask.shape[0], y_slice.stop + dilations)
                            x_start = max(0, x_slice.start - dilations)
                            x_end = min(labeled_mask.shape[1], x_slice.stop + dilations)
                            padded_slice = (slice(y_start, y_end), slice(x_start, x_end))

                            local_labels = labeled_mask[padded_slice]
                            instance_mask = local_labels == label

                            npx = int(np.sum(instance_mask))
                            if npx < min_instance_size:
                                continue

                            if dilations > 0:
                                instance_mask = ndi.binary_dilation(instance_mask, iterations=dilations-1)

                            # Set pixels count if not already set (at first valid source)
                            if 'n_pixels' not in inst_record:
                                inst_record['n_pixels'] = npx

                            if source['type'] == 'ratios':
                                for slice_idx in range(source['num_pages']):
                                    data_slice = source['handle'].pages[slice_idx].asarray()
                                    ratio_basename = source['ratio_basename']
                                    r_type_slice = f"{ratio_basename}_{slice_idx + 1}" if source['num_pages'] > 1 else ratio_basename
                                    col_name = f"{source['column_prefix']}{r_type_slice}"
                                    
                                    masked_vals = data_slice[padded_slice][instance_mask]
                                    vals = masked_vals[(masked_vals > 0) & np.isfinite(masked_vals)]
                                    if vals.size > 0:
                                        mean_ratio = float(np.nanmean(vals))
                                        inst_record[col_name] = mean_ratio
                                        has_any_data = True
                                        
                            elif source['type'] == 'sam_classmap':
                                patch = source['matrix'][padded_slice][instance_mask]
                                labels, counts = np.unique(patch, return_counts=True)
                                name_counts = {}
                                for lbl, cnt in zip(labels, counts):
                                    lbl_int = int(lbl)
                                    if lbl_int <= 0:
                                        continue
                                    class_idx = lbl_int - 1
                                    if class_idx < len(source['class_names']):
                                        name_counts[source['class_names'][class_idx]] = int(cnt)
                                class_total = sum(name_counts.values())
                                if class_total > 0:
                                    for mol, cnt in name_counts.items():
                                        pct = (cnt / class_total) * 100.0
                                        col_name = f"SAM_{mol}"
                                        inst_record[col_name] = float(pct)
                                        has_any_data = True
                                        
                            elif source['type'] == 'predictions':
                                preds = source['matrix'][padded_slice][instance_mask]
                                unique_classes, counts = np.unique(preds, return_counts=True)
                                class_counts = {}
                                for k, v in zip(unique_classes, counts):
                                    k_str = str(k).strip()
                                    if mol_groups is not None and k_str in mol_groups:
                                        mapped_name = mol_groups[k_str]
                                    else:
                                        mapped_name = k_str
                                    class_counts[mapped_name] = class_counts.get(mapped_name, 0) + int(v)
                                    
                                for c in classes_to_ignore_set:
                                    if c in class_counts:
                                        del class_counts[c]
                                        
                                class_total = sum(class_counts.values())
                                if class_total > 0:
                                    for mol, cnt in class_counts.items():
                                        pct = (cnt / class_total) * 100.0
                                        col_name = f"RF_{mol}"
                                        inst_record[col_name] = float(pct)
                                        has_any_data = True
                                        
                            elif source['type'] == 'hsi':
                                hsi_data = source['matrix']
                                num_channels = hsi_data.shape[0]
                                for ch_idx in range(num_channels):
                                    col_name = f"CH{ch_idx + 1}"
                                    masked_vals = hsi_data[ch_idx][padded_slice][instance_mask]
                                    vals = masked_vals[np.isfinite(masked_vals)]
                                    if vals.size > 0:
                                        inst_record[col_name] = float(np.nanmean(vals))
                                        has_any_data = True

                        if has_any_data:
                            records.append(inst_record)

        finally:
            for source in loaded_sources:
                if source['type'] == 'ratios' and 'handle' in source:
                    try:
                        source['handle'].close()
                    except Exception as e:
                        print(f"Failed to close TIFF handle: {e}")

        if not records:
            raise ValueError('No valid instances found across all masks.')

        df_out = pd.DataFrame.from_records(records)
        
        pct_cols = [c for c in df_out.columns if c.startswith(('RF_', 'SAM_'))]
        if pct_cols:
            df_out[pct_cols] = df_out[pct_cols].fillna(0.0)

        if output_flat_csv:
            df_out.to_csv(output_flat_csv, index=False)

        return df_out


    def quantify_unit_class_percentages_nested(self, units_dict, unit_mappings=None):
        """
        Calculate replicate-level percentages for all classes in all units.

        Input shape:
            units_dict: {
                unit_name: [
                    {'counts': {class_name: count, ...}, 'image_name': str, 'Source_ID': str, 'instance_label': str, 'Group_ID': str},  # replicate 0
                    {'counts': {class_name: count, ...}, 'image_name': str, 'Source_ID': str, 'instance_label': str, 'Group_ID': str},  # replicate 1
                    ...
                ],
                ...
            }
            unit_mappings: Optional dict mapping abbreviated unit names to full names

        Returns:
            DataFrame with columns [Unit, Molecule, Percentage, Replicate, Image_Name, Source_ID, Instance_Label, Group_ID]
        """
        # Convert to DataFrame
        records = []
        # count = 0
        for unit_name, pct_list in units_dict.items():
            # Extract base unit name (first part before underscore)
            base_unit = unit_name.split('_')[0] if '_' in unit_name else unit_name
            
            # Apply unit mapping if provided
            if unit_mappings:
                base_unit_lower = base_unit.lower()
                base_unit = unit_mappings.get(base_unit_lower, base_unit)
            
            for rep_idx, pct_entry in enumerate(pct_list):
                pct_dict = pct_entry['percentages']
                image_name = pct_entry['image_name']
                Source_ID = pct_entry['Source_ID']
                instance_label = pct_entry['instance_label']
                Group_ID = pct_entry.get('Group_ID', None)
                centroid_x = pct_entry.get('centroid_x', None)
                centroid_y = pct_entry.get('centroid_y', None)
                
                for molecule, percentage in pct_dict.items():
                    records.append({
                        'Unit': base_unit,
                        'Molecule': molecule,
                        'Percentage': percentage,
                        'Replicate': rep_idx + 1,
                        'Image_Name': image_name,
                        'Source_ID': Source_ID,
                        'Instance_Label': instance_label,
                        'Group_ID': Group_ID,
                        'centroid_x': centroid_x,
                        'centroid_y': centroid_y
                    })
        df = pd.DataFrame.from_records(records)
        return df


    def apply_unified_masking(self, *, mode, data_path=None, mask_list_path, prefix='CODEX', mask_dict=None,
                              subgroups=None, img_name=None, classes_to_ignore=None,
                              group_subclasses=False, dilations=2, min_instance_size=100, output_flat_csv=None):
        """
        Unified masking quantification for three modes: 'predictions', 'ratios', 'sam_classmap'.

        Returns a long-form pandas DataFrame with instance-level rows suitable for pivoting
        into the same `wide_df` shape used by `integratedPipeline.py`.
        """
        if mode not in ('predictions', 'ratios', 'sam_classmap'):
            raise ValueError("mode must be one of 'predictions', 'ratios', 'sam_classmap'")

        if mask_list_path is None:
            raise ValueError('mask_list_path is required')

        if mode in ('predictions',) and data_path is None:
            raise ValueError('data_path (CSV) required for predictions mode')

        if mode == 'ratios' and data_path is None:
            raise ValueError('data_path (ratio TIFF) required for ratios mode')

        # Defaults
        if classes_to_ignore is None:
            classes_to_ignore = ['Masked', 'Kidney Background', 'No Match']
        classes_to_ignore_set = {c.strip() for c in classes_to_ignore}

        # Load data sources as needed
        data_matrix = None
        tif_ratios = None
        sam_class_map = None
        ratio_basename = None

        if mode == 'predictions':
            data_df = pd.read_csv(data_path, header=None, index_col=False)
            data_matrix = data_df.values
            if img_name is None:
                img_name = os.path.basename(data_path).replace('_predictions.csv', '')

        elif mode == 'ratios':
            tif_ratios = tifffile.TiffFile(data_path)
            num_pages = len(tif_ratios.pages)
            ratio_basename = os.path.splitext(os.path.basename(data_path))[0]
            if img_name is None:
                img_name = os.path.splitext(os.path.basename(data_path))[0]
                if '-ratio' in img_name:
                    img_name = img_name.split('-')[0]

        elif mode == 'sam_classmap':  # sam_classmap
            sam_class_map = tifffile.imread(data_path)
            if img_name is None:
                img_name = os.path.splitext(os.path.basename(data_path))[0]

        records = []

        mask_files = glob(os.path.join(mask_list_path, '*-mask.tif'))
        if not mask_files:
            raise ValueError(f'No mask TIFF files found in {mask_list_path}')

        for mask_file in tqdm(mask_files, desc='Processing masks (unified)'):
            mask_basename = os.path.basename(mask_file)
            mask_img = tifffile.imread(mask_file)
            if mask_img.ndim > 2:
                mask_img = np.squeeze(mask_img)

            # Multi-channel mask handling
            if mask_img.ndim == 3 and mask_img.shape[0] > 1:
                if mask_dict is None:
                    raise Exception(f"Mask image {mask_basename} has multiple channels but no mask_dict provided for indexing.")

                for idx, unit in mask_dict.items():
                    try:
                        mask_slice = mask_img[idx]
                    except Exception:
                        print(f"Skipping {mask_basename} channel {idx}: index error")
                        continue

                    unit_name = unit
                    # if prefix and unit_name.startswith(f"{prefix}-"):
                    #     unit_name = unit_name[len(prefix) + 1:]

                    unit_name, subgroup = self._parse_mask_basename(unit_name, prefix, subgroups)

                    mask_slice = mask_slice > 0
                    if not np.any(mask_slice):
                        continue

                    instance_metadata = self._instance_masking(mask_slice, dilations=dilations, min_instance_size=min_instance_size)

                    for inst in instance_metadata:
                        npx = int(np.sum(inst['instance_mask']))
                        base_meta = {
                            'Image_Name': img_name,
                            'Unit': unit_name,
                            'Instance_Label': f"{int(inst['label'])}",
                            'Source_ID': img_name.split('-')[0] if img_name else None,
                            'Group_ID': subgroup,
                            'centroid_x': inst.get('centroid_x', None),
                            'centroid_y': inst.get('centroid_y', None),
                            'n_pixels': npx
                        }

                        if mode == 'predictions':
                            preds = data_matrix[inst['padded_slice']][inst['instance_mask']]
                            if preds.size == 0:
                                continue
                            unique, counts = np.unique(preds, return_counts=True)
                            class_counts = {str(k).strip(): int(v) for k, v in zip(unique, counts)}
                            for c in list(classes_to_ignore_set):
                                if c in class_counts:
                                    del class_counts[c]

                            if group_subclasses:
                                all_classes = self.molecule_names if hasattr(self, 'molecule_names') and self.molecule_names else list(class_counts.keys())
                                mix_classes = [c.strip() for c in all_classes if isinstance(c, str) and 'Mix' in c]
                                for mix_class in mix_classes:
                                    superclass = mix_class.replace(' Mix', '')
                                    subclasses = [c for c in class_counts.keys() if superclass in c and c != mix_class]
                                    if subclasses:
                                        combined = class_counts.get(mix_class, 0)
                                        for sub in subclasses:
                                            combined += class_counts.get(sub, 0)
                                            class_counts.pop(sub, None)
                                        class_counts[mix_class] = combined

                            total = sum(class_counts.values())
                            if total == 0:
                                continue
                            for mol, cnt in class_counts.items():
                                pct = (cnt / total) * 100.0
                                rec = base_meta.copy()
                                rec.update({'Molecule': mol, 'Percentage': float(pct)})
                                records.append(rec)

                        elif mode == 'sam_classmap':
                            if self.molecule_names is None:
                                raise ValueError('HSI_Visualizer.molecule_names not set; required for sam_classmap mode')
                            patch = sam_class_map[inst['padded_slice']][inst['instance_mask']]
                            labels, counts = np.unique(patch, return_counts=True)
                            mol_names_ext = ['No Match'] + list(self.molecule_names)
                            name_counts = {mol_names_ext[int(lbl)]: int(cnt) for lbl, cnt in zip(labels, counts) if int(lbl) < len(mol_names_ext)}
                            # Filter No Match
                            name_counts.pop('No Match', None)
                            total = sum(name_counts.values())
                            if total == 0:
                                continue
                            for mol, cnt in name_counts.items():
                                pct = (cnt / total) * 100.0
                                rec = base_meta.copy()
                                rec.update({'Molecule': mol, 'Percentage': float(pct)})
                                records.append(rec)

                        else:  # ratios
                            # For ratios, handle each page separately
                            for slice_idx in range(num_pages):
                                data_slice = tif_ratios.pages[slice_idx].asarray()
                                r_type_slice = f"{ratio_basename}_{slice_idx + 1}" if num_pages > 1 else ratio_basename
                                masked_vals = data_slice[inst['padded_slice']][inst['instance_mask']]
                                vals = masked_vals[(masked_vals > 0) & np.isfinite(masked_vals)]
                                if vals.size == 0:
                                    continue
                                mean_ratio = float(np.nanmean(vals))
                                rec = base_meta.copy()
                                rec.update({'Ratio_Type': r_type_slice, 'Mean_Ratio': mean_ratio})
                                records.append(rec)

            else:
                # Single-channel mask
                unit_name, subgroup = self._parse_mask_basename(mask_basename, prefix, subgroups)
                if subgroups is not None and subgroup is None:
                    continue

                mask_slice = mask_img > 0
                labeled_mask, num_features = ndi.label(mask_slice)
                if num_features == 0:
                    continue

                object_slices = ndi.find_objects(labeled_mask)
                instance_metadata = []
                for label, obj_slice in enumerate(object_slices, start=1):
                    if obj_slice is None:
                        continue
                    y_slice, x_slice = obj_slice
                    y_start = max(0, y_slice.start - dilations)
                    y_end = min(labeled_mask.shape[0], y_slice.stop + dilations)
                    x_start = max(0, x_slice.start - dilations)
                    x_end = min(labeled_mask.shape[1], x_slice.stop + dilations)
                    padded_slice = (slice(y_start, y_end), slice(x_start, x_end))
                    local_labels = labeled_mask[padded_slice]
                    instance_mask = local_labels == label
                    if int(np.sum(instance_mask)) < min_instance_size:
                        continue
                    if dilations > 0:
                        instance_mask = ndi.binary_dilation(instance_mask, iterations=dilations - 1)
                    rows, cols = np.nonzero(local_labels == label)
                    rows += padded_slice[0].start
                    cols += padded_slice[1].start
                    centroid_y = int(np.mean(rows))
                    centroid_x = int(np.mean(cols))
                    instance_metadata.append({'label': label, 'padded_slice': padded_slice, 'instance_mask': instance_mask, 'centroid_x': centroid_x, 'centroid_y': centroid_y})

                for inst in instance_metadata:
                    npx = int(np.sum(inst['instance_mask']))
                    base_meta = {
                        'Image_Name': img_name,
                        'Unit': unit_name,
                        'Instance_Label': f"{int(inst['label'])}",
                        'Source_ID': (img_name.split('-')[0] if img_name else None),
                        'Group_ID': subgroup,
                        'centroid_x': inst.get('centroid_x', None),
                        'centroid_y': inst.get('centroid_y', None),
                        'n_pixels': npx
                    }

                    if mode == 'predictions':
                        preds = data_matrix[inst['padded_slice']][inst['instance_mask']]
                        if preds.size == 0:
                            continue
                        unique, counts = np.unique(preds, return_counts=True)
                        class_counts = {str(k).strip(): int(v) for k, v in zip(unique, counts)}
                        for c in list(classes_to_ignore_set):
                            if c in class_counts:
                                del class_counts[c]

                        if group_subclasses:
                            all_classes = self.molecule_names if hasattr(self, 'molecule_names') and self.molecule_names else list(class_counts.keys())
                            mix_classes = [c.strip() for c in all_classes if isinstance(c, str) and 'Mix' in c]
                            for mix_class in mix_classes:
                                superclass = mix_class.replace(' Mix', '')
                                subclasses = [c for c in class_counts.keys() if superclass in c and c != mix_class]
                                if subclasses:
                                    combined = class_counts.get(mix_class, 0)
                                    for sub in subclasses:
                                        combined += class_counts.get(sub, 0)
                                        class_counts.pop(sub, None)
                                    class_counts[mix_class] = combined

                        total = sum(class_counts.values())
                        if total == 0:
                            continue
                        for mol, cnt in class_counts.items():
                            pct = (cnt / total) * 100.0
                            rec = base_meta.copy()
                            rec.update({'Molecule': mol, 'Percentage': float(pct)})
                            records.append(rec)

                    elif mode == 'sam_classmap':
                        if self.molecule_names is None:
                            raise ValueError('HSI_Visualizer.molecule_names not set; required for sam_classmap mode')
                        patch = sam_class_map[inst['padded_slice']][inst['instance_mask']]
                        labels, counts = np.unique(patch, return_counts=True)
                        mol_names_ext = ['No Match'] + list(self.molecule_names)
                        name_counts = {mol_names_ext[int(lbl)]: int(cnt) for lbl, cnt in zip(labels, counts) if int(lbl) < len(mol_names_ext)}
                        name_counts.pop('No Match', None)
                        total = sum(name_counts.values())
                        if total == 0:
                            continue
                        for mol, cnt in name_counts.items():
                            pct = (cnt / total) * 100.0
                            rec = base_meta.copy()
                            rec.update({'Molecule': mol, 'Percentage': float(pct)})
                            records.append(rec)

                    else:  # ratios
                        for slice_idx in range(num_pages):
                            data_slice = tif_ratios.pages[slice_idx].asarray()
                            r_type_slice = f"{ratio_basename}_{slice_idx + 1}" if num_pages > 1 else ratio_basename
                            masked_vals = data_slice[inst['padded_slice']][inst['instance_mask']]
                            vals = masked_vals[(masked_vals > 0) & np.isfinite(masked_vals)]
                            if vals.size == 0:
                                continue
                            mean_ratio = float(np.nanmean(vals))
                            rec = base_meta.copy()
                            rec.update({'Ratio_Type': r_type_slice, 'Mean_Ratio': mean_ratio})
                            records.append(rec)

        # Close tif handle if opened
        if tif_ratios is not None:
            tif_ratios.close()

        if not records:
            raise ValueError('No valid instances found across all masks.')

        df_out = pd.DataFrame.from_records(records)
        # Ensure consistent column ordering favorable for pivoting
        index_cols = ['Image_Name', 'Unit', 'Instance_Label', 'Source_ID']
        if 'Group_ID' in df_out.columns and df_out['Group_ID'].notna().any():
            index_cols.append('Group_ID')

        # Optionally write flat output as parquet when requested.
        if output_flat_csv:
            output_path = str(output_flat_csv)
            if output_path.lower().endswith('.parquet'):
                try:
                    df_out.to_parquet(output_path, index=False)
                except ImportError:
                    fallback_path = output_path[:-8] + '.csv'
                    df_out.to_csv(fallback_path, index=False)
            else:
                df_out.to_csv(output_path, index=False)

        return df_out


    def quantify_unit_ratio_means_nested(self, units_dict, unit_mappings=None):
        """
        Calculate replicate-level mean ratios for all ratio types in all units.

        Input shape:
            units_dict: {
                unit_name: [
                    {'ratios': {ratio_type: mean_value, ...}, 'image_name': str, 'Source_ID': str, 'instance_label': str, 'Group_ID': str},  # replicate 0
                    {'ratios': {ratio_type: mean_value, ...}, 'image_name': str, 'Source_ID': str, 'instance_label': str, 'Group_ID': str},  # replicate 1
                    ...
                ],
                ...
            }
            unit_mappings: Optional dict mapping abbreviated unit names to full names

        Returns:
            DataFrame with columns [Unit, Ratio_Type, Mean_Ratio, Replicate, Image_Name, Source_ID, Instance_Label, Group_ID]
        """
        records = []
        for unit_name, ratio_list in units_dict.items():
            # Extract base unit name (first part before underscore)
            base_unit = unit_name.split('_')[0] if '_' in unit_name else unit_name  

            if isinstance(ratio_list, dict):
                # Old format: single dict of ratios
                ratio_list = [{'ratios': ratio_list, 'image_name': 'Unknown', 'Source_ID': 'Unknown', 'instance_label': '0', 'Group_ID': None}]
            
            for rep_idx, ratio_entry in enumerate(ratio_list):
                # Check if new format (dict with 'ratios', 'image_name', 'Source_ID', 'instance_label', 'Group_ID')
                if isinstance(ratio_entry, dict) and 'ratios' in ratio_entry:
                    ratio_dict = ratio_entry['ratios']
                    image_name = ratio_entry.get('image_name', 'Unknown')
                    Source_ID = ratio_entry.get('Source_ID', 'Unknown')
                    instance_label = ratio_entry.get('instance_label', '0')
                    Group_ID = ratio_entry.get('Group_ID', None)
                    centroid_x = ratio_entry.get('centroid_x', None)
                    centroid_y = ratio_entry.get('centroid_y', None)
                else:
                    # Old format: just a dict of ratio values
                    ratio_dict = ratio_entry
                    image_name = 'Unknown'
                    Source_ID = 'Unknown'
                    instance_label = '0'
                    Group_ID = None
                    centroid_x = None
                    centroid_y = None
                

                
                # Apply unit mapping if provided
                if unit_mappings:
                    base_unit_lower = base_unit.lower()
                    base_unit = unit_mappings.get(base_unit_lower, base_unit)
                
                for ratio_type, mean_value in ratio_dict.items():
                    records.append({
                        'Unit': base_unit,
                        'Ratio_Type': ratio_type,
                        'Mean_Ratio': mean_value,
                        'Replicate': rep_idx + 1,
                        'Image_Name': image_name,
                        'Source_ID': Source_ID,
                        'Instance_Label': instance_label,
                        'Group_ID': Group_ID,
                        'centroid_x': centroid_x,
                        'centroid_y': centroid_y
                    })
        
        df = pd.DataFrame.from_records(records)
        return df

    def aggregate_molecule_groups(self, df, group_map, molecule_col='Molecule', value_col='Percentage', keep_original=True):
        """
        Aggregate fine-grained molecule classes into supergroups.
        
        Args:
            df: Long-form percentages DataFrame
            group_map: Dict of {SupergroupName: [Molecule1, Molecule2, ...]}
            molecule_col: Column containing molecule names
            value_col: Column containing values to sum
            keep_original: If True, original molecules are kept alongside the new supergroups (default True)
            
        Returns:
            DataFrame with aggregated "supergroups" added
        """
        if df is None or df.empty:
            return df
            
        new_df = df.copy()
        
        # Identify non-value and non-molecule columns to group by
        metadata_cols = [c for c in df.dropna(axis=1, how='all').columns if c not in [molecule_col, value_col]]
        
        # Track all molecules that are being aggregated
        aggregated_molecules = [m for sublist in group_map.values() for m in sublist]
        
        aggregated_results = []
        for group_name, molecules in group_map.items():
            # Filter for molecules in this group
            subset = df[df[molecule_col].isin(molecules)]
            if subset.empty:
                continue
                
            # Aggregate by metadata columns
            grouped = subset.groupby(metadata_cols, observed=True)[value_col].sum().reset_index()
            grouped[molecule_col] = group_name
            aggregated_results.append(grouped)
            
        if not aggregated_results:
            return new_df
            
        # Combine aggregated results
        agg_df = pd.concat(aggregated_results, ignore_index=True)
        
        if not keep_original:
            # Drop molecules that were aggregated
            new_df = new_df[~new_df[molecule_col].isin(aggregated_molecules)]
            
        # Append supergroups
        final_df = pd.concat([new_df, agg_df], ignore_index=True)
        return final_df

    def plot_composition_bar(self, df, grouping_cols, molecule_col='Molecule', value_col='Percentage', 
                             output_dir=None, title_prefix='', figsize=(15, 8), 
                             show_plots=False, palette=None, display_name_map=None, unit_order=None,
                             normalize=True):
        """
        Generate stacked bar charts showing the composition of molecules per group.
        
        Args:
            df: DataFrame containing the data.
            grouping_cols: List of columns to group by (e.g., ['Unit'] or ['Source_ID', 'Unit']).
            molecule_col: Column containing molecule names.
            value_col: Column containing values to sum/average.
            output_dir: Directory to save the plots.
            title_prefix: Prefix for the plot title.
            figsize: Tuple for figure size.
            show_plots: Whether to display plots interactively.
            palette: Optional colormap or list of colors.
            display_name_map: Optional mapping for molecule names in the legend.
            unit_order: Optional list to order the groups on the X-axis.
            normalize: If True, normalizes each bar to 100%.
        """
        import matplotlib.pyplot as plt
        import os

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # Group and average across replicates
        agg_df = df.groupby(grouping_cols + [molecule_col], observed=True)[value_col].mean().reset_index()
        
        # Pivot for stacked bar chart
        pivot_df = agg_df.pivot_table(index=grouping_cols, columns=molecule_col, values=value_col).fillna(0)
        
        # Normalize to 100% if requested
        if normalize:
            row_sums = pivot_df.sum(axis=1)
            # Avoid division by zero
            pivot_df = pivot_df.div(row_sums.where(row_sums > 0, 1), axis=0) * 100
        
        # Apply unit order if provided
        if unit_order:
            # Handle multi-index if grouping_cols has more than one column
            if len(grouping_cols) == 1:
                pivot_df = pivot_df.reindex(unit_order)
            else:
                # If multi-level, we sort by the last level (assumed to be Unit)
                unit_col = grouping_cols[-1]
                pivot_df = pivot_df.reset_index()
                # Only reorder if all categories are present or we allow NaNs
                existing_units = [u for u in unit_order if u in pivot_df[unit_col].values]
                pivot_df[unit_col] = pd.Categorical(pivot_df[unit_col], categories=unit_order, ordered=True)
                pivot_df = pivot_df.sort_values(grouping_cols).set_index(grouping_cols)
        
        # Map molecule names for legend if display_name_map is provided
        if display_name_map:
            pivot_df = pivot_df.rename(columns=display_name_map)
        
        # Plotting
        fig, ax = plt.subplots(figsize=figsize)
        pivot_df.plot(kind='bar', stacked=True, ax=ax, colormap=palette or 'tab20')
        
        # Title and Labels
        group_label = ", ".join(grouping_cols)
        norm_suffix = " (%)" if normalize else ""
        plt.title(f"{title_prefix} Molecule Composition per {group_label}", fontweight='bold', fontsize=16)
        plt.ylabel(f"Composition{norm_suffix}", fontweight='bold', fontsize=14)
        plt.xlabel(group_label, fontweight='bold', fontsize=14)
        
        # Legend styling
        plt.legend(title="Molecule", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        
        plt.tight_layout()
        
        if output_dir:
            # Create a safe filename
            safe_prefix = title_prefix.replace(' ', '_').replace(':', '').replace('/', '_').lower()
            safe_groups = "_".join(grouping_cols).lower()
            filename = f"{safe_prefix}_{safe_groups}_composition.png"
            plt.savefig(os.path.join(output_dir, filename), dpi=600, bbox_inches='tight')
            
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
            
        return pivot_df

    def plot_cluster_comparison_across_images(self, df, unit_name, molecule_list=None, 
                                             output_dir=None, value_col='Percentage',
                                             show_plots=False, display_name_map=None,
                                             unit_color_map=None, palette='viridis',
                                             display_units=None,
                                             remove_outliers=True):
        """
        Compare molecule percentages for a specific cluster across different images.
        Shows mean bars with overlaid individual data points.
        
        Args:
            df: DataFrame containing the data.
            unit_name: Name of the cluster (Unit) to analyze.
            molecule_list: List of molecules to compare. If "All" or None, uses all in the unit.
            output_dir: Directory to save the plots (will create a subfolder for the unit).
            value_col: Column containing values to plot.
            show_plots: Whether to display plots interactively.
            display_name_map: Optional mapping for molecule names in titles.
            unit_color_map: Not used here as we color by Source_ID usually.
            palette: Seaborn palette for images.
            remove_outliers: If True, applies 1.5 * IQR filtration per image.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os
        
        # Filter for the specific unit
        unit_df = df[df['Unit'] == unit_name].copy()
        if unit_df.empty:
            print(f"No data found for unit: {unit_name}")
            return
            
        if molecule_list == "All" or molecule_list is None:
            molecule_list = sorted(unit_df['Molecule'].unique())
        elif isinstance(molecule_list, str):
            molecule_list = [molecule_list]
            
        if output_dir:
            unit_safe = unit_name.replace(' ', '_').replace('/', '_')
            unit_dir = os.path.join(output_dir, 'cluster_comparison_across_images', unit_safe)
            os.makedirs(unit_dir, exist_ok=True)
            
        # Build global list of all unique image names across the entire input df
        # so that every plot has the same X-axis width regardless of which images have data
        all_image_names = sorted([str(n) for n in df['Image_Name'].dropna().unique()])
        
        for molecule in molecule_list:
            mol_df_stats = unit_df[unit_df['Molecule'] == molecule].copy()
            if mol_df_stats.empty:
                continue
            
            # 1. Prepare raw data for stats
            mol_df_stats = mol_df_stats.dropna(subset=[value_col])
            img_stats_raw = mol_df_stats.groupby('Image_Name')[value_col].agg(['mean', 'count'])
            valid_imgs_raw = img_stats_raw[(img_stats_raw['mean'] > 0) & (img_stats_raw['count'] > 1)].index.tolist()
            mol_df_stats = mol_df_stats[mol_df_stats['Image_Name'].isin(valid_imgs_raw)].copy()

            if mol_df_stats.empty:
                continue

            # 2. Prepare filtered data for plotting
            if remove_outliers:
                mol_df_plot = self._apply_outlier_filtration(mol_df_stats, value_col, ['Image_Name'])
            else:
                mol_df_plot = mol_df_stats

            if mol_df_plot.empty:
                continue

            # --- Cross-image significance testing on RAW data ---
            from scipy.stats import ttest_ind
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
            
            image_names_with_data = sorted(mol_df_stats['Image_Name'].unique())
            sig_symbols = {0.001: '***', 0.01: '**', 0.05: '*', 1.0: 'ns'}
            unit_significance = []
            
            if len(image_names_with_data) == 2:
                data1 = mol_df_stats[mol_df_stats['Image_Name'] == image_names_with_data[0]][value_col].values
                data2 = mol_df_stats[mol_df_stats['Image_Name'] == image_names_with_data[1]][value_col].values
                if len(data1) > 1 and len(data2) > 1:
                    _, p_val = ttest_ind(data1, data2, equal_var=True)
                    sig = 'ns'
                    for thresh, symb in sorted(sig_symbols.items()):
                        if p_val <= thresh: sig = symb; break
                    if sig != 'ns':
                        try:
                            idx1 = all_image_names.index(image_names_with_data[0])
                            idx2 = all_image_names.index(image_names_with_data[1])
                            unit_significance.append((idx1, idx2, sig))
                            print(f"    Significant: {molecule} — {image_names_with_data[0]} vs {image_names_with_data[1]}: p={p_val:.4f} ({sig})")
                        except ValueError: pass
            
            elif len(image_names_with_data) > 2:
                try:
                    res = pairwise_tukeyhsd(endog=mol_df_stats[value_col], groups=mol_df_stats['Image_Name'], alpha=0.05)
                    for row in res.summary().data[1:]:
                        if row[6]:  # reject is True
                            p_val_raw = row[3]
                            p_adj = 0.001 if isinstance(p_val_raw, str) and '<' in p_val_raw else float(p_val_raw)
                            sig = 'ns'
                            for thresh, symb in sorted(sig_symbols.items()):
                                if p_adj <= thresh: sig = symb; break
                            try:
                                idx1 = all_image_names.index(str(row[0]))
                                idx2 = all_image_names.index(str(row[1]))
                                unit_significance.append((idx1, idx2, sig))
                                print(f"    Significant: {molecule} — {row[0]} vs {row[1]}: p={p_adj:.4f} ({sig})")
                            except ValueError: pass
                except Exception: pass

            fig, ax = plt.subplots(figsize=(max(6, len(all_image_names) * 0.5), 6))
            
            # Plot bars and scatter using filtered data
            sns.barplot(data=mol_df_plot, x='Image_Name', y=value_col, ax=ax,
                        order=all_image_names, alpha=0.6, capsize=.1, errorbar='se', palette=palette)
            sns.stripplot(data=mol_df_plot, x='Image_Name', y=value_col, ax=ax,
                          order=all_image_names, color='black', alpha=0.5, jitter=True, size=4)

            # --- Draw significance markers ---
            if unit_significance:
                y_min, y_max = ax.get_ylim()
                line_height = 0.08
                unit_significance.sort(key=lambda x: (abs(x[0]-x[1]), min(x[0], x[1])))
                drawn_markers = []
                for x1, x2, sig in unit_significance:
                    if x1 > x2: x1, x2 = x2, x1
                    level = 0
                    while True:
                        collision = False
                        for (mx1, mx2), mlevel in drawn_markers:
                            if mlevel == level and not (x2 + 0.2 < mx1 or x1 - 0.2 > mx2):
                                collision = True; break
                        if not collision: break
                        level += 1
                    y_pos = y_max * (1.10 + level * line_height)
                    ax.plot([x1, x2], [y_pos, y_pos], 'k-', linewidth=1.0)
                    ax.text((x1 + x2) / 2, y_pos, sig, ha='center', va='bottom', fontsize=10, fontweight='bold')
                    drawn_markers.append(((x1, x2), level))
                max_level = max(m[1] for m in drawn_markers)
                ax.set_ylim(top=y_max * (1.20 + max_level * line_height))

            display_unit = display_units.get(unit_name, unit_name) if display_units else unit_name
            display_mol = display_name_map.get(molecule, molecule) if display_name_map else molecule
            plt.title(f"{display_unit}: {display_mol}", fontweight='bold', fontsize=14)
            plt.ylabel(f"Relative Probability Pct. (%)", fontweight='bold')
            plt.xlabel("")
            plt.xticks([])
            plt.grid(axis='y', linestyle='--', alpha=0.3)
            plt.tight_layout()
            
            if output_dir:
                mol_safe = str(molecule).replace(' ', '_').replace(':', '').replace('/', '_')
                plt.savefig(os.path.join(unit_dir, f"{mol_safe}_across_images.png"), dpi=600, bbox_inches='tight')
                
            if show_plots:
                plt.show()
            else:
                plt.close(fig)
        
        print(f"  Generated cross-image comparison plots for {unit_name} in {unit_dir if output_dir else 'memory'}")

    def plot_spatial_gradient_heatmap(self, df, molecule, value_col='Percentage',
                                      image_order=None, unit_order=None,
                                      output_dir=None, display_name_map=None,
                                      unit_display_map=None, normalize_axis='rows',
                                      cmap='coolwarm', figsize=None,
                                      min_images=2, show_plots=False):
        """
        Generate a heatmap showing how a molecule's mean value shifts across
        spatially-ordered images for each cluster/unit.

        Rows = clusters/units. Columns = images in their provided spatial order
        (e.g. inner medulla → inner cortex). Column labels are hidden so the
        axis reads as a continuous spatial gradient.

        Args:
            df: DataFrame containing the data (must have 'Unit', 'Molecule',
                'Image_Name', and value_col columns).
            molecule: The molecule/ratio to visualize.
            value_col: Column containing values (default 'Percentage').
            image_order: Ordered list of Image_Name values that defines the
                spatial axis (left → right). If None, uses sorted unique values.
            unit_order: Ordered list of Unit values for the Y-axis. If None,
                uses sorted unique values.
            output_dir: Directory to save the plot.
            display_name_map: Optional dict mapping molecule names to display names.
            unit_display_map: Optional dict mapping unit names to display names.
            normalize_axis: How to z-score the data before plotting.
                'rows'  — normalise each unit (row) across its images; highlights
                          the gradient shape within each cluster (default).
                'cols'  — normalise each image (column) across all units; highlights
                          which clusters are relatively high/low at each position.
                None    — no normalisation; raw values are plotted.
            cmap: Matplotlib colormap name (default 'coolwarm').
            figsize: Tuple (width, height). Auto-sized if None.
            min_images: Minimum number of images with data for a unit to be
                included (default 2).
            show_plots: Whether to display the plot interactively.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os

        # --- Filter to the requested molecule ---
        mol_df = df[df['Molecule'] == molecule].copy()
        if mol_df.empty:
            print(f"No data found for molecule: {molecule}")
            return

        # --- Build image order ---
        if image_order is None:
            image_order = sorted([str(n) for n in df['Image_Name'].dropna().unique()])
        image_order = [str(n) for n in image_order]

        # --- Compute per-unit, per-image means ---
        mol_df['Image_Name'] = mol_df['Image_Name'].astype(str)
        means = (mol_df.groupby(['Unit', 'Image_Name'])[value_col]
                       .mean()
                       .reset_index())

        # Pivot: rows = Unit, cols = Image_Name in provided order
        pivot = means.pivot(index='Unit', columns='Image_Name', values=value_col)

        # Keep only image columns that are in our ordered list
        available_images = [img for img in image_order if img in pivot.columns]
        pivot = pivot[available_images]

        # Drop units that have fewer than min_images non-null values
        pivot = pivot[pivot.notna().sum(axis=1) >= min_images]

        if pivot.empty:
            print(f"No units with >= {min_images} images for molecule: {molecule}")
            return

        # Apply unit order: use the display-name-order from unit_display_map.values() if available,
        # otherwise fall back to explicit unit_order list or alphabetical sort.
        if unit_display_map is not None:
            # Build ordering from the VALUES of unit_display_map (preserves insertion order)
            display_to_raw = {v: k for k, v in unit_display_map.items()}
            ordered_display_names = list(dict.fromkeys(unit_display_map.values()))  # unique, ordered
            ordered_units = [display_to_raw[d] for d in ordered_display_names if display_to_raw.get(d) in pivot.index]
            pivot = pivot.reindex(ordered_units)
        elif unit_order is not None:
            ordered_units = [u for u in unit_order if u in pivot.index]
            pivot = pivot.reindex(ordered_units)
        else:
            pivot = pivot.sort_index()

        # Map row labels to display names
        if unit_display_map:
            pivot.index = [unit_display_map.get(u, u) for u in pivot.index]

        # --- Z-score normalisation ---
        if normalize_axis == 'rows':
            # Normalise each unit (row) across its images
            row_means = pivot.mean(axis=1)
            row_stds  = pivot.std(axis=1).replace(0, 1)
            pivot = pivot.sub(row_means, axis=0).div(row_stds, axis=0)
        elif normalize_axis == 'cols':
            # Normalise each image (column) across all units
            col_means = pivot.mean(axis=0)
            col_stds  = pivot.std(axis=0).replace(0, 1)
            pivot = pivot.sub(col_means, axis=1).div(col_stds, axis=1)

        is_normalized = normalize_axis in ('rows', 'cols')

        # --- Fill NaN with 0 for smooth rendering ---
        data_arr = pivot.fillna(0).values

        # Compute colour limits symmetrically around 0
        vabs = np.nanmax(np.abs(data_arr)) if is_normalized else None
        vmin_plot = -vabs if vabs else None
        vmax_plot =  vabs if vabs else None

        # --- Plot using sns.heatmap with no cell lines ---
        n_rows, n_cols = pivot.shape
        if figsize is None:
            figsize = (max(8, n_cols * 0.35), max(4, n_rows * 0.45))

        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            pivot,
            ax=ax,
            cmap=cmap,
            center=0 if is_normalized else None,
            vmin=vmin_plot,
            vmax=vmax_plot,
            linewidths=0,       # no cell dividers at all
            cbar=False,         # colorbar saved separately
            xticklabels=False,
        )

        display_mol = display_name_map.get(molecule, molecule) if display_name_map else molecule
        norm_labels = {'rows': ' (row z-scored)', 'cols': ' (col z-scored)'}
        norm_note = norm_labels.get(normalize_axis, '')
        ax.set_title(f"{display_mol} — Spatial Gradient Across Images{norm_note}",
                     fontweight='bold', fontsize=13, pad=12)
        ax.set_xlabel("← Inner Medulla    |    Inner Cortex →", fontsize=10)
        ax.set_ylabel("")
        plt.tight_layout()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            mol_safe = str(molecule).replace(' ', '_').replace(':', '').replace('/', '_')

            # Save the main heatmap (no colorbar embedded)
            save_path = os.path.join(output_dir, f"spatial_gradient_{mol_safe}.png")
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
            print(f"  Saved spatial gradient heatmap: {save_path}")

            # Save the colorbar as a separate image
            cbar_fig, cbar_ax = plt.subplots(figsize=(1.2, max(3, n_rows * 0.3)))
            cbar_label_text = {'rows': 'Z-score (per unit)', 'cols': 'Z-score (per image)'}
            cbar_label = cbar_label_text.get(normalize_axis, value_col)
            norm = mcolors.Normalize(vmin=vmin_plot, vmax=vmax_plot)
            cbar = cbar_fig.colorbar(
                cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=cbar_ax,
                orientation='vertical',
            )
            cbar.set_label(cbar_label, fontsize=10)
            cbar_fig.tight_layout()
            cbar_path = os.path.join(output_dir, f"spatial_gradient_{mol_safe}_colorbar.png")
            cbar_fig.savefig(cbar_path, dpi=600, bbox_inches='tight')
            plt.close(cbar_fig)
            print(f"  Saved colorbar: {cbar_path}")

        if show_plots:
            plt.show()
        else:
            plt.close(fig)

    def plot_spatial_gradient_heatmap_multi(self, df, query_list, query_col='Query',value_col='Value',
                                            image_order=None, unit_order=None,
                                            output_dir=None, display_name_map=None,
                                            unit_display_map=None, normalize_axis='cols',
                                            cmap='coolwarm', figsize=None,
                                            min_images=2, show_plots=False):
        """
        Generate a spatial gradient heatmap with a transposed, multi-query layout:

          - Rows    = images ordered spatially (medulla top -> cortex bottom).
          - Columns = cluster/unit groups, each subdivided into len(query_list)
                      sub-columns -- one per molecule or ratio type in query_list.

        This lets you compare how multiple molecules behave across the same
        spatial gradient side-by-side for every cluster in one figure.

        Args:
            df          : DataFrame with columns 'Unit', 'Molecule', 'Image_Name',
                          and value_col.
            query_list  : Single molecule name (str) OR list of molecule/ratio names.
                          Each entry becomes one sub-column within every cluster group.
            value_col   : Column containing values (default 'Percentage').
            image_order : Ordered list of Image_Name values, medulla-first -> cortex-last.
                          Rows are plotted top-to-bottom in this order.
                          If None, sorted unique values are used.
            unit_order  : Ordered list of Unit names for the column groups (left -> right).
                          If None, uses unit_display_map insertion order or alphabetical.
            output_dir  : Directory to save outputs.
            display_name_map : Dict mapping molecule names -> display names (sub-column labels).
            unit_display_map : Dict mapping unit names -> display names (group header labels).
            normalize_axis :
                'cols'  -- z-score each (unit, molecule) sub-column across images (default).
                           Highlights the gradient shape within each cluster-molecule pair.
                'rows'  -- z-score each image row across all sub-columns.
                           Highlights which cluster-molecule pairs are high/low per image.
                None    -- no normalisation; raw values plotted.
            cmap        : Matplotlib colormap (default 'coolwarm').
            figsize     : Tuple (w, h). Auto-sized if None.
            min_images  : Minimum non-null image count for a unit to be included (default 2).
            show_plots  : Whether to show the plot interactively.
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        from matplotlib.transforms import blended_transform_factory
        import os

        # --- Normalise query_list to a list ---
        if isinstance(query_list, str):
            query_list = [query_list]

        # --- Filter df to requested molecules ---
        mol_df = df[df[query_col].isin(query_list)].copy()
        if mol_df.empty:
            print(f"No data found for query: {query_list}")
            return

        # --- Build image order (rows: medulla=top -> cortex=bottom) ---
        if image_order is None:
            image_order = sorted([str(n) for n in df['Image_Name'].dropna().unique()])
        image_order = [str(n) for n in image_order]

        # --- Compute per-unit, per-molecule, per-image means ---
        mol_df['Image_Name'] = mol_df['Image_Name'].astype(str)
        means = (
            mol_df
            .groupby(['Unit', query_col, 'Image_Name'])[value_col]
            .mean()
            .reset_index()
        )

        # --- Determine valid units (>= min_images across any of the query molecules) ---
        unit_img_counts = means.groupby('Unit')['Image_Name'].nunique()
        valid_units = unit_img_counts[unit_img_counts >= min_images].index.tolist()
        if not valid_units:
            print(f"No units with >= {min_images} images for query: {query_list}")
            return

        # --- Resolve unit ordering ---
        if unit_display_map is not None:
            display_to_raw = {v: k for k, v in unit_display_map.items()}
            ordered_display = list(dict.fromkeys(unit_display_map.values()))
            ordered_units = [
                display_to_raw[d] for d in ordered_display
                if display_to_raw.get(d) in valid_units
            ]
        elif unit_order is not None:
            ordered_units = [u for u in unit_order if u in valid_units]
        else:
            ordered_units = sorted(valid_units)

        if not ordered_units:
            print("No valid units after applying ordering.")
            return

        n_units  = len(ordered_units)
        n_mols   = len(query_list)
        n_images = len(image_order)
        n_cols   = n_units * n_mols   # total sub-columns

        # --- Build full data matrix (n_images x n_cols) ---
        # Column layout: [unit0_mol0, unit0_mol1, ..., unit1_mol0, unit1_mol1, ...]
        data_matrix = np.full((n_images, n_cols), np.nan)
        for ui, unit in enumerate(ordered_units):
            for mi, mol in enumerate(query_list):
                col_idx = ui * n_mols + mi
                subset = means[(means['Unit'] == unit) & (means[query_col] == mol)]
                for row_idx, img in enumerate(image_order):
                    row_data = subset[subset['Image_Name'] == img][value_col]
                    if not row_data.empty:
                        data_matrix[row_idx, col_idx] = row_data.iloc[0]

        # --- Apply normalisation ---
        is_normalized = normalize_axis in ('cols', 'rows')
        if normalize_axis == 'cols':
            # Z-score each sub-column (unit-molecule pair) independently across images
            col_means = np.nanmean(data_matrix, axis=0)
            col_stds  = np.nanstd(data_matrix,  axis=0)
            col_stds[col_stds == 0] = 1
            data_matrix = (data_matrix - col_means[np.newaxis, :]) / col_stds[np.newaxis, :]
        elif normalize_axis == 'rows':
            # Z-score each image row across all sub-columns
            row_means = np.nanmean(data_matrix, axis=1)
            row_stds  = np.nanstd(data_matrix,  axis=1)
            row_stds[row_stds == 0] = 1
            data_matrix = (data_matrix - row_means[:, np.newaxis]) / row_stds[:, np.newaxis]

        # Colour limits (symmetric around 0 when normalised)
        vabs = np.nanmax(np.abs(data_matrix)) if is_normalized else None
        vmin_plot = -vabs if vabs is not None else float(np.nanmin(data_matrix))
        vmax_plot =  vabs if vabs is not None else float(np.nanmax(data_matrix))

        # Replace NaN with 0 for rendering
        plot_matrix = np.where(np.isnan(data_matrix), 0.0, data_matrix)

        # --- Figure sizing ---
        if figsize is None:
            fig_w = max(6, n_cols * 0.55 + 2.5)
            fig_h = max(4, n_images * 0.38 + 1.8)
            figsize = (fig_w, fig_h)

        fig, ax = plt.subplots(figsize=figsize)

        norm_obj = mcolors.Normalize(vmin=vmin_plot, vmax=vmax_plot)
        ax.imshow(plot_matrix, aspect='auto', cmap=cmap,
                  norm=norm_obj, interpolation='nearest')

        ax.set_xlim(-0.5, n_cols - 0.5)
        ax.set_ylim(n_images - 0.5, -0.5)  # top=0 -> medulla at top

        # --- Column separator lines ---
        for ui in range(n_units):
            # Thin lines between sub-columns within the same unit group
            for mi in range(1, n_mols):
                x = ui * n_mols + mi - 0.5
                ax.axvline(x, color='white', linewidth=0.5, alpha=0.5, zorder=3)
            # Thick lines between unit groups
            if ui > 0:
                x = ui * n_mols - 0.5
                ax.axvline(x, color='white', linewidth=2.0, zorder=4)

        # --- X-axis ticks: molecule sub-column labels ---
        xtick_positions = list(range(n_cols))
        xtick_labels = []
        for _ in range(n_units):
            for mol in query_list:
                disp = display_name_map.get(mol, mol) if display_name_map else mol
                xtick_labels.append(disp)
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels, rotation=45, ha='right', fontsize=8)
        ax.tick_params(axis='x', which='both', length=0)
        ax.xaxis.set_tick_params(pad=2)

        # --- Y-axis: hide individual image labels; add directional label ---
        ax.set_yticks([])
        ax.set_ylabel("Medulla \u2191              \u2193 Cortex", fontsize=9, labelpad=6)

        # --- Unit group bracketed headers above the axes ---
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        bracket_y = 1.045
        tick_y0   = 1.01

        for ui, unit in enumerate(ordered_units):
            center_col = ui * n_mols + (n_mols - 1) / 2.0
            left_col   = ui * n_mols - 0.4
            right_col  = ui * n_mols + n_mols - 1 + 0.4
            unit_disp  = unit_display_map.get(unit, unit) if unit_display_map else unit

            # Horizontal bracket line spanning this unit's columns
            ax.plot([left_col, right_col], [bracket_y, bracket_y],
                    transform=trans, color='black', lw=1.0, clip_on=False)
            # Short vertical end-ticks of the bracket
            for x_end in (left_col, right_col):
                ax.plot([x_end, x_end], [tick_y0, bracket_y],
                        transform=trans, color='black', lw=1.0, clip_on=False)
            # Unit name text above the bracket
            ax.text(center_col, bracket_y + 0.005, unit_disp,
                    transform=trans, ha='center', va='bottom',
                    fontsize=9, fontweight='bold', clip_on=False)

        # --- Title ---
        norm_labels = {'cols': ' (col z-scored)', 'rows': ' (row z-scored)'}
        norm_note = norm_labels.get(normalize_axis, '')
        mols_display = ', '.join(
            (display_name_map.get(m, m) if display_name_map else m) for m in query_list
        )
        ax.set_title(f"Spatial Gradient: {mols_display}{norm_note}",
                     fontweight='bold', fontsize=12, pad=30)

        plt.tight_layout()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            mol_safe = '_'.join(
                str(m).replace(' ', '_').replace(':', '').replace('/', '_')
                for m in query_list
            )

            # Save main heatmap (no embedded colorbar)
            save_path = os.path.join(output_dir, f"spatial_gradient_multi_{mol_safe}.png")
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
            print(f"  Saved spatial gradient heatmap: {save_path}")

            # Save standalone colorbar
            cbar_fig, cbar_ax = plt.subplots(figsize=(1.2, max(3, n_images * 0.25)))
            cbar_label_text = {
                'cols': 'Z-score (per cluster-molecule column)',
                'rows': 'Z-score (per image row)',
            }
            cbar_label = cbar_label_text.get(normalize_axis, value_col)
            cbar = cbar_fig.colorbar(
                cm.ScalarMappable(norm=norm_obj, cmap=cmap),
                cax=cbar_ax, orientation='vertical',
            )
            cbar.set_label(cbar_label, fontsize=10)
            cbar_fig.tight_layout()
            cbar_path = os.path.join(output_dir, f"spatial_gradient_multi_{mol_safe}_colorbar.png")
            cbar_fig.savefig(cbar_path, dpi=600, bbox_inches='tight')
            plt.close(cbar_fig)
            print(f"  Saved colorbar: {cbar_path}")

        if show_plots:
            plt.show()
        else:
            plt.close(fig)

    def plot_spatial_gradient_heatmap_multi_centroid(
        self,
        df,
        query_list,
        query_col='Query',
        value_col='Value',
        spatial_col='centroid_x',
        image_col='Image_Name',
        n_bins=50,
        unit_col='Unit',
        unit_order=None,
        output_dir=None,
        display_name_map=None,
        unit_display_map=None,
        normalize_axis='cols',
        cmap='coolwarm',
        figsize=None,
        min_bins=2,
        show_plots=False,
        spatial_label=None,
        clip_std=3,
    ):
        """
        Generate a spatial gradient heatmap with a multi-query layout where the
        spatial axis (rows) is derived from a continuous coordinate column.
        
        Plots are generated for each individual image found in the dataframe.

          - Rows    = equal-width bins of *spatial_col* (low → high, top → bottom).
          - Columns = cluster/unit groups, each subdivided into len(query_list)
                      sub-columns — one per molecule or ratio type in query_list.

        Args:
            df            : DataFrame with columns for *unit_col*, *query_col*,
                            *spatial_col*, and *value_col*.
            query_list    : Single query name (str) OR list of query/ratio names.
            query_col     : Column containing query labels (default 'Query').
            value_col     : Column containing numeric values (default 'Value').
            spatial_col   : Column for the spatial axis (default 'centroid_x').
            image_col     : Column for image names (default 'Image_Name').
            n_bins        : Number of bins along *spatial_col* (default 50).
            unit_col      : Column for unit labels (default 'Unit').
            unit_order    : Ordered list of units for columns.
            output_dir    : Directory to save outputs.
            display_name_map : Mapping for query names.
            unit_display_map : Mapping for unit names.
            normalize_axis : 'cols' (z-score per sub-column) or 'rows' (z-score per row).
            cmap          : Colormap (default 'coolwarm').
            figsize       : Figure size (w, h).
            min_bins      : Min bins with data for a unit to be included.
            show_plots    : Whether to show plots.
            spatial_label : Y-axis label.
            clip_std      : Standard deviations at which to clip normalized values (default 3).
                            Outliers are marked with an asterisk (*).
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        from matplotlib.transforms import blended_transform_factory
        import os

        # --- Normalise query_list to a list ---
        if isinstance(query_list, str):
            query_list = [query_list]

        # --- Filter df to requested queries ---
        mol_df = df[df[query_col].isin(query_list)].copy()
        if mol_df.empty:
            print(f"No data found for query: {query_list}")
            return

        # --- Resolve Image List ---
        image_list = sorted(mol_df[image_col].dropna().unique()) if image_col in mol_df.columns else [None]
        
        for img_name in image_list:
            if img_name is not None:
                img_df = mol_df[mol_df[image_col] == img_name].copy()
                img_title_label = f" [{img_name}]"
                img_file_label = str(img_name).replace(' ', '_')
            else:
                img_df = mol_df.copy()
                img_title_label = ""
                img_file_label = "all"

            # --- Validate spatial column ---
            if spatial_col not in img_df.columns:
                print(f"spatial_col '{spatial_col}' not found for {img_name}. Skipping.")
                continue
                
            img_df = img_df.dropna(subset=[spatial_col, value_col])
            if img_df.empty:
                continue

            # --- Build spatial bins (image-specific range) ---
            s_min, s_max = img_df[spatial_col].min(), img_df[spatial_col].max()
            if s_min == s_max:
                print(f"Identical spatial values for {img_name}. Skipping.")
                continue

            bin_edges = np.linspace(s_min, s_max, n_bins + 1)
            img_df['_spatial_bin'] = np.searchsorted(bin_edges[1:], img_df[spatial_col].values, side='left')
            img_df['_spatial_bin'] = img_df['_spatial_bin'].clip(0, n_bins - 1)

            # --- Compute per-unit, per-query, per-bin means ---
            means = (
                img_df
                .groupby([unit_col, query_col, '_spatial_bin'])[value_col]
                .mean()
                .reset_index()
            )

            # --- Determine valid units ---
            unit_bin_counts = means.groupby(unit_col)['_spatial_bin'].nunique()
            valid_units = unit_bin_counts[unit_bin_counts >= min_bins].index.tolist()
            if not valid_units:
                print(f"No valid units (>= {min_bins} bins) for {img_name}. Skipping.")
                continue

            # --- Resolve unit ordering ---
            if unit_display_map is not None:
                display_to_raw = {v: k for k, v in unit_display_map.items()}
                ordered_display = list(dict.fromkeys(unit_display_map.values()))
                ordered_units = [display_to_raw[d] for d in ordered_display
                                 if display_to_raw.get(d) in valid_units]
            elif unit_order is not None:
                ordered_units = [u for u in unit_order if u in valid_units]
            else:
                ordered_units = sorted(valid_units)

            if not ordered_units:
                continue

            n_units, n_mols, n_rows = len(ordered_units), len(query_list), n_bins
            n_cols = n_units * n_mols

            # --- Build data matrix ---
            data_matrix = np.full((n_rows, n_cols), np.nan)
            for ui, unit in enumerate(ordered_units):
                for mi, mol in enumerate(query_list):
                    col_idx = ui * n_mols + mi
                    subset = means[(means[unit_col] == unit) & (means[query_col] == mol)]
                    for _, row in subset.iterrows():
                        data_matrix[int(row['_spatial_bin']), col_idx] = row[value_col]

            # --- Apply normalisation and clipping ---
            is_normalized = normalize_axis in ('cols', 'rows')
            outlier_indices = []
            
            if is_normalized:
                if normalize_axis == 'cols':
                    col_means = np.nanmean(data_matrix, axis=0)
                    col_stds  = np.nanstd(data_matrix,  axis=0)
                    col_stds[col_stds == 0] = 1
                    data_matrix = (data_matrix - col_means[np.newaxis, :]) / col_stds[np.newaxis, :]
                elif normalize_axis == 'rows':
                    row_means = np.nanmean(data_matrix, axis=1)
                    row_stds  = np.nanstd(data_matrix,  axis=1)
                    row_stds[row_stds == 0] = 1
                    data_matrix = (data_matrix - row_means[:, np.newaxis]) / row_stds[:, np.newaxis]
                
                # Identify and clip outliers
                if clip_std is not None:
                    outlier_mask = np.abs(data_matrix) > clip_std
                    outlier_indices = np.argwhere(outlier_mask)
                    data_matrix = np.clip(data_matrix, -clip_std, clip_std)

            # Colour limits
            vabs = clip_std if (is_normalized and clip_std) else np.nanmax(np.abs(data_matrix))
            vmin_plot, vmax_plot = (-vabs, vabs) if is_normalized else (np.nanmin(data_matrix), np.nanmax(data_matrix))
            plot_matrix = np.where(np.isnan(data_matrix), 0.0, data_matrix)

            # --- Figure sizing ---
            if figsize is None:
                fig_w = max(6, n_cols * 0.55 + 2.5)
                fig_h = max(4, n_rows * 0.08 + 1.8)
                figsize_img = (fig_w, fig_h)
            else:
                figsize_img = figsize

            fig, ax = plt.subplots(figsize=figsize_img)
            norm_obj = mcolors.Normalize(vmin=vmin_plot, vmax=vmax_plot)
            ax.imshow(plot_matrix, aspect='auto', cmap=cmap, norm=norm_obj, interpolation='nearest')

            # --- Mark outliers with asterisks ---
            for r, c in outlier_indices:
                # Contrast color for the asterisk
                val = plot_matrix[r, c]
                text_color = 'white' if abs(val) > vabs * 0.5 else 'black'
                ax.text(c, r, '*', color=text_color, ha='center', va='center', fontweight='bold', fontsize=10)

            ax.set_xlim(-0.5, n_cols - 0.5)
            ax.set_ylim(n_rows - 0.5, -0.5)

            # --- Separators ---
            for ui in range(n_units):
                for mi in range(1, n_mols):
                    x = ui * n_mols + mi - 0.5
                    ax.axvline(x, color='white', linewidth=0.5, alpha=0.5, zorder=3)
                if ui > 0:
                    x = ui * n_mols - 0.5
                    ax.axvline(x, color='white', linewidth=2.0, zorder=4)

            # --- X-axis ---
            xtick_positions = list(range(n_cols))
            xtick_labels = []
            for _ in range(n_units):
                for mol in query_list:
                    disp = display_name_map.get(mol, mol) if display_name_map else mol
                    xtick_labels.append(disp)
            ax.set_xticks(xtick_positions)
            ax.set_xticklabels(xtick_labels, rotation=45, ha='right', fontsize=8)
            ax.tick_params(axis='x', which='both', length=0)

            # --- Y-axis ---
            n_yticks = min(6, n_rows)
            ytick_indices = np.linspace(0, n_rows - 1, n_yticks, dtype=int)
            ytick_labels_vals = [f"{bin_edges[i]:.1f}" for i in ytick_indices]
            ax.set_yticks(ytick_indices)
            ax.set_yticklabels(ytick_labels_vals, fontsize=7)
            _spatial_label = spatial_label or f"← low {spatial_col}    |    high {spatial_col} →"
            ax.set_ylabel(_spatial_label, fontsize=9, labelpad=6)

            # --- Brackets ---
            trans = blended_transform_factory(ax.transData, ax.transAxes)
            for ui, unit in enumerate(ordered_units):
                center_col = ui * n_mols + (n_mols - 1) / 2.0
                left_col, right_col = ui * n_mols - 0.4, ui * n_mols + n_mols - 1 + 0.4
                unit_disp = unit_display_map.get(unit, unit) if unit_display_map else unit
                ax.plot([left_col, right_col], [1.045, 1.045], transform=trans, color='black', lw=1.0, clip_on=False)
                for x_end in (left_col, right_col):
                    ax.plot([x_end, x_end], [1.01, 1.045], transform=trans, color='black', lw=1.0, clip_on=False)
                ax.text(center_col, 1.05, unit_disp, transform=trans, ha='center', va='bottom', fontsize=9, fontweight='bold', clip_on=False)

            # --- Title ---
            # norm_labels = {'cols': ' (col z-scored)', 'rows': ' (row z-scored)'}
            # norm_note = norm_labels.get(normalize_axis, '')
            # mols_display = ', '.join((display_name_map.get(m, m) if display_name_map else m) for m in query_list)
            # ax.set_title(f"Spatial Gradient [{spatial_col}]{img_title_label}: {mols_display}{norm_note}", fontweight='bold', fontsize=12, pad=30)
            plt.tight_layout()

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                mol_safe = '_'.join(str(m).replace(' ', '_').replace(':', '').replace('/', '_') for m in query_list)
                spatial_safe = spatial_col.replace(' ', '_')
                save_path = os.path.join(output_dir, f"spatial_gradient_centroid_{img_file_label}_{spatial_safe}_{mol_safe}.png")
                plt.savefig(save_path, dpi=600, bbox_inches='tight')
                
                # Colorbar
                cbar_fig, cbar_ax = plt.subplots(figsize=(1.2, max(3, n_rows * 0.05)))
                cbar_label = {'cols': 'Z-score (col)', 'rows': 'Z-score (row)'}.get(normalize_axis, value_col)
                cbar = cbar_fig.colorbar(cm.ScalarMappable(norm=norm_obj, cmap=cmap), cax=cbar_ax, orientation='vertical')
                cbar.set_label(cbar_label, fontsize=10)
                cbar_fig.tight_layout()
                cbar_path = os.path.join(output_dir, f"spatial_gradient_centroid_{img_file_label}_{spatial_safe}_{mol_safe}_colorbar.png")
                cbar_fig.savefig(cbar_path, dpi=600, bbox_inches='tight')
                plt.close(cbar_fig)

            if show_plots:
                plt.show()
            else:
                plt.close(fig)


    def plot_centroid_spatial_gradient(
        self,
        df,
        ratio_col='Ratio_Type',
        value_col='Mean_Ratio',
        unit_col='Unit',
        image_col='Image_Name',
        x_col='centroid_x',
        y_col='centroid_y',
        ratio_order=None,
        unit_order=None,
        units_to_display=None,
        unit_display_names=None,
        unit_display_map=None,
        ratio_display_map=None,
        cmap='coolwarm',
        normalize='unit',
        point_size=6,
        alpha=0.85,
        raster_bins=200,
        show_raster=True,
        show_scatter=True,
        output_dir=None,
        show_plots=False,
        figsize=None,
        dpi=200,
    ):
        """
        For each image in *df*, produce a subplot grid (rows = Units, cols = Ratio_Types)
        where every cell is a spatial map of ``value_col`` coloured by the ratio value at
        each instance's ``(x_col, y_col)`` centroid.

        Two rendering modes can be combined:
          - **show_raster** (default True)  — a 2-D binned mean histogram rasterised with
            ``plt.imshow``, giving a smooth spatial background.
          - **show_scatter** (default True) — individual centroid dots overlaid on top,
            coloured by the same value.

        Args:
            df              : DataFrame with at least the columns listed below.
            ratio_col       : Column holding ratio / query type labels   (default 'Ratio_Type').
            value_col       : Column holding the numeric value to colour  (default 'Mean_Ratio').
            unit_col        : Column holding cell-type / unit labels      (default 'Unit').
            image_col       : Column holding image identifiers            (default 'Image_Name').
            x_col           : Column holding the X centroid coordinate    (default 'centroid_x').
            y_col           : Column holding the Y centroid coordinate    (default 'centroid_y').
            ratio_order       : Optional list specifying the column order of ratio types.
                                Defaults to sorted unique values.
            unit_order        : Optional list specifying the row order of units (raw names).
                                Defaults to sorted unique values.
            units_to_display  : Optional list of raw unit names to include.  Units not in
                                this list are silently dropped before ordering.  If None,
                                all units present in *df* are shown.
            unit_display_names: Optional **ordered** dict ``{raw_unit: display_name}``.
                                - Its **keys** define which units to show AND their row order
                                  (insertion order is respected).
                                - Its **values** are the Y-axis label strings.
                                When provided this takes precedence over both *units_to_display*
                                and *unit_order* / *unit_display_map* for ordering and labelling.
            unit_display_map  : Optional dict {raw_unit -> display_name} for Y-axis labels.
                                Used when *unit_display_names* is not provided.
            ratio_display_map : Optional dict {raw_ratio -> display_name} for column headers.
            cmap            : Matplotlib colormap (default 'coolwarm').
            normalize       : How to set vmin/vmax per subplot.
                              'unit'   — shared scale across all ratio types for the same unit
                                         (row-wise min/max).
                              'ratio'  — shared scale across all units for the same ratio type
                                         (column-wise min/max).
                              'global' — one shared scale across the whole figure.
                              None     — each subplot auto-scales independently.
            point_size      : Scatter marker size (default 6).
            alpha           : Scatter marker alpha (default 0.85).
            raster_bins     : Number of bins along each axis for the 2-D histogram raster
                              (default 200).
            show_raster     : Whether to show the 2-D binned mean background (default True).
            show_scatter    : Whether to overlay individual centroid dots (default True).
            output_dir      : Directory to save figures (one PNG per image).
            show_plots      : Whether to display figures interactively.
            figsize         : (width, height) override; auto-sized if None.
            dpi             : Resolution for saved figures (default 200).
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np
        import os

        required_cols = {ratio_col, value_col, unit_col, image_col, x_col, y_col}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing columns: {missing}")

        # ── Resolve ordering ────────────────────────────────────────────────
        all_ratios = sorted(df[ratio_col].dropna().unique()) if ratio_order is None else list(ratio_order)

        # unit_display_names is the primary authority: keys = selection + order + labels
        if unit_display_names is not None:
            # Keys define the selection and row order; values are the display labels
            _available = set(df[unit_col].dropna().unique())
            all_units = [u for u in unit_display_names.keys() if u in _available]
            # Build a merged display map: unit_display_names values override unit_display_map
            _merged_display = dict(unit_display_map or {})
            _merged_display.update(unit_display_names)
            unit_display_map = _merged_display
        elif unit_order is not None:
            all_units = list(unit_order)
        elif unit_display_map is not None:
            # unit_display_map insertion order drives row ordering
            display_to_raw = {v: k for k, v in unit_display_map.items()}
            ordered_display = list(dict.fromkeys(unit_display_map.values()))
            _available = set(df[unit_col].dropna().unique())
            all_units = [display_to_raw[d] for d in ordered_display
                         if display_to_raw.get(d) in _available]
        else:
            all_units = sorted(df[unit_col].dropna().unique())

        # Apply units_to_display filter (only when unit_display_names did NOT already filter)
        if units_to_display is not None and unit_display_names is None:
            _keep = set(units_to_display)
            all_units = [u for u in all_units if u in _keep]

        n_units  = len(all_units)
        n_ratios = len(all_ratios)

        if n_units == 0 or n_ratios == 0:
            print("No units or ratio types found — nothing to plot.")
            return

        # ── Pre-compute global colour limits per normalization mode ─────────
        def _lims(values):
            """Return (vmin, vmax) clipped to the 2nd–98th percentile range."""
            values = np.asarray(values, dtype=float)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                return 0.0, 1.0
            lo, hi = np.nanpercentile(values, 2), np.nanpercentile(values, 98)
            if lo == hi:
                lo, hi = lo - 0.5, hi + 0.5
            return float(lo), float(hi)

        # Pre-compute limit tables
        global_lims  = _lims(df[value_col])                                   # (lo, hi)
        unit_lims    = {u: _lims(df.loc[df[unit_col] == u, value_col])
                        for u in all_units}
        ratio_lims   = {r: _lims(df.loc[df[ratio_col] == r, value_col])
                        for r in all_ratios}

        def _subplot_lims(unit, ratio):
            if normalize == 'global':
                return global_lims
            elif normalize == 'unit':
                return unit_lims[unit]
            elif normalize == 'ratio':
                return ratio_lims[ratio]
            else:
                # independent — caller must pass actual values
                return None

        # ── Per-image loop ──────────────────────────────────────────────────
        image_names = sorted(df[image_col].dropna().unique())

        for image_name in image_names:
            img_df = df[df[image_col] == image_name].copy()
            if img_df.empty:
                continue

            # Only keep units / ratios present in this image
            present_units  = [u for u in all_units  if u in img_df[unit_col].values]
            present_ratios = [r for r in all_ratios if r in img_df[ratio_col].values]
            if not present_units or not present_ratios:
                continue

            n_rows = len(present_units)
            n_cols = len(present_ratios)

            # ── Figure sizing ────────────────────────────────────────────
            if figsize is None:
                fig_w = max(4 * n_cols + 1.5, 8)
                fig_h = max(3.2 * n_rows + 1.2, 5)
                _figsize = (fig_w, fig_h)
            else:
                _figsize = figsize

            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=_figsize,
                squeeze=False,
            )

            # Global spatial extent for this image (consistent axes across all subplots)
            x_all = img_df[x_col].values
            y_all = img_df[y_col].values
            x_min, x_max = float(np.nanmin(x_all)), float(np.nanmax(x_all))
            y_min, y_max = float(np.nanmin(y_all)), float(np.nanmax(y_all))
            pad_x = max((x_max - x_min) * 0.02, 10)
            pad_y = max((y_max - y_min) * 0.02, 10)

            for ri, unit in enumerate(present_units):
                unit_df = img_df[img_df[unit_col] == unit]
                unit_disp = (unit_display_map.get(unit, unit)
                             if unit_display_map else unit)

                for ci, ratio in enumerate(present_ratios):
                    ax = axes[ri][ci]
                    cell_df = unit_df[unit_df[ratio_col] == ratio].dropna(subset=[x_col, y_col, value_col])

                    lims = _subplot_lims(unit, ratio)
                    if lims is None and not cell_df.empty:
                        lims = _lims(cell_df[value_col])
                    elif lims is None:
                        lims = (0.0, 1.0)

                    vmin, vmax = lims
                    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

                    # ── Raster background ─────────────────────────────
                    if show_raster and not cell_df.empty:
                        xs = cell_df[x_col].values.astype(float)
                        ys = cell_df[y_col].values.astype(float)
                        zs = cell_df[value_col].values.astype(float)

                        # Build 2-D grid covering the full image extent
                        x_edges = np.linspace(x_min - pad_x, x_max + pad_x, raster_bins + 1)
                        y_edges = np.linspace(y_min - pad_y, y_max + pad_y, raster_bins + 1)

                        # Bin indices for each point
                        xi = np.clip(np.searchsorted(x_edges, xs, side='right') - 1,
                                     0, raster_bins - 1)
                        yi = np.clip(np.searchsorted(y_edges, ys, side='right') - 1,
                                     0, raster_bins - 1)

                        # Accumulate sum and count per bin
                        sum_grid   = np.full((raster_bins, raster_bins), np.nan)
                        count_grid = np.zeros((raster_bins, raster_bins), dtype=int)
                        np.add.at(count_grid, (yi, xi), 1)
                        tmp_sum = np.zeros((raster_bins, raster_bins), dtype=float)
                        np.add.at(tmp_sum, (yi, xi), zs)

                        mask = count_grid > 0
                        sum_grid[mask] = tmp_sum[mask] / count_grid[mask]

                        ax.imshow(
                            sum_grid,
                            origin='upper',
                            extent=[x_min - pad_x, x_max + pad_x,
                                    y_max + pad_y, y_min - pad_y],
                            aspect='equal',
                            cmap=cmap,
                            norm=norm,
                            interpolation='bilinear',
                            zorder=1,
                        )

                    # ── Scatter dots ──────────────────────────────────
                    if show_scatter and not cell_df.empty:
                        sc = ax.scatter(
                            cell_df[x_col],
                            cell_df[y_col],
                            c=cell_df[value_col],
                            cmap=cmap,
                            norm=norm,
                            s=point_size,
                            alpha=alpha,
                            linewidths=0,
                            zorder=2,
                        )

                    # ── Axes cosmetics ────────────────────────────────
                    ax.set_xlim(x_min - pad_x, x_max + pad_x)
                    ax.set_ylim(y_max + pad_y, y_min - pad_y)   # image-style: y inverted
                    # ax.set_aspect('equal')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)

                    # Column header (only top row)
                    if ri == 0:
                        ratio_disp = (ratio_display_map.get(ratio, ratio)
                                      if ratio_display_map else ratio)
                        ax.set_title(ratio_disp, fontsize=9, fontweight='bold', pad=4)

                    # Row label (only leftmost column)
                    if ci == 0:
                        ax.set_ylabel(unit_disp, fontsize=8, fontweight='bold',
                                      rotation=0, ha='right', va='center',
                                      labelpad=6)

                    # Empty-data notice
                    if cell_df.empty:
                        ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                                ha='center', va='center', fontsize=7, color='grey')

            # ── Shared colourbar (one per normalization scope) ────────────
            # We place a single colourbar on the right side of the figure
            import matplotlib.cm as cm
            if normalize in ('global', 'unit', 'ratio', None):
                # Use global limits for the shared colorbar label
                cb_vmin, cb_vmax = global_lims
                cb_norm = mcolors.Normalize(vmin=cb_vmin, vmax=cb_vmax)
                cb_ax = fig.add_axes([0.92, 0.15, 0.015, 0.70])
                cbar = fig.colorbar(
                    cm.ScalarMappable(norm=cb_norm, cmap=cmap),
                    cax=cb_ax,
                    orientation='vertical',
                )
                cbar.set_label(value_col.replace('_', ' '), fontsize=9)
                cbar.ax.tick_params(labelsize=7)

            # ── Figure title ──────────────────────────────────────────────
            norm_note = (
                f" — colour normalised per {'unit (row)' if normalize == 'unit' else 'ratio (col)' if normalize == 'ratio' else 'global'}"
                if normalize else ""
            )
            fig.suptitle(
                f"Spatial Centroid Gradient  |  {image_name}{norm_note}",
                fontsize=11, fontweight='bold', y=1.01,
            )

            plt.tight_layout(rect=[0, 0, 0.91, 1.0])

            # ── Save / show ───────────────────────────────────────────────
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                safe_name = str(image_name).replace(' ', '_').replace('/', '_').replace(':', '')
                save_path = os.path.join(output_dir, f"centroid_gradient_{safe_name}.png")
                plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
                print(f"  Saved: {save_path}")

            if show_plots:
                plt.show()
            else:
                plt.close(fig)

    def plot_topographical_intensity_map(self, image,
                                        downsample_factor=1, sigma=0, padding=0,
                                        cmap='magma', alpha=1.0, elevation=40, azimuth=90, 
                                        output_dir=None, show_plots=False):
        """
        Create a clean 3D topographical map based on image intensity.
        
        This treats the 2D image intensity as elevation (Z-axis) and renders a clean
        surface without axes, grid, or unit markers.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.ndimage import zoom
        import os

        # Crop by non-zero pixels
        non_zero_mask = image != 0

        # Calculate bounds of non-zero pixels
        coords = np.argwhere(non_zero_mask)
        min_y, min_x = coords.min(axis=0)
        max_y, max_x = coords.max(axis=0)

        # Add padding
        min_x = max(0, min_x - padding)
        max_x = min(image.shape[1], max_x + padding)
        min_y = max(0, min_y - padding)
        max_y = min(image.shape[0], max_y + padding)

        img_crop = image[min_y:max_y, min_x:max_x].copy()
            
        # Apply smoothing if requested
        if sigma > 0:
            img_crop = ndi.gaussian_filter(img_crop.astype(float), sigma=sigma)

        # Prepare image data for plotting
        if downsample_factor > 1:
            plot_image = zoom(img_crop, 1/downsample_factor, order=1)
        else:
            plot_image = img_crop
                
        h, w = plot_image.shape
        x_mesh = np.arange(0, w)
        y_mesh = np.arange(0, h)
        X, Y = np.meshgrid(x_mesh, y_mesh)
            
        fig = plt.figure(figsize=(12, 10), facecolor='white')
        ax = fig.add_subplot(111, projection='3d')
            
        # Use strides to handle large images efficiently
        r_stride = max(1, h // 100)
        c_stride = max(1, w // 100)
        
        rcount = 100
        ccount = int(rcount * h/w)
        # Plot the surface
        surf = ax.plot_surface(X, Y, plot_image, 
        rcount=rcount,
        ccount=ccount,
                               cmap=cmap, alpha=alpha, linewidth=0, 
                               antialiased=True, shade=False)
            
        # Set view and aspect
        ax.view_init(elev=elevation, azim=azimuth)
        ax.set_box_aspect([1, h/w, 0.3])
            
        # REMOVE ALL AXES, GRID, AND SPINES
        ax.set_axis_off()
        ax.grid(False)
            
        # Remove pane backgrounds (make them transparent)
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            
        plt.tight_layout()
            
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"topographical_map.png")
            plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
            print(f"Saved topographical map to {output_path}")
            
        if show_plots:
            plt.show()
        else:
            plt.close(fig)


    def generate_unit_heatmaps(self, df, value_col, grouping_col, output_dir, data_type='percentage',
                               figsize=(8, 6), show_plots=False, cmap=None, vmin=-2, vmax=2, display_name_map=None, unit_display_map=None, unit_order=None,
                               units_to_display=None, top_n=20, groups_to_display=None, remove_outliers=False, **kwargs):
        """
        Generate heatmaps showing each molecule across units (aggregating across Group_IDs).
        
        This creates z-score normalized heatmaps where each row is a molecule/ratio type
        and each column is a unit, with values aggregated across all Group_IDs within each unit.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing the data with columns for Unit, value_col, and grouping_col
        value_col : str
            Name of the column containing the values to aggregate (e.g., 'Percentage', 'Mean_Ratio')
        grouping_col : str
            Name of the column to group by (e.g., 'Molecule', 'Ratio_Type')
        output_dir : str
            Directory to save the heatmap plots
        data_type : str
            Type of data being visualized (for labeling)
        figsize : tuple
            Figure size (width, height)
        show_plots : bool
            Whether to display plots interactively
        cmap : str or Colormap
            Colormap for heatmap (default: SVG_BUBBLE_CMAP)
        vmin, vmax : float
            Min and max values for colormap normalization
        unit_order : list, optional
            Custom order for units on x-axis. Should be a list of unit names.
            If None, uses default ordering (units with both Group_IDs first, then single-Group_ID units)
        units_to_display : list, optional
            List of unit names to include; all others are excluded.
        top_n : int, optional
            Maximum number of top molecules to display (by z-score magnitude) when
            grouping_col is 'Molecule'. Default is 20. Ignored if groups_to_display is set.
        groups_to_display : list, optional
            Explicit list of molecule or ratio names to include. When provided, overrides
            top_n filtering and shows only these specific entries.
            
        Returns:
        --------
        dict : Dictionary with heatmap matrices and statistics
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if cmap is None:
            cmap = SVG_BUBBLE_CMAP

        # Filter to selected units if specified
        if units_to_display is not None:
            df = df[df['Unit'].isin(units_to_display)].copy()

        # Get unique molecules/ratio types and units
        unique_groups = df[grouping_col].unique()
        unique_groups = [group.strip() for group in unique_groups]
        unique_units = df['Unit'].unique()
        
        if len(unique_groups) < 2 or len(unique_units) < 2:
            print(f"Insufficient data for unit heatmap: {len(unique_groups)} groups, {len(unique_units)} units")
            return {}
        
        # Check if secondary grouping column exists (e.g., Group_ID, Treatment)
        secondary_group_col = kwargs.get('secondary_group_col', 'Group_ID' if 'Group_ID' in df.columns else None)
        has_Group_ID = secondary_group_col is not None and df[secondary_group_col].notna().any()
        
        # Apply outlier filtration before aggregation to ensure stable Z-scores
        filter_groups = ['Unit', secondary_group_col, grouping_col] if has_Group_ID else ['Unit', grouping_col]
        if remove_outliers:
            df_clean = self._apply_outlier_filtration(df, value_col, filter_groups)
        else:
            df_clean = df
        
        if has_Group_ID:
            # Calculate mean for each Unit-SecondaryGroup-Group combination
            agg_data = df_clean.groupby(['Unit', secondary_group_col, grouping_col])[value_col].agg(['mean', 'std', 'count']).reset_index()
            # Create a combined column label for Unit-SecondaryGroup using standardized delimiter
            agg_data['Unit_Group_Combo'] = agg_data['Unit'] + ' (' + agg_data[secondary_group_col].astype(str) + ')'
        else:
            # Calculate mean for each Unit-Group combination (no secondary groups)
            agg_data = df_clean.groupby(['Unit', grouping_col])[value_col].agg(['mean', 'std', 'count']).reset_index()
            agg_data['Unit_Group_Combo'] = agg_data['Unit']
        
        # Remove NaN and infinite values
        agg_data = agg_data.replace([np.inf, -np.inf], np.nan)
        agg_data = agg_data.dropna(subset=['mean'])
        
        if len(agg_data) < 2:
            print("Insufficient data after removing NaN/inf values")
            return {}
        
        # Calculate z-scores normalized to each group's own mean across all unit-Group_ID combinations
        # For each molecule/ratio, normalize to its mean across all units (and Group_IDs if present)
        def normalize_group(group):
            group_mean = group['mean'].mean()
            group_std = group['mean'].std()
            if group_std == 0:
                group['z_score'] = 0
            else:
                group['z_score'] = (group['mean'] - group_mean) / group_std
            return group
        
        agg_data = agg_data.groupby(grouping_col, group_keys=False).apply(normalize_group)
        
        # Create pivot table for heatmap: Groups as rows, Unit_Group_ID as columns
        try:
            if groups_to_display is not None:
                agg_sub = agg_data[agg_data[grouping_col].isin(groups_to_display)]
            elif grouping_col == 'Molecule':
                # Filter top N (or fewer) z-score magnitude values
                n_top = np.minimum(top_n, df[grouping_col].nunique())
                top_molecules = agg_data.iloc[agg_data['z_score'].abs().nlargest(n_top, keep='all').index][grouping_col].unique() 
                agg_sub = agg_data[agg_data[grouping_col].isin(top_molecules)]  
            else:
                agg_sub = agg_data

            # Standardized pivot logic: aggregate by mean to handle any potential duplicate Unit_Group_ID labels
            heatmap_matrix = agg_sub.pivot_table(index='Unit_Group_Combo', columns=grouping_col, values='z_score', aggfunc='mean')
            mean_matrix = agg_sub.pivot_table(index='Unit_Group_Combo', columns=grouping_col, values='mean', aggfunc='mean')
            
            # Ensure alignment
            mean_matrix = mean_matrix.reindex(index=heatmap_matrix.index, columns=heatmap_matrix.columns)
            
            # Mask values with 0 mean or NaN to ensure they are treated as missing/zero
            mask = (mean_matrix == 0) | mean_matrix.isna()
            heatmap_matrix = heatmap_matrix.mask(mask, np.nan)
            
            
            # Sort columns by display_name_map order if provided, otherwise alphabetically
            if display_name_map:
                # Get current columns (molecule/ratio names)
                current_columns = heatmap_matrix.columns.tolist()
                # Create ordered list: first items in display_name_map order, then remaining alphabetically
                ordered_columns = [k for k in display_name_map.keys() if k in current_columns]
                remaining_columns = sorted([g for g in current_columns if g not in display_name_map.keys()])
                new_columns = ordered_columns + remaining_columns
                heatmap_matrix = heatmap_matrix[new_columns]

                
            else:
                # Sort columns alphabetically
                heatmap_matrix = heatmap_matrix.sort_index(axis=1)
            
            # Sort rows based on unit_order if provided, otherwise use default ordering
            if unit_order is not None:
                # Custom ordering based on provided unit_order list
                all_rows = heatmap_matrix.index.tolist()
                ordered_rows = []
                for unit in unit_order:
                    # Stricter matching: only match rows starting with unit name or exactly equal to it
                    # (Standardized with bubble chart logic to prevent accidental substring matches)
                    matching_rows = [r for r in all_rows if r.startswith(unit + ' (') or r == unit]
                    ordered_rows.extend(sorted(matching_rows))
                # Only keep rows that match unit_order (exclude unlisted units)
                heatmap_matrix = heatmap_matrix.reindex(ordered_rows).dropna(how='all')
            elif has_Group_ID:
                # Default ordering: units with both Group_IDs first, then single-Group_ID units
                all_rows = heatmap_matrix.index.tolist()
                # Identify which units have both sub-groups
                unit_group_map = {}  # {base_unit: [list of unit_group_combo rows]}
                for row in all_rows:
                    if ' (' in row:
                        base_unit = row.split(' (')[0]
                        if base_unit not in unit_group_map:
                            unit_group_map[base_unit] = [row]
                        else:
                            unit_group_map[base_unit].append(row)
                    else:
                        if row not in unit_group_map:
                            unit_group_map[row] = [row]
                ordered_rows = []
                for base_unit, rows in unit_group_map.items():
                    if len(rows) > 1:
                        ordered_rows.extend(sorted(rows))
                for base_unit, rows in unit_group_map.items():
                    if len(rows) == 1:
                        ordered_rows.extend(rows)
                heatmap_matrix = heatmap_matrix.reindex(ordered_rows)
            
            # Store the data (no overall stats since each group normalized independently)
            heatmap_data = {
                'matrix': heatmap_matrix,
                'raw_data': agg_data
            }
            
            # Calculate dynamic figure size based on matrix dimensions
            n_rows = heatmap_matrix.shape[0]  # Number of unit-group combinations
            n_cols = heatmap_matrix.shape[1]  # Number of molecules/ratios
            
            # Increased scaling: ~0.5 inches per row, ~1.0 inches per column for better readability
            fig_width = max(3, 1.0 * n_cols + 2.5)  # Increased from 0.7 to 1.0
            fig_height = max(3, min(10, 0.5 * n_rows + 2.5))  # Increased from 0.35 to 0.5, increased max height

            # fig_width = max(6, 0.7 * n_cols + 2.5)
            # fig_height = max(3, min(10, 0.35 * n_rows + 2.5))
            
            # Create heatmap visualization
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            ax.set_facecolor('black')
            
            # Create heatmap without embedded colorbar
            sns.heatmap(heatmap_matrix, annot=False, cmap=cmap, 
                       center=0, vmin=vmin, vmax=vmax,
                       cbar=False,
                       linewidths=0.5, linecolor='white', ax=ax, square=False)
            
            # --- Colorbar saved separately ---
            import matplotlib.cm as cm
            import matplotlib.colors as mcolors
            cbar_fig, cbar_ax = plt.subplots(figsize=(1.2, max(3, n_rows * 0.35)))
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax) if (vmin < 0 and vmax > 0) else mcolors.Normalize(vmin=vmin, vmax=vmax)
            cbar = cbar_fig.colorbar(
                cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=cbar_ax,
                orientation='vertical',
            )
            cbar.set_label('Z-score (normalized to mean)', color='black', fontsize=14)
            cbar.ax.yaxis.set_tick_params(color='black', labelcolor='black', labelsize=12)
            cbar.outline.set_edgecolor('black')
            cbar_fig.tight_layout()
            cbar_path = os.path.join(output_dir, f"heatmap_unit_aggregated_{data_type}_colorbar.png")
            cbar_fig.savefig(cbar_path, dpi=600, bbox_inches='tight', facecolor='white')
            plt.close(cbar_fig)
            print(f"  Saved colorbar: {cbar_path}")
            
            # Add visual separators between different units if showing grouping combinations
            if has_Group_ID:
                row_names = heatmap_matrix.index.tolist()
                unit_changes = []
                prev_unit = None
                for idx, row_name in enumerate(row_names):
                    # Extract unit name (before the parenthesis)
                    current_unit = row_name.split(' (')[0] if ' (' in row_name else row_name
                    if prev_unit is not None and current_unit != prev_unit:
                        unit_changes.append(idx)
                    prev_unit = current_unit
                # Draw horizontal lines at unit boundaries (since units are now on rows)
                for boundary in unit_changes:
                    ax.axhline(y=boundary, color='gray', linewidth=1.4, zorder=10)
            
            # Formatting
            if has_Group_ID:
                # Update y-tick labels to use standardized format and apply unit_display_map
                current_labels = [label.get_text() for label in ax.get_yticklabels()]
                new_labels = []
                for label in current_labels:
                    # Extract unit name (before parenthesis)
                    if ' (' in label:
                        unit_part, Group_ID_part = label.split(' (', 1)
                        # Apply unit display map
                        if unit_display_map and unit_part in unit_display_map:
                            unit_part = unit_display_map[unit_part]
                        # Use full Group_ID name instead of abbreviation
                        Group_ID_full = Group_ID_part.rstrip(')')
                        if Group_ID_full in ['O', 'Other', '', None]:
                            new_labels.append(f"{unit_part}")
                        else:
                            new_labels.append(f"{unit_part} ({Group_ID_full})")
                    else:
                        # Apply unit display map to label without Group_ID
                        if unit_display_map and label in unit_display_map:
                            new_labels.append(unit_display_map[label])
                        else:
                            new_labels.append(label.replace('_', ' '))
                ax.set_yticklabels(new_labels, rotation=0, ha='right', fontsize=14)
                ax.set_ylabel('Unit (Group_ID)', fontweight='bold', fontsize=16)
            else:
                # Apply unit_display_map to y-axis labels (without Group_IDs)
                y_labels = []
                for label in ax.get_yticklabels():
                    original_name = label.get_text()
                    if unit_display_map and original_name in unit_display_map:
                        y_labels.append(unit_display_map[original_name])
                    else:
                        y_labels.append(original_name.replace('_', ' '))
                ax.set_yticklabels(y_labels, rotation=0, ha='right', fontsize=14)
                ax.set_ylabel('Unit', fontweight='bold', fontsize=16)
            
            ax.set_xlabel(grouping_col.replace("_", " "), fontweight='bold', fontsize=16)
            # Update title based on data type
            if data_type == 'percentage':
                ax.set_title('Normalized Molecule Percentage',
                           fontweight='bold', fontsize=18, pad=20)
            else:
                ax.set_title('Normalized Ratio',
                           fontweight='bold', fontsize=18, pad=20)
            
            # Apply display_name_map to x-axis labels (now molecules are on x-axis)
            x_labels = []
            for label in ax.get_xticklabels():
                original_name = label.get_text()
                if display_name_map and original_name in display_name_map:
                    mapped_name = display_name_map[original_name]
                    x_labels.append(mapped_name)
                else:
                    # Always add label even if not in display_name_map
                    x_labels.append(original_name.replace('_', ' '))
            ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=14)
            
            # Style the axes with light gray borders to match separator lines
            for spine in ax.spines.values():
                spine.set_edgecolor('lightgray')
                spine.set_linewidth(1.4)
            
            plt.tight_layout()
            
            # Save figure
            output_path = os.path.join(output_dir, 
                                      f"heatmap_unit_aggregated_{data_type}.png")
            plt.savefig(output_path, dpi=600, bbox_inches='tight')
            
            if show_plots:
                plt.show()
            else:
                plt.close(fig)
            
            print(f"\nGenerated unit-aggregated heatmap in {output_dir}")
            return heatmap_data
                
        except Exception as e:
            print(f"Error creating unit-aggregated heatmap: {str(e)}")
            return {}

    def generate_unit_bubble_charts(self, df, value_col, grouping_col, output_dir, data_type='percentage',
                                   show_plots=False, cmap=None, vmin=-2, vmax=2,
                                   display_name_map=None, unit_display_map=None, unit_order=None,
                                   units_to_display=None, top_n=20, groups_to_display=None, remove_outliers=False, **kwargs):
        """
        Generate a bubble chart showing each molecule across units (aggregating across Group_IDs).

        Mirrors generate_unit_heatmaps but renders bubbles instead of filled cells:
          - Bubble size encodes the magnitude of the significance (1-vs-rest T-test)
          - Bubble color encodes the sign/magnitude via the chosen colormap (z-score)
          - White figure/axes background
          - X-axis tick labels (molecules/ratios) are placed above the plot

        Standardized logic:
          - Statistics (bubble size) are always calculated on RAW data.
          - Visuals (bubble color/z-scores) respect the remove_outliers toggle.
        """
        import matplotlib.colors as mcolors
        from scipy.stats import ttest_ind
        from matplotlib.lines import Line2D

        os.makedirs(output_dir, exist_ok=True)

        if cmap is None:
            cmap = SVG_BUBBLE_CMAP

        # 1. Filter to selected units if specified
        if units_to_display is not None:
            df = df[df['Unit'].isin(units_to_display)].copy()

        # 2. Prepare raw data for statistics (clean NaNs/infs but keep outliers)
        df_stats = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[value_col]).copy()

        # 3. Prepare filtered data for plotting (if requested)
        secondary_group_col = kwargs.get('secondary_group_col', 'Group_ID' if 'Group_ID' in df.columns else None)
        has_Group_ID = secondary_group_col is not None and df[secondary_group_col].notna().any()
        filter_groups = ['Unit', secondary_group_col, grouping_col] if has_Group_ID else ['Unit', grouping_col]
        
        if remove_outliers:
            df_plot = self._apply_outlier_filtration(df, value_col, filter_groups)
        else:
            df_plot = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[value_col]).copy()

        # 4. Aggregate plotting data
        if has_Group_ID:
            agg_data = df_plot.groupby(['Unit', secondary_group_col, grouping_col])[value_col].agg(['mean', 'std', 'count']).reset_index()
            agg_data['Unit_Group_Combo'] = agg_data['Unit'] + ' (' + agg_data[secondary_group_col].astype(str) + ')'
        else:
            agg_data = df_plot.groupby(['Unit', grouping_col])[value_col].agg(['mean', 'std', 'count']).reset_index()
            agg_data['Unit_Group_Combo'] = agg_data['Unit']

        agg_data = agg_data.dropna(subset=['mean'])

        # Calculate Z-scores across units for each molecule (using the plotting data)
        def normalize_group(group):
            group_mean = group['mean'].mean()
            group_std = group['mean'].std()
            if group_std == 0:
                group['z_score'] = 0
            else:
                group['z_score'] = (group['mean'] - group_mean) / group_std
            return group

        agg_data = agg_data.groupby(grouping_col, group_keys=False).apply(normalize_group)

        try:
            # Filtering molecules
            if groups_to_display is not None:
                agg_sub = agg_data[agg_data[grouping_col].isin(groups_to_display)]
            elif grouping_col == 'Molecule':
                n_top = np.minimum(top_n, df[grouping_col].nunique())
                # Use absolute z-score for top_n selection
                top_molecules = agg_data.groupby(grouping_col)['z_score'].apply(lambda x: x.abs().max()).nlargest(n_top).index
                agg_sub = agg_data[agg_data[grouping_col].isin(top_molecules)]
            else:
                agg_sub = agg_data

            # Create matrices for plotting
            heatmap_matrix = agg_sub.pivot_table(index='Unit_Group_Combo', columns=grouping_col, values='z_score', aggfunc='mean')
            mean_matrix = agg_sub.pivot_table(index='Unit_Group_Combo', columns=grouping_col, values='mean', aggfunc='mean')
            
            # Sort columns
            if display_name_map:
                current_columns = heatmap_matrix.columns.tolist()
                ordered_columns = [k for k in display_name_map.keys() if k in current_columns]
                remaining_columns = sorted([g for g in current_columns if g not in display_name_map.keys()])
                heatmap_matrix = heatmap_matrix[ordered_columns + remaining_columns]
            else:
                heatmap_matrix = heatmap_matrix.sort_index(axis=1)

            # Sort rows based on unit_order if provided
            if unit_order is not None:
                all_rows = heatmap_matrix.index.tolist()
                ordered_rows = []
                for unit in unit_order:
                    matching_rows = [r for r in all_rows if r.startswith(unit + ' (') or r == unit]
                    ordered_rows.extend(sorted(matching_rows))
                heatmap_matrix = heatmap_matrix.reindex(ordered_rows).dropna(how='all')
            elif has_Group_ID:
                # Default ordering matching barplots
                all_rows = heatmap_matrix.index.tolist()
                unit_group_map = {}
                for row in all_rows:
                    base_unit = row.split(' (')[0] if ' (' in row else row
                    unit_group_map.setdefault(base_unit, []).append(row)
                
                ordered_rows = []
                # First units with multiple groups
                for base_unit in sorted(unit_group_map.keys()):
                    if len(unit_group_map[base_unit]) > 1:
                        ordered_rows.extend(sorted(unit_group_map[base_unit]))
                # Then units with single groups
                for base_unit in sorted(unit_group_map.keys()):
                    if len(unit_group_map[base_unit]) == 1:
                        ordered_rows.extend(unit_group_map[base_unit])
                heatmap_matrix = heatmap_matrix.reindex(ordered_rows)

            # Build display labels
            col_labels = [self._apply_display_name(c, display_name_map) for c in heatmap_matrix.columns]
            row_labels = []
            for row in heatmap_matrix.index:
                if ' (' in row:
                    unit_part, Group_ID_part = row.split(' (', 1)
                    unit_disp = unit_display_map[unit_part] if unit_display_map and unit_part in unit_display_map else unit_part
                    Group_ID_full = Group_ID_part.rstrip(')')
                    row_labels.append(f"{unit_disp} ({Group_ID_full})" if Group_ID_full not in ['O', 'Other', ''] else f"{unit_disp}")
                else:
                    row_labels.append(unit_display_map[row] if unit_display_map and row in unit_display_map else row.replace('_', ' '))

            # Figure sizing
            n_rows, n_cols = heatmap_matrix.shape
            spacing = 0.5
            fig_width = max(6, 0.7 * n_cols + 2.5)
            fig_height = max(3, min(10, 0.35 * n_rows + 2.5))
            
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')

            col_index = {col: i * spacing for i, col in enumerate(heatmap_matrix.columns)}
            row_index = {row: i * spacing for i, row in enumerate(heatmap_matrix.index)}

            xs, ys, sizes, color_vals = [], [], [], []
            xs_black, ys_black, sizes_black = [], [], []
            min_bubble_size = 50

            # 5. Loop through and calculate significance on RAW data
            for row_name in heatmap_matrix.index:
                for col_name in heatmap_matrix.columns:
                    z = heatmap_matrix.loc[row_name, col_name]
                    mean_val = mean_matrix.loc[row_name, col_name]
                    x_val, y_val = col_index[col_name], row_index[row_name]
                    
                    if pd.isna(mean_val) or mean_val == 0:
                        xs_black.append(x_val); ys_black.append(y_val); sizes_black.append(min_bubble_size)
                        continue
                        
                    # Extract unit and group info for stat lookup
                    matching_agg = agg_data[(agg_data['Unit_Group_Combo'] == row_name) & (agg_data[grouping_col] == col_name)]
                    if matching_agg.empty:
                        xs_black.append(x_val); ys_black.append(y_val); sizes_black.append(min_bubble_size); continue
                    
                    unit_name = matching_agg.iloc[0]['Unit']
                    secondary_val = matching_agg.iloc[0][secondary_group_col] if has_Group_ID else None
                    
                    # 1-vs-rest T-test on RAW data (df_stats)
                    if has_Group_ID and secondary_val is not None:
                        group_mask = (df_stats['Unit'] == unit_name) & (df_stats[secondary_group_col] == secondary_val) & (df_stats[grouping_col] == col_name)
                        other_mask = (~((df_stats['Unit'] == unit_name) & (df_stats[secondary_group_col] == secondary_val))) & (df_stats[grouping_col] == col_name)
                    else:
                        group_mask = (df_stats['Unit'] == unit_name) & (df_stats[grouping_col] == col_name)
                        other_mask = (df_stats['Unit'] != unit_name) & (df_stats[grouping_col] == col_name)

                    this_vals = df_stats[group_mask][value_col].values
                    other_vals = df_stats[other_mask][value_col].values
                    
                    p_val = 1.0
                    if len(this_vals) >= 2 and len(other_vals) >= 2:
                        # Assuming equal variance as requested (Tukey-Kramer assumption)
                        _, p_val = ttest_ind(this_vals, other_vals, equal_var=True)
                        if pd.isna(p_val): p_val = 1.0
                    
                    # Size based on significance
                    if p_val >= 0.05: size = min_bubble_size
                    elif p_val >= 0.01: size = 125
                    elif p_val >= 0.001: size = 200
                    else: size = 250
                    
                    xs.append(x_val); ys.append(y_val); sizes.append(size); color_vals.append(z)

            # Normalization and Plotting
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax) if vmin < 0 < vmax else mcolors.Normalize(vmin=vmin, vmax=vmax)
            if xs:
                ax.scatter(xs, ys, s=sizes, c=color_vals, cmap=cmap, norm=norm, edgecolors='none', zorder=3)
            if xs_black:
                ax.scatter(xs_black, ys_black, s=sizes_black, c='black', edgecolors='none', zorder=3)

            # Save colorbar
            import matplotlib.cm as cm
            cbar_fig, cbar_ax = plt.subplots(figsize=(1.2, max(3, n_rows * 0.35)))
            cbar = cbar_fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, orientation='vertical')
            cbar.set_label('Z-score (normalized)', color='black', fontsize=14)
            cbar_fig.savefig(os.path.join(output_dir, f"bubble_unit_aggregated_{data_type}_colorbar.png"), dpi=600, bbox_inches='tight', facecolor='white')
            plt.close(cbar_fig)

            # Horizontal separators
            if has_Group_ID:
                prev_u = None
                for ridx, row_nm in enumerate(heatmap_matrix.index):
                    curr_u = row_nm.split(' (')[0] if ' (' in row_nm else row_nm
                    if prev_u is not None and curr_u != prev_u:
                        ax.axhline(y=ridx * spacing - spacing * 0.5, color='#aaaaaa', linewidth=1.0, zorder=2)
                    prev_u = curr_u

            # Formatting
            ax.set_xticks([i * spacing for i in range(n_cols)])
            ax.set_yticks([i * spacing for i in range(n_rows)])
            ax.grid(True, color='#dddddd', linewidth=0.5, zorder=1)
            ax.xaxis.set_label_position('top'); ax.xaxis.tick_top()
            ax.set_xticklabels(col_labels, rotation=45, ha='left', fontsize=14, color='black')
            ax.set_yticklabels(row_labels, rotation=0, ha='right', fontsize=14, color='black')
            ax.set_ylabel('Unit (Group_ID)' if has_Group_ID else 'Unit', fontweight='bold', fontsize=16)
            ax.set_title('Normalized Molecule Percentage' if data_type == 'percentage' else 'Normalized Ratio', fontweight='bold', fontsize=18, pad=20)
            
            for spine in ax.spines.values(): spine.set_edgecolor('#cccccc')
            ax.set_xlim(-spacing * 0.5, (n_cols - 1) * spacing + spacing * 0.5)
            ax.set_ylim(-spacing * 0.5, (n_rows - 1) * spacing + spacing * 0.5)

            # Significance Legend
            legend_fig, legend_ax = plt.subplots(figsize=(2.5, 1.2))
            sig_handles = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=5, label='p ≥ 0.05', markeredgecolor='black', linewidth=0.5),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='p < 0.05', markeredgecolor='black', linewidth=0.5),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='p < 0.01', markeredgecolor='black', linewidth=0.5),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=12, label='p < 0.001', markeredgecolor='black', linewidth=0.5),
            ]
            legend_ax.legend(handles=sig_handles, loc='center', frameon=True, fontsize=11, title='Significance', ncol=2)
            legend_ax.axis('off')
            legend_fig.savefig(os.path.join(output_dir, "bubble_significance_legend.png"), dpi=600, bbox_inches='tight', facecolor='white')
            plt.close(legend_fig)

            plt.tight_layout()
            output_path = os.path.join(output_dir, f"bubble_unit_aggregated_{data_type}.png")
            plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
            if show_plots: plt.show()
            else: plt.close(fig)

            return {'matrix': heatmap_matrix, 'raw_data': agg_data}

        except Exception as e:
            print(f"Error creating unit-aggregated bubble chart: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}



    def generate_raw_ratio_bubble_chart(self, df, value_col='Mean_Ratio', grouping_col='Ratio_Type', 
                                        output_dir=None, show_plots=True, cmap=None, vmin=None, vmax=None,
                                        display_name_map=None, unit_display_map=None, unit_order=None,
                                        units_to_display=None, groups_to_display=None, remove_outliers=True):
        """
        Refactored raw ratio bubble chart aligning with generate_unit_bubble_charts logic.
        
        Bubble size encodes 1-vs-rest t-test significance (calculated on RAW data).
        Bubble color encodes the mean value (calculated on outlier-filtered data).
        Styling matches the standardized unit bubble charts (white background, labels above).
        """
        import matplotlib.colors as mcolors
        from scipy.stats import ttest_ind
        from matplotlib.lines import Line2D

        if output_dir is None:
            output_dir = 'raw_ratio_bubble_charts'
        os.makedirs(output_dir, exist_ok=True)
        
        if cmap is None:
            cmap = 'RdBu_r'

        try:
            # 1. Filter to selected units if specified
            if units_to_display is not None:
                df = df[df['Unit'].isin(units_to_display)].copy()

            # 2. Prepare raw data for statistics (clean NaNs/infs but keep outliers)
            df_stats = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[value_col]).copy()
            
            # 3. Prepare filtered data for plotting (if requested)
            has_Group_ID = 'Group_ID' in df.columns and df['Group_ID'].notna().any()
            filter_groups = ['Unit', 'Group_ID', grouping_col] if has_Group_ID else ['Unit', grouping_col]
            
            if remove_outliers:
                df_plot = self._apply_outlier_filtration(df, value_col, filter_groups)
            else:
                df_plot = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[value_col]).copy()
            
            # 4. Aggregate plotting data
            if has_Group_ID:
                agg_data = df_plot.groupby(['Unit', 'Group_ID', grouping_col])[value_col].agg(['mean', 'std', 'count']).reset_index()
                agg_data['Unit_Group_Combo'] = agg_data['Unit'] + ' (' + agg_data['Group_ID'].astype(str) + ')'
            else:
                agg_data = df_plot.groupby(['Unit', grouping_col])[value_col].agg(['mean', 'std', 'count']).reset_index()
                agg_data['Unit_Group_Combo'] = agg_data['Unit']

            # Filtering groups
            if groups_to_display is not None:
                agg_data = agg_data[agg_data[grouping_col].isin(groups_to_display)]

            # Create matrix for plotting
            raw_matrix = agg_data.pivot_table(index='Unit_Group_Combo', columns=grouping_col, values='mean', aggfunc='mean')
            raw_matrix = raw_matrix.fillna(0)

            # Sort columns
            if display_name_map:
                current_columns = raw_matrix.columns.tolist()
                ordered_columns = [k for k in display_name_map.keys() if k in current_columns]
                remaining_columns = sorted([g for g in current_columns if g not in display_name_map.keys()])
                raw_matrix = raw_matrix[ordered_columns + remaining_columns]
            else:
                raw_matrix = raw_matrix.sort_index(axis=1)

            # Sort rows based on unit_order
            if unit_order is not None:
                all_rows = raw_matrix.index.tolist()
                ordered_rows = []
                for unit in unit_order:
                    matching_rows = [r for r in all_rows if r.startswith(unit + ' (') or r == unit]
                    ordered_rows.extend(sorted(matching_rows))
                raw_matrix = raw_matrix.reindex(ordered_rows).dropna(how='all')
            elif has_Group_ID:
                all_rows = raw_matrix.index.tolist()
                unit_group_map = {}
                for row in all_rows:
                    base_unit = row.split(' (')[0] if ' (' in row else row
                    unit_group_map.setdefault(base_unit, []).append(row)
                ordered_rows = []
                for bu in sorted(unit_group_map.keys()):
                    if len(unit_group_map[bu]) > 1: ordered_rows.extend(sorted(unit_group_map[bu]))
                for bu in sorted(unit_group_map.keys()):
                    if len(unit_group_map[bu]) == 1: ordered_rows.extend(unit_group_map[bu])
                raw_matrix = raw_matrix.reindex(ordered_rows)

            # Labels
            col_labels = [self._apply_display_name(c, display_name_map) for c in raw_matrix.columns]
            row_labels = []
            for row in raw_matrix.index:
                if ' (' in row:
                    unit_part, Group_ID_part = row.split(' (', 1)
                    unit_disp = unit_display_map[unit_part] if unit_display_map and unit_part in unit_display_map else unit_part
                    Group_ID_full = Group_ID_part.rstrip(')')
                    row_labels.append(f"{unit_disp} ({Group_ID_full})" if Group_ID_full not in ['O', 'Other', ''] else f"{unit_disp}")
                else:
                    row_labels.append(unit_display_map[row] if unit_display_map and row in unit_display_map else row.replace('_', ' '))

            # Figure sizing
            n_rows, n_cols = raw_matrix.shape
            spacing = 0.5
            fig_width = max(6, 0.7 * n_cols + 2.5)
            fig_height = max(3, min(10, 0.35 * n_rows + 2.5))
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')

            col_index = {col: i * spacing for i, col in enumerate(raw_matrix.columns)}
            row_index = {row: i * spacing for i, row in enumerate(raw_matrix.index)}
            xs, ys, sizes, color_vals = [], [], [], []
            xs_black, ys_black, sizes_black = [], [], []

            global_mean = agg_data['mean'].mean()
            if vmin is None: vmin = agg_data['mean'].min()
            if vmax is None: vmax = agg_data['mean'].max()
            vcenter = global_mean

            min_bubble_size = 50

            # 5. Loop and calculate significance on RAW data
            for row_name in raw_matrix.index:
                for col_name in raw_matrix.columns:
                    val = raw_matrix.loc[row_name, col_name]
                    x_val, y_val = col_index[col_name], row_index[row_name]
                    
                    if pd.isna(val) or val == 0:
                        xs_black.append(x_val); ys_black.append(y_val); sizes_black.append(min_bubble_size)
                        continue
                        
                    matching_agg = agg_data[(agg_data['Unit_Group_Combo'] == row_name) & (agg_data[grouping_col] == col_name)]
                    if matching_agg.empty:
                        xs_black.append(x_val); ys_black.append(y_val); sizes_black.append(min_bubble_size); continue
                    
                    unit_nm = matching_agg.iloc[0]['Unit']
                    reg_nm = matching_agg.iloc[0]['Group_ID'] if has_Group_ID else None
                    
                    # 1-vs-rest T-test on RAW data
                    if has_Group_ID and reg_nm:
                        mask = (df_stats['Unit'] == unit_nm) & (df_stats['Group_ID'] == reg_nm) & (df_stats[grouping_col] == col_name)
                        other = (~((df_stats['Unit'] == unit_nm) & (df_stats['Group_ID'] == reg_nm))) & (df_stats[grouping_col] == col_name)
                    else:
                        mask = (df_stats['Unit'] == unit_nm) & (df_stats[grouping_col] == col_name)
                        other = (df_stats['Unit'] != unit_nm) & (df_stats[grouping_col] == col_name)
                    
                    this_v = df_stats[mask][value_col].values
                    other_v = df_stats[other][value_col].values
                    
                    p_val = 1.0
                    if len(this_v) >= 2 and len(other_v) >= 2:
                        _, p_val = ttest_ind(this_v, other_v, equal_var=True)
                        if pd.isna(p_val): p_val = 1.0
                    
                    if p_val >= 0.05: size = min_bubble_size
                    elif p_val >= 0.01: size = 125
                    elif p_val >= 0.001: size = 200
                    else: size = 250
                    
                    xs.append(x_val); ys.append(y_val); sizes.append(size); color_vals.append(val)

            # Normalization and Plotting
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax) if vmin < vcenter < vmax else mcolors.Normalize(vmin=vmin, vmax=vmax)
            if xs:
                ax.scatter(xs, ys, s=sizes, c=color_vals, cmap=cmap, norm=norm, edgecolors='none', zorder=3)
            if xs_black:
                ax.scatter(xs_black, ys_black, s=sizes_black, c='black', edgecolors='none', zorder=3)

            # Colorbar
            import matplotlib.cm as cm
            cbar_fig, cbar_ax = plt.subplots(figsize=(1.2, max(3, n_rows * 0.35)))
            cbar = cbar_fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, orientation='vertical')
            cbar.set_label('Mean Ratio', color='black', fontsize=14)
            cbar_fig.savefig(os.path.join(output_dir, "bubble_chart_raw_ratios_colorbar.png"), dpi=600, bbox_inches='tight', facecolor='white')
            plt.close(cbar_fig)

            # Formatting
            if has_Group_ID:
                prev_u = None
                for ridx, row_nm in enumerate(raw_matrix.index):
                    curr_u = row_nm.split(' (')[0] if ' (' in row_nm else row_nm
                    if prev_u is not None and curr_u != prev_u:
                        ax.axhline(y=ridx * spacing - spacing * 0.5, color='#aaaaaa', linewidth=1.0, zorder=2)
                    prev_u = curr_u

            ax.set_xticks([i * spacing for i in range(n_cols)])
            ax.set_yticks([i * spacing for i in range(n_rows)])
            ax.grid(True, color='#dddddd', linewidth=0.5, zorder=1)
            ax.xaxis.set_label_position('top'); ax.xaxis.tick_top()
            ax.set_xticklabels(col_labels, rotation=45, ha='left', fontsize=14, color='black')
            ax.set_yticklabels(row_labels, rotation=0, ha='right', fontsize=14, color='black')
            ax.set_ylabel('Unit (Group_ID)' if has_Group_ID else 'Unit', fontweight='bold', fontsize=16)
            ax.set_title('Raw Ratio Distribution', fontweight='bold', fontsize=18, pad=20)
            
            for spine in ax.spines.values(): spine.set_edgecolor('#cccccc')
            ax.set_xlim(-spacing * 0.5, (n_cols - 1) * spacing + spacing * 0.5)
            ax.set_ylim(-spacing * 0.5, (n_rows - 1) * spacing + spacing * 0.5)

            # Significance Legend
            legend_fig, legend_ax = plt.subplots(figsize=(2.5, 1.2))
            sig_handles = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=5, label='p ≥ 0.05', markeredgecolor='black', linewidth=0.5),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='p < 0.05', markeredgecolor='black', linewidth=0.5),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='p < 0.01', markeredgecolor='black', linewidth=0.5),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=12, label='p < 0.001', markeredgecolor='black', linewidth=0.5),
            ]
            legend_ax.legend(handles=sig_handles, loc='center', frameon=True, fontsize=11, title='Significance', ncol=2)
            legend_ax.axis('off')
            legend_fig.savefig(os.path.join(output_dir, "bubble_significance_legend_raw.png"), dpi=600, bbox_inches='tight', facecolor='white')
            plt.close(legend_fig)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "bubble_chart_raw_ratios.png"), dpi=600, bbox_inches='tight', facecolor='white')
            
            if show_plots: plt.show()
            else: plt.close()
            
            return {'matrix': raw_matrix, 'raw_data': agg_data}
        except Exception as e:
            print(f"Error creating raw ratio bubble chart: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}

    def plot_subgroup_gradient_heatmap(self, df, value_col, grouping_col, subgroup_order,
                                       output_dir=None, display_name_map=None,
                                       unit_display_map=None, unit_order=None,
                                       units_to_display=None, cmap='RdBu_r',
                                       min_size_to_plot=10, remove_outliers=True,
                                       show_values=True, cmap_scale=None):
        """
        Create a spatial gradient heatmap of ratios/molecules across ordered subgroups per unit.
        Also runs 1-Way ANOVA to evaluate regional statistical variance.

        Parameters
        ----------
        df : pandas.DataFrame
        value_col : str
            The data column name ('Percentage' or 'Mean_Ratio')
        grouping_col : str
            The metric column name ('Molecule' or 'Ratio_Type')
        subgroup_order : list
            Ordered spatial segments (e.g. ['P', 'M', 'CM', 'C3', 'C2', 'C1'])
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import os
        from scipy.stats import f_oneway

        if output_dir is None:
            output_dir = 'subgroup_gradient_heatmaps'
        os.makedirs(output_dir, exist_ok=True)

        # 1. Filter to selected units if specified
        if units_to_display is not None:
            df = df[df['Unit'].isin(units_to_display)].copy()

        # 2. Remove NaN and infinite values
        clean_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[value_col])
        
        if 'Group_ID' not in clean_df.columns:
            print("  Error: 'Group_ID' column required to establish subgroup cohorts.")
            return

        # Filter strictly to targeted subgroups
        clean_df = clean_df[clean_df['Group_ID'].isin(subgroup_order)].copy()

        unique_items = clean_df[grouping_col].dropna().unique()

        for item in unique_items:
            item_data = clean_df[clean_df[grouping_col] == item].copy()
            
            # Outlier Filtration
            if remove_outliers:
                item_data = self._apply_outlier_filtration(item_data, value_col, ['Unit', 'Group_ID'])
            
            if item_data.empty:
                continue

            # Clean data (NaNs/infs) but keep outliers for stats
            item_data_stats = clean_df[clean_df[grouping_col] == item].copy()
            
            # Outlier Filtration for plotting
            if remove_outliers:
                item_data_plot = self._apply_outlier_filtration(item_data_stats, value_col, ['Unit', 'Group_ID'])
            else:
                item_data_plot = item_data_stats
            
            if item_data_plot.empty:
                continue

            # Build Data Matrix (Plotting)
            agg_data_plot = item_data_plot.groupby(['Unit', 'Group_ID'])[value_col].agg(['mean', 'count']).reset_index()
            agg_data_plot = agg_data_plot[agg_data_plot['count'] >= min_size_to_plot]
            
            if agg_data_plot.empty:
                continue

            # Pivot
            pivot_df = agg_data_plot.pivot(index='Unit', columns='Group_ID', values='mean')
            valid_cols = [sg for sg in subgroup_order if sg in pivot_df.columns]
            pivot_df = pivot_df[valid_cols]
            
            if pivot_df.empty:
                continue

            # Reindex Rows
            if unit_order is not None:
                ordered_units = [u for u in unit_order if u in pivot_df.index]
                remaining_units = [u for u in pivot_df.index if u not in ordered_units]
                pivot_df = pivot_df.reindex(ordered_units + remaining_units)
            elif unit_display_map is not None:
                ordered_units = [u for u in unit_display_map.keys() if u in pivot_df.index]
                remaining_units = [u for u in pivot_df.index if u not in ordered_units]
                pivot_df = pivot_df.reindex(ordered_units + remaining_units)
            else:
                pivot_df = pivot_df.sort_index()

            # Statistical variations across subgroups (on RAW data)
            anova_results = {}
            spearman_results = {}
            rank_map = {sg: i for i, sg in enumerate(subgroup_order)}
            
            for unit in pivot_df.index:
                x_ranks = []
                y_means_raw = []
                
                for sg in valid_cols:
                    sg_vals_raw = item_data_stats[(item_data_stats['Unit'] == unit) & (item_data_stats['Group_ID'] == sg)][value_col].values
                    if len(sg_vals_raw) >= 2:
                        x_ranks.append(rank_map[sg])
                        y_means_raw.append(np.mean(sg_vals_raw))
                
                # 1. Welch's ANOVA (using raw data)
                unit_subset = item_data_stats[item_data_stats['Unit'] == unit]
                if unit_subset['Group_ID'].nunique() >= 2:
                    try:
                        import pingouin as pg
                        aov = pg.welch_anova(data=unit_subset, dv=value_col, between='Group_ID')
                        p_val = aov['p-unc'].iloc[0] if 'p-unc' in aov.columns else aov['p_unc'].iloc[0]
                        anova_results[unit] = p_val
                    except Exception:
                        anova_results[unit] = np.nan
                
                # 2. Spearman Correlation
                if len(x_ranks) >= 3:
                    from scipy.stats import spearmanr
                    try:
                        corr, s_p = spearmanr(x_ranks, y_means_raw)
                        spearman_results[unit] = (corr, s_p)
                    except Exception:
                        spearman_results[unit] = (np.nan, np.nan)

            # Rename indices and columns for display
            display_rows = [unit_display_map.get(row, row) if unit_display_map else row for row in pivot_df.index]
            right_side_labels = []
            for row in pivot_df.index:
                right_label = ""
                p_val = anova_results.get(row, 1.0)
                if p_val < 0.001: right_label += "***"
                elif p_val < 0.01: right_label += "**"
                elif p_val < 0.05: right_label += "*"
                
                if row in spearman_results:
                    corr, s_p = spearman_results[row]
                    if not np.isnan(corr):
                        if right_label: right_label += " "
                        right_label += f"(ρ={corr:.2f})"
                right_side_labels.append(right_label)

            # Plotting
            fig_width = max(5, len(valid_cols) * 1.2 + 2)
            fig_height = max(4, len(pivot_df) * 0.4 + 2)
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            display_item = display_name_map.get(item, item) if display_name_map else item

            # Limits
            vmin_p, vmax_p = pivot_df.min().min(), pivot_df.max().max()
            if cmap_scale and item in cmap_scale:
                s = cmap_scale[item]
                if isinstance(s, (list, tuple)) and len(s) == 2: vmin_p, vmax_p = s

            sns.heatmap(pivot_df, annot=show_values, fmt=".3f", cmap=cmap, cbar=False, linewidths=0.5, ax=ax,
                        xticklabels=valid_cols, yticklabels=display_rows, vmin=vmin_p, vmax=vmax_p)
            
            # Right Y-axis
            ax_r = ax.twinx()
            ax_r.set_ylim(ax.get_ylim()); ax_r.set_yticks(ax.get_yticks())
            ax_r.set_yticklabels(right_side_labels, rotation=0, ha='left', fontsize=10, color='black')
            for s in ['right', 'top', 'left', 'bottom']: ax_r.spines[s].set_visible(False)
            ax_r.tick_params(axis='y', which='both', length=0)
            
            ax.set_title(f"{display_item}", fontweight='bold', fontsize=14, pad=15)
            ax.set_xlabel("Subgroup", fontweight='bold', fontsize=12)
            ax.set_ylabel("Unit", fontweight='bold', fontsize=12)
            plt.tight_layout()
            
            item_safe = str(item).replace(':', '').replace('/', '_').replace(' ', '_')
            plt.savefig(os.path.join(output_dir, f"subgroup_gradient_{item_safe}.png"), dpi=600, bbox_inches='tight', facecolor='white')
            plt.close()

            # Standalone colorbar
            import matplotlib.cm as cm
            import matplotlib.colors as mcolors
            cb_fig, cb_ax = plt.subplots(figsize=(1.2, max(3, len(pivot_df) * 0.35)))
            norm = mcolors.Normalize(vmin=vmin_p, vmax=vmax_p)
            cbar = cb_fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cb_ax, orientation='vertical')
            cbar.set_label('Relative Prob. %' if value_col == 'Percentage' else 'Mean Ratio', color='black', fontsize=12)
            cb_fig.tight_layout()
            cb_fig.savefig(os.path.join(output_dir, f"subgroup_gradient_{item_safe}_colorbar.png"), dpi=600, bbox_inches='tight', facecolor='white')
            plt.close(cb_fig)

            print(f"  Generated subgroup gradient heatmap for {item}: {output_dir}")
            print(f"  Generated colorbar: {output_dir}")

    def plot_subgroup_gradient_heatmap_wide(self, wide_df, value_cols, subgroup_order,
                                            output_dir=None, display_name_map=None,
                                            unit_display_map=None, unit_order=None,
                                            units_to_display=None, cmap='RdBu_r',
                                            min_size_to_plot=10, remove_outliers=True,
                                            show_values=True, cmap_scale=None,
                                            cbar_label='Mean Ratio', display_significance=False):
        """
        Create a spatial gradient heatmap of ratios/molecules across ordered subgroups per unit,
        taking a wide format DataFrame where metrics are separate columns.
        Also runs 1-Way ANOVA to evaluate regional statistical variance.

        Parameters
        ----------
        wide_df : pandas.DataFrame
            Pivoted DataFrame containing metrics as columns.
        value_cols : str or list of str
            The column names in wide_df to plot (e.g. ['Lipid_to_Protein', 'Lipid_Unsaturation']).
        subgroup_order : list
            Ordered spatial segments (e.g. ['P', 'M', 'CM', 'C3', 'C2', 'C1'])
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import os
        from scipy.stats import f_oneway

        if output_dir is None:
            output_dir = 'subgroup_gradient_heatmaps'
        os.makedirs(output_dir, exist_ok=True)

        if isinstance(value_cols, str):
            value_cols = [value_cols]

        # 1. Filter to selected units if specified
        if units_to_display is not None:
            wide_df = wide_df[wide_df['Unit'].isin(units_to_display)].copy()

        if 'Group_ID' not in wide_df.columns:
            print("  Error: 'Group_ID' column required to establish subgroup cohorts.")
            return

        # Filter strictly to targeted subgroups
        wide_df = wide_df[wide_df['Group_ID'].isin(subgroup_order)].copy()

        for item in value_cols:
            if item not in wide_df.columns:
                print(f"  Warning: Column '{item}' not found in DataFrame.")
                continue

            # Clean data (NaNs/infs) but keep outliers for stats
            item_data_stats = wide_df[['Unit', 'Group_ID', item]].replace([np.inf, -np.inf], np.nan).dropna(subset=[item]).copy()
            
            # Outlier Filtration for plotting
            if remove_outliers:
                item_data_plot = self._apply_outlier_filtration(item_data_stats, item, ['Unit', 'Group_ID'])
            else:
                item_data_plot = item_data_stats

            if item_data_plot.empty:
                continue

            # Build Data Matrix (Plotting)
            agg_data_plot = item_data_plot.groupby(['Unit', 'Group_ID'])[item].agg(['mean', 'count']).reset_index()
            agg_data_plot = agg_data_plot[agg_data_plot['count'] >= min_size_to_plot]
            
            if agg_data_plot.empty:
                continue

            # Pivot
            pivot_df = agg_data_plot.pivot(index='Unit', columns='Group_ID', values='mean')
            valid_cols = [sg for sg in subgroup_order if sg in pivot_df.columns]
            pivot_df = pivot_df[valid_cols]
            
            if pivot_df.empty:
                continue

            # Reindex Rows
            if unit_order is not None:
                ordered_units = [u for u in unit_order if u in pivot_df.index]
                remaining_units = [u for u in pivot_df.index if u not in ordered_units]
                pivot_df = pivot_df.reindex(ordered_units + remaining_units)
            elif unit_display_map is not None:
                ordered_units = [u for u in unit_display_map.keys() if u in pivot_df.index]
                remaining_units = [u for u in pivot_df.index if u not in ordered_units]
                pivot_df = pivot_df.reindex(ordered_units + remaining_units)
            else:
                pivot_df = pivot_df.sort_index()

            # Save counts (using filtered data)
            pivot_counts = agg_data_plot.pivot(index='Unit', columns='Group_ID', values='count')
            pivot_counts = pivot_counts.reindex(index=pivot_df.index, columns=pivot_df.columns)
            item_safe = str(item).replace(':', '').replace('/', '_').replace(' ', '_')
            pivot_counts.to_csv(os.path.join(output_dir, f"subgroup_gradient_{item_safe}_counts.csv"))
            if display_significance:
                # Statistical variations across subgroups (on RAW data)
                anova_results = {}
                spearman_results = {}
                rank_map = {sg: i for i, sg in enumerate(subgroup_order)}

                for unit in pivot_df.index:
                    x_ranks = []
                    y_means_raw = []

                    for sg in valid_cols:
                        sg_vals_raw = item_data_stats[(item_data_stats['Unit'] == unit) & (item_data_stats['Group_ID'] == sg)][item].values
                        if len(sg_vals_raw) >= 2:
                            x_ranks.append(rank_map[sg])
                            y_means_raw.append(np.mean(sg_vals_raw))

                    # 1. Welch's ANOVA (using raw data)
                    unit_subset = item_data_stats[item_data_stats['Unit'] == unit]
                    if unit_subset['Group_ID'].nunique() >= 2:
                        try:
                            import pingouin as pg
                            aov = pg.welch_anova(data=unit_subset, dv=item, between='Group_ID')
                            p_val = aov['p-unc'].iloc[0] if 'p-unc' in aov.columns else aov['p_unc'].iloc[0]
                            anova_results[unit] = p_val
                        except Exception:
                            anova_results[unit] = np.nan

                    # 2. Spearman Correlation
                    if len(x_ranks) >= 3:
                        from scipy.stats import spearmanr
                        try:
                            corr, s_p = spearmanr(x_ranks, y_means_raw)
                            spearman_results[unit] = (corr, s_p)
                        except Exception:
                            spearman_results[unit] = (np.nan, np.nan)

                right_side_labels = []
                for row in pivot_df.index:
                    right_label = ""
                    p_val = anova_results.get(row, 1.0)
                    if p_val < 0.001: right_label += "***"
                    elif p_val < 0.01: right_label += "**"
                    elif p_val < 0.05: right_label += "*"

                    if row in spearman_results:
                        corr, s_p = spearman_results[row]
                        if not np.isnan(corr):
                            if right_label: right_label += " "
                            right_label += f"(ρ={corr:.2f})"
                    right_side_labels.append(right_label)
                            # Labels and display rows
            
            display_rows = [unit_display_map.get(row, row) if unit_display_map else row for row in pivot_df.index]

            

            # Plotting
            fig_height = max(4, len(pivot_df) * 0.4 + 2)
            fig_width = max(fig_height, len(valid_cols) * 1.2 + 2)

            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            display_item = display_name_map.get(item, item) if display_name_map else item

            # Limits
            vmin_p, vmax_p = pivot_df.min().min(), pivot_df.max().max()
            if cmap_scale and item in cmap_scale:
                s = cmap_scale[item]
                if isinstance(s, (list, tuple)) and len(s) == 2: vmin_p, vmax_p = s

            sns.heatmap(pivot_df, annot=show_values, fmt=".3f", cmap=cmap, cbar=False, linewidths=0.5, ax=ax,
                        xticklabels=valid_cols, yticklabels=display_rows, vmin=vmin_p, vmax=vmax_p)

            # Right Y-axis
            if display_significance:
                ax_r = ax.twinx()
                ax_r.set_ylim(ax.get_ylim()); ax_r.set_yticks(ax.get_yticks())
                ax_r.set_yticklabels(right_side_labels, rotation=0, ha='left', fontsize=10, color='black')
                for s in ['right', 'top', 'left', 'bottom']: ax_r.spines[s].set_visible(False)
                ax_r.tick_params(axis='y', which='both', length=0)

            ax.set_title(f"{display_item}", fontweight='bold', fontsize=14, pad=15)
            ax.set_xlabel("Subgroup", fontweight='bold', fontsize=12)
            ax.set_ylabel("Unit", fontweight='bold', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"subgroup_gradient_{item_safe}.png"), dpi=600, bbox_inches='tight', facecolor='white')
            plt.close()

            # Colorbar
            import matplotlib.cm as cm
            import matplotlib.colors as mcolors
            cb_fig, cb_ax = plt.subplots(figsize=(1.2, max(3, len(pivot_df) * 0.35)))
            norm = mcolors.Normalize(vmin=vmin_p, vmax=vmax_p)
            cbar = cb_fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cb_ax, orientation='vertical')
            cbar.set_label(cbar_label, color='black', fontsize=12)
            cb_fig.tight_layout()
            cb_fig.savefig(os.path.join(output_dir, f"subgroup_gradient_{item_safe}_colorbar.png"), dpi=600, bbox_inches='tight', facecolor='white')
            plt.close(cb_fig)

            print(f"  Generated subgroup gradient heatmap for {item}: {output_dir}")
            print(f"  Generated colorbar: {output_dir}")

    def plot_subgroup_gradient_heatmap_wide_transposed(self, wide_df, value_cols, subgroup_order,
                                                       output_dir=None, display_name_map=None,
                                                       unit_display_map=None, unit_order=None,
                                                       units_to_display=None, cmap='RdBu_r',
                                                       min_size_to_plot=10, remove_outliers=True,
                                                       show_values=True, cmap_scale=None,
                                                       cbar_label='Z-score'):
        """
        Create a spatial gradient heatmap of ratios/molecules across ordered subgroups per unit,
        taking a wide format DataFrame where metrics are separate columns.
        
        **Layout**: Subgroups are ROWS (in order), each column is a Unit+Ratio_Type combination.
        Columns are labeled with just the ratio type and visually separated by ratio type.
        
        Also runs 1-Way ANOVA to evaluate regional statistical variance.
        Each heatmap is z-scored independently, and all plots share the same
        color scale based on the maximum absolute z-score across items.

        Parameters
        ----------
        wide_df : pandas.DataFrame
            Pivoted DataFrame containing metrics as columns.
        value_cols : str or list of str
            The column names in wide_df to plot (e.g. ['Lipid_to_Protein', 'Lipid_Unsaturation']).
        subgroup_order : list
            Ordered spatial segments (e.g. ['P', 'M', 'CM', 'C3', 'C2', 'C1']) that will be rows.
        output_dir : str
            Directory to save plots.
        display_name_map : dict, optional
            Mapping of ratio type names to display names.
        unit_display_map : dict, optional
            Mapping of unit names to display names.
        unit_order : list, optional
            Custom order for units. If None, uses sorted order.
        units_to_display : list, optional
            List of unit names to include; all others are excluded.
        cmap : str
            Colormap for heatmap (default 'RdBu_r').
        min_size_to_plot : int
            Minimum number of valid data points required per cell.
        remove_outliers : bool
            Whether to remove outliers before plotting.
        show_values : bool
            Whether to annotate cells with values.
        cmap_scale : dict, optional
            Per-item custom vmin/vmax. Example: {'Ratio_A': [-1, 1], 'Ratio_B': [0, 2]}
        cbar_label : str
            Label for the colorbar (default 'Mean Ratio').
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import os
        import pandas as pd
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        from scipy.stats import f_oneway
        from scipy.stats import spearmanr

        if output_dir is None:
            output_dir = 'subgroup_gradient_heatmaps_transposed'
        os.makedirs(output_dir, exist_ok=True)

        if isinstance(value_cols, str):
            value_cols = [value_cols]

        # 1. Filter to selected units if specified
        if units_to_display is not None:
            wide_df = wide_df[wide_df['Unit'].isin(units_to_display)].copy()

        if 'Group_ID' not in wide_df.columns:
            print("  Error: 'Group_ID' column required to establish subgroup cohorts.")
            return

        # Filter strictly to targeted subgroups
        wide_df = wide_df[wide_df['Group_ID'].isin(subgroup_order)].copy()

        # 2. Determine unit order
        all_units = sorted(wide_df['Unit'].unique())
        if unit_order is not None:
            ordered_units = [u for u in unit_order if u in all_units]
            remaining_units = [u for u in all_units if u not in ordered_units]
            all_units = ordered_units + remaining_units

        plot_records = []

        for item in value_cols:
            if item not in wide_df.columns:
                print(f"  Warning: Column '{item}' not found in DataFrame.")
                continue

            # Clean data (NaNs/infs) but keep outliers for stats
            item_data_stats = wide_df[['Unit', 'Group_ID', item]].replace([np.inf, -np.inf], np.nan).dropna(subset=[item]).copy()

            # Outlier Filtration for plotting
            if remove_outliers:
                item_data_plot = self._apply_outlier_filtration(item_data_stats, item, ['Unit', 'Group_ID'])
            else:
                item_data_plot = item_data_stats

            # Build Data Matrix (Plotting): subgroups as rows, units as columns
            agg_data_plot = item_data_plot.groupby(['Group_ID', 'Unit'])[item].agg(['mean', 'count']).reset_index()
            agg_data_plot = agg_data_plot[agg_data_plot['count'] >= min_size_to_plot]

            if agg_data_plot.empty:
                continue

            pivot_df = agg_data_plot.pivot(index='Group_ID', columns='Unit', values='mean')
            valid_subgroups = [sg for sg in subgroup_order if sg in pivot_df.index]
            pivot_df = pivot_df.reindex(valid_subgroups)
            plot_units = [u for u in all_units if u in pivot_df.columns]
            pivot_df = pivot_df[plot_units]

            if pivot_df.empty or pivot_df.isnull().all().all():
                print(f"  Skipping {item}: no valid data after filtering.")
                continue

            display_item = display_name_map.get(item, item) if display_name_map else item
            item_safe = str(item).replace(':', '').replace('/', '_').replace(' ', '_')

            z_df = pivot_df.copy().astype(float)
            z_values = z_df.to_numpy(dtype=float)
            z_mean = np.nanmean(z_values)
            z_std = np.nanstd(z_values)
            if np.isnan(z_std) or z_std == 0:
                z_df = z_df - z_mean
            else:
                z_df = (z_df - z_mean) / z_std

            item_abs_max = np.nanmax(np.abs(z_df.to_numpy(dtype=float)))
            if np.isnan(item_abs_max):
                item_abs_max = 0

            # Save counts
            pivot_counts = agg_data_plot.pivot(index='Group_ID', columns='Unit', values='count')
            pivot_counts = pivot_counts.reindex(index=valid_subgroups, columns=plot_units)
            counts_csv_path = os.path.join(output_dir, f"subgroup_gradient_transposed_{item_safe}_counts.csv")
            pivot_counts.to_csv(counts_csv_path)

            # Statistical variations across subgroups (on RAW data)
            stats_rows = []
            rank_map = {sg: i for i, sg in enumerate(valid_subgroups)}

            for unit in plot_units:
                x_ranks = []
                y_means_raw = []

                for sg in valid_subgroups:
                    sg_vals_raw = item_data_stats[(item_data_stats['Unit'] == unit) & (item_data_stats['Group_ID'] == sg)][item].values
                    if len(sg_vals_raw) >= 2:
                        x_ranks.append(rank_map[sg])
                        y_means_raw.append(np.mean(sg_vals_raw))

                welch_p = np.nan
                unit_subset = item_data_stats[item_data_stats['Unit'] == unit]
                if unit_subset['Group_ID'].nunique() >= 2:
                    try:
                        import pingouin as pg
                        aov = pg.welch_anova(data=unit_subset, dv=item, between='Group_ID')
                        welch_p = aov['p-unc'].iloc[0] if 'p-unc' in aov.columns else aov['p_unc'].iloc[0]
                    except Exception:
                        welch_p = np.nan

                spearman_rho = np.nan
                spearman_p = np.nan
                if len(x_ranks) >= 3:
                    try:
                        corr, s_p = spearmanr(x_ranks, y_means_raw)
                        spearman_rho, spearman_p = corr, s_p
                    except Exception:
                        spearman_rho, spearman_p = np.nan, np.nan

                stats_rows.append({
                    'Unit': unit,
                    'Welch_ANOVA_p': welch_p,
                    'Spearman_rho': spearman_rho,
                    'Spearman_p': spearman_p,
                    'N_subgroups_used': len(x_ranks)
                })

            stats_df = pd.DataFrame(stats_rows)
            stats_csv_path = os.path.join(output_dir, f"subgroup_gradient_transposed_{item_safe}_stats.csv")
            stats_df.to_csv(stats_csv_path, index=False)
            plot_records.append({
                'item': item,
                'item_safe': item_safe,
                'display_item': display_item,
                'z_df': z_df,
                'display_rows': list(pivot_df.index),
                'display_cols': [f"{idx}" for idx, _ in enumerate(pivot_df.columns, start=1)],
                'n_rows': len(pivot_df.index),
                'item_abs_max': item_abs_max,
                'counts_csv_path': counts_csv_path,
                'stats_csv_path': stats_csv_path,
            })

        if not plot_records:
            print("No valid data to plot.")
            return

        global_abs_max = max(record['item_abs_max'] for record in plot_records)
        if np.isnan(global_abs_max) or global_abs_max == 0:
            global_abs_max = 1.0
        vmin_p, vmax_p = -global_abs_max, global_abs_max

        for record in plot_records:
            item = record['item']
            item_safe = record['item_safe']
            display_item = record['display_item']
            z_df = record['z_df']
            display_rows = record['display_rows']
            display_cols = record['display_cols']
            n_rows = record['n_rows']

            n_cols = len(z_df.columns)
            fig_width = max(16, n_cols * 0.6 + 2)
            fig_height = max(fig_width, n_rows * 0.5 + 2)

            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            sns.heatmap(
                z_df,
                annot=show_values,
                fmt=".3f",
                cmap=cmap,
                cbar=False,
                linewidths=1.5,
                linecolor='white',
                ax=ax,
                vmin=vmin_p,
                vmax=vmax_p
            )

            ax.set_xticklabels(display_cols,
                               rotation=0,
                               ha='center',
                               fontsize=fig_width * 2,
                               fontweight='bold')
            ax.set_yticklabels(display_rows, rotation=90, fontsize=fig_height * 2, fontweight='bold')
            ax.set_title(f"{display_item}", fontweight='bold', fontsize=fig_height * 2.5, pad=15)
            ax.set_xlabel("", fontweight='bold', fontsize=12)
            ax.set_ylabel("", fontweight='bold', fontsize=12)

            for _, spine in ax.spines.items():
                spine.set_visible(True)
                spine.set_linewidth(5)

            plt.tight_layout()
            png_path = os.path.join(output_dir, f"subgroup_gradient_transposed_{item_safe}.png")
            plt.savefig(png_path, dpi=600, bbox_inches='tight', transparent=True)
            plt.close()

            cb_fig, cb_ax = plt.subplots(figsize=(1.2, max(3, n_rows * 0.4)))
            norm = mcolors.Normalize(vmin=vmin_p, vmax=vmax_p)
            cbar = cb_fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cb_ax, orientation='vertical')
            cbar.set_label(cbar_label, color='black', fontsize=12)
            cb_fig.tight_layout()
            colorbar_path = os.path.join(output_dir, f"subgroup_gradient_transposed_{item_safe}_colorbar.png")
            cb_fig.savefig(colorbar_path, dpi=600, bbox_inches='tight', transparent=True)
            plt.close(cb_fig)

            print(f"  Generated transposed subgroup gradient heatmap for {item}: {png_path}")
            print(f"  Generated colorbar: {colorbar_path}")
            print(f"  Saved counts CSV: {record['counts_csv_path']}")
            print(f"  Saved stats CSV: {record['stats_csv_path']}")
    def _save_standalone_legend(self, handles, title, save_path, ncol=1, fontsize=11, title_fontsize=12):
        """Save a standalone legend figure as a PNG.

        Parameters
        ----------
        handles : list of matplotlib.patches.Patch (or Line2D)
            Legend handle objects with labels already set.
        title : str
            Legend title text.
        save_path : str
            Full file path (should end in .png) where the legend will be saved.
        ncol : int
            Number of columns in the legend.
        fontsize : int
            Font size for legend entry labels.
        title_fontsize : int
            Font size for the legend title.
        """
        import matplotlib.pyplot as plt
        import os

        # Create parent directory if it doesn't exist
        parent_dir = os.path.dirname(save_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        # Use enough columns to keep the legend readable but not too wide
        n = len(handles)
        if ncol == 1 and n > 6:
            ncol = 2

        fig, ax = plt.subplots(figsize=(max(2.5, ncol * 1.8 + 0.5), max(1.2, (n / ncol) * 0.35 + 0.6)))
        legend = ax.legend(
            handles=handles,
            loc='center',
            frameon=True,
            fontsize=fontsize,
            title=title,
            title_fontsize=title_fontsize,
            ncol=ncol,
        )
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.95)
        ax.axis('off')
        fig.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white', transparent=True)
        plt.close(fig)




    def create_individual_barplots(self, df, value_col, grouping_col, items_list=None,
                                    output_dir=None, data_type='percentage',
                                    show_plots=False, display_name_map=None,
                                    unit_color_map=None, unit_display_map=None,
                                    units_to_display=None,
                                    consolidate_Group_IDs=False,
                                    subgroup_list=None,
                                    compare_by="Group_ID",
                                    show_empty_subgroups=False,
                                    show_scatter=False,
                                    plot_type='bar',
                                    sig_style='brackets',
                                    use_patterns=True,
                                    min_size_to_plot=3,
                                    remove_outliers=True,
                                    orientation='vertical'):
        """
        Create individual plots for publication.
        Supports bar, box, or violin plots with optional Group_ID subgroup variants.
        
        Args:
            df: DataFrame containing the measurements.
            value_col: Column name for the values ('Percentage' or 'Mean_Ratio').
            grouping_col: Column name for the items ('Molecule' or 'Ratio_Type').
            items_list: List of molecules/ratios to plot.
            output_dir: Directory to save plots.
            consolidate_Group_IDs: If True, merges subgroups into base Units for cleaner comparison.
            compare_by: Column name or list of column names used to define subgroups (default: "Group_ID").
            plot_type: Plot style to render ('bar', 'box', or 'violin').
            sig_style: Controls how statistical significance is shown on the plot.
            min_size_to_plot: Minimum number of valid data points required to plot a unit (default: 3).
                'brackets'   — (default) traditional bracket lines with * symbols above bars.
                               Can get crowded when there are many significant pairs.
                'stars_only' — compact mode: a small * / ** / *** label is placed directly
                               above each bar in a significant pair.  No crossing lines.
                'export_only'— no on-plot markers at all; results are only written to CSV.
                A CSV of all significant pairs is always saved to output_dir (when set),
                regardless of the chosen style.
            orientation: 'vertical' (default) or 'horizontal'.
        """
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        from scipy.stats import f_oneway, ttest_ind
        import pingouin as pg


        # Layout tuning knobs for individual barplots:
        # - bar_width controls visual thickness of each bar
        # - same_unit_spacing controls spacing between subgroup bars of the same base unit
        # - between_unit_spacing controls spacing between different base units
        bar_width = 0.2
        same_unit_spacing = 0.21
        between_unit_spacing = 0.4 if consolidate_Group_IDs and not plot_type == 'bar' else bar_width if plot_type == 'bar' else 0.7

        orientation = str(orientation).lower().strip()
        if orientation not in {'vertical', 'horizontal'}:
            raise ValueError(f"Unsupported orientation '{orientation}'. Use 'vertical' or 'horizontal'.")

        horizontal = orientation == 'horizontal'

        if output_dir is None:
            output_dir = 'individual_barplots'
        os.makedirs(output_dir, exist_ok=True)

        plot_type = str(plot_type).lower().strip()
        if plot_type not in {'bar', 'box', 'violin'}:
            raise ValueError(f"Unsupported plot_type '{plot_type}'. Use 'bar', 'box', or 'violin'.")
        
        print(f"Units to display: {units_to_display if units_to_display else 'All'}")

        # Filter units if specified
        if units_to_display is not None:
            df = df[df['Unit'].isin(units_to_display)].copy()
        
        # Consolidation check
        if consolidate_Group_IDs:
            df['Plot_Unit'] = df['Unit']
        else:
            # Create Unit_(Subgroup) labels based on compare_by columns
            def make_unit_subgroup_label(row):
                # Handle single string or list of columns
                cols = [compare_by] if isinstance(compare_by, str) else compare_by
                
                label_parts = []
                for col in cols:
                    val = row.get(col)
                    if not pd.isna(val) and val not in [None, '', 'Other', 'O']:
                        label_parts.append(str(val))
                
                if not label_parts:
                    return str(row['Unit'])
                
                # Combine multiple column values if necessary
                subgroup_str = ", ".join(label_parts)
                return f"{row['Unit']} ({subgroup_str})"
            
            df['Plot_Unit'] = df.apply(make_unit_subgroup_label, axis=1)

            # --- IDENTIFY SUBGROUPS FOR DYNAMIC VISUALS ---
            if subgroup_list is None:
                all_groups = set()
                for unit in df['Plot_Unit'].unique():
                    if ' (' in unit:
                        gid = unit.split(' (')[-1].rstrip(')')
                        all_groups.add(gid)
                # Sort alphabetically only when auto-detecting
                subgroup_list = sorted(list(all_groups))
            else:
                # Use provided list as is (maintain order)
                subgroup_list = list(subgroup_list)
            
            visual_type = "patterns" if use_patterns else "shading"
            print(f"Subgroups identified for dynamic {visual_type}: {subgroup_list}")

        is_wide = grouping_col is None
        if is_wide and items_list is None:
            raise Exception("Must have items_list if using wide dataframe version.")

        elif items_list == "All" or items_list is None:
            items_list = sorted(df[grouping_col].unique())

        consolidated_unit_reference = None
        if consolidate_Group_IDs and show_empty_subgroups:
            if units_to_display is not None:
                consolidated_unit_reference = list(units_to_display)
            else:
                consolidated_unit_reference = sorted(df['Unit'].dropna().unique().tolist())


        # Iterate through items
        for item_name in items_list:
            if is_wide:
                current_value_col = item_name
                item_data = df.copy()
            else:
                current_value_col = value_col
                item_data = df[df[grouping_col] == item_name].copy()
        
                # Clean data using the current value column
                item_data.replace([np.inf, -np.inf], np.nan, inplace=True)
                item_data = item_data.dropna(subset=[current_value_col])
            
            if len(item_data) == 0:
                print(f"  Skipping {item_name}: no valid data")
                continue

            if "Ion" in item_name:
                # Normalize ion intensity values between 0 and 100
                item_data[current_value_col] = (item_data[current_value_col] - item_data[current_value_col].min()) / (item_data[current_value_col].max() - item_data[current_value_col].min() + 1e-9)
                tmp_outliers = False  # Don't apply outlier filtration to already normalized ion intensities
            else:
                tmp_outliers = remove_outliers

            # 1. Determine base units to plot
            if units_to_display is not None:
                base_units = units_to_display
            elif consolidated_unit_reference is not None:
                # Use the full unit list so consolidated plots keep spacing for
                # units that are absent from the current item.
                base_units = consolidated_unit_reference
            else:
                # Extract unique base units from the data and sort them
                all_raw_units = item_data['Plot_Unit'].unique()
                base_units = sorted(list(set(pu.split(' (')[0] if ' (' in pu else pu for pu in all_raw_units)))

            # 2. Build final ordered list of plot units
            if consolidate_Group_IDs:
                source_map = set(item_data['Plot_Unit'].unique())
                if show_empty_subgroups:
                    # Keep all base units, including those without data (for spacing consistency)
                    reordered_units = base_units
                else:
                    # Only include units that have data
                    reordered_units = [bu for bu in base_units if bu in source_map]
            else: 
                reordered_units = []
                for bu in base_units:
                    # Check if this base unit exists in data
                    existing_for_base = [pu for pu in item_data['Plot_Unit'].unique() if (pu.split(' (')[0] if ' (' in pu else pu) == bu]
                    
                    # For this base unit, iterate over subgroups in the requested order
                    added_for_this_base = []
                    for sg in subgroup_list:
                        pu = f"{bu} ({sg})"
                        if pu in existing_for_base:
                            reordered_units.append(pu)
                            added_for_this_base.append(pu)
                        elif show_empty_subgroups:
                            reordered_units.append(pu)
                            added_for_this_base.append(pu)
                    
                    # Safety catch: Add any remaining units for this base (e.g. base unit without subgroup)
                    for pu in existing_for_base:
                        if pu not in added_for_this_base:
                            reordered_units.append(pu)
            
            all_plot_units = reordered_units

            
            # Prepare data groups
            bar_means = []
            bar_stderrs = []
            colors = []
            hatches = []
            positions = []
            scatter_vals_per_pos = []  # list of (position, array_of_values) for scatter overlay
            plot_vals_per_pos = []     # list of arrays used to draw box/violin plots

            # Prepare plot arrays
            current_pos = 1.0
            unit_tick_positions = []
            unit_tick_labels = []
            unit_position_map = {}     # Map unit name -> tick position
            empty_positions = []       # Track positions of empty units
            unit_significance = []     # List of (pos1, pos2, sig_symbol)
            subgroup_data_by_base = {} # Map base_unit -> {group_id: (data, position)}
            empty_units = set()        # Track units with no actual data
            prev_base_unit = None
            
            for unit in all_plot_units:
                unit_data = item_data[item_data['Plot_Unit'] == unit]
                
                if (show_empty_subgroups or consolidated_unit_reference is not None) and len(unit_data) == 0:
                    # Add a placeholder with NaN value to ensure the unit appears in the plot
                    # This applies to both empty subgroups and empty consolidated units
                    unit_data = pd.DataFrame({current_value_col: [np.nan], 'Plot_Unit': [unit]})
                    has_data = False
                    empty_units.add(unit)  # Track as empty for stats exclusion
                else:
                    has_data = True

                if unit_data[current_value_col].count() < min_size_to_plot and has_data:
                    continue
                
                base_unit = unit.split(' (')[0] if ' (' in unit else unit

                # Add tighter spacing within a base unit and larger gaps between base units
                if prev_base_unit is not None:
                    if base_unit == prev_base_unit:
                        current_pos += same_unit_spacing
                    else:
                        current_pos += between_unit_spacing
                
                # Track empty position before adding to plots
                if not has_data:
                    empty_positions.append(current_pos)
                
                # Simple unit-based plot
                if remove_outliers and has_data:
                    unit_data_filtered = self._apply_outlier_filtration(unit_data, current_value_col, ['Plot_Unit'])
                else:
                    unit_data_filtered = unit_data
                filtered_vals = unit_data_filtered[current_value_col].values

                if len(filtered_vals) > 0:
                    bar_means.append(np.mean(filtered_vals))
                    bar_stderrs.append(np.std(filtered_vals) / np.sqrt(len(filtered_vals)))
                else:
                    raw_vals = unit_data[current_value_col].values
                    if len(raw_vals) > 0:
                        bar_means.append(np.mean(raw_vals))
                        bar_stderrs.append(np.std(raw_vals) / np.sqrt(len(raw_vals)))
                    else:
                        bar_means.append(np.nan)
                        bar_stderrs.append(0)
                
                # Store individual values for scatter overlay
                vals_for_plot = filtered_vals if len(filtered_vals) > 0 else unit_data[current_value_col].values
                scatter_vals_per_pos.append((current_pos, vals_for_plot))
                plot_vals_per_pos.append(vals_for_plot if len(vals_for_plot) > 0 else np.array([np.nan]))

                positions.append(current_pos)
                # Use unit_color_map if provided, with color variation for Group_ID subgroups
                base_color = unit_color_map.get(base_unit, '#888888') if unit_color_map else '#888888'
                
                # Extract Group_ID if present and apply dynamic visuals
                hatch = None
                if ' (' in unit:
                    # Extract Group_ID (e.g., 'CortexInner' from 'Unit (CortexInner)')
                    group_id = unit.split(' (')[-1].rstrip(')')
                    
                    if use_patterns:
                        # Use hatch patterns instead of color adjustment
                        hatch = self._get_hatch_for_group(group_id, subgroup_list=subgroup_list)
                        color = base_color
                    else:
                        # Apply dynamic color variations (shading)
                        color = self._adjust_color_for_group(base_color, group_id, subgroup_list=subgroup_list)
                else:
                    color = base_color
                
                colors.append(color)
                hatches.append(hatch)
                
                unit_tick_positions.append(current_pos)
                unit_position_map[unit] = current_pos

                # Collect subgroup data for later intra-unit significance testing.
                # Use raw values for stats even if plotting uses filtered values (per user request)
                if not consolidate_Group_IDs and ' (' in unit:
                    group_id = unit.split(' (')[-1].rstrip(')')
                    raw_vals = unit_data[current_value_col].values
                    if base_unit not in subgroup_data_by_base:
                        subgroup_data_by_base[base_unit] = {}
                    subgroup_data_by_base[base_unit][group_id] = (raw_vals, current_pos)


                prev_base_unit = base_unit

                # Map display name for unit
                display_unit = unit
                if unit_display_map:
                    base_unit = unit.split(' (')[0] if ' (' in unit else unit
                    if base_unit in unit_display_map:
                        mapped = unit_display_map[base_unit]
                        display_unit = unit.replace(base_unit, mapped)
                
                
                unit_tick_labels.append(display_unit)

            if not bar_means: continue

            # --- SIGNIFICANCE TESTING ---
            unit_significance = []
            sig_symbols = {0.001: '***', 0.01: '**', 0.05: '*', 1.0: 'ns'}

            # Keep a consistent aspect ratio across bar, box, and violin variants so the
            # rendered plots are directly comparable even when the orientation changes.
            # For vertical plots: 2:1 aspect ratio (taller than wider), so width/height = 0.5
            # For horizontal plots: 2:1 aspect ratio (wider than taller), so width/height = 2.0
            target_aspect_ratio = 1.5 if horizontal else 0.66

            def _match_aspect_ratio(preferred_width, preferred_height):
                preferred_width = max(preferred_width, 6)
                preferred_height = max(preferred_height, 4)

                if preferred_width / preferred_height > target_aspect_ratio:
                    preferred_height = preferred_width / target_aspect_ratio
                else:
                    preferred_width = preferred_height * target_aspect_ratio

                return preferred_width, preferred_height
            
            # ALWAYS perform statistical tests on the RAW data (before outlier removal)
            # while the visual bars/boxes will respect the remove_outliers flag.
            stat_df = item_data

            if consolidate_Group_IDs:
                # 1. INTER-UNIT: Compare all base units across the plot
                unique_units = [u for u in all_plot_units if u in unit_position_map and u not in empty_units]
                # Filter to only units that have at least 2 samples AND non-zero variance (Welch's ANOVA requirement)
                unique_units = [u for u in unique_units if len(stat_df[stat_df['Plot_Unit'] == u]) >= 2 
                                and stat_df[stat_df['Plot_Unit'] == u][current_value_col].std() > 1e-9]

                if len(unique_units) > 1:
                    analysis_df = stat_df[stat_df['Plot_Unit'].isin(unique_units)]
                    try:
                        # 1. Welch's ANOVA
                        aov = pg.welch_anova(data=analysis_df, dv=current_value_col, between='Plot_Unit')
                        if not aov.empty and 'p_unc' in aov.columns and aov['p_unc'].iloc[0] < 0.05:
                            # 2. Games-Howell Post-hoc
                            ph = pg.pairwise_gameshowell(data=analysis_df, dv=current_value_col, between='Plot_Unit')
                            for _, row in ph.iterrows():
                                u1, u2, p = row['A'], row['B'], row['pval']
                                if p < 0.05:
                                    sig = next((s for t, s in sig_symbols.items() if p <= t), 'ns')
                                    unit_significance.append((unit_position_map[u1], unit_position_map[u2], sig))
                    except Exception as e: print(f"    Inter-unit Pingouin testing failed: {e}")
            else:
                # 2. INTRA-UNIT: Compare subgroups within each base unit
                for b_unit, s_dict in subgroup_data_by_base.items():
                    # Filter out empty subgroup variants from this base unit
                    s_dict_filtered = {gid: (data, pos) for gid, (data, pos) in s_dict.items() 
                                      if f"{b_unit} ({gid})" not in empty_units}
                    g_ids = sorted(s_dict_filtered.keys())
                    
                    if len(g_ids) == 2:
                        # Welch's T-test for 2 subgroups
                        d1, p1 = s_dict_filtered[g_ids[0]]
                        d2, p2 = s_dict_filtered[g_ids[1]]
                        _, p = ttest_ind(d1, d2, equal_var=False)
                        if p < 0.05:
                            sig = next((s for t, s in sig_symbols.items() if p <= t), 'ns')
                            unit_significance.append((p1, p2, sig))
                    elif len(g_ids) > 2:
                        # 2b. Intra-unit Games-Howell using Pingouin
                        try:
                            unit_subset = stat_df[stat_df['Unit'] == b_unit]
                            # Filter out groups with zero variance to avoid linalg errors in Games-Howell
                            group_stds = unit_subset.groupby('Group_ID')[current_value_col].std()
                            valid_groups = group_stds[group_stds > 1e-9].index
                            unit_subset = unit_subset[unit_subset['Group_ID'].isin(valid_groups)]
                            aov = pg.welch_anova(data=unit_subset, dv=current_value_col, between='Group_ID')
                            if not aov.empty and 'p_unc' in aov.columns and aov['p_unc'].iloc[0] < 0.05:
                                if unit_subset['Group_ID'].nunique() > 2:
                                    ph = pg.pairwise_gameshowell(data=unit_subset, dv=current_value_col, between='Group_ID')
                                    for _, row in ph.iterrows():
                                        gid1, gid2, p = row['A'], row['B'], row['pval']
                                        if p < 0.05:
                                            sig = next((s for t, s in sig_symbols.items() if p <= t), 'ns')
                                            unit_significance.append((s_dict[gid1][1], s_dict[gid2][1], sig))
                        except Exception as e: print(f"    Intra-unit Pingouin testing failed for {b_unit}: {e}")


            # Plotting
            span_extent = positions[-1] + bar_width / 2
            value_extent = max([abs(v) for v in bar_means if np.isfinite(v)] + [1.0])
            if horizontal:
                fig_width = max(6, 4.0 + value_extent * 0.08, 2.5 + span_extent * 0.8)
                fig_height = max(4, 2.5 + span_extent * 0.8)
            else:
                fig_width = max(4, 2.5 + span_extent * 0.8)
                fig_height = max(6, 4.0 + value_extent * 0.08)

            fig_width, fig_height = _match_aspect_ratio(fig_width, fig_height)

            fig, ax = plt.subplots(figsize=(fig_width, fig_height))

            if plot_type == 'bar':
                if horizontal:
                    bars = ax.barh(positions, bar_means, height=bar_width, xerr=bar_stderrs,
                                error_kw=dict(linestyle='-', elinewidth=1.5, capsize=5, capthick=1.5),
                                color=colors, alpha=0.8, edgecolor='black', linewidth=2)
                else:
                    bars = ax.bar(positions, bar_means, width=bar_width, yerr=bar_stderrs,
                                error_kw=dict(linestyle='-', elinewidth=1.5, capsize=5, capthick=1.5),
                                color=colors, alpha=0.8, edgecolor='black', linewidth=2)

                # Apply hatches to individual bars
                for bar, h in zip(bars, hatches):
                    if h:
                        bar.set_hatch(h)
            elif plot_type == 'box':
                box = ax.boxplot(
                    plot_vals_per_pos,
                    positions=positions,
                    widths=bar_width * 0.9,
                    patch_artist=True,
                    showfliers=False,
                    vert=not horizontal,
                )
                for patch, color, h in zip(box['boxes'], colors, hatches):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.8)
                    patch.set_edgecolor('black')
                    patch.set_linewidth(1.5)
                    if h:
                        patch.set_hatch(h)
                for key in ['whiskers', 'caps', 'medians']:
                    for artist in box[key]:
                        artist.set_color('black')
                        artist.set_linewidth(1.2)
            else:  # violin
                # violinplot requires ≥2 data points per group (needs KDE).
                violin_vals, violin_positions, violin_colors, violin_hatches = [], [], [], []
                single_vals, single_positions, single_colors = [], [], []
                for vv, vp, vc, vh in zip(plot_vals_per_pos, positions, colors, hatches):
                    arr = np.asarray(vv)
                    # Keep only finite values
                    clean = arr[np.isfinite(arr)] if arr.size > 0 else arr
                    # Require at least two distinct finite values for KDE (violin);
                    # otherwise treat as a single-point group (render as a marker)
                    if clean.size >= 2 and np.unique(clean).size >= 2:
                        violin_vals.append(clean)
                        violin_positions.append(vp)
                        violin_colors.append(vc)
                        violin_hatches.append(vh)
                    elif clean.size == 1:
                        single_vals.append(clean[0])
                        single_positions.append(vp)
                        single_colors.append(vc)

                if violin_vals:
                    violin = ax.violinplot(
                        violin_vals,
                        positions=violin_positions,
                        widths=bar_width * 0.95,
                        showmeans=False,
                        showmedians=True,
                        showextrema=False,
                        vert=not horizontal,
                    )
                    for body, color, h in zip(violin['bodies'], violin_colors, violin_hatches):
                        body.set_facecolor(color)
                        body.set_alpha(0.7)
                        body.set_edgecolor('black')
                        body.set_linewidth(1.2)
                        if h:
                            body.set_hatch(h)
                    for key in ['cbars', 'cmins', 'cmaxes', 'cmedians']:
                        if key in violin:
                            violin[key].set_color('black')
                            violin[key].set_linewidth(1.0)

                # Single-point groups: render as a prominent scatter marker
                if single_vals:
                    ax.scatter(
                        single_positions, single_vals,
                        color=single_colors, edgecolors='black',
                        linewidths=1.0, s=2, zorder=5, marker='D',
                        label='_single_point'
                    )

            # --- Scatter overlay: individual data points jittered over each bar ---
            if show_scatter:
                rng = np.random.default_rng(seed=42)
                scatter_jitter = bar_width * 0.30  # jitter within ±30% of half bar width
                for (s_pos, s_vals), s_color in zip(scatter_vals_per_pos, colors):
                    if len(s_vals) == 0:
                        continue
                    elif len(s_vals) == 1:
                        # Single point: plot without jitter
                        if horizontal:
                            ax.scatter(
                                s_vals[0], s_pos,
                                color=s_color, edgecolors='black',
                                linewidths=1.0, s=60, zorder=2, marker='D',
                                label='_single_point_scatter'
                            )
                        else:
                            ax.scatter(
                                s_pos, s_vals[0],
                                color=s_color, edgecolors='black',
                                linewidths=1.0, s=60, zorder=2, marker='D',
                                label='_single_point_scatter'
                            )
                        continue
                    elif len(s_vals) > 100:
                        # Plot random subset of points if too many to avoid overplotting
                        s_rand_vals = rng.choice(s_vals, size=100, replace=False)
                        jitter = rng.uniform(-scatter_jitter, scatter_jitter, size=100)
                        if horizontal:
                            ax.scatter(
                                s_rand_vals,
                                s_pos + jitter,
                                color=s_color,
                                edgecolors='black',
                                linewidths=0.5,
                                s=2,
                                alpha=0.5,
                                zorder=4,
                            )
                        else:
                            ax.scatter(
                                s_pos + jitter,
                                s_rand_vals,
                                color=s_color,
                                edgecolors='black',
                                linewidths=0.5,
                                s=2,
                                alpha=0.5,
                                zorder=4,
                            )
                    else:
                        jitter = rng.uniform(-scatter_jitter, scatter_jitter, size=len(s_vals))
                        if horizontal:
                            ax.scatter(
                                s_vals,
                                s_pos + jitter,
                                color=s_color,
                                edgecolors='black',
                                linewidths=0.5,
                                s=2,
                                alpha=0.5,
                                zorder=4,
                            )
                        else:
                            ax.scatter(
                                s_pos + jitter,
                                s_vals,
                                color=s_color,
                                edgecolors='black',
                                linewidths=0.5,
                                s=2,
                                alpha=0.5,
                                zorder=4,
                            )

            # Labels and styling
            display_item = display_name_map.get(item_name, item_name) if display_name_map else item_name
            # y_label = 'Relative Prob. %' if data_type == 'percentage' else 'Mean Value'
            # y_label = ""
            # ax.set_ylabel(y_label, fontweight='bold', fontsize=14)
            if horizontal:
                ax.tick_params(axis='y', pad=6, labelsize=11)
                ax.set_yticks(unit_tick_positions)
                # Remove unit names from y-axis labels
                ax.set_yticklabels([])
                # Invert y-axis so bars appear in reverse order (top-to-bottom)
                ax.invert_yaxis()
                ax.tick_params(axis='x', pad=2, labelsize=11)
                
                # # Set consistent aspect ratio for the plot area (grid)
                # # Make plot grid square in data coordinates for consistent visual proportions
                # y_range = positions[-1] - positions[0] + bar_width if positions else 1.0
                # xlim = ax.get_xlim()
                # x_range = xlim[1] - xlim[0] if xlim else 1.0
                # aspect_ratio = y_range / x_range if x_range > 0 else 1.0
                # ax.set_aspect(aspect_ratio, adjustable='box')

            ax.set_title(f"{display_item}", fontweight='bold', fontsize=20, pad=15)
            
            # Hide category ticks only for the vertical layout; the horizontal layout
            # keeps the numeric x-axis visible.
            if not horizontal:
                ax.set_xticks([])
                ax.set_xticklabels([])
                ax.tick_params(axis='y', labelsize=16)

            # Legend 1: base unit colors only (no subgroup suffixes)
            unit_legend_patches = []
            if units_to_display is not None:
                base_units_for_legend = [u for u in units_to_display if any((pu.split(' (')[0] if ' (' in pu else pu) == u for pu in all_plot_units)]
            else:
                base_units_for_legend = []
                seen_base_units = set()
                for pu in all_plot_units:
                    bu = pu.split(' (')[0] if ' (' in pu else pu
                    if bu not in seen_base_units:
                        seen_base_units.add(bu)
                        base_units_for_legend.append(bu)

            for base_unit in base_units_for_legend:
                display_base = unit_display_map.get(base_unit, base_unit) if unit_display_map else base_unit
                base_color = unit_color_map.get(base_unit, '#888888') if unit_color_map else '#888888'
                unit_legend_patches.append(mpatches.Patch(color=base_color, label=display_base))

            # Always refresh legend so colors stay in sync with current run settings.
            unit_legend_path = os.path.join(output_dir, "unit_legend.png")
            self._save_standalone_legend(unit_legend_patches, "Units", unit_legend_path)

            # Legend 2: subgroup shading (lighter/darker) semantics
            # Use subgroup_list for ordering, but only include those that actually appear in all_plot_units
            subgroups_in_plot = set()
            for pu in all_plot_units:
                if ' (' in pu:
                    subgroups_in_plot.add(pu.split(' (')[-1].rstrip(')'))
            
            if subgroup_list is not None:
                subgroup_ids_present = [sg for sg in subgroup_list if sg in subgroups_in_plot]
            else:
                print("No subgroup_list provided; skipping subgroup legend.")
                subgroup_ids_present = []

            if subgroup_ids_present:
                subgroup_legend_patches = []
                neutral_base = '#888888'
                
                if not use_patterns:
                    neutral_luminance = np.mean(mcolors.to_rgb(neutral_base))

                for gid in subgroup_ids_present:
                    if use_patterns:
                        hatch = self._get_hatch_for_group(gid, subgroup_list=subgroup_list)
                        subgroup_legend_patches.append(
                            mpatches.Patch(facecolor=neutral_base, hatch=hatch, edgecolor='black', label=f"{gid}")
                        )
                    else:
                        shade_color = self._adjust_color_for_group(neutral_base, gid, subgroup_list=subgroup_list)
                        shade_luminance = np.mean(mcolors.to_rgb(shade_color))
                        shade_direction = 'lighter' if shade_luminance >= neutral_luminance else 'darker'
                        subgroup_legend_patches.append(
                            mpatches.Patch(color=shade_color, label=f"{gid} ({shade_direction})")
                        )

                legend_title = "Subgroup Patterns" if use_patterns else "Subgroup Shading"
                subgroup_legend_filename = "subgroup_pattern_legend.png" if use_patterns else "subgroup_shading_legend.png"
                subgroup_legend_path = os.path.join(output_dir, subgroup_legend_filename)
                self._save_standalone_legend(subgroup_legend_patches, legend_title, subgroup_legend_path)


            # --- Save significance results to CSV ---
            if output_dir and unit_significance:
                import csv
                import re
                item_safe_sig = re.sub(r'[<>:"/\\|?*]', '_', str(item_name))
                sig_csv_path = os.path.join(output_dir, f"{item_safe_sig}_significance.csv")
                # Build a reverse map: position -> unit label
                pos_to_label = {pos: lbl for pos, lbl in zip(unit_tick_positions, unit_tick_labels)}
                # Build a map: position -> mean value
                pos_to_mean = {pos: mean for pos, mean in zip(positions, bar_means)}
                with open(sig_csv_path, 'w', newline='') as csvf:
                    writer = csv.writer(csvf)
                    writer.writerow(['Group_1', 'Group_2', 'Significance', 'Greater_Mean'])
                    for x1, x2, sig_sym in unit_significance:
                        lbl1 = pos_to_label.get(x1, str(x1))
                        lbl2 = pos_to_label.get(x2, str(x2))
                        mean1 = pos_to_mean.get(x1, np.nan)
                        mean2 = pos_to_mean.get(x2, np.nan)
                        if np.isfinite(mean1) and np.isfinite(mean2):
                            greater = lbl1 if mean1 > mean2 else lbl2
                        else:
                            greater = "N/A"
                        writer.writerow([lbl1, lbl2, sig_sym, greater])
                print(f"  Saved significance table: {sig_csv_path}")

            # --- Draw significance markers on the plot ---
            _sig_style = str(sig_style).lower().strip()
            if unit_significance and _sig_style != 'export_only':
                if horizontal:
                    x_max = ax.get_xlim()[1]
                else:
                    y_max = ax.get_ylim()[1]

                if _sig_style == 'stars_only':
                    # Compact mode: pick strongest symbol per bar position
                    sym_rank = {'***': 3, '**': 2, '*': 1, 'ns': 0}
                    bar_sym_rank = {}
                    bar_sym_label = {}
                    for x1, x2, sig_sym in unit_significance:
                        rank = sym_rank.get(sig_sym, 0)
                        for pos in (x1, x2):
                            if rank > bar_sym_rank.get(pos, -1):
                                bar_sym_rank[pos] = rank
                                bar_sym_label[pos] = sig_sym

                    if horizontal:
                        star_x = x_max * 1.04 if x_max != 0 else 1.0
                        for pos, sym in bar_sym_label.items():
                            ax.text(star_x, pos, sym, ha='left', va='center',
                                    fontsize=11 if bar_width > 0.5 else 8, fontweight='bold', color='black')
                        ax.set_xlim(right=x_max * 1.15 if x_max != 0 else 1.15)
                    else:
                        star_y = y_max * 1.04
                        for pos, sym in bar_sym_label.items():
                            ax.text(pos, star_y, sym, ha='center', va='bottom',
                                    fontsize=11 if bar_width > 0.5 else 8, fontweight='bold', color='black')
                        ax.set_ylim(top=y_max * 1.15)

                else:  # 'brackets' (default)
                    line_height = 0.08
                    unit_significance.sort(key=lambda x: (abs(x[0]-x[1]), min(x[0], x[1])))
                    drawn_markers = []

                    for x1, x2, sig_sym in unit_significance:
                        if x1 > x2: x1, x2 = x2, x1

                        # Find first available stacking level
                        level = 0
                        while True:
                            collision = False
                            for (mx1, mx2), mlevel in drawn_markers:
                                if mlevel == level:
                                    if not (x2 + 0.2 < mx1 or x1 - 0.2 > mx2):
                                        collision = True
                                        break
                            if not collision:
                                break
                            level += 1

                        if horizontal:
                            x_pos = x_max * (1.10 + level * line_height)
                            ax.plot([x_pos, x_pos], [x1, x2], 'k-', linewidth=1.0)
                            ax.text(x_pos, (x1 + x2) / 2, sig_sym,
                                    ha='left', va='center', fontsize=10 if bar_width > 0.5 else 8, fontweight='bold')
                        else:
                            y_pos = y_max * (1.10 + level * line_height)
                            ax.plot([x1, x2], [y_pos, y_pos], 'k-', linewidth=1.0)
                            ax.text((x1 + x2) / 2, y_pos, sig_sym,
                                    ha='center', va='bottom', fontsize=10 if bar_width > 0.5 else 8, fontweight='bold')
                        drawn_markers.append(((x1, x2), level))

                    max_level = max(m[1] for m in drawn_markers) if drawn_markers else 0
                    if horizontal:
                        ax.set_xlim(right=x_max * (1.20 + max_level * line_height))
                        # Dynamically increase figure width if we have many levels
                        if max_level > 3:
                            fig.set_figwidth(fig_width + (max_level - 3) * 0.5)
                    else:
                        ax.set_ylim(top=y_max * (1.20 + max_level * line_height))
                        
                        # Dynamically increase figure height if we have many levels
                        if max_level > 3:
                            fig.set_figheight(fig_height + (max_level - 3) * 0.5)

            # --- Mark empty units with subtle vertical tick lines (flush with spine) ---
            if empty_positions:
                if horizontal:
                    x_min, x_max = ax.get_xlim()
                    tick_width = (x_max - x_min) * 0.01  # Small tick size
                    for y_pos in empty_positions:
                        # Draw a small horizontal tick line at the left edge, flush with spine
                        ax.plot([x_min, x_min + tick_width], [y_pos, y_pos], 'k-', linewidth=1.5, alpha=0.5, zorder=3)
                else:
                    y_min, y_max = ax.get_ylim()
                    tick_height = (y_max - y_min) * 0.01  # Small tick size
                    for x_pos in empty_positions:
                        # Draw a small vertical tick line at the bottom, starting from the spine (flush with x-axis)
                        ax.plot([x_pos, x_pos], [y_min, y_min + tick_height], 'k-', linewidth=1.5, alpha=0.5, zorder=3)

            # Grid and spines
            # Enable grid and send it behind the plot
            ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.7, zorder=0)
            ax.set_axisbelow(True)

            # Ensure the patch (background) is transparent so grid is visible
            ax.set_facecolor('white')  # Keep background white
            ax.patch.set_alpha(0)  
            for spine in ax.spines.values(): spine.set_edgecolor("#505050")
            
            # try:
            #     plt.tight_layout()
            # except Exception:
            #     # Fallback if decorations are too large for tight_layout
            #     pass
            # Save
            # Sanitize item_name for use as filename (remove/replace invalid Windows characters)
            import re
            item_safe = re.sub(r'[<>:"/\\|?*]', '_', str(item_name))
            orientation_suffix = '_horizontal' if horizontal else ''
            save_path = os.path.join(output_dir, f"{data_type}_{item_safe}_{plot_type}plot{orientation_suffix}.png")
            # print(f" Generated individual bar plot in {save_path}")

            plt.savefig(save_path, dpi=600, bbox_inches='tight', transparent=True)
            
            if show_plots: plt.show()
            else: plt.close(fig)

    def create_individual_barplots_horizontal(self, df, value_col, grouping_col, items_list=None,
                                               output_dir=None, data_type='percentage',
                                               show_plots=False, display_name_map=None,
                                               unit_color_map=None, unit_display_map=None,
                                               units_to_display=None,
                                               consolidate_Group_IDs=False,
                                               subgroup_list=None,
                                               compare_by="Group_ID",
                                               show_empty_subgroups=False,
                                               show_scatter=False,
                                               plot_type='bar',
                                               sig_style='brackets',
                                               use_patterns=True,
                                               min_size_to_plot=3,
                                               remove_outliers=True):
        """Create individual plots using the same workflow as create_individual_barplots, but horizontal."""
        return self.create_individual_barplots(
            df=df,
            value_col=value_col,
            grouping_col=grouping_col,
            items_list=items_list,
            output_dir=output_dir,
            data_type=data_type,
            show_plots=show_plots,
            display_name_map=display_name_map,
            unit_color_map=unit_color_map,
            unit_display_map=unit_display_map,
            units_to_display=units_to_display,
            consolidate_Group_IDs=consolidate_Group_IDs,
            subgroup_list=subgroup_list,
            compare_by=compare_by,
            show_empty_subgroups=show_empty_subgroups,
            show_scatter=show_scatter,
            plot_type=plot_type,
            sig_style=sig_style,
            use_patterns=use_patterns,
            min_size_to_plot=min_size_to_plot,
            remove_outliers=remove_outliers,
            orientation='horizontal',
        )

    def generate_spatial_value_maps(self, df, value_col, grouping_col, output_dir, 
                                   grid_size=(1000, 1000), cmap='magma', show_plots=False,
                                   vmin=None, vmax=None, selected_units=None):
        """
        Visualize the spatial distribution of a value (ratio or percentage) across an image.
        Creates a matrix of size (grid_size, grid_size) by binning centroid locations.
        
        Args:
            df: DataFrame containing the data (must have centroid_x, centroid_y)
            value_col: Column name to visualize (e.g., 'Mean_Ratio' or 'Percentage')
            grouping_col: Column name to group by (e.g., 'Ratio_Type' or 'Molecule')
            output_dir: Directory to save plots
            grid_size: Target dimension for the binned matrix. If None, uses 1:1 pixel mapping.
            cmap: Colormap to use.
            show_plots: Whether to display plots interactively.
            vmin, vmax: Range for colormap normalization.
            selected_units: Optional unit or list of units to filter by (e.g., 'Glomeruli').
        """

        os.makedirs(output_dir, exist_ok=True)

        if df is None or df.empty:
            print("No data available for spatial visualization.")
            return

        # Use Image_Name if available, else Source_ID
        img_col = 'Image_Name' if 'Image_Name' in df.columns else 'Source_ID'

        # Store full dataframe to determine global bounds per image
        full_df = df.copy()

        # Filter by selected units if provided
        if selected_units is not None:
            if isinstance(selected_units, str):
                selected_units = [selected_units]
            df = df[df['Unit'].isin(selected_units)].copy()
            if df.empty:
                print(f"No data found for selected units: {selected_units}")
                return
            print(f"Filtering spatial mapping for units: {selected_units}")

        # Iterate through unique images
        for img_name in df[img_col].unique():
            img_df = df[df[img_col] == img_name].copy()
            
            # Get global bounds for this image across ALL units to ensure alignment
            img_full_df = full_df[full_df[img_col] == img_name].dropna(subset=['centroid_x', 'centroid_y'])
            if img_full_df.empty:
                continue
                
            global_min_x, global_max_x = img_full_df['centroid_x'].min(), img_full_df['centroid_x'].max()
            global_min_y, global_max_y = img_full_df['centroid_y'].min(), img_full_df['centroid_y'].max()
            
            range_x = global_max_x - global_min_x
            range_y = global_max_y - global_min_y
            
            # Decide on matrix dimensions based on global range
            if grid_size is None:
                width = int(range_x) + 1
                height = int(range_y) + 1
            elif isinstance(grid_size, (list, tuple)) and len(grid_size) == 1:
                if range_x > range_y:
                    width = grid_size[0]
                    height = int((range_y / (range_x + 1e-8)) * grid_size[0])
                else:
                    height = grid_size[0]
                    width = int((range_x / (range_y + 1e-8)) * grid_size[0])
            elif isinstance(grid_size, (list, tuple)) and len(grid_size) == 2:
                width, height = grid_size[0], grid_size[1]
            else:
                # Single integer grid_size
                if range_x > range_y:
                    width = grid_size
                    height = int((range_y / (range_x + 1e-8)) * grid_size)
                else:
                    height = grid_size
                    width = int((range_x / (range_y + 1e-8)) * grid_size)
            
            width = max(1, int(width))
            height = max(1, int(height))
            
            for item_name in img_df[grouping_col].unique():
                subset = img_df[img_df[grouping_col] == item_name].copy()
                
                # Filter out points without coordinates
                subset = subset.dropna(subset=['centroid_x', 'centroid_y'])
                if subset.empty:
                    continue
                
                # Calculate bins relative to GLOBAL min/max
                if grid_size is None:
                    bin_x = (subset['centroid_x'] - global_min_x).astype(int)
                    bin_y = (subset['centroid_y'] - global_min_y).astype(int)
                else:
                    bin_x = ((subset['centroid_x'] - global_min_x) / (range_x + 1e-8) * (width - 1)).astype(int)
                    bin_y = ((subset['centroid_y'] - global_min_y) / (range_y + 1e-8) * (height - 1)).astype(int)
                
                # Create matrix
                matrix = np.full((height, width), np.nan)
                
                # Group by bins and calculate mean
                subset['bin_x'] = bin_x
                subset['bin_y'] = bin_y
                
                binned_data = subset.groupby(['bin_y', 'bin_x'])[value_col].mean().reset_index()
                
                for _, row in binned_data.iterrows():
                    by, bx = int(row['bin_y']), int(row['bin_x'])
                    if 0 <= by < height and 0 <= bx < width:
                        matrix[by, bx] = row[value_col]

                # Save tiff
                tifffile.imwrite(os.path.join(output_dir, f"{img_name}_{item_name}_{selected_units}.tif"), matrix, photometric='minisblack')
                
                # Plotting
                fig, ax = plt.subplots(figsize=(12, 10))
                
                # Use a background color for NaN values (empty areas)
                current_cmap = plt.get_cmap(cmap).copy()
                current_cmap.set_bad(color='#111111') # Near-black for empty
                
                im = ax.imshow(matrix, cmap=current_cmap, interpolation='nearest', 
                               origin='upper', vmin=vmin, vmax=vmax)
                
                # Add colorbar
                cbar = fig.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label(f"Mean {value_col.replace('_', ' ')}", fontsize=12, fontweight='bold')
                
                # Styling
                title_suffix = f" (Unit: {', '.join(selected_units)})" if selected_units else ""
                ax.set_title(f"Spatial Distribution: {item_name}{title_suffix}\nImage: {img_name}", 
                             fontweight='bold', fontsize=20, pad=20)
                
                if grid_size:
                    ax.set_xlabel(f"Binned X (Grid: {width})", fontsize=16)
                    ax.set_ylabel(f"Binned Y (Grid: {height})", fontsize=16)
                else:
                    ax.set_xlabel("X (pixels)", fontsize=16)
                    ax.set_ylabel("Y (pixels)", fontsize=16)
                
                plt.tight_layout()
                
                # Save
                item_safe = str(item_name).replace(':', '').replace('/', '_').replace(' ', '_')
                img_safe = str(img_name).replace(':', '').replace('/', '_').replace(' ', '_')
                save_path = os.path.join(output_dir, f"spatial_{value_col}_{item_safe}_{img_safe}_{selected_units}.png")
                plt.savefig(save_path, dpi=600, bbox_inches='tight')
                
                if show_plots:
                    plt.show()
                else:
                    plt.close(fig)
                
                print(f"Generated spatial map for {item_name} in {img_name}")

    def generate_directional_profiles(self, df, value_col, grouping_col, output_dir, pixel_size_um=1.0, 
                                     bin_ratio=10, normalize_auc=True, show_plots=False, 
                                     selected_units=None, directions=['x', 'y'],
                                     selected_items=None, combine_groups=True):
        """
        Visualize the variation of a value (ratio or percentage) across X and Y directions.
        Generates line graphs showing binned means with optional AUC normalization.
        
        Args:
            df: DataFrame containing the data (must have centroid_x, centroid_y)
            value_col: Column name to visualize (e.g., 'Mean_Ratio' or 'Percentage')
            grouping_col: Column name to group by (e.g., 'Ratio_Type' or 'Molecule')
            output_dir: Directory to save plots
            bin_ratio: The ratio of bin size to pixel size (e.g., 10 means 1 bin per 10 pixels).
            normalize_auc: If True, each line is normalized such that its sum equals 1.
            show_plots: Whether to display plots interactively.
            selected_units: Optional unit or list of units to filter by.
            directions: List of directions to plot ('x', 'y').
            selected_items: Optional list of specific items in grouping_col to include.
            combine_groups: If True, plots all groups in one figure. If False, creates separate figures.
        """
        os.makedirs(output_dir, exist_ok=True)

        if df is None or df.empty:
            print("No data available for directional profiles.")
            return

        # Use Image_Name if available, else Source_ID
        img_col = 'Image_Name' if 'Image_Name' in df.columns else 'Source_ID'
        full_df = df.copy()

        # Filter by selected units if provided
        if selected_units is not None:
            if isinstance(selected_units, str):
                selected_units = [selected_units]
            df = df[df['Unit'].isin(selected_units)].copy()
            if df.empty:
                print(f"No data found for selected units: {selected_units}")
                return

        # Filter by selected items if provided
        if selected_items is not None:
            if isinstance(selected_items, (str, int, float)):
                selected_items = [selected_items]
            df = df[df[grouping_col].isin(selected_items)].copy()
            if df.empty:
                print(f"No data found for selected items: {selected_items}")
                return

        for img_name in df[img_col].unique():
            img_df = df[df[img_col] == img_name].copy()
            
            # Get global bounds for this image
            img_full_df = full_df[full_df[img_col] == img_name].dropna(subset=['centroid_x', 'centroid_y'])
            if img_full_df.empty:
                continue
                
            for direction in directions:
                coord_col = 'centroid_x' if direction.lower() == 'x' else 'centroid_y'
                
                # Filter out points without coordinates
                img_df_clean = img_df.dropna(subset=[coord_col]).copy()
                if img_df_clean.empty:
                    continue
                
                # Determine bin edges based on full image range
                min_coord = img_full_df[coord_col].min()
                max_coord = img_full_df[coord_col].max()
                
                if max_coord == min_coord:
                    continue
                
                # Number of bins based on bin_ratio (pixels per bin)
                num_bins = max(1, int((max_coord - min_coord) / bin_ratio))
                
                # Create consistent bin edges
                bin_edges = np.linspace(min_coord, max_coord, num_bins + 1)
                img_df_clean['bin_idx'] = pd.cut(img_df_clean[coord_col], bins=bin_edges, labels=False, include_lowest=True)
                
                # Calculate mean per bin and group
                profile_data = img_df_clean.groupby(['bin_idx', grouping_col])[value_col].mean().reset_index()
                
                # Convert bin index to physical coordinates (microns)
                bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
                profile_data['coord_um'] = profile_data['bin_idx'].map(lambda i: bin_centers[int(i)] * pixel_size_um)
                
                # Normalize AUC if requested
                if normalize_auc:
                    for group in profile_data[grouping_col].unique():
                        mask = profile_data[grouping_col] == group
                        total_sum = profile_data.loc[mask, value_col].sum()
                        if total_sum > 0:
                            profile_data.loc[mask, value_col] = profile_data.loc[mask, value_col] / total_sum
                
                # Determine which groups to plot in this iteration
                unique_groups = profile_data[grouping_col].unique()
                if combine_groups:
                    group_sets = [unique_groups]
                else:
                    group_sets = [[g] for g in unique_groups]

                for groups_to_plot in group_sets:
                    plot_df = profile_data[profile_data[grouping_col].isin(groups_to_plot)]
                    if plot_df.empty:
                        continue
                        
                    # Plot
                    fig = plt.figure(figsize=(14, 6))
                    sns.lineplot(data=plot_df, x='coord_um', y=value_col, hue=grouping_col, linewidth=2.5, palette='viridis')
                    
                    # Formatting
                    group_title = "" if combine_groups else f" - {groups_to_plot[0]}"
                    plt.title(f"Spatial Profile Variation ({direction.upper()}): {img_name}{group_title}", 
                              fontweight='bold', fontsize=16, pad=20)
                    plt.xlabel(f"{direction.upper()} Distance (microns)", fontsize=14, fontweight='bold')
                    
                    y_label = f"Mean {value_col.replace('_', ' ')}"
                    if normalize_auc:
                        y_label += " (AUC Normalized)"
                    plt.ylabel(y_label, fontsize=14, fontweight='bold')
                    
                    plt.legend(title=grouping_col.replace('_', ' '), bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
                    plt.grid(True, linestyle='--', alpha=0.4)
                    
                    # Save
                    img_safe = str(img_name).replace(':', '').replace('/', '_').replace(' ', '_')
                    item_safe = "" if combine_groups else f"_{str(groups_to_plot[0]).replace(':', '').replace('/', '_').replace(' ', '_')}"
                    save_name = f"profile_{direction}_{img_safe}{item_safe}.png"
                        
                    save_path = os.path.join(output_dir, save_name)
                    plt.savefig(save_path, dpi=600, bbox_inches='tight')
                    
                    if show_plots:
                        plt.show()
                    else:
                        plt.close(fig)
                    
                    print(f"Generated {direction.upper()} directional profile for {img_name}{group_title}")

