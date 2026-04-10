import numpy as np
import matplotlib.pyplot as plt
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
    
    def __init__(self, mol_path, wavenumber_start=2700, wavenumber_end=3100, num_samples=61):
        """
        Initialize HSI Visualizer
        Args:
            wavenumber_start: Starting wavenumber for molecule dataset (default 2700)
            wavenumber_end: Ending wavenumber for molecule dataset (default 3100)
            num_samples: Number of samples in wavenumber range (default 61)
        """
        self.wavenumber_start = wavenumber_start
        self.wavenumber_end = wavenumber_end
        self.num_samples = num_samples

        print(f"Loading molecules from: {mol_path}")
        mol_data = np.load(mol_path, allow_pickle=True)
        self.normalized_molecules = mol_data['normalized_molecules']
        self.molecule_names = mol_data['molecule_names']
        print(f"HSI Visualizer initialized with {len(self.molecule_names)} molecules.")
    
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
        fig = plt.figure(figsize=(3, 2))
        fig.legend(handles=patches, title=title, loc='center', frameon=True)
        plt.axis('off')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
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
        
        # Filter the dataframe
        return df[(df[value_col] >= lower_bound) & (df[value_col] <= upper_bound)]


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
        # plt.show()
    
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
        Reconstruct probability stack for a specific image and save as TIFF
        
        Args:
            img_probabilities: Array of probabilities for a single image
            img_path: Path of the image to reconstruct
            output_path: Path to save the probability stack TIFF
            stats: Dictionary to image statistics
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

        # Save as TIFF with individual page labels
        try:
            # ImageJ format requires writing entire stack at once
            # Use contiguous series with proper metadata
            tifffile.imwrite(
                output_path,
                prob_stack.astype(np.float32),
                imagej=True,
                metadata={
                    'axes': 'ZYX',  # Stack as Z slices
                    'Labels': channel_names,  # List of string labels for each slice
                    'unit': 'pixel',
                    'spacing': 1.0,
                },   
            )
            
            print(f"Saved probability stack with {n_classes} channels to: {output_path}")
            
        except Exception as e:
            print(f"Warning: failed to save probability stack for {img_path}: {e}")

    def apply_rf_masking(self, prediction_csv_path=None, ratio_tiff_path=None, mask_list_path=None, prefix='CODEX', results_per_unit=None, 
                        subgroups=None, img_name=None, Source_ID=None, classes_to_ignore=None, group_subclasses=False):
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
            
        Returns:
            Dictionary with quantified data per unit and instance
        """
        if prediction_csv_path is None and ratio_tiff_path is None:
            raise ValueError("Must provide either prediction_csv_path or ratio_tiff_path")
        
        if prediction_csv_path is not None and ratio_tiff_path is not None:
            raise ValueError("Provide only one of prediction_csv_path or ratio_tiff_path, not both")
        
        # Set default classes to ignore if not provided
        if classes_to_ignore is None:
            classes_to_ignore = ['Masked', 'Kidney Background', 'No Match']
        classes_to_ignore_set = {cls.strip() for cls in classes_to_ignore}
        
        # Determine mode
        is_prediction_mode = prediction_csv_path is not None
        
        # Load data
        if is_prediction_mode:
            # Load CSV as matrix (no header, no index) - each cell is a predicted class name
            data_df = pd.read_csv(prediction_csv_path, header=None, index_col=False)
            data_matrix = data_df.values  # Convert to numpy array for easier indexing
            if img_name is None:
                img_name = os.path.basename(prediction_csv_path).replace('_predictions.csv', '')
        else: 
            data_img = tifffile.imread(ratio_tiff_path)
            if img_name is None:
                img_name = os.path.splitext(os.path.basename(ratio_tiff_path))[0]
                # Remove ratio type suffix (e.g., "image-Lipid_to_Protein-ratio" -> "image")
                if '-ratio' in img_name:
                    img_name = img_name.split('-')[0]
            
            # Extract ratio type from filename
            ratio_basename = os.path.basename(ratio_tiff_path).replace('.tif', '')
            ratio_type = ratio_basename.replace(f"{img_name}-", "").replace("-ratio", "")
        
        # Load mask files
        mask_files = glob(os.path.join(mask_list_path, '*[-mask][-Mask].tif'))
        if not mask_files:
            raise ValueError(f"No mask TIFF files found in {mask_list_path}")
        
        if results_per_unit is None:
            results_per_unit = {}
        
        # Process each mask file
        for mask_file in mask_files:
            # Load mask
            mask_img = tifffile.imread(mask_file)
            
            # Check if mask has extra dimensions and squeeze if needed
            if mask_img.ndim > 2:
                mask_img = np.squeeze(mask_img)
            
            # Ensure mask is 2D
            if mask_img.ndim != 2:
                print(f"Skipping {os.path.basename(mask_file)}: expected 2D mask, got shape {mask_img.shape}")
                continue
            
            # Extract unit name from mask filename
            mask_basename = os.path.basename(mask_file)
            # Remove prefix and mask suffix
            unit_name = mask_basename
            if prefix and unit_name.startswith(f"{prefix}-"):
                unit_name = unit_name[len(prefix)+1:]
            
            for suffix in ['-mask.tif', '-Mask.tif']:
                if unit_name.endswith(suffix):
                    unit_name = unit_name[:-len(suffix)]
            
            subgroup = None
            if "_" in unit_name and subgroups is not None:
                unit_parts = unit_name.split('_')
                if len(unit_parts) >= 2:
                    second_part = unit_parts[-1]
                    for s_name in subgroups:
                        # Match criteria: prefix or first-4-chars
                        if s_name.lower().startswith(second_part.lower()) or second_part.lower().startswith(s_name.lower()[:4]):
                            subgroup = s_name
                            break
                unit_name = unit_parts[0]
            
            # Check shape compatibility
            if is_prediction_mode:
                if data_matrix.shape != mask_img.shape:
                    print(f"Skipping {mask_basename}: shape mismatch (data: {data_matrix.shape}, mask: {mask_img.shape})")
                    continue
            else:
                if data_img.shape != mask_img.shape:
                    print(f"Skipping {mask_basename}: shape mismatch (data: {data_img.shape}, mask: {mask_img.shape})")
                    continue
            
            # Label mask for instance segmentation
            labeled_mask, num_features = ndi.label(mask_img > 0)
            if num_features == 0:
                continue
            
            # Minimum instance size threshold (in pixels)
            min_instance_size = 50
            
            # Process each connected component using local bounding boxes
            # to avoid repeatedly allocating full-image masks.
            object_slices = ndi.find_objects(labeled_mask)
            dilation_iterations = 1
            for label, obj_slice in enumerate(object_slices, start=1):
                if obj_slice is None:
                    continue

                # Add padding to the slice for dilation to avoid clipping at edges
                y_slice, x_slice = obj_slice
                y_start = max(0, y_slice.start - dilation_iterations)
                y_end = min(labeled_mask.shape[0], y_slice.stop + dilation_iterations)
                x_start = max(0, x_slice.start - dilation_iterations)
                x_end = min(labeled_mask.shape[1], x_slice.stop + dilation_iterations)
                
                padded_slice = (slice(y_start, y_end), slice(x_start, x_end))

                # Use padded slice to extract local label context
                local_labels = labeled_mask[padded_slice]
                instance_mask = local_labels == label

                # Check size before dilation
                instance_size = int(np.sum(instance_mask))
                if instance_size < min_instance_size:
                    continue
                
                # Apply dilation - now it has room to grow within the padded slice
                instance_mask = ndi.binary_dilation(instance_mask, iterations=dilation_iterations-1)
                
                # Check size after dilation
                if int(np.sum(instance_mask)) < min_instance_size:
                    continue
                
                if is_prediction_mode:
                    # Quantify predictions - extract class names at mask positions
                    predictions = data_matrix[padded_slice][instance_mask]
                    
                    # Check if we have valid predictions
                    if len(predictions) == 0:
                        continue
                    
                    # Count occurrences of each class
                    unique_classes, counts = np.unique(predictions, return_counts=True)
                    class_counts = {key.strip(): value for key, value in dict(zip(unique_classes, counts)).items()}
                    
                    # Remove ignored classes
                    for cls in classes_to_ignore_set:
                        if cls in class_counts:
                            del class_counts[cls]

                    if group_subclasses:
                        # Find mix classes available globally or fallback to present classes
                        if hasattr(self, 'molecule_names') and self.molecule_names is not None:
                            all_classes = self.molecule_names
                        else:
                            all_classes = list(class_counts.keys())
                            
                        mix_classes = [c.strip() for c in all_classes if isinstance(c, str) and "Mix" in c]
                        
                        for mix_class in mix_classes:
                            superclass = mix_class.replace(" Mix", "") # e.g., "TAG" from "TAG Mix"
                            
                            # Find subclasses present in current predictions that map to this mix class
                            subclasses = [c for c in class_counts.keys() if superclass in c and c != mix_class]
                            
                            if subclasses:
                                combined_count = class_counts.get(mix_class, 0)
                                for sub in subclasses:
                                    combined_count += class_counts[sub]
                                    del class_counts[sub]
                                class_counts[mix_class] = combined_count

                    total = sum(class_counts.values())
                    if total > 0:
                        pct = {k: (v / total * 100.0) for k, v in class_counts.items()}
                        
                        hierarchies = {}
                    else:
                        pct = {k: 0.0 for k in class_counts.items()}
                    
                    # Store prediction data
                    data_entry = {
                        'percentages': pct,
                        'image_name': img_name,
                        'Source_ID': Source_ID if Source_ID else img_name.split('-')[0],
                        'instance_label': f"{int(label)}"
                    }
                    if subgroup:
                        data_entry['Group_ID'] = subgroup
                    
                    results_per_unit.setdefault(unit_name, []).append(data_entry)
                
                else:
                    # Quantify ratios
                    masked_ratio = data_img[padded_slice][instance_mask]
                    
                    # Calculate mean ratio (excluding zeros)
                    non_zero_ratios = masked_ratio[masked_ratio > 0]
                    if non_zero_ratios.size > 0:
                        mean_ratio = np.mean(non_zero_ratios)
                        
                        # Store ratio data
                        ratio_entry = {
                            'ratios': {ratio_type: mean_ratio},
                            'image_name': img_name,
                            'Source_ID': Source_ID if Source_ID else img_name.split('-')[0],
                            'instance_label': f"{int(label)}"
                        }
                        if subgroup:
                            ratio_entry['Group_ID'] = subgroup
                        
                        results_per_unit.setdefault(unit_name, []).append(ratio_entry)
                    # If no non-zero ratios, skip this instance entirely
        return results_per_unit

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
                
                for molecule, percentage in pct_dict.items():
                    records.append({
                        'Unit': base_unit,
                        'Molecule': molecule,
                        'Percentage': percentage,
                        'Replicate': rep_idx + 1,
                        'Image_Name': image_name,
                        'Source_ID': Source_ID,
                        'Instance_Label': instance_label,
                        'Group_ID': Group_ID
                    })
        df = pd.DataFrame.from_records(records)
        return df


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
                else:
                    # Old format: just a dict of ratio values
                    ratio_dict = ratio_entry
                    image_name = 'Unknown'
                    Source_ID = 'Unknown'
                    instance_label = '0'
                    Group_ID = None
                

                
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
                        'Group_ID': Group_ID
                    })
        
        df = pd.DataFrame.from_records(records)
        return df

    def create_individual_boxplots(self, df, value_col, grouping_col, items_list,
                                  output_dir=None, data_type='percentage',
                                  figure_width=8.0, figure_height=5.0,
                                  show_plots=False, display_name_map=None,
                                  unit_color_map=None, unit_display_map=None,
                                  units_to_display=None,
                                  compare_by=None, compare_order=None,
                                  consolidate_Group_IDs=False):
        """
        Create individual box plots for publication, optionally comparing across sample types.
        
        Args:
            df: DataFrame containing the measurements.
            value_col: Column name for the values ('Percentage' or 'Mean_Ratio').
            grouping_col: Column name for the items ('Molecule' or 'Ratio_Type').
            items_list: List of molecules/ratios to plot.
            output_dir: Directory to save plots.
            compare_by: Column to compare within each unit (e.g., 'Sample_Type').
            compare_order: Ordered list of groups for the comparison column.
            consolidate_Group_IDs: If True, merges C/M Group_IDs into base Units for cleaner comparison.
        """
        import matplotlib.patches as mpatches
        from scipy.stats import f_oneway, ttest_ind

        if output_dir is None:
            output_dir = 'individual_boxplots'
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare color palette for comparison groups
        # Publication-quality colors: Teal, Orange, Gray or similar
        comparison_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # Filter units if specified
        if units_to_display is not None:
            df = df[df['Unit'].isin(units_to_display)].copy()
        
        # Consolidation check
        if consolidate_Group_IDs or compare_by == 'Group_ID':
            df['Plot_Unit'] = df['Unit']
        else:
            # Create Unit_Group_ID column if not consolidate
            def make_unit_Group_ID(row):
                Group_ID = row.get('Group_ID')
                if pd.isna(Group_ID) or Group_ID in [None, '', 'Other', 'O']:
                    return str(row['Unit'])
                # Abbreviate Group_ID
                reg_abbrev = str(Group_ID)[0] if len(str(Group_ID)) > 0 else ''
                if reg_abbrev in ['O', '']: return str(row['Unit'])
                return f"{row['Unit']} ({reg_abbrev})"
            df['Plot_Unit'] = df.apply(make_unit_Group_ID, axis=1)

        # Iterate through items
        for item_name in items_list:
            item_data = df[df[grouping_col] == item_name].copy()
            item_data.replace([np.inf, -np.inf], np.nan, inplace=True)
            item_data = item_data.dropna(subset=[value_col])
            
            if len(item_data) == 0:
                print(f"  Skipping {item_name}: no valid data")
                continue

            # Identify unique units to plot
            all_plot_units = sorted(item_data['Plot_Unit'].unique())
            
            # Prepare data groups
            plot_data = []
            group_labels = []
            colors = []
            positions = []
            
            current_pos = 1
            unit_tick_positions = []
            unit_tick_labels = []
            unit_group_positions = {}  # Map unit -> {group: position}
            
            # Sort units by original unit_order if possible
            # (Just uses alphabetical for Plot_Unit for now)
            
            for unit in all_plot_units:
                unit_data = item_data[item_data['Plot_Unit'] == unit]
                unit_start_pos = current_pos
                unit_group_positions[unit] = {}
                
                if compare_by and compare_by in item_data.columns:
                    # Get specific groups and their order
                    if compare_order:
                        groups = [g for g in compare_order if g in unit_data[compare_by].unique()]
                    else:
                        groups = sorted(unit_data[compare_by].unique())
                    
                    if not groups:
                        print(f"    WARNING: No comparison groups found for unit {unit} (column '{compare_by}')")
                    else:
                        print(f"    Found comparison groups for unit {unit}: {groups}")

                    for i, group in enumerate(groups):
                        group_data = unit_data[unit_data[compare_by] == group][value_col].values
                        if len(group_data) > 0:
                            plot_data.append(group_data)
                            unit_group_positions[unit][group] = current_pos
                            positions.append(current_pos)
                            colors.append(comparison_palette[i % len(comparison_palette)])
                            current_pos += 0.6
                    
                    # Calculate center for unit label
                    unit_tick_positions.append((unit_start_pos + current_pos - 0.6) / 2)
                    current_pos += 0.8  # Gap between units
                else:
                    # Simple unit-based plot
                    vals = unit_data[value_col].values
                    plot_data.append(vals)
                    positions.append(current_pos)
                    # Use unit_color_map if provided
                    base_unit = unit.split(' (')[0] if ' (' in unit else unit
                    color = unit_color_map.get(base_unit, '#888888') if unit_color_map else '#888888'
                    colors.append(color)
                    
                    unit_tick_positions.append(current_pos)
                    current_pos += 1.0

                # Map display name for unit
                display_unit = unit
                if unit_display_map:
                    base_unit = unit.split(' (')[0] if ' (' in unit else unit
                    if base_unit in unit_display_map:
                        mapped = unit_display_map[base_unit]
                        display_unit = unit.replace(base_unit, mapped)
                unit_tick_labels.append(display_unit)

            if not plot_data: continue

            # Plotting
            fig, ax = plt.subplots(figsize=(figure_width, figure_height))
            
            bp = ax.boxplot(plot_data, positions=positions, patch_artist=True,
                           widths=0.4, showfliers=False,
                           boxprops=dict(linewidth=1.2, edgecolor='black'),
                           medianprops=dict(color='black', linewidth=1.5),
                           whiskerprops=dict(color='black', linewidth=1.2),
                           capprops=dict(color='black', linewidth=1.2))
            
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.8)

            # Labels and styling
            ax.set_xticks(unit_tick_positions)
            ax.set_xticklabels(unit_tick_labels, rotation=45, ha='right', fontsize=12)
            
            display_item = display_name_map.get(item_name, item_name) if display_name_map else item_name
            y_label = 'Percentage (%)' if data_type == 'percentage' else 'Mean Ratio'
            ax.set_ylabel(y_label, fontweight='bold', fontsize=14)
            ax.set_title(f"{display_item} Distribution", fontweight='bold', fontsize=16, pad=15)
            
            # Simple Legend if comparing
            if compare_by and compare_order:
                legend_patches = [mpatches.Patch(color=comparison_palette[i], label=str(g).upper()) 
                                 for i, g in enumerate(compare_order)]
                ax.legend(handles=legend_patches, loc='upper right', frameon=True, fontsize=10)
            
            # Tukey HSD significance testing across units (when no compare_by is given)
            if not compare_by and len(all_plot_units) > 1:
                """
                Perform pairwise Tukey HSD test across displayed units.
                This tests whether the means of different units are significantly different.
                """
                # Prepare data for Tukey HSD: create DataFrame with Unit and value columns
                # Apply outlier filtration per group before pooling for Tukey
                item_data_filtered = self._apply_outlier_filtration(item_data, value_col, ['Plot_Unit'])
                
                records = []
                for _, row in item_data_filtered.iterrows():
                    records.append({'Unit': row['Plot_Unit'], 'Value': row[value_col]})
                
                if records and len(item_data_filtered['Plot_Unit'].unique()) > 1:
                    tukey_df = pd.DataFrame(records)
                    
                    try:
                        # Perform Tukey HSD test
                        tukey_result = pairwise_tukeyhsd(endog=tukey_df['Value'], 
                                                        groups=tukey_df['Unit'], 
                                                        alpha=0.05)
                        
                        # Extract p-values and create comparison dict
                        pvalues_dict = {}
                        for row in tukey_result.summary().data[1:]:  # Skip header
                            group1 = str(row[0])
                            group2 = str(row[1])
                            pval = float(row[3])  # Fixed: index 3 is p-adj, index 5 was upper CI bound
                            pvalues_dict[(group1, group2)] = pval
                        
                        # Draw significance bars directly on plot
                        sig_symbols = {
                            0.001: '***',
                            0.01: '**',
                            0.05: '*',
                            1.0: 'ns'
                        }
                        
                        # Sort comparisons by their span (smaller spans first to avoid overlaps)
                        sorted_comparisons = sorted(pvalues_dict.items(), 
                                                   key=lambda x: abs(unit_tick_positions[all_plot_units.index(x[0][1])] - 
                                                                      unit_tick_positions[all_plot_units.index(x[0][0])]))
                        
                        y_max = ax.get_ylim()[1]
                        line_height = 0.02
                        current_y_offset = 1
                        
                        for (unit1, unit2), pval in sorted_comparisons:
                            if unit1 not in all_plot_units or unit2 not in all_plot_units:
                                continue
                            
                            idx1 = all_plot_units.index(unit1)
                            idx2 = all_plot_units.index(unit2)
                            
                            if idx1 > idx2:
                                idx1, idx2 = idx2, idx1
                            
                            # Determine significance symbol
                            sig = 'ns'
                            for threshold, symbol in sorted(sig_symbols.items()):
                                if pval <= threshold:
                                    sig = symbol
                                    break
                            
                            if sig == 'ns':
                                continue  # Skip non-significant
                            
                            # Calculate bar y position and draw line
                            y_pos = y_max * (1.0 + line_height * current_y_offset)
                            x1, x2 = unit_tick_positions[idx1], unit_tick_positions[idx2]
                            ax.plot([x1, x2], [y_pos, y_pos], 'k-', linewidth=1.0)
                            ax.text((x1 + x2) / 2, y_pos, sig, ha='center', va='bottom', fontsize=10, fontweight='bold')
                            
                            current_y_offset += 1
                        
                        # Adjust y-limit to accommodate significance bars
                        ax.set_ylim(top=y_max * (1.0 + line_height * (current_y_offset + 1)))
                    except Exception as e:
                        print(f"    Warning: Could not perform Tukey HSD for {item_name}: {e}")

            # Inter-group significance testing (e.g., Cortex vs Medulla within each unit)
            elif compare_by and len(all_plot_units) >= 1:
                y_max = ax.get_ylim()[1]
                line_height = 0.05
                
                for unit in all_plot_units:
                    unit_data = item_data[item_data['Plot_Unit'] == unit]
                    groups_in_unit = sorted(unit_data[compare_by].unique())
                    if len(groups_in_unit) < 2:
                        continue
                        
                    # Prepare comparison data
                    comp_records = []
                    # Use filtered data for consistency
                    unit_filtered = self._apply_outlier_filtration(unit_data, value_col, [compare_by])
                    for group in groups_in_unit:
                        vals = unit_filtered[unit_filtered[compare_by] == group][value_col].values
                        for v in vals:
                            comp_records.append({'Group': group, 'Value': v})
                    
                    if not comp_records:
                        continue
                    comp_df = pd.DataFrame(comp_records)
                    
                    try:
                        # Perform Tukey or T-test
                        if len(groups_in_unit) == 2:
                            # Simple T-test for 2 groups
                            g1, g2 = groups_in_unit
                            v1 = comp_df[comp_df['Group'] == g1]['Value'].values
                            v2 = comp_df[comp_df['Group'] == g2]['Value'].values
                            if len(v1) >= 2 and len(v2) >= 2:
                                _, pval = ttest_ind(v1, v2, equal_var=False)
                                pvalues_dict = {(g1, g2): pval}
                            else:
                                pvalues_dict = {}
                        else:
                            # Tukey for >2 groups
                            tukey_res = pairwise_tukeyhsd(endog=comp_df['Value'], groups=comp_df['Group'], alpha=0.05)
                            pvalues_dict = {}
                            for row in tukey_res.summary().data[1:]:
                                pvalues_dict[(str(row[0]), str(row[1]))] = float(row[3])
                        
                        # Plot markers above unit cluster
                        current_unit_offset = 1
                        for (g1, g2), pval in pvalues_dict.items():
                            sig = 'ns'
                            sig_symbols = {0.001: '***', 0.01: '**', 0.05: '*', 1.0: 'ns'}
                            for threshold, symbol in sorted(sig_symbols.items()):
                                if pval <= threshold:
                                    sig = symbol
                                    break
                            
                            if sig == 'ns': continue
                            
                            x1 = unit_group_positions[unit][g1]
                            x2 = unit_group_positions[unit][g2]
                            y_pos = y_max * (1.0 + line_height * current_unit_offset)
                            
                            ax.plot([x1, x2], [y_pos, y_pos], 'k-', linewidth=0.8)
                            ax.text((x1 + x2) / 2, y_pos, sig, ha='center', va='bottom', fontsize=8, fontweight='bold')
                            current_unit_offset += 1
                            
                        # Update global y-limit if needed
                        if current_unit_offset > 1:
                            new_top = y_max * (1.0 + line_height * (current_unit_offset + 1))
                            if ax.get_ylim()[1] < new_top:
                                ax.set_ylim(top=new_top)
                                
                    except Exception as e:
                        print(f"    Warning: Could not perform inter-group significance for {item_name} in {unit}: {e}")
            
            # Inter-group significance testing (e.g., Cortex vs Medulla within each unit)
            elif compare_by and len(all_plot_units) >= 1:
                y_max = ax.get_ylim()[1]
                line_height = 0.05
                
                for unit in all_plot_units:
                    unit_data = item_data[item_data['Plot_Unit'] == unit]
                    groups_in_unit = sorted(unit_data[compare_by].unique())
                    if len(groups_in_unit) < 2:
                        continue
                        
                    # Prepare comparison data
                    comp_records = []
                    # Use filtered data for consistency
                    unit_filtered = self._apply_outlier_filtration(unit_data, value_col, [compare_by])
                    for group in groups_in_unit:
                        vals = unit_filtered[unit_filtered[compare_by] == group][value_col].values
                        for v in vals:
                            comp_records.append({'Group': group, 'Value': v})
                    
                    if not comp_records:
                        continue
                    comp_df = pd.DataFrame(comp_records)
                    
                    try:
                        # Perform Tukey or T-test
                        if len(groups_in_unit) == 2:
                            # Simple T-test for 2 groups
                            g1, g2 = groups_in_unit
                            v1 = comp_df[comp_df['Group'] == g1]['Value'].values
                            v2 = comp_df[comp_df['Group'] == g2]['Value'].values
                            if len(v1) >= 2 and len(v2) >= 2:
                                _, pval = ttest_ind(v1, v2, equal_var=False)
                                pvalues_dict = {(g1, g2): pval}
                            else:
                                pvalues_dict = {}
                        else:
                            # Tukey for >2 groups
                            tukey_res = pairwise_tukeyhsd(endog=comp_df['Value'], groups=comp_df['Group'], alpha=0.05)
                            pvalues_dict = {}
                            for row in tukey_res.summary().data[1:]:
                                pvalues_dict[(str(row[0]), str(row[1]))] = float(row[3])
                        
                        # Plot markers above unit cluster
                        current_unit_offset = 1
                        for (g1, g2), pval in pvalues_dict.items():
                            sig = 'ns'
                            sig_symbols = {0.001: '***', 0.01: '**', 0.05: '*', 1.0: 'ns'}
                            for threshold, symbol in sorted(sig_symbols.items()):
                                if pval <= threshold:
                                    sig = symbol
                                    break
                            
                            if sig == 'ns': continue
                            
                            x1 = unit_group_positions[unit][g1]
                            x2 = unit_group_positions[unit][g2]
                            y_pos = y_max * (1.0 + line_height * current_unit_offset)
                            
                            ax.plot([x1, x2], [y_pos, y_pos], 'k-', linewidth=0.8)
                            ax.text((x1 + x2) / 2, y_pos, sig, ha='center', va='bottom', fontsize=8, fontweight='bold')
                            current_unit_offset += 1
                            
                        # Update global y-limit if needed
                        if current_unit_offset > 1:
                            new_top = y_max * (1.0 + line_height * (current_unit_offset + 1))
                            if ax.get_ylim()[1] < new_top:
                                ax.set_ylim(top=new_top)
                                
                    except Exception as e:
                        print(f"    Warning: Could not perform inter-group significance for {item_name} in {unit}: {e}")

            # Grid and spines
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            for spine in ax.spines.values(): spine.set_edgecolor('#cccccc')
            
            plt.tight_layout()
            
            # Save
            item_safe = item_name.replace(':', '').replace('/', '_').replace(' ', '_')
            save_path = os.path.join(output_dir, f"{data_type}_{item_safe}_boxplot.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            if show_plots: plt.show()
            else: plt.close()

        return []

    def generate_unit_heatmaps(self, df, value_col, grouping_col, output_dir, data_type='percentage',
                               figsize=(8, 6), show_plots=False, cmap=None, vmin=-2, vmax=2, display_name_map=None, unit_display_map=None, unit_order=None,
                               units_to_display=None, top_n=20, groups_to_display=None, **kwargs):
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
        
        # if len(unique_groups) < 2 or len(unique_units) < 2:
        #     print(f"Insufficient data for unit heatmap: {len(unique_groups)} groups, {len(unique_units)} units")
        #     return {}
        
        # Check if secondary grouping column exists (e.g., Group_ID, Treatment)
        secondary_group_col = kwargs.get('secondary_group_col', 'Group_ID' if 'Group_ID' in df.columns else None)
        has_Group_ID = secondary_group_col is not None and df[secondary_group_col].notna().any()
        
        # Apply outlier filtration before aggregation to ensure stable Z-scores
        filter_groups = ['Unit', secondary_group_col, grouping_col] if has_Group_ID else ['Unit', grouping_col]
        df_clean = df
        # df_clean = self._apply_outlier_filtration(df, value_col, filter_groups)
        
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
        
        # if len(agg_data) < 2:
        #     print("Insufficient data after removing NaN/inf values")
        #     return {}
        
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
            fig_width = max(6, 1.0 * n_cols + 2.5)  # Increased from 0.7 to 1.0
            fig_height = max(3, min(14, 0.5 * n_rows + 2.5))  # Increased from 0.35 to 0.5, increased max height

            fig_width = max(6, 0.7 * n_cols + 2.5)
            fig_height = max(3, min(10, 0.35 * n_rows + 2.5))
            
            # Create heatmap visualization
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            ax.set_facecolor('black')
            
            # Create heatmap with journal-quality styling
            sns.heatmap(heatmap_matrix, annot=False, cmap=cmap, 
                       center=0, vmin=vmin, vmax=vmax,
                       cbar_kws={'label': 'Z-score (normalized to mean)', 'shrink': 0.8},
                       linewidths=0.5, linecolor='white', ax=ax, square=False)
            
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
                        # Abbreviate Group_ID to first letter only
                        Group_ID_abbrev = Group_ID_part.rstrip(')')[0] if Group_ID_part.rstrip(')') else Group_ID_part
                        if Group_ID_abbrev in ['O', 'Other', '', None]:
                            new_labels.append(f"{unit_part}")
                        else:
                            new_labels.append(f"{unit_part} ({Group_ID_abbrev})")
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
                                      f"heatmap_unit_aggregated_{data_type}.svg")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            
            if show_plots:
                plt.show()
            else:
                plt.close()
            
            print(f"\nGenerated unit-aggregated heatmap in {output_dir}")
            return heatmap_data
                
        except Exception as e:
            print(f"Error creating unit-aggregated heatmap: {str(e)}")
            return {}

    def generate_unit_bubble_charts(self, df, value_col, grouping_col, output_dir, data_type='percentage',
                                   show_plots=False, cmap=None, vmin=-2, vmax=2,
                                   display_name_map=None, unit_display_map=None, unit_order=None,
                                   units_to_display=None, top_n=20, groups_to_display=None, **kwargs):
        """
        Generate a bubble chart showing each molecule across units (aggregating across Group_IDs).

        Mirrors generate_unit_heatmaps but renders bubbles instead of filled cells:
          - Bubble size encodes the magnitude of the z-score (|z_score|)
          - Bubble color encodes the sign/magnitude via the chosen colormap
          - Black figure/axes background
          - White text for all labels and ticks
          - X-axis tick labels (molecules/ratios) are placed above the plot

        Parameters
        ----------
        df : pandas.DataFrame
        value_col : str
        grouping_col : str
        output_dir : str
        data_type : str
        show_plots : bool
        cmap : str
        vmin, vmax : float
        display_name_map : dict, optional
        unit_display_map : dict, optional
        unit_order : list, optional
        units_to_display : list, optional
            List of unit names to include; all others are excluded.
        top_n : int, optional
            Maximum number of top molecules to display (by z-score magnitude) when
            grouping_col is 'Molecule'. Default is 20. Ignored if groups_to_display is set.
        groups_to_display : list, optional
            Explicit list of molecule or ratio names to include. When provided, overrides
            top_n filtering and shows only these specific entries.

        Returns
        -------
        dict : {'matrix': heatmap_matrix, 'raw_data': agg_data}
        """
        import matplotlib.colors as mcolors

        os.makedirs(output_dir, exist_ok=True)

        if cmap is None:
            cmap = SVG_BUBBLE_CMAP

        # Filter to selected units if specified
        if units_to_display is not None:
            df = df[df['Unit'].isin(units_to_display)].copy()

        unique_groups = df[grouping_col].unique()
        unique_groups = [g.strip() for g in unique_groups]
        unique_units = df['Unit'].unique()

        # Check if secondary grouping column exists
        secondary_group_col = kwargs.get('secondary_group_col', 'Group_ID' if 'Group_ID' in df.columns else None)
        has_Group_ID = secondary_group_col is not None and df[secondary_group_col].notna().any()
        
        # Apply outlier filtration before aggregation
        filter_groups = ['Unit', secondary_group_col, grouping_col] if has_Group_ID else ['Unit', grouping_col]
        # df_clean = self._apply_outlier_filtration(df, value_col, filter_groups)

        df_clean = df
        
        if has_Group_ID:
            agg_data = df_clean.groupby(['Unit', secondary_group_col, grouping_col])[value_col].agg(['mean', 'std', 'count']).reset_index()
            agg_data['Unit_Group_Combo'] = agg_data['Unit'] + ' (' + agg_data[secondary_group_col].astype(str) + ')'
        else:
            agg_data = df_clean.groupby(['Unit', grouping_col])[value_col].agg(['mean', 'std', 'count']).reset_index()
            agg_data['Unit_Group_Combo'] = agg_data['Unit']

        agg_data = agg_data.replace([np.inf, -np.inf], np.nan)
        agg_data = agg_data.dropna(subset=['mean'])

        # if len(agg_data) < 2:
        #     print("Insufficient data after removing NaN/inf values")
        #     return {}

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
            # --- same filtering/pivoting logic as generate_unit_heatmaps ---
            if groups_to_display is not None:
                agg_sub = agg_data[agg_data[grouping_col].isin(groups_to_display)]
            elif grouping_col == 'Molecule':
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
            
            # Mask values with 0 mean or NaN
            mask = (mean_matrix == 0) | mean_matrix.isna()
            heatmap_matrix = heatmap_matrix.mask(mask, np.nan)

            # Sort columns
            if display_name_map:
                current_columns = heatmap_matrix.columns.tolist()
                ordered_columns = [k for k in display_name_map.keys() if k in current_columns]
                remaining_columns = sorted([g for g in current_columns if g not in display_name_map.keys()])
                heatmap_matrix = heatmap_matrix[ordered_columns + remaining_columns]
            else:
                heatmap_matrix = heatmap_matrix.sort_index(axis=1)

            # Sort rows
            if unit_order is not None:
                all_rows = heatmap_matrix.index.tolist()
                ordered_rows = []
                for unit in unit_order:
                    matching_rows = [r for r in all_rows if r.startswith(unit + ' (') or r == unit]
                    ordered_rows.extend(sorted(matching_rows))
                # Only keep rows that match unit_order (exclude unlisted units)
                heatmap_matrix = heatmap_matrix.reindex(ordered_rows).dropna(how='all')
            elif has_Group_ID:
                all_rows = heatmap_matrix.index.tolist()
                unit_group_map = {}
                for row in all_rows:
                    if ' (' in row:
                        base_unit = row.split(' (')[0]
                        unit_group_map.setdefault(base_unit, []).append(row)
                    else:
                        unit_group_map.setdefault(row, [row])
                ordered_rows = []
                for base_unit, rows in unit_group_map.items():
                    if len(rows) > 1:
                        ordered_rows.extend(sorted(rows))
                for base_unit, rows in unit_group_map.items():
                    if len(rows) == 1:
                        ordered_rows.extend(rows)
                heatmap_matrix = heatmap_matrix.reindex(ordered_rows)

            # Build display labels for columns (x-axis / molecules)
            col_labels = []
            for col in heatmap_matrix.columns:
                if display_name_map and col in display_name_map:
                    col_labels.append(display_name_map[col])
                else:
                    col_labels.append(col.replace('_', ' '))

            # Build display labels for rows (y-axis / units)
            row_labels = []
            for row in heatmap_matrix.index:
                if has_Group_ID and ' (' in row:
                    unit_part, Group_ID_part = row.split(' (', 1)
                    if unit_display_map and unit_part in unit_display_map:
                        unit_part = unit_display_map[unit_part]
                    Group_ID_abbrev = Group_ID_part.rstrip(')')[0] if Group_ID_part.rstrip(')') else Group_ID_part
                    if Group_ID_abbrev in ['O', 'Other', '', None]:
                        row_labels.append(f"{unit_part}")
                    else:
                        row_labels.append(f"{unit_part} ({Group_ID_abbrev})")
                else:
                    if unit_display_map and row in unit_display_map:
                        row_labels.append(unit_display_map[row])
                    else:
                        row_labels.append(row.replace('_', ' '))

            bubble_data = {'matrix': heatmap_matrix, 'raw_data': agg_data}

            # --- Figure sizing and spacing ---
            n_rows = heatmap_matrix.shape[0]
            n_cols = heatmap_matrix.shape[1]
            spacing = 0.5  # <1 brings bubbles closer together
            
            # Standardized scaling with heatmap: ~0.5 inches per row, ~1.0 inches per column
            fig_width = max(6, 0.7 * n_cols + 2.5)
            fig_height = max(3, min(10, 0.35 * n_rows + 2.5))
            
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')

            # --- Build scatter coordinates ---
            col_index = {col: i * spacing for i, col in enumerate(heatmap_matrix.columns)}
            row_index = {row: i * spacing for i, row in enumerate(heatmap_matrix.index)}

            xs, ys, sizes, color_vals = [], [], [], []
            xs_black, ys_black, sizes_black = [], [], []
            all_z = heatmap_matrix.values.flatten()
            max_abs_z = max(np.abs(all_z).max(), 0.1)
            max_bubble_size = 250
            min_bubble_size = 50

            # size/color calculations will use the synchronized heatmap_matrix and mean_matrix

            # Size bubbles based on 1-vs-rest t-test significance
            from scipy.stats import ttest_ind
            
            for row_name in heatmap_matrix.index:
                for col_name in heatmap_matrix.columns:
                    z = heatmap_matrix.loc[row_name, col_name]
                    mean_val = mean_matrix.loc[row_name, col_name]
                    x_val = col_index[col_name]
                    y_val = row_index[row_name]
                    
                    # If the value is 0 or nan, force smallest bubble and color black
                    if pd.isna(mean_val) or mean_val == 0:
                        xs_black.append(x_val)
                        ys_black.append(y_val)
                        sizes_black.append(min_bubble_size)
                    else:
                        # Get the corresponding agg_data row to extract Unit and Group_ID
                        matching_agg_rows = agg_data[(agg_data['Unit_Group_Combo'] == row_name) & 
                                                      (agg_data[grouping_col] == col_name)]
                        
                        if len(matching_agg_rows) == 0:
                            xs_black.append(x_val)
                            ys_black.append(y_val)
                            sizes_black.append(min_bubble_size)
                            continue
                        
                        unit_name = matching_agg_rows.iloc[0]['Unit']
                        secondary_val = matching_agg_rows.iloc[0][secondary_group_col] if has_Group_ID else None
                        
                        # Get data for this specific group combination from the already-filtered clean dataframe
                        if has_Group_ID and secondary_val is not None:
                            group_mask = (df_clean['Unit'] == unit_name) & (df_clean[secondary_group_col] == secondary_val) & (df_clean[grouping_col] == col_name)
                            other_mask = (~((df_clean['Unit'] == unit_name) & (df_clean[secondary_group_col] == secondary_val))) & (df_clean[grouping_col] == col_name)
                        else:
                            group_mask = (df_clean['Unit'] == unit_name) & (df_clean[grouping_col] == col_name)
                            other_mask = (df_clean['Unit'] != unit_name) & (df_clean[grouping_col] == col_name)

                        this_vals = df_clean[group_mask][value_col].values
                        other_vals = df_clean[other_mask][value_col].values
                        
                        # Filter outliers for the comparison group (rest of data for same molecule)
                        # Note: Per-group filtration is already handled by df_clean
                        
                        # Calculate p-value via t-test
                        if len(this_vals) >= 2 and len(other_vals) >= 2:
                            t_stat, p_val = ttest_ind(this_vals, other_vals, equal_var=False)
                            if pd.isna(p_val):
                                p_val = 1.0
                        else:
                            p_val = 1.0
                        
                        # Size based on significance
                        if p_val >= 0.05:
                            size = min_bubble_size
                        elif p_val < 0.05 and p_val >= 0.01:
                            size = 125
                        elif p_val < 0.01 and p_val >= 0.001:
                            size = 200
                        elif p_val < 0.001:
                            size = 250
                        
                        xs.append(x_val)
                        ys.append(y_val)
                        sizes.append(size)
                        color_vals.append(z)

            # Use TwoSlopeNorm only if vmin < 0 < vmax, otherwise use standard Norm
            if vmin < 0 and vmax > 0:
                norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            else:
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            # Plot normal colored bubbles
            if xs:
                sc = ax.scatter(xs, ys, s=sizes, c=color_vals, cmap=cmap, norm=norm,
                                edgecolors='none', zorder=3)
            else:
                # Dummy scatter for colorbar if no normal data
                sc = ax.scatter([0], [0], s=[0], c=[0], cmap=cmap, norm=norm)
                
            # Plot black bubbles for NaN/0
            if xs_black:
                ax.scatter(xs_black, ys_black, s=sizes_black, c='black',
                           edgecolors='none', zorder=3)

            # --- Colorbar ---
            cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label('Z-score (normalized to mean)', color='black', fontsize=14)
            cbar.ax.yaxis.set_tick_params(color='black', labelcolor='black', labelsize=12)
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color='black')
            cbar.outline.set_edgecolor('black')

            # --- Horizontal separator lines between different base units ---
            if has_Group_ID:
                prev_unit_name = None
                for ridx, row_name in enumerate(heatmap_matrix.index.tolist()):
                    current_unit = row_name.split(' (')[0] if ' (' in row_name else row_name
                    if prev_unit_name is not None and current_unit != prev_unit_name:
                        ax.axhline(y=ridx * spacing - spacing * 0.5, color='#aaaaaa', linewidth=1.0, zorder=2)
                    prev_unit_name = current_unit

            # --- Subtle grid ---
            ax.set_xticks([i * spacing for i in range(n_cols)])
            ax.set_yticks([i * spacing for i in range(n_rows)])
            ax.grid(True, color='#dddddd', linewidth=0.5, zorder=1)
            ax.set_axisbelow(True)

            # --- X-tick labels above the plot ---
            ax.xaxis.set_label_position('top')
            ax.xaxis.tick_top()
            ax.set_xticklabels(col_labels, rotation=45, ha='left', fontsize=14, color='black')

            # --- Y-tick labels ---
            ax.set_yticklabels(row_labels, rotation=0, ha='right', fontsize=14, color='black')

            # --- Axis labels ---
            ylabel = 'Unit (Group_ID)' if has_Group_ID else 'Unit'
            ax.set_ylabel(ylabel, fontweight='bold', fontsize=16, color='black')

            # --- Title ---
            title = 'Normalized Molecule Percentage' if data_type == 'percentage' else 'Normalized Ratio'
            ax.set_title(title, fontweight='bold', fontsize=18, pad=20, color='black')

            # --- Tick colors ---
            ax.tick_params(colors='black', which='both')

            # --- Spine colors ---
            for spine in ax.spines.values():
                spine.set_edgecolor('#cccccc')

            # --- Axis limits ---
            ax.set_xlim(-spacing * 0.5, (n_cols - 1) * spacing + spacing * 0.5)
            ax.set_ylim(-spacing * 0.5, (n_rows - 1) * spacing + spacing * 0.5)


            # --- Save significance legend as separate SVG ---
            from matplotlib.lines import Line2D
            significance_legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=5,
                label='p ≥ 0.05', markeredgecolor='black', linewidth=0.5),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8,
                label='p < 0.05', markeredgecolor='black', linewidth=0.5),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10,
                label='p < 0.01', markeredgecolor='black', linewidth=0.5),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=12,
                label='p < 0.001', markeredgecolor='black', linewidth=0.5),
            ]
            legend_fig, legend_ax = plt.subplots(figsize=(2.5, 1.2))
            legend = legend_ax.legend(handles=significance_legend_elements, loc='center', frameon=True, fontsize=11, title='Significance', title_fontsize=12, ncol=2)
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_alpha(0.95)
            legend_ax.axis('off')
            legend_svg_path = os.path.join(output_dir, f"bubble_significance_legend.svg")
            legend_fig.savefig(legend_svg_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(legend_fig)

            plt.tight_layout()

            # plt.tight_layout()

            output_path = os.path.join(output_dir, f"bubble_unit_aggregated_{data_type}.svg")
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

            if show_plots:
                plt.show()
            else:
                plt.close()

            print(f"\nGenerated unit-aggregated bubble chart in {output_dir}")
            return bubble_data

        except Exception as e:
            print(f"Error creating unit-aggregated bubble chart: {str(e)}")
            return {}



    def generate_raw_ratio_bubble_chart(self, df, value_col='Mean_Ratio', grouping_col='Ratio_Type', 
                                        output_dir=None, show_plots=True, cmap=None, vmin=None, vmax=None,
                                        display_name_map=None, unit_display_map=None, unit_order=None,
                                        units_to_display=None, groups_to_display=None):
        """
        Refactored raw ratio bubble chart aligning with generate_unit_bubble_charts logic.
        
        Bubble size encodes 1-vs-rest t-test significance.
        Bubble color encodes the raw mean value using RdBu_r colormap centered at global mean.
        Styling matches the standardized unit bubble charts (white background, labels above).

        Parameters
        ----------
        df : pandas.DataFrame
        value_col : str
        grouping_col : str
        output_dir : str
        show_plots : bool
        cmap : str, optional
        vmin, vmax : float, optional
        display_name_map : dict, optional
        unit_display_map : dict, optional
        unit_order : list, optional
        units_to_display : list, optional
        groups_to_display : list, optional

        Returns
        -------
        dict : {'matrix': raw_matrix, 'raw_data': agg_data}
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
            # Filter to selected units if specified
            if units_to_display is not None:
                df = df[df['Unit'].isin(units_to_display)].copy()

            # Remove NaN and infinite values
            clean_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[value_col])
            
            # if len(clean_df) < 2:
            #     print("Insufficient data for raw ratio bubble chart")
            #     return {}

            # Aggregation logic matching generate_unit_bubble_charts
            has_Group_ID = 'Group_ID' in clean_df.columns and clean_df['Group_ID'].notna().any()
            
            # Apply outlier filtration before aggregation
            filter_groups = ['Unit', 'Group_ID', grouping_col] if has_Group_ID else ['Unit', grouping_col]
            clean_df_filtered = self._apply_outlier_filtration(clean_df, value_col, filter_groups)
            
            if has_Group_ID:
                agg_data = clean_df_filtered.groupby(['Unit', 'Group_ID', grouping_col])[value_col].agg(['mean', 'std', 'count']).reset_index()
                agg_data['Unit_Group_Combo'] = agg_data['Unit'] + ' (' + agg_data['Group_ID'] + ')'
            else:
                agg_data = clean_df_filtered.groupby(['Unit', grouping_col])[value_col].agg(['mean', 'std', 'count']).reset_index()
                agg_data['Unit_Group_Combo'] = agg_data['Unit']

            # Filtering groups
            if groups_to_display is not None:
                agg_data = agg_data[agg_data[grouping_col].isin(groups_to_display)]

            # if len(agg_data) < 2:
            #     print("Insufficient data for raw ratio bubble chart after filtering")
            #     return {}

            # Raw values for matrix (instead of Z-score)
            raw_matrix = agg_data.pivot(index='Unit_Group_Combo', columns=grouping_col, values='mean')
            raw_matrix = raw_matrix.fillna(0)

            # Sort columns
            if display_name_map:
                current_columns = raw_matrix.columns.tolist()
                ordered_columns = [k for k in display_name_map.keys() if k in current_columns]
                remaining_columns = sorted([g for g in current_columns if g not in display_name_map.keys()])
                raw_matrix = raw_matrix[ordered_columns + remaining_columns]
            else:
                raw_matrix = raw_matrix.sort_index(axis=1)

            # Sort rows
            if unit_order is not None:
                all_rows = raw_matrix.index.tolist()
                ordered_rows = []
                for unit in unit_order:
                    matching_rows = [r for r in all_rows if r.startswith(unit + ' (') or r == unit]
                    ordered_rows.extend(sorted(matching_rows))
                # Only keep rows that match unit_order (exclude unlisted units)
                raw_matrix = raw_matrix.reindex(ordered_rows).dropna(how='all')
            elif has_Group_ID:
                all_rows = raw_matrix.index.tolist()
                unit_Group_ID_map = {}
                for row in all_rows:
                    if ' (' in row:
                        base_unit = row.split(' (')[0]
                        unit_Group_ID_map.setdefault(base_unit, []).append(row)
                    else:
                        unit_Group_ID_map.setdefault(row, [row])
                ordered_rows = []
                for base_unit, rows in unit_Group_ID_map.items():
                    if len(rows) > 1:
                        ordered_rows.extend(sorted(rows))
                for base_unit, rows in unit_Group_ID_map.items():
                    if len(rows) == 1:
                        ordered_rows.extend(rows)
                raw_matrix = raw_matrix.reindex(ordered_rows)

            # Display labels
            col_labels = [self._apply_display_name(c, display_name_map) for c in raw_matrix.columns]
            row_labels = []
            for row in raw_matrix.index:
                if has_Group_ID and ' (' in row:
                    unit_part, Group_ID_part = row.split(' (', 1)
                    unit_display = unit_display_map[unit_part] if unit_display_map and unit_part in unit_display_map else unit_part
                    Group_ID_abbrev = Group_ID_part.rstrip(')')[0] if Group_ID_part.rstrip(')') else Group_ID_part
                    if Group_ID_abbrev in ['O', 'Other', '', None]:
                        row_labels.append(f"{unit_display}")
                    else:
                        row_labels.append(f"{unit_display} ({Group_ID_abbrev})")
                else:
                    if unit_display_map and row in unit_display_map:
                        row_labels.append(unit_display_map[row])
                    else:
                        row_labels.append(row.replace('_', ' '))

            # Figure sizing
            n_rows, n_cols = raw_matrix.shape
            spacing = 0.5
            fig_width = max(6, 0.7 * n_cols + 2.5)
            fig_height = max(3, min(10, 0.35 * n_rows + 2.5))
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')

            # Build coordinates and calculate significance
            col_index = {col: i * spacing for i, col in enumerate(raw_matrix.columns)}
            row_index = {row: i * spacing for i, row in enumerate(raw_matrix.index)}
            xs, ys, sizes, color_vals = [], [], [], []
            xs_black, ys_black, sizes_black = [], [], []

            global_mean = agg_data['mean'].mean()
            global_max = max(agg_data['mean'].max(), global_mean + 0.1)
            global_min = min(agg_data['mean'].min(), global_mean - 0.1)
            
            # Set dynamic limits if not provided
            if vmin is None: vmin = global_min
            if vmax is None: vmax = global_max
            vcenter = global_mean

            min_bubble_size = 50

            for row_name in raw_matrix.index:
                for col_name in raw_matrix.columns:
                    val = raw_matrix.loc[row_name, col_name]
                    x_val, y_val = col_index[col_name], row_index[row_name]
                    
                    if pd.isna(val) or val == 0:
                        xs_black.append(x_val); ys_black.append(y_val); sizes_black.append(min_bubble_size)
                    else:
                        # Significance test matching generate_unit_bubble_charts
                        matching_agg = agg_data[(agg_data['Unit_Group_Combo'] == row_name) & (agg_data[grouping_col] == col_name)]
                        if len(matching_agg) == 0:
                            xs_black.append(x_val); ys_black.append(y_val); sizes_black.append(min_bubble_size); continue
                        
                        unit_nm = matching_agg.iloc[0]['Unit']
                        reg_nm = matching_agg.iloc[0]['Group_ID'] if 'Group_ID' in matching_agg.columns else None
                        
                        # Get data from the already-filtered clean_df_filtered for per-group consistency
                        if has_Group_ID and reg_nm:
                            unit_Group_ID_mask = (clean_df_filtered['Unit'] == unit_nm) & (clean_df_filtered['Group_ID'] == reg_nm) & (clean_df_filtered[grouping_col] == col_name)
                            other_mask = (~((clean_df_filtered['Unit'] == unit_nm) & (clean_df_filtered['Group_ID'] == reg_nm))) & (clean_df_filtered[grouping_col] == col_name)
                        else:
                            unit_Group_ID_mask = (clean_df_filtered['Unit'] == unit_nm) & (clean_df_filtered[grouping_col] == col_name)
                            other_mask = (clean_df_filtered['Unit'] != unit_nm) & (clean_df_filtered[grouping_col] == col_name)
                        
                        this_vals = clean_df_filtered[unit_Group_ID_mask][value_col].values
                        other_vals = clean_df_filtered[other_mask][value_col].values

                        # Note: Per-group filtration is already handled by clean_df_filtered

                        p_val = 1.0
                        if len(this_vals) >= 2 and len(other_vals) >= 2:
                            _, p_val = ttest_ind(this_vals, other_vals, equal_var=False)
                            if pd.isna(p_val): p_val = 1.0
                        
                        # Size logic
                        if p_val >= 0.05: size = min_bubble_size
                        elif p_val >= 0.01: size = 125
                        elif p_val >= 0.001: size = 200
                        else: size = 250
                        
                        xs.append(x_val); ys.append(y_val); sizes.append(size); color_vals.append(val)

            # Normalization and Scatter
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
            if xs:
                sc = ax.scatter(xs, ys, s=sizes, c=color_vals, cmap=cmap, norm=norm, edgecolors='none', zorder=3)
            else:
                sc = ax.scatter([0], [0], s=[0], c=[0], cmap=cmap, norm=norm)
            
            if xs_black:
                ax.scatter(xs_black, ys_black, s=sizes_black, c='black', edgecolors='none', zorder=3)

            # Colorbar
            cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label('Mean Ratio', color='black', fontsize=14)
            cbar.ax.yaxis.set_tick_params(color='black', labelcolor='black', labelsize=12)
            cbar.outline.set_edgecolor('black')

            # Styling
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

            # Significance Legend SVG
            legend_fig, legend_ax = plt.subplots(figsize=(2.5, 1.2))
            significance_legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=5, label='p ≥ 0.05', markeredgecolor='black', linewidth=0.5),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='p < 0.05', markeredgecolor='black', linewidth=0.5),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='p < 0.01', markeredgecolor='black', linewidth=0.5),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=12, label='p < 0.001', markeredgecolor='black', linewidth=0.5),
            ]
            legend = legend_ax.legend(handles=significance_legend_elements, loc='center', frameon=True, fontsize=11, title='Significance', title_fontsize=12, ncol=2)
            legend_ax.axis('off')
            legend_fig.savefig(os.path.join(output_dir, "bubble_significance_legend_raw.svg"), dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(legend_fig)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "bubble_chart_raw_ratios.svg"), dpi=300, bbox_inches='tight', facecolor='white')
            
            if show_plots: plt.show()
            else: plt.close()
            
            return {'matrix': raw_matrix, 'raw_data': agg_data}
        except Exception as e:
            print(f"Error creating raw ratio bubble chart: {str(e)}")
            return {}

    def create_individual_barplots(self, df, value_col, grouping_col, items_list=None,
                                    output_dir=None, data_type='percentage',
                                    show_plots=False, display_name_map=None,
                                    unit_color_map=None, unit_display_map=None,
                                    units_to_display=None,
                                    compare_by=None, compare_order=None,
                                    consolidate_Group_IDs=False):
        """
        Create individual bar plots for publication, optionally comparing across sample types.
        Shows mean ± standard error for each unit.
        
        Args:
            df: DataFrame containing the measurements.
            value_col: Column name for the values ('Percentage' or 'Mean_Ratio').
            grouping_col: Column name for the items ('Molecule' or 'Ratio_Type').
            items_list: List of molecules/ratios to plot.
            output_dir: Directory to save plots.
            compare_by: Column to compare within each unit (e.g., 'Sample_Type').
            compare_order: Ordered list of groups for the comparison column.
            consolidate_Group_IDs: If True, merges C/M Group_IDs into base Units for cleaner comparison.
        """
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        from scipy.stats import f_oneway, ttest_ind

        if output_dir is None:
            output_dir = 'individual_barplots'
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare color palette for comparison groups
        comparison_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        if compare_by == 'Group_ID':
            # Specific colors for Cortex/Medulla if that's the comparison
            group_colors = {'Cortex': '#4dabf7', 'Medulla': '#ff922b', 'C': '#4dabf7', 'M': '#ff922b'}
        
        print(f"Units to display: {units_to_display if units_to_display else 'All'}")

        # Filter units if specified
        if units_to_display is not None:
            df = df[df['Unit'].isin(units_to_display)].copy()
        
        # Consolidation check
        if consolidate_Group_IDs or compare_by == 'Group_ID':
            df['Plot_Unit'] = df['Unit']
        else:
            # Create Unit_Group_ID column if not consolidate
            def make_unit_Group_ID(row):
                Group_ID = row.get('Group_ID')
                if pd.isna(Group_ID) or Group_ID in [None, '', 'Other', 'O']:
                    return str(row['Unit'])
                # Abbreviate Group_ID
                reg_abbrev = str(Group_ID)[0] if len(str(Group_ID)) > 0 else ''
                if reg_abbrev in ['O', '']: return str(row['Unit'])
                return f"{row['Unit']} ({reg_abbrev})"
            df['Plot_Unit'] = df.apply(make_unit_Group_ID, axis=1)

        if items_list == "All" or items_list is None:
            items_list = sorted(df[grouping_col].unique())

        # Iterate through items
        for item_name in items_list:
            item_data = df[df[grouping_col] == item_name].copy()
            item_data.replace([np.inf, -np.inf], np.nan, inplace=True)
            item_data = item_data.dropna(subset=[value_col])
            
            if len(item_data) == 0:
                print(f"  Skipping {item_name}: no valid data")
                continue

            # Identify unique units to plot
            all_plot_units = sorted(item_data['Plot_Unit'].unique())
            
            # Identify all unique comparison groups for this item to ensure consistency across units
            all_item_groups = []
            if compare_by and compare_by in item_data.columns:
                if compare_order:
                    # Use provided order but filter to what exists in this item's data
                    all_item_groups = [g for g in compare_order if g in item_data[compare_by].unique()]
                else:
                    all_item_groups = sorted(item_data[compare_by].unique())

            
            # Prepare data groups
            bar_means = []
            bar_stderrs = []
            colors = []
            positions = []
            
            current_pos = 1
            unit_tick_positions = []
            unit_tick_labels = []
            unit_group_positions = {}  # Map unit -> {group: position}
            unit_significance = []     # List of (pos1, pos2, sig_symbol)
            
            for unit in all_plot_units:
                unit_data = item_data[item_data['Plot_Unit'] == unit]
                unit_start_pos = current_pos
                
                if compare_by and compare_by in item_data.columns:
                    # Iterate through all possible groups for this item to maintain consistent positions/colors
                    for i, group in enumerate(all_item_groups):
                        group_data = unit_data[unit_data[compare_by] == group]

                        if not group_data.empty:
                            unit_group_positions.setdefault(unit, {})[group] = current_pos
                            # Consistency: Remove outliers using IQR method for both bar heights and statistical tests
                            filtered_group = self._apply_outlier_filtration(group_data, value_col, [compare_by])
                            filtered_group_vals = filtered_group[value_col].values
                            
                            if len(filtered_group_vals) > 0:
                                bar_means.append(np.mean(filtered_group_vals))
                                bar_stderrs.append(np.std(filtered_group_vals) / np.sqrt(len(filtered_group_vals)))
                            else:
                                # Fallback if everything filtered (unlikely)
                                raw_vals = group_data[value_col].values
                                bar_means.append(np.mean(raw_vals))
                                bar_stderrs.append(np.std(raw_vals) / np.sqrt(len(raw_vals)))
                            
                            positions.append(current_pos)
                            
                            # Use specific colors for Group_ID if available
                            if compare_by == 'Group_ID' and group in group_colors:
                                colors.append(group_colors[group])
                            else:
                                colors.append(comparison_palette[i % len(comparison_palette)])
                                
                        # Increment even if empty to leave a gap (keeps bars aligned across units)
                        current_pos += 0.45 
                    
                    # Calculate center for unit label
                    unit_tick_positions.append((unit_start_pos + current_pos - 0.45) / 2)
                    
                    # --- Intra-unit statistical analysis ---
                    # Only collect groups that actually have data for this specific unit
                    unit_groups_with_data = [g for g in all_item_groups if not unit_data[unit_data[compare_by] == g].empty]
                    if compare_by is not None and len(unit_groups_with_data) >= 1:
                        group_data_list = []
                        group_labels = []
                        for group in unit_groups_with_data:
                            g_data = unit_data[unit_data[compare_by] == group]
                            g_data_filtered = self._apply_outlier_filtration(g_data, value_col, [compare_by])
                            if not g_data_filtered.empty:
                                group_data_list.append(g_data_filtered[value_col].values)
                                group_labels.append(group)
                        
                        if len(group_data_list) >= 2:
                            # Perform local stats
                            sig_symbols = {0.001: '***', 0.01: '**', 0.05: '*', 1.0: 'ns'}
                            
                            if len(group_data_list) == 2:
                                # T-test for 2 groups
                                t_stat, p_val = ttest_ind(group_data_list[0], group_data_list[1], equal_var=False)
                                
                                sig = 'ns'
                                for thresh, symb in sorted(sig_symbols.items()):
                                    if p_val <= thresh: sig = symb; break
                                
                                # Use same logic for positions regardless of significance level
                                # but only draw the bar if it's NOT 'ns' (unless the user wants 'ns' bars too)
                                # Currently we only append to unit_significance if it's NOT 'ns'
                                if sig != 'ns':
                                    pos1 = unit_group_positions[unit][group_labels[0]]
                                    pos2 = unit_group_positions[unit][group_labels[1]]
                                    unit_significance.append((pos1, pos2, sig))
                            else:
                                # ANOVA + Tukey for >2 groups
                                try:
                                    f_stat, p_val = f_oneway(*group_data_list)
                                    if p_val < 0.05:
                                        # Flatten for Tukey
                                        tukey_records = []
                                        for k, g_vals in enumerate(group_data_list):
                                            for val in g_vals:
                                                tukey_records.append({'G': group_labels[k], 'V': val})
                                        tk_df = pd.DataFrame(tukey_records)
                                        tk_res = pairwise_tukeyhsd(tk_df['V'], tk_df['G'])
                                        
                                        # Add all significant pairs
                                        for row in tk_res.summary().data[1:]:
                                            g1, g2, p_adj = row[0], row[1], row[3]
                                            if p_adj < 0.05:
                                                sig = 'ns'
                                                for thresh, symb in sorted(sig_symbols.items()):
                                                    if p_adj <= thresh: sig = symb; break
                                                p1 = unit_group_positions[unit][str(g1)]
                                                p2 = unit_group_positions[unit][str(g2)]
                                                unit_significance.append((p1, p2, sig))
                                                print(f"    SIGNIFICANT: {g1} vs {g2} in {unit}: p_adj={p_adj:.4f} ({sig})")
                                except: pass
                    
                    current_pos += 1.0  # Gap between units
                else:
                    # Simple unit-based plot
                    # Remove outliers using shared IQR method for both bar heights and statistical tests
                    unit_data_filtered = self._apply_outlier_filtration(unit_data, value_col, ['Plot_Unit'])
                    filtered_vals = unit_data_filtered[value_col].values

                    if len(filtered_vals) > 0:
                        bar_means.append(np.mean(filtered_vals))
                        bar_stderrs.append(np.std(filtered_vals) / np.sqrt(len(filtered_vals)))
                    else:
                        raw_vals = unit_data[value_col].values
                        if len(raw_vals) > 0:
                            bar_means.append(np.mean(raw_vals))
                            bar_stderrs.append(np.std(raw_vals) / np.sqrt(len(raw_vals)))
                        else:
                            bar_means.append(0)
                            bar_stderrs.append(0)
                        
                    positions.append(current_pos)
                    # Use unit_color_map if provided
                    base_unit = unit.split(' (')[0] if ' (' in unit else unit
                    color = unit_color_map.get(base_unit, '#888888') if unit_color_map else '#888888'
                    colors.append(color)
                    
                    unit_tick_positions.append(current_pos)
                    current_pos += 1.0

                # Map display name for unit
                display_unit = unit
                if unit_display_map:
                    base_unit = unit.split(' (')[0] if ' (' in unit else unit
                    if base_unit in unit_display_map:
                        mapped = unit_display_map[base_unit]
                        display_unit = unit.replace(base_unit, mapped)
                
                
                unit_tick_labels.append(display_unit)

            if not bar_means: continue

            # Plotting
            fig_height = 6
            fig_width = 4

            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            
            bars = ax.bar(positions, bar_means, width=0.5, yerr=bar_stderrs, 
                        error_kw=dict(linestyle='-', elinewidth=1.5, capsize=5, capthick=1.5),
                        color=colors, alpha=0.8, edgecolor='black', linewidth=2)
            
            # Determine which display names have duplicates across units_to_display (if provided)
            # This MUST be done BEFORE label cleaning so we can match against full display names
            display_names_with_duplicates = set()
            
            if units_to_display is not None and unit_display_map:
                # Map each base unit to its display name
                unit_to_display_name = {}
                for base_unit in units_to_display:
                    if base_unit in unit_display_map:
                        unit_to_display_name[base_unit] = unit_display_map[base_unit]
                    else:
                        unit_to_display_name[base_unit] = base_unit
                
                # Count how many base units map to each display name
                from collections import Counter
                display_name_counts = Counter(unit_to_display_name.values())
                display_names_with_duplicates = {name for name, count in display_name_counts.items() if count > 1}
                
            # Add (N) suffixes to labels that have duplicates (BEFORE cleaning labels)
            if display_names_with_duplicates:
                # Directly check if label is in the duplicated names set
                label_indices = {name: 0 for name in display_names_with_duplicates}
                deduplicated_labels = []
                for label in unit_tick_labels:
                    if label in label_indices:
                        label_indices[label] += 1
                        deduplicated_labels.append(f"{label} (N{label_indices[label]})")
                    else:
                        deduplicated_labels.append(label)
                unit_tick_labels = deduplicated_labels
            else:
                # Also check for duplicates within this plot as a fallback
                from collections import Counter
                label_counts = Counter(unit_tick_labels)
                duplicate_labels = {label for label in label_counts if label_counts[label] > 1}
                
                if duplicate_labels:
                    label_indices = {label: 0 for label in duplicate_labels}
                    deduplicated_labels = []
                    for label in unit_tick_labels:
                        if label in label_indices:
                            label_indices[label] += 1
                            deduplicated_labels.append(f"{label} (N{label_indices[label]})")
                        else:
                            deduplicated_labels.append(label)
                    unit_tick_labels = deduplicated_labels
            
            # Clean up labels: extract display name if pipe-delimited
            # cleaned_labels = []
            # for label in unit_tick_labels:
            #     # If label contains "|", take only the part after it
            #     if "|" in label:
            #         label = label.split("|", 1)[1].strip()
            #     cleaned_labels.append(label)
            # unit_tick_labels = cleaned_labels

            # Labels and styling
            ax.set_xticks(unit_tick_positions)
            ax.set_xticklabels(unit_tick_labels, rotation=90, ha='center', fontsize=12)
            
            display_item = display_name_map.get(item_name, item_name) if display_name_map else item_name
            y_label = 'Relative Percentage (%)' if data_type == 'percentage' else 'Mean Ratio'
            ax.set_ylabel(y_label, fontweight='bold', fontsize=14)
            
            title_suffix = f" by {compare_by.replace('_', ' ')}" if compare_by else ""
            ax.set_title(f"{display_item} Distribution{title_suffix}", fontweight='bold', fontsize=16, pad=15)
            
            # Legend if comparing
            if compare_by:
                # Use the same all_item_groups as used for the bars for legend consistency
                legend_labels = [str(g).lower().capitalize() for g in all_item_groups]
                
                legend_patches = []
                for i, label in enumerate(legend_labels):

                    # Match label back to group_colors if possible
                    color_found = False
                    if compare_by == 'Group_ID':
                        for k, v in group_colors.items():
                            if k.upper() == label:
                                legend_patches.append(mpatches.Patch(color=v, label=label))
                                color_found = True
                                break
                    
                    if not color_found:
                        legend_patches.append(mpatches.Patch(color=comparison_palette[i % len(comparison_palette)], label=label))
                
                # Only save standalone legend once per call
                legend_path = os.path.join(output_dir, "comparison_legend.png")
                if not os.path.exists(legend_path):
                    self._save_standalone_legend(legend_patches, "Comparison Groups", legend_path)

            # Note: We will draw all significance markers (intra-unit and inter-unit) at the end
            # to ensure proper vertical stacking.
            
            # Tukey HSD significance testing across units (only when not comparing subgroups to avoid clutter)
            if not compare_by and len(all_plot_units) > 1:
                # Prepare data for Tukey HSD
                item_data_filtered = self._apply_outlier_filtration(item_data, value_col, ['Plot_Unit'])
                records = []
                for _, row in item_data_filtered.iterrows():
                    records.append({'Unit': row['Plot_Unit'], 'Value': row[value_col]})
                
                if records and len(item_data_filtered['Plot_Unit'].unique()) > 1:
                    tukey_df = pd.DataFrame(records)
                    try:
                        tukey_result = pairwise_tukeyhsd(endog=tukey_df['Value'], groups=tukey_df['Unit'], alpha=0.05)
                        
                        sig_symbols = {0.001: '***', 0.01: '**', 0.05: '*', 1.0: 'ns'}
                        for row in tukey_result.summary().data[1:]:
                            unit1, unit2, p_adj = str(row[0]), str(row[1]), float(row[3])
                            if p_adj < 0.05:
                                sig = 'ns'
                                for thresh, symb in sorted(sig_symbols.items()):
                                    if p_adj <= thresh: sig = symb; break
                                
                                # Use centers for unit positions
                                p1 = unit_tick_positions[all_plot_units.index(unit1)]
                                p2 = unit_tick_positions[all_plot_units.index(unit2)]
                                unit_significance.append((p1, p2, sig))
                    except Exception as e:
                        print(f"    Warning: Tukey HSD failed for {item_name}: {e}")
            
            # --- Draw ALL significance markers (Intra and Inter) ---
            if unit_significance:
                y_max = ax.get_ylim()[1]
                line_height = 0.08  # Increased from 0.06 for better symbol clearance
                # Sort by span (smaller spans first) then position
                unit_significance.sort(key=lambda x: (abs(x[0]-x[1]), min(x[0], x[1])))
                
                # Adaptive stacking: Track occupied Y-ranges per X-interval
                # We'll use a simple greedy stacking
                drawn_markers = [] # List of (x_range, y_level)
                
                for x1, x2, sig in unit_significance:
                    if x1 > x2: x1, x2 = x2, x1
                    
                    # Find first available y_level (above bars + padding)
                    level = 0
                    while True:
                        collision = False
                        for (mx1, mx2), mlevel in drawn_markers:
                            if mlevel == level:
                                # Check if intervals overlap
                                if not (x2 + 0.1 < mx1 or x1 - 0.1 > mx2):
                                    collision = True
                                    break
                        if not collision:
                            break
                        level += 1
                    
                    # Increased base gap from 1.05 to 1.10
                    y_pos = y_max * (1.10 + level * line_height)
                    ax.plot([x1, x2], [y_pos, y_pos], 'k-', linewidth=1.0)
                    ax.text((x1 + x2) / 2, y_pos, sig, ha='center', va='bottom', fontsize=10, fontweight='bold')
                    drawn_markers.append(((x1, x2), level))
                
                # Expand Y-limit properly for highest level
                max_level = max([m[1] for m in drawn_markers]) if drawn_markers else 0
                ax.set_ylim(top=y_max * (1.20 + max_level * line_height))
            
            # Grid and spines
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            for spine in ax.spines.values(): spine.set_edgecolor('#cccccc')
            
            plt.tight_layout()
            # Save
            item_safe = item_name.replace(':', '').replace('/', '_').replace(' ', '_')
            if compare_by:
                item_safe += f"_by_{compare_by}"

            save_path = os.path.join(output_dir, f"{data_type}_{item_safe}_barplot.png")
            # print(f" Generated individual bar plot in {save_path}")

            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            if show_plots: plt.show()
            else: plt.close()
