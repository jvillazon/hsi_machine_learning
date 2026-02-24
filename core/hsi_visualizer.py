import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tifffile
from glob import glob
import os
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.ndimage as ndi

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
            ax_prob.set_yticks(range(len(sorted_probs)))
            ax_prob.set_yticklabels(sorted_names, fontsize=9)
            ax_prob.set_xlabel('Probability', fontsize=10)
            ax_prob.set_title('Top 5 Predictions', fontsize=11, fontweight='bold')
            ax_prob.set_xlim([0, 1])
            ax_prob.grid(True, alpha=0.3, axis='x')

    
    def visualize_class_spectra(self, predictions, img_path=None, dataset=None, class_name=None, class_idx=None, 
                                 max_spectra=None, show_individual=True, show_reference=True):
        """
        Visualize all spectra predicted as a specific class, showing mean ± std and optionally individual spectra.
        
        Args:
            predictions: Either:
                - DataFrame loaded from prediction CSV (with molecule names in cells)
                - Array/list of predictions (indices or class names)
            img_path: Path to the original .tif image file (required if predictions is a DataFrame)
            dataset: Dataset object (required if predictions is array/list, not needed for DataFrame)
            class_name: Name of the class to visualize (e.g., 'Cholesterol')
            class_idx: Index of the class to visualize (alternative to class_name)
            max_spectra: Maximum number of individual spectra to plot (default None = all, can be slow)
            show_individual: Whether to show individual spectra (default True)
            show_reference: Whether to show reference molecule spectrum (default True)
            
        Returns:
            dict: Statistics including mean_spectrum, std_spectrum, num_spectra
        """
        
        # Determine which class to visualize
        if class_name is not None:
            if class_name not in self.molecule_names:
                raise ValueError(f"Class '{class_name}' not found in molecule names: {self.molecule_names}")
            target_class_idx = np.where(self.molecule_names == class_name)[0][0]
            target_class_name = class_name
        elif class_idx is not None:
            target_class_idx = class_idx
            target_class_name = self.molecule_names[class_idx] if class_idx < len(self.molecule_names) else f"Class {class_idx}"
        else:
            raise ValueError("Must provide either class_name or class_idx")
        
        # Handle DataFrame input (from CSV)
        if isinstance(predictions, pd.DataFrame):
            if dataset is None:
                raise ValueError("dataset must be provided when predictions is a DataFrame")
            
            # Get predictions as numpy array
            pred_matrix = predictions.values
            height, width = pred_matrix.shape
            
            # Find all pixels predicted as target class
            if pred_matrix.dtype == object or isinstance(pred_matrix[0, 0], str):
                # String predictions (molecule names)
                class_mask = (pred_matrix == target_class_name)
            else:
                # Numeric predictions
                class_mask = (pred_matrix == target_class_idx)
            
            # Get row and column indices where class was predicted
            row_indices, col_indices = np.where(class_mask)
            num_spectra = len(row_indices)
            
            print(f"Found {num_spectra} pixels predicted as '{target_class_name}'")
            
            if num_spectra == 0:
                print("No spectra found for this class!")
                return None
            
            # Limit number of spectra if specified
            if max_spectra is not None and num_spectra > max_spectra:
                print(f"Limiting to {max_spectra} random spectra for visualization")
                sample_idx = np.random.choice(num_spectra, size=max_spectra, replace=False)
                row_indices = row_indices[sample_idx]
                col_indices = col_indices[sample_idx]
            
            # Extract spectra using dataset's __getitem__ method
            print("Extracting spectra from dataset...")
            all_spectra = []
            
            # Need to determine which image index this corresponds to
            # Assuming predictions CSV is for a single image, find the image index
            if img_path is not None:
                # Find the image index in dataset
                img_idx = None
                for i, path in enumerate(dataset.img_list):
                    if os.path.basename(path) == os.path.basename(img_path) or path == img_path:
                        img_idx = i
                        break
                
                if img_idx is None:
                    raise ValueError(f"Image path {img_path} not found in dataset")
                
                # Get the starting index for this image in the dataset
                img_start_idx = dataset.img_size[img_idx] if img_idx > 0 else 0
                
                # Convert 2D positions to dataset indices
                for row, col in zip(row_indices, col_indices):
                    # Convert 2D position to 1D offset within image
                    pixel_offset = row * width + col
                    # Get global dataset index
                    global_idx = img_start_idx + pixel_offset
                    
                    try:
                        spectrum, _ = dataset[int(global_idx)]
                        # Convert to numpy if needed
                        if hasattr(spectrum, 'numpy'):
                            spectrum = spectrum.numpy()
                        all_spectra.append(spectrum)
                    except Exception as e:
                        print(f"Warning: Could not load spectrum at index {global_idx}: {e}")
                        continue
            else:
                raise ValueError("img_path must be provided when using DataFrame predictions")
        
        # Handle array/list input (from model predictions)
        else:
            if dataset is None:
                raise ValueError("dataset must be provided when predictions is an array/list")
            
            # Flatten predictions if it's a list of arrays (per-image predictions)
            if isinstance(predictions, list):
                predictions_flat = np.concatenate([np.array(p) for p in predictions])
            else:
                predictions_flat = np.array(predictions)
            
            # Convert string predictions to indices if needed
            if predictions_flat.dtype == object or isinstance(predictions_flat[0], str):
                # Map molecule names to indices
                pred_indices = np.zeros(len(predictions_flat), dtype=int)
                for i, mol_name in enumerate(self.molecule_names):
                    pred_indices[predictions_flat == mol_name] = i
                predictions_flat = pred_indices
            
            # Find all indices where this class was predicted
            class_mask = predictions_flat == target_class_idx
            class_indices = np.where(class_mask)[0]
            
            num_spectra = len(class_indices)
            print(f"\nFound {num_spectra} spectra predicted as '{target_class_name}' (class {target_class_idx})")
            
            if num_spectra == 0:
                print("No spectra found for this class!")
                return None
            
            # Limit number of spectra if specified
            if max_spectra is not None and num_spectra > max_spectra:
                print(f"Limiting to {max_spectra} random spectra for visualization")
                sample_indices = np.random.choice(class_indices, size=max_spectra, replace=False)
            else:
                sample_indices = class_indices
            
            # Collect all spectra for this class
            print("Collecting spectra...")
            all_spectra = []
            for idx in sample_indices:
                try:
                    spectrum, _ = dataset[int(idx)]
                    # Convert to numpy if needed
                    if hasattr(spectrum, 'numpy'):
                        spectrum = spectrum.numpy()
                    all_spectra.append(spectrum)
                except Exception as e:
                    print(f"Warning: Could not load spectrum at index {idx}: {e}")
                    continue
        
        if len(all_spectra) == 0:
            print("Could not load any spectra!")
            return None
        
        # Stack spectra and compute statistics
        spectra_array = np.array(all_spectra)
        mean_spectrum = np.mean(spectra_array, axis=0)
        std_spectrum = np.std(spectra_array, axis=0)
        
        # Create wavenumber axis
        wavenumber = np.linspace(self.wavenumber_start, self.wavenumber_end, self.num_samples)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot individual spectra if requested
        if show_individual:
            for i, spectrum in enumerate(all_spectra):
                ax.plot(wavenumber, spectrum, alpha=0.1, color='blue', linewidth=0.5)
        
        # Plot mean ± std
        ax.plot(wavenumber, mean_spectrum, color='darkblue', linewidth=2, label=f'Mean (n={len(all_spectra)})')
        ax.fill_between(wavenumber, mean_spectrum - std_spectrum, mean_spectrum + std_spectrum, 
                        color='blue', alpha=0.3, label='±1 Std Dev')
        
        # Plot reference molecule spectrum if requested
        if show_reference and target_class_idx < len(self.normalized_molecules):
            ref_spectrum = self.normalized_molecules[target_class_idx]
            # Scale reference to match mean spectrum max for comparison
            ref_max = np.max(ref_spectrum)
            mean_max = np.max(mean_spectrum)
            scaled_ref = ref_spectrum * (mean_max / ref_max) if ref_max > 0 else ref_spectrum
            ax.plot(wavenumber, scaled_ref, '--', color='red', linewidth=2, 
                   label=f'Reference: {target_class_name}', alpha=0.7)
        
        ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=12)
        ax.set_ylabel("Normalized Intensity", fontsize=12)
        ax.set_title(f"Spectra Predicted as '{target_class_name}' (n={len(all_spectra)})", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        # plt.show()
        
        # Return statistics
        stats = {
            'class_name': target_class_name,
            'class_idx': target_class_idx,
            'num_spectra': len(all_spectra),
            'mean_spectrum': mean_spectrum,
            'std_spectrum': std_spectrum,
            'all_spectra': spectra_array
        }
        
        return stats

    
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

    def show_random_predictions(self, dataset, predictions, probabilities, num_images=3, spectra_per_image=2, exclude_classes=None):
        """
        Display random predictions from different images in a single subplot
        
        Args:
            predictions: Array of predictions from predict()
            probabilities: Array of probabilities from predict()
            num_images: Number of images to sample from
            spectra_per_image: Number of spectra to show per image
            exclude_classes: List of class names to exclude from visualization (e.g., ['No Match'])
        """

        num_images = min(num_images, len(dataset.img_list))
        total_spectra = num_images * spectra_per_image
        
        # Create 2-column subplot: spectra on left, probabilities on right
        fig, axes = plt.subplots(total_spectra, 2, figsize=(12, 3 * total_spectra))
        if total_spectra == 1:
            axes = axes.reshape(1, 2)
        
        plot_idx = 0
        for img_idx in range(num_images):
            # Find valid index range for this image
            start_idx = dataset.img_size[img_idx]
            end_idx = dataset.img_size[img_idx + 1]
            img_size = end_idx - start_idx
            
            # Sample random indices, filtering out excluded classes if specified
            sampled_count = 0
            max_attempts = 1000
            attempts = 0
            random_indices = []
            
            while sampled_count < spectra_per_image and attempts < max_attempts:
                idx = start_idx + np.random.randint(0, img_size)
                
                # Check if this spectrum should be excluded
                if exclude_classes is not None:
                    pred_class = predictions[idx]
                    mol_name = self.molecule_names[pred_class]
                    if mol_name in exclude_classes:
                        attempts += 1
                        continue
                
                random_indices.append(idx)
                sampled_count += 1
                attempts += 1
            
            # Show spectra with predictions
            for idx in random_indices:
                spectra, _ = dataset[idx]
                self.visualize_spectrum_in_axes(
                    ax_spectrum=axes[plot_idx, 0],
                    ax_prob=axes[plot_idx, 1],
                    spectrum=spectra,
                    prediction=predictions[idx],
                    probabilities=probabilities[idx],
                    img_idx=img_idx
                )
                plot_idx += 1
        
        plt.tight_layout()
        plt.show()

    def apply_rf_masking(self, prediction_csv_path=None, ratio_tiff_path=None, mask_list_path=None, results_per_unit=None, 
                         stats=None, regions=None, img_name=None, sample_name=None, classes_to_ignore=None):
        """
        Apply masks and quantify predictions or ratios for each instance.
        
        Args:
            prediction_csv_path: Path to prediction CSV file (for prediction quantification)
            ratio_tiff_path: Path to ratio TIFF file (for ratio quantification)
            mask_list_path: Path to folder containing mask TIFF files
            stats: Dictionary of image statistics
            regions: List of region names for classification
            img_name: Image name (without extension)
            sample_name: Sample name
            classes_to_ignore: List of class names to ignore in quantification (default: ['Masked', 'Kidney Background', 'No Match'])
            
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
            raise ValueError("No mask TIFF files found in the specified directory")
        
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
            
            # Extract unit name from mask filename (e.g., "CODEX-Glom_cortex-mask.tif" -> "Glom_cortex")
            mask_basename = os.path.basename(mask_file)
            unit_name = mask_basename.replace('CODEX-', '').replace('-mask.tif', '')
            
            # Check shape compatibility
            if is_prediction_mode:
                # Matrix shape should match mask shape
                if data_matrix.shape != mask_img.shape:
                    print(f"Skipping {mask_basename}: shape mismatch (data: {data_matrix.shape}, mask: {mask_img.shape})")
                    continue
            else:
                if data_img.shape != mask_img.shape:
                    print(f"Skipping {mask_basename}: shape mismatch (data: {data_img.shape}, mask: {mask_img.shape})")
                    continue
            
            # Label mask for instance segmentation
            labeled_mask, num_features = ndi.label(mask_img > 0)
            
            # Minimum instance size threshold (in pixels)
            min_instance_size = 250
            
            # Determine region from unit_name
            region = 'Other'
            if regions is not None:
                unit_parts = unit_name.split('_')
                if len(unit_parts) >= 2:
                    second_part = unit_parts[-1]
                    for region_name in regions:
                        if region_name.lower().startswith(second_part.lower()) or second_part.lower().startswith(region_name.lower()[:4]):
                            region = region_name
                            break
            
            # Process each instance
            for label in range(1, labeled_mask.max() + 1):
                instance_mask = labeled_mask == label
                
                # Check size before dilation
                instance_size = np.sum(instance_mask)
                if instance_size < min_instance_size:
                    continue
                
                # Apply dilation
                instance_mask = ndi.binary_dilation(instance_mask, iterations=1)
                
                # Check size after dilation
                if np.sum(instance_mask) < min_instance_size:
                    continue
                
                if is_prediction_mode:
                    # Quantify predictions - extract class names at mask positions
                    predictions = data_matrix[instance_mask]
                    
                    # Check if we have valid predictions
                    if len(predictions) == 0:
                        continue
                    
                    # Count occurrences of each class
                    unique_classes, counts = np.unique(predictions, return_counts=True)
                    class_counts = dict(zip(unique_classes, counts))
                    
                    # Remove ignored classes
                    for cls in classes_to_ignore:
                        if cls in class_counts:
                            del class_counts[cls]

                    total = sum(class_counts.values())
                    if total > 0:
                        pct = {k: (v / total * 100.0) for k, v in class_counts.items()}
                    else:
                        pct = {k: 0.0 for k in class_counts.keys()}
                    
                    # Store prediction data
                    data_entry = {
                        'percentages': pct,
                        'image_name': img_name,
                        'sample_name': sample_name if sample_name else img_name.split('-')[0],
                        'instance_label': f"{int(label)}",
                        'region': region
                    }
                    
                    if unit_name not in results_per_unit:
                        results_per_unit[unit_name] = [data_entry]
                    else:
                        results_per_unit[unit_name].append(data_entry)
                
                else:
                    # Quantify ratios
                    masked_ratio = data_img[instance_mask]
                    
                    # Calculate mean ratio (excluding zeros)
                    non_zero_ratios = masked_ratio[masked_ratio > 0]
                    if len(non_zero_ratios) > 0:
                        mean_ratio = np.mean(non_zero_ratios)
                        
                        # Store ratio data
                        ratio_entry = {
                            'ratios': {ratio_type: mean_ratio},
                            'image_name': img_name,
                            'sample_name': sample_name if sample_name else img_name.split('-')[0],
                            'instance_label': f"{int(label)}",
                            'region': region
                        }
                        
                        if unit_name not in results_per_unit:
                            results_per_unit[unit_name] = [ratio_entry]
                        else:
                            results_per_unit[unit_name].append(ratio_entry)
                    # If no non-zero ratios, skip this instance entirely

        
        return results_per_unit
    
    def mask_tiff_by_classes(self, tiff_path, csv_path, classes_to_mask, mask_value=0):
        """
        Mask out specific classes in a TIFF image based on predictions from a CSV file.
        Pixels belonging to specified classes will be set to the mask_value.
        
        Args:
            tiff_path: Path to the input TIFF file (can be 2D or hyperspectral)
            csv_path: Path to the prediction CSV file
            classes_to_mask: List of class names (molecule names) to mask out
            output_path: Path to save the masked TIFF (if None, adds '_masked' to original filename)
            mask_value: Value to set for masked pixels (default 0)
            
        Returns:
            masked_image: The masked TIFF image as a numpy array
        """
        
        # Load the TIFF image
        image = tifffile.imread(tiff_path)
        
        # Load the prediction CSV
        pred_df = pd.read_csv(csv_path)
        pred_matrix = pred_df.values
        
        # Determine if image is 2D or 3D (hyperspectral)
        if image.ndim == 2:
            # 2D image (height, width)
            if pred_matrix.shape != image.shape:
                raise ValueError(f"Prediction matrix shape {pred_matrix.shape} doesn't match image shape {image.shape}")
            is_hyperspectral = False
        elif image.ndim == 3:
            # 3D hyperspectral image (channels, height, width)
            if pred_matrix.shape != image.shape[1:]:
                raise ValueError(f"Prediction matrix shape {pred_matrix.shape} doesn't match image spatial dimensions {image.shape[1:]}")
            is_hyperspectral = True
        else:
            raise ValueError(f"Unsupported image dimensions: {image.ndim}. Expected 2 or 3.")
        
        # Create a binary mask for pixels to be masked out
        mask = np.zeros(pred_matrix.shape, dtype=bool)
        
        for class_name in classes_to_mask:
            # Find pixels belonging to this class
            class_mask = (pred_matrix == class_name)
            
            # Add to overall mask
            mask = mask | class_mask
        
        # Apply mask to image
        # Create a copy to avoid modifying original
        masked_image = image.copy()
        
        # Mask pixels based on image type
        if is_hyperspectral:
            # Mask all channels for the selected pixels
            masked_image[:, mask] = mask_value
        else:
            # Mask 2D image directly
            masked_image[mask] = mask_value
        
        return masked_image


    def quantify_predictions(self, input_csv_path, classes_to_ignore=None):
        """
        Quantify the number of pixels per class in a prediction CSV file.
        
        Args:
            input_csv_path: Path to the prediction CSV file
            
        Returns:
            class_counts: Dictionary with class names as keys and pixel counts as values
        """
        
        # Load masked prediction CSV
        pred_df = pd.read_csv(input_csv_path)
        
        pred_array = pred_df.to_numpy()
        
        # Count occurrences of each class
        unique, counts = np.unique(pred_array, return_counts=True)
        class_counts = dict(zip(unique, counts))

        if classes_to_ignore is not None:
            for cls in classes_to_ignore:
                if cls in class_counts:
                    del class_counts[cls]
        
        return class_counts

    def quantify_unit_class_percentages_nested(self, units_dict, unit_mappings=None):
        """
        Calculate replicate-level percentages for all classes in all units.

        Input shape:
            units_dict: {
                unit_name: [
                    {'counts': {class_name: count, ...}, 'image_name': str, 'sample_name': str, 'instance_label': str, 'region': str},  # replicate 0
                    {'counts': {class_name: count, ...}, 'image_name': str, 'sample_name': str, 'instance_label': str, 'region': str},  # replicate 1
                    ...
                ],
                ...
            }
            unit_mappings: Optional dict mapping abbreviated unit names to full names

        Returns:
            DataFrame with columns [Unit, Molecule, Percentage, Replicate, Image_Name, Sample_Name, Instance_Label, Region]
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
                sample_name = pct_entry['sample_name']
                instance_label = pct_entry['instance_label']
                region = pct_entry.get('region', None)
                
                for molecule, percentage in pct_dict.items():
                    records.append({
                        'Unit': base_unit,
                        'Molecule': molecule,
                        'Percentage': percentage,
                        'Replicate': rep_idx + 1,
                        'Image_Name': image_name,
                        'Sample_Name': sample_name,
                        'Instance_Label': instance_label,
                        'Region': region
                    })
        df = pd.DataFrame.from_records(records)
        return df

    def quantify_ratio_tiff(self, tiff_path, classes_to_ignore=None):
        """
        Calculate the mean ratio value for non-zero pixels in a masked ratio TIFF file.
        
        Args:
            tiff_path: Path to the masked ratio TIFF file
            classes_to_ignore: Not used here but kept for API consistency
            
        Returns:
            mean_ratio: Mean value of non-zero pixels, or NaN if no valid pixels
        """
        # Load the ratio TIFF
        ratio_img = tifffile.imread(tiff_path)
        
        # Create mask for non-zero (non-masked) pixels
        non_zero_mask = ratio_img > 0
        
        
        # Calculate mean of non-zero pixels
        if np.sum(non_zero_mask) > 0:
            mean_ratio = np.mean(ratio_img[non_zero_mask])
        else:
            mean_ratio = np.nan
        
        return mean_ratio

    def quantify_unit_ratio_means_nested(self, units_dict, unit_mappings=None):
        """
        Calculate replicate-level mean ratios for all ratio types in all units.

        Input shape:
            units_dict: {
                unit_name: [
                    {'ratios': {ratio_type: mean_value, ...}, 'image_name': str, 'sample_name': str, 'instance_label': str, 'region': str},  # replicate 0
                    {'ratios': {ratio_type: mean_value, ...}, 'image_name': str, 'sample_name': str, 'instance_label': str, 'region': str},  # replicate 1
                    ...
                ],
                ...
            }
            unit_mappings: Optional dict mapping abbreviated unit names to full names

        Returns:
            DataFrame with columns [Unit, Ratio_Type, Mean_Ratio, Replicate, Image_Name, Sample_Name, Instance_Label, Region]
        """
        records = []
        for unit_name, ratio_list in units_dict.items():
            # Be tolerant if old format (list of dicts with ratio values) is provided
            if isinstance(ratio_list, dict):
                # Old format: single dict of ratios
                ratio_list = [{'ratios': ratio_list, 'image_name': 'Unknown', 'sample_name': 'Unknown', 'instance_label': '0', 'region': None}]
            
            for rep_idx, ratio_entry in enumerate(ratio_list):
                # Check if new format (dict with 'ratios', 'image_name', 'sample_name', 'instance_label', 'region')
                if isinstance(ratio_entry, dict) and 'ratios' in ratio_entry:
                    ratio_dict = ratio_entry['ratios']
                    image_name = ratio_entry.get('image_name', 'Unknown')
                    sample_name = ratio_entry.get('sample_name', 'Unknown')
                    instance_label = ratio_entry.get('instance_label', '0')
                    region = ratio_entry.get('region', None)
                else:
                    # Old format: just a dict of ratio values
                    ratio_dict = ratio_entry
                    image_name = 'Unknown'
                    sample_name = 'Unknown'
                    instance_label = '0'
                    region = None
                
                # Extract base unit name (first part before underscore)
                base_unit = unit_name.split('_')[0] if '_' in unit_name else unit_name
                
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
                        'Sample_Name': sample_name,
                        'Instance_Label': instance_label,
                        'Region': region
                    })
        
        df = pd.DataFrame.from_records(records)
        return df
    
    def prediction_csv_to_rgb_tiff(self, csv_path, stats, output_path=None, use_molecule_colors=True, 
                                     molecules_to_zero=None):
        """
        Convert a prediction CSV to an RGB TIFF where each classifier class 
        gets a consistent predefined color.
        
        Args:
            csv_path: Path to the prediction CSV file
            stats: Dictionary containing image statistics (e.g., pixel size)
            output_path: Path to save the RGB TIFF (if None, replaces .csv with _rgb.tif)
            use_molecule_colors: If True, use consistent color palette; if False, use random colors
            molecules_to_zero: List of molecule names to make black (zero out).
            
        Returns:
            None (saves RGB TIFF to disk)
        """
        
        # Load the CSV - it's a matrix with molecule names
        df = pd.read_csv(csv_path)
        pred_matrix = df.values
        height, width = pred_matrix.shape
        
        # Get all possible classes from the classifier (using molecule_names from visualizer)
        if self.molecule_names is not None:
            all_classes = [str(name) for name in self.molecule_names]
        else:
            # Fallback to unique values in the prediction matrix
            all_classes = sorted(np.unique(pred_matrix).tolist())
        
        n_classes = len(all_classes)
        
        # Create color map for ALL possible classes (not just those present in this image)
        if use_molecule_colors:
            # Use a perceptually distinct color palette
            colors = sns.color_palette("husl", n_classes)
            # Convert from 0-1 range to 0-255 range
            colors_255 = [(int(r*255), int(g*255), int(b*255)) for r, g, b in colors]
        else:
            # Generate random colors
            np.random.seed(42)  # For reproducibility
            colors_255 = [(np.random.randint(0, 256), 
                          np.random.randint(0, 256), 
                          np.random.randint(0, 256)) for _ in range(n_classes)]
        
        # Create mapping from class name to color
        color_map = {}
        molecules_to_zero = molecules_to_zero or []
        
        for i, class_name in enumerate(all_classes):
            if class_name in molecules_to_zero:
                # Set to black
                color_map[class_name] = (0, 0, 0)
            # elif class_name.endswith('Background'):
            #     # Set to light gray
            #     color_map[class_name] = (200, 200, 200)
            else:
                color_map[class_name] = colors_255[i]
        
        # Create RGB image (height, width, 3)
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Assign colors to pixels based on their predicted class
        for class_name, color in color_map.items():
            mask = pred_matrix == class_name
            rgb_image[mask] = color

        
        # Save as TIFF
        if output_path is None:
            output_path = csv_path.replace('.csv', '_rgb.tif')
        
        tifffile.imwrite(
            output_path, 
            rgb_image, 
            photometric='rgb',
            imagej=True,
            metadata={
                'axes': 'CYX',
                'unit': 'um',
                'PhysicalSizeX': stats['pixel_size_x'],
                'PhysicalSizeXUnit': 'um',
                'PhysicalSizeY': stats['pixel_size_y'],
                'PhysicalSizeYUnit': 'um',
            }
        )
    
    def one_way_anova_unit_comparison(self, df, value_col, grouping_col, output_dir=None,
                                      data_type='percentage', figsize=(5, 5),
                                      show_plots=True, display_name_map=None, unit_color_map=None, unit_display_map=None):
        """
        Perform one-way ANOVA comparing units for each molecule/ratio type,
        followed by Tukey HSD post-hoc test. Creates bar plots with significance bars.
        
        Args:
            df: DataFrame with columns [Unit, {Molecule or Ratio_Type}, {Percentage or Mean_Ratio}, 
                                        Replicate, Image_Name, Sample_Name]
            display_name_map: Optional dict mapping original names to display names for plots
            unit_color_map: Optional dict mapping unit names to hex color codes (e.g., {'U1': '#1f77b4'})
            value_col: Name of the value column ('Percentage' or 'Mean_Ratio')
            grouping_col: Name of the grouping column ('Molecule' or 'Ratio_Type')
            output_dir: Directory to save plots (if None, uses current directory)
            data_type: Type of data ('percentage' or 'ratio') for labeling
            figsize: Figure size for plots (width, height)
            show_plots: Whether to display plots interactively
            
        Returns:
            anova_results: Dictionary with ANOVA and Tukey HSD results for each molecule/ratio type
        """
        from scipy import stats
        
        # Create output directory if needed
        if output_dir is None:
            output_dir = 'anova_unit_comparison'
        os.makedirs(output_dir, exist_ok=True)
        
        # Get unique molecules/ratio types
        unique_groups = df[grouping_col].unique()
        
        # Sort by display_name_map order if provided, otherwise alphabetically
        if display_name_map:
            # Create ordered list: first items in display_name_map order, then remaining alphabetically
            ordered_groups = [k for k in display_name_map.keys() if k in unique_groups]
            remaining = sorted([g for g in unique_groups if g not in display_name_map.keys()])
            sorted_groups = ordered_groups + remaining
        else:
            sorted_groups = sorted(unique_groups)
        
        anova_results = {}
        significant_molecules = []  # Track which molecules have significance
         
        print("\nGenerating box plots...")
        
        for group_value in sorted_groups:
            # Filter data for this molecule/ratio type
            group_data = df[df[grouping_col] == group_value].copy()
            
            # Get unique units
            units = sorted(group_data['Unit'].unique())
            
            # Skip if only one unit
            if len(units) < 2:
                continue

            # Remove value if inf or NaN
            group_data.replace([np.inf, -np.inf], np.nan, inplace=True)
            group_data = group_data.dropna(subset=[value_col])
            
            # Prepare data for ANOVA and skip if any group is empty
            unit_groups = [group_data[group_data['Unit'] == unit][value_col].values 
                          for unit in units if len(group_data[group_data['Unit'] == unit]) > 0]
            
            # # Skip if any group is empty
            # if any(len(g) == 0 for g in unit_groups):
            #     continue
            
            # # Check for NaN or infinite values
            # if any(np.any(np.isnan(g)) or np.any(np.isinf(g)) for g in unit_groups):
            #     print(f"Warning: {group_value} contains NaN or infinite values, skipping...")
            #     continue
            
            # Check if all values are identical (no variance)
            all_values = np.concatenate(unit_groups)
            if len(np.unique(all_values)) == 1:
                print(f"Warning: {group_value} has no variance (all values identical), skipping...")
                continue
            
            # Remove outliers using IQR method
            Q1 = np.percentile(all_values, 25)
            Q3 = np.percentile(all_values, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Filter out outliers from the dataset
            group_data_clean = group_data[
                (group_data[value_col] >= lower_bound) & 
                (group_data[value_col] <= upper_bound)
            ].copy()
            
            # Check if we still have enough data after outlier removal
            if len(group_data_clean) < 3:
                print(f"Warning: {group_value} has insufficient data after outlier removal, skipping...")
                continue
            
            # Verify each unit still has data
            units_with_data = [unit for unit in units if len(group_data_clean[group_data_clean['Unit'] == unit]) > 0]
            if len(units_with_data) < 2:
                print(f"Warning: {group_value} has less than 2 units with data after outlier removal, skipping...")
                continue
            
            try:
                import warnings
                from statsmodels.tools.sm_exceptions import ValueWarning
                
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=ValueWarning)
                    
                    # Get region information from DataFrame
                    # Now Unit column has base unit name and Region column has the region
                    # We need to create unit-region combinations for analysis
                    
                    # Create a combined Unit_Region column for grouping
                    group_data_clean['Unit_Region'] = group_data_clean.apply(
                        lambda row: f"{row['Unit']}_{row['Region']}" if pd.notna(row.get('Region')) else row['Unit'],
                        axis=1
                    )
                    
                    unit_regions = group_data_clean['Unit_Region'].unique()
                    unit_info = {}  # {unit_region: {'base': base_unit, 'region': region}}
                    base_units = {}  # {base_unit: [unit_region_names]}
                    
                    for unit_region in unit_regions:
                        # Extract base unit and region
                        unit_region_data = group_data_clean[group_data_clean['Unit_Region'] == unit_region]
                        if len(unit_region_data) > 0:
                            base_unit = unit_region_data['Unit'].iloc[0]
                            region = unit_region_data['Region'].iloc[0] if 'Region' in unit_region_data.columns else 'Other'
                            if pd.isna(region):
                                region = 'Other'
                        else:
                            continue
                        
                        unit_info[unit_region] = {'base': base_unit, 'region': region}
                        if base_unit not in base_units:
                            base_units[base_unit] = []
                        base_units[base_unit].append(unit_region)
                    
                    # 1. WITHIN-UNIT COMPARISONS: Compare first two regions for each base unit
                    within_unit_sig_pairs = []
                    
                    for base_unit, unit_region_list in base_units.items():
                        if len(unit_region_list) >= 2:  # Only if we have multiple regions
                            # Group by region
                            regions_map = {}
                            for ur in unit_region_list:
                                region = unit_info[ur]['region']
                                if region not in regions_map:
                                    regions_map[region] = []
                                regions_map[region].append(ur)
                            
                            # Compare first two regions found
                            region_names = sorted(regions_map.keys())
                            if len(region_names) >= 2:
                                region1_units = regions_map[region_names[0]]
                                region2_units = regions_map[region_names[1]]
                                
                                if region1_units and region2_units:
                                    region1_unit = region1_units[0]
                                    region2_unit = region2_units[0]
                                    region1_data = group_data_clean[group_data_clean['Unit_Region'] == region1_unit][value_col].values
                                    region2_data = group_data_clean[group_data_clean['Unit_Region'] == region2_unit][value_col].values
                                    
                                    if len(region1_data) > 0 and len(region2_data) > 0:
                                        # Perform t-test for paired comparison
                                        from scipy.stats import ttest_ind
                                        t_stat, p_val = ttest_ind(region1_data, region2_data)
                                        
                                        if p_val < 0.05:
                                            within_unit_sig_pairs.append((region1_unit, region2_unit, p_val))
                    
                    # 2. BETWEEN-UNIT COMPARISONS: Compare base units within same region
                    # Analyze all regions present in the data
                    between_unit_sig_pairs = []
                    
                    # Get all unique regions from the data
                    all_regions = set(unit_info[u]['region'] for u in unit_info.keys())
                    
                    for region in sorted(all_regions):
                        # Get unit_regions in this region
                        region_unit_regions = [u for u in unit_info.keys() if unit_info[u]['region'] == region]
                        
                        if len(region_unit_regions) < 2:
                            continue
                        
                        # Get data for each unit_region in this region
                        region_base_data = {}
                        for unit_region in region_unit_regions:
                            unit_data = group_data_clean[group_data_clean['Unit_Region'] == unit_region][value_col].values
                            if len(unit_data) > 0:
                                region_base_data[unit_region] = unit_data
                        
                        # Only perform ANOVA if we have at least 2 units with data
                        if len(region_base_data) >= 2:
                            # Perform one-way ANOVA on units within this region
                            region_groups = [region_base_data[u] for u in sorted(region_base_data.keys())]
                            f_stat, p_value = stats.f_oneway(*region_groups)
                            is_significant = p_value < 0.05
                            
                            # Perform Tukey HSD if significant
                            if is_significant:
                                # Create DataFrame with unit labels
                                region_df = pd.DataFrame()
                                for unit, values in region_base_data.items():
                                    temp_df = pd.DataFrame({
                                        'unit': [unit] * len(values),
                                        'value': values
                                    })
                                    region_df = pd.concat([region_df, temp_df], ignore_index=True)
                                
                                tukey_region = pairwise_tukeyhsd(
                                    endog=region_df['value'],
                                    groups=region_df['unit'],
                                    alpha=0.05
                                )
                                
                                tukey_summary = tukey_region.summary()
                                for row in tukey_summary.data[1:]:
                                    unit1 = str(row[0])
                                    unit2 = str(row[1])
                                    p_adj = float(row[3])
                                    reject = row[-1]
                                    
                                    if reject:
                                        between_unit_sig_pairs.append((unit1, unit2, p_adj))
                    
                    # Track if this molecule has significance
                    if within_unit_sig_pairs or between_unit_sig_pairs:
                        sig_info = f"{group_value}: "
                        sig_parts = []
                        if within_unit_sig_pairs:
                            sig_parts.append(f"{len(within_unit_sig_pairs)} within-unit")
                        if between_unit_sig_pairs:
                            sig_parts.append(f"{len(between_unit_sig_pairs)} between-unit")
                        sig_info += ", ".join(sig_parts)
                        significant_molecules.append(sig_info)
                    
                    # Store results
                    anova_results[group_value] = {
                        'within_unit_pairs': within_unit_sig_pairs,
                        'between_unit_pairs': between_unit_sig_pairs,
                        'units': units_with_data,
                        'unit_info': unit_info,
                        'base_units': base_units,
                        'group_data': group_data_clean
                    }
                    
                    # Create visualization with both types of significance bars
                    self._plot_unit_comparison(group_data, group_value, value_col, 
                                              data_type, units_with_data, 
                                              within_unit_sig_pairs, between_unit_sig_pairs,
                                              unit_info, output_dir,
                                              figsize, show_plots, anova_results, display_name_map, unit_color_map, unit_display_map, show_outliers=False)
                    
            except Exception as e:
                # Silently skip errors
                continue
        
        # Print summary of significant molecules
        if significant_molecules:
            print("\nMolecules/ratios with significant differences:")
            for sig_info in significant_molecules:
                print(f"  • {sig_info}")
        else:
            print("\nNo significant differences found.")
        
        return anova_results
    
    def _add_significance_bars(self, ax, sorted_units, positions, upper_whisker_values, means,
                               within_unit_sig_pairs, between_unit_sig_pairs, fontsize=9, linewidth=1.2):
        """
        Add significance bars with asterisks to a box plot axis.
        
        This function handles the spanning logic to ensure bars clear all intermediate data
        when connecting non-adjacent units.
        
        Args:
            ax: Matplotlib axis object
            sorted_units: List of unit names in plot order
            positions: List of x-axis positions for each unit
            upper_whisker_values: List of upper whisker values for each unit
            means: Array of mean values for each unit
            within_unit_sig_pairs: List of (unit1, unit2, p_val) tuples for within-unit comparisons
            between_unit_sig_pairs: List of (unit1, unit2, p_adj) tuples for between-unit comparisons
            fontsize: Font size for asterisks (default 9)
            linewidth: Line width for significance bars (default 1.2)
        """
        max_height = max(upper_whisker_values) if upper_whisker_values else 0
        
        # 1. Add WITHIN-UNIT significance bars (region comparisons within same unit)
        max_within_bar_height = 0
        if within_unit_sig_pairs and max_height > 0:
            y_offset_within = (max_height - min(means)) * 0.13
            
            for unit1, unit2, p_val in within_unit_sig_pairs:
                try:
                    i1 = sorted_units.index(unit1)
                    i2 = sorted_units.index(unit2)
                    
                    pos1 = positions[i1]
                    pos2 = positions[i2]
                    
                    # Find the max upper whisker of all units between i1 and i2 (inclusive)
                    min_idx = min(i1, i2)
                    max_idx = max(i1, i2)
                    local_max_height = max(upper_whisker_values[min_idx:max_idx+1])
                    
                    bar_height = local_max_height + y_offset_within * 1.0
                    max_within_bar_height = max(max_within_bar_height, bar_height)
                    
                    # Draw the bar (solid line for within-unit)
                    ax.plot([pos1, pos1, pos2, pos2], 
                           [bar_height - y_offset_within*0.2, bar_height, bar_height, bar_height - y_offset_within*0.2],
                           'k-', linewidth=linewidth, zorder=10)
                    
                    # Determine asterisks
                    if p_val < 0.0001:
                        asterisks = '****'
                    elif p_val < 0.001:
                        asterisks = '***'
                    elif p_val < 0.01:
                        asterisks = '**'
                    else:
                        asterisks = '*'
                    
                    ax.text((pos1 + pos2) / 2, bar_height, asterisks, 
                           ha='center', va='bottom', fontsize=fontsize, fontweight='bold', zorder=10)
                except (ValueError, IndexError):
                    continue
        
        # 2. Add BETWEEN-UNIT significance bars (same region comparisons)
        if between_unit_sig_pairs and max_height > 0:
            y_offset_between = (max_height - min(means)) * 0.13
            
            # Track occupied y-ranges for each x-range to prevent overlaps
            # List of tuples: (x_start, x_end, y_height)
            occupied_bars = []
            
            for unit1, unit2, p_adj in between_unit_sig_pairs:
                try:
                    i1 = sorted_units.index(unit1)
                    i2 = sorted_units.index(unit2)
                    
                    pos1 = positions[i1]
                    pos2 = positions[i2]
                    
                    # Find the max upper whisker of all units between i1 and i2 (inclusive)
                    min_idx = min(i1, i2)
                    max_idx = max(i1, i2)
                    local_max_height = max(upper_whisker_values[min_idx:max_idx+1])
                    
                    # Start between-unit bars above both data and any within-unit bars
                    if max_within_bar_height > 0:
                        base_height = max(max_within_bar_height, local_max_height + y_offset_between * 1.0) + y_offset_between * 0.8
                    else:
                        base_height = local_max_height + y_offset_between * 1.6
                    
                    # Find the lowest available height that doesn't overlap with existing bars
                    bar_height = base_height
                    min_pos = min(pos1, pos2)
                    max_pos = max(pos1, pos2)
                    
                    # Check for overlaps with existing bars
                    max_attempts = 20
                    for attempt in range(max_attempts):
                        overlaps = False
                        for occ_x_start, occ_x_end, occ_y in occupied_bars:
                            # Check if x-ranges overlap
                            x_overlap = not (max_pos < occ_x_start or min_pos > occ_x_end)
                            # Check if this height would overlap (with small margin)
                            y_overlap = abs(bar_height - occ_y) < y_offset_between * 0.7
                            
                            if x_overlap and y_overlap:
                                overlaps = True
                                break
                        
                        if not overlaps:
                            break
                        
                        # Move up if there's an overlap
                        bar_height += y_offset_between * 0.8
                    
                    # Record this bar's position
                    occupied_bars.append((min_pos, max_pos, bar_height))
                    
                    # Draw the bar (dashed line for between-unit)
                    ax.plot([pos1, pos1, pos2, pos2], 
                           [bar_height - y_offset_between*0.25, bar_height, bar_height, bar_height - y_offset_between*0.25],
                           'k--', linewidth=linewidth, dashes=(4, 2), zorder=9)
                    
                    # Determine asterisks
                    if p_adj < 0.0001:
                        asterisks = '****'
                    elif p_adj < 0.001:
                        asterisks = '***'
                    elif p_adj < 0.01:
                        asterisks = '**'
                    else:
                        asterisks = '*'
                    
                    ax.text((pos1 + pos2) / 2, bar_height, asterisks, 
                           ha='center', va='bottom', fontsize=fontsize, fontweight='bold', zorder=9)
                except (ValueError, IndexError, KeyError):
                    continue
    
    def _plot_unit_comparison(self, data, group_name, value_col, data_type,
                             units, within_unit_sig_pairs, between_unit_sig_pairs,
                             unit_info, output_dir, figsize, show_plots, anova_results, display_name_map=None, unit_color_map=None, unit_display_map=None, show_outliers=False):
        """
        Create bar plot with significance bars for unit comparisons.
        Shows two types of significance:
        - Within-unit: Region comparisons for same base unit (e.g., first vs second region)
        - Between-unit: Base unit comparisons within same region
        
        Args:
            data: Filtered DataFrame for one molecule/ratio type
            group_name: Name of the molecule/ratio type
            value_col: Name of the value column
            data_type: Type of data ('percentage' or 'ratio')
            units: List of unit names
            within_unit_sig_pairs: List of (unit1, unit2, p_value) for region comparisons within same unit
            between_unit_sig_pairs: List of (base1, base2, p_value) for base unit comparisons
            unit_info: Dictionary mapping unit names to {'base': base_unit, 'region': region}
            output_dir: Directory to save plot
            figsize: Figure size
            show_plots: Whether to display plot interactively
            anova_results: Dictionary containing ANOVA results
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create a combined Unit_Region for plotting
        data['Unit_Region'] = data.apply(
            lambda row: f"{row['Unit']}_{row['Region']}" if pd.notna(row.get('Region')) else row['Unit'],
            axis=1
        )
        
        # Group units by base unit and region
        unit_regions = {}  # {unit_region: region}
        unit_groups = {}  # {base_unit: {region_name: unit_region}}
        
        for unit_region in data['Unit_Region'].unique():
            unit_region_data = data[data['Unit_Region'] == unit_region]
            if len(unit_region_data) > 0:
                base_unit = unit_region_data['Unit'].iloc[0]
                region = unit_region_data['Region'].iloc[0] if 'Region' in unit_region_data.columns else 'Other'
                if pd.isna(region):
                    region = 'Other'
                
                unit_regions[unit_region] = region
                if base_unit not in unit_groups:
                    unit_groups[base_unit] = {}
                unit_groups[base_unit][region] = unit_region
        
        # Create sorted unit order: group paired units together
        sorted_units = []
        for base_unit in sorted(unit_groups.keys()):
            # Add regions in sorted order
            for region in sorted(unit_groups[base_unit].keys()):
                sorted_units.append(unit_groups[base_unit][region])
        
        # Prepare data for boxplot using sorted order
        plot_data = [data[data['Unit_Region'] == unit_region][value_col].values for unit_region in sorted_units]
        
        # Create positions with gaps between different base units
        positions = []
        current_pos = 0
        prev_base = None
        for unit_region in sorted_units:
            # Extract base unit
            unit_region_data = data[data['Unit_Region'] == unit_region]
            if len(unit_region_data) > 0:
                base_unit = unit_region_data['Unit'].iloc[0]
            else:
                base_unit = unit_region.split('_')[0]
            
            # Add gap if we're starting a new base unit
            if prev_base is not None and base_unit != prev_base:
                current_pos += 0.4  # Reduced gap between different base units (was 0.7)
            
            positions.append(current_pos)
            current_pos += 0.35  # Tighter spacing within same base unit (was 0.5)
            prev_base = base_unit
        
        # Create color palette using helper function
        base_color_map = self._create_color_map(sorted(unit_groups.keys()), unit_color_map)
        
        # Get all unique regions in sorted order
        all_regions_in_data = sorted(set(unit_regions.values()))
        
        # Assign colors based on region
        colors = []
        for unit_region in sorted_units:
            region = unit_regions.get(unit_region, 'Other')
            unit_region_data = data[data['Unit_Region'] == unit_region]
            if len(unit_region_data) > 0:
                base_unit = unit_region_data['Unit'].iloc[0]
            else:
                base_unit = unit_region.split('_')[0]
            
            if region == all_regions_in_data[0] if all_regions_in_data else 'Other':
                colors.append(base_color_map[base_unit])
            elif len(all_regions_in_data) > 1 and region == all_regions_in_data[1]:
                colors.append(self._create_lighter_shade(base_color_map[base_unit]))
            else:
                colors.append((0.85, 0.85, 0.85))
        
        # Create boxplot with custom positions
        bp = ax.boxplot(plot_data, positions=positions, patch_artist=True,
                        widths=0.28, showfliers=show_outliers,
                        boxprops=dict(facecolor='white', edgecolor='none', linewidth=0),
                        medianprops=dict(color='black', linewidth=1.5),
                        whiskerprops=dict(color='black', linewidth=1.0),
                        capprops=dict(color='black', linewidth=1.0),
                        flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, 
                                      linestyle='none', markeredgecolor='none', alpha=0.4))
        
        # Color the boxes with full opacity for cleaner appearance - no outline
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
            patch.set_edgecolor('none')
            patch.set_linewidth(0)
        
        # Calculate means and upper whisker values using helper function
        means = np.array([np.mean(d) if len(d) > 0 else 0 for d in plot_data])
        upper_whisker_values = self._calculate_upper_whisker_values(plot_data)
        max_height = max(upper_whisker_values)
        
        # Add significance bars using the helper function
        self._add_significance_bars(ax, sorted_units, positions, upper_whisker_values, means,
                                    within_unit_sig_pairs, between_unit_sig_pairs, fontsize=13, linewidth=1.5)
        
        # Set x-ticks and labels to strictly follow sorted_units (ordered_columns)
        ax.set_xticks(positions)
        ax.set_xticklabels(sorted_units, rotation=45, ha='center', fontsize=4)
        ax.set_xlabel('Unit (Region)', fontweight='bold', fontsize=14)
        ax.set_ylabel(f'{value_col.replace("_", " ")} (mean ± SEM)', fontweight='bold', fontsize=14)
        
        # Title using helper function
        display_name = self._apply_display_name(group_name, display_name_map)
        title = f'{display_name.replace("_", " ")}'
        ax.set_title(title, fontweight='bold', fontsize=20, pad=20)
        
        ax.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=0.5, color='gray')
        ax.set_axisbelow(True)  # Place grid behind plot elements
        
        # Set y-axis tick label font size
        ax.tick_params(axis='y', labelsize=10)
        
        # Set background color and create box with spines
        ax.set_facecolor('#f8f8f8')  # Light gray background
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        # Make all spines the same color and width
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.2)
        
        plt.tight_layout()
        
        # Save figure
        safe_name = group_name.replace('/', '_').replace('\\', '_').replace(' ', '_')
        output_path = os.path.join(output_dir,
                                   f"anova_unit_{data_type}_{safe_name}.svg")
        plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
        
        if show_plots:
            plt.show()
        else:
            plt.close()

    def create_multi_panel_boxplots(self, df, value_col, grouping_col, molecules_list,
                                     nrows, ncols, output_dir=None, data_type='percentage',
                                     figsize=None, show_plots=True, display_name_map=None,
                                     unit_color_map=None, unit_display_map=None):
        """
        Create a multi-panel figure with box plots for selected molecules/ratios.
        
        Args:
            df: DataFrame with columns [Unit, {Molecule or Ratio_Type}, {Percentage or Mean_Ratio}, 
                                        Region, Replicate, Image_Name, Sample_Name]
            value_col: Name of the value column ('Percentage' or 'Mean_Ratio')
            grouping_col: Name of the grouping column ('Molecule' or 'Ratio_Type')
            molecules_list: List of molecule/ratio names to plot
            nrows: Number of rows in subplot grid
            ncols: Number of columns in subplot grid
            output_dir: Directory to save plot (if None, uses current directory)
            data_type: Type of data ('percentage' or 'ratio') for labeling
            figsize: Figure size (width, height). If None, auto-calculated based on grid
            show_plots: Whether to display plot interactively
            display_name_map: Optional dict mapping original names to display names
            unit_color_map: Optional dict mapping unit names to hex color codes
            unit_display_map: Optional dict mapping full unit names to abbreviated names
            
        Returns:
            fig: The created figure object
        """
        from scipy import stats
        
        # Create output directory if needed
        if output_dir is None:
            output_dir = 'multi_panel_boxplots'
        os.makedirs(output_dir, exist_ok=True)
        
        # Use fixed subplot dimensions for consistency across different figures
        # Each subplot is 3 inches wide and 2.5 inches tall
        subplot_width = 3.0
        subplot_height = 2.5
        
        if figsize is None:
            # Calculate total figure size: subplots + spacing + legend
            # wspace and hspace add proportional spacing, so we need extra space
            figsize = (ncols * subplot_width + (ncols - 1) * 0.3 * subplot_width, 
                      nrows * subplot_height + (nrows - 1) * 0.4 * subplot_height + 1.5)
        
        # Create figure with subplots
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        
        # Flatten axes array for easier iteration
        if nrows == 1 and ncols == 1:
            axes = np.array([axes])
        axes = axes.flatten() if nrows > 1 or ncols > 1 else [axes]
        
        # Get unique units for legend
        unique_units = sorted(df['Unit'].unique())
        
        # Track which unit-region combinations actually appear in the data
        all_unit_regions = set()
        for group_value in molecules_list:
            group_data = df[df[grouping_col] == group_value].copy()
            if len(group_data) > 0:
                group_data['Unit_Region'] = group_data.apply(
                    lambda row: f"{row['Unit']}_{row['Region']}" if pd.notna(row.get('Region')) else row['Unit'],
                    axis=1
                )
                all_unit_regions.update(group_data['Unit_Region'].unique())
        
        # Determine which units have multiple regions vs single region (calculate once, use throughout)
        unit_region_structure = {}  # {unit: {'regions': set(), 'has_multiple': bool}}
        for unit in unique_units:
            unit_regions_found = set()
            for ur in all_unit_regions:
                if ur.startswith(f'{unit}_'):
                    region = ur.split('_', 1)[1]
                    unit_regions_found.add(region)
            unit_region_structure[unit] = {
                'regions': unit_regions_found,
                'has_multiple': len(unit_regions_found) >= 2
            }
        
        # Create color mapping using helper function
        base_color_map = self._create_color_map(unique_units, unit_color_map)
        
        # Iterate through molecules and create box plots
        for idx, group_value in enumerate(molecules_list):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            
            # Filter data for this molecule/ratio type
            group_data = df[df[grouping_col] == group_value].copy()
            
            if len(group_data) == 0:
                ax.axis('off')
                continue
            
            # Remove NaN and inf values
            group_data.replace([np.inf, -np.inf], np.nan, inplace=True)
            group_data = group_data.dropna(subset=[value_col])
            
            if len(group_data) == 0:
                ax.axis('off')
                continue
            
            # Create Unit_Region column
            group_data['Unit_Region'] = group_data.apply(
                lambda row: f"{row['Unit']}_{row['Region']}" if pd.notna(row.get('Region')) else row['Unit'],
                axis=1
            )
            
            # Group units by base unit and region
            unit_regions = {}
            unit_groups = {}
            
            for unit_region in group_data['Unit_Region'].unique():
                unit_region_data = group_data[group_data['Unit_Region'] == unit_region]
                if len(unit_region_data) > 0:
                    base_unit = unit_region_data['Unit'].iloc[0]
                    region = unit_region_data['Region'].iloc[0] if 'Region' in unit_region_data.columns else 'Other'
                    if pd.isna(region):
                        region = 'Other'
                    
                    unit_regions[unit_region] = region
                    if base_unit not in unit_groups:
                        unit_groups[base_unit] = {}
                    unit_groups[base_unit][region] = unit_region
            
            # Create sorted unit order: units with multiple regions first, then single-region units
            # Use pre-calculated unit_region_structure
            units_with_multiple = []
            units_with_single = []
            
            for base_unit in sorted(unit_groups.keys()):
                unit_regions_in_group = list(unit_groups[base_unit].values())
                # Check using pre-calculated structure
                if unit_region_structure[base_unit]['has_multiple']:
                    units_with_multiple.extend(unit_regions_in_group)
                else:
                    units_with_single.extend(unit_regions_in_group)

            # Combine: units with multiple regions first, then single-region units
            sorted_units = units_with_multiple + units_with_single
            
            # Prepare data for boxplot
            plot_data = [group_data[group_data['Unit_Region'] == unit_region][value_col].values 
                        for unit_region in sorted_units]
            
            # Create positions
            positions = []
            current_pos = 0
            prev_base = None
            for unit_region in sorted_units:
                unit_region_data = group_data[group_data['Unit_Region'] == unit_region]
                if len(unit_region_data) > 0:
                    base_unit = unit_region_data['Unit'].iloc[0]
                else:
                    base_unit = unit_region.split('_')[0]
                
                if prev_base is not None and base_unit != prev_base:
                    current_pos += 0.4
                
                positions.append(current_pos)
                current_pos += 0.35
                prev_base = base_unit
            
            # Create colors
            # Get all unique regions in this subplot's data for color assignment
            all_regions_present = sorted(set(unit_regions.values()))
            
            colors = []
            for unit_region in sorted_units:
                region = unit_regions.get(unit_region, 'Other')
                unit_region_data = group_data[group_data['Unit_Region'] == unit_region]
                if len(unit_region_data) > 0:
                    base_unit = unit_region_data['Unit'].iloc[0]
                else:
                    base_unit = unit_region.split('_')[0]
                
                if region == all_regions_present[0] if all_regions_present else 'Other':
                    colors.append(base_color_map[base_unit])
                elif len(all_regions_present) > 1 and region == all_regions_present[1]:
                    colors.append(self._create_lighter_shade(base_color_map[base_unit]))
                else:
                    colors.append((0.85, 0.85, 0.85))
            
            # Create boxplot
            bp = ax.boxplot(plot_data, positions=positions, patch_artist=True,
                           widths=0.25, showfliers=False,
                           boxprops=dict(facecolor='white', edgecolor='black', linewidth=1.0),
                           medianprops=dict(color='black', linewidth=1.5),
                           whiskerprops=dict(color='black', linewidth=1.0),
                           capprops=dict(color='black', linewidth=1.0))
            
            # Color the boxes
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.85)
                patch.set_edgecolor('black')
                patch.set_linewidth(1.0)
            
            # Calculate means and upper whisker values using helper function
            means = np.array([np.mean(d) if len(d) > 0 else 0 for d in plot_data])
            upper_whisker_values = self._calculate_upper_whisker_values(plot_data)
            max_height = max(upper_whisker_values) if upper_whisker_values else 0
            
            # Remove outliers using IQR method for statistical testing only
            all_values = np.concatenate([d for d in plot_data if len(d) > 0])
            Q1 = np.percentile(all_values, 25)
            Q3 = np.percentile(all_values, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Filter out outliers from the dataset for statistical testing
            group_data_clean = group_data[
                (group_data[value_col] >= lower_bound) & 
                (group_data[value_col] <= upper_bound)
            ].copy()
            
            # Perform statistical tests on cleaned data (without outliers)
            # Get unit info for this molecule
            unit_info = {}
            base_units = {}
            
            for unit_region in sorted_units:
                unit_region_data = group_data[group_data['Unit_Region'] == unit_region]
                if len(unit_region_data) > 0:
                    base_unit = unit_region_data['Unit'].iloc[0]
                    region = unit_region_data['Region'].iloc[0] if 'Region' in unit_region_data.columns else 'Other'
                    if pd.isna(region):
                        region = 'Other'
                    
                    unit_info[unit_region] = {'base': base_unit, 'region': region}
                    
                    if base_unit not in base_units:
                        base_units[base_unit] = []
                    base_units[base_unit].append(unit_region)
            
            # Within-unit comparisons - use cleaned data
            # Match single panel logic: only compare first two regions for each base unit
            within_unit_sig_pairs = []
            for base_unit, unit_region_list in base_units.items():
                if len(unit_region_list) >= 2:  # Only if we have multiple regions
                    # Group by region
                    regions_map = {}
                    for ur in unit_region_list:
                        region = unit_info[ur]['region']
                        if region not in regions_map:
                            regions_map[region] = []
                        regions_map[region].append(ur)
                    
                    # Compare first two regions (alphabetically sorted)
                    region_names = sorted(regions_map.keys())
                    if len(region_names) >= 2:
                        region1_units = regions_map[region_names[0]]
                        region2_units = regions_map[region_names[1]]
                        
                        if region1_units and region2_units:
                            region1_unit = region1_units[0]  # Take first unit from region 1
                            region2_unit = region2_units[0]  # Take first unit from region 2
                            region1_data = group_data_clean[group_data_clean['Unit_Region'] == region1_unit][value_col].values
                            region2_data = group_data_clean[group_data_clean['Unit_Region'] == region2_unit][value_col].values
                            
                            if len(region1_data) > 0 and len(region2_data) > 0:
                                try:
                                    from scipy.stats import ttest_ind
                                    t_stat, p_val = ttest_ind(region1_data, region2_data)
                                    
                                    if p_val < 0.05:
                                        within_unit_sig_pairs.append((region1_unit, region2_unit, p_val))
                                except:
                                    pass
            
            # Between-unit comparisons (same region) - use cleaned data
            # Test all unique regions found in the data
            between_unit_sig_pairs = []
            all_unique_regions = sorted(set(unit_info[u]['region'] for u in unit_info.keys()))
            for region in all_unique_regions:
                # Get unit_regions in this region
                region_unit_regions = [u for u in unit_info.keys() if unit_info[u]['region'] == region]
                
                if len(region_unit_regions) < 2:
                    continue
                
                # Get data for each unit_region in this region
                region_base_data = {}
                for unit_region in region_unit_regions:
                    unit_data = group_data_clean[group_data_clean['Unit_Region'] == unit_region][value_col].values
                    if len(unit_data) > 0:
                        region_base_data[unit_region] = unit_data
                
                # Only perform ANOVA if we have at least 2 units with data
                if len(region_base_data) >= 2:
                    try:
                        # Perform one-way ANOVA on units within this region
                        region_groups = [region_base_data[u] for u in sorted(region_base_data.keys())]
                        f_stat, p_value = stats.f_oneway(*region_groups)
                        is_significant = p_value < 0.05
                        
                        # Perform Tukey HSD if significant
                        if is_significant:
                            from statsmodels.stats.multicomp import pairwise_tukeyhsd
                            
                            # Create DataFrame with unit labels
                            region_df = pd.DataFrame()
                            for unit, values in region_base_data.items():
                                temp_df = pd.DataFrame({
                                    'unit': [unit] * len(values),
                                    'value': values
                                })
                                region_df = pd.concat([region_df, temp_df], ignore_index=True)
                            
                            tukey_region = pairwise_tukeyhsd(
                                endog=region_df['value'],
                                groups=region_df['unit'],
                                alpha=0.05
                            )
                            
                            tukey_summary = tukey_region.summary()
                            for row in tukey_summary.data[1:]:
                                unit1 = str(row[0])
                                unit2 = str(row[1])
                                p_adj = float(row[3])
                                reject = row[-1]
                                
                                if reject:
                                    between_unit_sig_pairs.append((unit1, unit2, p_adj))
                    except:
                        pass
            
            # Add significance bars using the helper function
            self._add_significance_bars(ax, sorted_units, positions, upper_whisker_values, means,
                                       within_unit_sig_pairs, between_unit_sig_pairs, fontsize=9, linewidth=1.5)
            
            # Set title using helper function
            display_name = self._apply_display_name(group_value, display_name_map)
            ax.set_title(display_name.replace("_", " "), fontweight='bold', fontsize=14)
            
            # Remove x-axis tick labels
            ax.set_xticklabels([])
            ax.set_xlabel('')
            
            # Set y-axis label
            if idx % ncols == 0:  # Only leftmost column
                ax.set_ylabel(f'{value_col.replace("_", " ")}', fontweight='bold', fontsize=12)
            
            # Add grid
            ax.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=0.5, color='gray')
            ax.set_axisbelow(True)
            ax.tick_params(axis='y', labelsize=10)
            
            # Set background
            ax.set_facecolor('#f8f8f8')
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1.0)
        
        # Hide unused subplots
        for idx in range(len(molecules_list), len(axes)):
            axes[idx].axis('off')
        
        # Create legend - separate units with two regions from units with one region
        from matplotlib.patches import Patch
        
        # Use pre-calculated unit_region_structure
        units_with_both_regions = [u for u in unique_units if unit_region_structure[u]['has_multiple']]
        units_with_single_region = [u for u in unique_units if not unit_region_structure[u]['has_multiple'] and len(unit_region_structure[u]['regions']) == 1]
        
        # Build legend elements with units with both regions first, then single-region units
        first_row_elements = []
        second_row_elements = []
        
        # Add units with both regions to first row
        for unit in units_with_both_regions:
            display_unit = unit_display_map.get(unit, unit) if unit_display_map else unit
            
            # Get the regions for this unit
            unit_regions = [ur for ur in all_unit_regions if ur.startswith(f'{unit}_')]
            region_names = sorted(set([ur.split('_')[-1] for ur in unit_regions]))
            
            # Add patches for each region (use abbreviation for label)
            for idx, region in enumerate(region_names[:2]):  # Limit to first 2 regions
                region_abbrev = region[0]  # First letter abbreviation
                
                if idx == 0:
                    # First region uses base color
                    first_row_elements.append(Patch(facecolor=base_color_map[unit], 
                                                edgecolor='none', 
                                                label=f'{display_unit} ({region_abbrev})',
                                                alpha=0.85))
                else:
                    # Second region uses lighter color
                    lighter_color = self._create_lighter_shade(base_color_map[unit])
                    first_row_elements.append(Patch(facecolor=lighter_color, 
                                                edgecolor='none', 
                                                label=f'{display_unit} ({region_abbrev})',
                                                alpha=0.85))
        
        # Add units with only one region to second row (include region abbreviation)
        for unit in units_with_single_region:
            display_unit = unit_display_map.get(unit, unit) if unit_display_map else unit
            
            # Find which region this unit has
            unit_regions = [ur for ur in all_unit_regions if ur.startswith(f'{unit}_')]
            if unit_regions:
                region_name = unit_regions[0].split('_')[-1]
                region_abbrev = region_name[0]  # First letter abbreviation
                # Get all unique regions in dataset to determine color
                all_dataset_regions = sorted(set([ur.split('_')[-1] for ur in all_unit_regions if '_' in ur]))
                
                # Use base color for first alphabetical region, lighter for others
                if all_dataset_regions and region_name == all_dataset_regions[0]:
                    second_row_elements.append(Patch(facecolor=base_color_map[unit], 
                                                edgecolor='none', 
                                                label=f'{display_unit} ({region_abbrev})',
                                                alpha=0.85))
                else:
                    lighter_color = self._create_lighter_shade(base_color_map[unit])
                    second_row_elements.append(Patch(facecolor=lighter_color, 
                                                edgecolor='none', 
                                                label=f'{display_unit} ({region_abbrev})',
                                                alpha=0.85))
        
        # Calculate total entries and items per row
        num_dual_region = len(first_row_elements)
        num_single_region = len(second_row_elements)
        total_entries = num_dual_region + num_single_region
        
        # Divide by 2 to get number of columns per row (round up)
        import math
        ncols = math.ceil(total_entries / 2)
        
        # Distribute dual-region entries first across both rows, then add single-region entries
        legend_elements = []
        
        # Split dual-region entries between two rows
        dual_first_half = first_row_elements[:ncols]
        dual_second_half = first_row_elements[ncols:]
        
        # Combine: first row gets first half of dual-region + single-region entries
        first_row = dual_first_half
        remaining_slots_first_row = ncols - len(first_row)
        if remaining_slots_first_row > 0:
            first_row.extend(second_row_elements[:remaining_slots_first_row])
            remaining_single = second_row_elements[remaining_slots_first_row:]
        else:
            remaining_single = second_row_elements
        
        # Second row gets second half of dual-region + remaining single-region entries
        second_row = dual_second_half + remaining_single
        
        # Pad second row if needed
        while len(second_row) < ncols:
            second_row.append(Patch(facecolor='none', edgecolor='none', label=''))
        
        # Combine both rows
        legend_elements = first_row + second_row
        
        # Create legend with ncols columns (forces two rows)
        fig.legend(handles=legend_elements, loc='lower center', ncol=ncols, 
                  bbox_to_anchor=(0.5, -0.02), fontsize=13, frameon=True, 
                  fancybox=True, shadow=False, columnspacing=1.0, handletextpad=0.5)
        
        plt.tight_layout(rect=[0, 0.12, 1, 1])
        
        # Save figure
        output_path = os.path.join(output_dir, f"multi_panel_{data_type}_boxplots.svg")
        plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        return fig

    def generate_region_heatmaps(self, df, value_col, grouping_col, output_dir=None,
                                  data_type='percentage', figsize=(8, 6), 
                                  show_plots=True, cmap='RdBu_r', vmin=-2, vmax=2, display_name_map=None, unit_display_map=None):
        """
        Generate z-score normalized heatmaps showing variability across regions.
        
        This creates separate heatmaps for each molecule/ratio type, where rows are units
        and columns are regions. Values are normalized using z-scores relative to the
        overall mean across all unit-region combinations.
        Values are normalized to the inter-region and inter-unit mean (z-score normalization).
        
        Args:
            df: DataFrame with columns [Unit, {Molecule or Ratio_Type}, {Percentage or Mean_Ratio}, Region, ...]
            value_col: Name of the value column ('Percentage' or 'Mean_Ratio')
            grouping_col: Name of the grouping column ('Molecule' or 'Ratio_Type')
            output_dir: Directory to save plots (if None, uses current directory)
            data_type: Type of data ('percentage' or 'ratio') for labeling
            figsize: Figure size for plots (width, height)
            show_plots: Whether to display plots interactively
            cmap: Colormap to use for heatmap
            vmin, vmax: Min and max values for colormap scale
            
        Returns:
            heatmap_data: Dictionary with heatmap matrices for each molecule/ratio type
        """
        # Create output directory if needed
        if output_dir is None:
            output_dir = 'region_heatmaps_normalized'
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if Region column exists
        if 'Region' not in df.columns:
            print("Warning: Region column not found in DataFrame. Cannot generate heatmaps.")
            return {}
        
        # Get unique molecules/ratio types
        unique_groups = df[grouping_col].unique()
        
        # Sort by display_name_map order if provided, otherwise alphabetically
        if display_name_map:
            # Create ordered list: first items in display_name_map order, then remaining alphabetically
            ordered_groups = [k for k in display_name_map.keys() if k in unique_groups]
            remaining = sorted([g for g in unique_groups if g not in display_name_map.keys()])
            sorted_groups = ordered_groups + remaining
        else:
            sorted_groups = sorted(unique_groups)
        
        heatmap_data = {}
        
        for group_value in sorted_groups:
            # Filter data for this molecule/ratio type
            group_data = df[df[grouping_col] == group_value].copy()
            
            # Skip if insufficient data
            if len(group_data) < 2:
                continue
            
            # Remove NaN and infinite values
            group_data = group_data.replace([np.inf, -np.inf], np.nan)
            group_data = group_data.dropna(subset=[value_col])
            
            if len(group_data) < 2:
                continue
            
            # Calculate mean and std for each Unit-Region combination
            agg_data = group_data.groupby(['Unit', 'Region'])[value_col].agg(['mean', 'std', 'count']).reset_index()
            
            # Skip if not enough data
            if len(agg_data) < 2:
                continue
            
            # Calculate z-scores normalized to this molecule's own mean across all unit-region combinations
            molecule_mean = agg_data['mean'].mean()
            molecule_std = agg_data['mean'].std()
            
            if molecule_std == 0:
                print(f"Warning: {group_value} has zero variance, skipping...")
                continue
            
            # Calculate z-scores (normalized to this molecule's mean across units and regions)
            agg_data['z_score'] = (agg_data['mean'] - molecule_mean) / molecule_std
            
            # Create pivot table for heatmap: Units as rows, Regions as columns
            try:
                heatmap_matrix = agg_data.pivot(index='Unit', columns='Region', values='z_score')
                
                # Fill missing unit-region combinations with -3 to show as extremely low
                heatmap_matrix = heatmap_matrix.fillna(-3)
                
                # Store the data
                heatmap_data[group_value] = {
                    'matrix': heatmap_matrix,
                    'raw_data': agg_data,
                    'molecule_mean': molecule_mean,
                    'molecule_std': molecule_std
                }
                
                # Calculate dynamic figure size based on matrix dimensions
                n_rows = heatmap_matrix.shape[0]  # Number of units
                n_cols = heatmap_matrix.shape[1]  # Number of regions
                # Scale: ~1.3 inch per column, ~0.5 inches per row for molecules (increased from 0.4)
                # Cap at 6 inches max height to match ratio heatmaps
                fig_width = max(6, min(10, 1.3 * n_cols + 2))  # +2 for labels/colorbar
                if data_type == 'percentage':
                    fig_height = max(3, min(6, 0.5 * n_rows + 2))  # Increased row height for molecules
                else:
                    fig_height = max(3, min(6, 0.4 * n_rows + 2))  # Keep original for ratios
                
                # Create heatmap visualization
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                
                # Create heatmap with journal-quality styling
                sns.heatmap(heatmap_matrix, annot=False, cmap=cmap, 
                           center=0, vmin=vmin, vmax=vmax,
                           cbar_kws={'label': f'Z-score (normalized to {group_value.replace("_", " ")} mean)', 
                                    'shrink': 0.8},
                           linewidths=0.5, linecolor='white', ax=ax, square=False)
                
                # Formatting
                ax.set_xlabel('Region', fontweight='bold', fontsize=14)
                ax.set_ylabel('Unit', fontweight='bold', fontsize=14)
                display_name = self._apply_display_name(group_value, display_name_map)
                ax.set_title(f'{display_name.replace("_", " ")}',
                           fontweight='bold', fontsize=16, pad=20)
                
                # Rotate labels and replace underscores with spaces
                x_labels = [label.get_text().replace('_', ' ') for label in ax.get_xticklabels()]
                ax.set_xticklabels(x_labels, rotation=0, ha='center', fontsize=12)
                # Apply unit_display_map to y-axis labels (units)
                y_labels = []
                for label in ax.get_yticklabels():
                    original_name = label.get_text()
                    if unit_display_map and original_name in unit_display_map:
                        y_labels.append(unit_display_map[original_name])
                    else:
                        y_labels.append(original_name.replace('_', ' '))
                ax.set_yticklabels(y_labels, rotation=0, fontsize=12)
                
                # Style the axes with light gray borders to match separator lines
                for spine in ax.spines.values():
                    spine.set_edgecolor('lightgray')
                    spine.set_linewidth(1.0)
                
                plt.tight_layout()
                
                # Add thick border around the heatmap (after tight_layout)
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor('black')
                    spine.set_linewidth(2.5)
                
                # Save figure
                safe_name = group_value.replace('/', '_').replace('\\', '_').replace(' ', '_')
                output_path = os.path.join(output_dir, 
                                          f"heatmap_normalized_{data_type}_{safe_name}.svg")
                plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
                
                if show_plots:
                    plt.show()
                else:
                    plt.close()
                    
            except Exception as e:
                print(f"Error creating heatmap for {group_value}: {str(e)}")
                continue
        
        print(f"\nGenerated {len(heatmap_data)} heatmaps in {output_dir}")
        
        # Export top 3 most significant molecules for each unit-region
        self._export_top_molecules_per_unit_region(heatmap_data, output_dir, display_name_map, 
                                                   grouping_col, data_type)
        
        return heatmap_data
    
    def _export_top_molecules_per_unit_region(self, heatmap_data, output_dir, display_name_map=None,
                                             grouping_col='Molecule', data_type='percentage'):
        """
        Export top 3 most significant (highest positive z-score) and bottom 3 
        (most negative z-score) molecules for each unit-region combination.
        
        Args:
            heatmap_data: Dictionary from generate_region_heatmaps containing z-score data
            output_dir: Directory to save CSV files
            display_name_map: Optional mapping for molecule display names
            grouping_col: Name of the grouping column ('Molecule' or 'Ratio_Type')
            data_type: Type of data ('percentage' or 'ratio')
        """
        if not heatmap_data:
            return
        
        # Determine column names based on data type
        is_ratio = (data_type == 'ratio')
        item_col = 'Ratio_Type' if is_ratio else 'Molecule'
        item_display_col = 'Ratio_Display' if is_ratio else 'Molecule_Display'
        
        # Collect all z-scores for each unit-region across all molecules
        all_data = []
        
        for molecule, data in heatmap_data.items():
            matrix = data['matrix']
            raw_data = data['raw_data']
            display_name = self._apply_display_name(molecule, display_name_map)
            
            # Iterate through the matrix
            for unit in matrix.index:
                for region in matrix.columns:
                    z_score = matrix.loc[unit, region]
                    if pd.notna(z_score):
                        # Get the actual mean value from raw_data
                        raw_value = raw_data[(raw_data['Unit'] == unit) & 
                                            (raw_data['Region'] == region)]['mean'].values
                        actual_value = raw_value[0] if len(raw_value) > 0 else np.nan
                        
                        all_data.append({
                            'Unit': unit,
                            'Region': region,
                            'Unit_Region': f"{unit}_{region}",
                            item_col: molecule,
                            item_display_col: display_name,
                            'Z_Score': z_score,
                            'Actual_Value': actual_value
                        })
        
        if not all_data:
            return
        
        # Convert to DataFrame
        df_all = pd.DataFrame(all_data)
        
        # For each unit-region, find top 3 and bottom 3
        results = []
        
        for unit_region in df_all['Unit_Region'].unique():
            unit_region_data = df_all[df_all['Unit_Region'] == unit_region].copy()
            unit_region_data = unit_region_data.sort_values('Z_Score', ascending=False)
            
            unit = unit_region_data['Unit'].iloc[0]
            region = unit_region_data['Region'].iloc[0]
            
            # Top 3 (highest positive z-scores)
            top_3 = unit_region_data.head(3)
            for rank, (idx, row) in enumerate(top_3.iterrows(), 1):
                result_dict = {
                    'Unit': unit,
                    'Region': region,
                    'Unit_Region': unit_region,
                    'Ranking': 'Top',
                    'Rank': rank,
                    item_col: row[item_col],
                    item_display_col: row[item_display_col],
                    'Z_Score': row['Z_Score'],
                    'Actual_Value': row['Actual_Value']
                }
                results.append(result_dict)
            
            # Bottom 3 (most negative z-scores)
            bottom_3 = unit_region_data.tail(3).iloc[::-1]  # Reverse to show most negative first
            for rank, (idx, row) in enumerate(bottom_3.iterrows(), 1):
                result_dict = {
                    'Unit': unit,
                    'Region': region,
                    'Unit_Region': unit_region,
                    'Ranking': 'Bottom',
                    'Rank': rank,
                    item_col: row[item_col],
                    item_display_col: row[item_display_col],
                    'Z_Score': row['Z_Score'],
                    'Actual_Value': row['Actual_Value']
                }
                results.append(result_dict)
        
        # Create DataFrame and save
        results_df = pd.DataFrame(results)
        output_path = os.path.join(output_dir, 'top_bottom_molecules_by_unit_region.csv')
        results_df.to_csv(output_path, index=False)
        print(f"Saved top/bottom molecules summary to: {output_path}")

    def generate_raw_ratio_heatmap(self, df, value_col='Mean_Ratio', grouping_col='Ratio_Type', 
                                    output_dir=None, figsize=(8, 6), show_plots=True, 
                                    cmap='RdBu_r', vmin=None, vmax=None, display_name_map=None, unit_display_map=None, unit_order=None):
        """
        Generate a non-normalized heatmap showing raw mean ratio values across units and regions.
        
        This creates a single heatmap where rows are ratio types and columns are Unit-Region combinations,
        displaying the actual mean ratio values without normalization.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with columns [Unit, Ratio_Type, Mean_Ratio, Region, ...]
        value_col : str
            Name of the column containing ratio values (default: 'Mean_Ratio')
        grouping_col : str
            Name of the column to group by (default: 'Ratio_Type')
        output_dir : str
            Directory to save the heatmap plot
        figsize : tuple
            Figure size (width, height)
        show_plots : bool
            Whether to display plot interactively
        cmap : str
            Colormap for heatmap (default: 'viridis')
        vmin, vmax : float
            Min and max values for colormap (if None, uses data range)
        unit_order : list, optional
            Custom order for units on x-axis. Should be a list of unit names.
            If None, uses default ordering (units with both regions first, then single-region units)
            
        Returns:
        --------
        dict : Dictionary with heatmap matrix and statistics
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if Region column exists
        # Check if Region column exists
        has_region = 'Region' in df.columns and df['Region'].notna().any()
        
        if has_region:
            # Calculate mean for each Unit-Region-Group combination
            agg_data = df.groupby(['Unit', 'Region', grouping_col])[value_col].agg(['mean', 'std', 'count']).reset_index()
            # Create a combined column label for Unit-Region using standardized delimiter
            agg_data['Unit_Region'] = agg_data['Unit'] + ' (' + agg_data['Region'] + ')'
        else:
            # Calculate mean for each Unit-Group combination (no regions)
            agg_data = df.groupby(['Unit', grouping_col])[value_col].agg(['mean', 'std', 'count']).reset_index()
            agg_data['Unit_Region'] = agg_data['Unit']
        
        # Remove NaN and infinite values
        agg_data = agg_data.replace([np.inf, -np.inf], np.nan)
        agg_data = agg_data.dropna(subset=['mean'])
        
        if len(agg_data) < 2:
            print("Insufficient data after removing NaN/inf values")
            return {}
        
        # Create pivot table for heatmap: Ratio types as rows, Unit_Region as columns
        try:
            heatmap_matrix = agg_data.pivot(index=grouping_col, columns='Unit_Region', values='mean')
            
            # Sort rows by display_name_map order if provided, otherwise alphabetically
            if display_name_map:
                # Get current index (molecule/ratio names)
                current_index = heatmap_matrix.index.tolist()
                # Create ordered list: first items in display_name_map order, then remaining alphabetically
                ordered_index = [k for k in display_name_map.keys() if k in current_index]
                remaining_index = sorted([g for g in current_index if g not in display_name_map.keys()])
                new_index = ordered_index + remaining_index
                heatmap_matrix = heatmap_matrix.reindex(new_index)
            else:
                # Sort rows alphabetically
                heatmap_matrix = heatmap_matrix.sort_index()
            
            # Sort columns based on unit_order if provided, otherwise use default ordering
            if unit_order is not None:
                # Custom ordering based on provided unit_order list
                all_columns = heatmap_matrix.columns.tolist()
                ordered_columns = []
                for unit in unit_order:
                    # Add all columns that start with this unit name
                    matching_cols = [col for col in all_columns if col.startswith(unit + ' (') or col == unit]
                    ordered_columns.extend(sorted(matching_cols))
                # Add any remaining columns not in unit_order
                remaining_cols = [col for col in all_columns if col not in ordered_columns]
                ordered_columns.extend(sorted(remaining_cols))
                heatmap_matrix = heatmap_matrix[ordered_columns]
            elif has_region:
                # Default ordering: units with both regions first, then single-region units
                all_columns = heatmap_matrix.columns.tolist()
                # Identify which units have both regions
                unit_region_map = {}  # {base_unit: [list of unit_region columns]}
                for col in all_columns:
                    if ' (' in col:
                        base_unit = col.split(' (')[0]
                        if base_unit not in unit_region_map:
                            unit_region_map[base_unit] = [col]
                        else:
                            unit_region_map[base_unit].append(col)
                    else:
                        if col not in unit_region_map:
                            unit_region_map[col] = [col]
                ordered_columns = []
                for base_unit, cols in unit_region_map.items():
                    if len(cols) > 1:
                        ordered_columns.extend(sorted(cols))
                for base_unit, cols in unit_region_map.items():
                    if len(cols) == 1:
                        ordered_columns.extend(cols)
                heatmap_matrix = heatmap_matrix[ordered_columns]
                
            # Store the data (no overall stats since each group normalized independently)
            heatmap_data = {
                'matrix': heatmap_matrix,
                'raw_data': agg_data
            }
            
            # Calculate dynamic figure size based on matrix dimensions
            n_rows = heatmap_matrix.shape[0]  # Number of molecules/ratios
            n_cols = heatmap_matrix.shape[1]  # Number of unit-region combinations
            # Scale: ~0.7 inches per column, ~0.35 inches per row, with min/max bounds for consistency
            fig_width = max(6, min(10, 0.7 * n_cols + 2.5))  # +2.5 for labels/colorbar
            fig_height = max(3, min(6, 0.35 * n_rows + 2))  # +2 for title/labels
            
            # Create heatmap visualization
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            
            # Create heatmap with journal-quality styling
            # Using RdBu_r with center at 0.5 for ratio data (blue=0, white=0.5, red=1)
            sns.heatmap(heatmap_matrix, annot=False, cmap='RdBu_r', 
                       center=0.5, vmin=0, vmax=1,
                       cbar_kws={'label': 'Mean Value', 'shrink': 0.8},
                       linewidths=0.5, linecolor='white', ax=ax, square=False)
            
            # Add visual separators between different units if showing regions
            if has_region:
                col_names = heatmap_matrix.columns.tolist()
                unit_changes = []
                prev_unit = None
                for idx, col_name in enumerate(col_names):
                    # Extract unit name (before the parenthesis)
                    current_unit = col_name.split(' (')[0] if ' (' in col_name else col_name
                    if prev_unit is not None and current_unit != prev_unit:
                        unit_changes.append(idx)
                    prev_unit = current_unit
                # Draw vertical lines at unit boundaries
                for boundary in unit_changes:
                    ax.axvline(x=boundary, color='gray', linewidth=1.4, zorder=10)
            
            # Formatting
            if has_region:
                # Update x-tick labels to use standardized format and apply unit_display_map
                current_labels = [label.get_text() for label in ax.get_xticklabels()]
                new_labels = []
                for label in current_labels:
                    # Extract unit name (before parenthesis)
                    if ' (' in label:
                        unit_part, region_part = label.split(' (', 1)
                        # Apply unit display map
                        if unit_display_map and unit_part in unit_display_map:
                            unit_part = unit_display_map[unit_part]
                        # Abbreviate region to first letter only
                        region_abbrev = region_part.rstrip(')')[0] if region_part.rstrip(')') else region_part
                        new_labels.append(f"{unit_part} ({region_abbrev})")
                    else:
                        # Apply unit display map to label without region
                        if unit_display_map and label in unit_display_map:
                            new_labels.append(unit_display_map[label])
                        else:
                            new_labels.append(label.replace('_', ' '))
                ax.set_xticklabels(new_labels, rotation=45, ha='center', fontsize=10)
                ax.set_xlabel('Unit (Region)', fontweight='bold', fontsize=14)
            else:
                # Apply unit_display_map to x-axis labels (without regions)
                x_labels = []
                for label in ax.get_xticklabels():
                    original_name = label.get_text()
                    if unit_display_map and original_name in unit_display_map:
                        x_labels.append(unit_display_map[original_name])
                    else:
                        x_labels.append(original_name.replace('_', ' '))
                ax.set_xticklabels(x_labels, rotation=45, ha='center', fontsize=10)
                ax.set_xlabel('Unit', fontweight='bold', fontsize=14)
            
            ax.set_ylabel(grouping_col.replace("_", " "), fontweight='bold', fontsize=14)
            ax.set_title(f'Raw {grouping_col.replace("_", " ")} Values',
                       fontweight='bold', fontsize=16, pad=20)
            
            # Apply display_name_map to y-axis labels
            y_labels = []
            for label in ax.get_yticklabels():
                original_name = label.get_text()
                if display_name_map and original_name in display_name_map:
                    y_labels.append(display_name_map[original_name])
                else:
                    y_labels.append(original_name.replace('_', ' '))
            ax.set_yticklabels(y_labels, rotation=0, fontsize=12)
            
            # Style the axes with light gray borders to match separator lines
            for spine in ax.spines.values():
                spine.set_edgecolor('lightgray')
                spine.set_linewidth(1.4)

            plt.tight_layout()

            # Save figure
            output_path = os.path.join(output_dir, 'heatmap_raw_ratios.svg')
            plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)

            if show_plots:
                plt.show()
            else:
                plt.close()

            print(f"\nGenerated raw ratio heatmap in {output_dir}")
            return heatmap_data
        except Exception as e:
            print(f"Error creating raw ratio heatmap: {str(e)}")
            return {}


    def generate_unit_heatmaps(self, df, value_col, grouping_col, output_dir, data_type='percentage',
                               figsize=(8, 6), show_plots=False, cmap='RdBu_r', vmin=-2, vmax=2, display_name_map=None, unit_display_map=None, unit_order=None):
        """
        Generate heatmaps showing each molecule across units (aggregating across regions).
        
        This creates z-score normalized heatmaps where each row is a molecule/ratio type
        and each column is a unit, with values aggregated across all regions within each unit.
        
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
        cmap : str
            Colormap for heatmap
        vmin, vmax : float
            Min and max values for colormap normalization
        unit_order : list, optional
            Custom order for units on x-axis. Should be a list of unit names.
            If None, uses default ordering (units with both regions first, then single-region units)
            
        Returns:
        --------
        dict : Dictionary with heatmap matrices and statistics
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get unique molecules/ratio types and units
        unique_groups = df[grouping_col].unique()
        unique_units = df['Unit'].unique()
        
        if len(unique_groups) < 2 or len(unique_units) < 2:
            print(f"Insufficient data for unit heatmap: {len(unique_groups)} groups, {len(unique_units)} units")
            return {}
        
        # Check if Region column exists
        has_region = 'Region' in df.columns and df['Region'].notna().any()
        
        if has_region:
            # Calculate mean for each Unit-Region-Group combination
            agg_data = df.groupby(['Unit', 'Region', grouping_col])[value_col].agg(['mean', 'std', 'count']).reset_index()
            # Create a combined column label for Unit-Region using standardized delimiter
            agg_data['Unit_Region'] = agg_data['Unit'] + ' (' + agg_data['Region'] + ')'
        else:
            # Calculate mean for each Unit-Group combination (no regions)
            agg_data = df.groupby(['Unit', grouping_col])[value_col].agg(['mean', 'std', 'count']).reset_index()
            agg_data['Unit_Region'] = agg_data['Unit']
        
        # Remove NaN and infinite values
        agg_data = agg_data.replace([np.inf, -np.inf], np.nan)
        agg_data = agg_data.dropna(subset=['mean'])
        
        if len(agg_data) < 2:
            print("Insufficient data after removing NaN/inf values")
            return {}
        
        # Calculate z-scores normalized to each group's own mean across all unit-region combinations
        # For each molecule/ratio, normalize to its mean across all units (and regions if present)
        def normalize_group(group):
            group_mean = group['mean'].mean()
            group_std = group['mean'].std()
            if group_std == 0:
                group['z_score'] = 0
            else:
                group['z_score'] = (group['mean'] - group_mean) / group_std
            return group
        
        agg_data = agg_data.groupby(grouping_col, group_keys=False).apply(normalize_group)
        
        # Create pivot table for heatmap: Groups as rows, Unit_Region as columns
        try:
            heatmap_matrix = agg_data.pivot(index=grouping_col, columns='Unit_Region', values='z_score')
            
            # Fill missing unit-region combinations with -3 to show as extremely low
            heatmap_matrix = heatmap_matrix.fillna(-3)
            
            # Sort rows by display_name_map order if provided, otherwise alphabetically
            if display_name_map:
                # Get current index (molecule/ratio names)
                current_index = heatmap_matrix.index.tolist()
                # Create ordered list: first items in display_name_map order, then remaining alphabetically
                ordered_index = [k for k in display_name_map.keys() if k in current_index]
                remaining_index = sorted([g for g in current_index if g not in display_name_map.keys()])
                new_index = ordered_index + remaining_index
                heatmap_matrix = heatmap_matrix.reindex(new_index)
            else:
                # Sort rows alphabetically
                heatmap_matrix = heatmap_matrix.sort_index()
            
            # Sort columns based on unit_order if provided, otherwise use default ordering
            if unit_order is not None:
                # Custom ordering based on provided unit_order list
                all_columns = heatmap_matrix.columns.tolist()
                ordered_columns = []
                for unit in unit_order:
                    # Add all columns that start with this unit name
                    matching_cols = [col for col in all_columns if col.startswith(unit + ' (') or col == unit]
                    ordered_columns.extend(sorted(matching_cols))
                # Add any remaining columns not in unit_order
                remaining_cols = [col for col in all_columns if col not in ordered_columns]
                ordered_columns.extend(sorted(remaining_cols))
                heatmap_matrix = heatmap_matrix[ordered_columns]
            elif has_region:
                # Default ordering: units with both regions first, then single-region units
                all_columns = heatmap_matrix.columns.tolist()
                # Identify which units have both regions
                unit_region_map = {}  # {base_unit: [list of unit_region columns]}
                for col in all_columns:
                    if ' (' in col:
                        base_unit = col.split(' (')[0]
                        if base_unit not in unit_region_map:
                            unit_region_map[base_unit] = [col]
                        else:
                            unit_region_map[base_unit].append(col)
                    else:
                        if col not in unit_region_map:
                            unit_region_map[col] = [col]
                ordered_columns = []
                for base_unit, cols in unit_region_map.items():
                    if len(cols) > 1:
                        ordered_columns.extend(sorted(cols))
                for base_unit, cols in unit_region_map.items():
                    if len(cols) == 1:
                        ordered_columns.extend(cols)
                heatmap_matrix = heatmap_matrix[ordered_columns]
            
            # Store the data (no overall stats since each group normalized independently)
            heatmap_data = {
                'matrix': heatmap_matrix,
                'raw_data': agg_data
            }
            
            # Calculate dynamic figure size based on matrix dimensions
            n_rows = heatmap_matrix.shape[0]  # Number of molecules/ratios
            n_cols = heatmap_matrix.shape[1]  # Number of unit-region combinations
            # Scale: ~0.7 inches per column, ~0.35 inches per row, with min/max bounds for consistency
            fig_width = max(6, min(10, 0.7 * n_cols + 2.5))  # +2.5 for labels/colorbar
            fig_height = max(3, min(6, 0.35 * n_rows + 2))  # +2 for title/labels
            
            # Create heatmap visualization
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            
            # Create heatmap with journal-quality styling
            sns.heatmap(heatmap_matrix, annot=False, cmap=cmap, 
                       center=0, vmin=vmin, vmax=vmax,
                       cbar_kws={'label': 'Z-score (normalized to mean)', 'shrink': 0.8},
                       linewidths=0.5, linecolor='white', ax=ax, square=False)
            
            # Add visual separators between different units if showing regions
            if has_region:
                col_names = heatmap_matrix.columns.tolist()
                unit_changes = []
                prev_unit = None
                for idx, col_name in enumerate(col_names):
                    # Extract unit name (before the parenthesis)
                    current_unit = col_name.split(' (')[0] if ' (' in col_name else col_name
                    if prev_unit is not None and current_unit != prev_unit:
                        unit_changes.append(idx)
                    prev_unit = current_unit
                # Draw vertical lines at unit boundaries
                for boundary in unit_changes:
                    ax.axvline(x=boundary, color='gray', linewidth=1.4, zorder=10)
            
            # Formatting
            if has_region:
                # Update x-tick labels to use standardized format and apply unit_display_map
                current_labels = [label.get_text() for label in ax.get_xticklabels()]
                new_labels = []
                for label in current_labels:
                    # Extract unit name (before parenthesis)
                    if ' (' in label:
                        unit_part, region_part = label.split(' (', 1)
                        # Apply unit display map
                        if unit_display_map and unit_part in unit_display_map:
                            unit_part = unit_display_map[unit_part]
                        # Abbreviate region to first letter only
                        region_abbrev = region_part.rstrip(')')[0] if region_part.rstrip(')') else region_part
                        new_labels.append(f"{unit_part} ({region_abbrev})")
                    else:
                        # Apply unit display map to label without region
                        if unit_display_map and label in unit_display_map:
                            new_labels.append(unit_display_map[label])
                        else:
                            new_labels.append(label.replace('_', ' '))
                ax.set_xticklabels(new_labels, rotation=45, ha='center', fontsize=10)
                ax.set_xlabel('Unit (Region)', fontweight='bold', fontsize=14)
            else:
                # Apply unit_display_map to x-axis labels (without regions)
                x_labels = []
                for label in ax.get_xticklabels():
                    original_name = label.get_text()
                    if unit_display_map and original_name in unit_display_map:
                        x_labels.append(unit_display_map[original_name])
                    else:
                        x_labels.append(original_name.replace('_', ' '))
                ax.set_xticklabels(x_labels, rotation=45, ha='center', fontsize=10)
                ax.set_xlabel('Unit', fontweight='bold', fontsize=14)
            
            ax.set_ylabel(grouping_col.replace("_", " "), fontweight='bold', fontsize=14)
            # Update title based on data type
            if data_type == 'percentage':
                ax.set_title('Normalized Molecule Percentage',
                           fontweight='bold', fontsize=16, pad=20)
            else:
                ax.set_title('Normalized Ratio',
                           fontweight='bold', fontsize=16, pad=20)
            
            # Apply display_name_map to y-axis labels
            y_labels = []
            for label in ax.get_yticklabels():
                original_name = label.get_text()
                if display_name_map and original_name in display_name_map:
                    y_labels.append(display_name_map[original_name])
                else:
                    y_labels.append(original_name.replace('_', ' '))
            ax.set_yticklabels(y_labels, rotation=0, fontsize=12)
            
            # Style the axes with light gray borders to match separator lines
            for spine in ax.spines.values():
                spine.set_edgecolor('lightgray')
                spine.set_linewidth(1.4)
            
            plt.tight_layout()
            
            # Save figure
            output_path = os.path.join(output_dir, 
                                      f"heatmap_unit_aggregated_{data_type}.svg")
            plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
            
            if show_plots:
                plt.show()
            else:
                plt.close()
            
            print(f"\nGenerated unit-aggregated heatmap in {output_dir}")
            return heatmap_data
                
        except Exception as e:
            print(f"Error creating unit-aggregated heatmap: {str(e)}")
            return {}

    def generate_region_bubble_charts(self, df, value_col, grouping_col, output_dir=None,
                                       data_type='percentage', figsize=(10, 6), 
                                       show_plots=True, display_name_map=None, unit_display_map=None):
        """
        Generate bubble charts showing variability across regions.
        
        Creates separate bubble charts for each molecule/ratio type, where x-axis shows regions,
        y-axis shows units, and bubble size/color represent z-score normalized values.
        
        Args:
            df: DataFrame with columns [Unit, {Molecule or Ratio_Type}, {Percentage or Mean_Ratio}, Region, ...]
            value_col: Name of the value column ('Percentage' or 'Mean_Ratio')
            grouping_col: Name of the grouping column ('Molecule' or 'Ratio_Type')
            output_dir: Directory to save plots (if None, uses current directory)
            data_type: Type of data ('percentage' or 'ratio') for labeling
            figsize: Figure size for plots (width, height)
            show_plots: Whether to display plots interactively
            display_name_map: Optional mapping for display names
            unit_display_map: Optional mapping for unit abbreviations
            
        Returns:
            bubble_data: Dictionary with bubble chart data for each molecule/ratio type
        """
        # Create output directory if needed
        if output_dir is None:
            output_dir = 'region_bubble_charts_normalized'
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if Region column exists
        if 'Region' not in df.columns:
            print("Warning: Region column not found in DataFrame. Cannot generate bubble charts.")
            return {}
        
        # Get unique molecules/ratio types
        unique_groups = df[grouping_col].unique()
        
        # Sort by display_name_map order if provided, otherwise alphabetically
        if display_name_map:
            ordered_groups = [k for k in display_name_map.keys() if k in unique_groups]
            remaining = sorted([g for g in unique_groups if g not in display_name_map.keys()])
            sorted_groups = ordered_groups + remaining
        else:
            sorted_groups = sorted(unique_groups)
        
        bubble_data = {}
        
        for group_value in sorted_groups:
            # Filter data for this molecule/ratio type
            group_data = df[df[grouping_col] == group_value].copy()
            
            # Skip if insufficient data
            if len(group_data) < 2:
                continue
            
            # Remove NaN and infinite values
            group_data = group_data.replace([np.inf, -np.inf], np.nan)
            group_data = group_data.dropna(subset=[value_col])
            
            if len(group_data) < 2:
                continue
            
            # Calculate mean for each Unit-Region combination
            agg_data = group_data.groupby(['Unit', 'Region'])[value_col].agg(['mean', 'std', 'count']).reset_index()
            
            # Skip if not enough data
            if len(agg_data) < 2:
                continue
            
            # Calculate z-scores normalized to this molecule's mean
            molecule_mean = agg_data['mean'].mean()
            molecule_std = agg_data['mean'].std()
            
            if molecule_std == 0:
                print(f"Warning: {group_value} has zero variance, skipping...")
                continue
            
            agg_data['z_score'] = (agg_data['mean'] - molecule_mean) / molecule_std
            
            # Store the data
            bubble_data[group_value] = {
                'data': agg_data,
                'molecule_mean': molecule_mean,
                'molecule_std': molecule_std
            }
            
            # Create bubble chart
            fig, ax = plt.subplots(figsize=figsize)
            
            # Get unique regions and units
            regions = sorted(agg_data['Region'].unique())
            units = sorted(agg_data['Unit'].unique())
            
            # Create position mappings
            region_pos = {r: i for i, r in enumerate(regions)}
            unit_pos = {u: i for i, u in enumerate(units)}
            
            # Plot bubbles
            for _, row in agg_data.iterrows():
                x = region_pos[row['Region']]
                y = unit_pos[row['Unit']]
                z = row['z_score']
                
                # Bubble size proportional to absolute z-score
                size = np.abs(z) * 200 + 50  # Scale factor for visibility
                
                # Color based on sign of z-score
                color = '#d62728' if z > 0 else '#1f77b4'  # Red for positive, blue for negative
                
                ax.scatter(x, y, s=size, c=color, alpha=0.6, edgecolors='black', linewidth=1)
            
            # Set axis labels and ticks
            ax.set_xticks(range(len(regions)))
            ax.set_yticks(range(len(units)))
            
            # Apply display names
            region_labels = [r.replace('_', ' ') for r in regions]
            ax.set_xticklabels(region_labels, rotation=0, ha='center', fontsize=12)
            
            unit_labels = []
            for u in units:
                if unit_display_map and u in unit_display_map:
                    unit_labels.append(unit_display_map[u])
                else:
                    unit_labels.append(u.replace('_', ' '))
            ax.set_yticklabels(unit_labels, rotation=0, fontsize=12)
            
            # Labels and title
            ax.set_xlabel('Region', fontweight='bold', fontsize=14)
            ax.set_ylabel('Unit', fontweight='bold', fontsize=14)
            display_name = self._apply_display_name(group_value, display_name_map)
            ax.set_title(f'{display_name.replace("_", " ")}',
                       fontweight='bold', fontsize=16, pad=20)
            
            # Add legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', 
                      markersize=10, alpha=0.6, label='Above Mean', markeredgecolor='black'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', 
                      markersize=10, alpha=0.6, label='Below Mean', markeredgecolor='black')
            ]
            ax.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=10)
            
            # Grid
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            
            plt.tight_layout()
            
            # Save figure
            safe_name = group_value.replace('/', '_').replace(' ', '_')
            output_path = os.path.join(output_dir, f"bubble_chart_{safe_name}.svg")
            plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
            
            if show_plots:
                plt.show()
            else:
                plt.close()
        
        print(f"\nGenerated {len(bubble_data)} region bubble charts in {output_dir}")
        return bubble_data

    def generate_unit_bubble_charts(self, df, value_col, grouping_col, output_dir, data_type='percentage',
                                     show_plots=True, display_name_map=None, unit_display_map=None):
        """
        Generate unit-aggregated bubble charts (aggregating across regions).
        
        Creates a single bubble chart showing all molecules/ratios, where x-axis shows units,
        y-axis shows molecules/ratios, and bubble size/color represent z-score normalized values.
        
        Args:
            df: DataFrame with columns [Unit, {Molecule or Ratio_Type}, {Percentage or Mean_Ratio}, ...]
            value_col: Name of the value column
            grouping_col: Name of the grouping column
            output_dir: Directory to save plots
            data_type: Type of data ('percentage' or 'ratio')
            show_plots: Whether to display plots interactively
            display_name_map: Optional mapping for display names
            unit_display_map: Optional mapping for unit abbreviations
            
        Returns:
            bubble_data: Dictionary with bubble chart data
        """
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Remove NaN and infinite values
            clean_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[value_col])
            
            if len(clean_df) < 2:
                print("Insufficient data for unit-aggregated bubble chart")
                return {}
            
            # Aggregate across regions (if Region column exists)
            if 'Region' in clean_df.columns:
                agg_data = clean_df.groupby(['Unit', grouping_col])[value_col].agg(['mean', 'std', 'count']).reset_index()
            else:
                agg_data = clean_df.groupby(['Unit', grouping_col])[value_col].agg(['mean', 'std', 'count']).reset_index()
            
            # Calculate global z-scores (normalized across all units and molecules)
            global_mean = agg_data['mean'].mean()
            global_std = agg_data['mean'].std()
            
            if global_std == 0:
                print("Warning: Zero variance in data, cannot generate bubble chart")
                return {}
            
            agg_data['z_score'] = (agg_data['mean'] - global_mean) / global_std
            
            # Create bubble chart
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Get unique units and molecules
            units = sorted(agg_data['Unit'].unique())
            molecules = sorted(agg_data[grouping_col].unique())
            
            # Apply display name ordering if provided
            if display_name_map:
                ordered_molecules = [k for k in display_name_map.keys() if k in molecules]
                remaining = sorted([m for m in molecules if m not in display_name_map.keys()])
                molecules = ordered_molecules + remaining
            
            # Create position mappings
            unit_pos = {u: i for i, u in enumerate(units)}
            molecule_pos = {m: i for i, m in enumerate(molecules)}
            
            # Plot bubbles
            for _, row in agg_data.iterrows():
                x = unit_pos[row['Unit']]
                y = molecule_pos[row[grouping_col]]
                z = row['z_score']
                
                # Bubble size proportional to absolute z-score
                size = np.abs(z) * 200 + 50
                
                # Color based on sign of z-score
                color = '#d62728' if z > 0 else '#1f77b4'
                
                ax.scatter(x, y, s=size, c=color, alpha=0.6, edgecolors='black', linewidth=1)
            
            # Set axis labels and ticks
            ax.set_xticks(range(len(units)))
            ax.set_yticks(range(len(molecules)))
            
            # Apply display names
            unit_labels = []
            for u in units:
                if unit_display_map and u in unit_display_map:
                    unit_labels.append(unit_display_map[u])
                else:
                    unit_labels.append(u.replace('_', ' '))
            ax.set_xticklabels(unit_labels, rotation=45, ha='right', fontsize=10)
            
            molecule_labels = []
            for m in molecules:
                if display_name_map and m in display_name_map:
                    molecule_labels.append(display_name_map[m])
                else:
                    molecule_labels.append(m.replace('_', ' '))
            ax.set_yticklabels(molecule_labels, rotation=0, fontsize=12)
            
            # Labels and title
            ax.set_xlabel('Unit', fontweight='bold', fontsize=14)
            ax.set_ylabel(grouping_col.replace("_", " "), fontweight='bold', fontsize=14)
            
            if data_type == 'percentage':
                ax.set_title('Normalized Percentage', fontweight='bold', fontsize=16, pad=20)
            else:
                ax.set_title('Normalized Ratio', fontweight='bold', fontsize=16, pad=20)
            
            # Add legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', 
                      markersize=10, alpha=0.6, label='Above Mean', markeredgecolor='black'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', 
                      markersize=10, alpha=0.6, label='Below Mean', markeredgecolor='black')
            ]
            ax.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=10)
            
            # Grid
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            
            plt.tight_layout()
            
            # Save figure
            output_path = os.path.join(output_dir, f"bubble_chart_unit_aggregated_{data_type}.svg")
            plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
            
            if show_plots:
                plt.show()
            else:
                plt.close()
            
            print(f"\nGenerated unit-aggregated bubble chart in {output_dir}")
            return {'data': agg_data, 'global_mean': global_mean, 'global_std': global_std}
                
        except Exception as e:
            print(f"Error creating unit-aggregated bubble chart: {str(e)}")
            return {}

    def generate_raw_ratio_bubble_chart(self, df, value_col='Mean_Ratio', grouping_col='Ratio_Type', 
                                        output_dir=None, show_plots=True, display_name_map=None, 
                                        unit_display_map=None):
        """
        Generate raw (non-normalized) ratio bubble chart.
        
        Creates a bubble chart showing raw ratio values without z-score normalization.
        Bubble size represents the raw ratio value magnitude.
        
        Args:
            df: DataFrame with columns [Unit, Ratio_Type, Mean_Ratio, ...]
            value_col: Name of the value column
            grouping_col: Name of the grouping column
            output_dir: Directory to save plots
            show_plots: Whether to display plots interactively
            display_name_map: Optional mapping for display names
            unit_display_map: Optional mapping for unit abbreviations
            
        Returns:
            bubble_data: Dictionary with bubble chart data
        """
        if output_dir is None:
            output_dir = 'raw_ratio_bubble_charts'
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Remove NaN and infinite values
            clean_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[value_col])
            
            if len(clean_df) < 2:
                print("Insufficient data for raw ratio bubble chart")
                return {}
            
            # Aggregate across regions if Region column exists
            if 'Region' in clean_df.columns:
                agg_data = clean_df.groupby(['Unit', grouping_col])[value_col].agg(['mean', 'std', 'count']).reset_index()
            else:
                agg_data = clean_df.groupby(['Unit', grouping_col])[value_col].agg(['mean', 'std', 'count']).reset_index()
            
            # Create bubble chart
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Get unique units and ratio types
            units = sorted(agg_data['Unit'].unique())
            ratios = sorted(agg_data[grouping_col].unique())
            
            # Apply display name ordering if provided
            if display_name_map:
                ordered_ratios = [k for k in display_name_map.keys() if k in ratios]
                remaining = sorted([r for r in ratios if r not in display_name_map.keys()])
                ratios = ordered_ratios + remaining
            
            # Create position mappings
            unit_pos = {u: i for i, u in enumerate(units)}
            ratio_pos = {r: i for i, r in enumerate(ratios)}
            
            # Calculate global stats for sizing
            global_mean = agg_data['mean'].mean()
            global_max = agg_data['mean'].max()
            
            # Plot bubbles
            for _, row in agg_data.iterrows():
                x = unit_pos[row['Unit']]
                y = ratio_pos[row[grouping_col]]
                value = row['mean']
                
                # Bubble size proportional to raw value (normalized to max)
                if global_max > 0:
                    size = (value / global_max) * 500 + 50
                else:
                    size = 100
                
                # Color gradient based on value relative to mean
                if value > global_mean:
                    color = '#d62728'  # Red for above mean
                else:
                    color = '#1f77b4'  # Blue for below mean
                
                ax.scatter(x, y, s=size, c=color, alpha=0.6, edgecolors='black', linewidth=1)
            
            # Set axis labels and ticks
            ax.set_xticks(range(len(units)))
            ax.set_yticks(range(len(ratios)))
            
            # Apply display names
            unit_labels = []
            for u in units:
                if unit_display_map and u in unit_display_map:
                    unit_labels.append(unit_display_map[u])
                else:
                    unit_labels.append(u.replace('_', ' '))
            ax.set_xticklabels(unit_labels, rotation=45, ha='right', fontsize=10)
            
            ratio_labels = []
            for r in ratios:
                if display_name_map and r in display_name_map:
                    ratio_labels.append(display_name_map[r])
                else:
                    ratio_labels.append(r.replace('_', ' '))
            ax.set_yticklabels(ratio_labels, rotation=0, fontsize=12)
            
            # Labels and title
            ax.set_xlabel('Unit', fontweight='bold', fontsize=14)
            ax.set_ylabel('Ratio Type', fontweight='bold', fontsize=14)
            ax.set_title('Raw Ratio Values', fontweight='bold', fontsize=16, pad=20)
            
            # Add legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', 
                      markersize=10, alpha=0.6, label='Above Mean', markeredgecolor='black'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', 
                      markersize=10, alpha=0.6, label='Below Mean', markeredgecolor='black')
            ]
            ax.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=10)
            
            # Grid
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            
            plt.tight_layout()
            
            # Save figure
            output_path = os.path.join(output_dir, "bubble_chart_raw_ratios.svg")
            plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
            
            if show_plots:
                plt.show()
            else:
                plt.close()
            
            print(f"\nGenerated raw ratio bubble chart in {output_dir}")
            return {'data': agg_data, 'global_mean': global_mean, 'global_max': global_max}
                
        except Exception as e:
            print(f"Error creating raw ratio bubble chart: {str(e)}")
            return {}

    def generate_class_spectra_comparison_plots(
        self, dataset, prediction_dir, output_dir,
        srs_params_path='params_dataset/srs_params_61.npz',
        display_name_map=None,
        classes_to_exclude=None
    ):
        """
        Generate SVG plots comparing the standardized mean predicted spectrum
        (with std shading) against the clean reference molecule for each class.

        Args:
            dataset: HSI_Unlabeled_Dataset instance (provides raw spectra)
            prediction_dir: Path to rf_outputs directory containing per-image
                            subfolders with _predictions.csv files
            output_dir: Directory where SVG plots will be saved
            srs_params_path: Path to srs_params .npz file containing background
            display_name_map: Optional dict mapping molecule names to display names
            classes_to_exclude: Optional list of class names to skip (e.g. ['No Match'])
        """
        from core.hsi_normalization import spectral_standardization

        os.makedirs(output_dir, exist_ok=True)

        # Load background from srs_params
        srs_path = srs_params_path if srs_params_path.endswith('.npz') else srs_params_path + '.npz'
        srs_data = np.load(srs_path)
        background = srs_data['background']
        ch_start_val = int(srs_data['ch_start'])

        # Build wavenumber axis
        wavenumbers = np.linspace(self.wavenumber_start, self.wavenumber_end, self.num_samples)

        # Collect per-image prediction CSVs
        csv_paths = []
        for root, dirs, files in os.walk(prediction_dir):
            for f in files:
                if f.endswith('_predictions.csv'):
                    csv_paths.append(os.path.join(root, f))

        if not csv_paths:
            print(f"No prediction CSVs found in {prediction_dir}")
            return

        print(f"\nFound {len(csv_paths)} prediction CSV(s)")

        # Map image basenames to dataset image paths for quick lookup
        img_path_map = {}
        for img_path in dataset.img_list:
            basename_no_ext = os.path.splitext(os.path.basename(img_path))[0]
            img_path_map[basename_no_ext] = img_path

        # Accumulate raw spectra per class across all images
        class_spectra = {}  # {class_name: list of 1D spectra arrays}

        from tqdm import tqdm as tqdm_bar
        for csv_path in tqdm_bar(csv_paths, desc="Collecting spectra per class"):
            # Infer image name from CSV filename (e.g. "img_name_predictions.csv" -> "img_name")
            csv_basename = os.path.basename(csv_path)
            img_name_no_ext = csv_basename.replace('_predictions.csv', '')

            if img_name_no_ext not in img_path_map:
                print(f"  Warning: no dataset image for {img_name_no_ext}, skipping")
                continue

            img_path = img_path_map[img_name_no_ext]

            # Load predictions matrix (each cell is a molecule name string)
            pred_df = pd.read_csv(csv_path, header=None)
            pred_matrix = pred_df.values  # shape (height, width), dtype object/str

            # Load RAW spectra (no normalization) — spectral_standardization
            # handles its own normalization internally, so feeding pre-normalized
            # data would cause double-normalization artifacts
            image = tifffile.memmap(img_path, mode='r')
            image_spectra = image.reshape(image.shape[0], -1).T  # (n_pixels, n_channels)
            image_spectra = np.flip(image_spectra, axis=1).astype(np.float32)
            stats = dataset.image_stats[img_path]
            height = stats['height']
            width = stats['width']

            # Flatten predictions to match pixel ordering
            pred_flat = pred_matrix.flatten()  # length = height * width

            # Sanity check
            if len(pred_flat) != image_spectra.shape[0]:
                print(f"  Warning: size mismatch for {img_name_no_ext}: "
                      f"preds={len(pred_flat)}, spectra={image_spectra.shape[0]}, skipping")
                continue

            # Group pixel indices by predicted class
            unique_classes = np.unique(pred_flat)
            for cls_name in unique_classes:
                cls_name_str = str(cls_name)
                if classes_to_exclude and cls_name_str in classes_to_exclude:
                    continue
                mask = pred_flat == cls_name
                cls_spectra = image_spectra[mask]  # (n_matched_pixels, n_wavenumbers)
                if cls_name_str not in class_spectra:
                    class_spectra[cls_name_str] = []
                class_spectra[cls_name_str].append(cls_spectra)

        if not class_spectra:
            print("No spectra collected for any class.")
            return

        # Concatenate spectra per class and generate plots
        print(f"\nGenerating comparison plots for {len(class_spectra)} classes...")

        sorted_classes = sorted(class_spectra.keys())
        for cls_name in tqdm_bar(sorted_classes, desc="Generating SVGs"):
            spectra_list = class_spectra.pop(cls_name)  # pop to free memory progressively
            all_spectra = np.concatenate(spectra_list, axis=0)  # (N, n_wavenumbers)
            del spectra_list
            n_spectra = all_spectra.shape[0]

            if n_spectra < 2:
                print(f"  Skipping '{cls_name}': only {n_spectra} spectrum(a)")
                continue

            # Apply spectral_standardization to all spectra at once, then compute mean/std
            try:
                all_standardized = spectral_standardization(
                    all_spectra,
                    wavenum_1=self.wavenumber_start,
                    wavenum_2=self.wavenumber_end,
                    num_samp=self.num_samples,
                    background=background
                )
            except Exception as e:
                print(f"  Warning: standardization failed for '{cls_name}': {e}")
                all_standardized = all_spectra

            standardized_mean = np.mean(all_standardized, axis=0)
            standardized_std = np.std(all_standardized, axis=0)
            del all_spectra, all_standardized  # free memory

            # Find the matching reference molecule spectrum
            ref_spectrum = None
            mol_idx = None
            for i, mol_name in enumerate(self.molecule_names):
                if mol_name == cls_name:
                    mol_idx = i
                    break

            if mol_idx is not None:
                ref_raw = self.normalized_molecules[mol_idx]
                # Min-max normalize reference to [0, 1]
                ref_min = np.min(ref_raw)
                ref_max = np.max(ref_raw)
                if ref_max - ref_min > 1e-8:
                    ref_spectrum = (ref_raw - ref_min) / (ref_max - ref_min)
                else:
                    ref_spectrum = ref_raw

            # --- Plot ---
            fig, ax = plt.subplots(figsize=(8, 5))

            # Mean standardized spectrum
            ax.plot(wavenumbers, standardized_mean, color='#1f77b4', linewidth=1.5,
                    label=f'Mean predicted ({n_spectra:,} px)')

            # Std shading from standardized spectra
            ax.fill_between(
                wavenumbers,
                standardized_mean - standardized_std,
                standardized_mean + standardized_std,
                alpha=0.25, color='#1f77b4', label='± 1 std'
            )

            # Reference molecule
            if ref_spectrum is not None:
                ax.plot(wavenumbers, ref_spectrum, color='#d62728', linewidth=1.5,
                        linestyle='--', label='Reference')

            # Display name for title
            display_name = cls_name
            if display_name_map and cls_name in display_name_map:
                display_name = display_name_map[cls_name]

            ax.set_title(display_name, fontsize=14, fontweight='bold')
            ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=11)
            ax.set_ylabel('Intensity (a.u.)', fontsize=11)
            ax.legend(fontsize=9, frameon=True, edgecolor='0.8')
            ax.tick_params(labelsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            fig.tight_layout()

            # Save as SVG
            safe_name = cls_name.replace('/', '_').replace(' ', '_').replace(':', '_')
            svg_path = os.path.join(output_dir, f"{safe_name}.svg")
            fig.savefig(svg_path, format='svg', bbox_inches='tight')
            plt.close(fig)

        print(f"\nSaved {len(sorted_classes)} SVG plots to {output_dir}")
