import os
import shutil
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import shap
import joblib
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import sys 
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.hsi_normalization import spectral_standardization
import matplotlib.colors as _mc


# ---------------------------------------------------------------------------
# Lipid superclass color palette
#
# Each entry:  (superclass_name, [keywords_to_match], base_hex_color)
# Matching is case-insensitive substring on the class label string.
# Classes not matching any entry are grouped under "Other" (#999999).
# To add / rename superclasses edit this list only.
# ---------------------------------------------------------------------------
_LIPID_SUPERCLASSES = [
    ("Phospholipids",
     ["PC", "PE", "PS", "PI", "PG", "Cardiolipin", "CDP",
      "LPA", "LPC", "LPE", "LPS", "LPI", "LPG",
      "DOPC", "DOPE", "DSPC", "DPPC", "Mix"],
     "#4477AA"),   # blue

    ("Sterols",
     ["Cholesterol", "CE(", "Sterol"],
     "#CCBB44"),   # gold

    ("Glycerolipids",
     ["TAG", "DAG", "TG(", "DG(", "Glycerol"],
     "#228833"),   # green

    ("Sphingolipids",
     ["SM", "Sphingosine", "GlcCer", "GalCer",
      "Cer", "LacCer", "Glucosylceramide"],
     "#AA3377"),   # magenta

    ("Fatty Acids",
     ["Stearic", "Palmitic", "DHA", "TPA",
      "Fatty",  "Acid",
      "Lactate", "Glucose"],
     "#EE6677"),   # red-orange

    ("Carnitine",
        ["Carnitine", "CAR"],
        "#FF5733"),   # bright red
    
    ("Background",
     ["Background", "background"],
        "#000000"),  # black
    ("No Match",
     ["No Match", "no match"],
     "#BBBBBB"),   # light gray

     
]


def _get_superclass(class_label: str):
    """Return (superclass_name, base_hex_color) for a class label."""
    label_upper = str(class_label).upper()
    for sc_name, keywords, color in _LIPID_SUPERCLASSES:
        for kw in keywords:
            if kw.upper() in label_upper:
                return sc_name, color
    return "Other", "#999999"


def _lighten(hex_color: str, factor: float) -> str:
    """Interpolate hex_color toward white by factor (0 = original, 1 = white)."""
    rgb = np.array(_mc.to_rgb(hex_color))
    return _mc.to_hex(np.clip(rgb + (1.0 - rgb) * factor, 0, 1))


def _superclass_color_palette(class_labels):
    """
    Build a {class_label: hex_color} palette.

    Classes that share a superclass receive tonal variants of the superclass
    base color (darkest → lightest for first → last member of each group).
    Classes not matching any superclass are colored with sequential grays.
    """
    groups = {}  # sc_name → {'color': hex, 'labels': [...]}
    for lbl in class_labels:
        sc_name, sc_color = _get_superclass(str(lbl))
        if sc_name not in groups:
            groups[sc_name] = {'color': sc_color, 'labels': []}
        groups[sc_name]['labels'].append(lbl)

    palette = {}
    for info in groups.values():
        base = info['color']
        labels = info['labels']
        n = len(labels)
        for i, lbl in enumerate(labels):
            # Spread [0%, 45%] lighter; base color is the darkest shade
            palette[lbl] = _lighten(base, i / max(n, 1) * 0.45)
    return palette

def aggregate_shap_results(results_list):
    """
    Aggregate a list of per-image dicts returned by compute_shap_values into a
    single dict with the same structure as a single compute_shap_values return value.

    Aggregation rules
    -----------------
    shap_values            : list of (n_samples_total, n_features) arrays per class
                             — rows concatenated across images
    class_shap             : dict {cls: (n_samples_total, n_features)}
                             — rows concatenated across images
    feature_importance_by_class : dict {cls: (n_features,)}
                             — mean of per-image importance arrays
    expected_value         : scalar mean (or array mean) across images
    spectra_subset         : (n_samples_total, n_features) — rows concatenated
    """
    if not results_list:
        return {}

    # ---- shap_values (list of per-class 2-D arrays) ----------------------
    n_classes = len(results_list[0]['shap_values'])
    agg_shap_values = [
        np.concatenate([r['shap_values'][i] for r in results_list], axis=0)
        for i in range(n_classes)
    ]

    # ---- class_shap (dict {cls_name: 2-D array}) -------------------------
    all_cls_names = list(results_list[0]['class_shap'].keys())
    agg_class_shap = {
        cls: np.concatenate([r['class_shap'][cls] for r in results_list], axis=0)
        for cls in all_cls_names
    }

    # ---- feature_importance_by_class (dict {cls_name: 1-D array}) --------
    agg_importance = {
        cls: np.mean(
            np.stack([r['feature_importance_by_class'][cls] for r in results_list], axis=0),
            axis=0
        )
        for cls in all_cls_names
    }

    # ---- expected_value (scalar or 1-D array) ----------------------------
    ev_list = [r['expected_value'] for r in results_list]
    agg_expected = np.mean(np.stack(ev_list, axis=0), axis=0)

    # ---- spectra_subset (2-D array) --------------------------------------
    agg_spectra = np.concatenate([r['spectra_subset'] for r in results_list], axis=0)

    return {
        'shap_values': agg_shap_values,
        'class_shap': agg_class_shap,
        'expected_value': agg_expected,
        'feature_importance_by_class': agg_importance,
        'spectra_subset': agg_spectra,
    }

class HSI_Classifier:
    def __init__(self, dataset, visualizer, model_path, labeled_dataset, output_base=None):
        """
        Initialize HSI Classifier
        
        Args:
            dataset: HSI_Unlabeled_Dataset instance
            visualizer: HSI_Visualizer instance for results visualization
            model_path: Path to the saved .joblib classifier model
            labeled_dataset: HSI_Labeled_Dataset instance for reference spectra for weighting
            output_base: Optional base directory for saving outputs

        """
        self.dataset = dataset
        self.visualizer = visualizer
        self.labeled_dataset = labeled_dataset
        # Load the classifier model
        print(f"\nLoading classifier model from {model_path}")
        self.model = joblib.load(model_path)

        # Save outputs if requested
        if output_base is None:
            try:
                parent_dir = os.path.abspath(os.path.join(self.dataset.img_list[0], os.pardir, os.pardir))
            except Exception:
                parent_dir = os.path.abspath(os.path.join(os.getcwd()))
            self.output_base = os.path.join(parent_dir, f'{os.path.basename(model_path).split(".")[0]}_outputs')
        else:
            self.output_base = output_base


    def predict(self, generate_shap=True, shap_n_samples=500, shap_background=100, alpha=None):
        """
        Perform classifier inference by processing each image individually.
        
        After base model predictions, probabilities are refined in two stages:
          1. SAM spectral weighting (batch_spectral_weighting)
          2. SAM-weighted SHAP feature importance weighting (shap_probability_weighting)
        A SHAP report is saved per image when generate_shap=True.
        
        Args:
            generate_shap: bool - whether to compute and save SAM-weighted SHAP per image
            shap_n_samples: int - number of pixel spectra to sample for SHAP computation
            shap_background: int - number of background samples for the SHAP explainer
            
        Returns:
            predictions: numpy array of shape (n_samples,) containing class predictions
            probabilities: numpy array of shape (n_samples, n_classes) containing class probabilities
            timing_stats: dictionary containing timing statistics
        """
        import time
        self.background = self.labeled_dataset.background
        labeled_spectra, labels = self.labeled_dataset.get_reference()
        predictions = []
        probabilities = []
        self.alpha = alpha

        if generate_shap:
            shap_global_results = []

        timing_stats = {
            'image_times': [],
            'start_time': datetime.now(),
        }
        
        print(f"\nProcessing {len(self.dataset.img_list)} images...")

        # Create output directory if it doesn't exist
        os.makedirs(self.output_base, exist_ok=True)

        # Process each image
        for img_idx, img_path in enumerate(tqdm(self.dataset.img_list, desc="Processing images")):
            img_start = time.time()
            
            # Get image statistics and shape
            img_name = os.path.basename(img_path)
            print(f"Processing image: {img_name}")
            stats = self.dataset.image_stats[img_path]
            height = stats['height']
            width = stats['width']
            image_shape = (height, width)

            # Load and process image spectra
            image_spectra = self.dataset.load_and_process_image(img_path)

            # Visualize image spectra
            def predict_chunk(chunk):
                probs = self.model.predict_proba(chunk)
                preds = self.model.predict(chunk)
                return preds, probs
            
            # Split data into chunks for parallel processing
            chunk_size = 100000  # Adjust based on memory
            n_chunks = int(np.ceil(len(image_spectra) / chunk_size))
            chunks = np.array_split(image_spectra, n_chunks)
            
            # Process chunks in parallel with progress bar
            with tqdm(total=n_chunks, desc="Processing chunks in parallel") as pbar:
                def update_progress(*a):
                    pbar.update(1)
                    return None
                    
                results = Parallel(n_jobs=-1, backend="threading")(
                    delayed(predict_chunk)(chunk) for chunk in chunks
                )
            
            # Combine results
            img_preds = np.concatenate([r[0] for r in results])
            img_probs = np.concatenate([r[1] for r in results])
            
            if self.alpha is not None:
                final_probs = self.batch_spectral_weighting(img_probs, image_spectra, labeled_spectra)
                final_probs = final_probs.astype(np.float32)
                final_preds = self.model.classes_[np.argmax(final_probs, axis=1)]
            else:
                final_probs = img_probs
                final_probs = final_probs.astype(np.float32)
                final_preds = img_preds

            # Flag differences between original and SAM-weighted predictions
            differences = img_preds != final_preds
            num_differences = np.sum(differences)
            print(f"\nNumber of spectra with different predictions after spectral weighting: {num_differences}")

            # Apply SHAP probability refinement for this image and then aggregate feature importance across images if requested
            if generate_shap:
                shap_results = self.compute_shap_values(
                    spectra=image_spectra,
                    labels=labels,
                    background_samples=shap_background,
                    n_samples=shap_n_samples,
                    approximate=True,
                    shap_batch_size=1000,
                    n_jobs=-1
                )
                shap_global_results.append(shap_results)

            probabilities.append(final_probs)
            predictions.append(final_preds)
        

            img_folder = os.path.join(self.output_base, os.path.splitext(img_name)[0])
            os.makedirs(img_folder, exist_ok=True)

            # # Save SHAP report for this image to aggregate feature importance across images later
            # if shap_results is not None:
            #     shap_dir = os.path.join(img_folder, 'shap_report')
            #     # Always recreate shap_report to remove stale results
            #     if os.path.exists(shap_dir):
            #         shutil.rmtree(shap_dir)
            #     os.makedirs(shap_dir)
            #     wavenumbers = np.linspace(self.dataset.wavenumber_start, self.dataset.wavenumber_end, self.dataset.num_samples)
            #     importance_df = self.get_global_feature_importance(shap_results, wavenumbers)
            #     importance_df.to_csv(
            #         os.path.join(shap_dir, 'shap_importance.csv'), index=False
            #     )
            #     print(f"Saved SHAP feature importance to {shap_dir}")
            #     plt.ioff()
            #     self.plot_shap_feature_importance(
            #         shap_results, wavenumbers,
            #         output_path=os.path.join(shap_dir, 'shap_importance.png')
            #     )
            #     plt.close('all')
            #     if wavenumbers is not None:
            #         # Choose random class index to plot (or 0 if out of range)
            #         random_class_indeces = np.random.choice(len(shap_results['shap_values']), size=min(3, len(shap_results['shap_values'])), replace=False)
            #         for cls_idx in random_class_indeces:
            #             self.plot_shap_spectra(
            #                 shap_results, wavenumbers, class_idx=cls_idx,
            #                 output_path=os.path.join(shap_dir, f'shap_spectrum_class_{cls_idx}.png')
            #             )
            #             plt.close('all')

            csv_path = os.path.join(img_folder, f"{os.path.splitext(img_name)[0]}_predictions.csv")
            # Save predictions DataFrame as CSV
            self.visualizer.create_prediction_csv(
                img_predictions=final_preds,
                img_shape=image_shape,
                img_path=img_path,
                output_path=csv_path,
            )
            

            # Reshape to (height, width, n_classes)
            n_classes = final_probs.shape[1]
            prob_reshaped = final_probs.reshape((height, width, n_classes))
            # Assuming "no match" is the last class
            no_match_slice = prob_reshaped[:, :, -1]
            # Create mask where no_match probability > 0.1
            no_match_mask = no_match_slice > 0.1
            # Invert the mask (True where we want to KEEP the probabilities)
            keep_mask = ~no_match_mask
            # Apply inverted mask to all slices EXCEPT the "no match" slice
            # Expand mask to match probability dimensions
            keep_mask_expanded = keep_mask[:, :, np.newaxis]  # Shape: (height, width, 1)
            prob_masked = prob_reshaped.copy()
            # Multiply all slices except the last one (no match) by the inverted mask
            prob_masked[:, :, :-1] = prob_masked[:, :, :-1] * keep_mask_expanded
            # The "no match" slice (last one) remains unchanged
            # Reshape back to original format (n_pixels, n_classes)
            img_probs_masked = prob_masked.reshape(-1, n_classes)

            # Save probabilities TIFF
            tiff_path = os.path.join(img_folder, f"{os.path.splitext(img_name)[0]}_probabilities.tif")
            self.visualizer.create_probability_stack(
                img_probabilities=img_probs_masked,
                img_shape=image_shape,
                img_path=img_path,
                output_path=tiff_path,
                stats=stats
            )

            # Update timing statistics
            img_end = time.time()
            img_time = img_end - img_start
            timing_stats['image_times'].append({
                'image_idx': img_idx,
                'image_name': os.path.basename(img_path),
                'total_time': img_time,
                'num_spectra': len(image_spectra)
            })
            
            # Print progress for this image
            spectra_per_sec = len(image_spectra) / img_time
            print(f"\nImage {img_idx + 1}/{len(self.dataset.img_list)}:")
            print(f"Processed {len(image_spectra)} spectra in {img_time:.2f}s")
            print(f"Processing speed: {spectra_per_sec:.2f} spectra/second")


        if generate_shap:
            # Aggregate SHAP results across all images into a single structure
            # that matches the layout returned by compute_shap_values.
            shap_global_results = aggregate_shap_results(shap_global_results)
            shap_dir = os.path.join(self.output_base, 'shap_report_global')
            # if os.path.exists(shap_dir):
            #     shutil.rmtree(shap_dir)
            os.makedirs(shap_dir, exist_ok=True)
            wavenumbers = np.linspace(self.dataset.wavenumber_start, self.dataset.wavenumber_end, self.dataset.num_samples)
            importance_df = self.get_global_feature_importance(shap_global_results, wavenumbers)
            importance_df.to_csv(os.path.join(shap_dir, 'global_feature_importance.csv'), index=False)
            print(f"\nSaved global SHAP feature importance to {shap_dir}")
            plt.ioff()
            self.plot_shap_feature_importance(
                    shap_global_results, wavenumbers,
                    output_path=os.path.join(shap_dir, 'shap_importance.png')
                )
            plt.close('all')
            if wavenumbers is not None:
                # Choose random class index to plot (or 0 if out of range)
                random_class_indeces = np.random.choice(len(shap_global_results['shap_values']), size=min(3, len(shap_global_results['shap_values'])), replace=False)
                for cls_idx in random_class_indeces:
                    self.plot_shap_spectra(
                        shap_global_results, wavenumbers, class_idx=cls_idx,
                        output_path=os.path.join(shap_dir, f'shap_spectrum_class_{cls_idx}.png')
                    )
                    plt.close('all')
                
        # Concatenate results
        # predictions = np.concatenate(predictions)
        # probabilities = np.concatenate(probabilities)

        # Compute aggregate statistics
        total_spectra = sum(x['num_spectra'] for x in timing_stats['image_times'])
        total_processing_time = sum(x['total_time'] for x in timing_stats['image_times'])
        avg_speed = total_spectra / total_processing_time if total_processing_time > 0 else 0
        
        print("\nProcessing Summary:")
        print(f"Total spectra processed: {total_spectra}")
        print(f"Total processing time: {total_processing_time:.2f}s")
        print(f"Average processing speed: {avg_speed:.2f} spectra/second")
        
        # return predictions, probabilities, timing_stats
    

    
    def batch_spectral_weighting(self, img_probs, image_spectra, labeled_spectra):
        """
        Apply spectral weighting to probabilities in batches to manage memory usage.
        
        Args:
            img_probs: numpy array of shape (n_pixels, n_classes) containing class probabilities
            image_spectra: numpy array of shape (n_pixels, n_wavenumbers) containing spectra for the image
            labeled_spectra: numpy array of shape (n_reference, n_wavenumbers) containing reference spectra
        Returns:
            weighted_probs: numpy array of shape (n_pixels, n_classes) containing spectral weighted probabilities
        """
        
        n_pixels = img_probs.shape[0]
        weighted_probs = np.zeros_like(img_probs)
        chunk_size = 100000  # Adjust based on memory
        n_chunks = int(np.ceil(n_pixels / chunk_size))

        for i in tqdm(range(n_chunks), desc="Standardize, SAM, and weight in chunks"):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, n_pixels)
            spectra_chunk = image_spectra[start_idx:end_idx]
            prob_chunk = img_probs[start_idx:end_idx]
            # Standardize
            try:
                standardized_chunk = spectral_standardization(
                    spectra_chunk,
                    wavenum_1=self.visualizer.wavenumber_start,
                    wavenum_2=self.visualizer.wavenumber_end,
                    num_samp=self.visualizer.num_samples,
                    background=self.background
                )
            except Exception as e:
                print(f"  Warning: standardization failed for chunk {i}: {e}")
                standardized_chunk = spectra_chunk
            # Compute SAM weights for this chunk
            sam_weights_chunk = compute_sam_weights(standardized_chunk, labeled_spectra)
            aligned_sam_weights_chunk = sam_weights_chunk[:, self.model.classes_]
            # Apply weights to probabilities
            weighted_prob_chunk = np.zeros_like(prob_chunk)
            for j in range(prob_chunk.shape[1]):
                weighted_prob_chunk[:, j] = prob_chunk[:, j] * self.alpha * aligned_sam_weights_chunk[:, j % aligned_sam_weights_chunk.shape[1]]
            weighted_probs[start_idx:end_idx] = weighted_prob_chunk
        weighted_probs = np.nan_to_num(weighted_probs, nan=0.0)
        return weighted_probs

    def compute_shap_values(self, spectra, labels, background_samples=100, n_samples=None,
                             approximate=True, shap_batch_size=1000, n_jobs=-1):
        """
        Compute SHAP values for spectral data using the classifier model.
        
        Args:
            spectra: numpy array of shape (n_samples, n_features) - spectral data to explain
            background_samples: int - number of background samples for SHAP explainer
                                 (only used for non-tree models)
            n_samples: int - optional limit on samples to explain (for memory efficiency)
            approximate: bool - use faster approximate SHAP for tree models (default True)
            shap_batch_size: int - number of samples per parallel batch (default 1000)
            n_jobs: int - number of parallel jobs; -1 uses all cores (default -1)
            
        Returns:
            dict containing shap_values, expected_value, feature_importance_by_class, spectra_subset
        """
        # Subsample if needed for memory efficiency
        if n_samples is not None and len(spectra) > n_samples:
            indices = np.random.choice(len(spectra), n_samples, replace=False)
            spectra_subset = spectra[indices]
        else:
            spectra_subset = spectra

        # Unwrap CalibratedClassifierCV to access the underlying estimator for SHAP.
        # calibrated_classifiers_[0].estimator holds the base model for sklearn >= 1.2;
        # older versions use .base_estimator instead.
        from sklearn.calibration import CalibratedClassifierCV
        base_model = self.model
        if isinstance(self.model, CalibratedClassifierCV):
            try:
                base_model = self.model.calibrated_classifiers_[0].estimator
            except AttributeError:
                base_model = self.model.calibrated_classifiers_[0].base_estimator
            print("Detected CalibratedClassifierCV — using base estimator for TreeExplainer.")

        is_tree_model = hasattr(base_model, 'estimators_')

        if is_tree_model:
            # tree_path_dependent: no background data needed and faster than interventional
            print("Creating TreeExplainer (tree_path_dependent, no background required)...")
            explainer = shap.TreeExplainer(
                base_model,
                feature_perturbation="tree_path_dependent"
            )

            # Split into batches and compute in parallel using threading backend
            n_batches = max(1, int(np.ceil(len(spectra_subset) / shap_batch_size)))
            batches = np.array_split(spectra_subset, n_batches)
            print(f"Computing SHAP values for {len(spectra_subset)} samples "
                  f"in {n_batches} batch(es) "
                  f"(approximate={approximate}, n_jobs={n_jobs})...")

            def _compute_batch(batch):
                return explainer.shap_values(
                    batch,
                    check_additivity=False,  # skip validation pass — ~30-50% faster
                    approximate=approximate  # tree-path approximation when True
                )

            if n_jobs != 1 and n_batches > 1:
                batch_results = Parallel(n_jobs=n_jobs, backend="threading")(
                    delayed(_compute_batch)(batch)
                    for batch in tqdm(batches, desc="SHAP batches")
                )
            else:
                batch_results = [
                    _compute_batch(batch)
                    for batch in tqdm(batches, desc="SHAP batches")
                ]

            # Concatenate across batches
            if isinstance(batch_results[0], list):
                shap_values = [
                    np.concatenate([r[i] for r in batch_results], axis=0)
                    for i in range(len(batch_results[0]))
                ]
            else:
                shap_values = np.concatenate(batch_results, axis=0)

            # Normalize to list of (n_samples, n_features) arrays per class.
            # Newer SHAP versions return a 3D ndarray instead of a list:
            #   (n_samples, n_features, n_classes)  — SHAP >= 0.40
            #   (n_classes, n_samples, n_features)  — older 3D format
            n_actual_features = spectra_subset.shape[1]
            if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                if shap_values.shape[2] == n_actual_features:
                    # Shape is (n_classes, n_samples, n_features)
                    shap_values = [shap_values[i] for i in range(shap_values.shape[0])]
                elif shap_values.shape[1] == n_actual_features:
                    # Shape is (n_samples, n_features, n_classes)
                    shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
                else:
                    print(f"Warning: unexpected SHAP array shape {shap_values.shape} "
                          f"(expected n_features={n_actual_features}). "
                          f"Converting along axis 0.")
                    shap_values = [shap_values[i] for i in range(shap_values.shape[0])]

            expected_value = explainer.expected_value

        else:
            # For SVM or other non-tree models use KernelExplainer with background data
            if len(spectra_subset) > background_samples:
                background_idx = np.random.choice(len(spectra_subset), background_samples, replace=False)
                background_data = spectra_subset[background_idx]
            else:
                background_data = spectra_subset

            print(f"Creating KernelExplainer with {len(background_data)} background samples...")
            explainer = shap.KernelExplainer(self.model.predict_proba, background_data)
            print(f"Computing SHAP values for {len(spectra_subset)} samples...")
            shap_values = explainer.shap_values(spectra_subset)
            expected_value = explainer.expected_value

        # Use model's class ordering — never np.unique(labels), which sorts alphabetically
        # and can differ from self.model.classes_.
        # self.model.classes_ contains integer indices (mol_idx from __getitem__).
        # Map each index to its molecule name string so downstream dicts are
        # keyed by human-readable names (needed for superclass color matching).
        ordered_classes = self.model.classes_
        mol_names = self.labeled_dataset.molecule_names  # shape (n_classes,)
        class_shap = {}
        feature_importance_by_class = {}

        # Compute feature importance by class
        for cls_idx, cls in enumerate(ordered_classes):
            # Resolve human-readable molecule name from integer class index
            cls_name = mol_names[cls] if cls < len(mol_names) else str(cls)

            if isinstance(shap_values, list):
                cls_shap = shap_values[cls_idx] if cls_idx < len(shap_values) else shap_values[-1]
            else:
                cls_shap = shap_values[cls_idx]

            feature_importance = np.mean(np.abs(cls_shap), axis=0)
            class_shap[cls_name] = cls_shap
            feature_importance_by_class[cls_name] = feature_importance

        return {
            'shap_values': shap_values,
            'class_shap': class_shap,
            'expected_value': expected_value,
            'feature_importance_by_class': feature_importance_by_class,
            'spectra_subset': spectra_subset
        }


    @staticmethod
    def _is_ignored_class(cls_name: str) -> bool:
        """Return True for classes that should be excluded from SHAP/importance displays."""
        name = str(cls_name)
        return name == 'No Match' or 'Background' in name

    def get_global_feature_importance(self, shap_results, wavenumbers=None):
        """
        Compute global feature importance SHAP results.
        
        Args:
            shap_results: dict returned by compute_shap_values
            wavenumbers: optional array of wavenumber values for feature names
            
        Returns:
            DataFrame with feature importance per class and overall
        """
        feature_importance_by_class = {
            k: v for k, v in shap_results['feature_importance_by_class'].items()
            if not self._is_ignored_class(k)
        }
        n_features = len(next(iter(feature_importance_by_class.values())))
        
        # Create feature names — guard against length mismatch between wavenumbers and SHAP output
        if wavenumbers is not None and len(wavenumbers) == n_features:
            feature_names = [f'{wn:.1f} cm⁻¹' for wn in wavenumbers]
        else:
            if wavenumbers is not None and len(wavenumbers) != n_features:
                print(f"Warning: wavenumbers length ({len(wavenumbers)}) != n_features ({n_features}). "
                      f"Falling back to generic feature names.")
            feature_names = [f'Feature_{i}' for i in range(n_features)]
        
        # Build DataFrame
        data = {'Feature': feature_names}
        for cls, importance in feature_importance_by_class.items():
            data[f'Class_{cls}'] = importance
            
        df = pd.DataFrame(data)
        
        # Add overall importance (mean across classes)
        class_cols = [c for c in df.columns if c.startswith('Class_')]
        df['Overall'] = df[class_cols].mean(axis=1)
        
        # Sort by overall importance
        df = df.sort_values('Overall', ascending=False).reset_index(drop=True)
        
        return df

    def plot_shap_feature_importance(self, shap_results, wavenumbers=None, 
                                      top_n=20, output_path=None, figsize=(22, 14)):
        """
        Plot SHAP feature importance by class.
        
        Args:
            shap_results: dict returned by compute_shap_values
            wavenumbers: optional array of wavenumber values for feature names
            top_n: number of top features to display
            output_path: optional path to save the figure
            figsize: tuple for figure size
        """
        importance_df = self.get_global_feature_importance(shap_results, wavenumbers)
        
        # Get top features by overall importance
        top_df = importance_df.head(top_n)
        
        # Prepare data for stacked bar chart
        class_cols = [c for c in top_df.columns if c.startswith('Class_')]

        # Filter out ignored classes (No Match, Background) from the stacked bars
        class_cols = [c for c in class_cols if not self._is_ignored_class(c.replace('Class_', ''))]

        # Sort classes so same-superclass segments are contiguous in the stack
        raw_labels = [col.replace('Class_', '') for col in class_cols]
        sorted_pairs = sorted(
            zip(class_cols, raw_labels),
            key=lambda t: (_get_superclass(t[1])[0], t[1])
        )
        class_cols_sorted, raw_labels_sorted = zip(*sorted_pairs) if sorted_pairs else ([], [])

        # Build superclass-aware color palette from the sorted label order so that
        # tonal shading goes darkest → lightest in the order bars are drawn
        color_palette = _superclass_color_palette(list(raw_labels_sorted))

        # Separator gap = 1% of the widest total bar
        total_values = top_df[list(class_cols_sorted)].sum(axis=1).values
        #sep_width = max(total_values.max() * 0.01, 1e-9)
        sep_width = 0

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Plot 1: Stacked bar chart of class contributions
        ax1 = axes[0]
        x = np.arange(len(top_df))
        bottom = np.zeros(len(top_df))

        # Track one legend handle per superclass (avoid duplicates)
        seen_superclasses = {}
        prev_sc = None

        for col, lbl in zip(class_cols_sorted, raw_labels_sorted):
            sc_name, _ = _get_superclass(lbl)
            color = color_palette[lbl]

            # Thin white spacer between superclass groups
            if prev_sc is not None and sc_name != prev_sc:
                ax1.barh(x, np.full(len(x), sep_width), left=bottom,
                         color='white', linewidth=0)
                bottom += sep_width

            ax1.barh(x, top_df[col].values, left=bottom,
                     label=f"{lbl} ({sc_name})", color=color)
            bottom += top_df[col].values

            # One representative patch per superclass for the grouped legend
            if sc_name not in seen_superclasses:
                seen_superclasses[sc_name] = mpatches.Patch(
                    color=color, label=sc_name
                )
            prev_sc = sc_name

        # Build legend: one patch per superclass
        group_handles = list(seen_superclasses.values())

        ax1.set_yticks(x)
        ax1.set_yticklabels(top_df['Feature'].values, fontsize=16)
        ax1.set_xlabel('SHAP Importance', fontsize=20)
        ax1.set_title(f'Top {top_n} Features by Class', fontsize=24)
        ax1.tick_params(axis='x', labelsize=16)
        ax1.legend(
            handles=group_handles,
            labels=[h.get_label() for h in group_handles],
            title='Species Type',
            loc='lower right',
            fontsize=16,
            title_fontsize=20
        )
        ax1.invert_yaxis()
        
        # Plot 2: Overall importance
        ax2 = axes[1]
        ax2.barh(x, top_df['Overall'].values, color='steelblue')
        ax2.set_yticks(x)
        ax2.set_yticklabels(top_df['Feature'].values, fontsize=16)
        ax2.set_xlabel('Mean SHAP Importance', fontsize=20)
        ax2.set_title(f'Top {top_n} Features (Overall)', fontsize=24)
        ax2.tick_params(axis='x', labelsize=16)
        ax2.invert_yaxis()
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved feature importance plot to {output_path}")
        
        plt.show()
        
        return fig

    def plot_shap_spectra(self, shap_results, wavenumbers, class_idx=0, 
                          sample_idx=0, output_path=None):
        """
        Plot SHAP explanation overlaid on spectrum for a specific sample.
        
        Args:
            shap_results: dict returned by compute_shap_values
            wavenumbers: array of wavenumber values
            class_idx: class index to explain
            sample_idx: sample index to explain
            output_path: optional path to save the figure
        """
        # Reference spectra for class
        spectra, labels = self.labeled_dataset.get_reference()

        # Resolve molecule name from model's integer class index
        cls_int = self.model.classes_[class_idx] if class_idx < len(self.model.classes_) else self.model.classes_[-1]
        cls = self.labeled_dataset.molecule_names[cls_int] if cls_int < len(self.labeled_dataset.molecule_names) else str(cls_int)

        # Find the reference spectrum for this class
        matching = spectra[labels == cls]
        if len(matching) == 0:
            print(f"Warning: no reference spectrum found for class '{cls}'. Using first spectrum.")
            spectrum = spectra[0]
        else:
            spectrum = matching[0]

        shap_values = shap_results['class_shap']

        if cls not in shap_values:
            print(f"Warning: class '{cls}' not found in SHAP results. Available: {list(shap_values.keys())}")
            cls = list(shap_values.keys())[class_idx] if class_idx < len(shap_values) else list(shap_values.keys())[0]

        shap_vals = shap_values[cls][sample_idx]
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        # Plot spectrum
        ax1 = axes[0]
        ax1.plot(wavenumbers, spectrum, 'b-', linewidth=2)
        ax1.set_ylabel('Intensity', fontsize=20)
        ax1.set_title(f'Spectrum for Class {cls}', fontsize=18)
        ax1.tick_params(axis='both', labelsize=16)
        ax1.grid(True, alpha=0.3)
        
        # Plot SHAP values with color coding
        ax2 = axes[1]
        colors = np.where(shap_vals > 0, 'red', 'blue')
        ax2.bar(wavenumbers, shap_vals, color=colors, width=(wavenumbers[1]-wavenumbers[0])*0.8)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Wavenumber (cm⁻¹)', fontsize=20)
        ax2.set_ylabel('SHAP Value', fontsize=20)
        ax2.set_title(f'Feature Attribution for Class {cls}', fontsize=24)
        ax2.tick_params(axis='both', labelsize=16)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved SHAP spectra plot to {output_path}")
        
        plt.show()
        
        return fig

    def generate_shap_report(self, shap_results, wavenumbers=None, n_samples=500, 
                              background_samples=100, output_dir=None):
        """
        Generate a comprehensive SHAP explanation report.
        
        Args:
            wavenumbers: array of wavenumber values for feature names
            n_samples: number of samples to use for SHAP computation
            background_samples: number of background samples for SHAP explainer
            output_dir: directory to save report outputs
            
        Returns:
            dict containing all SHAP analysis results
        """
        if self.labeled_dataset is None:
            raise ValueError("labeled_dataset required for SHAP analysis")
        
        # Get reference spectra and labels
        labeled_spectra, labels = self.labeled_dataset.get_reference()
        
        # Get sample of unlabeled spectra for explanation
        img_path = self.dataset.img_list[0]
        image_spectra = self.dataset.load_and_process_image(img_path)
        
        print(f"Computing SHAP values for {n_samples} samples...")
        
        
        # Generate outputs
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save feature importance
            importance_df = self.get_global_feature_importance(shap_results, wavenumbers)
            csv_path = os.path.join(output_dir, 'shap_importance.csv')
            importance_df.to_csv(csv_path, index=False)
            print(f"Saved feature importance to {csv_path}")
            
            # Save importance plot
            plot_path = os.path.join(output_dir, 'shap_importance.png')
            self.plot_shap_feature_importance(shap_results, wavenumbers, 
                                               output_path=plot_path)
            
            # Save example SHAP spectra plots
            if wavenumbers is not None:
                for cls_idx in range(min(3, len(shap_results['shap_values']))):
                    spec_path = os.path.join(output_dir, f'shap_spectrum_class_{cls_idx}.png')
                    self.plot_shap_spectra(shap_results, wavenumbers, 
                                           class_idx=cls_idx, output_path=spec_path)
        
        return shap_results
    
def compute_sam_weights(query_spectra, reference_spectra):
    """
    Compute Spectral Angle Mapper (SAM) similarity weights between query and reference spectra.
    
    Args:
        query_spectra: numpy array of shape (n_query, n_features)
        reference_spectra: numpy array of shape (n_reference, n_features)
        
    Returns:
        sam_weights: numpy array of shape (n_query, n_reference) containing SAM-based weights
    """
    # Normalize spectra
    query_norm = query_spectra / (np.linalg.norm(query_spectra, axis=1, keepdims=True) + 1e-10)  # Add small value to avoid division by zero
    ref_norm = reference_spectra / (np.linalg.norm(reference_spectra, axis=1, keepdims=True) + 1e-10)
    
    # Compute cosine similarity via dot product
    similarity = np.dot(query_norm, ref_norm.T)
    
    # Clip to valid range and compute spectral angle
    cos_sim_matrix = np.clip(similarity, -1, 1)
    thetas = np.arccos(cos_sim_matrix)
    
    # Convert angles to weights (smaller angle = higher weight)
    # Normalized to [0, 1] where 1 is perfect match
    sam_weights = 1 - (thetas / np.pi)
    
    return sam_weights