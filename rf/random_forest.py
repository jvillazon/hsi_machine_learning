import os
import numpy as np
# import pandas as pd
from datetime import datetime
from tqdm import tqdm
import joblib
from joblib import Parallel, delayed

class HSI_RandomForest:
    def __init__(self, dataset, visualizer, output_base=None):
        """
        Initialize HSI Random Forest Classifier
        
        Args:
            dataset: HSI_Unlabeled_Dataset instance
            visualizer: Optional HSI_Visualizer instance for results visualization
        """
        self.dataset = dataset
        self.visualizer = visualizer
        # Save outputs if requested
        if output_base is None:
            try:
                parent_dir = os.path.abspath(os.path.join(self.dataset.img_list[0], os.pardir, os.pardir))
            except Exception:
                parent_dir = os.path.abspath(os.path.join(os.getcwd()))
            self.output_base = os.path.join(parent_dir, 'rf_outputs')


    def predict(self, model_path, output_base=None):
        """
        Perform Random Forest inference by processing each image individually
        
        Args:
            model_path: Path to the saved .joblib Random Forest model
            output_base: Optional base directory for saving outputs
                        If None, will create 'rf_outputs' next to first image
            
        Returns:
            predictions: numpy array of shape (n_samples,) containing class predictions
            probabilities: numpy array of shape (n_samples, n_classes) containing class probabilities
            timing_stats: dictionary containing timing statistics
        """
        import time
        
        # Load the Random Forest model
        print(f"\nLoading Random Forest model from {model_path}")
        rf_model = joblib.load(model_path)
        
        predictions = []
        probabilities = []
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
                probs = rf_model.predict_proba(chunk)
                preds = rf_model.predict(chunk)
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
            
            predictions.append(img_preds)
            probabilities.append(img_probs)

            # Flip predictions and probabilities if needed
            # img_preds = np.flip(img_preds, axis=0)  # Flip if needed
            # img_probs = np.flip(img_probs, axis=0)  # Flip if needed

            img_folder = os.path.join(self.output_base, os.path.splitext(img_name)[0])
            os.makedirs(img_folder, exist_ok=True)

            csv_path = os.path.join(img_folder, f"{os.path.splitext(img_name)[0]}_predictions.csv")
            # Save predictions DataFrame as CSV
            self.visualizer.create_prediction_csv(
                img_predictions=img_preds, 
                img_shape=image_shape,
                img_path=img_path,
                output_path=csv_path,   
            )
            
            # Apply "no match" mask to probabilities before saving
            # Reshape to (height, width, n_classes)
            n_classes = img_probs.shape[1]
            prob_reshaped = img_probs.reshape((height, width, n_classes))
            
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
            
            tiff_path = os.path.join(img_folder, f"{os.path.splitext(img_name)[0]}_probabilities.tif")
            # Save probabilities TIFF
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
        
        # Concatenate results
        predictions = np.concatenate(predictions)
        probabilities = np.concatenate(probabilities)
        
        # Compute aggregate statistics
        total_spectra = sum(x['num_spectra'] for x in timing_stats['image_times'])
        total_processing_time = sum(x['total_time'] for x in timing_stats['image_times'])
        avg_speed = total_spectra / total_processing_time if total_processing_time > 0 else 0
        
        print("\nProcessing Summary:")
        print(f"Total spectra processed: {total_spectra}")
        print(f"Total processing time: {total_processing_time:.2f}s")
        print(f"Average processing speed: {avg_speed:.2f} spectra/second")
        
        return predictions, probabilities, timing_stats
    

