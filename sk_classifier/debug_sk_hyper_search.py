
import os
import numpy as np


import sys 
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.hsi_labeled_dataset import HSI_Labeled_Dataset, create_dataloaders

import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, log_loss

import time

import optuna.visualization.matplotlib as vis
import matplotlib.pyplot as plt


def loader_to_numpy(loader):
    X, y = zip(*[(X.numpy(), y.numpy()) for X, y in loader])
    return np.concatenate(X), np.concatenate(y)

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

def create_objective(X_train, y_train, X_val, y_val, sam_weights, label_array):
    # Reserve 20% of training data for Platt calibration in every trial.
    # The same deterministic split is reused across trials so calibration quality
    # is comparable and the val set stays fully held-out.
    cal_split = int(len(X_train) * 0.8)
    X_fit,  y_fit  = X_train[:cal_split], y_train[:cal_split]
    X_cal,  y_cal  = X_train[cal_split:], y_train[cal_split:]

    def objective(trial):
        # Hyperparameters ---
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 5, 30)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
        alpha = trial.suggest_float('alpha', 0.1, 2.0)  # Controls SAM weighting influence
        
        # Start time
        start_time = time.time()

        # 1. Fit base RF on the 80% fit set
        rf = RandomForestClassifier(n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf,
                                    n_jobs=-1)
        rf.fit(X_fit, y_fit)

        # 2. Platt scaling: fit sigmoid layer on the reserved 20% calibration set
        calibrated_rf = CalibratedClassifierCV(rf, cv='prefit', method='sigmoid')
        calibrated_rf.fit(X_cal, y_cal)

        # 3. Get calibrated probabilities on the held-out validation set
        probs = calibrated_rf.predict_proba(X_val)

        # Log calibration quality as trial user attributes (visible in Optuna dashboard)
        base_probs = rf.predict_proba(X_val)
        trial.set_user_attr('log_loss_base',  round(log_loss(y_val, base_probs), 6))
        trial.set_user_attr('log_loss_platt', round(log_loss(y_val, probs),      6))

        # rf.classes_ are integer indices into label_array — use directly to reorder SAM columns
        aligned_sam_weights = sam_weights[:, rf.classes_]  # (n_val, n_classes), aligned

        # Apply alpha as a power to the weights to control 'strictness'
        weighted_probs = probs * (aligned_sam_weights ** alpha)
        
        # Re-normalize and get final predictions
        weighted_probs = np.nan_to_num(weighted_probs, nan=0.0)  # Handle any NaNs
        final_preds = rf.classes_[np.argmax(weighted_probs, axis=1)]  # Map back to class labels
        
        # End time measurement
        end_time = time.time()

        # Using Macro F1-score is better for many lipid subtypes (imbalanced data)
        score = f1_score(y_val, final_preds, average='macro')
        elapsed_time = end_time - start_time
        
        return score, elapsed_time
    return objective

def print_current_score(study, frozen_trial):
    # This is the score of the trial that JUST finished
    current_score, current_elapsed_time = frozen_trial.values
    print(f"Trial {frozen_trial.number} Score: {current_score}, Time: {current_elapsed_time:.2f} seconds")

def main():
    
    dataset = HSI_Labeled_Dataset(
        molecule_dataset_path='molecule_dataset/lipid_subtype_wn_61_test',
        srs_params_path='params_dataset/srs_params_61',
        num_samples_per_class=10000,
        normalize_per_molecule=False,
        compute_min_max=True,
        noise_multiplier=0.5
    )

    

    train_loader, val_loader, test_loader = create_dataloaders(
        dataset,
        batch_size=4096,   # large batch = fewer Python iterations
        train_ratio=0.7,
        val_ratio=0.15,
        seed=42,
    )

    
    # Create train/val split (test set not used in hyperparameter search)
    X_train, y_train = loader_to_numpy(train_loader)
    X_val,   y_val   = loader_to_numpy(val_loader)


    reference_library_array, label_array = dataset.get_reference()
    sam_weights = compute_sam_weights(X_val, reference_library_array)

    objective = create_objective(X_train, y_train, X_val, y_val, sam_weights, label_array)

    study = optuna.create_study(directions=["maximize", "minimize"])
    study.optimize(objective, n_trials=50, callbacks=[print_current_score])

    print(f"Top models on the Pareto Front: {len(study.best_trials)}")

    for t in study.best_trials:
        print(f"Trial {t.number}: F1={t.values[0]:.4f}, Time={t.values[1]:.4f}s | Params: {t.params}")

    output_dir = "hp_search_plots"
    os.makedirs(output_dir, exist_ok=True)


    # 1. Pareto Front (The most important plot for Multi-Objective)
    # This shows the trade-off between F1-Score (Objective 0) and Time (Objective 1)
    fig_pareto = vis.plot_pareto_front(study, target_names=["F1-Score", "Inference Time"])
    fig_pareto.get_figure().savefig(os.path.join(output_dir, "lipid_pareto_front.png"), dpi=200, bbox_inches="tight")

    # 2. Parameter Importance for F1-Score
    # This identifies which hyperparameters (like alpha vs. max_depth) drive accuracy
    fig_param_f1 = vis.plot_param_importances(study, target=lambda t: t.values[0], target_name="F1-Score")
    fig_param_f1.get_figure().savefig(os.path.join(output_dir, "param_importance_f1.png"), bbox_inches="tight")

    # 3. Optimization History
    # Shows how the search converged
    fig_history = vis.plot_optimization_history(study, target=lambda t: t.values[0], target_name="F1-Score")
    fig_history.get_figure().savefig(os.path.join(output_dir, "optimization_history_f1.png"), bbox_inches="tight")

    # 4. Contour Plot (Interaction between Alpha and Prob_Threshold)
    # This helps visualize the 'Goldilocks zone' for your SAM-weighting logic
    fig_contour = vis.plot_contour(study, params=["min_samples_split", "max_depth"], target=lambda t: t.values[0])
    fig_contour.get_figure().savefig(os.path.join(output_dir, "contour_min_samples_split_vs_max_depth.png"), bbox_inches="tight")

if __name__ == "__main__":
    main()