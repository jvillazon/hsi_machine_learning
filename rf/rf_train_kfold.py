"""
Train Random Forest classifiers using k-fold cross-validation with hyperparameter tuning
"""

import numpy as np
from core.hsi_labeled_dataset import HSI_Labeled_Dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
from tqdm import tqdm
import os


def train_rf_with_kfold(dataset, param_grid, n_folds=5, train_val_ratio=0.85, 
                        save_best_model=True, output_dir='models'):
    """
    Train Random Forest with k-fold cross-validation and hyperparameter tuning
    
    Parameters:
    -----------
    dataset : HSI_Labeled_Dataset
        The dataset to use for training
    param_grid : dict
        Dictionary of parameters to search over
        Example: {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    n_folds : int
        Number of folds for cross-validation (default: 5)
    train_val_ratio : float
        Ratio of data to use for training+validation vs test (default: 0.85)
    save_best_model : bool
        Whether to save the best model (default: True)
    output_dir : str
        Directory to save outputs (default: 'models')
        
    Returns:
    --------
    dict : Dictionary containing:
        - 'best_params': Best parameters found
        - 'best_score': Best cross-validation score
        - 'cv_results': Full cross-validation results
        - 'test_accuracy': Accuracy on held-out test set
        - 'test_results': Full test set results
        - 'best_model': The trained best model
    """
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    print("\n" + "=" * 80)
    print("K-FOLD CROSS-VALIDATION WITH HYPERPARAMETER TUNING")
    print("=" * 80)
    
    # Create stratified split indices (balanced per class)
    train_val_size_per_class = int(dataset.num_samples_per_class * train_val_ratio)
    test_size_per_class = dataset.num_samples_per_class - train_val_size_per_class
    
    print("\nCreating stratified splits:")
    print(f"  Train+Val: {train_val_size_per_class} samples per class ({train_val_size_per_class * dataset.n_molecules} total)")
    print(f"  Test:      {test_size_per_class} samples per class ({test_size_per_class * dataset.n_molecules} total)")
    
    np.random.seed(42)
    train_val_indices = []
    test_indices = []
    
    for mol_idx in range(dataset.n_molecules):
        # All indices for this class
        class_start = mol_idx * dataset.num_samples_per_class
        class_indices = np.arange(class_start, class_start + dataset.num_samples_per_class)
        
        # Shuffle and split
        np.random.shuffle(class_indices)
        train_val_indices.extend(class_indices[:train_val_size_per_class])
        test_indices.extend(class_indices[train_val_size_per_class:])
    
    # Shuffle the combined indices
    np.random.shuffle(train_val_indices)
    np.random.shuffle(test_indices)
    
    # Generate training+validation data
    print(f"\nGenerating {len(train_val_indices)} train+val samples...")
    X_train_val = []
    y_train_val = []
    for i in tqdm(train_val_indices, desc="Loading train+val data"):
        spec, label = dataset[i]
        X_train_val.append(spec.numpy())
        y_train_val.append(label)
    X_train_val = np.array(X_train_val)
    y_train_val = np.array(y_train_val)
    
    # Generate test data
    print(f"Generating {len(test_indices)} test samples...")
    X_test = []
    y_test = []
    for i in tqdm(test_indices, desc="Loading test data"):
        spec, label = dataset[i]
        X_test.append(spec.numpy())
        y_test.append(label)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print(f"\nDataset split complete:")
    print(f"  Training + Validation: {len(X_train_val)} samples")
    print(f"  Test set: {len(X_test)} samples")
    print(f"\nParameter grid:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    print(f"\nNumber of folds: {n_folds}")
    
    # Create the base model
    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Setup stratified k-fold cross-validation
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Setup GridSearchCV
    print(f"\nTotal parameter combinations to test: ", end="")
    n_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"{n_combinations}")
    print(f"Total model fits: {n_combinations * n_folds}")
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )
    
    # Perform grid search
    print("\n" + "=" * 80)
    print("PERFORMING GRID SEARCH")
    print("=" * 80)
    grid_search.fit(X_train_val, y_train_val)
    
    # Print results
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 80)
    print(f"\nBest parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"\nBest cross-validation score: {grid_search.best_score_:.4f}")
    
    # Get top 5 parameter combinations
    cv_results = grid_search.cv_results_
    top_indices = np.argsort(cv_results['mean_test_score'])[-5:][::-1]
    
    print("\nTop 5 parameter combinations:")
    print("-" * 80)
    for i, idx in enumerate(top_indices, 1):
        params = cv_results['params'][idx]
        mean_score = cv_results['mean_test_score'][idx]
        std_score = cv_results['std_test_score'][idx]
        print(f"\n{i}. Score: {mean_score:.4f} (+/- {std_score:.4f})")
        for param, value in params.items():
            print(f"   {param}: {value}")
    
    # Evaluate best model on test set
    print("\n" + "=" * 80)
    print("EVALUATING BEST MODEL ON TEST SET")
    print("=" * 80)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    class_report = classification_report(y_test, y_pred, 
                                         target_names=dataset.molecule_names)
    print(class_report)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    print("\n" + "=" * 80)
    print("GENERATING CONFUSION MATRIX")
    print("=" * 80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Unnormalized confusion matrix
    fig1, ax1 = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=dataset.molecule_names,
                yticklabels=dataset.molecule_names,
                cbar_kws={'label': 'Count'},
                ax=ax1)
    ax1.set_xlabel('Predicted Label', fontweight='bold', fontsize=12)
    ax1.set_ylabel('True Label', fontweight='bold', fontsize=12)
    ax1.set_title('Random Forest Confusion Matrix (K-Fold CV)', 
                  fontweight='bold', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = f'plots/rf_kfold_confusion_matrix_{timestamp}.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig2, ax2 = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=dataset.molecule_names,
                yticklabels=dataset.molecule_names,
                cbar_kws={'label': 'Proportion'},
                ax=ax2)
    ax2.set_xlabel('Predicted Label', fontweight='bold', fontsize=12)
    ax2.set_ylabel('True Label', fontweight='bold', fontsize=12)
    ax2.set_title('Random Forest Confusion Matrix - Normalized (K-Fold CV)',
                  fontweight='bold', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_norm_path = f'plots/rf_kfold_confusion_matrix_normalized_{timestamp}.png'
    plt.savefig(cm_norm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nConfusion matrices saved to:")
    print(f"  - {cm_path}")
    print(f"  - {cm_norm_path}")
    
    # Plot parameter importance (how much each parameter affects performance)
    print("\n" + "=" * 80)
    print("ANALYZING PARAMETER IMPORTANCE")
    print("=" * 80)
    
    # Save best model if requested
    if save_best_model:
        model_path = os.path.join(output_dir, f'rf_kfold_best_{timestamp}.joblib')
        joblib.dump(best_model, model_path)
        print(f"\nBest model saved to: {model_path}")
        
        # Save parameter search results
        results_path = os.path.join(output_dir, f'rf_kfold_results_{timestamp}.joblib')
        joblib.dump(grid_search.cv_results_, results_path)
        print(f"CV results saved to: {results_path}")
    
    # Return results
    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': cv_results,
        'test_accuracy': test_accuracy,
        'test_results': {
            'y_true': y_test,
            'y_pred': y_pred,
            'confusion_matrix': cm,
            'classification_report': class_report
        },
        'best_model': best_model,
        'grid_search': grid_search
    }


if __name__ == '__main__':
    
    # Load dataset
    print("=" * 80)
    print("LOADING DATASET")
    print("=" * 80)
    
    dataset = HSI_Labeled_Dataset(
        molecule_dataset_path='molecule_dataset/lipid_subtype_CH_61',
        srs_params_path='params_dataset/srs_params_61',
        num_samples_per_class=1000,  # Reduced from 20000 for faster training
        normalize_per_molecule=False,
        compute_min_max=True
    )
    
    # Visualize dataset samples
    print("\n" + "=" * 80)
    print("VISUALIZING DATASET SAMPLES")
    print("=" * 80)
    dataset.visualize_dataset_samples(train_ratio=0.7, val_ratio=0.15, num_samples_per_class=3)
    
    # Define parameter grid for hyperparameter tuning
    # You can customize this based on your needs
    
    # Option 1: Comprehensive search (slower but more thorough)
    param_grid_comprehensive = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Option 2: Quick search (faster, fewer parameters)
    param_grid_quick = {
        'n_estimators': [200, 300],
        'max_depth': [None, 20, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    # Option 3: Custom focused search (recommended to start)
    param_grid_focused = {
        'n_estimators': [400, 500],
        'max_depth': [20, 30],
        'min_samples_split': [2],
        'min_samples_leaf': [2]
    }
    
    # Choose which parameter grid to use
    param_grid = param_grid_focused
    
    # Run k-fold cross-validation with hyperparameter tuning
    results = train_rf_with_kfold(
        dataset=dataset,
        param_grid=param_grid,
        n_folds=5,
        train_val_ratio=0.85,
        save_best_model=True,
        output_dir='models'
    )
    
    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"\nBest Cross-Validation Score: {results['best_score']:.4f}")
    print(f"Test Set Accuracy: {results['test_accuracy']:.4f}")
    print(f"\nBest Parameters:")
    for param, value in results['best_params'].items():
        print(f"  {param}: {value}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
