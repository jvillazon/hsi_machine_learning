"""
HSI Model Trainer - Unified training interface for sklearn and PyTorch models.

This module provides a flexible training interface that works with:
- sklearn models (RandomForest, SVM, etc.)
- PyTorch neural networks
- Custom models with fit/predict interface

Uses HSI_Labeled_Dataset for synthetic data generation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
from tqdm import tqdm
import joblib
import os
import time

from core.hsi_labeled_dataset import HSI_Labeled_Dataset, create_dataloaders
from sk_classifier.debug_sk_classifier_stratified_kfold import compute_sam_weights    


class HSI_Trainer:
    """
    Unified trainer for HSI classification models.
    
    Supports both sklearn and PyTorch models with a consistent interface.
    """
    
    def __init__(self, dataset, model, model_type='sklearn'):
        """
        Initialize trainer.
        
        Parameters
        ----------
        dataset : HSI_Labeled_Dataset
            The dataset to use for training
        model : sklearn model or torch.nn.Module
            The model to train
        model_type : str
            'sklearn' or 'pytorch'
        """
        self.dataset = dataset
        self.model = model
        self.model_type = model_type.lower()
        self.calibrated_model = None  # Set by train_sklearn_classifier when use_platt=True
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'train_f1': [], 'val_f1': []}
        
        if self.model_type not in ['sklearn', 'pytorch']:
            raise ValueError("model_type must be 'sklearn' or 'pytorch'")

    @property
    def active_model(self):
        """Return the Platt-calibrated model if fitted, otherwise the base model."""
        return self.calibrated_model if self.calibrated_model is not None else self.model

    def loader_to_numpy(self, loader):
        X, y = zip(*[(X.numpy(), y.numpy()) for X, y in loader])
        return np.concatenate(X), np.concatenate(y)

    
    def train_sklearn_classifier(self, train_ratio=0.7, val_ratio=0.15, verbose=True, seed=42, use_platt=True, **kwargs):
        """
        Train sklearn model using synthetic dataset.
        
        Parameters
        ----------
        dataset : HSI_Labeled_Dataset
            Dataset for generating synthetic data
        train_ratio : float
            Fraction of data for training
        val_ratio : float
            Fraction of data for validation
        verbose : bool
            Print training progress
        seed : int
            Random seed for reproducibility
            
        Returns
        -------
        dict : Training metrics
        """

        # Create stratified split indices (balanced per class)
        train_size_per_class = int(self.dataset.num_samples_per_class * train_ratio)
        val_size_per_class = int(self.dataset.num_samples_per_class * val_ratio)
        
        if verbose:
            print("\nCreating stratified splits:")
            print(f"  Train: {train_size_per_class} samples per class ({train_size_per_class * self.dataset.n_molecules} total)")
            print(f"  Val:   {val_size_per_class} samples per class ({val_size_per_class * self.dataset.n_molecules} total)")
        
        train_loader, val_loader, test_loader = create_dataloaders(
            self.dataset,
            batch_size=256,
            train_ratio=0.7,
            val_ratio=0.15,
        seed=seed,
        )
            # Create train/val split (test set not used in hyperparameter search)
        self.X_train, self.y_train = self.loader_to_numpy(train_loader)
        self.X_val,   self.y_val   = self.loader_to_numpy(val_loader)
        self.X_test,  self.y_test  = self.loader_to_numpy(test_loader)

        # Timing info
        if verbose:
            start_time = time.time()

        # Train model
        if verbose:
            print("\nTraining model...")
        self.model.fit(self.X_train, self.y_train)

        # --- Platt scaling (post-hoc probability calibration) ---
        self.calibrated_model = None
        if use_platt:
            if verbose:
                print("\nFitting Platt scaling (sigmoid calibration) on validation set...")
            calib = CalibratedClassifierCV(estimator=self.model, cv='prefit', method='sigmoid')
            calib.fit(self.X_val, self.y_val)
            self.calibrated_model = calib
            if verbose:
                base_probs  = self.model.predict_proba(self.X_val)
                calib_probs = self.calibrated_model.predict_proba(self.X_val)
                # Multiclass Brier: mean sum-of-squared-errors over class probability vectors
                n_cls = len(self.model.classes_)
                one_hot = np.eye(n_cls)[self.y_val]  # (N, C)
                base_brier  = float(np.mean(np.sum((base_probs  - one_hot) ** 2, axis=1)))
                calib_brier = float(np.mean(np.sum((calib_probs - one_hot) ** 2, axis=1)))
                base_ll  = log_loss(self.y_val, base_probs)
                calib_ll = log_loss(self.y_val, calib_probs)
                print("  Platt calibration quality on validation set:")
                print(f"    Log-loss — Before: {base_ll:.4f}  After: {calib_ll:.4f}  (Δ {calib_ll - base_ll:+.4f})")
                print(f"    Brier    — Before: {base_brier:.4f}  After: {calib_brier:.4f}  (Δ {calib_brier - base_brier:+.4f})")

        if verbose:
            end_time = time.time()
        
        # Train and validation predictions (with optional SAM weighting)
        try:
            if kwargs['sam_weighting'] and 'alpha' in kwargs:

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

                # If SAM weights are provided, apply them to the validation predictions
                self.alpha = kwargs['alpha']
                self.reference_libray_array, label_array = self.dataset.get_reference()
                aligned_val_weights = compute_sam_weights(self.X_val, self.reference_libray_array)
                aligned_train_weights = compute_sam_weights(self.X_train, self.reference_libray_array)

                # Get base predictions and probabilities
                val_probs = self.active_model.predict_proba(self.X_val)
                aligned_weights = aligned_val_weights[:, self.model.classes_]  # Align SAM weights with model's class order
                weighted_val_probs = val_probs * (1 + self.alpha * aligned_weights)  # Simple weighting scheme
                val_pred = self.model.classes_[np.argmax(weighted_val_probs, axis=1)]

                train_prob = self.active_model.predict_proba(self.X_train)
                aligned_train_weights = aligned_train_weights[:, self.model.classes_]  # Align SAM weights with model's class order
                weighted_train_probs = train_prob * (1 + self.alpha * aligned_train_weights)
                train_pred = self.model.classes_[np.argmax(weighted_train_probs, axis=1)]
            else:
                self.reference_libray_array = None  # No SAM weighting used
                # No SAM weighting, just use base predictions
                train_pred = self.active_model.predict(self.X_train)
                val_pred = self.active_model.predict(self.X_val)
        
            train_acc = accuracy_score(self.y_train, train_pred)
            val_acc = accuracy_score(self.y_val, val_pred)

            train_score = f1_score(self.y_train, train_pred, average='weighted')
            val_score = f1_score(self.y_val, val_pred, average='weighted')
            
            if verbose:
                print(f"Training completed in {end_time - start_time:.2f} seconds")
                print(f"Training Accuracy: {train_acc:.4f}")
                print(f"Validation Accuracy: {val_acc:.4f}")
            
            return {
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'train_score': train_score,
                'val_score': val_score,
                'train_predictions': train_pred,
                'val_predictions': val_pred,
                'train_f1': train_score,
                'val_f1': val_score,
                'platt_calibrated': self.calibrated_model is not None,
            }
        except Exception as e:
            print(f"Error during training: {e}")
    
    def train_pytorch(self, train_ratio=0.7, val_ratio=0.15, 
                     batch_size=32, epochs=10, lr=0.001, device='cpu', verbose=True):
        """
        Train PyTorch model using synthetic dataset.
        
        Parameters
        ----------
        dataset : HSI_Labeled_Dataset
            Dataset for generating synthetic data
        train_ratio : float
            Fraction of data for training
        val_ratio : float
            Fraction of data for validation
        batch_size : int
            Batch size for training
        epochs : int
            Number of training epochs
        lr : float
            Learning rate
        device : str
            'cpu' or 'cuda'
        verbose : bool
            Print training progress
            
        Returns
        -------
        dict : Training history
        """
        
        if verbose:
            print("=" * 80)
            print("TRAINING PYTORCH MODEL")
            print("=" * 80)
        
        # Create dataloaders
        train_loader, val_loader, _ = create_dataloaders(
            self.dataset, batch_size=batch_size, 
            train_ratio=train_ratio, val_ratio=val_ratio
        )
        
        # Setup training
        self.model = self.model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', disable=not verbose)
            for spectra, labels in pbar:
                spectra, labels = spectra.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = self.model(spectra)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                if verbose:
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{100.*train_correct/train_total:.2f}%'
                    })
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for spectra, labels in val_loader:
                    spectra, labels = spectra.to(device), labels.to(device)
                    outputs = self.model(spectra)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            # Record metrics
            epoch_train_loss = train_loss / len(train_loader)
            epoch_train_acc = train_correct / train_total
            epoch_val_loss = val_loss / len(val_loader)
            epoch_val_acc = val_correct / val_total
            
            self.history['train_loss'].append(epoch_train_loss)
            self.history['train_acc'].append(epoch_train_acc)
            self.history['val_loss'].append(epoch_val_loss)
            self.history['val_acc'].append(epoch_val_acc)
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"train_loss: {epoch_train_loss:.4f}, train_acc: {epoch_train_acc:.4f}, "
                      f"val_loss: {epoch_val_loss:.4f}, val_acc: {epoch_val_acc:.4f}")
        
        if verbose:
            print("\n" + "=" * 80)
            print("TRAINING COMPLETE")
            print("=" * 80)
        
        return self.history
    
    def train(self, **kwargs):
        """
        Unified training interface - automatically dispatches to correct method.
        
        Parameters
        ----------
        dataset : HSI_Labeled_Dataset
            Dataset for training
        **kwargs : dict
            Training parameters (see train_sklearn or train_pytorch)
            
        Returns
        -------
        Training results
        """
        if self.model_type == 'sklearn':
            return self.train_sklearn(**kwargs)
        else:
            return self.train_pytorch(**kwargs)
    
    def evaluate(self, batch_size=32, device='cpu'):
        """
        Evaluate model on dataset.
        
        Parameters
        ----------
        dataset : HSI_Labeled_Dataset or DataLoader
            Data to evaluate on
        batch_size : int
            Batch size (only for PyTorch)
        device : str
            Device for PyTorch models
            
        Returns
        -------
        dict : Evaluation metrics
        """
        if self.model_type == 'sklearn':
            # Generate test data
            X_test, y_test = self.X_test, self.y_test
            test_pred = self.active_model.predict(X_test)

            if self.reference_libray_array is not None:
                # If SAM weights were used during training, apply them to test predictions
                test_probs = self.active_model.predict_proba(X_test)
                aligned_test_weights = compute_sam_weights(X_test, self.reference_libray_array)[:, self.model.classes_]
                weighted_test_probs = test_probs * (1 + self.alpha * aligned_test_weights)  
                test_pred = self.model.classes_[np.argmax(weighted_test_probs, axis=1)]
            else:
                test_pred = self.active_model.predict(X_test)

            acc = accuracy_score(y_test, test_pred)
            score = f1_score(y_test, test_pred, average='weighted')

            return {
                'accuracy': acc,
                'score': score,
                'predictions': test_pred,
                'true_labels': y_test,
                'confusion_matrix': confusion_matrix(y_test, test_pred),
                'classification_report': classification_report(y_test, test_pred)
            }

        else:
            # PyTorch evaluation
            if not isinstance(self.dataset, DataLoader):
                _, _, test_loader = create_dataloaders(self.dataset, batch_size=batch_size)
            else:
                test_loader = self.dataset
            
            self.model.eval()
            self.model = self.model.to(device)
            
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for spectra, labels in test_loader:
                    spectra = spectra.to(device)
                    outputs = self.model(spectra)
                    _, predicted = outputs.max(1)
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.numpy())
            
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            acc = accuracy_score(all_labels, all_preds)
            score = f1_score(all_labels, all_preds, average='weighted')
            return {
                'accuracy': acc,
                'score': score,
                'predictions': all_preds,
                'true_labels': all_labels,
                'confusion_matrix': confusion_matrix(all_labels, all_preds),
                'classification_report': classification_report(all_labels, all_preds)
            }
    
    def plot_confusion_matrix(self, results, class_names=None, figsize=(12, 10), 
                             normalize=False, save_path=None, title=None):
        """
        Plot confusion matrix from evaluation results.
        
        Parameters
        ----------
        results : dict
            Results dictionary from evaluate() containing 'confusion_matrix'
        class_names : list, optional
            List of class names for labels. If None, uses numeric labels
        figsize : tuple
            Figure size (width, height)
        normalize : bool
            If True, normalize confusion matrix to show percentages
        save_path : str, optional
            Path to save the figure. If None, displays only
        title : str, optional
            Custom title for the plot
            
        Returns
        -------
        fig, ax : matplotlib figure and axis objects
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        cm = results['confusion_matrix']
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            cbar_label = 'Normalized Frequency'
        else:
            fmt = 'd'
            cbar_label = 'Count'
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                   xticklabels=class_names if class_names is not None else 'auto',
                   yticklabels=class_names if class_names is not None else 'auto',
                   cbar_kws={'label': cbar_label},
                   ax=ax)
        
        if title is None:
            title = 'Confusion Matrix' + (' (Normalized)' if normalize else '')
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        # Rotate labels if they're long
        if class_names is not None:
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        
        return fig, ax
    
    def save(self, filepath):
        """
        Save trained model.
        
        Parameters
        ----------
        filepath : str
            Path to save model
        """
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        
        if self.model_type == 'sklearn':
            joblib.dump(self.model, filepath)
            print(f"Base sklearn model saved to: {filepath}")
            if self.calibrated_model is not None:
                platt_path = filepath.replace('.joblib', '_platt.joblib')
                joblib.dump(self.calibrated_model, platt_path)
                print(f"Platt-calibrated model saved to: {platt_path}")
                print(f"  → Set model_path='{platt_path}' in classify.py to use calibrated probabilities.")
        else:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'history': self.history
            }, filepath)
            print(f"PyTorch model saved to: {filepath}")
    
    def load(self, filepath):
        """
        Load trained model.
        
        Parameters
        ----------
        filepath : str
            Path to saved model
        """
        if self.model_type == 'sklearn':
            self.model = joblib.load(filepath)
            print(f"Sklearn model loaded from: {filepath}")
        else:
            checkpoint = torch.load(filepath)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.history = checkpoint.get('history', self.history)
            print(f"PyTorch model loaded from: {filepath}")


# Example usage functions
def train_random_forest(dataset, n_estimators=300, max_depth=None, **kwargs):
    """
    Train a Random Forest classifier.
    
    Parameters
    ----------
    dataset : HSI_Labeled_Dataset
        Training dataset
    n_estimators : int
        Number of trees
    max_depth : int or None
        Maximum tree depth
    **kwargs : dict
        Additional training parameters
        
    Returns
    -------
    HSI_Trainer : Trained model wrapper
    """
    from sklearn.ensemble import RandomForestClassifier
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    
    trainer = HSI_Trainer(model, model_type='sklearn')
    trainer.train(dataset, **kwargs)
    
    return trainer


def train_svm(dataset, C=1.0, kernel='rbf', **kwargs):
    """
    Train an SVM classifier.
    
    Parameters
    ----------
    dataset : HSI_Labeled_Dataset
        Training dataset
    C : float
        Regularization parameter
    kernel : str
        Kernel type
    **kwargs : dict
        Additional training parameters
        
    Returns
    -------
    HSI_Trainer : Trained model wrapper
    """
    from sklearn.svm import SVC
    
    model = SVC(C=C, kernel=kernel, random_state=42)
    
    trainer = HSI_Trainer(model, model_type='sklearn')
    trainer.train(dataset, **kwargs)
    
    return trainer


def train_neural_network(dataset, model, epochs=10, lr=0.001, batch_size=32, device='cpu', **kwargs):
    """
    Train a PyTorch neural network.
    
    Parameters
    ----------
    dataset : HSI_Labeled_Dataset
        Training dataset
    model : torch.nn.Module
        Neural network model
    epochs : int
        Number of training epochs
    lr : float
        Learning rate
    batch_size : int
        Batch size
    device : str
        'cpu' or 'cuda'
    **kwargs : dict
        Additional training parameters
        
    Returns
    -------
    HSI_Trainer : Trained model wrapper
    """
    trainer = HSI_Trainer(model, model_type='pytorch')
    trainer.train(dataset, epochs=epochs, lr=lr, batch_size=batch_size, device=device, **kwargs)
    
    return trainer


