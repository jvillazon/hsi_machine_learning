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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import joblib
import os
import time


class HSI_Trainer:
    """
    Unified trainer for HSI classification models.
    
    Supports both sklearn and PyTorch models with a consistent interface.
    """
    
    def __init__(self, model, model_type='sklearn'):
        """
        Initialize trainer.
        
        Parameters
        ----------
        model : sklearn model or torch.nn.Module
            The model to train
        model_type : str
            'sklearn' or 'pytorch'
        """
        self.model = model
        self.model_type = model_type.lower()
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        if self.model_type not in ['sklearn', 'pytorch']:
            raise ValueError("model_type must be 'sklearn' or 'pytorch'")
    
    def train_sklearn(self, dataset, train_ratio=0.7, val_ratio=0.15, verbose=True, seed=42):
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
        if verbose:
            print("=" * 80)
            print("TRAINING SKLEARN MODEL")
            print("=" * 80)
        
        # Create stratified split indices (balanced per class)
        train_size_per_class = int(dataset.num_samples_per_class * train_ratio)
        val_size_per_class = int(dataset.num_samples_per_class * val_ratio)
        
        if verbose:
            print("\nCreating stratified splits:")
            print(f"  Train: {train_size_per_class} samples per class ({train_size_per_class * dataset.n_molecules} total)")
            print(f"  Val:   {val_size_per_class} samples per class ({val_size_per_class * dataset.n_molecules} total)")
        
        np.random.seed(seed)
        train_indices = []
        val_indices = []
        
        for mol_idx in range(dataset.n_molecules):
            # All indices for this class
            class_start = mol_idx * dataset.num_samples_per_class
            class_indices = np.arange(class_start, class_start + dataset.num_samples_per_class)
            
            # Shuffle and split
            np.random.shuffle(class_indices)
            train_indices.extend(class_indices[:train_size_per_class])
            val_indices.extend(class_indices[train_size_per_class:train_size_per_class + val_size_per_class])
        
        # Shuffle the combined indices
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        
        # Generate training data
        if verbose:
            print(f"\nGenerating {len(train_indices)} training samples...")
        X_train = []
        y_train = []
        for i in tqdm(train_indices, disable=not verbose):
            spec, label = dataset[i]
            X_train.append(spec.numpy())
            y_train.append(label)
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Generate validation data
        if verbose:
            print(f"Generating {len(val_indices)} validation samples...")
        X_val = []
        y_val = []
        for i in tqdm(val_indices, disable=not verbose):
            spec, label = dataset[i]
            X_val.append(spec.numpy())
            y_val.append(label)
        X_val = np.array(X_val)
        y_val = np.array(y_val)
        
        # Timing info
        if verbose:
            start_time = time.time()

        # Train model
        if verbose:
            print("\nTraining model...")
        self.model.fit(X_train, y_train)

        if verbose:
            end_time = time.time()
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        
        if verbose:
            print("\n" + "=" * 80)
            print("TRAINING COMPLETE")
            print("=" * 80)     
            print(f"Training time: {end_time - start_time:.2f} seconds")
            print(f"Training Accuracy: {train_acc:.4f}")
            print(f"Validation Accuracy: {val_acc:.4f}")
        
        return {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'train_predictions': train_pred,
            'val_predictions': val_pred
        }
    
    def train_pytorch(self, dataset, train_ratio=0.7, val_ratio=0.15, 
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
        from hsi_labeled_dataset import create_dataloaders
        
        if verbose:
            print("=" * 80)
            print("TRAINING PYTORCH MODEL")
            print("=" * 80)
        
        # Create dataloaders
        train_loader, val_loader, _ = create_dataloaders(
            dataset, batch_size=batch_size, 
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
    
    def train(self, dataset, **kwargs):
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
            return self.train_sklearn(dataset, **kwargs)
        else:
            return self.train_pytorch(dataset, **kwargs)
    
    def evaluate(self, dataset, batch_size=32, device='cpu'):
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
            X_test = []
            y_test = []
            for i in range(len(dataset)):
                spec, label = dataset[i]
                X_test.append(spec.numpy())
                y_test.append(label)
            X_test = np.array(X_test)
            y_test = np.array(y_test)
            
            y_pred = self.model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            return {
                'accuracy': acc,
                'predictions': y_pred,
                'true_labels': y_test,
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred)
            }
        else:
            # PyTorch evaluation
            if not isinstance(dataset, DataLoader):
                from hsi_labeled_dataset import create_dataloaders
                _, _, test_loader = create_dataloaders(dataset, batch_size=batch_size)
            else:
                test_loader = dataset
            
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
            
            return {
                'accuracy': acc,
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
            print(f"Sklearn model saved to: {filepath}")
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
