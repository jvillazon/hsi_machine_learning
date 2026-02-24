"""
Train and compare denoising autoencoders with different loss parameters.

Tests different weight combinations for Spectral_Preserving_Loss:
- w_shape: Pearson correlation (shape similarity)
- w_grad: Gradient loss (peak positions)
- w_curv: Curvature loss (peak widths)
- w_mse: Scale-invariant MSE (pixel-level reconstruction)

Evaluates which loss combination produces best spectral similarity and accuracy.
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import from core/
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn  # unused import, can be removed
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm

from core.hsi_labeled_dataset import HSI_Denoising_Dataset, create_denoising_dataloaders
from denoising.denoise_loss import Spectral_Preserving_Loss, Metrics
from denoising.denoise_model import DenoisingAutoencoder
from denoising.denoise_model_residual import ResidualDenoisingAutoencoder


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_shape = 0
    total_mse = 0

    for batch_data in tqdm(train_loader):
        # Handle 3-tuple return (noisy, clean, class_idx)
        noisy, clean = batch_data[0], batch_data[1]
        
        noisy = noisy.to(device)
        clean = clean.to(device)
        
        # Add channel dimension if needed: (B, L) -> (B, 1, L)
        if noisy.dim() == 2:
            noisy = noisy.unsqueeze(1)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(noisy)
        
        # Compute loss
        loss = criterion(output, clean)
        
        # Get loss components (need to pass pred and target)
        components = criterion.get_loss_components(output, clean)
        total_shape += components['shape']
        total_mse += components['mse']
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    n_batches = len(train_loader)
    return {
        'total_loss': total_loss / n_batches,
        'shape_loss': total_shape / n_batches,
        'mse_loss': total_mse / n_batches
    }


def validate(model, val_loader, criterion, device, show_per_class=False, epoch=None, model_name=None, save_dir=None):
    """Validate model with optional per-class visualization."""
    model.eval()
    total_loss = 0
    total_shape = 0
    total_mse = 0
    all_similarities = []
    
    # Per-class tracking for visualization
    if show_per_class:
        class_samples = {}  # {class_idx: {'noisy': [], 'clean': [], 'output': []}}
    
    with torch.no_grad():
        for batch_data in val_loader:
            # Handle both 2-tuple and 3-tuple returns
            if len(batch_data) == 3:
                noisy, clean, class_indices = batch_data
            else:
                noisy, clean = batch_data
                class_indices = None
            
            noisy = noisy.to(device)
            clean = clean.to(device)
            
            # Add channel dimension if needed: (B, L) -> (B, 1, L)
            if noisy.dim() == 2:
                noisy = noisy.unsqueeze(1)
            
            # Forward pass
            output = model(noisy)
            
            # Compute loss
            loss = criterion(output, clean)
            
            # Get loss components (need to pass pred and target)
            components = criterion.get_loss_components(output, clean)
            total_shape += components['shape']
            total_mse += components['mse']
            
            # Compute spectral similarity (cosine similarity)
            # Remove channel dimension for similarity computation: (B, 1, L) -> (B, L)
            output_flat = output.squeeze(1) if output.dim() == 3 else output
            clean_flat = clean.squeeze(1) if clean.dim() == 3 else clean
            similarity = Metrics.compute_spectral_similarity(output_flat, clean_flat)
            all_similarities.append(similarity.item())
            
            total_loss += loss.item()
            
            # Collect samples for per-class visualization
            if show_per_class and class_indices is not None:
                for i in range(len(class_indices)):
                    cls_idx = class_indices[i].item()
                    
                    if cls_idx not in class_samples:
                        class_samples[cls_idx] = {'noisy': [], 'clean': [], 'output': []}
                    
                    # Only store first sample per class for visualization
                    if len(class_samples[cls_idx]['noisy']) == 0:
                        class_samples[cls_idx]['noisy'].append(noisy[i].squeeze().cpu())
                        class_samples[cls_idx]['clean'].append(clean_flat[i].cpu())
                        class_samples[cls_idx]['output'].append(output_flat[i].cpu())
    
    n_batches = len(val_loader)
    mean_similarity = np.mean(all_similarities) if all_similarities else 0.0
    
    results = {
        'total_loss': total_loss / n_batches,
        'shape_loss': total_shape / n_batches,
        'mse_loss': total_mse / n_batches,
        'spectral_similarity': mean_similarity
    }
    
    # Create per-class visualization
    if show_per_class and class_samples and save_dir is not None:
        visualize_per_class(class_samples, epoch, model_name, save_dir)
    
    return results


def visualize_per_class(class_samples, epoch, model_name, save_dir):
    """Create visualization of clean, noisy, and reconstructed spectra for each class."""
    n_classes = len(class_samples)
    
    # Create grid layout
    n_cols = 5
    n_rows = (n_classes + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    axes = axes.flatten() if n_classes > 1 else [axes]
    
    for idx, (cls_idx, samples) in enumerate(sorted(class_samples.items())):
        ax = axes[idx]
        
        noisy = samples['noisy'][0].numpy()
        clean = samples['clean'][0].numpy()
        output = samples['output'][0].numpy()
        
        # Align output to clean range for visualization
        output_aligned = output.copy()
        clean_min, clean_max = clean.min(), clean.max()
        output_min, output_max = output.min(), output.max()
        output_aligned = (output - output_min) / (output_max - output_min + 1e-8)
        output_aligned = output_aligned * (clean_max - clean_min) + clean_min
        
        # Plot
        ax.plot(noisy, color='red', alpha=0.3, linewidth=1, label='Noisy')
        ax.plot(clean, color='blue', linewidth=1.5, label='Clean')
        ax.plot(output_aligned, color='green', linewidth=1.5, linestyle='--', label='Reconstructed', alpha=0.7)
        
        # Compute metrics
        from denoising.denoise_loss import corr_loss
        similarity = 1.0 - corr_loss(torch.tensor(output).unsqueeze(0), 
                                     torch.tensor(clean).unsqueeze(0)).item()
        mse = np.mean((output_aligned - clean) ** 2)
        
        ax.set_title(f'Class {cls_idx}\nSim: {similarity:.3f}, MSE: {mse:.4f}', fontsize=8)
        ax.set_xlabel('Wavenumber', fontsize=7)
        ax.set_ylabel('Intensity', fontsize=7)
        ax.tick_params(labelsize=6)
        if idx == 0:
            ax.legend(fontsize=6, loc='upper right')
        ax.grid(True, alpha=0.2)
    
    # Hide unused subplots
    for idx in range(n_classes, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'{model_name} - Per-Class Denoising (Epoch {epoch+1})', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # Save
    save_path = save_dir / f'per_class_epoch_{epoch+1}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Per-class visualization saved to: {save_path}")
    plt.close()


def plot_denoising_progress(model, val_loader, device, epoch, model_name, save_dir, num_samples=3):
    """Plot denoising examples during training."""
    model.eval()
    
    # Get one batch - handle 3-tuple return (noisy, clean, class_idx)
    batch_data = next(iter(val_loader))
    noisy, clean = batch_data[0], batch_data[1]
    
    noisy = noisy.to(device)
    clean = clean.to(device)
    
    # Add channel dimension if needed: (B, L) -> (B, 1, L)
    if noisy.dim() == 2:
        noisy = noisy.unsqueeze(1)
    
    # Get reconstructions
    with torch.no_grad():
        output = model(noisy)
    
    # Move to CPU for plotting
    noisy = noisy.squeeze(1).cpu()  # Remove channel dim for plotting
    clean = clean.cpu()
    output = output.cpu()
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 1, figsize=(8, 3*num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        ax = axes[i]
        
        # Align reconstructed to clean target for visualization
        # Shift and scale output to match clean range
        output_aligned = output[i].clone()
        clean_min, clean_max = clean[i].min(), clean[i].max()
        output_min, output_max = output[i].min(), output[i].max()
        output_aligned = (output[i] - output_min) / (output_max - output_min + 1e-8)
        output_aligned = output_aligned * (clean_max - clean_min) + clean_min
        
        # Plot spectra
        ax.plot(noisy[i].numpy(), color='red', alpha=0.4, linewidth=1.5, label='Noisy Input')
        ax.plot(clean[i].numpy(), color='blue', linewidth=2.5, label='Clean Target', zorder=3)
        ax.plot(output_aligned.numpy(), color='green', linewidth=2, linestyle='--', 
                label='Reconstructed (aligned)', alpha=0.8, zorder=2)
        
        # Compute similarity using Pearson correlation (1 - corr_loss)
        from denoising.denoise_loss import corr_loss
        similarity = 1.0 - corr_loss(output[i:i+1], clean[i:i+1]).item()
        
        # Compute MSE on aligned reconstruction
        mse = torch.nn.functional.mse_loss(output_aligned, clean[i]).item()
        
        ax.set_title(f'Sample {i+1} - Similarity: {similarity:.4f}, MSE: {mse:.6f}', fontsize=10)
        ax.set_xlabel('Wavenumber Index')
        ax.set_ylabel('Normalized Intensity')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name} - Epoch {epoch+1}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # Save
    filename = save_dir / f'{model_name}_epoch_{epoch+1:03d}.png'
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()


def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs, device, model_name, save_dir, resume=True):
    """Train model and track metrics."""
    
    print(f"Training: {model_name}")

    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_shape': [],
        'val_shape': [],
        'train_mse': [],
        'val_mse': [],
        'val_similarity': [],
        'epoch_times': []
    }
    
    best_val_loss = float('inf')
    best_epoch = 0
    start_epoch = 0

    # Resume from checkpoint if available
    checkpoint_path = save_dir / f'{model_name}_best.pth'
    if resume and checkpoint_path.exists():
        print(f"  Resuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        best_val_loss = ckpt.get('val_loss', float('inf'))
        start_epoch = ckpt.get('epoch', 0) + 1
        history = ckpt.get('history', history)
        best_epoch = ckpt.get('epoch', 0)
        print(f"  Resumed at epoch {start_epoch}, best val loss so far: {best_val_loss:.6f}")
    
    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - start_time
        
        # Store metrics
        history['train_loss'].append(train_metrics['total_loss'])
        history['val_loss'].append(val_metrics['total_loss'])
        history['train_shape'].append(train_metrics['shape_loss'])
        history['val_shape'].append(val_metrics['shape_loss'])
        history['train_mse'].append(train_metrics['mse_loss'])
        history['val_mse'].append(val_metrics['mse_loss'])
        history['val_similarity'].append(val_metrics['spectral_similarity'])
        history['epoch_times'].append(epoch_time)
        
        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'history': history
            }, save_dir / f'{model_name}_best.pth')
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_metrics['total_loss']:.6f} "
                  f"(shape: {train_metrics['shape_loss']:.6f}, "
                  f"mse: {train_metrics['mse_loss']:.6f})")
            print(f"  Val Loss:   {val_metrics['total_loss']:.6f} "
                  f"(shape: {val_metrics['shape_loss']:.6f}, "
                  f"mse: {val_metrics['mse_loss']:.6f})")
            print(f"  Val Similarity: {val_metrics['spectral_similarity']:.6f}")
        
        # Show per-class visualization every 25 epochs
        if (epoch + 1) % 25 == 0:
            print(f"\n--- Creating Per-Class Visualization at Epoch {epoch+1} ---")
            _ = validate(model, val_loader, criterion, device, show_per_class=True, 
                        epoch=epoch, model_name=model_name, save_dir=save_dir)
        
        # Plot denoising examples every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            plot_denoising_progress(model, val_loader, device, epoch, model_name, save_dir, num_samples=3)
            print(f"  ✓ Denoising visualization saved for epoch {epoch+1}")
    
    print(f"\nBest validation loss: {best_val_loss:.6f} at epoch {best_epoch+1}")
    print(f"Average epoch time: {np.mean(history['epoch_times']):.2f}s")
    
    return history


def plot_training_curves(histories, model_names, save_dir):
    """Plot training curves for all models."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Total loss
    ax = axes[0, 0]
    for history, name in zip(histories, model_names):
        epochs = range(1, len(history['train_loss']) + 1)
        ax.plot(epochs, history['train_loss'], '--', alpha=0.6, label=f'{name} (train)')
        ax.plot(epochs, history['val_loss'], '-', linewidth=2, label=f'{name} (val)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Shape loss
    ax = axes[0, 1]
    for history, name in zip(histories, model_names):
        epochs = range(1, len(history['val_shape']) + 1)
        ax.plot(epochs, history['val_shape'], '-', linewidth=2, label=name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Shape Loss (Pearson)')
    ax.set_title('Validation Shape Loss')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # MSE loss
    ax = axes[0, 2]
    for history, name in zip(histories, model_names):
        epochs = range(1, len(history['val_mse']) + 1)
        ax.plot(epochs, history['val_mse'], '-', linewidth=2, label=name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Scale-Invariant MSE')
    ax.set_title('Validation MSE Loss')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Spectral similarity
    ax = axes[1, 0]
    for history, name in zip(histories, model_names):
        epochs = range(1, len(history['val_similarity']) + 1)
        ax.plot(epochs, history['val_similarity'], '-', linewidth=2, label=name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Spectral Similarity (Pearson)')
    ax.set_title('Validation Spectral Similarity')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Epoch times
    ax = axes[1, 1]
    avg_times = [np.mean(h['epoch_times']) for h in histories]
    bars = ax.bar(model_names, avg_times, alpha=0.7)
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Average Epoch Time')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    # Add values on bars
    for bar, time_val in zip(bars, avg_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.2f}s', ha='center', va='bottom', fontsize=9)
    
    # Final metrics comparison
    ax = axes[1, 2]
    final_similarities = [h['val_similarity'][-1] for h in histories]
    bars = ax.bar(model_names, final_similarities, alpha=0.7, color='green')
    ax.set_ylabel('Spectral Similarity')
    ax.set_title('Final Validation Similarity')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([min(final_similarities) - 0.01, 1.0])
    # Add values on bars
    for bar, sim_val in zip(bars, final_similarities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{sim_val:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Training curves saved to: {save_dir / 'training_comparison.png'}")
    plt.show()


def visualize_reconstructions(models, model_names, test_loader, device, save_dir, num_samples=4):
    """Visualize reconstructions from all models."""
    # Get one batch - handle 3-tuple return (noisy, clean, class_idx)
    batch_data = next(iter(test_loader))
    noisy, clean = batch_data[0], batch_data[1]
    
    noisy = noisy.to(device)
    clean = clean.to(device)
    
    # Add channel dimension if needed: (B, L) -> (B, 1, L)
    if noisy.dim() == 2:
        noisy = noisy.unsqueeze(1)
    
    # Get reconstructions from all models
    reconstructions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            output = model(noisy)
        reconstructions.append(output.cpu())
    
    noisy = noisy.squeeze(1).cpu()  # Remove channel dim for plotting
    clean = clean.cpu()
    
    # Plot
    n_models = len(models)
    fig, axes = plt.subplots(num_samples, n_models + 2, figsize=(4*(n_models+2), 4*num_samples))
    
    for i in range(num_samples):
        # Noisy input
        ax = axes[i, 0]
        ax.plot(noisy[i].numpy(), color='red', alpha=0.7, label='Noisy Input')
        ax.plot(clean[i].numpy(), color='blue', linewidth=2, label='Clean Target')
        ax.set_title(f'Sample {i+1}: Input vs Target')
        ax.set_xlabel('Wavenumber Index')
        ax.set_ylabel('Normalized Intensity')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Reconstructions from each model
        for j, (recon, name) in enumerate(zip(reconstructions, model_names)):
            ax = axes[i, j + 1]
            
            # Align reconstructed to clean target for visualization
            recon_aligned = recon[i].clone()
            clean_min, clean_max = clean[i].min(), clean[i].max()
            recon_min, recon_max = recon[i].min(), recon[i].max()
            recon_aligned = (recon[i] - recon_min) / (recon_max - recon_min + 1e-8)
            recon_aligned = recon_aligned * (clean_max - clean_min) + clean_min
            
            ax.plot(noisy[i].numpy(), color='red', alpha=0.2, linewidth=1, label='Noisy')
            ax.plot(clean[i].numpy(), color='blue', linewidth=2.5, label='Target', zorder=3)
            ax.plot(recon_aligned.numpy(), color='green', linewidth=2, linestyle='--', 
                    label='Reconstructed', alpha=0.8, zorder=2)
            
            # Compute similarity using Pearson correlation (1 - corr_loss)
            from denoising.denoise_loss import corr_loss
            similarity = 1.0 - corr_loss(recon[i:i+1], clean[i:i+1]).item()
            mse = torch.nn.functional.mse_loss(recon_aligned, clean[i]).item()
            
            ax.set_title(f'{name}\nSim: {similarity:.4f}, MSE: {mse:.6f}', fontsize=9)
            ax.set_xlabel('Wavenumber Index')
            ax.set_ylabel('Normalized Intensity')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        
        # Residuals comparison
        ax = axes[i, -1]
        for recon, name in zip(reconstructions, model_names):
            residual = (recon[i] - clean[i]).numpy()
            ax.plot(residual, label=name, linewidth=1.5, alpha=0.8)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_title('Residuals (Reconstructed - Target)')
        ax.set_xlabel('Wavenumber Index')
        ax.set_ylabel('Residual')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'reconstruction_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Reconstructions saved to: {save_dir / 'reconstruction_comparison.png'}")
    plt.show()


def evaluate_test_set(models, model_names, test_loader, criterion, device):
    """Evaluate all models on test set."""
    
    print("TEST SET EVALUATION")

    
    results = []
    for model, name in zip(models, model_names):
        metrics = validate(model, test_loader, criterion, device)
        results.append(metrics)
        
        print(f"\n{name}:")
        print(f"  Total Loss:          {metrics['total_loss']:.6f}")
        print(f"  Shape Loss:          {metrics['shape_loss']:.6f}")
        print(f"  MSE Loss:            {metrics['mse_loss']:.6f}")
        print(f"  Spectral Similarity: {metrics['spectral_similarity']:.6f}")
    
    # Find best model
    best_idx = np.argmax([r['spectral_similarity'] for r in results])
    
    print(f"BEST MODEL: {model_names[best_idx]}")
    print(f"  Spectral Similarity: {results[best_idx]['spectral_similarity']:.6f}")

    
    return results


def main():
    # Configuration
    BATCH_SIZE = 128
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SAVE_DIR = Path('denoising/denoising_results_loss_comparison')
    SAVE_DIR.mkdir(exist_ok=True)

    
    # Create dataset with mixture classes
    print("Loading Denoising Dataset...")
    
    # Create ALL unique mixture pairs from all molecules (including "No Match")
    mixture_pairs = None  # None = create all possible pairs
    
    dataset = HSI_Denoising_Dataset(
        molecule_dataset_path='molecule_dataset/lipid_subtype_wn_61_test',
        srs_params_path='params_dataset/srs_params_61',
        num_samples_per_class=2000,
        noise_multiplier=2.0,  # Increase noise for denoising task
        exclude_molecules=[],  # Include all molecules including "No Match"
        create_mixtures=True,
        mixture_pairs=mixture_pairs,  # None = creates all unique pairs
        instance_clean_target=True,   # Fix 3: clean = same SNR+bg instance, no noise
    )

    
    print("Dataset Summary:")

    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_denoising_dataloaders(
        dataset,
        batch_size=BATCH_SIZE,
        train_ratio=0.7,
        val_ratio=0.15,
        seed=42
    )
    
    # Define loss function configurations to test
    # Testing shape + MSE only (no gradient or curvature)
    loss_configs = [
        # Modify these configurations as needed
        {'name': 'aMSE + Prom - Residual', 'w_shape': 0.0, 'w_grad': 0.0, 'w_curv': 0.0, 'w_mse': 0.8, 'w_prom': 0.2},
    ]

    
    print("Training Configuration:")

    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Device: {DEVICE}")
    print(f"Results will be saved to: {SAVE_DIR}")
    
    # Train models with different loss configurations
    models = []
    histories = []
    model_names = []
    criterions = []
    
    for config in loss_configs:
        
        print(f"Testing Loss Configuration: {config['name']}")
    
        
        # Create loss function with specific weights
        criterion = Spectral_Preserving_Loss(
            w_shape=config['w_shape'],
            w_grad=config['w_grad'],
            w_curv=config['w_curv'],
            w_mse=config['w_mse'],
            w_prom=config['w_prom']
        )
        criterions.append(criterion)
        
        # Create model (same architecture for all)
        # ResidualDenoisingAutoencoder: output = noisy - noise_estimate
        # prevents hallucination of peaks not present in input
        model = ResidualDenoisingAutoencoder(
            in_channels=1,
            base_channels=16,
            kernels=[3, 5, 7],
        ).to(DEVICE)
        
        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {n_params:,}")
        
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # Train
        history = train_model(
            model, train_loader, val_loader, criterion, optimizer,
            NUM_EPOCHS, DEVICE, config['name'], SAVE_DIR, resume=False
        )
        
        models.append(model)
        histories.append(history)
        model_names.append(config['name'])
    
    # Plot training curves
    plot_training_curves(histories, model_names, SAVE_DIR)
    
    # Load best models for evaluation
    
    print("Loading Best Models for Evaluation")
    for i, (model, name) in enumerate(zip(models, model_names)):
        checkpoint_path = SAVE_DIR / f"{name}_best.pth"
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded {name} from epoch {checkpoint['epoch']+1}")
    
    # Evaluate on test set with corresponding loss functions
    
    print("Test Set Evaluation with Best Models")

    
    results = []
    for model, name, criterion in zip(models, model_names, criterions):
        metrics = validate(model, test_loader, criterion, DEVICE)
        results.append(metrics)
        
        print(f"\n{name}:")
        print(f"  Total Loss:          {metrics['total_loss']:.6f}")
        print(f"  Shape Loss:          {metrics['shape_loss']:.6f}")
        print(f"  MSE Loss:            {metrics['mse_loss']:.6f}")
        print(f"  Spectral Similarity: {metrics['spectral_similarity']:.6f}")
    
    # Find best model by spectral similarity
    best_idx = np.argmax([r['spectral_similarity'] for r in results])
    
    print(f"BEST MODEL: {model_names[best_idx]}")
    print(f"  Spectral Similarity: {results[best_idx]['spectral_similarity']:.6f}")
    print("  Loss Configuration:")
    print(f"    w_shape: {loss_configs[best_idx]['w_shape']}")
    print(f"    w_grad:  {loss_configs[best_idx]['w_grad']}")
    print(f"    w_curv:  {loss_configs[best_idx]['w_curv']}")
    print(f"    w_mse:   {loss_configs[best_idx]['w_mse']}")
    print(f"    w_prom:  {loss_configs[best_idx]['w_prom']}")

    
    # Visualize reconstructions
    visualize_reconstructions(models, model_names, test_loader, DEVICE, SAVE_DIR, num_samples=4)
    
    # Save summary of results
    summary_path = SAVE_DIR / 'loss_comparison_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("Loss Configuration Comparison Results\n")
        f.write("=" * 70 + "\n\n")
        
        for i, (name, config, result) in enumerate(zip(model_names, loss_configs, results)):
            f.write(f"{i+1}. {name}\n")
            f.write(f"   Weights: w_shape={config['w_shape']}, w_grad={config['w_grad']}, ")
            f.write(f"w_curv={config['w_curv']}, w_mse={config['w_mse']}, w_prom={config['w_prom']}\n")
            f.write(f"   Spectral Similarity: {result['spectral_similarity']:.6f}\n")
            f.write(f"   Total Loss: {result['total_loss']:.6f}\n")
            f.write(f"   Shape Loss: {result['shape_loss']:.6f}\n")
            f.write(f"   MSE Loss: {result['mse_loss']:.6f}\n")
            f.write("\n")
        
        f.write("=" * 70 + "\n")
        f.write(f"BEST: {model_names[best_idx]} (Similarity: {results[best_idx]['spectral_similarity']:.6f})\n")
    
    print(f"\n✓ Summary saved to: {summary_path}")
    print("\nTRAINING COMPLETE!")
    print(f"All results saved to: {SAVE_DIR}")


if __name__ == '__main__':
    main()
