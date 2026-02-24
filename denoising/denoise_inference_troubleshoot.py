"""
Run inference on test dataset using trained denoising models.

This script:
1. Loads a trained denoising model (.pth file)
2. Runs inference on a test dataset
3. Computes evaluation metrics (correlation, MSE, spectral similarity)
4. Visualizes denoised results
5. Saves quantitative results to CSV
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import from core/
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import argparse

from core.hsi_labeled_dataset import HSI_Denoising_Dataset, create_denoising_dataloaders
from denoising.denoise_model import DenoisingAutoencoder
from denoising.denoise_loss import Spectral_Preserving_Loss, Metrics


def load_model(model_path, device='cpu'):
    """
    Load trained denoising model from .pth file.
    
    Args:
        model_path: Path to .pth checkpoint file
        device: Device to load model to ('cpu' or 'cuda')
    
    Returns:
        model: Loaded DenoisingAutoencoder model
    """
    print(f"Loading model from: {model_path}")
    
    # Initialize model architecture (must match training)
    model = DenoisingAutoencoder(
        in_channels=1,
        base_channels=16,
        latent_dim=128,
        kernels=[3, 5, 7]
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Loaded from epoch: {checkpoint.get('epoch', 'unknown')}")
            best_val_loss = checkpoint.get('best_val_loss', 'unknown')
            if isinstance(best_val_loss, (int, float)):
                print(f"  Best val loss: {best_val_loss:.6f}")
            else:
                print(f"  Best val loss: {best_val_loss}")
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    return model


def evaluate_model(model, test_loader, device, dataset=None):
    """
    Evaluate model on test set and compute comprehensive metrics.
    
    Args:
        model: DenoisingAutoencoder model
        test_loader: DataLoader for test set
        device: Device to run inference on
        dataset: Original dataset to get molecule names
    
    Returns:
        results: Dictionary containing all evaluation metrics
    """
    model.eval()
    
    # Storage for metrics
    all_noisy = []
    all_clean = []
    all_denoised = []
    all_class_indices = []
    
    correlations = []
    mse_scores = []
    cosine_similarities = []
    
    print("\nRunning inference on test set...")
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Denoising"):
            # Handle 3-tuple return (noisy, clean, class_idx)
            noisy, clean, class_idx = batch_data[0], batch_data[1], batch_data[2]
            
            noisy = noisy.to(device)
            clean = clean.to(device)
            
            # Add channel dimension if needed: (B, L) -> (B, 1, L)
            if noisy.dim() == 2:
                noisy = noisy.unsqueeze(1)
            
            # Denoise
            denoised = model(noisy)
            
            # Remove channel dimension for metrics
            denoised_flat = denoised if denoised.dim() == 2 else denoised.squeeze(1)
            clean_flat = clean if clean.dim() == 2 else clean.squeeze(1)
            noisy_flat = noisy.squeeze(1) if noisy.dim() == 3 else noisy
            
            # Compute metrics for this batch
            # 1. Pearson correlation
            for i in range(denoised_flat.shape[0]):
                denoised_np = denoised_flat[i].cpu().numpy()
                clean_np = clean_flat[i].cpu().numpy()
                
                # Pearson correlation
                denoised_centered = denoised_np - denoised_np.mean()
                clean_centered = clean_np - clean_np.mean()
                corr = np.corrcoef(denoised_centered, clean_centered)[0, 1]
                correlations.append(corr)
                
                # MSE
                mse = np.mean((denoised_np - clean_np) ** 2)
                mse_scores.append(mse)
            
            # 2. Cosine similarity (batch-wise)
            cos_sim = Metrics.compute_spectral_similarity(denoised_flat, clean_flat)
            cosine_similarities.append(cos_sim.item())
            
            # Store for visualization
            all_noisy.append(noisy_flat.cpu())
            all_clean.append(clean_flat.cpu())
            all_denoised.append(denoised_flat.cpu())
            all_class_indices.append(class_idx.cpu())
    
    # Concatenate all results
    all_noisy = torch.cat(all_noisy, dim=0)
    all_clean = torch.cat(all_clean, dim=0)
    all_denoised = torch.cat(all_denoised, dim=0)
    all_class_indices = torch.cat(all_class_indices, dim=0)
    
    # Compute summary statistics
    results = {
        'num_samples': len(all_denoised),
        'mean_correlation': np.mean(correlations),
        'std_correlation': np.std(correlations),
        'median_correlation': np.median(correlations),
        'min_correlation': np.min(correlations),
        'max_correlation': np.max(correlations),
        'mean_mse': np.mean(mse_scores),
        'std_mse': np.std(mse_scores),
        'median_mse': np.median(mse_scores),
        'mean_cosine_similarity': np.mean(cosine_similarities),
        'std_cosine_similarity': np.std(cosine_similarities),
        'all_correlations': correlations,
        'all_mse': mse_scores,
        'noisy_spectra': all_noisy,
        'clean_spectra': all_clean,
        'denoised_spectra': all_denoised,
        'class_indices': all_class_indices,
        'dataset': dataset
    }
    
    return results


def print_results(results, model_name):
    """Print evaluation results in a formatted table."""
    print(f"\n{'='*70}")
    print(f"INFERENCE RESULTS: {model_name}")
    print(f"{'='*70}")
    print(f"\nTest Set Size: {results['num_samples']} samples")
    print(f"\n{'Metric':<30} {'Mean':<12} {'Std':<12} {'Median':<12}")
    print("-" * 70)
    print(f"{'Pearson Correlation':<30} {results['mean_correlation']:<12.6f} {results['std_correlation']:<12.6f} {results['median_correlation']:<12.6f}")
    print(f"{'MSE (Reconstruction)':<30} {results['mean_mse']:<12.6f} {results['std_mse']:<12.6f} {results['median_mse']:<12.6f}")
    print(f"{'Cosine Similarity':<30} {results['mean_cosine_similarity']:<12.6f} {results['std_cosine_similarity']:<12.6f} {'-':<12}")
    print("-" * 70)
    print(f"{'Correlation Range':<30} [{results['min_correlation']:.6f}, {results['max_correlation']:.6f}]")
    print(f"{'='*70}\n")


def visualize_results(results, model_name, save_dir, num_examples=10):
    """
    Visualize denoising results.
    
    Args:
        results: Dictionary from evaluate_model()
        model_name: Name of the model for plot titles
        save_dir: Directory to save figures
        num_examples: Number of random examples to plot
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    noisy = results['noisy_spectra']
    clean = results['clean_spectra']
    denoised = results['denoised_spectra']
    
    # 1. Plot random examples
    print(f"\nGenerating visualizations...")
    
    # Select random indices
    n_samples = len(denoised)
    random_indices = np.random.choice(n_samples, min(num_examples, n_samples), replace=False)
    
    fig, axes = plt.subplots(num_examples, 1, figsize=(12, 3*num_examples))
    if num_examples == 1:
        axes = [axes]
    
    for i, idx in enumerate(random_indices):
        ax = axes[i]
        
        noisy_spec = noisy[idx].numpy()
        clean_spec = clean[idx].numpy()
        denoised_spec = denoised[idx].numpy()
        
        # Align denoised to clean target for visualization
        # Shift and scale denoised output to match clean range
        denoised_aligned = denoised_spec.copy()
        clean_min, clean_max = clean_spec.min(), clean_spec.max()
        denoised_min, denoised_max = denoised_spec.min(), denoised_spec.max()
        denoised_aligned = (denoised_spec - denoised_min) / (denoised_max - denoised_min + 1e-8)
        denoised_aligned = denoised_aligned * (clean_max - clean_min) + clean_min
        
        # Compute metrics for this sample (on original denoised, not aligned)
        denoised_centered = denoised_spec - denoised_spec.mean()
        clean_centered = clean_spec - clean_spec.mean()
        corr = np.corrcoef(denoised_centered, clean_centered)[0, 1]
        
        # MSE on aligned reconstruction for better interpretability
        mse_aligned = np.mean((denoised_aligned - clean_spec) ** 2)
        
        # Plot
        ax.plot(noisy_spec, label='Noisy Input', color='red', alpha=0.4, linewidth=1.5)
        ax.plot(clean_spec, label='Clean Target', color='blue', linewidth=2.5, zorder=3)
        ax.plot(denoised_aligned, label='Denoised Output (aligned)', color='green', 
                linewidth=2, linestyle='--', alpha=0.8, zorder=2)
        
        ax.set_title(f'Sample {idx} - Correlation: {corr:.4f}, MSE (aligned): {mse_aligned:.6f}', fontsize=10)
        ax.set_xlabel('Wavenumber Index')
        ax.set_ylabel('Normalized Intensity')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name} - Denoising Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = save_dir / f'{model_name}_examples.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved examples to: {save_path}")
    plt.close()
    
    # 2. Plot correlation distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Correlation histogram
    ax = axes[0]
    ax.hist(results['all_correlations'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(results['mean_correlation'], color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {results["mean_correlation"]:.4f}')
    ax.axvline(results['median_correlation'], color='green', linestyle='--', linewidth=2,
               label=f'Median: {results["median_correlation"]:.4f}')
    ax.set_xlabel('Pearson Correlation', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Correlation Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # MSE histogram
    ax = axes[1]
    ax.hist(results['all_mse'], bins=50, color='coral', alpha=0.7, edgecolor='black')
    ax.axvline(results['mean_mse'], color='red', linestyle='--', linewidth=2,
               label=f'Mean: {results["mean_mse"]:.6f}')
    ax.axvline(results['median_mse'], color='green', linestyle='--', linewidth=2,
               label=f'Median: {results["median_mse"]:.6f}')
    ax.set_xlabel('MSE (Reconstruction Error)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('MSE Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name} - Performance Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = save_dir / f'{model_name}_metrics_distribution.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved metrics distribution to: {save_path}")
    plt.close()
    
    # 3. Visualize "No Match" samples specifically
    if 'class_indices' in results and results['dataset'] is not None:
        visualize_no_match_samples(results, model_name, save_dir)


def visualize_no_match_samples(results, model_name, save_dir):
    """Visualize samples from 'No Match' class specifically."""
    dataset = results['dataset']
    class_indices = results['class_indices']
    
    # Find "No Match" class index
    no_match_idx = None
    for i, name in enumerate(dataset.molecule_names):
        if name == 'No Match':
            no_match_idx = i
            break
    
    if no_match_idx is None:
        print("  ⚠ 'No Match' class not found in dataset")
        return
    
    # Find all samples from "No Match" class
    no_match_mask = class_indices == no_match_idx
    no_match_count = no_match_mask.sum().item()
    
    if no_match_count == 0:
        print(f"  ⚠ No 'No Match' samples found in test set")
        return
    
    print(f"\n  Visualizing {no_match_count} 'No Match' samples...")
    
    # Get No Match samples
    noisy_no_match = results['noisy_spectra'][no_match_mask]
    clean_no_match = results['clean_spectra'][no_match_mask]
    denoised_no_match = results['denoised_spectra'][no_match_mask]
    
    # Plot up to 10 examples
    num_to_plot = min(10, no_match_count)
    fig, axes = plt.subplots(num_to_plot, 1, figsize=(12, 3*num_to_plot))
    if num_to_plot == 1:
        axes = [axes]
    
    for i in range(num_to_plot):
        ax = axes[i]
        
        noisy_spec = noisy_no_match[i].numpy()
        clean_spec = clean_no_match[i].numpy()
        denoised_spec = denoised_no_match[i].numpy()
        
        # Align denoised to clean range for visualization
        clean_min, clean_max = clean_spec.min(), clean_spec.max()
        denoised_min, denoised_max = denoised_spec.min(), denoised_spec.max()
        denoised_aligned = (denoised_spec - denoised_min) / (denoised_max - denoised_min + 1e-8)
        denoised_aligned = denoised_aligned * (clean_max - clean_min) + clean_min
        
        # Plot
        ax.plot(noisy_spec, color='red', alpha=0.3, linewidth=1.5, label='Noisy Input')
        ax.plot(clean_spec, color='blue', linewidth=2, label='Clean Target')
        ax.plot(denoised_aligned, color='green', linewidth=2, linestyle='--', 
                label='Denoised Output', alpha=0.8)
        
        # Compute correlation
        clean_centered = clean_spec - clean_spec.mean()
        denoised_centered = denoised_spec - denoised_spec.mean()
        corr = np.corrcoef(clean_centered, denoised_centered)[0, 1]
        mse = np.mean((denoised_aligned - clean_spec) ** 2)
        
        ax.set_title(f'"No Match" Sample {i+1} | Correlation: {corr:.4f}, MSE: {mse:.6f}', 
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Wavenumber Index', fontsize=10)
        ax.set_ylabel('Normalized Intensity', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name} - "No Match" Class Denoising Results', 
                fontsize=14, fontweight='bold', y=1.001)
    plt.tight_layout()
    
    save_path = save_dir / f'{model_name}_no_match_samples.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved 'No Match' visualization to: {save_path}")
    plt.close()


def save_results_csv(results, model_name, save_path):
    """Save detailed results to CSV file."""
    # Create dataframe with per-sample metrics
    df = pd.DataFrame({
        'sample_idx': range(len(results['all_correlations'])),
        'pearson_correlation': results['all_correlations'],
        'mse': results['all_mse']
    })
    
    save_path = Path(save_path)
    df.to_csv(save_path, index=False)
    print(f"  ✓ Saved detailed results to: {save_path}")
    
    # Also save summary statistics
    summary_path = save_path.parent / f"{model_name}_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Inference Results: {model_name}\n")
        f.write("="*70 + "\n\n")
        f.write(f"Test Set Size: {results['num_samples']} samples\n\n")
        f.write(f"Pearson Correlation:\n")
        f.write(f"  Mean:   {results['mean_correlation']:.6f}\n")
        f.write(f"  Std:    {results['std_correlation']:.6f}\n")
        f.write(f"  Median: {results['median_correlation']:.6f}\n")
        f.write(f"  Range:  [{results['min_correlation']:.6f}, {results['max_correlation']:.6f}]\n\n")
        f.write(f"MSE (Reconstruction Error):\n")
        f.write(f"  Mean:   {results['mean_mse']:.6f}\n")
        f.write(f"  Std:    {results['std_mse']:.6f}\n")
        f.write(f"  Median: {results['median_mse']:.6f}\n\n")
        f.write(f"Cosine Similarity:\n")
        f.write(f"  Mean:   {results['mean_cosine_similarity']:.6f}\n")
        f.write(f"  Std:    {results['std_cosine_similarity']:.6f}\n")
    
    print(f"  ✓ Saved summary to: {summary_path}")


def run_inference(model_path, dataset_config, output_dir='denoising/inference_results', 
                  batch_size=32, num_examples=10, device='cpu'):
    """
    Complete inference pipeline.
    
    Args:
        model_path: Path to trained model .pth file
        dataset_config: Dict with dataset parameters
        output_dir: Directory to save results
        batch_size: Batch size for inference
        num_examples: Number of examples to visualize
        device: Device to run inference on
    """
    model_path = Path(model_path)
    model_name = model_path.stem  # e.g., 'Shape+MSE_best'
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\n{'='*70}")
    print(f"DENOISING MODEL INFERENCE")
    print(f"{'='*70}")
    print(f"Model: {model_name}")
    print(f"Model path: {model_path}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    
    # Load model
    model = load_model(model_path, device=device)
    
    # Create dataset and test loader
    print(f"\nLoading dataset...")
    dataset = HSI_Denoising_Dataset(
        molecule_dataset_path=dataset_config['molecule_dataset_path'],
        srs_params_path=dataset_config['srs_params_path'],
        num_samples_per_class=dataset_config.get('num_samples_per_class', 1000),
        exclude_molecules=dataset_config.get('exclude_molecules', None),
        noise_multiplier=dataset_config.get('noise_multiplier', 1.0)
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_denoising_dataloaders(
        dataset,
        batch_size=batch_size,
        train_ratio=0.7,
        val_ratio=0.15,
        seed=42
    )
    
    # Run inference
    results = evaluate_model(model, test_loader, device=device, dataset=dataset)
    
    # Print results
    print_results(results, model_name)
    
    # Visualize results
    visualize_results(results, model_name, output_dir, num_examples=num_examples)
    
    # Save to CSV
    csv_path = output_dir / f'{model_name}_results.csv'
    save_results_csv(results, model_name, csv_path)
    
    print(f"\n{'='*70}")
    print(f"INFERENCE COMPLETE")
    print(f"{'='*70}\n")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference with trained denoising model')
    parser.add_argument('--model', type=str, 
                        default='denoising/denoising_results_loss_comparison/Correlation+MSE_best.pth',
                        help='Path to trained model .pth file')
    parser.add_argument('--molecule-dataset', type=str,
                        default='molecule_dataset/lipid_subtype_wn-61.npz',
                        help='Path to molecule dataset')
    parser.add_argument('--srs-params', type=str,
                        default='params_dataset/srs_params_61.npz',
                        help='Path to SRS parameters')
    parser.add_argument('--samples-per-class', type=int, default=1000,
                        help='Number of samples per molecule class')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--num-examples', type=int, default=10,
                        help='Number of examples to visualize')
    parser.add_argument('--output-dir', type=str, default='denoising/inference_results',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run inference on (cpu or cuda)')
    parser.add_argument('--exclude-no-match', action='store_true',
                        help='Exclude "No Match" molecule from dataset')
    
    args = parser.parse_args()
    
    # Configure dataset
    dataset_config = {
        'molecule_dataset_path': args.molecule_dataset,
        'srs_params_path': args.srs_params,
        'num_samples_per_class': args.samples_per_class,
        'exclude_molecules': ['No Match'] if args.exclude_no_match else None,
        'noise_multiplier': 1.0
    }
    
    # Run inference
    results = run_inference(
        model_path=args.model,
        dataset_config=dataset_config,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_examples=args.num_examples,
        device=args.device
    )
