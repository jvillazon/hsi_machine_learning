# Cluster images on encoded latent space to visualize how the model is grouping spectra
import os

import sys
from pathlib import Path

# Add parent directory to path so we can import from core/
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch import device, nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE

from core.hsi_labeled_dataset import HSI_Denoising_Dataset, create_denoising_dataloaders
from denoising.denoise_model import DenoisingAutoencoder
from denoising.denoise_model_residual import ResidualDenoisingAutoencoder

# Freeze weights of encoder portion for clustering visualization
def load_model_(model_path, model_class, device='cpu'):
    """Load the denoising model from file."""
    if model is None:
        model = model_class(
            in_channels=1,
            base_channels=16,
            latent_dim=128,
            kernels=[3,5,7]
        )
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

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

    model.to(device)
    model.eval()

    return model


# Example usage of encoder into latent space for clustering
def main():
    # Dataset Configuration
    img_dir = r"D:\ADATA Backup\HuBMAP\HuBMAP CODEX\data"
    num_samp = 61
    wn_1 = 2700
    wn_2 = 3100
    ch_start = int((2800 - wn_1) / (wn_2 - wn_1) * num_samp)

    # Model Configuration
    model_path = r"denoising/denoising_models/aMSE + Prom_best.pth"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model_(model_path, ResidualDenoisingAutoencoder, device=device)
    # Create dataset and dataloader
    dataset = HSI_Denoising_Dataset(
        img_dir=img_dir,
        num_samp=num_samp,
        wn_1=wn_1,
        wn_2=wn_2,
        ch_start=ch_start
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    # Collect latent representations
    latent_vectors = []
    with torch.no_grad():
        for noisy, clean in dataloader:
            noisy = noisy.to(device)
            # Pass through encoder to get latent representation
            x = noisy.unsqueeze(1)  # Ensure shape (B, 1, L)
            for enc in model.encs:
                x, _ = enc(x)
            latent_vectors.append(x.cpu())
    latent_vectors = torch.cat(latent_vectors, dim=0)  # Shape (N, latent_dim)
    print("Latent representations shape:", latent_vectors.shape)

    # Use DBSCAN on latent_vectors.cpu().numpy() to cluster and visualize
    latent_np = latent_vectors.cpu().numpy()
    clustering = DBSCAN(eps=0.5, min_samples=5).fit(latent_np)
    print("DBSCAN cluster labels:", clustering.labels_)

    # Optionally, use t-SNE for 2D visualization of latent space
    tsne = TSNE(n_components=2, random_state=42)
    latent_2d = tsne.fit_transform(latent_np)
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=clustering.labels_, cmap='tab10')
    plt.colorbar()
