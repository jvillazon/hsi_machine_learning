# Cluster images on encoded latent space to visualize how the model is grouping spectra
import os

import sys
from pathlib import Path

# Add parent directory to path so we can import from core/
sys.path.insert(0, str(Path(__file__).parent.parent))



import numpy as np
import torch
from torch import device, nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE

from core.hsi_unlabeled_dataset import HSI_Unlabeled_Dataset
from core.hsi_labeled_dataset import HSI_Denoising_Dataset, create_denoising_dataloaders
from denoising.denoise_model import DenoisingAutoencoder
from denoising.denoise_model_residual import ResidualDenoisingAutoencoder

# Freeze weights of encoder portion for clustering visualization
def load_model_(model_path, model_class, device='cpu'):

    """Load the denoising model from file."""
    if not os.path.exists(model_path):
        print("No trained model exists.")
        return None
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
    img_dir = r"D:\ADATA Backup\HuBMAP\HuBMAP Xenium\Xenium HSI\data"
    num_samp = 61
    wn_1 = 2700
    wn_2 = 3100
    ch_start = int((2800 - wn_1) / (wn_2 - wn_1) * num_samp)

    # Model Configuration
    model_path = r"denoising/denoising_models/aMSE + Prom_best.pth"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model_(model_path, ResidualDenoisingAutoencoder, device=device)
    # Create dataset and dataloader
        # Load dataset
    dataset = HSI_Unlabeled_Dataset(
        img_dir=img_dir,
        ch_start=ch_start,
        transform=None,
        image_normalization=True,
        min_max_normalization=False,
        num_samples=num_samp,
        wavenumber_start=wn_1,
        wavenumber_end=wn_2,
    )

    # Efficient per-image latent encoding and clustering
    batch_size = 512
    latent_vectors = []
    pixel_img_indices = []
    pixel_flat_indices = []
    img_shapes = []
    print("Encoding all images in batches...")
    with torch.no_grad():
        for img_idx, img_path in enumerate(tqdm(dataset.img_list, desc="Processing images")):
            stats = dataset.image_stats[img_path]
            height = stats['height']
            width = stats['width']
            img_shapes.append((height, width))
            image_spectra = dataset.load_and_process_image(img_path)
            image_tensor = torch.from_numpy(image_spectra).unsqueeze(1).to(device)  # (n_pixels, 1, n_wavenumbers)
            image_dataset = TensorDataset(image_tensor)
            image_dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False)
            running_flat_idx = 0
            for batch in image_dataloader:
                batch_tensor = batch[0]
                x = batch_tensor
                for enc in model.encs:
                    x, _ = enc(x)
                latent_vectors.append(x.cpu())
                # Track pixel indices for cluster reconstruction
                n = x.shape[0]
                pixel_img_indices.extend([img_idx]*n)
                pixel_flat_indices.extend(range(running_flat_idx, running_flat_idx+n))
                running_flat_idx += n
    latent_vectors = torch.cat(latent_vectors, dim=0)
    print("Latent representations shape:", latent_vectors.shape)

    # Cluster and visualize
    latent_np = latent_vectors.cpu().numpy()
    clustering = DBSCAN(eps=0.5, min_samples=5).fit(latent_np)
    cluster_labels = clustering.labels_
    print("DBSCAN cluster labels:", cluster_labels)

    tsne = TSNE(n_components=2, random_state=42)
    latent_2d = tsne.fit_transform(latent_np)
    plt.figure(figsize=(8, 6))
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=cluster_labels, cmap='tab10', s=2)
    plt.title('t-SNE of Latent Space (colored by DBSCAN cluster)')
    plt.colorbar(label='Cluster Label')
    plt.tight_layout()
    plt.show()

    # Reconstruct cluster-labeled images
    cluster_images = [np.full(shape, -1, dtype=int) for shape in img_shapes]
    for i in range(len(cluster_labels)):
        img_idx = pixel_img_indices[i]
        flat_idx = pixel_flat_indices[i]
        height, width = img_shapes[img_idx]
        row, col = np.unravel_index(flat_idx, (height, width))
        cluster_images[img_idx][row, col] = cluster_labels[i]

    # Plot cluster-labeled images
    for img_idx, cluster_img in enumerate(cluster_images):
        plt.figure(figsize=(6, 5))
        plt.imshow(cluster_img, cmap='tab10', interpolation='nearest')
        plt.title(f'Image {img_idx+1} Cluster Map')
        plt.colorbar(label='Cluster Label')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()