"""Evaluate denoising model on hSRS data."""
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import tifffile

core_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'core'))
denoising_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'denoising'))
if core_path not in sys.path:
    sys.path.insert(0, core_path)
if denoising_path not in sys.path:
    sys.path.insert(0, denoising_path)

from hsi_unlabeled_dataset import HSI_Unlabeled_Dataset
from denoise_model import DenoisingAutoencoder



def load_model(model_path, model=None, device='cpu'):
    """Load the denoising model from file."""
    if model is None:
        model = DenoisingAutoencoder(
            in_channels=1,
            base_channels=16,
            latent_dim=128,
            kernels=[3,5,7]
        )
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

    model.to(device)
    model.eval()

    return model

def main():
    """Main function to run denoising inference."""
    # Dataset Configuration
    img_dir = r"/Users/jorgevillazon/Documents/files/codex-srs/HuBMAP .tif files for Jorge Part 1/data"
    num_samp = 61
    wn_1 = 2700
    wn_2 = 3100
    ch_start = int((2800 - wn_1) / (wn_2 - wn_1) * num_samp)

    # Model Configuration
    model_path = r"denoising/denoising_models/Correlation+MSE_best.pth"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Output Configuration
    parent_dir = os.path.dirname(img_dir)
    output_dir = os.path.join(parent_dir, "denoised_images")

    # Load dataset
    dataset = HSI_Unlabeled_Dataset(
        img_dir=img_dir,
        ch_start=ch_start,
        transform=None,
        image_normalization=False,
        min_max_normalization=True,
        num_samples=num_samp,
        wavenumber_start=wn_1,
        wavenumber_end=wn_2,
    )

    # Load model
    model = load_model(model_path, device=device)

    # Denoising inference
    os.makedirs(output_dir, exist_ok=True)

    
    with torch.no_grad():
        for img_idx, img_path in enumerate(tqdm(dataset.img_list, desc="Processing images")):

            # Get image name and statistics
            img_name = os.path.basename(img_path)
            stats = dataset.image_stats[img_path]
            height = stats['height']
            width = stats['width']
            image_shape = (height, width)

            # Load and preprocess image
            image_spectra = dataset.load_and_process_image(img_path)

            # Convert to tensor and reshape for model input
            image_tensor = torch.from_numpy(image_spectra).unsqueeze(1).to(device)  # Shape: (n_pixels, 1, n_wavenumbers)
            
            # Batch processing if needed
            batch_size = 1024

            # Denoise in batches
            image_dataset = TensorDataset(image_tensor)
            image_dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False)
            output_tensors = []

            print(f"Denoising image: {img_name}")
            for batch in tqdm(image_dataloader, desc="Denoising batches"):
                batch_tensor = batch[0]
                denoised_batch = model(batch_tensor)
                output_tensors.append(denoised_batch)
            denoised_tensor = torch.cat(output_tensors, dim=0)

            denoised_image = denoised_tensor.squeeze(1).reshape(height, width, -1).permute(2, 0, 1)  # Shape: (channels, height, width)

            # Save denoised image
            denoised_image_np = denoised_image.cpu().numpy()  # Shape: (channels, height, width)
            save_path = os.path.join(output_dir, f"denoised_{img_name}")
            tifffile.imwrite(save_path, denoised_image_np.astype(np.float32))

            print(f"Saved denoised image to {save_path}")

if __name__ == "__main__":
    main()