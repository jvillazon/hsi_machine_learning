import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import glob
import numpy as np
import pandas as pd
import tifffile
import rampy as rp
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F


# Adapted from HSRS-Contrast by Zhi Li
def normalize_spectrum(spectrum, method="intensity"):
    """
    Normalize spectrum using different methods
    Args:
        spectrum: input spectrum
        method: normalization method ('minmax', 'intensity', 'area')
    Returns:
        normalized spectrum
    """
    if not np.any(spectrum):
        return spectrum

    if method == "intensity":
        max_val = np.max(np.abs(spectrum))
        if max_val != 0:
            return spectrum / max_val
        return spectrum

    elif method == "minmax":
        min_val = np.min(spectrum)
        max_val = np.max(spectrum)
        if max_val - min_val != 0:
            return (spectrum - min_val) / (max_val - min_val)
        return spectrum

    elif method == "area":
        area = np.trapz(spectrum)
        if area != 0:
            return spectrum / area
        return spectrum

    raise ValueError(f"Unknown normalization method: {method}")


def smooth_spectrum(spectrum, lamda=0.2):
    """
    Smooth spectrum using Whittaker smoothing
    Args:
        spectrum: input spectrum
        lamda: smoothing parameter
    Returns:
        smoothed spectrum
    """
    if not np.any(spectrum):
        return spectrum

    smoothed = rp.smooth(np.arange(len(spectrum)), spectrum, method="whittaker", Lambda=lamda)
    return smoothed


def load_water_baseline(water_baseline_path: str, spectra_start: float = 2700, spectra_end: float = 3100):
    """Load and process the water baseline data"""
    try:
        baseline_data = pd.read_csv(water_baseline_path, header=None).values.flatten()
        x_original = np.linspace(spectra_start, spectra_end, len(baseline_data))

        # Reverse the baseline data order (as done in original code)
        baseline_data = baseline_data[::-1]
        baseline_interpolator = interp1d(
            x_original,
            baseline_data,
            kind="cubic",
            fill_value=(baseline_data[0], baseline_data[-1]),
            bounds_error=False,
        )
        print(f"Water baseline loaded from {water_baseline_path}")
        return baseline_interpolator
    except Exception as e:
        print(f"Could not load water baseline: {e}. Processing will continue without water subtraction.")
        return None


def interpolate_spectrum_to_62(spectrum, current_wavenumbers, target_wavenumbers):
    """
    Interpolate spectrum to have exactly 62 spectra size
    Args:
        spectrum: input spectrum
        current_wavenumbers: current wavenumber grid
        target_wavenumbers: target wavenumber grid (length 62)
    Returns:
        interpolated spectrum with 62 spectra size
    """
    if len(spectrum) == 62:
        return spectrum

    # Check if spectrum is all zeros (use any for efficiency)
    if not np.any(spectrum):
        return np.zeros(62, dtype=np.float32)

    # Interpolate to 62 channels
    interpolator = interp1d(
        current_wavenumbers, spectrum, kind="linear", fill_value="extrapolate", bounds_error=False
    )

    interpolated = interpolator(target_wavenumbers)
    return interpolated.astype(np.float32)


def load_hSRS_image(image_path, spectra_start=2700, spectra_end=3100, return_mask_info=False):
    """
    Load hyperspectral SRS image and reshape to (n_pixels, spectra_size)
    Args:
        image_path: path to the .tif hyperspectral image
        spectra_start: starting wavenumber
        spectra_end: ending wavenumber
    Returns:
        reshaped_spectra: array of shape (n_pixels, spectra_size) interpolated to 62 channels
        wavenumbers: wavenumber grid used
        mask: optional boolean mask of valid spatial pixels (height x width)
        coords: optional array of (y, x) positions for valid pixels
    """
    # Load image stack
    image = tifffile.imread(image_path)  # (N, height, width)
    image = np.flip(image, axis=0)  # flip the image because the hypserspectra is from 3100 to 2700
    if len(image.shape) != 3:
        raise ValueError("Image should be a hyperspectral image stack")

    N, height, width = image.shape

    # Create wavenumber grids
    current_wavenumbers = np.linspace(spectra_start, spectra_end, N)
    target_wavenumbers = np.linspace(spectra_start, spectra_end, 62)  # Always 62 channels

    # Create mask for valid pixels (non-zero intensity)
    pixel_intensities = np.sum(image, axis=0)
    mask = pixel_intensities > 0
    coords = np.column_stack(np.where(mask))

    # Reshape to (n_pixels, spectra_size) - only valid pixels
    reshaped_spectra = image[:, mask].T  # Transpose to get (n_pixels, spectra_size)

    # Interpolate all spectra to 62 channels
    if N != 62:
        interpolated_spectra = np.zeros((reshaped_spectra.shape[0], 62), dtype=np.float32)

        for i in range(reshaped_spectra.shape[0]):
            interpolated_spectra[i] = interpolate_spectrum_to_62(
                reshaped_spectra[i], current_wavenumbers, target_wavenumbers
            )

        reshaped_spectra = interpolated_spectra

    if return_mask_info:
        return reshaped_spectra, target_wavenumbers, mask, coords
    return reshaped_spectra, target_wavenumbers, coords


def apply_water_subtraction(spectra, baseline_interpolator, wavenumbers):
    """
    Apply water baseline subtraction to spectra
    Args:
        spectra: input spectra of shape (n_pixels, spectra_size)
        baseline_interpolator: interpolator for water baseline
        wavenumbers: wavenumber length 62
    Returns:
        water subtracted spectra
    """
    if baseline_interpolator is None:
        print("No water baseline available. Skipping water subtraction.")
        return spectra

    water_baseline = baseline_interpolator(wavenumbers)
    water_baseline = np.clip(water_baseline, 0, 1)

    corrected_spectra = np.zeros_like(spectra, dtype=np.float32)

    for i in range(spectra.shape[0]):
        if not np.any(spectra[i]):
            continue

        spectrum = spectra[i].astype(np.float32)
        spectrum = np.maximum(spectrum, 0)

        # Scale baseline using mean of first few points
        scale_factor = np.mean(spectrum[:5])
        scaled_baseline = water_baseline * scale_factor

        # Subtract baseline and left point
        spectrum = spectrum - scaled_baseline
        left_point = np.mean(spectrum[:5])
        spectrum = spectrum - left_point
        spectrum = np.maximum(spectrum, 0)

        corrected_spectra[i] = spectrum

    return corrected_spectra


def preprocess_spectra(
    spectra, baseline_interpolator, wavenumbers, apply_water_sub=True, norm_method="intensity"
):
    """
    Complete preprocessing pipeline for spectra
    Args:
        spectra: input spectra of shape (n_pixels, spectra_size)
        baseline_interpolator: interpolator for water baseline
        wavenumbers: wavenumber grid
        apply_water_sub: whether to apply water subtraction
        norm_method: normalization method
    Returns:
        preprocessed spectra
    """
    # Water subtraction
    if apply_water_sub:
        spectra = apply_water_subtraction(spectra, baseline_interpolator, wavenumbers)

    # Normalization
    spectra = normalize_spectrum(spectra, norm_method)

    return spectra


def sample_spectra(spectra, n_samples, random_state=42):
    """
    Sample spectra randomly
    Args:
        spectra: input spectra of shape (n_pixels, spectra_size)
        n_samples: number of samples to select
        random_state: random seed
    Returns:
        subsampled spectra
    """
    n_pixels = spectra.shape[0]

    if n_samples >= n_pixels:
        print(f"Requested {n_samples} samples but only {n_pixels} available. Returning all.")
        return spectra

    np.random.seed(random_state)
    indices = np.random.choice(n_pixels, n_samples, replace=False)

    return spectra[indices]


class hSRSDataset(Dataset):
    """PyTorch Dataset for hSRS spectra"""

    def __init__(self, spectra, labels=None, transform=None):
        """
        Initialize dataset
        Args:
            spectra: spectral data of shape (n_samples, spectra_size)
            labels: optional labels
            transform: optional transform function
        """
        self.spectra = torch.FloatTensor(spectra)
        self.labels = torch.LongTensor(labels) if labels is not None else None
        self.transform = transform

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        spectrum = self.spectra[idx]

        if self.transform:
            spectrum = self.transform(spectrum)

        if self.labels is not None:
            return spectrum, self.labels[idx]
        else:
            return spectrum


def process_multiple_hSRS_images(
    image_paths,
    water_baseline_path="data/water_HSI_76.csv",
    spectra_start=2700,
    spectra_end=3100,
    n_samples_per_image=50000,
    apply_water_sub=True,
    norm_method="intensity",
    random_state=42,
):
    """
    Process multiple hSRS images and combine them
    Args:
        image_paths: list of paths to .tif files
        water_baseline_path: path to water baseline file
        spectra_start: starting wavenumber
        spectra_end: ending wavenumber
        n_samples_per_image: number of spectra to sample from each image
        apply_water_subtraction: whether to apply water subtraction
        norm_method: normalization method
        random_state: random seed
    Returns:
        combined_spectra: combined and processed spectra (all with 62 channels)
        image_labels: labels indicating which image each spectrum came from
    """
    # Load water baseline once
    baseline_interpolator = load_water_baseline(water_baseline_path, spectra_start, spectra_end)

    all_spectra = []
    image_labels = []

    for i, image_path in enumerate(image_paths):
        try:
            # Load image (automatically interpolated to 62 channels)
            spectra, wavenumbers = load_hSRS_image(image_path, spectra_start, spectra_end)

            # Preprocess
            processed_spectra = preprocess_spectra(
                spectra,
                baseline_interpolator,
                wavenumbers,
                apply_water_sub=apply_water_sub,
                norm_method=norm_method,
            )

            # Remove zero spectra
            valid_mask = np.any(processed_spectra, axis=1)
            processed_spectra = processed_spectra[valid_mask]

            # Subsample
            if n_samples_per_image > 0:
                processed_spectra = sample_spectra(processed_spectra, n_samples_per_image, random_state)

            # Add to collection
            all_spectra.append(processed_spectra)
            image_labels.extend([os.path.basename(image_path)] * len(processed_spectra))

            print(
                f"Processed {image_path}: {len(processed_spectra)} spectra with {processed_spectra.shape[1]} channels"
            )

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_spectra:
        raise ValueError("No valid spectra found in any image")

    # Combine all spectra (all should now have 62 channels)
    combined_spectra = np.vstack(all_spectra)

    print(f"Combined {len(image_paths)} images: {combined_spectra.shape} (all with 62 channels)")

    return combined_spectra, image_labels


def save_processed_data(
    spectra, image_labels, output_dir="data", train_ratio=0.8, random_state=42, save_format="npz"
):
    """
    Save processed spectra with train/validation split
    Args:
        spectra: processed spectra array
        image_labels: labels indicating source image for each spectrum
        output_dir: output directory
        train_ratio: ratio for train split
        random_state: random seed
        save_format: 'npz', 'csv', or 'both'
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create train/validation split
    indices = np.arange(len(spectra))
    train_idx, val_idx = train_test_split(
        indices, train_size=train_ratio, random_state=random_state, stratify=image_labels
    )

    train_spectra = spectra[train_idx]
    val_spectra = spectra[val_idx]
    train_labels = [image_labels[i] for i in train_idx]
    val_labels = [image_labels[i] for i in val_idx]

    if save_format in ["npz", "both"]:
        # Save as NPZ (recommended for large datasets)
        np.savez_compressed(
            os.path.join(output_dir, "train_data.npz"), spectra=train_spectra, labels=train_labels
        )
        np.savez_compressed(os.path.join(output_dir, "val_data.npz"), spectra=val_spectra, labels=val_labels)

    if save_format in ["csv", "both"]:
        train_df = pd.DataFrame(train_spectra)
        train_df["image_label"] = train_labels
        train_df.to_csv(os.path.join(output_dir, "train_data.csv"), index=False)

        val_df = pd.DataFrame(val_spectra)
        val_df["image_label"] = val_labels
        val_df.to_csv(os.path.join(output_dir, "val_data.csv"), index=False)


def load_processed_data(data_dir="data", file_format="npz"):
    """
    Load processed train/validation data
    Args:
        data_dir: directory containing the data files
        file_format: 'npz' or 'csv'
    Returns:
        train_spectra, val_spectra, train_labels, val_labels
    """
    if file_format == "npz":
        train_data = np.load(os.path.join(data_dir, "train_data.npz"), allow_pickle=True)
        val_data = np.load(os.path.join(data_dir, "val_data.npz"), allow_pickle=True)

        train_spectra = train_data["spectra"]
        val_spectra = val_data["spectra"]
        train_labels = train_data["labels"].tolist()
        val_labels = val_data["labels"].tolist()

    elif file_format == "csv":
        train_df = pd.read_csv(os.path.join(data_dir, "train_data.csv"))
        val_df = pd.read_csv(os.path.join(data_dir, "val_data.csv"))

        train_spectra = train_df.drop("image_label", axis=1).values
        val_spectra = val_df.drop("image_label", axis=1).values
        train_labels = train_df["image_label"].tolist()
        val_labels = val_df["image_label"].tolist()

    else:
        raise ValueError(f"Unsupported file format: {file_format}")

    return train_spectra, val_spectra, train_labels, val_labels


def create_dataloaders(
    train_spectra, val_spectra, train_labels=None, val_labels=None, batch_size=1024, num_workers=4
):
    """
    Create PyTorch DataLoaders for training and validation
    Args:
        train_spectra: training spectra
        val_spectra: validation spectra
        train_labels: training labels (optional)
        val_labels: validation labels (optional)
        batch_size: batch size
        num_workers: number of workers for data loading
    Returns:
        train_loader, val_loader
    """
    # Convert string labels to numeric if provided
    if train_labels is not None:
        unique_labels = list(set(train_labels + val_labels))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

        train_numeric_labels = [label_to_idx[label] for label in train_labels]
        val_numeric_labels = [label_to_idx[label] for label in val_labels]
    else:
        train_numeric_labels = None
        val_numeric_labels = None

    # Create datasets
    train_dataset = hSRSDataset(train_spectra, train_numeric_labels)
    val_dataset = hSRSDataset(val_spectra, val_numeric_labels)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader


def main(
    data_dir,
    water_baseline_path: str = "data/water_76.csv",
    output_dir="data",
    spectra_start=2700,
    spectra_end=3100,
    n_samples_per_image=50000,
    train_ratio=0.8,
    save_format="npz",
    apply_water_sub=False,
    norm_method="intensity",
    random_state=42,
):
    """
    Main preprocessing pipeline for multiple hSRS images
    Args:
        data_dir: directory containing .tif files
        water_baseline_path: path to water baseline file
        output_dir: output directory for processed data
        spectra_start: starting wavenumber
        spectra_end: ending wavenumber
        n_samples_per_image: number of spectra to sample per image
        train_ratio: ratio for train split
        save_format: output format ('npz', 'csv', or 'both')
        apply_water_subtraction: whether to apply water subtraction
        norm_method: normalization method
        random_state: random seed
    """
    # Find all .tif files
    image_paths = glob.glob(os.path.join(data_dir, "*.tif"))
    if not image_paths:
        raise ValueError(f"No .tif files found in {data_dir}")

    print(f"Found {len(image_paths)} .tif files")

    # Process all images
    combined_spectra, image_labels = process_multiple_hSRS_images(
        image_paths=image_paths,
        water_baseline_path=water_baseline_path,
        spectra_start=spectra_start,
        spectra_end=spectra_end,
        n_samples_per_image=n_samples_per_image,
        apply_water_sub=apply_water_sub,
        norm_method=norm_method,
        random_state=random_state,
    )

    # Save processed data
    save_processed_data(
        spectra=combined_spectra,
        image_labels=image_labels,
        output_dir=output_dir,
        train_ratio=train_ratio,
        random_state=random_state,
        save_format=save_format,
    )

    print("Preprocessing pipeline completed successfully!")


if __name__ == "__main__":
    main(
        data_dir="hSRS",  # Directory containing .tif files
        water_baseline_path="hSRS/water_76.csv",
        output_dir="data",
        spectra_start=2700,
        spectra_end=3100,
        n_samples_per_image=50000,
        train_ratio=0.8,
        save_format="npz",
        apply_water_sub=False,
        norm_method="intensity",
        random_state=42,
    )





