"""
Loss functions for HSI denoising autoencoder.

This module provides the Spectral_Preserving_Loss class which combines:
- Pearson correlation (shape similarity)
- Gradient loss (peak positions)
- Curvature loss (peak widths)

Optimized for hyperspectral imaging with noise and background variability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as f


def corr_loss(y_pred, y_true):
    """Pearson correlation loss - baseline and scale invariant.
    
    Computes 1 - Pearson correlation coefficient between predicted and target spectra.
    This loss is invariant to linear transformations (scale + offset), making it ideal
    for spectra with varying baselines and amplitudes.
    
    Args:
        y_pred: predicted spectra, shape [batch_size, num_wavenumbers]
        y_true: target spectra, shape [batch_size, num_wavenumbers]
    
    Returns:
        scalar loss value (0 = perfect correlation, 2 = opposite)
    """
    # Ensure float tensors
    y_pred = y_pred.float()
    y_true = y_true.float()

    # Subtract mean along the sequence dimension
    y_pred_mean = y_pred - y_pred.mean(dim=1, keepdim=True)
    y_true_mean = y_true - y_true.mean(dim=1, keepdim=True)

    # Compute numerator and denominator
    numerator = torch.sum(y_pred_mean * y_true_mean, dim=1)
    denominator = torch.sqrt(torch.sum(y_pred_mean**2, dim=1) * torch.sum(y_true_mean**2, dim=1) + 1e-8)

    # Pearson correlation per sample
    corr = numerator / denominator

    # Loss = 1 - correlation, average over batch
    loss = torch.mean(1 - corr)
    return loss


def gradient_loss(x, y):
    """First derivative loss - preserves peak positions and slopes.
    
    Computes MSE of first derivatives (finite differences).
    This helps preserve spectral shape transitions, peak positions, and slopes.
    Derivative operations naturally cancel constant baseline offsets.
    
    Args:
        x: predicted spectra (B, L) or (B, 1, L)
        y: target spectra (B, L) or (B, 1, L)
    
    Returns:
        scalar loss value
    """
    # Flatten to (B, L) if needed
    if x.dim() == 3:
        x = x.squeeze(1)
    if y.dim() == 3:
        y = y.squeeze(1)
    
    # Compute first derivative (gradient) using finite differences
    x_grad = x[:, 1:] - x[:, :-1]
    y_grad = y[:, 1:] - y[:, :-1]
    
    return f.mse_loss(x_grad, y_grad)


def second_derivative_loss(x, y):
    """Second derivative loss - preserves peak width and curvature.
    
    Computes MSE of second derivatives (curvature).
    This helps preserve peak sharpness/width and local curvature features.
    Second derivative emphasizes fine spectral details.
    
    Args:
        x: predicted spectra (B, L) or (B, 1, L)
        y: target spectra (B, L) or (B, 1, L)
    
    Returns:
        scalar loss value
    """
    # Flatten to (B, L) if needed
    if x.dim() == 3:
        x = x.squeeze(1)
    if y.dim() == 3:
        y = y.squeeze(1)
    
    # Compute second derivative using finite differences
    x_grad2 = x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]
    y_grad2 = y[:, 2:] - 2 * y[:, 1:-1] + y[:, :-2]
    
    return f.mse_loss(x_grad2, y_grad2)


def magnitude_weighted_mse(x, y):
    """MSE with magnitude-based weighting.
    
    Normalizes both predicted and target by their standard deviations before
    computing MSE. Additionally, weights errors by the magnitude of the target signal,
    penalizing errors in peaks/high-value regions more than baseline/low-value regions.
    
    This makes the loss focus on matching the signal pattern and emphasizes
    accurate reconstruction of important spectral features (peaks) over noise-level baseline.
    
    Args:
        x: predicted spectra (B, L) or (B, 1, L)
        y: target spectra (B, L) or (B, 1, L)
    
    Returns:
        scalar loss value
    """
    # Flatten to (B, L) if needed
    if x.dim() == 3:
        x = x.squeeze(1)
    if y.dim() == 3:
        y = y.squeeze(1)
    
    # Normalize each spectrum by its own standard deviation
    x_std = x.std(dim=1, keepdim=True) + 1e-8
    y_std = y.std(dim=1, keepdim=True) + 1e-8
    
    x_normalized = x / x_std
    y_normalized = y / y_std
    
    # Compute per-element squared error
    squared_error = (x_normalized - y_normalized) ** 2
    
    # Create magnitude-based weights from target signal
    # Normalize target to [0, 1] per spectrum for consistent weighting
    y_min = y_normalized.min(dim=1, keepdim=True)[0]
    y_max = y_normalized.max(dim=1, keepdim=True)[0]
    y_norm_01 = (y_normalized - y_min) / (y_max - y_min + 1e-8)
    
    # Weight: emphasize high-magnitude regions (peaks)
    # Use squared magnitude to strongly emphasize peaks: w = (1 + α * mag^2)
    # α controls the strength of weighting (higher α = more emphasis on peaks)
    alpha = 2.0  # Weighting strength parameter
    weights = 1.0 + alpha * (y_norm_01 ** 2)
    
    # Apply weights to squared error
    weighted_squared_error = squared_error * weights
    
    # Return mean of weighted errors
    return weighted_squared_error.mean()

def amplitude_weighted_mse(x, y, p=2.0):
    """Amplitude-weighted MSE - emphasizes residuals at high-magnitude peaks.
    
    Weights each point's squared error by the target amplitude raised to power p.
    Higher p = stronger focus on peak regions. Unlike scale_invariant_mse, this
    operates on the raw signal so absolute magnitude drives weighting directly.
    
    Loss = mean( |y|^p * (x - y)^2 ) / mean( |y|^p )
    
    Args:
        x: predicted spectra (B, L) or (B, 1, L)
        y: target spectra (B, L) or (B, 1, L)
        p: amplitude weighting exponent (default 2.0; higher = more peak focus)
    
    Returns:
        scalar loss value
    """
    # Flatten to (B, L) if needed
    if x.dim() == 3:
        x = x.squeeze(1)
    if y.dim() == 3:
        y = y.squeeze(1)

    weights = y.abs() ** p + 1e-8
    weighted_sq_err = weights * (x - y) ** 2
    return (weighted_sq_err / weights).mean()


def prominence_weighted_mse(x, y, alpha=2.0, baseline_window=5):
    """Prominence-weighted MSE - emphasizes residuals at prominent peaks.
    
    Weights squared errors by each point's prominence: how much it rises above
    the local baseline (rolling minimum). Points at peak tips receive high weight;
    flat baseline regions receive low weight. Fully scale-invariant per-spectrum
    since prominence is normalized to [0, 1] within each spectrum.
    
    Prominence_i = max(0, y_i - local_min_i)
    Weight_i     = 1 + alpha * (prominence_i / max_prominence)
    
    Args:
        x: predicted spectra (B, L) or (B, 1, L)
        y: target spectra (B, L) or (B, 1, L)
        alpha: weight scaling factor (default 2.0; higher = more peak focus)
        baseline_window: rolling window size for local baseline estimation (default 5)
    
    Returns:
        scalar loss value
    """
    # Flatten to (B, L) if needed
    if x.dim() == 3:
        x = x.squeeze(1)
    if y.dim() == 3:
        y = y.squeeze(1)

    # Estimate local baseline as rolling minimum via max_pool1d on negated signal
    padding = baseline_window // 2
    local_min = -f.max_pool1d(
        -y.unsqueeze(1), kernel_size=baseline_window, stride=1, padding=padding
    ).squeeze(1)  # shape (B, L)

    # Prominence: how much each point rises above its local baseline
    prominence = (y - local_min).clamp(min=0)  # shape (B, L)

    # Normalize prominence to [0, 1] per spectrum
    prom_max = prominence.max(dim=1, keepdim=True)[0] + 1e-8
    prom_norm = prominence / prom_max

    # Weights: 1 at baseline, (1 + alpha) at the most prominent peak
    weights = 1.0 + alpha * prom_norm

    weighted_sq_err = weights * (x - y) ** 2
    return (weighted_sq_err / weights).mean()


def mean_squared_log_error(x, y):
    """Mean Squared Logarithmic Error (MSLE) - emphasizes relative differences.
    
    Computes MSE of log-transformed spectra. This loss emphasizes relative
    differences and is less sensitive to large absolute errors in high-value regions,
    while still penalizing errors in low-value regions. It can help preserve overall
    spectral shape and relative peak intensities, especially when spectra have
    wide dynamic range.
    
    Args:
        x: predicted spectra (B, L) or (B, 1, L)
        y: target spectra (B, L) or (B, 1, L)
    
    Returns:
        scalar loss value
    """
    # Flatten to (B, L) if needed
    if x.dim() == 3:
        x = x.squeeze(1)
    if y.dim() == 3:
        y = y.squeeze(1)
    

    # Add small constant to avoid log(0)
    epsilon = 1e-8
    x = x - x.min() + epsilon
    y = y - y.min() + epsilon 
    x_log = torch.log(x)
    y_log = torch.log(y)
    
    return f.mse_loss(x_log, y_log)


class Spectral_Preserving_Loss(nn.Module):
    """Comprehensive spectral loss that preserves shape, peaks, width, and position.
    
    This loss combines multiple components to preserve spectral features:
    
    1. Shape similarity (Pearson correlation): Overall spectral shape
    2. Gradient loss (1st derivative): Peak positions and slopes  
    3. Curvature loss (2nd derivative): Peak width and sharpness
    4. Magnitude-weighted MSE: Pixel-level reconstruction accuracy
    5. Prominence-weighted MSE: Emphasizes errors at prominent peaks
    
    Loss = w_shape * pearson_loss + 
           w_grad * gradient_loss + 
           w_curv * curvature_loss +
           w_mse * magnitude_weighted_mse +
           w_prom * prominence_weighted_mse
    
    Based on comprehensive testing:
    - Point-wise MSE removed (poor noise/background robustness)
    - Pearson correlation preferred over SAM (baseline invariant)
    - Magnitude-weighted MSE added for reconstruction accuracy without scale sensitivity
    - Prominence-weighted MSE added to emphasize errors at prominent peaks  
    - Default weights optimized for discrimination + reconstruction
    
    Args:
        w_shape (float): Weight for Pearson correlation, default 0.4
        w_grad (float): Weight for gradient (1st derivative), default 0.25
        w_curv (float): Weight for curvature (2nd derivative), default 0.15
        w_mse (float): Weight for Magnitude-weighted MSE, default 0.2
        w_prom (float): Weight for Prominence-weighted MSE, default 0.05
    """
    
    def __init__(self, 
                 w_shape: float = 0.4,
                 w_grad: float = 0.25,
                 w_curv: float = 0.15,
                 w_mse: float = 0.2,
                 w_prom: float = 0.05):
        super().__init__()
        
        # Store weights
        self.w_shape = float(w_shape)
        self.w_grad = float(w_grad)
        self.w_curv = float(w_curv)
        self.w_mse = float(w_mse)
        self.w_prom = float(w_prom)
        
        # Normalize weights to sum to 1
        total = self.w_shape + self.w_grad + self.w_curv + self.w_mse + self.w_prom
        self.w_shape /= total
        self.w_grad /= total
        self.w_curv /= total
        self.w_mse /= total
        self.w_prom /= total
        
        print("Spectral_Preserving_Loss initialized:")
        print(f"  w_shape={self.w_shape:.3f}, w_grad={self.w_grad:.3f}, "
              f"w_curv={self.w_curv:.3f}, w_mse={self.w_mse:.3f}, w_prom={self.w_prom:.3f}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute comprehensive spectral loss.
        
        Args:
            pred: predicted spectra (B, L) or (B, 1, L)
            target: target spectra (B, L) or (B, 1, L)
        
        Returns:
            weighted combination of losses
        """
        # Ensure float and flatten if needed
        pred = pred.float()
        target = target.float()
        
        if pred.dim() == 3 and pred.shape[1] == 1:
            pred_flat = pred.squeeze(1)
        else:
            pred_flat = pred.view(pred.shape[0], -1)
        
        if target.dim() == 3 and target.shape[1] == 1:
            target_flat = target.squeeze(1)
        else:
            target_flat = target.view(target.shape[0], -1)
        
        # Component 1: Shape similarity (Pearson correlation)
        shape_loss = corr_loss(pred_flat, target_flat)
        
        # Component 2: Gradient loss (preserves peak positions and slopes)
        grad_loss = gradient_loss(pred_flat, target_flat)
        
        # Component 3: Curvature loss (preserves peak width and sharpness)
        curv_loss = second_derivative_loss(pred_flat, target_flat)
        
        # Component 4: Magnitude-weighted MSE (pixel-level reconstruction)
        mse_loss = amplitude_weighted_mse(pred_flat, target_flat)
        # mse_loss = mean_squared_log_error(pred_flat, target_flat)

        # Component 5: Prominence-weighted MSE (emphasizes errors at prominent peaks)
        prom_loss = prominence_weighted_mse(pred_flat, target_flat)
        
        # Combine with weights
        total_loss = (self.w_shape * shape_loss + 
                     self.w_grad * grad_loss + 
                     self.w_curv * curv_loss +
                     self.w_mse * mse_loss +
                     self.w_prom * prom_loss)
        
        return total_loss
    
    def get_loss_components(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """
        Get individual loss components for monitoring.
        
        Returns:
            dict with keys: 'shape', 'grad', 'curv', 'mse', 'prom', 'total'
        """
        pred = pred.float()
        target = target.float()
        
        if pred.dim() == 3 and pred.shape[1] == 1:
            pred_flat = pred.squeeze(1)
        else:
            pred_flat = pred.view(pred.shape[0], -1)
        
        if target.dim() == 3 and target.shape[1] == 1:
            target_flat = target.squeeze(1)
        else:
            target_flat = target.view(target.shape[0], -1)
        
        # Compute all components
        shape_loss = corr_loss(pred_flat, target_flat)
        grad_loss = gradient_loss(pred_flat, target_flat)
        curv_loss = second_derivative_loss(pred_flat, target_flat)
        mse_loss = magnitude_weighted_mse(pred_flat, target_flat)
        prom_loss = prominence_weighted_mse(pred_flat, target_flat)
        total_loss = (self.w_shape * shape_loss + 
                     self.w_grad * grad_loss + 
                     self.w_curv * curv_loss +
                     self.w_mse * mse_loss +
                     self.w_prom * prom_loss)
        
        return {
            'shape': shape_loss.item(),
            'grad': grad_loss.item(),
            'curv': curv_loss.item(),
            'mse': mse_loss.item(),
            'prom': prom_loss.item(),
            'total': total_loss.item()
        }


class Metrics:
    """Utility class for computing classification and spectral metrics."""
    
    @staticmethod
    def compute_spectral_similarity(pred_spectra, true_spectra):
        """Compute cosine similarity between predicted and true spectra.
        
        Args:
            pred_spectra: predicted spectra tensor
            true_spectra: target spectra tensor
        
        Returns:
            mean cosine similarity across batch
        """
        return torch.nn.functional.cosine_similarity(pred_spectra, true_spectra, dim=1).mean()
