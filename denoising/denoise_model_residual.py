"""
Residual denoising autoencoder for HSI spectra.

Structurally identical to DenoisingAutoencoder in denoise_model.py, but predicts
the NOISE to subtract rather than the clean signal directly:

    output = input - model(input)

This prevents hallucination of spectral peaks: the model can only attenuate
or preserve what is already present in the input — it cannot add energy at
wavenumbers where the input has none.

The clean target and loss functions are unchanged. Only the forward pass differs.
The original DenoisingAutoencoder in denoise_model.py is preserved intact.

Usage
-----
from denoising.denoise_model_residual import ResidualDenoisingAutoencoder

model = ResidualDenoisingAutoencoder(
    in_channels=1,
    base_channels=16,
    kernels=[3, 5, 7],
)
output = model(noisy)   # output = noisy - predicted_noise
loss = criterion(output, clean)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Re-use the shared building blocks from the original module so that
# checkpoints for the two architectures remain independent but the
# underlying conv/attention code stays in one place.
from denoising.denoise_model import EncoderBlock1D, DecoderBlock1D


class ResidualDenoisingAutoencoder(nn.Module):
    """U-Net style autoencoder that predicts residual noise.

    Architecture is identical to DenoisingAutoencoder. The only difference
    is in forward(): the network output is treated as a noise estimate and
    subtracted from the input rather than returned directly.

        output = input - noise_estimate

    Args:
        in_channels (int): Number of input channels (default 1).
        base_channels (int): Feature channels in first encoder layer (default 16).
        latent_dim (int): Unused, kept for API compatibility (default 128).
        kernels (list[int]): Kernel sizes for multi-scale encoder/decoder blocks
            (default [3, 5, 7]).
        residual_scale (float): Multiplier on the noise estimate before subtraction.
            Values < 1.0 soften the correction (default 1.0 = full subtraction).
    """

    def __init__(self,
                 in_channels: int = 1,
                 base_channels: int = 16,
                 latent_dim: int = 128,
                 kernels: list = None,
                 residual_scale: float = 1.0):
        super().__init__()
        if kernels is None:
            kernels = [3, 5, 7]
        self.kernels = kernels
        self.residual_scale = residual_scale

        # --- Encoder ---
        input_channels = in_channels
        self.encs = nn.ModuleList()
        for k in self.kernels[::-1]:
            use_attention = (k == self.kernels[0])
            self.encs.append(EncoderBlock1D(
                input_channels,
                base_channels,
                kernel=k,
                use_attention=use_attention,
            ))
            input_channels = base_channels
            base_channels = base_channels * 2

        base_channels = base_channels // 4

        # --- Decoder ---
        self.decs = nn.ModuleList()
        for k in self.kernels:
            if k == self.kernels[-1]:
                base_channels = 1
            self.decs.append(DecoderBlock1D(
                input_channels,
                base_channels,
                kernel=k,
            ))
            input_channels = base_channels
            base_channels = base_channels // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: noisy input spectra, shape (B, 1, L) or (B, L)

        Returns:
            denoised spectra, shape (B, L)
                = input - residual_scale * noise_estimate
        """
        # Remember original input for subtraction (ensure (B, 1, L))
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x_input = x  # (B, 1, L)

        # --- Encoder pass ---
        x_n = [x_input]
        for n, enc in enumerate(self.encs):
            x_enc, self.attn_space = enc(x_n[n])
            x_n.append(x_enc)

        # --- Decoder pass with skip connections ---
        noise_est = x_n[-1]
        for n, dec in enumerate(self.decs[:-1]):
            noise_est = dec(noise_est, skip=x_n[len(self.encs) - 1 - n])
        noise_est = self.decs[-1](noise_est)

        # Fix length mismatches
        if noise_est.size(-1) > x_input.size(-1):
            noise_est = noise_est[..., :x_input.size(-1)]
        elif noise_est.size(-1) < x_input.size(-1):
            pad = x_input.size(-1) - noise_est.size(-1)
            noise_est = F.pad(noise_est, (0, pad))

        # Residual subtraction: output = input - noise_estimate
        denoised = x_input - self.residual_scale * noise_est

        return denoised.squeeze(1)  # (B, L)
