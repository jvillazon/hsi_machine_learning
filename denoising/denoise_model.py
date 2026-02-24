import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sympy.solvers.diophantine.diophantine import reconstruct
from torch.onnx import register_custom_op_symbolic
#

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling1D(nn.Module):
    """
    Learns to weight features across the sequence dimension.
    This performs attention-weighted average pooling.
    """
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x):
        # x: (B, C, L)
        attn_scores = self.query(x)  # (B, 1, L)
        attn_weights = F.softmax(attn_scores, dim=-1)  # across sequence
        pooled = torch.sum(attn_weights * x, dim=-1, keepdim=True)  # weighted average
        return pooled  # (B, C, 1)

class EncoderBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, use_attention=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(out_ch, out_ch, kernel_size=kernel, padding=kernel//2),
            nn.LeakyReLU(0.1)
        )
        self.attn = AttentionPooling1D(out_ch) if use_attention else None

    def forward(self, x):
        x = self.conv(x)
        pooled = None
        if self.attn:
            pooled = self.attn(x)
            # broadcast pooled info to all positions
            x = x + pooled.expand_as(x)
        return x, pooled

class DecoderBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(out_ch, out_ch, kernel_size=kernel, padding=kernel//2),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x, skip=None):
        x = self.deconv(x)
        if skip is not None:
            # handle odd-length mismatches
            if x.shape[-1] != skip.shape[-1]:
                diff = skip.shape[-1] - x.shape[-1]
                x = F.pad(x, (0, diff))
            x = x + skip  # skip connection
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_attention=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1)
        )
        self.attn = AttentionPooling1D(out_ch) if use_attention else None

    def forward(self, x):
        x = self.conv(x)
        if self.attn:
            pooled = self.attn(x)
            # broadcast pooled info to all positions
            x = x + pooled.expand_as(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x, skip=None):
        x = self.deconv(x)
        if skip is not None:
            # handle odd-length mismatches
            if x.shape[-1] != skip.shape[-1]:
                diff = skip.shape[-1] - x.shape[-1]
                x = F.pad(x, (0, diff))
            x = x + skip  # skip connection
        return x


class DenoisingAutoencoder(nn.Module):
    def __init__(self, in_channels=1, base_channels = 16, latent_dim=128, kernels=[3, 5, 7]):
        super().__init__()
        self.kernels = kernels

        input_channels = in_channels
        self.encs = nn.ModuleList([])
        for k in self.kernels[::-1]:
            if k != self.kernels[0]:
                attn = False
            else:
                attn = True
            self.encs.append(EncoderBlock1D(
                input_channels,
                base_channels,
                kernel=k,
                use_attention=attn
            ))
            input_channels = base_channels
            base_channels = base_channels * 2

        base_channels = base_channels // 4

        self.decs = nn.ModuleList([])
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


    def forward(self, x):
        # Encoder
        # x1 = self.enc1(x)
        # x2 = self.enc2(x1)
        # latent = self.enc3(x2)
        x_n = [x]
        for n, enc in enumerate(self.encs):
            x_enc, self.attn_space = enc(x_n[n])
            x_n.append(x_enc)
        latent = x_n[-1]
        reconstructed = latent

        for n, dec in enumerate(self.decs[:-1]):
            reconstructed = dec(reconstructed, skip=x_n[len(self.encs)-1-n])
        reconstructed = self.decs[-1](reconstructed)
        
        # Handle size mismatches
        if reconstructed.size(-1) > x.size(-1):
            reconstructed = reconstructed[..., :x.size(-1)]
        elif reconstructed.size(-1) < x.size(-1):
            pad = x.size(-1) - reconstructed.size(-1)
            reconstructed = F.pad(reconstructed, (0, pad))
        
        return reconstructed.squeeze(1)





class DAE(nn.Module):
    def __init__(self, input_channels=1, base_channels=8, kernels=[3, 5, 7], sample_steps=3):
        super(DAE, self).__init__()

        self.sample_steps = sample_steps
        self.kernels = kernels
        self.down_layers = []
        in_channels = input_channels
        base_channels = base_channels

        for k in self.kernels:
            self.down_layers.append(
                nn.Conv1d(
                    in_channels,
                    base_channels,
                    kernel_size=k,
                    padding=k//2,
                    padding_mode="reflect")
            ),
            self.down_layers.append(nn.LeakyReLU()),
            print(f"kernel_down:{in_channels, base_channels}")
            in_channels = base_channels
            base_channels *= 2



        for steps in range(self.sample_steps):
            self.down_layers.append(
                nn.Conv1d(
                    in_channels,
                    base_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    padding_mode='reflect'
                )
            ),
            self.down_layers.append(nn.LeakyReLU()),
            print(f"down_sample:{in_channels, base_channels}")
            in_channels = base_channels
            base_channels *= 2

        self.down_layers.append(nn.Dropout(0.7))


        self.encoder = nn.Sequential(*self.down_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_channels, in_channels * (61 // self.sample_steps**2))

        self.up_layers = []
        in_channels = base_channels // 2
        out_channels = in_channels // 2
        for steps in range(self.sample_steps):
            self.up_layers.append(
                nn.ConvTranspose1d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
            )
            self.up_layers.append(nn.LeakyReLU()),
            print(f"Kernel_up:{in_channels, out_channels}")
            in_channels = out_channels
            out_channels = out_channels // 2


        for k in self.kernels[:-1]:
            self.up_layers.append(
                nn.ConvTranspose1d(
                    in_channels,
                    out_channels,
                    kernel_size=k,
                    padding=k//2,
                )
            ),
            self.up_layers.append(nn.LeakyReLU()),
            print(f"up_sample:{in_channels, out_channels}")
            in_channels = out_channels
            out_channels = out_channels // 2

        self.up_layers.append(
            nn.ConvTranspose1d(
                in_channels,
                input_channels,
                kernel_size=self.kernels[-1],
                padding=self.kernels[-1]//2,
            )
        )
        self.up_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*self.up_layers)

    def forward(self, x):
        latent = self.encoder(x)
        # # pool = self.pool(latent)
        latent = latent.squeeze(-1)
        c = latent.size(1)
        pool = self.fc(latent)
        pool = pool.view(pool.size(0), c, 61//(self.sample_steps**2))
        reconstructed = self.decoder(pool)

        if reconstructed.size(-1) > x.size(-1):
            reconstructed = reconstructed[..., :61]
        elif reconstructed.size(-1) < x.size(-1):
            pad = x.size(-1) - reconstructed.size(-1)
            reconstructed = F.pad(reconstructed, (0, pad))
        return reconstructed.squeeze(1)




class RamanDenoise(nn.Module):
    def __init__(self, input_channels=1, base_channels=8, output_padding=1):
        super(RamanDenoise, self).__init__()
    #
      # Encoder: 5 Conv + ReLU layers with Max Pooling
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, base_channels, kernel_size=3, stride=2),  # [B, 1, N] -> [B, b, N]
            nn.ReLU(),
            # nn.MaxPool1d(2),  # Downsample by 2 (half the size) -> [B, 16, N/2]

            nn.Conv1d(base_channels, base_channels * 2, kernel_size=5, stride=2),  # [B, b, N/2] -> [B, b * 2, N/2]
            nn.ReLU(),
            # nn.MaxPool1d(2),  # Downsample by 2 again -> [B, 32, N/4]

            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2),  # [B, b * 2, N/2] -> [B, b * 4, N/8]
            nn.ReLU(),

            nn.Conv1d( base_channels * 4, base_channels * 8, kernel_size=3, padding=1, padding_mode='reflect'),  # [B, b * 4, N/8] -> [B, b * 8, N/8]
            nn.ReLU(),

            nn.Conv1d(base_channels * 8, base_channels * 16, kernel_size=3, padding=1, padding_mode='reflect'),  # [B, b * 8, N/8] -> [B, b * 16, N/8]
            nn.ReLU(),
        )

        # self.pool = nn.AdaptiveAvgPool1d(1)
        # self.fc = nn.Linear(base_channels * 16, base_channels * 4 * 61)

        # Decoder: Using ConvTranspose1d to upsample and reconstruct the input
        self.decoder = nn.Sequential(
            # nn.ConvTranspose1d(base_channels * 32, base_channels * 16, kernel_size=3, padding=1, padding_mode='zeros'),
            # # [B, 256, N/4] -> [B, 128, N/2]
            # nn.ReLU(),

            nn.ConvTranspose1d(base_channels * 16, base_channels * 8, kernel_size=3, padding=1, padding_mode='zeros'),  # [B, 256, N/4] -> [B, 128, N/2]
            nn.ReLU(),

            nn.ConvTranspose1d( base_channels * 8,  base_channels * 4, kernel_size=3, padding=1, padding_mode='zeros'),  # [B, 128, N/2] -> [B, 64, N]
            nn.ReLU(),

            nn.ConvTranspose1d(base_channels * 4, base_channels * 2, kernel_size=3, stride=2, output_padding=output_padding),  # [B, 32, N] -> [B, 16, N]
            nn.ReLU(),

            nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size=3, stride=2, output_padding=output_padding),  # [B, 32, N] -> [B, 16, N]
            nn.ReLU(),

            nn.ConvTranspose1d(base_channels, 1, kernel_size=3, stride=2),  # [B, 16, N] -> [B, 1, N]
            nn.Sigmoid()  # For outputting values in the range [0, 1] (or [min, max] for normalized data)
        )

    def forward(self, x):
        latent = self.encoder(x)
        # pool = self.pool(latent)
        # pool = pool.squeeze(-1)
        # pool = self.fc(pool)
        reconstructed = self.decoder(latent)

        if reconstructed.size(-1) > x.size(-1):
            reconstructed = reconstructed[..., :61]
        elif reconstructed.size(-1) < x.size(-1):
            pad = x.size(-1) - reconstructed.size(-1)
            reconstructed = F.pad(reconstructed, (0, pad))
        return reconstructed.squeeze(1)
