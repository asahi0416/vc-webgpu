"""Causal Content Encoder and Causal Pitch (F0) Conditioner."""

import math
from typing import Tuple

import torch
import torch.nn as nn

from src.models.audio_dsp import HOP_LENGTH, N_FFT, N_MELS, SAMPLE_RATE
from src.models.causal_blocks import CausalConvNeXtBlock, ChannelLayerNorm


class CausalContentEncoder(nn.Module):
    """Extracts speaker-agnostic phonetic content features causally from log-mel."""

    def __init__(
        self,
        n_mels: int = N_MELS,
        dim: int = 256,
        bottleneck_dim: int = 64,
        num_layers: int = 3,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.dim = dim
        self.pad_len = kernel_size - 1

        self.in_proj = nn.Conv1d(n_mels, dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                CausalConvNeXtBlock(
                    dim=dim,
                    kernel_size=kernel_size,
                    mlp_ratio=2,
                    use_cond=False,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_norm = ChannelLayerNorm(dim)
        # Information bottleneck strips residual timbre while preserving linguistic content
        self.bottleneck = nn.Sequential(
            nn.Conv1d(dim, bottleneck_dim, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(bottleneck_dim, dim, kernel_size=1),
        )

    @staticmethod
    def _normalize_frame_mel(log_mel: torch.Tensor) -> torch.Tensor:
        """Per-frame normalization across mel bins to suppress volume/spectral tilt."""
        mean = log_mel.mean(dim=1, keepdim=True)
        var = ((log_mel - mean) ** 2).mean(dim=1, keepdim=True)
        return (log_mel - mean) * torch.rsqrt(var + 1e-5)

    def forward(
        self,
        log_mel: torch.Tensor,
        cache_in: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            log_mel: Input log-mel spectrogram of shape (B, n_mels, T).
            cache_in: Layer caches of shape (num_layers, B, dim, pad_len).

        Returns:
            content_feat: Content representation of shape (B, dim, T).
            cache_out: Updated layer caches of shape (num_layers, B, dim, pad_len).
        """
        h = self.in_proj(self._normalize_frame_mel(log_mel))
        next_caches = []
        for i, block in enumerate(self.blocks):
            h, c_out = block(h, cache_in[i], cond_vec=None)
            next_caches.append(c_out)

        content_feat = self.bottleneck(self.out_norm(h))
        cache_out = torch.stack(next_caches, dim=0)
        return content_feat, cache_out


class CausalPitchConditioner(nn.Module):
    """Causal F0 estimator, semitone shifter, and harmonic sinusoidal embedder."""

    def __init__(
        self,
        n_mels: int = N_MELS,
        sample_rate: int = SAMPLE_RATE,
        n_fft: int = N_FFT,
        hidden_dim: int = 128,
        out_dim: int = 256,
        num_layers: int = 2,
        kernel_size: int = 7,
        n_harmonics: int = 16,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.pad_len = kernel_size - 1
        # Bins covering 0 Hz to ~1000 Hz
        freq_resolution = float(sample_rate) / float(n_fft)
        self.low_freq_bins = min(int(1000.0 / freq_resolution) + 1, n_fft // 2 + 1)

        in_dim = self.low_freq_bins + n_mels
        self.in_proj = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, kernel_size=1),
            ChannelLayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [
                CausalConvNeXtBlock(
                    dim=hidden_dim,
                    kernel_size=kernel_size,
                    mlp_ratio=2,
                    use_cond=False,
                )
                for _ in range(num_layers)
            ]
        )
        self.f0_head = nn.Conv1d(hidden_dim, 2, kernel_size=1)

        # Log-frequency bounds [50 Hz, 1100 Hz]
        self.log_f0_min = math.log(50.0)
        self.log_f0_range = math.log(1100.0) - math.log(50.0)

        # Harmonic embedding frequencies
        harmonics = torch.arange(1, n_harmonics + 1, dtype=torch.float32).view(1, n_harmonics, 1)
        self.register_buffer("harmonics", harmonics)
        # Sin + Cos of harmonics (2 * n_harmonics) + normalized log_f0 + uv flag
        embed_in_dim = n_harmonics * 2 + 2
        self.embed_proj = nn.Sequential(
            nn.Conv1d(embed_in_dim, out_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(out_dim, out_dim, kernel_size=1),
        )

    def forward(
        self,
        mag: torch.Tensor,
        log_mel: torch.Tensor,
        pitch_shift_semitones: torch.Tensor,
        cache_in: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Estimate F0 causally, apply pitch shift, and return harmonic embedding.

        Args:
            mag: Linear magnitude spectrum (B, n_freq_bins, T).
            log_mel: Log-mel spectrogram (B, n_mels, T).
            pitch_shift_semitones: Semitone shift tensor of shape (B, 1).
            cache_in: Layer caches of shape (num_layers, B, hidden_dim, pad_len).

        Returns:
            f0_feat: Harmonic pitch conditioning tensor (B, out_dim, T).
            shifted_f0: Shifted F0 in Hz of shape (B, 1, T).
            uv: Voiced probability in [0, 1] of shape (B, 1, T).
            cache_out: Updated layer caches of shape (num_layers, B, hidden_dim, pad_len).
        """
        low_mag = torch.log(torch.clamp(mag[:, : self.low_freq_bins, :], min=1e-5))
        x = torch.cat([low_mag, log_mel], dim=1)
        h = self.in_proj(x)

        next_caches = []
        for i, block in enumerate(self.blocks):
            h, c_out = block(h, cache_in[i], cond_vec=None)
            next_caches.append(c_out)

        pred = self.f0_head(h)  # (B, 2, T)
        log_f0 = self.log_f0_min + torch.sigmoid(pred[:, 0:1, :]) * self.log_f0_range
        uv = torch.sigmoid(pred[:, 1:2, :])

        # Shift F0 in Hz: f0 * 2^(semitones / 12)
        shift_factor = torch.pow(2.0, pitch_shift_semitones.unsqueeze(-1) / 12.0)
        f0_hz = torch.exp(log_f0)
        shifted_f0 = f0_hz * shift_factor

        # Harmonic sinusoidal embedding masked by voiced probability
        norm_hz = (shifted_f0 / 1000.0) * self.harmonics * (2.0 * math.pi)
        sin_feat = torch.sin(norm_hz) * uv
        cos_feat = torch.cos(norm_hz) * uv
        shifted_log_norm = ((torch.log(torch.clamp(shifted_f0, min=10.0)) - self.log_f0_min) / self.log_f0_range) * uv

        raw_embed = torch.cat([sin_feat, cos_feat, shifted_log_norm, uv], dim=1)
        f0_feat = self.embed_proj(raw_embed)
        cache_out = torch.stack(next_caches, dim=0)
        return f0_feat, shifted_f0, uv, cache_out
