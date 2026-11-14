"""Causal Vocos Vocoder using Causal ConvNeXt blocks and COLA CausalISTFT."""

from typing import Tuple

import torch
import torch.nn as nn

from src.models.audio_dsp import CausalISTFT, HOP_LENGTH, N_FFT, N_MELS
from src.models.causal_blocks import CausalConvNeXtBlock, ChannelLayerNorm


class CausalVocosVocoder(nn.Module):
    """Lightweight causal Vocos-style vocoder mapping converted mel to waveform."""

    def __init__(
        self,
        mel_dim: int = N_MELS,
        dim: int = 256,
        num_layers: int = 4,
        kernel_size: int = 7,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
    ) -> None:
        super().__init__()
        self.mel_dim = mel_dim
        self.dim = dim
        self.num_layers = num_layers
        self.pad_len = kernel_size - 1
        self.n_freq_bins = n_fft // 2 + 1

        self.in_proj = nn.Conv1d(mel_dim, dim, kernel_size=1)
        self.in_norm = ChannelLayerNorm(dim)
        self.spk_proj = nn.Linear(dim, dim)

        self.blocks = nn.ModuleList(
            [
                CausalConvNeXtBlock(
                    dim=dim,
                    kernel_size=kernel_size,
                    mlp_ratio=2,
                    use_cond=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_norm = ChannelLayerNorm(dim)
        self.head = nn.Conv1d(dim, self.n_freq_bins * 2, kernel_size=1)
        self.istft = CausalISTFT(n_fft=n_fft, hop_length=hop_length)

    def forward(
        self,
        mel: torch.Tensor,
        spk_emb: torch.Tensor,
        vocoder_cache_in: torch.Tensor,
        ola_cache_in: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Synthesize waveform causally from mel spectrogram.

        Args:
            mel: Converted log-mel spectrogram of shape (B, mel_dim, T).
            spk_emb: Speaker embedding of shape (B, dim).
            vocoder_cache_in: ConvNeXt caches of shape (num_layers, B, dim, pad_len).
            ola_cache_in: Causal iSTFT Overlap-Add buffer of shape (B, 720).

        Returns:
            audio_out: Output waveform chunk of shape (B, T * hop_length).
            vocoder_cache_out: Updated ConvNeXt caches of shape (num_layers, B, dim, pad_len).
            ola_cache_out: Updated OLA tail buffer of shape (B, 720).
        """
        cond_vec = self.spk_proj(spk_emb)
        h = self.in_norm(self.in_proj(mel))

        next_caches = []
        for i, block in enumerate(self.blocks):
            h, c_out = block(h, vocoder_cache_in[i], cond_vec=cond_vec)
            next_caches.append(c_out)

        spec_params = self.head(self.out_norm(h))
        log_mag, phase = torch.chunk(spec_params, 2, dim=1)
        phase_sin = torch.sin(phase)
        phase_cos = torch.cos(phase)

        audio_out, ola_cache_out = self.istft(log_mag, phase_sin, phase_cos, ola_cache_in)
        vocoder_cache_out = torch.stack(next_caches, dim=0)
        return audio_out, vocoder_cache_out, ola_cache_out
