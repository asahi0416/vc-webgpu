"""Zero-shot Speaker Encoder with Multi-scale Conv1d blocks and Attentive Statistics Pooling."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.audio_dsp import CausalMelSpectrogram, N_MELS, STFT_CACHE_LEN
from src.models.causal_blocks import ChannelLayerNorm


class Res2ConvBlock(nn.Module):
    """Multi-scale 1D Res2Net-style convolutional block for speaker feature extraction."""

    def __init__(self, dim: int = 256, scale: int = 4, kernel_size: int = 5) -> None:
        super().__init__()
        if dim % scale != 0:
            raise ValueError("dim must be divisible by scale.")
        self.scale = scale
        self.width = dim // scale
        padding = kernel_size // 2

        self.in_proj = nn.Conv1d(dim, dim, kernel_size=1)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(self.width, self.width, kernel_size=kernel_size, padding=padding)
                for _ in range(scale - 1)
            ]
        )
        self.norm = ChannelLayerNorm(dim)
        self.act = nn.SiLU()
        self.out_proj = nn.Conv1d(dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.act(self.in_proj(x))
        splits = torch.split(h, self.width, dim=1)

        outputs = [splits[0]]
        prev = None
        for i, conv in enumerate(self.convs):
            inp = splits[i + 1] if prev is None else splits[i + 1] + prev
            prev = self.act(conv(inp))
            outputs.append(prev)

        out = torch.cat(outputs, dim=1)
        out = self.out_proj(self.norm(out))
        return residual + out


class AttentiveStatisticsPooling(nn.Module):
    """Attentive Statistics Pooling aggregating temporal features into fixed speaker vector."""

    def __init__(self, dim: int = 256, attn_dim: int = 128) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(dim, attn_dim, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(attn_dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool (B, dim, T) into (B, 2 * dim) weighted mean and std."""
        weights = F.softmax(self.attn(x), dim=-1)  # (B, dim, T)
        mean = torch.sum(x * weights, dim=-1)  # (B, dim)
        var = torch.sum(((x - mean.unsqueeze(-1)) ** 2) * weights, dim=-1)
        std = torch.sqrt(torch.clamp(var, min=1e-6))
        return torch.cat([mean, std], dim=1)


class SpeakerEncoder(nn.Module):
    """Extracts a 256-dim L2-normalized speaker embedding from ~3s reference audio."""

    def __init__(
        self,
        n_mels: int = N_MELS,
        hidden_dim: int = 256,
        spk_dim: int = 256,
        num_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.mel_extractor = CausalMelSpectrogram(n_mels=n_mels)
        self.stft_cache_len = STFT_CACHE_LEN

        self.in_conv = nn.Sequential(
            nn.Conv1d(n_mels, hidden_dim, kernel_size=5, padding=2),
            ChannelLayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [Res2ConvBlock(dim=hidden_dim, scale=4, kernel_size=5) for _ in range(num_blocks)]
        )
        self.asp = AttentiveStatisticsPooling(dim=hidden_dim, attn_dim=128)
        self.out_norm = nn.LayerNorm(hidden_dim * 2)
        self.out_proj = nn.Linear(hidden_dim * 2, spk_dim)

    def encode_mel(self, log_mel: torch.Tensor) -> torch.Tensor:
        """Compute L2-normalized speaker embedding directly from a log-mel spectrogram."""
        h = self.in_conv(log_mel)
        for block in self.blocks:
            h = block(h)

        stats = self.asp(h)  # (B, 2 * hidden_dim)
        emb = self.out_proj(self.out_norm(stats))
        return F.normalize(emb, p=2.0, dim=-1)

    def forward(self, ref_audio: torch.Tensor) -> torch.Tensor:
        """Compute L2-normalized speaker embedding from reference waveform.

        Args:
            ref_audio: Reference waveform tensor of shape (B, T_samples).
                T_samples should be >= 240 (truncated to multiple of hop_length).

        Returns:
            spk_emb: L2-normalized speaker embedding of shape (B, spk_dim).
        """
        hop = self.mel_extractor.hop_length
        usable_len = (ref_audio.shape[1] // hop) * hop
        audio = ref_audio[:, :usable_len]

        # Build zero cache cleanly without dynamic shape ops unfriendly to WebGPU
        stft_cache = audio[:, :1].unsqueeze(1).expand(-1, 1, self.stft_cache_len) * 0.0
        log_mel, _, _ = self.mel_extractor(audio, stft_cache)  # (B, n_mels, T_frames)
        return self.encode_mel(log_mel)
