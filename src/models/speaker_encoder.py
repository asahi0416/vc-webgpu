"""Zero-shot Speaker Encoder with Multi-scale Conv1d blocks and Attentive Statistics Pooling."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.audio_dsp import (
    CausalMelSpectrogram,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
    STFT_CACHE_LEN,
    _hz_to_mel_slaney,
    _mel_to_hz_slaney,
    build_harmonic_comb_bank,
)
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
        self.n_mels = n_mels
        self.spk_dim = spk_dim
        self.mel_extractor = CausalMelSpectrogram(n_mels=n_mels)
        self.stft_cache_len = STFT_CACHE_LEN

        comb_bank, log_f0_cands = build_harmonic_comb_bank(
            sr=SAMPLE_RATE, n_fft=N_FFT, n_cands=96, f0_min=70.0, f0_max=360.0
        )
        self.register_buffer("comb_bank", comb_bank.unsqueeze(0))
        self.register_buffer("log_f0_cands", log_f0_cands)

        mel_min = _hz_to_mel_slaney(torch.tensor(0.0, dtype=torch.float64))
        mel_max = _hz_to_mel_slaney(torch.tensor(float(SAMPLE_RATE) / 2.0, dtype=torch.float64))
        mel_centers = torch.linspace(mel_min, mel_max, n_mels + 2, dtype=torch.float64)[1:-1]
        hz_centers = _mel_to_hz_slaney(mel_centers)
        low_mel_log_hz = torch.log(torch.clamp(hz_centers[1:9], min=70.0, max=360.0)).to(torch.float32)
        self.register_buffer("low_mel_log_hz", low_mel_log_hz.view(1, 8, 1))

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
        self.acoustic_dim = n_mels + 1
        neural_dim = max(16, spk_dim - self.acoustic_dim)
        self.out_proj = nn.Linear(hidden_dim * 2, neural_dim)

    def encode_mel(
        self, log_mel: torch.Tensor, mag: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute L2-normalized speaker embedding encoding voiced envelope, pitch register, and timbre."""
        frame_mean = log_mel.mean(dim=1, keepdim=True)
        w_voiced = F.softmax((frame_mean - frame_mean.max(dim=-1, keepdim=True)[0]) * 2.0, dim=-1)
        mu_ref = (log_mel * w_voiced).sum(dim=-1)  # (B, n_mels)
        mu_shape = mu_ref - mu_ref.mean(dim=-1, keepdim=True)
        emb_env = torch.clamp(mu_shape / 24.0, -0.25, 0.25)

        if mag is not None:
            sal = torch.matmul(self.comb_bank, torch.log(torch.clamp(mag, min=1e-3)))
            comb_probs = F.softmax(sal * 12.0, dim=1)
            f0_log_frames = (comb_probs * self.log_f0_cands).sum(dim=1, keepdim=True)
            max_sal, _ = sal.max(dim=1, keepdim=True)
            low_energy = mag[:, 2:60, :].mean(dim=1, keepdim=True)
            uv_w = (
                w_voiced
                * torch.sigmoid((max_sal - 0.28) * 10.0)
                * torch.sigmoid((low_energy - 0.05) * 20.0)
                + 1e-5
            )
            uv_w = uv_w / uv_w.sum(dim=-1, keepdim=True)
            ref_log_f0 = (f0_log_frames * uv_w).sum(dim=-1)
        else:
            low_probs = F.softmax(log_mel[:, 1:9, :], dim=1)
            f0_log_frames = (low_probs * self.low_mel_log_hz).sum(dim=1, keepdim=True)
            ref_log_f0 = (f0_log_frames * w_voiced).sum(dim=-1)

        emb_f0 = torch.clamp((ref_log_f0 - math.log(180.0)) * 0.25, -0.25, 0.25)
        raw_ac = torch.cat([emb_env, emb_f0], dim=-1)
        ac_sq = (raw_ac * raw_ac).sum(dim=-1, keepdim=True)
        ac_scale = torch.clamp(math.sqrt(0.80) * torch.rsqrt(torch.clamp(ac_sq, min=1e-6)), max=1.0)
        emb_ac = raw_ac * ac_scale

        h = self.in_conv(log_mel)
        for block in self.blocks:
            h = block(h)

        stats = self.asp(h)  # (B, 2 * hidden_dim)
        neural_unit = F.normalize(self.out_proj(self.out_norm(stats)), p=2.0, dim=-1)
        neural_scale = torch.sqrt(torch.clamp(1.0 - (emb_ac * emb_ac).sum(dim=-1, keepdim=True), min=0.20))
        emb = torch.cat([emb_ac, neural_unit * neural_scale], dim=-1)
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
        log_mel, mag, _ = self.mel_extractor(audio, stft_cache)  # (B, n_mels, T_frames)
        return self.encode_mel(log_mel, mag=mag)
