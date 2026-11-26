"""Causal Content Encoder and Causal Pitch (F0) Conditioner."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.audio_dsp import (
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
    build_harmonic_comb_bank,
    build_mel_envelope_and_vtl_banks,
)
from src.models.causal_blocks import CausalConvNeXtBlock, ChannelLayerNorm


class _TrainingBottleneckNoise(nn.Module):
    """Inject additive Gaussian noise during training to limit continuous speaker channel capacity."""

    def __init__(self, noise_std: float = 0.25) -> None:
        super().__init__()
        self.noise_std = noise_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.noise_std > 0.0:
            return x + self.noise_std * torch.randn_like(x)
        return x


class CausalContentEncoder(nn.Module):
    """Extracts speaker-agnostic phonetic content features causally from log-mel."""

    def __init__(
        self,
        n_mels: int = N_MELS,
        dim: int = 256,
        bottleneck_dim: int = 48,
        num_layers: int = 3,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.dim = dim
        self.pad_len = kernel_size - 1

        mel_env_mat, _ = build_mel_envelope_and_vtl_banks(n_mels=n_mels)
        self.register_buffer("mel_env_mat", mel_env_mat)

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
        # Information bottleneck + training noise strips residual timbre while preserving linguistic content
        self.bottleneck = nn.Sequential(
            nn.Conv1d(dim, bottleneck_dim, kernel_size=1),
            nn.Tanh(),
            ChannelLayerNorm(bottleneck_dim),
            _TrainingBottleneckNoise(noise_std=0.05),
            nn.Conv1d(bottleneck_dim, dim, kernel_size=1),
        )

    def _normalize_frame_mel(self, log_mel: torch.Tensor) -> torch.Tensor:
        """Smooth narrowband harmonics and normalize per frame while preserving vowel formants."""
        padded = torch.cat(
            [
                log_mel[:, :1, :],
                log_mel[:, :1, :],
                log_mel,
                log_mel[:, -1:, :],
                log_mel[:, -1:, :],
            ],
            dim=1,
        )
        smoothed = (
            padded[:, :-4, :]
            + padded[:, 1:-3, :]
            + padded[:, 2:-2, :]
            + padded[:, 3:-1, :]
            + padded[:, 4:, :]
        ) * 0.2
        mean = smoothed.mean(dim=1, keepdim=True)
        var = ((smoothed - mean) ** 2).mean(dim=1, keepdim=True)
        return (smoothed - mean) * torch.rsqrt(var + 1e-5)

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
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.pad_len = kernel_size - 1
        # Bins covering 0 Hz to ~1000 Hz
        freq_resolution = float(sample_rate) / float(n_fft)
        self.low_freq_bins = min(int(1000.0 / freq_resolution) + 1, n_fft // 2 + 1)

        comb_bank, log_f0_cands = build_harmonic_comb_bank(
            sr=sample_rate, n_fft=n_fft, n_cands=128, f0_min=85.0, f0_max=750.0
        )
        self.register_buffer("comb_bank", comb_bank.unsqueeze(0), persistent=False)
        self.register_buffer("log_f0_cands", log_f0_cands, persistent=False)

        # Causal time-domain YIN buffers (non-persistent so existing checkpoints load cleanly)
        self.yin_win = 600
        self.yin_hop = 240
        self.yin_min_tau = 32   # 750 Hz @ 24 kHz
        self.yin_max_tau = 282  # 85.1 Hz @ 24 kHz
        self.yin_n_lags = self.yin_max_tau - self.yin_min_tau + 1  # 251
        self.yin_span = self.yin_win + self.yin_max_tau  # 882
        self.yin_offset = max(0, n_fft - self.yin_span)  # 78
        self.register_buffer(
            "yin_eye",
            torch.eye(self.yin_span, dtype=torch.float32).unsqueeze(1),
            persistent=False,
        )
        lags = torch.arange(self.yin_min_tau, self.yin_max_tau + 1, dtype=torch.long).unsqueeze(1)
        win_pos = torch.arange(self.yin_win, dtype=torch.long).unsqueeze(0)
        yin_idx = (self.yin_span - self.yin_win - lags) + win_pos  # (251, 600)
        self.register_buffer("yin_idx", yin_idx, persistent=False)
        self.register_buffer(
            "yin_tau_counts",
            torch.arange(1, self.yin_n_lags + 1, dtype=torch.float32).view(1, 1, self.yin_n_lags),
            persistent=False,
        )

        in_dim = self.low_freq_bins + n_mels + 2
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
        self.f0_head = nn.Conv1d(hidden_dim, 3, kernel_size=1)
        self.spk_f0_head = nn.Sequential(
            nn.Linear(out_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

        # Log-frequency bounds [50 Hz, 1100 Hz] and smooth speaker base register bounds [85 Hz, 320 Hz]
        self.log_f0_min = math.log(50.0)
        self.log_f0_range = math.log(1100.0) - math.log(50.0)
        self.log_base_min = math.log(85.0)
        self.log_base_range = math.log(320.0) - math.log(85.0)
        self.last_src_f0_hz: Optional[torch.Tensor] = None
        self.last_src_base_log_f0: Optional[torch.Tensor] = None
        self.last_tgt_base_log_f0: Optional[torch.Tensor] = None

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

    @torch.no_grad()
    def _estimate_yin_f0(self, audio_buf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Causal time-domain YIN F0 estimator (ONNX/WebGPU friendly, no Python loops)."""
        x_slice = audio_buf[:, self.yin_offset :]
        frames = F.conv1d(x_slice.unsqueeze(1), self.yin_eye, stride=self.yin_hop).transpose(1, 2)
        x_ref = frames[:, :, self.yin_span - self.yin_win : self.yin_span]  # (B, T, 600)
        x_lag = frames[:, :, self.yin_idx]  # (B, T, 251, 600)
        diff = x_ref.unsqueeze(2) - x_lag
        d = (diff * diff).sum(dim=-1)  # (B, T, 251)

        cum = torch.cumsum(d, dim=-1)
        cmnd = d * self.yin_tau_counts / (cum + 1e-8)

        below = cmnd < 0.22
        is_min = (cmnd <= F.pad(cmnd[..., :-1], (1, 0), value=1.0)) & (
            cmnd <= F.pad(cmnd[..., 1:], (0, 1), value=1.0)
        )
        valid_dip = below & is_min
        has_dip = valid_dip.any(dim=-1)
        first_idx = torch.argmax(valid_dip.to(torch.int64), dim=-1)
        global_idx = torch.argmin(cmnd, dim=-1)
        best_idx = torch.where(has_dip, first_idx, global_idx)

        c_idx = torch.clamp(best_idx, min=1, max=self.yin_n_lags - 2)
        y_m1 = torch.gather(cmnd, -1, (c_idx - 1).unsqueeze(-1)).squeeze(-1)
        y_0 = torch.gather(cmnd, -1, c_idx.unsqueeze(-1)).squeeze(-1)
        y_p1 = torch.gather(cmnd, -1, (c_idx + 1).unsqueeze(-1)).squeeze(-1)
        denom = 2.0 * (y_m1 - 2.0 * y_0 + y_p1)
        delta = torch.where(
            denom.abs() > 1e-6, (y_m1 - y_p1) / denom, torch.zeros_like(denom)
        ).clamp(-0.5, 0.5)

        tau_est = float(self.yin_min_tau) + best_idx.to(cmnd.dtype) + delta
        yin_f0_hz = float(self.sample_rate) / torch.clamp(
            tau_est, min=float(self.yin_min_tau), max=float(self.yin_max_tau)
        )
        min_cmnd = torch.gather(cmnd, -1, best_idx.unsqueeze(-1)).squeeze(-1)
        frame_rms = torch.sqrt((x_ref * x_ref).mean(dim=-1) + 1e-10)
        yin_uv = torch.sigmoid((0.35 - min_cmnd) * 18.0) * torch.sigmoid((frame_rms - 0.005) * 400.0)
        yin_log_f0 = torch.log(torch.clamp(yin_f0_hz, min=50.0, max=800.0)).unsqueeze(1)
        return yin_log_f0, yin_uv.unsqueeze(1)

    def forward(
        self,
        mag: torch.Tensor,
        log_mel: torch.Tensor,
        pitch_shift_semitones: torch.Tensor,
        cache_in: torch.Tensor,
        spk_emb: Optional[torch.Tensor] = None,
        audio_buf: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Estimate F0 causally via YIN (or harmonic comb fallback) + ConvNeXt smoothing."""
        low_band_energy = mag[:, 2:60, :].mean(dim=1, keepdim=True)
        if audio_buf is not None:
            comb_log_f0, yin_uv = self._estimate_yin_f0(audio_buf)
            comb_uv = yin_uv * torch.sigmoid((low_band_energy - 0.015) * 100.0)
        else:
            sal = torch.matmul(self.comb_bank, torch.log(torch.clamp(mag, min=1e-3)))
            comb_probs = F.softmax(sal * 35.0, dim=1)
            comb_log_f0 = (comb_probs * self.log_f0_cands).sum(dim=1, keepdim=True)
            max_sal, _ = sal.max(dim=1, keepdim=True)
            # Scale-tolerant voicing gate: reject weak/noisy consonant frames that cause octave jumps
            comb_uv = torch.sigmoid((max_sal - 0.72) * 10.0) * torch.sigmoid(
                (low_band_energy - 0.025) * 80.0
            )
        comb_log_norm = (comb_log_f0 - math.log(180.0)) * comb_uv

        low_mag = torch.log(torch.clamp(mag[:, : self.low_freq_bins, :], min=1e-5))
        low_mag_norm = low_mag - low_mag.mean(dim=1, keepdim=True)
        log_mel_norm = log_mel - log_mel.mean(dim=1, keepdim=True)
        x = torch.cat([low_mag_norm, log_mel_norm, comb_log_norm, comb_uv], dim=1)
        h = self.in_proj(x)

        next_caches = []
        for i, block in enumerate(self.blocks):
            h, c_out = block(h, cache_in[i], cond_vec=None)
            next_caches.append(c_out)

        pred = self.f0_head(h)  # (B, 3, T)
        src_base_log_f0 = self.log_base_min + torch.sigmoid(pred[:, 2:3, :]) * self.log_base_range
        # Track harmonic comb F0 faithfully across singing & speech (70-750 Hz) with tiny causal smoothing
        src_log_f0 = (
            comb_uv * comb_log_f0
            + (1.0 - comb_uv) * src_base_log_f0
            + 0.06 * torch.tanh(pred[:, 0:1, :])
        )
        uv = comb_uv * torch.sigmoid(pred[:, 1:2, :] + 0.5)

        if spk_emb is not None:
            ref_log_f0 = (
                spk_emb[:, self.n_mels : self.n_mels + 1].unsqueeze(-1) / 0.25
                + math.log(180.0)
            )
            tgt_base_log_f0 = torch.clamp(
                ref_log_f0 + 0.08 * torch.tanh(self.spk_f0_head(spk_emb)).unsqueeze(-1),
                min=self.log_base_min,
                max=math.log(340.0),
            )
        else:
            tgt_base_log_f0 = src_base_log_f0
        # Preserve input melody/pitch at 0 st (user controls octave/key via pitch_shift_semitones)
        adapted_log_f0 = src_log_f0

        self.last_src_f0_hz = torch.exp(src_log_f0)
        self.last_src_base_log_f0 = src_base_log_f0
        self.last_tgt_base_log_f0 = tgt_base_log_f0

        # Shift adapted F0 in Hz: f0 * 2^(semitones / 12)
        shift_factor = torch.pow(2.0, pitch_shift_semitones.unsqueeze(-1) / 12.0)
        shifted_f0 = torch.clamp(torch.exp(adapted_log_f0) * shift_factor, min=50.0, max=800.0)

        # Harmonic sinusoidal embedding masked by voiced probability
        norm_hz = (shifted_f0 / 1000.0) * self.harmonics * (2.0 * math.pi)
        sin_feat = torch.sin(norm_hz) * uv
        cos_feat = torch.cos(norm_hz) * uv
        shifted_log_norm = ((torch.log(torch.clamp(shifted_f0, min=10.0)) - self.log_f0_min) / self.log_f0_range) * uv

        raw_embed = torch.cat([sin_feat, cos_feat, shifted_log_norm, uv], dim=1)
        f0_feat = self.embed_proj(raw_embed)
        cache_out = torch.stack(next_caches, dim=0)
        return f0_feat, shifted_f0, uv, cache_out
