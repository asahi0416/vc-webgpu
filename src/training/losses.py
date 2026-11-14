"""Training losses and pure-PyTorch causal F0 ground-truth extractor."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.audio_dsp import HOP_LENGTH, N_FFT, SAMPLE_RATE


@torch.no_grad()
def extract_f0_autocorr(
    audio: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    f0_min: float = 65.0,
    f0_max: float = 500.0,
    voicing_thresh: float = 0.50,
    rms_thresh: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract frame-wise ground-truth F0 (Hz) and voiced flag using low-pass NSDF autocorrelation."""
    pad_left = n_fft - hop_length
    padded = F.pad(audio, (pad_left, 0))
    frames = padded.unfold(dimension=-1, size=n_fft, step=hop_length)  # (B, T_frames, N)
    batch_size, t_frames, win_len = frames.shape

    rms = torch.sqrt(torch.mean(frames * frames, dim=-1) + 1e-12)  # (B, T_frames)
    centered = frames - frames.mean(dim=-1, keepdim=True)

    # Low-pass filter in RFFT domain (keep <= 500Hz intact, taper to 900Hz) to avoid formant octave jumps
    fft_size = 1
    while fft_size < 2 * win_len:
        fft_size *= 2
    spec_raw = torch.fft.rfft(centered, n=fft_size, dim=-1)
    freqs = torch.linspace(
        0.0, float(sample_rate) / 2.0, spec_raw.shape[-1], device=audio.device, dtype=audio.dtype
    )
    lp_gain = torch.clamp(1.0 - (freqs - 500.0) / 400.0, 0.0, 1.0)
    frames_lp = torch.fft.irfft(spec_raw * lp_gain, n=fft_size, dim=-1)[:, :, :win_len]
    frames_lp = frames_lp - frames_lp.mean(dim=-1, keepdim=True)

    spec = torch.fft.rfft(frames_lp, n=fft_size, dim=-1)
    acf = torch.fft.irfft(spec.real * spec.real + spec.imag * spec.imag, n=fft_size, dim=-1)
    acf = acf[:, :, :win_len]  # (B, T_frames, N)

    # Compute exact denominator m(tau) = sum_{n=0}^{N-1-tau} x^2 + sum_{n=tau}^{N-1} x^2
    sq = frames_lp * frames_lp
    prefix_sq = torch.cumsum(sq, dim=-1)
    total_energy = prefix_sq[:, :, -1:]  # (B, T_frames, 1)

    left_energy = torch.flip(prefix_sq, dims=[-1])  # index tau corresponds to N-1-tau
    shifted_prefix = F.pad(prefix_sq[:, :, :-1], (1, 0))
    right_energy = total_energy - shifted_prefix
    denom = left_energy + right_energy + 1e-8

    nsdf = (2.0 * acf) / denom  # Normalized Square Difference Function in [-1, 1]

    min_lag = max(2, int(float(sample_rate) / f0_max))
    max_lag = min(win_len - 2, int(float(sample_rate) / f0_min))

    sub_nsdf = nsdf[:, :, min_lag : max_lag + 1]  # (B, T_frames, L)
    max_val, _ = sub_nsdf.max(dim=-1, keepdim=True)

    # Detect local peaks inside [min_lag+1, max_lag-1]
    left = sub_nsdf[:, :, :-2]
    center = sub_nsdf[:, :, 1:-1]
    right = sub_nsdf[:, :, 2:]
    is_peak = (center > left) & (center >= right)
    strong_peak = is_peak & (center >= 0.95 * torch.clamp(max_val, min=0.0))

    has_peak = strong_peak.any(dim=-1)
    first_peak_offset = torch.argmax(strong_peak.to(torch.int64), dim=-1)  # index in center (offset +1)
    raw_argmax = torch.argmax(sub_nsdf, dim=-1)

    best_rel_idx = torch.where(has_peak, first_peak_offset + 1, raw_argmax)
    best_lag = best_rel_idx + min_lag  # (B, T_frames)

    # Gather peak correlation value
    best_corr = torch.gather(nsdf, dim=-1, index=best_lag.unsqueeze(-1)).squeeze(-1)

    # Parabolic interpolation for sub-sample lag precision
    lag_clamped = torch.clamp(best_lag, min=min_lag + 1, max=max_lag - 1)
    y_l = torch.gather(nsdf, dim=-1, index=(lag_clamped - 1).unsqueeze(-1)).squeeze(-1)
    y_c = torch.gather(nsdf, dim=-1, index=lag_clamped.unsqueeze(-1)).squeeze(-1)
    y_r = torch.gather(nsdf, dim=-1, index=(lag_clamped + 1).unsqueeze(-1)).squeeze(-1)
    denom_para = 2.0 * (y_l - 2.0 * y_c + y_r)
    delta = torch.where(
        denom_para.abs() > 1e-6,
        (y_l - y_r) / denom_para,
        torch.zeros_like(denom_para),
    )
    delta = torch.clamp(delta, -0.5, 0.5)
    refined_lag = lag_clamped.to(torch.float32) + delta

    voiced = (rms > rms_thresh) & (best_corr > voicing_thresh)
    f0_hz = torch.where(
        voiced,
        float(sample_rate) / torch.clamp(refined_lag, min=1.0),
        torch.full_like(refined_lag, 150.0),
    )

    return f0_hz.unsqueeze(1), voiced.to(torch.float32).unsqueeze(1)


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT spectral convergence + log-magnitude L1 loss."""

    def __init__(
        self,
        fft_sizes: Tuple[int, ...] = (480, 960, 1920),
        hop_sizes: Tuple[int, ...] = (120, 240, 480),
        win_lengths: Tuple[int, ...] = (480, 960, 1920),
    ) -> None:
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

        for idx, win_len in enumerate(win_lengths):
            self.register_buffer(
                f"window_{idx}",
                torch.hann_window(win_len, periodic=True, dtype=torch.float32),
            )

    def _stft_mag(
        self, x: torch.Tensor, n_fft: int, hop: int, win_len: int, window: torch.Tensor
    ) -> torch.Tensor:
        if x.shape[-1] < n_fft:
            x = F.pad(x, (0, n_fft - x.shape[-1]))
        spec = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop,
            win_length=win_len,
            window=window,
            return_complex=True,
        )
        # Add eps inside sqrt to avoid NaN gradients at exact zero amplitude
        return torch.sqrt(spec.real * spec.real + spec.imag * spec.imag + 1e-7)

    def forward(self, pred_audio: torch.Tensor, target_audio: torch.Tensor) -> torch.Tensor:
        total_sc = 0.0
        total_mag = 0.0
        num_res = len(self.fft_sizes)

        for idx, (n_fft, hop, win_len) in enumerate(
            zip(self.fft_sizes, self.hop_sizes, self.win_lengths)
        ):
            window = getattr(self, f"window_{idx}")
            mag_pred = self._stft_mag(pred_audio, n_fft, hop, win_len, window)
            mag_tgt = self._stft_mag(target_audio, n_fft, hop, win_len, window)

            sc = torch.norm(mag_tgt - mag_pred, p="fro", dim=(-2, -1)) / (
                torch.norm(mag_tgt, p="fro", dim=(-2, -1)) + 1e-5
            )
            log_l1 = F.l1_loss(
                torch.log(torch.clamp(mag_pred, min=1e-5)),
                torch.log(torch.clamp(mag_tgt, min=1e-5)),
            )
            total_sc = total_sc + sc.mean()
            total_mag = total_mag + log_l1

        return (total_sc + total_mag) / float(num_res)


class SpeakerContrastiveLoss(nn.Module):
    """Cosine contrastive loss separating distinct speakers and grouping same-speaker samples."""

    def __init__(self, margin: float = 0.2) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, spk_emb: torch.Tensor, speaker_ids: torch.Tensor) -> torch.Tensor:
        batch_size = spk_emb.shape[0]
        if batch_size < 2:
            return spk_emb.sum() * 0.0

        sim = torch.matmul(spk_emb, spk_emb.transpose(0, 1))  # (B, B)
        eye = torch.eye(batch_size, device=spk_emb.device, dtype=torch.bool)
        same_spk = (speaker_ids.unsqueeze(1) == speaker_ids.unsqueeze(0)) & (~eye)
        diff_spk = speaker_ids.unsqueeze(1) != speaker_ids.unsqueeze(0)

        loss = spk_emb.sum() * 0.0
        if same_spk.any():
            pos_loss = ((1.0 - sim[same_spk]) ** 2).mean()
            loss = loss + pos_loss
        if diff_spk.any():
            neg_loss = (F.relu(sim[diff_spk] - self.margin) ** 2).mean()
            loss = loss + neg_loss
        return loss
