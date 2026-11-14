"""ONNX/WebGPU-compatible Causal STFT, Mel-Spectrogram, and COLA iSTFT modules.

Designed without complex64 tensors or custom operators so that all operations
export cleanly to ONNX opset 17 and execute efficiently on WebGPU.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

SAMPLE_RATE = 24000
N_FFT = 960
HOP_LENGTH = 240
N_MELS = 80
N_FREQ_BINS = N_FFT // 2 + 1  # 481
STFT_CACHE_LEN = N_FFT - HOP_LENGTH  # 720


def _hz_to_mel_slaney(freqs: torch.Tensor) -> torch.Tensor:
    """Convert frequency in Hz to Slaney mel scale."""
    f_min = 0.0
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = torch.log(torch.tensor(6.4, dtype=torch.float64)) / 27.0

    mels = (freqs - f_min) / f_sp
    log_region = freqs >= min_log_hz
    mels = torch.where(
        log_region,
        min_log_mel + torch.log(torch.clamp(freqs, min=min_log_hz) / min_log_hz) / logstep,
        mels,
    )
    return mels


def _mel_to_hz_slaney(mels: torch.Tensor) -> torch.Tensor:
    """Convert Slaney mel scale to frequency in Hz."""
    f_min = 0.0
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = torch.log(torch.tensor(6.4, dtype=torch.float64)) / 27.0

    freqs = f_min + f_sp * mels
    log_region = mels >= min_log_mel
    freqs = torch.where(
        log_region,
        min_log_hz * torch.exp(logstep * (mels - min_log_mel)),
        freqs,
    )
    return freqs


def build_mel_filterbank(
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    n_mels: int = N_MELS,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> torch.Tensor:
    """Build a Slaney-normalized mel filterbank matrix of shape (n_mels, n_fft // 2 + 1)."""
    if f_max is None:
        f_max = float(sr) / 2.0

    n_freqs = n_fft // 2 + 1
    fft_freqs = torch.linspace(0.0, float(sr) / 2.0, n_freqs, dtype=torch.float64)

    mel_min = _hz_to_mel_slaney(torch.tensor(f_min, dtype=torch.float64))
    mel_max = _hz_to_mel_slaney(torch.tensor(f_max, dtype=torch.float64))
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2, dtype=torch.float64)
    hz_points = _mel_to_hz_slaney(mel_points)

    f_diff = hz_points[1:] - hz_points[:-1]
    ramps = hz_points.unsqueeze(1) - fft_freqs.unsqueeze(0)

    lower = -ramps[:-2] / f_diff[:-1].unsqueeze(1)
    upper = ramps[2:] / f_diff[1:].unsqueeze(1)
    fb = torch.maximum(torch.zeros_like(lower), torch.minimum(lower, upper))

    # Slaney area normalization
    enorm = 2.0 / (hz_points[2:] - hz_points[:-2])
    fb *= enorm.unsqueeze(1)
    return fb.to(torch.float32)


class CausalMelSpectrogram(nn.Module):
    """Causal STFT and log-mel spectrogram extractor using Conv1d."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        n_mels: int = N_MELS,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_freq_bins = n_fft // 2 + 1
        self.cache_len = n_fft - hop_length

        window = torch.hann_window(n_fft, periodic=True, dtype=torch.float64)
        n = torch.arange(n_fft, dtype=torch.float64).unsqueeze(0)  # (1, N)
        k = torch.arange(self.n_freq_bins, dtype=torch.float64).unsqueeze(1)  # (F, 1)
        angles = 2.0 * torch.pi * k * n / float(n_fft)

        cos_basis = torch.cos(angles) * window.unsqueeze(0)
        sin_basis = -torch.sin(angles) * window.unsqueeze(0)
        # Concatenate real and imag kernels: (2 * F, 1, N)
        stft_kernel = torch.cat([cos_basis, sin_basis], dim=0).unsqueeze(1).to(torch.float32)

        self.register_buffer("stft_kernel", stft_kernel)
        self.register_buffer(
            "mel_basis",
            build_mel_filterbank(sr=sample_rate, n_fft=n_fft, n_mels=n_mels),
        )

    def forward(
        self,
        audio_chunk: torch.Tensor,
        stft_cache_in: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract causal log-mel spectrogram and magnitude spectrum.

        Args:
            audio_chunk: Input waveform tensor of shape (B, T_samples),
                where T_samples is a multiple of hop_length.
            stft_cache_in: Previous waveform tail of shape (B, 1, cache_len).

        Returns:
            log_mel: Log-mel spectrogram of shape (B, n_mels, T_frames).
            mag: Linear magnitude spectrum of shape (B, n_freq_bins, T_frames).
            stft_cache_out: Updated waveform tail of shape (B, 1, cache_len).
        """
        audio_in = audio_chunk.unsqueeze(1)  # (B, 1, T_samples)
        combined = torch.cat([stft_cache_in, audio_in], dim=-1)  # (B, 1, cache_len + T_samples)
        stft_cache_out = combined[:, :, -self.cache_len :]

        spec = F.conv1d(combined, self.stft_kernel, stride=self.hop_length, padding=0)
        real = spec[:, : self.n_freq_bins, :]
        imag = spec[:, self.n_freq_bins :, :]

        mag = torch.sqrt(real * real + imag * imag + 1e-9)
        mel = torch.matmul(self.mel_basis, mag)
        log_mel = torch.log(torch.clamp(mel, min=1e-5))
        return log_mel, mag, stft_cache_out


class CausalISTFT(nn.Module):
    """Causal inverse STFT with COLA Overlap-Add synthesis using standard tensor ops."""

    def __init__(
        self,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
    ) -> None:
        super().__init__()
        if n_fft % hop_length != 0:
            raise ValueError("n_fft must be an integer multiple of hop_length.")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_freq_bins = n_fft // 2 + 1
        self.cache_len = n_fft - hop_length
        self.n_subblocks = n_fft // hop_length

        window = torch.hann_window(n_fft, periodic=True, dtype=torch.float64)
        # Compute exact COLA normalization factor for analysis*synthesis window product
        cola_sum = torch.zeros(hop_length, dtype=torch.float64)
        for i in range(self.n_subblocks):
            w_sub = window[i * hop_length : (i + 1) * hop_length]
            cola_sum = cola_sum + w_sub * w_sub
        # Mean COLA factor (1.5 for 75% overlap Hann window)
        synth_window = window / cola_sum.repeat(self.n_subblocks)

        n = torch.arange(n_fft, dtype=torch.float64).unsqueeze(1)  # (N, 1)
        k = torch.arange(self.n_freq_bins, dtype=torch.float64).unsqueeze(0)  # (1, F)
        angles = 2.0 * torch.pi * k * n / float(n_fft)

        # Scale factors: DC (k=0) and Nyquist (k=N/2) appear once; others appear twice
        scale = torch.full((1, self.n_freq_bins), 2.0 / float(n_fft), dtype=torch.float64)
        scale[0, 0] = 1.0 / float(n_fft)
        if n_fft % 2 == 0:
            scale[0, -1] = 1.0 / float(n_fft)

        cos_synth = (torch.cos(angles) * scale) * synth_window.unsqueeze(1)
        # x[n] = (1/N) * sum (R_k cos - I_k sin) -> imag term has minus sign
        sin_synth = (-torch.sin(angles) * scale) * synth_window.unsqueeze(1)

        # Weight for Conv1d(in_channels=2*F, out_channels=N, kernel_size=1): (N, 2*F, 1)
        idft_kernel = torch.cat([cos_synth, sin_synth], dim=1).unsqueeze(-1).to(torch.float32)
        self.register_buffer("idft_kernel", idft_kernel)

    def forward(
        self,
        log_mag: torch.Tensor,
        phase_sin: torch.Tensor,
        phase_cos: torch.Tensor,
        ola_cache_in: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Synthesize time-domain waveform chunk causally using Overlap-Add.

        Args:
            log_mag: Log-magnitude spectrum of shape (B, n_freq_bins, T_frames).
            phase_sin: Sine of phase of shape (B, n_freq_bins, T_frames).
            phase_cos: Cosine of phase of shape (B, n_freq_bins, T_frames).
            ola_cache_in: Previous OLA tail buffer of shape (B, cache_len).

        Returns:
            audio_out: Reconstructed waveform of shape (B, T_frames * hop_length).
            ola_cache_out: Updated OLA tail buffer of shape (B, cache_len).
        """
        mag = torch.exp(torch.clamp(log_mag, max=10.0))
        real = mag * phase_cos
        imag = mag * phase_sin
        spec = torch.cat([real, imag], dim=1)  # (B, 2 * F, T_frames)

        frames_time = F.conv1d(spec, self.idft_kernel, stride=1, padding=0)  # (B, N, T_frames)
        batch_size = frames_time.shape[0]
        t_frames = frames_time.shape[2]
        out_samples = t_frames * self.hop_length

        # Vectorized shift-and-add OLA across all subblocks without Python loops
        canvas = F.pad(ola_cache_in, (0, out_samples))  # (B, cache_len + out_samples)
        for i in range(self.n_subblocks):
            subblock = frames_time[
                :, i * self.hop_length : (i + 1) * self.hop_length, :
            ]  # (B, hop, T_frames)
            # Transpose to (B, T_frames, hop) and flatten to (B, out_samples)
            flat_sub = subblock.transpose(1, 2).reshape(batch_size, out_samples)
            left_pad = i * self.hop_length
            right_pad = (self.n_subblocks - 1 - i) * self.hop_length
            canvas = canvas + F.pad(flat_sub, (left_pad, right_pad))

        audio_out = canvas[:, :out_samples]
        ola_cache_out = canvas[:, out_samples:]
        return audio_out, ola_cache_out
