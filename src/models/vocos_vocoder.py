"""Causal Harmonic-Plus-Noise Source-Filter Vocos Vocoder with COLA CausalISTFT."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.audio_dsp import (
    CausalISTFT,
    CausalMelSpectrogram,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
    build_cepstral_min_phase_matrix,
    build_mel_filterbank,
)
from src.models.causal_blocks import CausalConvNeXtBlock, ChannelLayerNorm


class CausalVocosVocoder(nn.Module):
    """Causal NSF-Vocos vocoder combining F0 harmonic-plus-noise excitation with spectral filtering."""

    def __init__(
        self,
        mel_dim: int = N_MELS,
        dim: int = 256,
        num_layers: int = 4,
        kernel_size: int = 7,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        n_harmonics: int = 20,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError(f"num_layers must be >= 2 (got {num_layers})")

        self.mel_dim = mel_dim
        self.dim = dim
        # Keep external cache tensor count = num_layers (4); slot num_layers-1 stores excitation caches.
        self.num_layers = num_layers
        self.active_layers = num_layers - 1
        self.pad_len = kernel_size - 1
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.cache_len = n_fft - hop_length
        self.n_freq_bins = n_fft // 2 + 1
        self.n_harmonics = n_harmonics
        self.exc_slots = self.cache_len // self.pad_len  # 720 // 6 = 120

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
                for _ in range(self.active_layers)
            ]
        )
        self.out_norm = ChannelLayerNorm(dim)
        self.head = nn.Conv1d(dim, self.n_freq_bins * 2, kernel_size=1)
        self.istft = CausalISTFT(n_fft=n_fft, hop_length=hop_length)

        # Pseudo-inverse mel lifter for smooth spectral envelope initialization
        fb = build_mel_filterbank(sr=SAMPLE_RATE, n_fft=n_fft, n_mels=mel_dim)
        log_fb_sum = torch.log(torch.clamp(fb.sum(dim=1), min=1e-6)).view(1, mel_dim, 1)
        self.register_buffer("log_fb_sum", log_fb_sum)
        fb_pinv = fb.T / (fb.T.sum(dim=-1, keepdim=True) + 1e-6)
        self.register_buffer("mel_lifter", fb_pinv.unsqueeze(0))

        # Local frequency envelope smoothing matrix (33 bins ~= 825 Hz) to whiten excitation
        freq_idx = torch.arange(self.n_freq_bins)
        freq_mat = ((freq_idx.unsqueeze(0) - freq_idx.unsqueeze(1)).abs() <= 16).to(torch.float32)
        freq_mat = freq_mat / freq_mat.sum(dim=-1, keepdim=True)
        self.register_buffer("freq_smooth_mat", freq_mat.unsqueeze(0))

        fft_freqs = torch.linspace(0.0, float(SAMPLE_RATE) / 2.0, self.n_freq_bins).view(
            1, self.n_freq_bins, 1
        )
        self.register_buffer("fft_freqs", fft_freqs)
        # Exact linear causal minimum-phase Hilbert operator over filter log-magnitude
        self.register_buffer("min_phase_mat", build_cepstral_min_phase_matrix(n_fft=n_fft))

        # Causal STFT kernel for harmonic-plus-noise excitation analysis
        stft_ref = CausalMelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=n_fft, hop_length=hop_length, n_mels=mel_dim
        )
        self.register_buffer("stft_kernel", stft_ref.stft_kernel.clone())

        # Fixed harmonic and deterministic frame-noise excitation buffers
        harmonics = torch.arange(1, n_harmonics + 1, dtype=torch.float32).view(1, n_harmonics, 1)
        self.register_buffer("harmonics", harmonics)
        # Glottal -12 dB/octave rolloff reduces metallic buzz and improves phase stability
        self.register_buffer("harm_weights", 1.0 / harmonics)

        # Schroeder phase offsets prevent coherent zero-phase impulse spikes across harmonics
        k_idx = torch.arange(1, n_harmonics + 1, dtype=torch.float32)
        theta_k = (math.pi * k_idx * (k_idx - 1.0) / float(n_harmonics)).view(1, n_harmonics, 1)
        self.register_buffer("cos_schroeder", torch.cos(theta_k))
        self.register_buffer("sin_schroeder", torch.sin(theta_k))

        alpha_ramp = (
            torch.arange(1, hop_length + 1, dtype=torch.float32).view(1, 1, 1, hop_length)
            / float(hop_length)
        )
        self.register_buffer("alpha_ramp", alpha_ramp)

        sample_ramp = alpha_ramp * (float(hop_length) / float(SAMPLE_RATE))
        self.register_buffer("sample_ramp", sample_ramp)

        rng = torch.Generator()
        rng.manual_seed(42)
        noise_basis = torch.randn(mel_dim, hop_length, generator=rng, dtype=torch.float32) / math.sqrt(
            float(mel_dim)
        )
        self.register_buffer("noise_basis", noise_basis)
        # Gentle high-pass noise basis for voiced aspiration with zero chunk-boundary dependency
        hp_spec = torch.fft.rfft(noise_basis, dim=-1)
        hp_freqs = torch.linspace(0.0, 1.0, hp_spec.shape[-1], dtype=torch.float32)
        hp_gain = 0.20 + 0.80 * torch.clamp((hp_freqs - 0.15) / 0.35, 0.0, 1.0)
        noise_hp_basis = torch.fft.irfft(hp_spec * hp_gain.unsqueeze(0), n=hop_length, dim=-1)
        self.register_buffer("noise_hp_basis", noise_hp_basis)

    @staticmethod
    def _smooth_mel(mel: torch.Tensor) -> torch.Tensor:
        """Apply 5-bin smoothing across frequency axis without ONNX 3D edge-pad ops."""
        m_pad = torch.cat(
            [mel[:, :1, :], mel[:, :1, :], mel, mel[:, -1:, :], mel[:, -1:, :]], dim=1
        )
        return (
            m_pad[:, :-4, :]
            + m_pad[:, 1:-3, :]
            + m_pad[:, 2:-2, :]
            + m_pad[:, 3:-1, :]
            + m_pad[:, 4:, :]
        ) * 0.2

    def _synthesize_excitation_spectrum(
        self,
        f0_hz: torch.Tensor,
        uv: torch.Tensor,
        noise_seed: torch.Tensor,
        stft_exc_cache: torch.Tensor,
        cos_m1_state: torch.Tensor,
        sin_state: torch.Tensor,
        prev_f0_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate sample-level continuous-F0 Schroeder-phase harmonic+noise excitation and analyze its STFT."""
        batch_size, _, t_frames = f0_hz.shape
        cos0 = cos_m1_state + 1.0
        sin0 = sin_state

        # Initialize prev_f0 on stream start (when cached state is 0) to exact starting F0 for causal equivalence
        prev_f0_init = torch.where(prev_f0_state > 1.0, prev_f0_state, f0_hz[:, :, :1])
        prev_f0_seq = torch.cat([prev_f0_init, f0_hz[:, :, :-1]], dim=-1)

        # Quadratic intra-frame phase sweeps smoothly from prev_f0 to curr f0 with C1 boundary continuity
        hop_sec = float(self.hop_length) / float(SAMPLE_RATE)
        d_raw = 0.5 * (prev_f0_seq + f0_hz) * hop_sec
        d_frac = d_raw - torch.floor(d_raw)
        cum_cycles = torch.cumsum(d_frac, dim=-1)
        prev_cycles = torch.cat([cum_cycles[:, :, :1] * 0.0, cum_cycles[:, :, :-1]], dim=-1)

        prev_rad = (2.0 * math.pi) * prev_cycles
        cp = torch.cos(prev_rad)
        sp = torch.sin(prev_rad)
        c_start = cos0 * cp - sin0 * sp
        s_start = sin0 * cp + cos0 * sp

        m_rad = (2.0 * math.pi * hop_sec) * (
            prev_f0_seq.unsqueeze(-1) * self.alpha_ramp
            + 0.5 * (f0_hz - prev_f0_seq).unsqueeze(-1) * (self.alpha_ramp * self.alpha_ramp)
        )
        cm = torch.cos(m_rad)
        sm = torch.sin(m_rad)
        out_samples = t_frames * self.hop_length
        c1 = (c_start.unsqueeze(-1) * cm - s_start.unsqueeze(-1) * sm).reshape(
            batch_size, 1, out_samples
        )
        s1 = (s_start.unsqueeze(-1) * cm + c_start.unsqueeze(-1) * sm).reshape(
            batch_size, 1, out_samples
        )

        # Angle-addition harmonic series with Schroeder phase dispersion k = 1..n_harmonics
        sin_harms = [s1 * self.cos_schroeder[:, 0:1, :] + c1 * self.sin_schroeder[:, 0:1, :]]
        ck, sk = c1, s1
        for k_idx in range(1, self.n_harmonics):
            ck_next = ck * c1 - sk * s1
            sk_next = sk * c1 + ck * s1
            ck, sk = ck_next, sk_next
            sin_harms.append(
                sk * self.cos_schroeder[:, k_idx : k_idx + 1, :]
                + ck * self.sin_schroeder[:, k_idx : k_idx + 1, :]
            )
        sin_bank = torch.cat(sin_harms, dim=1)

        f0_samples = (
            f0_hz.unsqueeze(-1).expand(batch_size, 1, t_frames, self.hop_length).reshape(batch_size, 1, out_samples)
        )
        uv_samples = (
            uv.unsqueeze(-1).expand(batch_size, 1, t_frames, self.hop_length).reshape(batch_size, 1, out_samples)
        )
        harm_freq = self.harmonics * f0_samples
        nyq_mask = (harm_freq < (SAMPLE_RATE * 0.45)).to(f0_hz.dtype)
        # Taper high harmonics (> 5.2 kHz) gently into breathiness without turning [2.3k-5.2k] into noise
        hf_taper = torch.sigmoid((5200.0 - harm_freq) / 900.0)
        harm_wave = (self.harm_weights * nyq_mask * hf_taper * sin_bank).sum(dim=1, keepdim=True)
        # Subtle causal per-frame glottal shimmer softens rigid periodic pulse buzz
        shimmer = (
            (1.0 + 0.06 * torch.tanh(noise_seed[:, :1, :]))
            .unsqueeze(-1)
            .expand(batch_size, 1, t_frames, self.hop_length)
            .reshape(batch_size, 1, out_samples)
        )
        harm_wave = harm_wave * shimmer
        noise_wav = torch.matmul(noise_seed.transpose(1, 2), self.noise_basis).reshape(
            batch_size, 1, out_samples
        )
        noise_hp = torch.matmul(noise_seed.transpose(1, 2), self.noise_hp_basis).reshape(
            batch_size, 1, out_samples
        )
        exc = (
            uv_samples * harm_wave * 0.085
            + uv_samples * noise_hp * 0.009
            + (1.0 - 0.90 * uv_samples) * noise_wav * 0.032
        ).squeeze(1)

        combined = torch.cat([stft_exc_cache, exc.unsqueeze(1)], dim=-1)
        next_stft_exc_cache = combined[:, :, -self.cache_len :]
        spec = F.conv1d(combined, self.stft_kernel, stride=self.hop_length, padding=0)
        real = spec[:, : self.n_freq_bins, :]
        imag = spec[:, self.n_freq_bins :, :]
        exc_mag = torch.sqrt(real * real + imag * imag + 1e-9)
        local_env = torch.matmul(self.freq_smooth_mat, exc_mag)
        exc_norm = torch.clamp(exc_mag / (local_env + 1e-5), min=0.10, max=4.0)
        cos_exc = real / exc_mag
        sin_exc = imag / exc_mag

        end_rad = (2.0 * math.pi) * cum_cycles[:, :, -1:]
        c_end = cos0 * torch.cos(end_rad) - sin0 * torch.sin(end_rad)
        s_end = sin0 * torch.cos(end_rad) + cos0 * torch.sin(end_rad)
        unit_norm = torch.sqrt(c_end * c_end + s_end * s_end + 1e-12)
        next_cos_m1 = (c_end / unit_norm) - 1.0
        next_sin = s_end / unit_norm
        next_f0 = f0_hz[:, :, -1:]
        return exc_norm, cos_exc, sin_exc, next_stft_exc_cache, next_cos_m1, next_sin, next_f0

    def forward(
        self,
        mel: torch.Tensor,
        spk_emb: torch.Tensor,
        vocoder_cache_in: torch.Tensor,
        ola_cache_in: torch.Tensor,
        f0_hz: Optional[torch.Tensor] = None,
        uv: Optional[torch.Tensor] = None,
        noise_seed: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Synthesize waveform causally using mel envelope filtering over continuous F0 excitation."""
        batch_size, _, t_frames = mel.shape
        if f0_hz is None:
            f0_hz = torch.full((batch_size, 1, t_frames), 200.0, device=mel.device, dtype=mel.dtype)
        if uv is None:
            uv = torch.full((batch_size, 1, t_frames), 0.9, device=mel.device, dtype=mel.dtype)
        if noise_seed is None:
            noise_seed = torch.zeros_like(mel)

        cond_vec = self.spk_proj(spk_emb)
        h = self.in_norm(self.in_proj(mel))

        next_caches = []
        for i, block in enumerate(self.blocks):
            h, c_out = block(h, vocoder_cache_in[i], cond_vec=cond_vec)
            next_caches.append(c_out)

        spec_params = self.head(self.out_norm(h))
        raw_env, raw_disp = torch.chunk(spec_params, 2, dim=1)
        delta_env = 2.5 * torch.tanh(raw_env)
        base_log_env = torch.matmul(self.mel_lifter, mel - self.log_fb_sum)
        filt_log_mag = base_log_env + delta_env
        # Keep harmonics clean up to ~5.2 kHz while tapering Nyquist harmonics gently
        hf_phase_mask = torch.clamp((self.fft_freqs - 1200.0) / 1500.0, 0.0, 1.0)
        phase_disp = torch.matmul(self.min_phase_mat, filt_log_mag) + 0.04 * torch.tanh(raw_disp) * hf_phase_mask

        state_slot = vocoder_cache_in[self.active_layers]
        stft_exc_cache = state_slot[:, : self.exc_slots, :].reshape(batch_size, 1, self.cache_len)
        cos_m1_state = state_slot[:, self.exc_slots : self.exc_slots + 1, 0:1]
        sin_state = state_slot[:, self.exc_slots : self.exc_slots + 1, 1:2]
        prev_f0_state = state_slot[:, self.exc_slots : self.exc_slots + 1, 2:3]

        (
            exc_norm,
            cos_exc,
            sin_exc,
            next_exc_cache,
            next_cos_m1,
            next_sin,
            next_f0,
        ) = self._synthesize_excitation_spectrum(
            f0_hz, uv, noise_seed, stft_exc_cache, cos_m1_state, sin_state, prev_f0_state
        )

        cos_d = torch.cos(phase_disp)
        sin_d = torch.sin(phase_disp)
        cos_p = cos_exc * cos_d - sin_exc * sin_d
        sin_p = sin_exc * cos_d + cos_exc * sin_d

        sub_f0_cut = 3.5 * uv * torch.clamp(
            1.0 - self.fft_freqs / torch.clamp(0.70 * f0_hz, min=45.0, max=185.0),
            0.0,
            1.0,
        )
        total_log_mag = (
            base_log_env
            + delta_env
            + 0.85 * torch.log(torch.clamp(exc_norm, min=0.10, max=4.0))
            - sub_f0_cut
        )
        audio_out, ola_cache_out = self.istft(total_log_mag, sin_p, cos_p, ola_cache_in)

        slot_exc = next_exc_cache.reshape(batch_size, self.exc_slots, self.pad_len)
        slot_phase = torch.cat(
            [next_cos_m1, next_sin, next_f0, state_slot[:, self.exc_slots : self.exc_slots + 1, 3:] * 0.0],
            dim=-1,
        )
        slot_tail = state_slot[:, self.exc_slots + 1 :, :] * 0.0
        next_slot = torch.cat([slot_exc, slot_phase, slot_tail], dim=1)
        next_caches.append(next_slot)

        vocoder_cache_out = torch.stack(next_caches, dim=0)
        return audio_out, vocoder_cache_out, ola_cache_out
