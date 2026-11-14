"""Causal Rectified Flow & Consistency Flow Matching Decoder."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.audio_dsp import N_MELS, build_mel_envelope_and_vtl_banks
from src.models.causal_blocks import (
    CausalConvNeXtBlock,
    ChannelLayerNorm,
    SinusoidalTimeEmbedding,
)


class CausalFlowMatchingDecoder(nn.Module):
    """Causal ConvNeXt vector-field predictor supporting few-step streaming Euler ODE."""

    def __init__(
        self,
        mel_dim: int = N_MELS,
        dim: int = 256,
        num_layers: int = 6,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        self.mel_dim = mel_dim
        self.dim = dim
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.pad_len = kernel_size - 1

        self.time_embed = SinusoidalTimeEmbedding(dim)
        self.spk_proj = nn.Linear(dim, dim)

        self.in_proj = nn.Conv1d(mel_dim + dim, dim, kernel_size=1)
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
        self.out_proj = nn.Conv1d(dim, mel_dim, kernel_size=1)

        mel_env_mat, vtl_warp_bank = build_mel_envelope_and_vtl_banks(n_mels=mel_dim)
        self.register_buffer("mel_env_mat", mel_env_mat)
        self.register_buffer("vtl_warp_bank", vtl_warp_bank)
        self.register_buffer(
            "warp_indices",
            torch.arange(vtl_warp_bank.shape[1], dtype=torch.float32).view(1, -1, 1),
        )
        x_mel = torch.linspace(-1.0, 1.0, mel_dim, dtype=torch.float32)
        canon_tilt = 0.2425 - 2.3392 * x_mel - 0.6541 * (x_mel ** 2) + 1.0055 * (x_mel ** 3)
        canon_tilt = (canon_tilt - canon_tilt.mean()).view(1, mel_dim, 1)
        self.register_buffer("canon_tilt", canon_tilt)
        self.register_buffer("freq_slope", (-x_mel).view(1, mel_dim, 1))

    def compute_target_base_mel(
        self,
        src_mel: torch.Tensor,
        spk_emb: torch.Tensor,
        vtl_shift: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build causal zero-shot speaker-neutral VTL-warped formant + target speaker EQ anchor mel."""
        tgt_shape_raw = (spk_emb[:, : self.mel_dim] * 24.0).unsqueeze(-1)
        tgt_broad = torch.matmul(self.mel_env_mat, tgt_shape_raw)
        tgt_shape = tgt_broad - tgt_broad.mean(dim=1, keepdim=True)

        # Light 3-bin smooth suppresses harmonic comb ripple while keeping crisp vowel formants
        pad3 = torch.cat([src_mel[:, :1, :], src_mel, src_mel[:, -1:, :]], dim=1)
        sm3 = (pad3[:, :-2, :] + pad3[:, 1:-1, :] + pad3[:, 2:, :]) * (1.0 / 3.0)
        src_frame_mean = src_mel.mean(dim=1, keepdim=True)
        src_formant = sm3 - src_frame_mean

        if vtl_shift is None:
            vtl_shift = spk_emb[:, self.mel_dim : self.mel_dim + 1].unsqueeze(-1) / 0.25

        warp_pos = torch.clamp(2.0 + vtl_shift * 3.8, 0.0, 4.0)
        warp_w = torch.clamp(1.0 - torch.abs(warp_pos - self.warp_indices), min=0.0)
        all_warped = torch.matmul(self.vtl_warp_bank, src_formant.unsqueeze(1))
        w_formant = (all_warped * warp_w.unsqueeze(2)).sum(dim=1)
        w_formant = w_formant - w_formant.mean(dim=1, keepdim=True)

        # Strip source speaker's broad spectral tilt so Flow starts near target envelope without source leakage
        w_broad = torch.matmul(self.mel_env_mat, w_formant)
        w_broad = w_broad - w_broad.mean(dim=1, keepdim=True)
        w_detail = w_formant - w_broad

        voice_gate = torch.sigmoid((src_frame_mean + 8.3) * 2.0)
        voiced_norm = -4.8 + 1.2 * torch.tanh((src_frame_mean + 5.6) / 1.8)
        norm_frame_mean = src_frame_mean * (1.0 - voice_gate) + voice_gate * voiced_norm
        return norm_frame_mean + 0.25 * w_detail + voice_gate * tgt_shape

    def forward_velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond_feat: torch.Tensor,
        spk_emb: torch.Tensor,
        layer_caches_in: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict velocity field v_theta(x_t, t | cond_feat, spk_emb) causally.

        Args:
            x_t: Intermediate noisy mel state of shape (B, mel_dim, T).
            t: Timestep tensor of shape (B, 1) in [0, 1].
            cond_feat: Combined content + F0 conditioning of shape (B, dim, T).
            spk_emb: Speaker embedding of shape (B, dim).
            layer_caches_in: Causal conv caches of shape (num_layers, B, dim, pad_len).

        Returns:
            v: Predicted velocity field of shape (B, mel_dim, T).
            layer_caches_out: Updated caches of shape (num_layers, B, dim, pad_len).
        """
        cond_vec = self.time_embed(t) + self.spk_proj(spk_emb)  # (B, dim)
        h = self.in_proj(torch.cat([x_t, cond_feat], dim=1))

        next_caches = []
        for i, block in enumerate(self.blocks):
            h, c_out = block(h, layer_caches_in[i], cond_vec=cond_vec)
            next_caches.append(c_out)

        v = self.out_proj(self.out_norm(h))
        layer_caches_out = torch.stack(next_caches, dim=0)
        return v, layer_caches_out

    def forward_euler_streaming(
        self,
        x_0: torch.Tensor,
        cond_feat: torch.Tensor,
        spk_emb: torch.Tensor,
        flow_cache_in: torch.Tensor,
        num_steps: int = 2,
        base_mel: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run few-step causal Euler ODE integration anchored at target spectral envelope."""
        anchor = base_mel if base_mel is not None else torch.zeros_like(x_0)
        x = anchor + 0.04 * x_0
        dt = 1.0 / float(num_steps)
        t_template = x_0[:, :1, 0] * 0.0  # (B, 1) zero tensor safe for ONNX tracing

        next_step_caches = []
        for k in range(num_steps):
            t_k = t_template + (float(k) * dt)
            v_k, cache_out_k = self.forward_velocity(
                x, t_k, cond_feat, spk_emb, flow_cache_in[k]
            )
            x = x + dt * v_k
            next_step_caches.append(cache_out_k)

        if base_mel is not None:
            x = anchor + 2.80 * torch.tanh((x - anchor) / 2.80)

        flow_cache_out = torch.stack(next_step_caches, dim=0)
        return x, flow_cache_out

    def _zero_layer_caches(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(
            self.num_layers,
            batch_size,
            self.dim,
            self.pad_len,
            device=device,
            dtype=dtype,
        )

    def compute_rectified_flow_loss(
        self,
        x_1: torch.Tensor,
        cond_feat: torch.Tensor,
        spk_emb: torch.Tensor,
        base_mel: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute Optimal-Transport Rectified Flow matching loss from base_mel + low noise."""
        batch_size = x_1.shape[0]
        anchor = base_mel if base_mel is not None else torch.zeros_like(x_1)
        x_0 = anchor + 0.04 * torch.randn_like(x_1)
        t = torch.rand(batch_size, 1, device=x_1.device, dtype=x_1.dtype)
        t_view = t.unsqueeze(-1)

        x_t = (1.0 - t_view) * x_0 + t_view * x_1
        target_v = x_1 - x_0

        caches = self._zero_layer_caches(batch_size, x_1.device, x_1.dtype)
        v_pred, _ = self.forward_velocity(x_t, t, cond_feat, spk_emb, caches)
        return F.mse_loss(v_pred, target_v)

    def compute_consistency_loss(
        self,
        x_1: torch.Tensor,
        cond_feat: torch.Tensor,
        spk_emb: torch.Tensor,
        delta_t: float = 0.05,
        base_mel: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute Consistency Flow Matching loss for 1-2 step distillation."""
        batch_size = x_1.shape[0]
        anchor = base_mel if base_mel is not None else torch.zeros_like(x_1)
        x_0 = anchor + 0.04 * torch.randn_like(x_1)
        t = torch.rand(batch_size, 1, device=x_1.device, dtype=x_1.dtype) * (1.0 - delta_t)
        t_next = t + delta_t

        t_view = t.unsqueeze(-1)
        t_next_view = t_next.unsqueeze(-1)

        x_t = (1.0 - t_view) * x_0 + t_view * x_1
        x_t_next = (1.0 - t_next_view) * x_0 + t_next_view * x_1
        caches = self._zero_layer_caches(batch_size, x_1.device, x_1.dtype)

        v_curr, _ = self.forward_velocity(x_t, t, cond_feat, spk_emb, caches)
        f_curr = x_t + (1.0 - t_view) * v_curr

        with torch.no_grad():
            v_next, _ = self.forward_velocity(x_t_next, t_next, cond_feat, spk_emb, caches)
            f_next = x_t_next + (1.0 - t_next_view) * v_next

        consistency_loss = F.mse_loss(f_curr, f_next)
        velocity_loss = F.mse_loss(v_curr, x_1 - x_0)
        return consistency_loss + 0.5 * velocity_loss
