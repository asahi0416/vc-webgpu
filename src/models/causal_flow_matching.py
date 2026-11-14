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

    def compute_target_base_mel(
        self,
        src_mel: torch.Tensor,
        spk_emb: torch.Tensor,
        vtl_shift: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build causal zero-shot target envelope + VTL-shifted formant anchor mel spectrogram."""
        tgt_shape_raw = (spk_emb[:, : self.mel_dim] * 26.0).unsqueeze(-1)
        tgt_broad = torch.matmul(self.mel_env_mat, tgt_shape_raw)
        tgt_mix = 0.65 * tgt_shape_raw + 0.35 * tgt_broad
        tgt_shape = tgt_mix - tgt_mix.mean(dim=1, keepdim=True)

        src_broad = torch.matmul(self.mel_env_mat, src_mel)
        src_shape = src_broad - src_broad.mean(dim=1, keepdim=True)
        src_detail = src_mel - src_broad

        if vtl_shift is None:
            vtl_shift = spk_emb[:, self.mel_dim : self.mel_dim + 1].unsqueeze(-1) / 0.25

        warp_pos = torch.clamp(2.0 + vtl_shift * 3.5, 0.0, 4.0)
        warp_w = torch.clamp(1.0 - torch.abs(warp_pos - self.warp_indices), min=0.0)
        all_warped = torch.matmul(self.vtl_warp_bank, src_detail.unsqueeze(1))
        warped_detail = (all_warped * warp_w.unsqueeze(2)).sum(dim=1)

        src_frame_mean = src_mel.mean(dim=1, keepdim=True)
        voice_gate = torch.sigmoid((src_frame_mean + 8.8) * 2.0)
        norm_frame_mean = -6.8 + 1.8 * torch.tanh((src_frame_mean + 6.8) / 1.8)
        return (
            norm_frame_mean
            + voice_gate * tgt_shape
            + (1.0 - voice_gate) * src_shape
            + 0.55 * warped_detail
        )

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
            x = anchor + 1.0 * torch.tanh(x - anchor)

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
