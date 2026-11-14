"""Causal Conv1d, Causal ConvNeXt, AdaLN/FiLM conditioning, and time embeddings."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Continuous timestep embedding for Flow Matching t in [0, 1]."""

    def __init__(self, dim: int, max_period: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half_dim, dtype=torch.float32)
            / float(max(half_dim - 1, 1))
        )
        self.register_buffer("freqs", freqs * 1000.0)

        self.mlp = nn.Sequential(
            nn.Linear(half_dim * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed timestep tensor t of shape (B, 1) into (B, dim)."""
        args = t * self.freqs.unsqueeze(0)  # (B, half_dim)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class ChannelLayerNorm(nn.Module):
    """LayerNorm applied directly across channel dimension of (B, C, T) tensors."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        normed = (x - mean) * torch.rsqrt(var + self.eps)
        return normed * self.weight + self.bias


class CausalConv1d(nn.Module):
    """Strictly causal 1D convolution with explicit ring/tail state caching."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pad_len = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=0,
            groups=groups,
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        cache_in: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply causal convolution.

        Args:
            x: Input tensor of shape (B, C_in, T).
            cache_in: Left context cache of shape (B, C_in, pad_len).
                Required when pad_len > 0; if None, zero-padded left context is used.

        Returns:
            out: Output tensor of shape (B, C_out, T).
            cache_out: Updated left context of shape (B, C_in, pad_len) or None.
        """
        if self.pad_len == 0:
            return self.conv(x), None

        if cache_in is None:
            cache_in = torch.zeros(
                x.shape[0],
                self.in_channels,
                self.pad_len,
                dtype=x.dtype,
                device=x.device,
            )
        combined = torch.cat([cache_in, x], dim=-1)
        cache_out = combined[:, :, -self.pad_len :]
        out = self.conv(combined)
        return out, cache_out


class CausalConvNeXtBlock(nn.Module):
    """Causal ConvNeXt block with optional AdaLN/FiLM conditioning."""

    def __init__(
        self,
        dim: int = 256,
        kernel_size: int = 7,
        mlp_ratio: int = 2,
        use_cond: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.pad_len = kernel_size - 1
        self.use_cond = use_cond

        self.dwconv = CausalConv1d(dim, dim, kernel_size=kernel_size, groups=dim)
        self.norm = ChannelLayerNorm(dim)
        self.pwconv1 = nn.Conv1d(dim, dim * mlp_ratio, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(dim * mlp_ratio, dim, kernel_size=1)

        if use_cond:
            self.cond_act = nn.SiLU()
            self.cond_proj = nn.Linear(dim, dim * 3)
            # Initialize gate bias to 1.0 and scale/shift to small values for smooth gradient flow
            nn.init.zeros_(self.cond_proj.bias)
            with torch.no_grad():
                self.cond_proj.bias[2 * dim :] = 1.0
        else:
            self.gamma = nn.Parameter(torch.ones(1, dim, 1) * 0.1)

    def forward(
        self,
        x: torch.Tensor,
        cache_in: torch.Tensor,
        cond_vec: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through Causal ConvNeXt block.

        Args:
            x: Feature tensor of shape (B, dim, T).
            cache_in: Causal convolution cache of shape (B, dim, pad_len).
            cond_vec: Optional conditioning vector of shape (B, dim).

        Returns:
            out: Output tensor of shape (B, dim, T).
            cache_out: Updated cache of shape (B, dim, pad_len).
        """
        residual = x
        h, cache_out = self.dwconv(x, cache_in)
        h = self.norm(h)

        if self.use_cond:
            if cond_vec is None:
                raise ValueError("cond_vec must be provided when use_cond=True.")
            mod = self.cond_proj(self.cond_act(cond_vec)).unsqueeze(-1)  # (B, 3 * dim, 1)
            scale, shift, gate = torch.chunk(mod, 3, dim=1)
            h = h * (1.0 + scale) + shift
            h = self.pwconv2(self.act(self.pwconv1(h)))
            out = residual + gate * h
        else:
            h = self.pwconv2(self.act(self.pwconv1(h)))
            out = residual + self.gamma * h

        return out, cache_out
