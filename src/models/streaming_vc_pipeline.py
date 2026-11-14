"""Integrated Causal Streaming Flow Matching VC pipeline and wrapper."""

from typing import Dict, Tuple

import torch
import torch.nn as nn

from src.models.audio_dsp import (
    CausalMelSpectrogram,
    HOP_LENGTH,
    N_MELS,
    STFT_CACHE_LEN,
)
from src.models.causal_flow_matching import CausalFlowMatchingDecoder
from src.models.content_f0_extractor import CausalContentEncoder, CausalPitchConditioner
from src.models.speaker_encoder import SpeakerEncoder
from src.models.vocos_vocoder import CausalVocosVocoder


class StreamingVCCore(nn.Module):
    """Streaming chunk-by-chunk voice conversion core (ONNX/WebGPU exportable)."""

    def __init__(self, num_flow_steps: int = 2, dim: int = 256) -> None:
        super().__init__()
        self.num_flow_steps = num_flow_steps
        self.dim = dim

        self.mel_extractor = CausalMelSpectrogram()
        self.content_encoder = CausalContentEncoder(dim=dim, num_layers=3, kernel_size=7)
        self.pitch_conditioner = CausalPitchConditioner(
            hidden_dim=128, out_dim=dim, num_layers=2, kernel_size=7
        )
        self.flow_decoder = CausalFlowMatchingDecoder(
            mel_dim=N_MELS, dim=dim, num_layers=6, kernel_size=7
        )
        self.vocoder = CausalVocosVocoder(
            mel_dim=N_MELS, dim=dim, num_layers=4, kernel_size=7
        )

    def forward(
        self,
        audio_chunk: torch.Tensor,
        spk_emb: torch.Tensor,
        pitch_shift_semitones: torch.Tensor,
        noise_seed: torch.Tensor,
        stft_cache_in: torch.Tensor,
        content_cache_in: torch.Tensor,
        f0_cache_in: torch.Tensor,
        flow_cache_in: torch.Tensor,
        vocoder_cache_in: torch.Tensor,
        ola_cache_in: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Run one causal streaming pass (single chunk or full sequence).

        Args:
            audio_chunk: Waveform of shape (B, T_samples), multiple of 240.
            spk_emb: L2-normalized speaker embedding (B, 256).
            pitch_shift_semitones: Pitch shift in semitones (B, 1).
            noise_seed: Standard normal noise for Flow Matching (B, 80, T_frames).
            stft_cache_in: (B, 1, 720)
            content_cache_in: (3, B, 256, 6)
            f0_cache_in: (2, B, 128, 6)
            flow_cache_in: (num_flow_steps, 6, B, 256, 6)
            vocoder_cache_in: (4, B, 256, 6)
            ola_cache_in: (B, 720)

        Returns:
            audio_out: Converted waveform (B, T_samples)
            stft_cache_out, content_cache_out, f0_cache_out,
            flow_cache_out, vocoder_cache_out, ola_cache_out
        """
        log_mel, mag, stft_cache_out = self.mel_extractor(audio_chunk, stft_cache_in)
        content_feat, content_cache_out = self.content_encoder(log_mel, content_cache_in)
        f0_feat, _, _, f0_cache_out = self.pitch_conditioner(
            mag, log_mel, pitch_shift_semitones, f0_cache_in
        )

        cond_feat = content_feat + f0_feat
        converted_mel, flow_cache_out = self.flow_decoder.forward_euler_streaming(
            noise_seed,
            cond_feat,
            spk_emb,
            flow_cache_in,
            num_steps=self.num_flow_steps,
        )

        audio_out, vocoder_cache_out, ola_cache_out = self.vocoder(
            converted_mel, spk_emb, vocoder_cache_in, ola_cache_in
        )
        return (
            audio_out,
            stft_cache_out,
            content_cache_out,
            f0_cache_out,
            flow_cache_out,
            vocoder_cache_out,
            ola_cache_out,
        )


class StreamingFlowMatchingVC(nn.Module):
    """End-to-end zero-shot causal Flow Matching VC pipeline."""

    def __init__(self, num_flow_steps: int = 2, dim: int = 256) -> None:
        super().__init__()
        self.speaker_encoder = SpeakerEncoder(spk_dim=dim)
        self.core = StreamingVCCore(num_flow_steps=num_flow_steps, dim=dim)

    def init_streaming_caches(
        self,
        batch_size: int = 1,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Create initial zero-filled streaming caches."""
        ce = self.core.content_encoder
        pc = self.core.pitch_conditioner
        fd = self.core.flow_decoder
        voc = self.core.vocoder

        stft_cache = torch.zeros(batch_size, 1, STFT_CACHE_LEN, device=device, dtype=dtype)
        content_cache = torch.zeros(
            ce.num_layers, batch_size, ce.dim, ce.pad_len, device=device, dtype=dtype
        )
        f0_cache = torch.zeros(
            pc.num_layers, batch_size, pc.hidden_dim, pc.pad_len, device=device, dtype=dtype
        )
        flow_cache = torch.zeros(
            self.core.num_flow_steps,
            fd.num_layers,
            batch_size,
            fd.dim,
            fd.pad_len,
            device=device,
            dtype=dtype,
        )
        vocoder_cache = torch.zeros(
            voc.num_layers, batch_size, voc.dim, voc.pad_len, device=device, dtype=dtype
        )
        ola_cache = torch.zeros(batch_size, STFT_CACHE_LEN, device=device, dtype=dtype)
        return (
            stft_cache,
            content_cache,
            f0_cache,
            flow_cache,
            vocoder_cache,
            ola_cache,
        )

    def encode_speaker(self, ref_audio: torch.Tensor) -> torch.Tensor:
        """Encode reference audio waveform into speaker embedding."""
        return self.speaker_encoder(ref_audio)

    def compute_training_losses(
        self,
        source_audio: torch.Tensor,
        target_audio: torch.Tensor,
        ref_audio: torch.Tensor,
        pitch_shift_semitones: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute Flow Matching and Consistency Distillation losses for training."""
        batch_size = source_audio.shape[0]
        device = source_audio.device
        dtype = source_audio.dtype

        spk_emb = self.speaker_encoder(ref_audio)

        # Extract source conditioning
        stft_zero = torch.zeros(batch_size, 1, STFT_CACHE_LEN, device=device, dtype=dtype)
        src_mel, src_mag, _ = self.core.mel_extractor(source_audio, stft_zero)

        ce = self.core.content_encoder
        pc = self.core.pitch_conditioner
        ce_cache = torch.zeros(ce.num_layers, batch_size, ce.dim, ce.pad_len, device=device, dtype=dtype)
        pc_cache = torch.zeros(pc.num_layers, batch_size, pc.hidden_dim, pc.pad_len, device=device, dtype=dtype)

        content_feat, _ = ce(src_mel, ce_cache)
        f0_feat, _, _, _ = pc(src_mag, src_mel, pitch_shift_semitones, pc_cache)
        cond_feat = content_feat + f0_feat

        # Target mel ground truth
        tgt_mel, _, _ = self.core.mel_extractor(target_audio, stft_zero)

        rf_loss = self.core.flow_decoder.compute_rectified_flow_loss(tgt_mel, cond_feat, spk_emb)
        cm_loss = self.core.flow_decoder.compute_consistency_loss(tgt_mel, cond_feat, spk_emb)
        total_loss = rf_loss + 0.5 * cm_loss
        return {
            "total_loss": total_loss,
            "rf_loss": rf_loss,
            "consistency_loss": cm_loss,
        }

    def get_model_stats(self) -> Dict[str, float | int]:
        """Return parameter counts and model size estimates in MB."""
        spk_params = sum(p.numel() for p in self.speaker_encoder.parameters())
        core_params = sum(p.numel() for p in self.core.parameters())
        total_params = spk_params + core_params
        return {
            "speaker_encoder_params": spk_params,
            "streaming_core_params": core_params,
            "total_params": total_params,
            "fp32_size_mb": round(total_params * 4 / (1024 * 1024), 2),
            "fp16_size_mb": round(total_params * 2 / (1024 * 1024), 2),
        }
