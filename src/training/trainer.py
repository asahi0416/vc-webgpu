"""End-to-end trainer, checkpointing, and ONNX exporter for Causal Flow Matching VC."""

from pathlib import Path
from typing import Dict, Optional

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.audio_dsp import HOP_LENGTH, N_MELS, SAMPLE_RATE, STFT_CACHE_LEN
from src.models.streaming_vc_pipeline import StreamingFlowMatchingVC
from src.training.dataset import apply_mel_formant_perturbation
from src.training.losses import (
    MultiResolutionSTFTLoss,
    SpeakerContrastiveLoss,
    extract_f0_autocorr,
)


class VCTrainer:
    """Trainer coordinating Flow Matching, Consistency, F0, Speaker, and Vocoder losses."""

    def __init__(
        self,
        model: StreamingFlowMatchingVC,
        lr: float = 3e-4,
        total_steps: int = 1000,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.device = torch.device(device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, betas=(0.8, 0.99), weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, total_steps), eta_min=lr * 0.05
        )
        self.stft_loss_fn = MultiResolutionSTFTLoss().to(self.device)
        self.spk_loss_fn = SpeakerContrastiveLoss(margin=0.2).to(self.device)
        self.global_step = 0

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute one end-to-end training step and return scalar loss metrics."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        audio = batch["audio"].to(self.device)  # (B, T_samples)
        ref_audio = batch["ref_audio"].to(self.device)  # (B, ref_samples)
        speaker_ids = batch["speaker_id"].to(self.device)  # (B,)
        batch_size = audio.shape[0]

        # 1. Speaker embedding & contrastive loss
        spk_emb = self.model.speaker_encoder(ref_audio)
        spk_loss = self.spk_loss_fn(spk_emb, speaker_ids)

        # 2. Target acoustic features & causal ground-truth F0
        stft_zero = torch.zeros(batch_size, 1, STFT_CACHE_LEN, device=self.device)
        tgt_mel, tgt_mag, _ = self.model.core.mel_extractor(audio, stft_zero)
        gt_f0, gt_uv = extract_f0_autocorr(audio)
        ref_len = (min(ref_audio.shape[1], SAMPLE_RATE) // HOP_LENGTH) * HOP_LENGTH
        ref_f0, ref_uv = extract_f0_autocorr(ref_audio[:, :ref_len])

        # 3. Information perturbation on source mel -> Content & Pitch estimation
        perturbed_mel = apply_mel_formant_perturbation(tgt_mel.detach())
        ce = self.model.core.content_encoder
        pc = self.model.core.pitch_conditioner
        ce_cache = torch.zeros(ce.num_layers, batch_size, ce.dim, ce.pad_len, device=self.device)
        pc_cache = torch.zeros(pc.num_layers, batch_size, pc.hidden_dim, pc.pad_len, device=self.device)

        content_feat, _ = ce(perturbed_mel, ce_cache)
        pitch_shift = torch.zeros(batch_size, 1, device=self.device)
        f0_feat, pred_f0, pred_uv, _ = pc(
            tgt_mag, perturbed_mel, pitch_shift, pc_cache, spk_emb=spk_emb
        )

        # F0 L1 loss on voiced frames + Voiced/Unvoiced BCE + Register F0 supervision
        src_f0_hz = pc.last_src_f0_hz if pc.last_src_f0_hz is not None else pred_f0
        log_pred_f0 = torch.log(torch.clamp(src_f0_hz, min=20.0))
        log_gt_f0 = torch.log(torch.clamp(gt_f0, min=20.0))
        log_ref_f0 = torch.log(torch.clamp(ref_f0, min=20.0))
        voiced_mask = gt_uv > 0.5
        if voiced_mask.any():
            f0_l1 = F.l1_loss(log_pred_f0[voiced_mask], log_gt_f0[voiced_mask])
        else:
            f0_l1 = pred_f0.sum() * 0.0
        uv_bce = F.binary_cross_entropy(torch.clamp(pred_uv, 1e-4, 1.0 - 1e-4), gt_uv)

        base_terms = []
        src_base = pc.last_src_base_log_f0
        tgt_base = pc.last_tgt_base_log_f0
        for b in range(batch_size):
            vm_src = gt_uv[b] > 0.5
            if vm_src.any() and src_base is not None:
                utt_src_mean = log_gt_f0[b][vm_src].mean()
                base_terms.append(F.l1_loss(src_base[b], utt_src_mean.expand_as(src_base[b])))
            vm_ref = ref_uv[b] > 0.5
            if vm_ref.any() and tgt_base is not None:
                utt_ref_mean = log_ref_f0[b][vm_ref].mean()
                base_terms.append(F.l1_loss(tgt_base[b], utt_ref_mean.expand_as(tgt_base[b])))
        base_f0_loss = torch.stack(base_terms).mean() if base_terms else pred_f0.sum() * 0.0
        if src_base is not None and src_base.shape[-1] > 1:
            base_smooth_loss = (src_base[:, :, 1:] - src_base[:, :, :-1]).pow(2).mean()
        else:
            base_smooth_loss = pred_f0.sum() * 0.0
        f0_loss = f0_l1 + 0.5 * uv_bce + 2.0 * base_f0_loss + 12.0 * base_smooth_loss

        # 4. Rectified Flow Matching + Consistency + 2-step Euler reconstruction losses
        cond_feat = content_feat + f0_feat
        fd = self.model.core.flow_decoder
        vtl_shift = (
            pc.last_tgt_base_log_f0 - pc.last_src_base_log_f0
            if (pc.last_tgt_base_log_f0 is not None and pc.last_src_base_log_f0 is not None)
            else None
        )
        base_in_mel = 0.7 * tgt_mel.detach() + 0.3 * perturbed_mel
        base_mel = fd.compute_target_base_mel(base_in_mel, spk_emb, vtl_shift=vtl_shift)
        rf_loss = fd.compute_rectified_flow_loss(tgt_mel, cond_feat, spk_emb, base_mel=base_mel)
        cm_loss = fd.compute_consistency_loss(tgt_mel, cond_feat, spk_emb, base_mel=base_mel)

        noise = torch.randn_like(tgt_mel)
        flow_cache = torch.zeros(
            self.model.core.num_flow_steps,
            fd.num_layers,
            batch_size,
            fd.dim,
            fd.pad_len,
            device=self.device,
        )
        pred_mel, _ = fd.forward_euler_streaming(
            noise,
            cond_feat,
            spk_emb,
            flow_cache,
            num_steps=self.model.core.num_flow_steps,
            base_mel=base_mel,
        )
        mel_l1_loss = F.l1_loss(pred_mel, tgt_mel)

        # Evaluate mel speaker consistency without backprop into SpeakerEncoder parameters
        spk_params = list(self.model.speaker_encoder.parameters())
        prev_req_grad = [p.requires_grad for p in spk_params]
        try:
            for p in spk_params:
                p.requires_grad_(False)
            pred_spk_emb = self.model.speaker_encoder.encode_mel(pred_mel)
        finally:
            for p, req in zip(spk_params, prev_req_grad):
                p.requires_grad_(req)
        mel_spk_loss = (1.0 - (pred_spk_emb * spk_emb.detach()).sum(dim=-1)).mean()

        # Cross-speaker shuffled envelope consistency loss
        if batch_size > 1:
            perm_idx = torch.roll(torch.arange(batch_size, device=self.device), shifts=1)
            spk_shuf = spk_emb[perm_idx].detach()
            f0_feat_shuf, _, _, _ = pc(
                tgt_mag, perturbed_mel, pitch_shift, pc_cache, spk_emb=spk_shuf
            )
            vtl_shuf = (
                pc.last_tgt_base_log_f0 - pc.last_src_base_log_f0
                if (pc.last_tgt_base_log_f0 is not None and pc.last_src_base_log_f0 is not None)
                else None
            )
            base_mel_shuf = fd.compute_target_base_mel(
                perturbed_mel, spk_shuf, vtl_shift=vtl_shuf
            )
            pred_mel_shuf, _ = fd.forward_euler_streaming(
                noise,
                content_feat + f0_feat_shuf,
                spk_shuf,
                flow_cache,
                num_steps=self.model.core.num_flow_steps,
                base_mel=base_mel_shuf,
            )
            v_w = torch.sigmoid((tgt_mel[:, 2:30, :].mean(dim=1, keepdim=True) + 5.5) * 2.0)
            v_w = v_w / (v_w.sum(dim=-1, keepdim=True) + 1e-6)
            pred_utt_mean = (pred_mel_shuf * v_w).sum(dim=-1, keepdim=True)
            pred_env_shuf = torch.matmul(fd.mel_env_mat, pred_utt_mean)
            pred_shape_shuf = pred_env_shuf - pred_env_shuf.mean(dim=1, keepdim=True)
            tgt_shape_raw = (spk_shuf[:, :N_MELS] * 24.0).unsqueeze(-1)
            tgt_env_shuf = torch.matmul(fd.mel_env_mat, tgt_shape_raw)
            tgt_shape_shuf = tgt_env_shuf - tgt_env_shuf.mean(dim=1, keepdim=True)
            cross_env_loss = (
                (pred_shape_shuf - tgt_shape_shuf).abs().mean()
                + 0.8 * F.l1_loss(pred_mel_shuf, base_mel_shuf.detach())
            )
        else:
            cross_env_loss = pred_mel.sum() * 0.0

        # 5. Causal NSF Vocos Vocoder loss (aligned by STFT_CACHE_LEN=720 causal synthesis delay)
        voc = self.model.core.vocoder
        voc_cache = torch.zeros(voc.num_layers, batch_size, voc.dim, voc.pad_len, device=self.device)
        ola_cache = torch.zeros(batch_size, STFT_CACHE_LEN, device=self.device)

        voc_in_mel = 0.7 * tgt_mel.detach() + 0.3 * pred_mel.detach()
        pred_audio, _, _ = voc(
            voc_in_mel,
            spk_emb,
            voc_cache,
            ola_cache,
            f0_hz=gt_f0,
            uv=gt_uv,
            noise_seed=noise,
        )

        delay = STFT_CACHE_LEN
        if audio.shape[1] > delay:
            aligned_pred = pred_audio[:, delay:]
            aligned_tgt = audio[:, :-delay]
        else:
            aligned_pred = pred_audio
            aligned_tgt = audio

        stft_loss = self.stft_loss_fn(aligned_pred, aligned_tgt)
        aligned_frames = aligned_pred.shape[1] // HOP_LENGTH
        if aligned_frames > 0:
            trimmed_len = aligned_frames * HOP_LENGTH
            pred_out_mel, _, _ = self.model.core.mel_extractor(
                aligned_pred[:, :trimmed_len], stft_zero
            )
            tgt_out_mel, _, _ = self.model.core.mel_extractor(
                aligned_tgt[:, :trimmed_len], stft_zero
            )
            voc_mel_loss = F.l1_loss(pred_out_mel, tgt_out_mel)
        else:
            voc_mel_loss = aligned_pred.sum() * 0.0
        vocoder_loss = stft_loss + 0.5 * voc_mel_loss

        total_loss = (
            rf_loss
            + 0.5 * cm_loss
            + 0.5 * mel_l1_loss
            + 1.0 * f0_loss
            + 0.5 * spk_loss
            + 0.5 * mel_spk_loss
            + 0.5 * cross_env_loss
            + vocoder_loss
        )

        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1

        return {
            "total_loss": float(total_loss.item()),
            "rf_loss": float(rf_loss.item()),
            "cm_loss": float(cm_loss.item()),
            "mel_l1": float(mel_l1_loss.item()),
            "f0_loss": float(f0_loss.item()),
            "spk_loss": float(spk_loss.item()),
            "vocoder_loss": float(vocoder_loss.item()),
        }

    def save_checkpoint(
        self,
        path: str | Path,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Path:
        ckpt_path = Path(path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "metrics": metrics or {},
        }
        torch.save(payload, str(ckpt_path))
        return ckpt_path

    def load_checkpoint(self, path: str | Path) -> Dict[str, object]:
        payload = torch.load(str(path), map_location=self.device)
        self.model.load_state_dict(payload["model_state"])
        if "optimizer_state" in payload:
            self.optimizer.load_state_dict(payload["optimizer_state"])
        if "scheduler_state" in payload:
            self.scheduler.load_state_dict(payload["scheduler_state"])
        self.global_step = int(payload.get("global_step", 0))
        return payload

    def export_onnx_models(
        self,
        output_dir: str | Path,
        chunk_frames: int = 4,
        ref_seconds: float = 3.0,
        opset_version: int = 17,
    ) -> Dict[str, Path]:
        """Export current trained weights directly to speaker_encoder.onnx and streaming_vc_core.onnx."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        was_training = self.model.training
        self.model.eval()
        orig_device = self.device
        # Export on CPU for clean WebGPU/ORT compatibility
        model_cpu = self.model.to("cpu")

        ref_samples = int(ref_seconds * SAMPLE_RATE)
        dummy_ref = torch.randn(1, ref_samples, dtype=torch.float32)
        spk_onnx = out_path / "speaker_encoder.onnx"

        with torch.no_grad():
            torch.onnx.export(
                model_cpu.speaker_encoder,
                (dummy_ref,),
                str(spk_onnx),
                opset_version=opset_version,
                input_names=["ref_audio"],
                output_names=["spk_emb"],
                dynamic_axes={"ref_audio": {0: "batch", 1: "num_samples"}, "spk_emb": {0: "batch"}},
                do_constant_folding=True,
            )
            onnx.checker.check_model(str(spk_onnx))

            chunk_samples = chunk_frames * HOP_LENGTH
            caches = model_cpu.init_streaming_caches(batch_size=1, device="cpu", dtype=torch.float32)
            core_inputs = (
                torch.randn(1, chunk_samples, dtype=torch.float32),
                torch.randn(1, 256, dtype=torch.float32),
                torch.zeros(1, 1, dtype=torch.float32),
                torch.randn(1, N_MELS, chunk_frames, dtype=torch.float32),
                *caches,
            )
            core_onnx = out_path / "streaming_vc_core.onnx"
            torch.onnx.export(
                model_cpu.core,
                core_inputs,
                str(core_onnx),
                opset_version=opset_version,
                input_names=[
                    "audio_chunk",
                    "spk_emb",
                    "pitch_shift_semitones",
                    "noise_seed",
                    "stft_cache_in",
                    "content_cache_in",
                    "f0_cache_in",
                    "flow_cache_in",
                    "vocoder_cache_in",
                    "ola_cache_in",
                ],
                output_names=[
                    "audio_out",
                    "stft_cache_out",
                    "content_cache_out",
                    "f0_cache_out",
                    "flow_cache_out",
                    "vocoder_cache_out",
                    "ola_cache_out",
                ],
                do_constant_folding=True,
            )
            onnx.checker.check_model(str(core_onnx))

        self.model.to(orig_device)
        if was_training:
            self.model.train()

        return {"speaker_encoder": spk_onnx, "streaming_vc_core": core_onnx}
