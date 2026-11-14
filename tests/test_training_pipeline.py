"""Unit tests for training losses, datasets, trainer, and checkpointing."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.audio_dsp import SAMPLE_RATE
from src.models.streaming_vc_pipeline import StreamingFlowMatchingVC
from src.training.dataset import SyntheticVocalDataset, WAVVoiceDataset
from src.training.losses import extract_f0_autocorr
from src.training.trainer import VCTrainer


class TestTrainingPipeline(unittest.TestCase):
    """Verify F0 ground-truth extraction, datasets, training convergence, and checkpointing."""

    def test_f0_autocorr_accuracy(self) -> None:
        """extract_f0_autocorr must detect 220 Hz harmonics accurately and silent frames as unvoiced."""
        duration_sec = 0.2  # 4800 samples = 20 frames
        n_samples = int(SAMPLE_RATE * duration_sec)
        t = torch.arange(n_samples, dtype=torch.float32) / float(SAMPLE_RATE)
        f0_target = 220.0
        voiced_wave = (
            0.5 * torch.sin(2.0 * torch.pi * f0_target * t)
            + 0.25 * torch.sin(2.0 * torch.pi * 2.0 * f0_target * t)
            + 0.15 * torch.sin(2.0 * torch.pi * 3.0 * f0_target * t)
        ).unsqueeze(0)
        silent_wave = torch.zeros(1, n_samples, dtype=torch.float32)

        gt_f0, gt_uv = extract_f0_autocorr(voiced_wave)
        self.assertEqual(gt_f0.shape, (1, 1, 20))
        self.assertEqual(gt_uv.shape, (1, 1, 20))

        # Steady frames (after 3-frame STFT causal warmup) must be voiced around 220 Hz
        steady_f0 = gt_f0[0, 0, 4:].mean().item()
        steady_uv = gt_uv[0, 0, 4:].mean().item()
        print(f"[F0 Test] Estimated steady F0={steady_f0:.2f} Hz, UV={steady_uv:.2f}")
        self.assertGreater(steady_uv, 0.8)
        self.assertLess(abs(steady_f0 - f0_target), 10.0)

        _, silent_uv = extract_f0_autocorr(silent_wave)
        self.assertLess(silent_uv.mean().item(), 0.1)

    def test_wav_and_synthetic_datasets(self) -> None:
        """Verify WAVVoiceDataset resampling and SyntheticVocalDataset shapes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "spk_a").mkdir()
            (root / "spk_b").mkdir()

            # Write WAVs at 16kHz and 48kHz to exercise resampling
            t16 = np.linspace(0, 0.5, 8000, endpoint=False, dtype=np.float32)
            sf.write(root / "spk_a" / "u1.wav", 0.4 * np.sin(2 * np.pi * 150.0 * t16), 16000)

            t48 = np.linspace(0, 0.5, 24000, endpoint=False, dtype=np.float32)
            sf.write(root / "spk_b" / "u1.wav", 0.4 * np.sin(2 * np.pi * 260.0 * t48), 48000)

            wav_ds = WAVVoiceDataset(root, segment_frames=16, ref_seconds=1.0)
            self.assertEqual(len(wav_ds), 2)
            item = wav_ds[0]
            self.assertEqual(item["audio"].shape, (16 * 240,))
            self.assertEqual(item["ref_audio"].shape, (SAMPLE_RATE,))
            self.assertTrue(torch.isfinite(item["audio"]).all())

        synth_ds = SyntheticVocalDataset(num_speakers=4, num_samples=12, segment_frames=16, ref_seconds=1.0)
        self.assertEqual(len(synth_ds), 12)
        s_item = synth_ds[1]
        self.assertEqual(s_item["audio"].shape, (16 * 240,))
        self.assertEqual(s_item["ref_audio"].shape, (SAMPLE_RATE,))
        self.assertEqual(int(s_item["speaker_id"].item()), 1)

    def test_trainer_loss_reduction_and_checkpoint(self) -> None:
        """Verify VCTrainer reduces total loss over steps and preserves state via checkpoint."""
        torch.manual_seed(42)
        synth_ds = SyntheticVocalDataset(num_speakers=4, num_samples=8, segment_frames=16, ref_seconds=1.0)
        loader = DataLoader(synth_ds, batch_size=4, shuffle=False)
        fixed_batch = next(iter(loader))

        model = StreamingFlowMatchingVC(num_flow_steps=2)
        trainer = VCTrainer(model=model, lr=1e-3, total_steps=20, device="cpu")

        initial_losses = []
        for _ in range(2):
            m = trainer.train_step(fixed_batch)
            initial_losses.append(m["total_loss"])
        loss_start = sum(initial_losses) / len(initial_losses)

        later_losses = []
        for _ in range(10):
            m = trainer.train_step(fixed_batch)
            later_losses.append(m["total_loss"])
        loss_end = sum(later_losses[-2:]) / 2.0

        print(f"[Trainer Test] Start loss={loss_start:.4f} -> End loss={loss_end:.4f}")
        self.assertLess(loss_end, loss_start, f"Loss did not decrease: {loss_start} -> {loss_end}")

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_ckpt.pt"
            trainer.save_checkpoint(ckpt_path, epoch=1, metrics=m)

            # Reload into a fresh model & trainer
            model2 = StreamingFlowMatchingVC(num_flow_steps=2)
            trainer2 = VCTrainer(model=model2, lr=1e-3, total_steps=20, device="cpu")
            trainer2.load_checkpoint(ckpt_path)

            self.assertEqual(trainer2.global_step, trainer.global_step)
            for p1, p2 in zip(model.parameters(), model2.parameters()):
                self.assertTrue(torch.allclose(p1, p2))

            onnx_paths = trainer2.export_onnx_models(output_dir=Path(tmpdir) / "onnx")
            self.assertTrue(onnx_paths["speaker_encoder"].exists())
            self.assertTrue(onnx_paths["streaming_vc_core"].exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
