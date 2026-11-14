"""Tests for Causal Flow Matching VC equivalence, training gradients, and ONNX export."""

import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.export_onnx import export_models
from src.models.audio_dsp import CausalISTFT, CausalMelSpectrogram, HOP_LENGTH, N_FFT, N_MELS, STFT_CACHE_LEN
from src.models.streaming_vc_pipeline import StreamingFlowMatchingVC


class TestCausalFlowMatchingVC(unittest.TestCase):
    """Comprehensive verification suite for Causal Flow Matching VC."""

    def test_causal_stft_istft_chunk_equivalence(self) -> None:
        """Verify CausalMelSpectrogram + CausalISTFT matches between full and chunked runs."""
        torch.manual_seed(42)
        mel_mod = CausalMelSpectrogram().eval()
        istft_mod = CausalISTFT().eval()

        n_chunks = 5
        chunk_samples = 4 * HOP_LENGTH  # 960 samples (40ms)
        total_samples = n_chunks * chunk_samples
        audio = torch.randn(1, total_samples, dtype=torch.float32)

        with torch.no_grad():
            # Full-sequence pass
            stft_c_full = torch.zeros(1, 1, STFT_CACHE_LEN)
            ola_c_full = torch.zeros(1, STFT_CACHE_LEN)
            mel_full, mag_full, stft_c_full_out = mel_mod(audio, stft_c_full)

            # Use real/imag phase from dummy spec via sin/cos
            phase_angle = torch.linspace(0.0, 3.14, mag_full.shape[1]).view(1, -1, 1).expand_as(mag_full)
            log_mag_full = torch.log(torch.clamp(mag_full, min=1e-5))
            wav_full, ola_c_full_out = istft_mod(
                log_mag_full,
                torch.sin(phase_angle),
                torch.cos(phase_angle),
                ola_c_full,
            )

            # Chunk-by-chunk pass
            stft_c = torch.zeros(1, 1, STFT_CACHE_LEN)
            ola_c = torch.zeros(1, STFT_CACHE_LEN)
            mel_chunks = []
            wav_chunks = []
            for idx in range(n_chunks):
                chunk = audio[:, idx * chunk_samples : (idx + 1) * chunk_samples]
                mel_c, mag_c, stft_c = mel_mod(chunk, stft_c)
                p_c = torch.linspace(0.0, 3.14, mag_c.shape[1]).view(1, -1, 1).expand_as(mag_c)
                wav_c, ola_c = istft_mod(
                    torch.log(torch.clamp(mag_c, min=1e-5)),
                    torch.sin(p_c),
                    torch.cos(p_c),
                    ola_c,
                )
                mel_chunks.append(mel_c)
                wav_chunks.append(wav_c)

            mel_cat = torch.cat(mel_chunks, dim=-1)
            wav_cat = torch.cat(wav_chunks, dim=-1)

        mel_err = (mel_full - mel_cat).abs().max().item()
        wav_err = (wav_full - wav_cat).abs().max().item()
        self.assertLess(mel_err, 1e-5, f"Mel mismatch: {mel_err}")
        self.assertLess(wav_err, 1e-5, f"Waveform mismatch: {wav_err}")
        self.assertLess((stft_c_full_out - stft_c).abs().max().item(), 1e-6)
        self.assertLess((ola_c_full_out - ola_c).abs().max().item(), 1e-6)

    def test_pipeline_causal_equivalence(self) -> None:
        """Full 4800-sample batch run must match 5x 960-sample streaming chunk runs."""
        torch.manual_seed(1234)
        pipeline = StreamingFlowMatchingVC(num_flow_steps=2).eval()

        n_chunks = 5
        chunk_frames = 4
        chunk_samples = chunk_frames * HOP_LENGTH  # 960 samples = 40ms
        total_samples = n_chunks * chunk_samples  # 4800 samples = 200ms

        full_audio = torch.randn(1, total_samples, dtype=torch.float32)
        ref_audio = torch.randn(1, 24000 * 3, dtype=torch.float32)
        pitch_shift = torch.tensor([[5.0]], dtype=torch.float32)
        full_noise = torch.randn(1, N_MELS, n_chunks * chunk_frames, dtype=torch.float32)

        with torch.no_grad():
            spk_emb = pipeline.encode_speaker(ref_audio)

            # 1. Full sequence pass
            init_caches_full = pipeline.init_streaming_caches(batch_size=1)
            full_out, *full_caches_out = pipeline.core(
                full_audio,
                spk_emb,
                pitch_shift,
                full_noise,
                *init_caches_full,
            )

            # 2. Sequential 40ms chunk pass
            caches = pipeline.init_streaming_caches(batch_size=1)
            chunk_outputs = []
            for i in range(n_chunks):
                audio_c = full_audio[:, i * chunk_samples : (i + 1) * chunk_samples]
                noise_c = full_noise[:, :, i * chunk_frames : (i + 1) * chunk_frames]
                out_c, *caches = pipeline.core(
                    audio_c,
                    spk_emb,
                    pitch_shift,
                    noise_c,
                    *caches,
                )
                chunk_outputs.append(out_c)
            streamed_out = torch.cat(chunk_outputs, dim=-1)

        max_err = (full_out - streamed_out).abs().max().item()
        print(f"[Causal Equivalence] Max absolute error over {total_samples} samples: {max_err:.3e}")
        self.assertLess(max_err, 1e-4, f"Streaming equivalence violated: {max_err}")

        for idx, (c_full, c_stream) in enumerate(zip(full_caches_out, caches)):
            c_err = (c_full - c_stream).abs().max().item()
            self.assertLess(c_err, 1e-4, f"Cache[{idx}] mismatch: {c_err}")

    def test_training_losses_and_gradients(self) -> None:
        """Verify Flow Matching loss, Consistency loss, and Vocoder backward gradients."""
        torch.manual_seed(7)
        pipeline = StreamingFlowMatchingVC(num_flow_steps=2).train()

        source_audio = torch.randn(2, 4800, dtype=torch.float32)
        target_audio = torch.randn(2, 4800, dtype=torch.float32)
        ref_audio = torch.randn(2, 24000, dtype=torch.float32)
        pitch_shift = torch.tensor([[0.0], [12.0]], dtype=torch.float32)

        losses = pipeline.compute_training_losses(
            source_audio, target_audio, ref_audio, pitch_shift
        )
        total_loss = losses["total_loss"]
        self.assertTrue(torch.isfinite(total_loss))
        total_loss.backward()

        # Check gradients exist and are finite across key trainable components
        for name in ["speaker_encoder", "core.content_encoder", "core.pitch_conditioner", "core.flow_decoder"]:
            submod = pipeline.get_submodule(name)
            grads = [p.grad for p in submod.parameters() if p.grad is not None]
            self.assertGreater(len(grads), 0, f"No gradients in {name}")
            for g in grads:
                self.assertTrue(torch.isfinite(g).all(), f"Non-finite gradient in {name}")

        # Also verify Vocoder loss & backward
        pipeline.zero_grad()
        spk_emb = pipeline.encode_speaker(ref_audio)
        dummy_mel = torch.randn(2, N_MELS, 20)
        voc_cache = torch.zeros(pipeline.core.vocoder.num_layers, 2, 256, 6)
        ola_cache = torch.zeros(2, STFT_CACHE_LEN)
        pred_wav, _, _ = pipeline.core.vocoder(dummy_mel, spk_emb, voc_cache, ola_cache)
        voc_loss = torch.nn.functional.l1_loss(pred_wav, target_audio)
        voc_loss.backward()
        voc_grads = [p.grad for p in pipeline.core.vocoder.parameters() if p.grad is not None]
        self.assertGreater(len(voc_grads), 0)
        for g in voc_grads:
            self.assertTrue(torch.isfinite(g).all())

    def test_onnx_export_parity_and_latency(self) -> None:
        """Export ONNX models, verify parity with PyTorch, size < 100MB, and RTF < 1.0."""
        torch.manual_seed(99)
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = export_models(output_dir=tmpdir, chunk_frames=4, ref_seconds=3.0, num_flow_steps=2)
            spk_onnx = paths["speaker_encoder"]
            core_onnx = paths["streaming_vc_core"]

            total_mb = (spk_onnx.stat().st_size + core_onnx.stat().st_size) / (1024 * 1024)
            self.assertLess(total_mb, 100.0, f"ONNX size too large: {total_mb:.2f} MB")

            # Recreate exact PyTorch reference model with same weights?
            # Since export_models instantiates inside, let's verify ORT with random inputs vs exported model
            # Better: export directly from an instantiated pipeline or reload?
            # Let's check ORT runs cleanly on speaker_encoder and streaming_vc_core, and also verify parity using a fresh export helper.
            so = ort.SessionOptions()
            so.intra_op_num_threads = 4
            spk_sess = ort.InferenceSession(str(spk_onnx), sess_options=so, providers=["CPUExecutionProvider"])
            core_sess = ort.InferenceSession(str(core_onnx), sess_options=so, providers=["CPUExecutionProvider"])

            # Test dynamic length reference audio (2.5s = 60000 samples)
            ref_np = np.random.randn(1, 60000).astype(np.float32)
            (spk_emb_np,) = spk_sess.run(["spk_emb"], {"ref_audio": ref_np})
            self.assertEqual(spk_emb_np.shape, (1, 256))
            self.assertAlmostEqual(float(np.linalg.norm(spk_emb_np)), 1.0, places=4)

            # Test streaming core (40ms chunk = 960 samples)
            chunk_np = np.random.randn(1, 960).astype(np.float32)
            pitch_np = np.array([[4.0]], dtype=np.float32)
            noise_np = np.random.randn(1, N_MELS, 4).astype(np.float32)
            feed = {
                "audio_chunk": chunk_np,
                "spk_emb": spk_emb_np,
                "pitch_shift_semitones": pitch_np,
                "noise_seed": noise_np,
                "stft_cache_in": np.zeros((1, 1, STFT_CACHE_LEN), dtype=np.float32),
                "content_cache_in": np.zeros((3, 1, 256, 6), dtype=np.float32),
                "f0_cache_in": np.zeros((2, 1, 128, 6), dtype=np.float32),
                "flow_cache_in": np.zeros((2, 6, 1, 256, 6), dtype=np.float32),
                "vocoder_cache_in": np.zeros((4, 1, 256, 6), dtype=np.float32),
                "ola_cache_in": np.zeros((1, STFT_CACHE_LEN), dtype=np.float32),
            }

            # Warmup
            for _ in range(3):
                outs = core_sess.run(None, feed)
            self.assertEqual(outs[0].shape, (1, 960))

            # Benchmark 30 chunks (1.2 seconds of audio)
            n_iters = 30
            t0 = time.perf_counter()
            for _ in range(n_iters):
                outs = core_sess.run(None, feed)
                # Feed updated caches back
                feed["stft_cache_in"] = outs[1]
                feed["content_cache_in"] = outs[2]
                feed["f0_cache_in"] = outs[3]
                feed["flow_cache_in"] = outs[4]
                feed["vocoder_cache_in"] = outs[5]
                feed["ola_cache_in"] = outs[6]
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            avg_chunk_ms = elapsed_ms / n_iters
            rtf = avg_chunk_ms / 40.0
            print(
                f"[ORT CPU Benchmark] Avg 40ms chunk inference: {avg_chunk_ms:.2f} ms | "
                f"RTF: {rtf:.3f} | Total ONNX size: {total_mb:.2f} MB"
            )
            self.assertLess(avg_chunk_ms, 40.0, f"CPU inference slower than real-time: {avg_chunk_ms:.2f} ms")

    def test_pytorch_onnx_exact_parity(self) -> None:
        """Verify PyTorch and ONNX Runtime produce numerically identical outputs."""
        torch.manual_seed(2026)
        pipeline = StreamingFlowMatchingVC(num_flow_steps=2).eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            spk_path = out_dir / "spk.onnx"
            core_path = out_dir / "core.onnx"

            ref_audio = torch.randn(1, 72000, dtype=torch.float32)
            audio_chunk = torch.randn(1, 960, dtype=torch.float32)
            spk_emb = pipeline.encode_speaker(ref_audio).detach()
            pitch_shift = torch.tensor([[-3.0]], dtype=torch.float32)
            noise_seed = torch.randn(1, N_MELS, 4, dtype=torch.float32)
            caches = pipeline.init_streaming_caches(batch_size=1)

            with torch.no_grad():
                pt_spk = pipeline.speaker_encoder(ref_audio).numpy()
                pt_outs = pipeline.core(audio_chunk, spk_emb, pitch_shift, noise_seed, *caches)
                pt_audio = pt_outs[0].numpy()

                torch.onnx.export(
                    pipeline.speaker_encoder,
                    (ref_audio,),
                    str(spk_path),
                    opset_version=17,
                    input_names=["ref_audio"],
                    output_names=["spk_emb"],
                    dynamic_axes={"ref_audio": {0: "batch", 1: "num_samples"}, "spk_emb": {0: "batch"}},
                )
                torch.onnx.export(
                    pipeline.core,
                    (audio_chunk, spk_emb, pitch_shift, noise_seed, *caches),
                    str(core_path),
                    opset_version=17,
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
                )

            spk_sess = ort.InferenceSession(str(spk_path), providers=["CPUExecutionProvider"])
            core_sess = ort.InferenceSession(str(core_path), providers=["CPUExecutionProvider"])

            (ort_spk,) = spk_sess.run(["spk_emb"], {"ref_audio": ref_audio.numpy()})
            spk_diff = float(np.max(np.abs(pt_spk - ort_spk)))
            self.assertLess(spk_diff, 1e-4, f"SpeakerEncoder ORT mismatch: {spk_diff}")

            ort_outs = core_sess.run(
                None,
                {
                    "audio_chunk": audio_chunk.numpy(),
                    "spk_emb": spk_emb.numpy(),
                    "pitch_shift_semitones": pitch_shift.numpy(),
                    "noise_seed": noise_seed.numpy(),
                    "stft_cache_in": caches[0].numpy(),
                    "content_cache_in": caches[1].numpy(),
                    "f0_cache_in": caches[2].numpy(),
                    "flow_cache_in": caches[3].numpy(),
                    "vocoder_cache_in": caches[4].numpy(),
                    "ola_cache_in": caches[5].numpy(),
                },
            )
            audio_diff = float(np.max(np.abs(pt_audio - ort_outs[0])))
            print(f"[ORT Parity] Speaker diff: {spk_diff:.3e} | Audio chunk diff: {audio_diff:.3e}")
            self.assertLess(audio_diff, 1e-4, f"StreamingVCCore ORT mismatch: {audio_diff}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
