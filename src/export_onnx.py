"""Export Causal Flow Matching VC models to ONNX (opset 17) for WebGPU and ORT."""

import os
import sys
from pathlib import Path

import onnx
import torch

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.audio_dsp import HOP_LENGTH, N_MELS, SAMPLE_RATE
from src.models.streaming_vc_pipeline import StreamingFlowMatchingVC


def export_models(
    output_dir: Path | str = PROJECT_ROOT / "models",
    chunk_frames: int = 4,
    ref_seconds: float = 3.0,
    num_flow_steps: int = 2,
    opset_version: int = 17,
) -> dict[str, Path]:
    """Export speaker_encoder.onnx and streaming_vc_core.onnx."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pipeline = StreamingFlowMatchingVC(num_flow_steps=num_flow_steps)
    pipeline.eval()

    stats = pipeline.get_model_stats()
    print(f"Model stats: {stats}")

    # 1. Export SpeakerEncoder
    ref_samples = int(ref_seconds * SAMPLE_RATE)
    dummy_ref_audio = torch.randn(1, ref_samples, dtype=torch.float32)
    spk_onnx_path = out_path / "speaker_encoder.onnx"

    with torch.no_grad():
        torch.onnx.export(
            pipeline.speaker_encoder,
            (dummy_ref_audio,),
            str(spk_onnx_path),
            opset_version=opset_version,
            input_names=["ref_audio"],
            output_names=["spk_emb"],
            dynamic_axes={
                "ref_audio": {0: "batch", 1: "num_samples"},
                "spk_emb": {0: "batch"},
            },
            do_constant_folding=True,
        )
    onnx.checker.check_model(str(spk_onnx_path))

    # 2. Export StreamingVCCore (40ms chunk = 4 frames = 960 samples)
    chunk_samples = chunk_frames * HOP_LENGTH
    dummy_audio_chunk = torch.randn(1, chunk_samples, dtype=torch.float32)
    dummy_spk_emb = torch.randn(1, 256, dtype=torch.float32)
    dummy_pitch_shift = torch.zeros(1, 1, dtype=torch.float32)
    dummy_noise_seed = torch.randn(1, N_MELS, chunk_frames, dtype=torch.float32)
    caches = pipeline.init_streaming_caches(batch_size=1, device="cpu", dtype=torch.float32)

    core_inputs = (
        dummy_audio_chunk,
        dummy_spk_emb,
        dummy_pitch_shift,
        dummy_noise_seed,
        *caches,
    )
    core_input_names = [
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
    ]
    core_output_names = [
        "audio_out",
        "stft_cache_out",
        "content_cache_out",
        "f0_cache_out",
        "flow_cache_out",
        "vocoder_cache_out",
        "ola_cache_out",
    ]
    core_onnx_path = out_path / "streaming_vc_core.onnx"

    with torch.no_grad():
        torch.onnx.export(
            pipeline.core,
            core_inputs,
            str(core_onnx_path),
            opset_version=opset_version,
            input_names=core_input_names,
            output_names=core_output_names,
            do_constant_folding=True,
        )
    onnx.checker.check_model(str(core_onnx_path))

    spk_mb = spk_onnx_path.stat().st_size / (1024 * 1024)
    core_mb = core_onnx_path.stat().st_size / (1024 * 1024)
    print(f"Exported {spk_onnx_path.name}: {spk_mb:.2f} MB")
    print(f"Exported {core_onnx_path.name}: {core_mb:.2f} MB")
    print(f"Total ONNX size: {spk_mb + core_mb:.2f} MB")

    return {
        "speaker_encoder": spk_onnx_path,
        "streaming_vc_core": core_onnx_path,
    }


if __name__ == "__main__":
    export_models()
