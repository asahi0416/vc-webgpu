"""Automatic Hugging Face Japanese multi-speaker dataset preparation with rollback."""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from src.models.audio_dsp import SAMPLE_RATE

# Avoid Windows symlink privilege warnings from huggingface_hub cache.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

DEFAULT_HF_REPO = "litagin/jvnv_corpus_v1_no_nv"
DEFAULT_SPEAKERS: Tuple[str, ...] = ("F1", "F2", "M1", "M2")


class TransactionalBackup:
    """Context manager that snapshots target files/directories and rolls back on failure."""

    def __init__(self, targets: Iterable[str | Path]) -> None:
        unique: List[Path] = []
        seen: set[Path] = set()
        for raw in targets:
            resolved = Path(raw).resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique.append(resolved)
        # Snapshot & restore shallowest paths first so nested targets stay consistent.
        self.targets: List[Path] = sorted(unique, key=lambda p: len(p.parts))
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self._records: List[Tuple[Path, bool, bool, Path | None]] = []

    def __enter__(self) -> "TransactionalBackup":
        self._tmpdir = tempfile.TemporaryDirectory(prefix="vc_backup_")
        backup_root = Path(self._tmpdir.name)
        self._records.clear()

        for idx, target in enumerate(self.targets):
            if target.exists():
                is_dir = target.is_dir()
                snap = backup_root / f"item_{idx}_{target.name}"
                if is_dir:
                    shutil.copytree(target, snap)
                else:
                    shutil.copy2(target, snap)
                self._records.append((target, True, is_dir, snap))
            else:
                self._records.append((target, False, False, None))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        try:
            if exc_type is not None:
                for target, existed, is_dir, snap in self._records:
                    if target.is_dir() and not target.is_symlink():
                        shutil.rmtree(target)
                    elif target.exists() or target.is_symlink():
                        target.unlink()

                    if existed and snap is not None and snap.exists():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if is_dir:
                            shutil.copytree(snap, target)
                        else:
                            shutil.copy2(snap, target)
        finally:
            if self._tmpdir is not None:
                self._tmpdir.cleanup()
                self._tmpdir = None
        return False


def _trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    top_db: float = 40.0,
    pad_sec: float = 0.05,
) -> np.ndarray:
    """Trim leading and trailing silence relative to signal peak."""
    peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
    if peak < 1e-6:
        return audio

    threshold = max(1e-5, peak * (10.0 ** (-top_db / 20.0)))
    active = np.flatnonzero(np.abs(audio) > threshold)
    if active.size == 0:
        return audio

    pad = int(pad_sec * sample_rate)
    start = max(0, int(active[0]) - pad)
    end = min(len(audio), int(active[-1]) + 1 + pad)
    return audio[start:end]


def _preprocess_waveform(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int = SAMPLE_RATE,
    min_duration_sec: float = 0.5,
) -> np.ndarray:
    """Convert to mono float32, resample to target_sr, trim silence, peak normalize to 0.9, and validate."""
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=-1).astype(np.float32)

    if arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError("Invalid waveform: empty or non-finite values detected")

    if orig_sr <= 0:
        raise ValueError(f"Invalid source sample rate: {orig_sr}")

    if orig_sr != target_sr:
        gcd = math.gcd(target_sr, orig_sr)
        arr = resample_poly(arr, target_sr // gcd, orig_sr // gcd).astype(np.float32)
        if not np.isfinite(arr).all():
            raise ValueError("Non-finite values produced after resampling")

    min_samples = int(min_duration_sec * target_sr)
    trimmed = _trim_silence(arr, target_sr)
    if len(trimmed) >= min_samples:
        arr = trimmed

    if len(arr) < min_samples:
        raise ValueError(
            f"Audio too short after preprocessing ({len(arr)} samples < {min_samples})"
        )

    peak = float(np.max(np.abs(arr)))
    if peak < 1e-6:
        raise ValueError("Audio contains only silence")

    normalized = ((arr / peak) * 0.9).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("Non-finite values after peak normalization")
    return normalized


def _is_dataset_ready(
    output_dir: Path,
    speakers: Tuple[str, ...],
    samples_per_speaker: int,
    target_sr: int,
    min_duration_sec: float,
) -> bool:
    """Check whether preprocessed speaker WAV directories already satisfy sample counts and rate."""
    if not output_dir.exists() or samples_per_speaker <= 0:
        return False

    for spk in speakers:
        spk_dir = output_dir / spk
        if not spk_dir.is_dir():
            return False
        wavs = sorted(spk_dir.glob("*.wav"))
        if len(wavs) < samples_per_speaker:
            return False
        for wav_path in wavs[:samples_per_speaker]:
            try:
                info = sf.info(wav_path)
            except Exception:
                return False
            if info.samplerate != target_sr or info.duration < min_duration_sec:
                return False
    return True


def fetch_and_prepare_hf_dataset(
    output_dir: str | Path = "data/jvnv",
    repo_id: str = DEFAULT_HF_REPO,
    speakers: Tuple[str, ...] = DEFAULT_SPEAKERS,
    samples_per_speaker: int = 6,
    target_sr: int = SAMPLE_RATE,
    min_duration_sec: float = 0.5,
) -> Path:
    """Download public Japanese speech WAVs from Hugging Face Hub and preprocess into 24kHz mono."""
    if samples_per_speaker <= 0:
        raise ValueError(f"samples_per_speaker must be positive, got {samples_per_speaker}")

    out_dir = Path(output_dir)
    if _is_dataset_ready(out_dir, speakers, samples_per_speaker, target_sr, min_duration_sec):
        print(
            f"[HFDataset] Reusing prepared dataset at {out_dir} "
            f"({len(speakers)} speakers x >= {samples_per_speaker} WAVs @ {target_sr}Hz)"
        )
        return out_dir

    from huggingface_hub import HfApi, hf_hub_download

    print(f"[HFDataset] Listing files from Hugging Face dataset '{repo_id}'...")
    repo_files = HfApi().list_repo_files(repo_id=repo_id, repo_type="dataset")

    speaker_files: Dict[str, List[str]] = {spk: [] for spk in speakers}
    for rel_path in sorted(repo_files):
        if not rel_path.lower().endswith(".wav"):
            continue
        for spk in speakers:
            if rel_path.startswith(f"{spk}/"):
                speaker_files[spk].append(rel_path)
                break

    for spk in speakers:
        if len(speaker_files[spk]) < samples_per_speaker:
            raise RuntimeError(
                f"Speaker '{spk}' in '{repo_id}' has only {len(speaker_files[spk])} WAV files "
                f"(requested {samples_per_speaker})"
            )

    total_written = 0
    with tempfile.TemporaryDirectory(prefix="vc_hf_stage_") as tmpdir:
        staging_dir = Path(tmpdir)
        for spk in speakers:
            spk_stage = staging_dir / spk
            spk_stage.mkdir(parents=True, exist_ok=True)
            selected = speaker_files[spk][:samples_per_speaker]

            for rel_path in selected:
                cached_wav = hf_hub_download(
                    repo_id=repo_id,
                    filename=rel_path,
                    repo_type="dataset",
                )
                audio, orig_sr = sf.read(cached_wav, dtype="float32", always_2d=False)
                processed = _preprocess_waveform(
                    audio,
                    orig_sr=orig_sr,
                    target_sr=target_sr,
                    min_duration_sec=min_duration_sec,
                )
                dest_path = spk_stage / Path(rel_path).name
                sf.write(dest_path, processed, target_sr, subtype="PCM_16")
                total_written += 1

        out_dir.mkdir(parents=True, exist_ok=True)
        for spk in speakers:
            target_spk = out_dir / spk
            if target_spk.exists():
                shutil.rmtree(target_spk)
            shutil.copytree(staging_dir / spk, target_spk)

    print(
        f"[HFDataset] Prepared {total_written} WAV files "
        f"({len(speakers)} speakers x {samples_per_speaker}, {target_sr}Hz mono) -> {out_dir}"
    )
    return out_dir
