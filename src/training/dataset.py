"""Datasets and information-perturbation utilities for self-supervised Flow Matching VC."""

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
from scipy import signal
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.models.audio_dsp import HOP_LENGTH, SAMPLE_RATE


def apply_mel_formant_perturbation(
    log_mel: torch.Tensor,
    warp_range: Tuple[float, float] = (0.86, 1.16),
    tilt_strength: float = 0.5,
) -> torch.Tensor:
    """Perturb source log-mel along the frequency axis to strip speaker timbre.

    Forces CausalContentEncoder to extract speaker-agnostic linguistic content
    from non-parallel single-speaker training waveforms.
    """
    batch_size, n_mels, t_frames = log_mel.shape
    perturbed_list = []

    for b in range(batch_size):
        item = log_mel[b : b + 1]  # (1, n_mels, T)
        factor = float(torch.empty(1).uniform_(warp_range[0], warp_range[1]).item())
        warped_mels = max(16, int(round(n_mels * factor)))

        # Interpolate along frequency axis (treat T as batch and M as spatial 1D)
        item_t = item.permute(0, 2, 1).reshape(t_frames, 1, n_mels)  # (T, 1, M)
        stretched = F.interpolate(item_t, size=warped_mels, mode="linear", align_corners=False)

        if warped_mels >= n_mels:
            cropped = stretched[:, :, :n_mels]
        else:
            pad_top = n_mels - warped_mels
            cropped = F.pad(stretched, (0, pad_top), mode="replicate")

        warped = cropped.reshape(1, t_frames, n_mels).permute(0, 2, 1)  # (1, M, T)

        # Random spectral tilt across mel bins
        tilt_slope = float(torch.empty(1).uniform_(-tilt_strength, tilt_strength).item())
        tilt = torch.linspace(
            -tilt_slope, tilt_slope, n_mels, device=log_mel.device, dtype=log_mel.dtype
        ).view(1, n_mels, 1)
        perturbed_list.append(warped + tilt)

    return torch.cat(perturbed_list, dim=0)


def _load_and_resample(path: Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = np.mean(audio, axis=1)
    if sr != target_sr and len(mono) > 0:
        gcd = math.gcd(int(sr), int(target_sr))
        up = int(target_sr) // gcd
        down = int(sr) // gcd
        mono = signal.resample_poly(mono, up, down).astype(np.float32)
    peak = float(np.max(np.abs(mono))) if len(mono) > 0 else 0.0
    if peak > 1e-4:
        mono = mono / max(1.0, peak)
    return mono.astype(np.float32)


def _crop_or_loop(audio: np.ndarray, target_len: int, rng: np.random.Generator) -> np.ndarray:
    if len(audio) == 0:
        return np.zeros(target_len, dtype=np.float32)
    if len(audio) < target_len:
        reps = int(math.ceil(float(target_len) / float(len(audio))))
        audio = np.tile(audio, reps)
    if len(audio) == target_len:
        return audio.astype(np.float32)
    start = int(rng.integers(0, len(audio) - target_len + 1))
    return audio[start : start + target_len].astype(np.float32)


class WAVVoiceDataset(Dataset):
    """Multi-speaker WAV dataset supporting <data_dir>/<speaker>/*.wav or flat *.wav."""

    def __init__(
        self,
        data_dir: str | Path,
        segment_frames: int = 16,
        ref_seconds: float = 3.0,
        sample_rate: int = SAMPLE_RATE,
        seed: int = 42,
        ssl_cache_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.segment_frames = segment_frames
        self.segment_samples = segment_frames * HOP_LENGTH
        self.ref_samples = int(ref_seconds * sample_rate)
        self.sample_rate = sample_rate
        self.rng = np.random.default_rng(seed)
        self.ssl_cache_dir = Path(ssl_cache_dir) if ssl_cache_dir else None

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")

        exts = {".wav", ".flac", ".ogg"}
        self.speaker_files: Dict[int, List[Path]] = {}
        self.items: List[Tuple[int, Path]] = []

        subdirs = sorted([p for p in self.data_dir.iterdir() if p.is_dir()])
        if subdirs:
            spk_idx = 0
            for d in subdirs:
                files = sorted([f for f in d.rglob("*") if f.suffix.lower() in exts])
                if files:
                    self.speaker_files[spk_idx] = files
                    for f in files:
                        self.items.append((spk_idx, f))
                    spk_idx += 1
        else:
            files = sorted([f for f in self.data_dir.rglob("*") if f.suffix.lower() in exts])
            for idx, f in enumerate(files):
                self.speaker_files[idx] = [f]
                self.items.append((idx, f))

        if not self.items:
            raise ValueError(f"No audio files found under {self.data_dir}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        spk_id, audio_path = self.items[index]
        wav = _load_and_resample(audio_path, target_sr=self.sample_rate)
        if len(wav) == 0:
            wav = np.zeros(self.segment_samples, dtype=np.float32)
        if len(wav) < self.segment_samples:
            reps = int(math.ceil(float(self.segment_samples) / float(len(wav))))
            wav = np.tile(wav, reps)
        if len(wav) == self.segment_samples:
            start = 0
        else:
            start = int(self.rng.integers(0, len(wav) - self.segment_samples + 1))
        segment = wav[start : start + self.segment_samples].astype(np.float32)

        # Sample reference clip from another utterance of the same speaker when available
        candidates = self.speaker_files[spk_id]
        ref_path = candidates[int(self.rng.integers(0, len(candidates)))]
        ref_wav = wav if ref_path == audio_path else _load_and_resample(ref_path, self.sample_rate)
        ref_clip = _crop_or_loop(ref_wav, self.ref_samples, self.rng)

        item: Dict[str, torch.Tensor] = {
            "audio": torch.from_numpy(segment),
            "ref_audio": torch.from_numpy(ref_clip),
            "speaker_id": torch.tensor(spk_id, dtype=torch.long),
        }

        if self.ssl_cache_dir is not None:
            ssl_path = self.ssl_cache_dir / audio_path.relative_to(self.data_dir).with_suffix(".pt")
            if ssl_path.exists():
                ssl = torch.load(ssl_path, map_location="cpu", weights_only=True).to(torch.float32)  # (768, T_50hz)
                seg_50 = max(1, (self.segment_frames + 1) // 2)
                if ssl.shape[1] < seg_50:
                    reps_s = int(math.ceil(float(seg_50) / float(max(1, ssl.shape[1]))))
                    ssl = ssl.repeat(1, reps_s)
                # 50Hz SSL frame = 20ms = 480 samples at 24kHz
                s50 = min(max(0, int(round(float(start) / 480.0))), max(0, ssl.shape[1] - seg_50))
                ssl_seg = ssl[:, s50 : s50 + seg_50].unsqueeze(0)  # (1, 768, seg_50)
                ssl_100 = torch.nn.functional.interpolate(
                    ssl_seg, size=self.segment_frames, mode="linear", align_corners=False
                ).squeeze(0)
                item["ssl_feat"] = ssl_100

        return item


class SyntheticVocalDataset(Dataset):
    """Zero-dependency synthetic multi-speaker formant vocal dataset for warmstart & testing."""

    def __init__(
        self,
        num_speakers: int = 8,
        num_samples: int = 200,
        segment_frames: int = 16,
        ref_seconds: float = 3.0,
        sample_rate: int = SAMPLE_RATE,
        seed: int = 123,
    ) -> None:
        super().__init__()
        self.num_speakers = num_speakers
        self.num_samples = num_samples
        self.segment_samples = segment_frames * HOP_LENGTH
        self.ref_samples = int(ref_seconds * sample_rate)
        self.sample_rate = sample_rate
        self.rng = np.random.default_rng(seed)

        # Assign distinct acoustic signatures (base pitch and 3 vowel formants) per speaker
        self.speaker_profiles = []
        for s in range(num_speakers):
            ratio = float(s) / max(1, num_speakers - 1)
            base_f0 = 105.0 + ratio * 195.0  # 105 Hz to 300 Hz
            f1 = 350.0 + ((s * 97) % 300)
            f2 = 1100.0 + ((s * 241) % 900)
            f3 = 2400.0 + ((s * 313) % 700)
            self.speaker_profiles.append((base_f0, f1, f2, f3))

    def _synthesize_utterance(self, spk_id: int, num_samples: int) -> np.ndarray:
        base_f0, f1, f2, f3 = self.speaker_profiles[spk_id]
        t = np.arange(num_samples, dtype=np.float32) / float(self.sample_rate)

        # Smooth intonational contour + vibrato
        contour_rate = float(self.rng.uniform(1.5, 4.0))
        contour_phase = float(self.rng.uniform(0.0, 2.0 * np.pi))
        f0_t = base_f0 * (1.0 + 0.08 * np.sin(2.0 * np.pi * contour_rate * t + contour_phase))
        phase = 2.0 * np.pi * np.cumsum(f0_t) / float(self.sample_rate)

        # Harmonic sum weighted by speaker-specific Gaussian formant envelopes
        wave = np.zeros(num_samples, dtype=np.float32)
        for h in range(1, 16):
            freq = base_f0 * h
            if freq > self.sample_rate * 0.45:
                break
            g1 = math.exp(-0.5 * ((freq - f1) / 140.0) ** 2)
            g2 = math.exp(-0.5 * ((freq - f2) / 220.0) ** 2)
            g3 = math.exp(-0.5 * ((freq - f3) / 300.0) ** 2)
            amp = (0.65 / h) + 0.45 * g1 + 0.30 * g2 + 0.18 * g3
            wave += float(amp) * np.sin(h * phase)

        # Amplitude envelope with occasional unvoiced breath noise segments
        env_mod = 0.55 + 0.45 * np.sin(2.0 * np.pi * float(self.rng.uniform(2.0, 5.5)) * t)
        breath = self.rng.normal(0.0, 0.015, size=num_samples).astype(np.float32)
        out = 0.25 * env_mod * wave + breath
        peak = float(np.max(np.abs(out)))
        if peak > 0.9:
            out = out * (0.9 / peak)
        return out.astype(np.float32)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        spk_id = index % self.num_speakers
        segment = self._synthesize_utterance(spk_id, self.segment_samples)
        ref_clip = self._synthesize_utterance(spk_id, self.ref_samples)
        return {
            "audio": torch.from_numpy(segment),
            "ref_audio": torch.from_numpy(ref_clip),
            "speaker_id": torch.tensor(spk_id, dtype=torch.long),
        }
