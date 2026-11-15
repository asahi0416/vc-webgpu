"""Pre-extract speaker-agnostic ContentVec features for fast causal distillation training."""

import argparse
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel


def _load_16k_mono(path: Path) -> np.ndarray:
    wav, sr = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    if sr != 16000:
        x_old = np.linspace(0.0, 1.0, len(wav), endpoint=False)
        n_new = max(1, int(round(len(wav) * 16000.0 / float(sr))))
        x_new = np.linspace(0.0, 1.0, n_new, endpoint=False)
        wav = np.interp(x_new, x_old, wav).astype(np.float32)
    # Standard HuBERT / ContentVec per-utterance normalization
    mean = float(wav.mean())
    std = float(wav.std())
    return ((wav - mean) / max(std, 1e-5)).astype(np.float32)


def extract_cache(
    data_dir: Path,
    out_dir: Path,
    model_id: str = "fwerkor/content-vec-best",
    batch_size: int = 8,
    max_files: int = 0,
) -> None:
    exts = {".wav", ".flac", ".ogg"}
    files = sorted([f for f in data_dir.rglob("*") if f.suffix.lower() in exts])
    if max_files > 0:
        files = files[:max_files]

    pending = []
    for f in files:
        rel = f.relative_to(data_dir).with_suffix(".pt")
        dst = out_dir / rel
        if not dst.exists():
            pending.append((f, dst))

    print(f"[SSL-Cache] Total audio={len(files)}, pending={len(pending)} -> {out_dir}")
    if not pending:
        return

    model = AutoModel.from_pretrained(model_id)
    model.eval()
    t0 = time.perf_counter()

    with torch.inference_mode():
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            wavs = [_load_16k_mono(src) for src, _ in batch]
            lens = [len(w) for w in wavs]
            max_len = max(lens)
            x = torch.zeros(len(wavs), max_len, dtype=torch.float32)
            mask = torch.zeros(len(wavs), max_len, dtype=torch.long)
            for i, w in enumerate(wavs):
                x[i, : len(w)] = torch.from_numpy(w)
                mask[i, : len(w)] = 1

            out = model(x, attention_mask=mask).last_hidden_state  # (B, T_50hz, 768)
            for i, (_, dst) in enumerate(batch):
                # ContentVec CNN stride is 320 samples (20ms at 16kHz)
                t_valid = max(1, (lens[i] - 80) // 320)
                t_valid = min(t_valid, out.shape[1])
                feat = out[i, :t_valid].transpose(0, 1).contiguous().to(torch.float16)  # (768, T)
                dst.parent.mkdir(parents=True, exist_ok=True)
                torch.save(feat, dst)

            done = start + len(batch)
            if done % 160 == 0 or done == len(pending):
                dt = time.perf_counter() - t0
                rate = done / max(dt, 1e-3)
                eta = (len(pending) - done) / max(rate, 1e-3)
                print(f"[SSL-Cache] {done}/{len(pending)} ({rate:.1f} files/s, ETA {eta/60:.1f}m)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ContentVec SSL cache")
    parser.add_argument("--data-dir", type=str, default="data/mixed")
    parser.add_argument("--out-dir", type=str, default="data/mixed_ssl")
    parser.add_argument("--model-id", type=str, default="fwerkor/content-vec-best")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-files", type=int, default=0)
    args = parser.parse_args()
    extract_cache(
        Path(args.data_dir),
        Path(args.out_dir),
        model_id=args.model_id,
        batch_size=args.batch_size,
        max_files=args.max_files,
    )


if __name__ == "__main__":
    main()
