"""Build pseudo-speaker folders from Common Voice JA (no client_id in this mirror).

Rows are ordered in contiguous speaker runs; runs are split where gender/age
changes or adjacent speaker-embedding cosine similarity drops below a threshold.
"""

import argparse
import io
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from huggingface_hub import hf_hub_download

from src.models.streaming_vc_pipeline import StreamingFlowMatchingVC
from src.training.hf_dataset import TransactionalBackup, _preprocess_waveform

REPO = "OpenSpeechHub/Common-Voice-17-Ja"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=2)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "data" / "mixed"))
    ap.add_argument("--jvnv-dir", default=str(PROJECT_ROOT / "data" / "jvnv"))
    ap.add_argument("--min-run", type=int, default=8)
    ap.add_argument("--clips-per-speaker", type=int, default=10)
    ap.add_argument("--max-speakers", type=int, default=60)
    ap.add_argument("--sim-threshold", type=float, default=0.5)
    ap.add_argument("--checkpoint", default=str(PROJECT_ROOT / "checkpoints" / "vc_latest.pt"))
    args = ap.parse_args()

    model = StreamingFlowMatchingVC(num_flow_steps=2)
    ck = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ck["model_state"], strict=True)
    model.eval()

    out = Path(args.out)
    with TransactionalBackup([out]):
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        jv = Path(args.jvnv_dir)
        if jv.exists():
            for d in sorted(p for p in jv.iterdir() if p.is_dir()):
                shutil.copytree(d, out / f"jvnv_{d.name}")

        n_spk = 0
        for sh in range(args.shards):
            path = hf_hub_download(
                REPO, f"data/train-{sh:05d}-of-00352.parquet", repo_type="dataset"
            )
            pf = pq.ParquetFile(path)
            rows = []
            for g in range(pf.num_row_groups):
                rows += pf.read_row_group(g, columns=["audio", "gender", "age"]).to_pylist()

            clips, embs, meta = [], [], []
            for r in rows:
                try:
                    a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
                    a = _preprocess_waveform(a, sr, min_duration_sec=1.0)
                except Exception:
                    continue
                with torch.no_grad():
                    e = model.encode_speaker(torch.from_numpy(a[:72000])[None]).flatten()
                clips.append(a)
                embs.append(e / e.norm())
                meta.append((r["gender"], r["age"]))

            runs, cur = [], [0]
            for i in range(1, len(clips)):
                same = meta[i] == meta[i - 1] and float(embs[i] @ embs[i - 1]) >= args.sim_threshold
                if same:
                    cur.append(i)
                else:
                    runs.append(cur)
                    cur = [i]
            runs.append(cur)
            good = [r for r in runs if len(r) >= args.min_run]
            print(f"[CV] shard {sh}: clips={len(clips)} runs={len(runs)} usable(>= {args.min_run})={len(good)}")

            for run in good:
                if n_spk >= args.max_speakers:
                    break
                d = out / f"cv_{n_spk:03d}"
                d.mkdir()
                for k, idx in enumerate(run[: args.clips_per_speaker]):
                    sf.write(d / f"{k:03d}.wav", clips[idx], 24000, subtype="PCM_16")
                n_spk += 1
        print(f"[CV] pseudo-speakers written: {n_spk} -> {out}")
        if n_spk == 0:
            raise RuntimeError("No usable pseudo-speaker runs found")


if __name__ == "__main__":
    main()
