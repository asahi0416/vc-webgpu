#!/usr/bin/env python3
"""
Upload ONNX models and assets to Hugging Face Hub.

Usage:
  python scripts/upload_to_hf.py --repo your-username/vc-web-models [--token hf_xxx]
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


def main():
    parser = argparse.ArgumentParser(description="Upload voice changer models to Hugging Face Hub")
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="Hugging Face model repository ID (e.g. 'your-username/vc-web-models')",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face write token (optional if logged in via `huggingface-cli login`)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private repository if it does not exist (default: public)",
    )
    parser.add_argument(
        "--include-knn",
        action="store_true",
        help="Also upload kNN-VC models (wavlm, hifigan) in addition to Seed-VC",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("[Error] huggingface_hub is not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi(token=args.token)

    # 1. Ensure repository exists
    repo_id = args.repo.strip()
    print(f"[*] Checking Hugging Face model repository: {repo_id}")
    try:
        create_repo(repo_id=repo_id, repo_type="model", private=args.private, exist_ok=True, token=args.token)
        print(f"[+] Repository confirmed: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"[!] Warning when creating repository: {e}")

    # 2. Select files to upload
    # Essential Seed-VC files
    seed_files = [
        "seedvc/dit_student_fp16.onnx",
        "seedvc/whisper_small_enc.onnx",
        "seedvc/bigvgan.onnx",
        "seedvc/fcpe.onnx",
        "seedvc/campplus_fp16.onnx",
        "seedvc/length_regulator_fp16.onnx",
        "seedvc/mel_banks.bin",
        "seedvc/mel_banks.json",
        "seedvc/fcpe_mel_128x513.bin",
    ]

    # FlowVC files
    flow_files = [
        "speaker_encoder.onnx",
        "streaming_vc_core.onnx",
    ]

    # Optional kNN files
    knn_files = [
        "knn/wavlm6_w1500.onnx",
        "knn/wavlm6_w1500_int8mm.onnx",
        "knn/hifigan_dyn.onnx",
    ]

    target_files = seed_files + flow_files
    if args.include_knn:
        target_files += knn_files

    print(f"\n[*] Preparing to upload {len(target_files)} model files...")

    for rel_path in target_files:
        local_path = MODELS_DIR / rel_path
        if not local_path.exists():
            print(f"[!] File not found locally, skipping: {local_path}")
            continue

        size_mb = local_path.stat().st_size / (1024 * 1024)
        repo_path = f"models/{rel_path}"
        print(f"[*] Uploading: {rel_path} ({size_mb:.1f} MB) -> {repo_path} ...", flush=True)

        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"    [OK] Uploaded {rel_path}")
        except Exception as e:
            print(f"    [FAIL] Error uploading {rel_path}: {e}")

    # 3. Create or update README.md in the model repo
    readme_content = f"""---
license: apache-2.0
tags:
- voice-conversion
- webgpu
- onnx
---

# ONNX Models for WebGPU Real-Time Voice Changer

Models hosted for WebGPU real-time browser inference.
Live Demo supported via Vercel / Cloudflare Pages.

### Included Models:
- **Seed-VC**: 1-Step Distilled DiT (`dit_student_fp16.onnx`), Whisper Encoder (`whisper_small_enc.onnx`), BigVGAN (`bigvgan.onnx`), FCPE (`fcpe.onnx`), CAMPPlus (`campplus_fp16.onnx`)
- **Causal FlowVC**: Fast causal flow matching model (`streaming_vc_core.onnx`, `speaker_encoder.onnx`)
"""
    try:
        api.upload_file(
            path_or_fileobj=readme_content.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
        )
        print("[+] Created README.md on Hugging Face")
    except Exception as e:
        print(f"[!] Warning on README upload: {e}")

    print("\n" + "=" * 60)
    print("SUCCESS: Model upload completed!")
    print(f"Repository URL: https://huggingface.co/{repo_id}")
    print("\nHow to use in Web App:")
    print(f"1. Open your Vercel URL with parameter:")
    print(f"   https://your-domain.vercel.app/?hf_repo={repo_id}")
    print(f"2. Or set window.__VC_DEFAULT_HF_REPO__ = '{repo_id}' in web/config.js")
    print("=" * 60)


if __name__ == "__main__":
    main()
