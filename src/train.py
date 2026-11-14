"""CLI training entrypoint for Causal Flow Matching VC."""

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.streaming_vc_pipeline import StreamingFlowMatchingVC
from src.training.dataset import SyntheticVocalDataset, WAVVoiceDataset
from src.training.hf_dataset import (
    DEFAULT_HF_REPO,
    TransactionalBackup,
    fetch_and_prepare_hf_dataset,
)
from src.training.trainer import VCTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Causal Flow Matching VC")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to directory containing speaker WAV folders (uses synthetic warmstart if omitted).",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help="Max steps per epoch (defaults to full dataset length)",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate")
    parser.add_argument(
        "--segment-frames",
        type=int,
        default=16,
        help="Training segment length in 10ms frames (16 frames = 160ms)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "vc_latest.pt"),
        help="Checkpoint save path",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional checkpoint path to resume training from",
    )
    parser.add_argument(
        "--export-onnx",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export updated ONNX models after training completes",
    )
    parser.add_argument(
        "--onnx-dir",
        type=str,
        default=str(PROJECT_ROOT / "models"),
        help="Directory for exported ONNX models",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to train on ('auto', 'cuda', or 'cpu')",
    )
    parser.add_argument(
        "--auto-hf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-download and preprocess Japanese speech from Hugging Face Hub if --data-dir is omitted",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=DEFAULT_HF_REPO,
        help="Hugging Face dataset repository ID",
    )
    parser.add_argument(
        "--samples-per-speaker",
        type=int,
        default=6,
        help="Number of WAV samples per speaker to prepare from Hugging Face",
    )
    parser.add_argument(
        "--hf-data-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "jvnv"),
        help="Directory for prepared Hugging Face WAV dataset",
    )
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Force synthetic warmstart dataset instead of Hugging Face real speech",
    )
    args = parser.parse_args()

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )

    use_auto_hf = (not args.data_dir) and (not args.synthetic_only) and args.auto_hf
    backup_targets = [Path(args.save_path)]
    if use_auto_hf:
        backup_targets.insert(0, Path(args.hf_data_dir))
    if args.export_onnx:
        onnx_dir = Path(args.onnx_dir)
        backup_targets.extend(
            [onnx_dir / "speaker_encoder.onnx", onnx_dir / "streaming_vc_core.onnx"]
        )

    with TransactionalBackup(backup_targets):
        dataset = _build_dataset(args, use_auto_hf=use_auto_hf)
        _run_training(args, dataset=dataset, device=device)


def _build_dataset(args: argparse.Namespace, use_auto_hf: bool):
    if args.data_dir:
        print(f"[Train] Loading WAV dataset from: {args.data_dir}")
        return WAVVoiceDataset(
            data_dir=args.data_dir,
            segment_frames=args.segment_frames,
        )

    if use_auto_hf:
        hf_dir = fetch_and_prepare_hf_dataset(
            output_dir=args.hf_data_dir,
            repo_id=args.hf_repo,
            samples_per_speaker=args.samples_per_speaker,
        )
        print(f"[Train] Loading Hugging Face WAV dataset from: {hf_dir}")
        return WAVVoiceDataset(
            data_dir=hf_dir,
            segment_frames=args.segment_frames,
        )

    num_synth = max(40, (args.steps_per_epoch or 20) * args.batch_size)
    print(f"[Train] Running synthetic multi-speaker warmstart ({num_synth} samples)")
    return SyntheticVocalDataset(
        num_speakers=8,
        num_samples=num_synth,
        segment_frames=args.segment_frames,
    )


def _build_balanced_sampler(dataset):
    """Balance rare vs studio speakers cleanly when speaker file counts differ strongly."""
    if not hasattr(dataset, "speaker_files") or not hasattr(dataset, "items"):
        return None
    spk_counts = {sid: len(files) for sid, files in dataset.speaker_files.items() if files}
    if len(spk_counts) <= 1:
        return None
    counts = sorted(spk_counts.values())
    med_cnt = counts[len(counts) // 2]
    large_spks = {sid for sid, c in spk_counts.items() if c > med_cnt * 2}
    if not large_spks or len(large_spks) == len(spk_counts):
        return None

    from torch.utils.data import WeightedRandomSampler

    small_spks = set(spk_counts.keys()) - large_spks
    weights = []
    for spk_id, _ in dataset.items:
        c = spk_counts[spk_id]
        if spk_id in large_spks:
            weights.append(0.25 / (len(large_spks) * c))
        else:
            weights.append(0.75 / (len(small_spks) * c))
    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)


def _run_training(args: argparse.Namespace, dataset, device: str) -> None:
    sampler = _build_balanced_sampler(dataset)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        drop_last=len(dataset) >= args.batch_size,
    )

    steps_per_ep = args.steps_per_epoch or max(1, len(loader))
    total_steps = args.epochs * steps_per_ep

    model = StreamingFlowMatchingVC(num_flow_steps=2)
    trainer = VCTrainer(model=model, lr=args.lr, total_steps=total_steps, device=device)

    start_epoch = 1
    if args.resume:
        ckpt = trainer.load_checkpoint(args.resume)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"[Train] Resumed from {args.resume} at step {trainer.global_step}")

    print(
        f"[Train] Device={device} | Epochs={args.epochs} | "
        f"Steps/Epoch={steps_per_ep} | Batch={args.batch_size}"
    )

    last_metrics = {}
    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0 = time.perf_counter()
        ep_totals: dict[str, float] = {}
        steps_done = 0

        for step_idx, batch in enumerate(loader):
            if step_idx >= steps_per_ep:
                break
            metrics = trainer.train_step(batch)
            for k, v in metrics.items():
                ep_totals[k] = ep_totals.get(k, 0.0) + v
            steps_done += 1

        avg_metrics = {k: v / max(1, steps_done) for k, v in ep_totals.items()}
        last_metrics = avg_metrics
        dt = time.perf_counter() - t0
        print(
            f"[Epoch {epoch}/{start_epoch + args.epochs - 1}] ({dt:.1f}s) "
            f"total={avg_metrics['total_loss']:.4f} | "
            f"rf={avg_metrics['rf_loss']:.4f} | cm={avg_metrics['cm_loss']:.4f} | "
            f"mel={avg_metrics['mel_l1']:.4f} | f0={avg_metrics['f0_loss']:.4f} | "
            f"spk={avg_metrics['spk_loss']:.4f} | voc={avg_metrics['vocoder_loss']:.4f}"
        )

    saved = trainer.save_checkpoint(
        args.save_path, epoch=start_epoch + args.epochs - 1, metrics=last_metrics
    )
    print(f"[Train] Saved checkpoint: {saved}")

    if args.export_onnx:
        paths = trainer.export_onnx_models(output_dir=args.onnx_dir)
        spk_mb = paths["speaker_encoder"].stat().st_size / (1024 * 1024)
        core_mb = paths["streaming_vc_core"].stat().st_size / (1024 * 1024)
        print(
            f"[Train] Exported ONNX: {paths['speaker_encoder'].name} ({spk_mb:.2f} MB), "
            f"{paths['streaming_vc_core'].name} ({core_mb:.2f} MB)"
        )


if __name__ == "__main__":
    main()
