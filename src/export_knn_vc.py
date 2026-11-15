"""Export kNN-VC (WavLM-Large layer-6 encoder + prematched HiFi-GAN) to browser-ready ONNX.

kNN-VC: Baas et al., "Voice Conversion With Just Nearest Neighbors" (Interspeech 2023, MIT license,
https://github.com/bshall/knn-vc). Pretrained weights are downloaded from the upstream GitHub release.

Outputs (models/knn/):
  wavlm6_w1500_int8mm.onnx : audio [1, 24080] float32 @16 kHz -> feat [1, 75, 1024]
                             (first 6 transformer layers of WavLM-Large, MatMul weights int8)
  hifigan_dyn.onnx         : feat [1, T, 1024] -> wav [1, 1, T*320] float32 @16 kHz (T is dynamic)
"""

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORK = PROJECT_ROOT / "third_party" / "knn-vc"
RELEASE = "https://github.com/bshall/knn-vc/releases/download/v0.1/"
LAYER = 6
WIN_FRAMES = 75
WIN_SAMPLES = 400 + 320 * (WIN_FRAMES - 1)  # 24080


def _download(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 1_000_000:
        return
    print(f"[knn] downloading {url}")
    urllib.request.urlretrieve(url, str(dst))


def prepare_assets(work: Path) -> None:
    work.mkdir(parents=True, exist_ok=True)
    if not (work / "repo" / "wavlm").exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/bshall/knn-vc.git", str(work / "repo")],
            check=True,
        )
    _download(RELEASE + "WavLM-Large.pt", work / "WavLM-Large.pt")
    _download(RELEASE + "prematch_g_02500000.pt", work / "prematch_g_02500000.pt")


def export_knn_vc(work: Path = DEFAULT_WORK, out_dir: Path = PROJECT_ROOT / "models" / "knn") -> None:
    prepare_assets(work)
    sys.path.insert(0, str(work / "repo"))
    from hifigan.models import Generator as HiFiGAN  # noqa: WPS433
    from hifigan.utils import AttrDict  # noqa: WPS433
    from wavlm.WavLM import WavLM, WavLMConfig  # noqa: WPS433

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = work / "onnx"
    tmp.mkdir(exist_ok=True)

    ck = torch.load(work / "WavLM-Large.pt", map_location="cpu", weights_only=False)
    wavlm = WavLM(WavLMConfig(ck["cfg"]))
    wavlm.load_state_dict(ck["model"])
    wavlm.eval()
    wavlm.encoder.layers = wavlm.encoder.layers[:LAYER]
    torch.nn.utils.remove_weight_norm(wavlm.encoder.pos_conv[0])

    class Enc(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m.extract_features(x, output_layer=LAYER, ret_layer_results=False)[0]

    enc = Enc(wavlm).eval()
    x = torch.randn(1, WIN_SAMPLES) * 0.05
    fp32 = tmp / "wavlm6_w1500.onnx"
    torch.onnx.export(enc, x, str(fp32), input_names=["audio"], output_names=["feat"], opset_version=17)

    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8 = out_dir / "wavlm6_w1500_int8mm.onnx"
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])
    print(f"[knn] wrote {int8} ({int8.stat().st_size / 1e6:.0f} MB)")

    with open(work / "repo" / "hifigan" / "config_v1_wavlm.json") as f:
        h = AttrDict(json.load(f))
    gen = HiFiGAN(h)
    gen.load_state_dict(torch.load(work / "prematch_g_02500000.pt", map_location="cpu", weights_only=False)["generator"])
    gen.eval()
    gen.remove_weight_norm()
    c = torch.randn(1, 19, 1024)
    hp = out_dir / "hifigan_dyn.onnx"
    torch.onnx.export(
        gen,
        c,
        str(hp),
        input_names=["feat"],
        output_names=["wav"],
        opset_version=17,
        dynamic_axes={"feat": {1: "frames"}, "wav": {2: "samples"}},
    )
    print(f"[knn] wrote {hp} ({hp.stat().st_size / 1e6:.0f} MB)")

    import onnxruntime as ort

    s1 = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        ref = enc(x).numpy()
    o = s1.run(None, {"audio": x.numpy()})[0]
    cos = (o[0] * ref[0]).sum(-1) / (np.linalg.norm(o[0], axis=-1) * np.linalg.norm(ref[0], axis=-1))
    print(f"[knn] wavlm int8 vs torch cos mean={cos.mean():.4f} min={cos.min():.4f}")
    s2 = ort.InferenceSession(str(hp), providers=["CPUExecutionProvider"])
    for t in (19, 29):
        cc = torch.randn(1, t, 1024)
        with torch.inference_mode():
            r = gen(cc).numpy()
        w = s2.run(None, {"feat": cc.numpy()})[0]
        print(f"[knn] hifigan T={t}: out {w.shape}, max abs diff vs torch={np.abs(w - r).max():.2e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "models" / "knn")
    a = ap.parse_args()
    export_knn_vc(a.work_dir, a.out_dir)
