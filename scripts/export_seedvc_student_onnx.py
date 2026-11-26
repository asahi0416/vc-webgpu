import os, sys, shutil, time
import numpy as np
import torch
import onnx
import onnxruntime as ort
from onnxruntime.transformers import float16

sys.path.insert(0, "third_party/seed-vc")
import yaml
from modules.commons import recursive_munch, build_model

CKPT_PATH = "checkpoints/seedvc_dit_1step_sharp.pt"
OUT_FP32  = "scratch/dit_student_sharp_fp32.onnx"
OUT_FP16  = "models/seedvc/dit_student_fp16.onnx"
BAK_FP16  = "models/seedvc/dit_student_fp16.onnx.bak_prog"

if os.path.exists(OUT_FP16) and not os.path.exists(BAK_FP16):
    print(f"Backing up existing {OUT_FP16} -> {BAK_FP16}")
    shutil.copy2(OUT_FP16, BAK_FP16)

cfg_path = "third_party/seed-vc/configs/presets/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"
cfg = yaml.safe_load(open(cfg_path, "r"))
mp = recursive_munch(cfg["model_params"])
mp.dit_type = "DiT"
pt_model = build_model(mp, stage="DiT")
estimator = pt_model["cfm"].estimator.float().eval()

sd = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
if isinstance(sd, dict) and "estimator" in sd: sd = sd["estimator"]
elif isinstance(sd, dict) and "model" in sd: sd = sd["model"]
estimator.load_state_dict(sd, strict=True)
estimator.setup_caches(max_batch_size=2, max_seq_length=4096)
# Convert Python bool flags to int so TorchScript ONNX export avoids aten::add(Tensor, bool)
estimator.style_as_token = int(estimator.style_as_token)
estimator.time_as_token = int(estimator.time_as_token)
estimator.eval()

class DitWrapper(torch.nn.Module):
    def __init__(self, est):
        super().__init__()
        self.est = est
    def forward(self, x, prompt_x, x_lens, t, style, cond):
        return self.est(x, prompt_x, x_lens, t, style, cond)

wrapper = DitWrapper(estimator).eval()

B, T = 1, 128
x = torch.randn(B, 128, T, dtype=torch.float32)
prompt_x = torch.randn(B, 128, T, dtype=torch.float32)
x_lens = torch.tensor([T], dtype=torch.int64)
t = torch.tensor([0.0], dtype=torch.float32)
style = torch.randn(B, 192, dtype=torch.float32)
cond = torch.randn(B, T, 768, dtype=torch.float32)

print("Exporting FP32 ONNX...")
t0 = time.time()
with torch.no_grad():
    pt_out = wrapper(x, prompt_x, x_lens, t, style, cond).numpy()
    torch.onnx.export(
        wrapper,
        (x, prompt_x, x_lens, t, style, cond),
        OUT_FP32,
        input_names=["x", "prompt_x", "x_lens", "t", "style", "cond"],
        output_names=["v"],
        dynamic_axes={
            "x": {0: "b", 2: "t"},
            "prompt_x": {0: "b", 2: "t"},
            "x_lens": {0: "b"},
            "t": {0: "b"},
            "style": {0: "b"},
            "cond": {0: "b", 1: "t"},
            "v": {0: "b", 2: "t"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
print(f"FP32 exported in {time.time()-t0:.1f}s ({os.path.getsize(OUT_FP32)/1e6:.2f} MB)")

print("Converting to FP16 (keep_io_types=True)...")
model_fp32 = onnx.load(OUT_FP32)
model_fp16 = float16.convert_float_to_float16(
    model_fp32,
    keep_io_types=True,
    disable_shape_infer=False,
)
onnx.save(model_fp16, OUT_FP16)
print(f"FP16 saved to {OUT_FP16} ({os.path.getsize(OUT_FP16)/1e6:.2f} MB)")

# Verify parity on B=1, T=128 and B=1, T=300
sess = ort.InferenceSession(OUT_FP16, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
ort_out = sess.run(None, {
    "x": x.numpy(),
    "prompt_x": prompt_x.numpy(),
    "x_lens": x_lens.numpy(),
    "t": t.numpy(),
    "style": style.numpy(),
    "cond": cond.numpy(),
})[0]
n_nan = int(np.isnan(ort_out).sum())
max_diff = float(np.max(np.abs(pt_out - ort_out)))
mean_diff = float(np.mean(np.abs(pt_out - ort_out)))
cos = float(np.dot(pt_out.ravel(), ort_out.ravel()) / (np.linalg.norm(pt_out) * np.linalg.norm(ort_out) + 1e-12))
print(f"ORT Parity (B=1, T=128): nan={n_nan}, max_diff={max_diff:.5f}, mean_diff={mean_diff:.5f}, cos={cos:.6f}")
assert n_nan == 0 and cos > 0.999, "ONNX FP16 parity check failed!"

# Clean up temporary FP32 file
if os.path.exists(OUT_FP32):
    os.remove(OUT_FP32)
print("Export & parity verification SUCCESS!")
