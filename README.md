# ComfyUI-RelayVSR-V100

English | [简体中文](README.zh-CN.md)

One node that gives RelayVSR's flash decoder its fused attention back on sm_70 GPUs.
Nothing in ComfyUI or `comfy_kitchen` is modified — remove the node and you are back to stock.

**15 s 1080p video (360 frames): 143.8 s → 74.2 s (1.94×).**

---

## What it does

RelayVSR's flash decoder uses GQA (`num_heads=8`, `kv_groups=2`). On sm_70, PyTorch's
`EFFICIENT_ATTENTION` backend (cutlass `fmha_cutlassF_f16_aligned_32x128_rf_sm70`) only
accepts MHA, so GQA makes `can_use_efficient_attention()` return False and SDPA silently
falls back to the math backend: QK^T → standalone softmax kernel → PV, with a fully
materialised S×S matrix. No error, no warning — just slower.

This node expands the K/V projection weights along the head dimension so the module
becomes MHA, then sets `kv_groups = num_heads`. The upstream
`k.view(b, 1, H, W, self.kv_groups, head_dim)` still holds, so no forward code is
rewritten and no global function is patched.

GQA is defined as "the i-th kv head serves r consecutive q heads", which is exactly what
`repeat_interleave` does — the expansion is mathematically exact, not an approximation.

Wiring:

```
[Load VAE] ──► [RelayVSR V100 Attention Fix] ──► [VSR Decode]
```

The node lives under `advanced/v100`. It deep-copies the VAE and only patches the copy,
so deleting the node restores stock behaviour immediately. The copy costs **~39 ms and
+78 MB of VRAM** (62 MB of weights plus 16 MB for the expanded K/V projections). It is
built once per VAE object and reused afterwards; it stays resident while that cached copy
is alive, and since it carries its own patcher, ComfyUI can still offload it when VRAM
gets tight.

Non-GQA models are passed through untouched after a single
`no GQA attention found` log line.

## Why it is needed

On the same shape (B=1, 8 heads, S=4096, head_dim=64, fp16):

| Path | TFLOPs | Kernel |
|---|---|---|
| SDPA with GQA (incl. `enable_gqa=True`) | 4.30 | math, unfused |
| SDPA with MHA | **34.36** | **fused cutlass** |

`enable_gqa=True` also takes the math path on sm_70 — it looks like GQA support but
costs the same 8×.

## Results

**Environment**: Python 3.12.10 / Tesla V100-SXM2-16GB (sm_70) / PyTorch 2.8.0+cu128 / CUDA 12.8
**Weights**: `relayvsr_x4_int8_convrot.safetensors` + `relayvsr_flash_vae_x4_fp32.safetensors`
**Input**: 24 fps, fp16, `detail=1.0`　**Timing**: `perf_counter` + `synchronize`, no profiler

All numbers below are **sm_70 only**. sm_75 (Tesla T4 / RTX 20 series) hits the same
math fallback and is covered by the same code path, but it has **not been benchmarked** —
no performance claim is made for it.

### Decoder (the part this node affects)

| Input → Output | Frames | Duration | Keyframes | Baseline | This node | Speedup |
|---|---|---|---|---|---|---|
| 480×270 → 1920×1080 | 120 | 5 s | 9 | 47.76 s | **24.79 s** | **1.93×** |
| 480×270 → 1920×1080 | 240 | 10 s | 17 | 95.86 s | **49.49 s** | **1.94×** |
| 480×270 → 1920×1080 | 360 | 15 s | 25 | 143.84 s | **74.15 s** | **1.94×** |
| 640×360 → 2560×1440 | 120 | 5 s | 9 | 95.96 s | **51.31 s** | **1.87×** |

The speedup does not change with clip length (1.93–1.94× across 5 / 10 / 15 s), so there is
no per-sequence overhead added.

### Full pipeline (decoder + Wan backbone)

The backbone runs once per 15 frames, ~743 ms per keyframe, and is unaffected by this node.

| Clip | Backbone | Decode | Total | Throughput |
|---|---|---|---|---|
| 5 s 1080p (9 keyframes) | 6.7 s | 47.8 → 24.8 s | **54.5 → 31.5 s** | 2.20 → 3.81 FPS |
| 15 s 1080p (25 keyframes) | 18.6 s | 143.8 → 74.2 s | **162.4 → 92.8 s** | 2.22 → 3.88 FPS |

End-to-end **1.73–1.75×** — lower than the decoder-only figure because the backbone takes
12–20% of the total and is untouched.

### Numerical equivalence

Same input, 360-frame output compared per pixel: **PSNR 66.5 dB**, mean abs diff 1.9e-4.
For reference, the same model at fp16 vs fp32 gives 65.9 dB — the error introduced here is
below fp16's own rounding noise.

## Supported hardware

The node activates on `compute_capability < (8, 0)`.

| Architecture | Devices | Status |
|---|---|---|
| sm_70 | Tesla V100, Quadro GV100, Titan V | ✅ tested (Tesla V100-SXM2-16GB) |
| sm_75 | Tesla T4, RTX 20 series | same root cause, not tested |
| sm_80+ | A100, RTX 30 series and newer | native GQA support, not needed |

## Notes

- The copy is explicitly moved to `vae.device` / `vae.vae_dtype`; ComfyUI's
  `load_models_gpu` only moves the original model held by the patcher.
- One deep copy per VAE object, cached and reused; re-applying never expands twice. The
  cache lives and dies with that VAE object (or a restart), so switching VAE rebuilds it.
- `V100_GQA_VERBOSE=1` prints per-layer expansion details.
- Applies to the x4 and x2 flash decoders only; it does not change output resolution or
  `detail` behaviour.
- Uninstall: delete this folder from `custom_nodes` and restart ComfyUI.

## License

Apache-2.0 — see [LICENSE](LICENSE).
