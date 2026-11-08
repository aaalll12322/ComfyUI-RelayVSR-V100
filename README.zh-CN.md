# ComfyUI-RelayVSR-V100

[English](README.md) | 简体中文

一个节点，让 RelayVSR 的 Flash 解码器在 sm_70 显卡上重新用上 fused attention。
不修改任何 ComfyUI 或 comfy_kitchen 文件，删掉节点即刻恢复。

15 秒 1080p 视频（360 帧）：**143.8 s → 74.2 s（1.94×）**。

---

## 实现了什么

把 RelayVSR Flash 解码器里 **GQA 的 K/V 投影权重按 head 展开成 MHA**，
让 PyTorch 的 cutlass fused attention 接管。

工作流接法：

```
[Load VAE] ──► [RelayVSR V100 Attention Fix] ──► [VSR Decode]
```

节点位置 `advanced/v100`。节点深拷贝 VAE 后只改副本，所以删掉节点即刻恢复原状。
副本开销约 **39 ms、显存 +78 MB**（62 MB 权重副本 + 16 MB 展开后的 K/V 投影），
按 VAE 对象只构建一次、之后复用；在缓存存活期间常驻，且自带独立 patcher，
显存紧张时 ComfyUI 仍可将其 offload 到 CPU。

传入非 GQA 模型时打一行 `no GQA attention found` 后原样透传。

## 为什么需要

RelayVSR 的结构是「Wan 2.2 5B 每 15 帧画一个关键帧 + Flash 解码器逐帧补中间帧」，
约 85% 的时间花在那个逐帧解码器上。而解码器用的是 GQA（`num_heads=8, kv_groups=2`）。

sm_70 上 PyTorch 的 `EFFICIENT_ATTENTION` 后端（cutlass
`fmha_cutlassF_f16_aligned_32x128_rf_sm70`）**只接受 MHA**。GQA 会让
`can_use_efficient_attention()` 返回 False，SDPA 静默退到 math 后端——
QK^T → 独立 softmax kernel → PV，物化完整 S×S。不报错、不警告，只是慢。

同一形状（B=1, 8 heads, S=4096, head_dim=64, fp16）：

| 路径 | TFLOPs | kernel |
|---|---|---|
| SDPA（GQA，含 `enable_gqa=True`） | 4.30 | math，分步 |
| SDPA（MHA） | **34.36** | **fused cutlass** |

`enable_gqa=True` 在 sm_70 上同样是 math 路径，看着"支持 GQA"但没用。

GQA 的定义是「第 i 个 kv head 服务连续 r 个 q head」，所以按 head
`repeat_interleave` 展开 K/V 权重、再把 `kv_groups` 设为 `num_heads`，数学上等价。
上游代码里的 `k.view(b, 1, H, W, self.kv_groups, head_dim)` 自然成立，
不需要重写任何 forward。

## 测试结果

**环境**：Python 3.12.10 / Tesla V100-SXM2-16GB（sm_70）/ PyTorch 2.8.0+cu128 / CUDA 12.8
**权重**：`relayvsr_x4_int8_convrot.safetensors` + `relayvsr_flash_vae_x4_fp32.safetensors`
**输入**：24 fps，fp16，`detail=1.0`　**计时**：`perf_counter` + `synchronize`，无 profiler

以下数据**全部来自 sm_70**。sm_75（Tesla T4 / RTX 20 系）会遇到同一个 math 回退、
走同一段代码，但**没有做过基准测试**，因此不对其速度作任何承诺。

### 解码器（本插件影响的部分）

| 原始 → 输出 | 帧数 | 时长 | 关键帧 | 原始 | 本插件 | 提速 |
|---|---|---|---|---|---|---|
| 480×270 → 1920×1080 | 120 | 5 s | 9 | 47.76 s | **24.79 s** | **1.93×** |
| 480×270 → 1920×1080 | 240 | 10 s | 17 | 95.86 s | **49.49 s** | **1.94×** |
| 480×270 → 1920×1080 | 360 | 15 s | 25 | 143.84 s | **74.15 s** | **1.94×** |
| 640×360 → 2560×1440 | 120 | 5 s | 9 | 95.96 s | **51.31 s** | **1.87×** |

提速与视频长度无关（5 / 10 / 15 秒均为 1.93–1.94×），说明没有随序列增长的额外开销。

### 整条流程（解码 + Wan 主干）

主干每 15 帧过一次，单个关键帧约 743 ms，不受本插件影响。

| 视频 | 主干 | 解码 | 合计 | 吞吐 |
|---|---|---|---|---|
| 5 s 1080p（9 关键帧） | 6.7 s | 47.8 → 24.8 s | **54.5 → 31.5 s** | 2.20 → 3.81 FPS |
| 15 s 1080p（25 关键帧） | 18.6 s | 143.8 → 74.2 s | **162.4 → 92.8 s** | 2.22 → 3.88 FPS |

端到端 **1.73–1.75×**——低于解码器单项，因为主干占了约 12–20% 且不受影响。

### 数值等价性

同一输入、360 帧输出逐像素比较：**PSNR 66.5 dB**，mean abs diff 1.9e-4。
参考：同模型 fp16 与 fp32 的 PSNR 为 65.9 dB，即本插件误差不超过 fp16 自身精度。

## 适用设备

插件按 `compute_capability < (8, 0)` 判断是否生效。

| 架构 | 代表设备 | 状态 |
|---|---|---|
| sm_70 | Tesla V100、Quadro GV100、Titan V | ✅ 实测（Tesla V100-SXM2-16GB） |
| sm_75 | Tesla T4、RTX 20 系 | 同一根因，未实测 |
| sm_80+ | A100、RTX 30 系及以上 | 原生支持 GQA，不需要 |

## 注意

- 副本显式 `.to(vae.device, vae.vae_dtype)`：ComfyUI 的 `load_models_gpu` 只搬 patcher 里的原模型
- 同一 VAE 对象只拷贝一次，之后走缓存，重复应用不会二次展开；缓存随该 VAE 对象（或重启）释放，换 VAE 会重新构建
- `V100_GQA_VERBOSE=1` 可逐层打印展开信息
- 仅供 x4 与 x2 的 Flash 解码器；不改变输出分辨率、不改 `detail` 行为
- 彻底移除：删除 `custom_nodes/ComfyUI-RelayVSR-V100` 后重启

## 许可

Apache-2.0，全文见 [LICENSE](LICENSE)。
