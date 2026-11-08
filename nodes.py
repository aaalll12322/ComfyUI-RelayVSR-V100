"""
RelayVSR V100 Attention Fix — 节点版（可移除）
==============================================

在 sm_70 / sm_75（Tesla V100、Quadro GV100、Titan V、Tesla T4 等）上，PyTorch 的
SDPA `EFFICIENT_ATTENTION` 后端（cutlass fmha，例如
`fmha_cutlassF_f16_aligned_32x128_rf_sm70`）只接受 MHA。
GQA 模型（K/V head 少于 Q）会让 `can_use_efficient_attention()` 返回 False，
SDPA 静默退化到 math 后端：QK^T -> 独立 softmax -> PV，实测慢约 8 倍。

本插件只做一件事：
把 GQA 的 K/V 投影权重在 head 维上展开成 MHA 的形状，并把 `kv_groups` 改为 `num_heads`。
之后上游代码里的 `view(b, 1, H, W, self.kv_groups, head_dim)` 自然成立，
cutlass fmha 接管。不重写任何 forward、不改任何全局函数。

数学等价：GQA 的定义就是「第 i 个 kv head 服务连续 r 个 q head」，
按 head 展开权重精确复现该语义；RMSNorm 作用在 head_dim 上，重复 head 输入相同，
输出也相同。

可移除：节点**深拷贝**传入的 VAE，只改副本，原对象不动。
所以从工作流删掉节点 = 立刻恢复原状，不需要重启 ComfyUI。
"""

from __future__ import annotations

import copy
import logging
import os

import torch

LOGGER = logging.getLogger("comfyui.v100_gqa")

# 设为 1 时逐层打印展开信息，排查用
_VERBOSE = os.environ.get("V100_GQA_VERBOSE", "0").strip().lower() in ("1", "true", "on", "yes")

# GQA attention 里参与 K/V 投影的四个 Linear（RelayVSR Flash 解码器的命名）
_KV_PROJ_NAMES = ("key", "value", "condition_key", "condition_value")

__all__ = [
    "expand_attention",
    "patch_module_tree",
    "RelayVSRGQAFix",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]


def _expand_linear(lin: torch.nn.Module, kv_heads: int, head_dim: int, r: int) -> None:
    """把 (kv_heads*head_dim, in) 的投影按 head 展开成 (heads*head_dim, in)。"""
    w = lin.weight.data
    out_f, in_f = w.shape
    if out_f != kv_heads * head_dim:
        raise ValueError(
            f"weight out_features {out_f} != kv_heads*head_dim {kv_heads * head_dim}")
    w2 = w.reshape(kv_heads, head_dim, in_f).repeat_interleave(r, dim=0).reshape(-1, in_f)
    lin.weight = torch.nn.Parameter(w2.contiguous(), requires_grad=False)
    if getattr(lin, "bias", None) is not None:
        b2 = lin.bias.data.reshape(kv_heads, head_dim).repeat_interleave(r, dim=0).reshape(-1)
        lin.bias = torch.nn.Parameter(b2.contiguous(), requires_grad=False)


def expand_attention(attn: torch.nn.Module) -> bool:
    """就地把一个 GQA attention 模块展开成 MHA。已是 MHA / 已展开 / 结构不符则 False。"""
    nh = getattr(attn, "num_heads", None)
    kv = getattr(attn, "kv_groups", None)
    hd = getattr(attn, "head_dim", None)
    if not (isinstance(nh, int) and isinstance(kv, int) and isinstance(hd, int)):
        return False
    if kv <= 0 or nh <= 0 or kv >= nh or nh % kv != 0:
        return False
    if getattr(attn, "_v100_gqa_expanded", False):
        return False

    # 先全部校验，再动手，避免展开一半留下坏状态
    for name in _KV_PROJ_NAMES:
        lin = getattr(attn, name, None)
        if lin is None or not hasattr(lin, "weight") or lin.weight is None:
            return False
        if lin.weight.shape[0] != kv * hd:
            return False

    r = nh // kv
    try:
        with torch.no_grad():
            for name in _KV_PROJ_NAMES:
                _expand_linear(getattr(attn, name), kv, hd, r)
    except Exception as exc:                                   # pragma: no cover
        LOGGER.warning("[V100-GQA] expand failed on %s: %s", type(attn).__name__, exc)
        return False

    attn.kv_groups = nh
    attn._v100_gqa_expanded = True
    attn._v100_gqa_orig_kv = kv
    return True


def patch_module_tree(root: torch.nn.Module) -> int:
    """遍历整棵模块树，展开所有符合结构的 GQA attention。返回展开层数。

    判据严格（num_heads / kv_groups / head_dim 三个整数属性 + key/value/
    condition_key/condition_value 四个投影且维度自洽），fused-qkv 或 MHA 模块不会命中。
    """
    n = 0
    for name, m in root.named_modules():
        if expand_attention(m):
            n += 1
            if _VERBOSE:
                LOGGER.info("[V100-GQA] %s: %d -> %d heads",
                            name, getattr(m, "_v100_gqa_orig_kv", 0), m.num_heads)
    return n


def _has_gqa(root: torch.nn.Module) -> bool:
    for m in root.modules():
        nh = getattr(m, "num_heads", None)
        kv = getattr(m, "kv_groups", None)
        if isinstance(nh, int) and isinstance(kv, int) and 0 < kv < nh and nh % kv == 0:
            return True
    return False


class RelayVSRGQAFix:
    """把 VAE 里的 GQA attention 展开成 MHA，在 sm_70 上恢复 fused attention。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vae": ("VAE",)}}

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "apply"
    CATEGORY = "advanced/v100"
    DESCRIPTION = (
        "Expands GQA K/V projections to MHA so PyTorch's fused cutlass attention works "
        "on sm_70 (V100). Mathematically exact, patches a deep copy only — delete the node "
        "to revert. ~2x faster RelayVSR flash decoding. "
        "展开 GQA 的 K/V 投影为 MHA，让 V100 上的 fused attention 生效，约 2 倍解码加速。"
    )

    def apply(self, vae):
        model = getattr(vae, "first_stage_model", None)
        if model is None:
            LOGGER.info("[V100-GQA] input is not a VAE with first_stage_model; pass through")
            return (vae,)

        if not _has_gqa(model):
            LOGGER.info("[V100-GQA] no GQA attention found (model is MHA or fused-qkv); "
                        "pass through unchanged")
            return (vae,)

        # 同一个 VAE 只拷贝一次：deepcopy 约 39 ms，展开后副本约 78 MB(fp16)，
        # 反复出图时没必要每次都复制。
        cached = getattr(vae, "_v100_gqa_cache", None)
        if cached is not None and getattr(cached, "first_stage_model", None) is not None:
            return (cached,)

        # 深拷贝：只改副本，保证删掉节点即恢复原状（VAE 在 ComfyUI 里是缓存共享的）
        try:
            new_model = copy.deepcopy(model)
        except Exception as exc:                               # pragma: no cover
            LOGGER.warning("[V100-GQA] deepcopy failed (%s); falling back to in-place patch", exc)
            n = patch_module_tree(model)
            LOGGER.info("[V100-GQA] %d layers expanded in place (restart to revert)", n)
            return (vae,)

        # 关键：副本必须落在 VAE 的运行设备与精度上。
        # ComfyUI 的 load_models_gpu() 只会搬运 patcher 管理的「原」模型，
        # 副本不在其中，若留在 offload 设备（CPU）就会报
        # "Input type (cuda.HalfTensor) and weight type (HalfTensor) should be the same"。
        _kw = {}
        _dev = getattr(vae, "device", None)
        _dt = getattr(vae, "vae_dtype", None)
        if _dev is not None:
            _kw["device"] = _dev
        if _dt is not None:
            _kw["dtype"] = _dt
        if _kw:
            try:
                new_model = new_model.to(**_kw)
            except Exception as exc:                           # pragma: no cover
                LOGGER.warning("[V100-GQA] could not move copy to %s: %s", _kw, exc)

        n = patch_module_tree(new_model)
        if n == 0:                                             # 拷贝后展开失败，退回原对象
            LOGGER.warning("[V100-GQA] found GQA candidates but none matched the "
                           "expected projection layout; pass through unchanged")
            return (vae,)

        new_vae = copy.copy(vae)                               # 浅拷贝外壳，共享 device/dtype 配置
        new_vae.first_stage_model = new_model
        # decode 走 self.first_stage_model.decode(...)，不经过 patcher，
        # 所以替换模块树即可。这里给副本建一个独立 patcher（不动原 patcher），
        # 让 ComfyUI 的显存管理认识它，显存紧张时能正常 offload 而不是死占。
        # 注意 model_override 是二元组 (model, (backup, backup_buffers,
        # object_patches_backup, pinned))，直接传 Module 会在 model_override[0] 处抛异常。
        try:
            _p = getattr(vae, "patcher", None)
            if _p is not None and hasattr(_p, "get_clone_model_override"):
                _ovr = _p.get_clone_model_override()
                new_vae.patcher = _p.clone(model_override=(new_model, _ovr[1]))
        except Exception as exc:
            LOGGER.info("[V100-GQA] patcher clone skipped (%s: %s); copy stays resident",
                        type(exc).__name__, str(exc)[:60])

        try:
            vae._v100_gqa_cache = new_vae                       # 供下次执行直接复用
        except Exception:
            pass

        LOGGER.info("[V100-GQA] %d attention layers expanded GQA -> MHA on a copy "
                    "(fused SDPA enabled, original VAE untouched)", n)
        return (new_vae,)


NODE_CLASS_MAPPINGS = {
    "RelayVSRGQAFix": RelayVSRGQAFix,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RelayVSRGQAFix": "RelayVSR V100 Attention Fix",
}
