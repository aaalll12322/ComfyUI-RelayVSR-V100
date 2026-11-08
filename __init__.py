# ComfyUI-RelayVSR-V100
#
# 在 sm_70 上，PyTorch 的 cutlass fmha 只接受 MHA。RelayVSR 的 Flash 解码器用 GQA
# （8 heads / 2 kv heads），于是 SDPA 静默退化到 math 后端，实测慢约 8 倍
# （4.30T vs 34.36T）。
#
# 本插件不改全局函数、不重写上游 forward：只把 K/V 投影权重按 head 展开成 MHA，
# 并把 kv_groups 设为 num_heads，上游的 view(...) 自然成立，fused kernel 接管。
#
# 用法：工作流里加 "RelayVSR V100 Attention Fix" 节点，串在 VAE 与 VSR Decode 之间。
# 详见 README.md / README.zh-CN.md。
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
