# okc: problem=muse_moe_w4a16
"""Routed NVFP4 experts using SGLang's packed-weight Marlin operators.

Unlike the PyTorch starter, GEMMs visit only routed local token/expert pairs
and dequantize weight tiles inside the matrix kernel. prepare() repacks the
32 local experts once; no expanded BF16 weight matrices are retained.

Keep Marlin's optimized packed GEMMs, but fuse small-batch route preparation
and post-expert normalization and use smaller EP8 token tiles. See REPORT.md
for cold-L2 CUDA-graph measurements against the supplied SGLang baseline.
"""

from types import SimpleNamespace
from typing import TypedDict

import torch
import triton
import triton.language as tl
from sgl_kernel.scalar_type import scalar_types
from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    fused_post_expert_norm_weighted_sum,
)
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import (
    prepare_moe_nvfp4_layer_for_marlin,
)


class Weights(TypedDict):
    w13: torch.Tensor          # uint8 [256, 2I, H/2], gate rows [0, I) then up rows [I, 2I), packed E2M1
    w13_sf: torch.Tensor       # float8_e4m3fn [256, 2I, H/16]
    w13_gs: torch.Tensor       # float32 [256]
    w2: torch.Tensor           # uint8 [256, H, I/2]
    w2_sf: torch.Tensor        # float8_e4m3fn [256, H, I/16]
    w2_gs: torch.Tensor        # float32 [256]
    post_norm_w: torch.Tensor  # bfloat16 [H], the post-expert RMSNorm's weight (applied as 1 + w)


class Config(TypedDict):
    bs: int           # tokens, M
    d_hidden: int     # H = 4096, the width the MoE block runs its experts at
    d_expert: int     # I = 4096
    n_experts: int    # 256 experts in the model
    local_start: int  # this GPU computes experts [local_start, local_start + local_count)
    local_count: int  # 32: one GPU under 8-way expert parallelism
    top_k: int        # 8
    sharding: str     # "ep8"


class Inputs(TypedDict):
    hidden_states: torch.Tensor  # bfloat16 [M, H]
    topk_ids: torch.Tensor       # int32 [M, 8], global expert ids (0-255)
    topk_weights: torch.Tensor   # float32 [M, 8]
    config: Config


class State(TypedDict):
    w13: torch.Tensor
    w13_s: torch.Tensor
    w13_gs: torch.Tensor
    w2: torch.Tensor
    w2_s: torch.Tensor
    w2_gs: torch.Tensor
    expert_map: torch.Tensor
    workspace: torch.Tensor
    norm_w: torch.Tensor
    n_experts: int
    local_start: int
    local_count: int
    d_hidden: int
    d_expert: int


LOCAL_START = 0
LOCAL_COUNT = 32
POST_NORM_EPS = 1e-8


def prepare(weights: Weights) -> State:
    """Repack the EP8 bank once, preserving the packed NVFP4 representation.

    The task places experts 0-31 on this GPU. No routing or activation data is
    available here or cached in state. Marlin's helper also permutes scales
    into the layout its GEMMs expect.
    """
    local = slice(LOCAL_START, LOCAL_START + LOCAL_COUNT)
    d_expert = weights["w13"].shape[1] // 2
    n_experts = weights["w13"].shape[0]
    device = weights["w13"].device
    layer = SimpleNamespace(
        quant_config=SimpleNamespace(group_size=16),
        moe_runner_config=SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=d_expert,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"][local].contiguous(),
        w2_weight=weights["w2"][local].contiguous(),
        w13_weight_scale=weights["w13_sf"][local].contiguous(),
        w2_weight_scale=weights["w2_sf"][local].contiguous(),
        w13_weight_scale_2=weights["w13_gs"][local].contiguous(),
        w2_weight_scale_2=weights["w2_gs"][local].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)

    # Global IDs map to local bank indices; -1 marks routes owned elsewhere.
    expert_map = torch.full((n_experts,), -1, dtype=torch.int32, device=device)
    expert_map[local] = torch.arange(LOCAL_COUNT, dtype=torch.int32, device=device)
    return {
        "w13": layer.w13_weight.data,
        "w13_s": layer.w13_weight_scale.data,
        "w13_gs": layer.w13_weight_scale_2.data,
        "w2": layer.w2_weight.data,
        "w2_s": layer.w2_weight_scale.data,
        "w2_gs": layer.w2_weight_scale_2.data,
        "expert_map": expert_map,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
        # Match SGLang's BF16 effective post-expert norm weight.
        "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16),
        "n_experts": n_experts,
        "local_start": LOCAL_START,
        "local_count": LOCAL_COUNT,
        "d_hidden": weights["w2"].shape[1],
        "d_expert": d_expert,
    }


def _block_size_m(num_tokens: int, top_k: int, local_count: int) -> int:
    """Measured EP8 tiles: reduce padding without excessive weight rereads."""
    return 8 if num_tokens <= 32 else 16


@triton.jit
def _small_batch_routes(
    GlobalIds, LocalIds, Sorted, BlockExperts, Padded,
    SLOTS: tl.constexpr, LOCAL_START: tl.constexpr, LOCAL_COUNT: tl.constexpr,
    BM: tl.constexpr, BLOCK_SLOTS: tl.constexpr, BLOCK_EXPERTS: tl.constexpr,
    BLOCKS_PER_EXPERT: tl.constexpr,
):
    """Map, count, prefix-sum and pad a small routing table in one CTA."""
    slots = tl.arange(0, BLOCK_SLOTS)
    experts = tl.arange(0, BLOCK_EXPERTS)
    ids = tl.load(GlobalIds + slots, slots < SLOTS, other=-1) - LOCAL_START
    local = (ids >= 0) & (ids < LOCAL_COUNT) & (slots < SLOTS)
    tl.store(LocalIds + slots, tl.where(local, ids, -1), slots < SLOTS)
    matches = (experts[:, None] == ids[None, :]) & local[None, :]
    counts = tl.sum(matches.to(tl.int32), 1)
    padded_counts = tl.cdiv(counts, BM) * BM
    offsets = tl.cumsum(padded_counts, 0) - padded_counts
    ranks = tl.cumsum(matches.to(tl.int32), 1) - 1
    tl.store(Sorted + offsets[:, None] + ranks, slots[None, :], matches)
    # Padding stores are disjoint from valid-route stores. No scratch clear or
    # inter-CTA synchronization is required, including the all-nonlocal case.
    pad = tl.arange(0, BM)
    tl.store(
        Sorted + offsets[:, None] + counts[:, None] + pad[None, :], SLOTS,
        pad[None, :] < (padded_counts - counts)[:, None],
    )
    blocks = tl.arange(0, BLOCKS_PER_EXPERT)
    tl.store(
        BlockExperts + offsets[:, None] // BM + blocks[None, :], experts[:, None],
        blocks[None, :] < (padded_counts // BM)[:, None],
    )
    tl.store(Padded, tl.sum(padded_counts, 0))


def _routes(global_ids, state, block_size):
    tokens, top_k = global_ids.shape
    slots = tokens * top_k
    if tokens > 32:
        local_ids = state["expert_map"][global_ids.to(torch.int32)].contiguous()
        sorted_ids, experts, padded = moe_align_block_size(
            local_ids, block_size, state["local_count"]
        )
        return local_ids, sorted_ids, experts, padded
    device = global_ids.device
    count = state["local_count"]
    capacity = triton.cdiv(slots + count * (block_size - 1), block_size) * block_size
    local_ids = torch.empty((tokens, top_k), device=device, dtype=torch.int32)
    sorted_ids = torch.empty((capacity,), device=device, dtype=torch.int32)
    experts = torch.empty((capacity // block_size,), device=device, dtype=torch.int32)
    padded = torch.empty((1,), device=device, dtype=torch.int32)
    _small_batch_routes[(1,)](
        global_ids, local_ids, sorted_ids, experts, padded,
        SLOTS=slots, LOCAL_START=state["local_start"], LOCAL_COUNT=count,
        BM=block_size, BLOCK_SLOTS=triton.next_power_of_2(slots),
        BLOCK_EXPERTS=triton.next_power_of_2(count),
        BLOCKS_PER_EXPERT=triton.next_power_of_2(triton.cdiv(slots, block_size)),
        num_warps=4,
    )
    return local_ids, sorted_ids, experts, padded


@triton.jit
def _small_batch_norm_sum(
    Y, Scores, Norm, LocalIds, Out,
    H: tl.constexpr, TOP_K: tl.constexpr, BLOCK_H: tl.constexpr,
    EPS: tl.constexpr,
):
    """Avoid SGLang's multi-kernel PyTorch fallback below 32 tokens."""
    token = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    norm = tl.load(Norm + cols, cols < H, other=0).to(tl.float32)
    acc = tl.full((BLOCK_H,), 0, tl.float32)
    for route in tl.static_range(TOP_K):
        slot = token * TOP_K + route
        # Marlin leaves nonlocal scratch unwritten. Branch before loading it.
        if tl.load(LocalIds + slot) >= 0:
            y = tl.load(Y + slot * H + cols, cols < H, other=0).to(tl.float32)
            z = y * norm
            inv_rms = tl.rsqrt(tl.sum(z * z, 0) / H + EPS)
            score = tl.load(Scores + slot)
            acc += (z * inv_rms) * score
    tl.store(Out + token * H + cols, acc, cols < H)


def _post_norm_sum(y, scores, norm_w, local_ids):
    tokens, top_k, hidden = y.shape
    if tokens >= 32:
        return fused_post_expert_norm_weighted_sum(
            y, scores, norm_weight=norm_w, eps=POST_NORM_EPS,
            weight_first=True, slot_expert_ids=local_ids,
        )
    out = torch.empty((tokens, hidden), device=y.device, dtype=y.dtype)
    _small_batch_norm_sum[(tokens,)](
        y, scores, norm_w, local_ids, out,
        H=hidden, TOP_K=top_k, BLOCK_H=triton.next_power_of_2(hidden),
        EPS=POST_NORM_EPS, num_warps=8, num_stages=2,
    )
    return out


def kernel_fn(data: Inputs, state: State) -> torch.Tensor:
    """Compute this GPU's routed expert contribution as BF16 [M, H].

    All routing decisions stay on device. Allocations depend only on shapes;
    nonlocal slots may be unwritten scratch and must be masked by every
    consumer. Do not move routing weights into either GEMM: normalization
    must occur before the weighted sum.
    """
    cfg = data["config"]
    if (cfg["local_start"], cfg["local_count"]) != (
        state["local_start"], state["local_count"]
    ):
        raise ValueError("prepare() packs only the task's EP8 experts 0-31")

    x = data["hidden_states"].contiguous()
    num_tokens, size_k = x.shape
    size_n = state["d_expert"]
    top_k = cfg["top_k"]
    gemm1_n = 2 * size_n
    topk_weights = data["topk_weights"].to(torch.float32).contiguous()
    block_size_m = _block_size_m(num_tokens, top_k, state["local_count"])
    local_ids, sorted_ids, block_experts, n_padded = _routes(
        data["topk_ids"].contiguous(), state, block_size_m
    )

    # Keep the baseline's scratch aliasing: h is dead once SwiGLU completes,
    # so the down projection can reuse its storage on the same CUDA stream.
    slots = num_tokens * top_k
    inter = torch.empty((slots, size_n), device=x.device, dtype=x.dtype)
    scratch = torch.empty(
        (slots * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype
    )
    h = scratch[: slots * gemm1_n].view(slots, gemm1_n)
    y = scratch[: slots * size_k].view(slots, size_k)
    common = dict(
        moe_block_size=block_size_m,
        mul_topk_weights=False,
        is_ep=True,
        b_q_type=scalar_types.float4_e2m1f,
        is_k_full=True,
        use_atomic_add=False,
        use_fp32_reduce=True,
        is_zp_float=False,
    )
    h = moe_wna16_marlin_gemm(
        x, h, state["w13"], None, state["w13_s"], state["w13_gs"],
        None, None, None, state["workspace"],
        sorted_ids, block_experts, n_padded, topk_weights,
        top_k=top_k, size_m=num_tokens, size_n=gemm1_n, size_k=size_k,
        **common,
    )
    silu_and_mul(h, inter, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        inter, y, state["w2"], None, state["w2_s"], state["w2_gs"],
        None, None, None, state["workspace"],
        sorted_ids, block_experts, n_padded, topk_weights,
        top_k=1, size_m=slots, size_n=size_k, size_k=size_n,
        **common,
    ).view(num_tokens, top_k, size_k)
    return _post_norm_sum(y, topk_weights, state["norm_w"], local_ids)
