# okc: problem=muse_moe_w4a16
"""Column-major, Split-K routed GEMMs with tile-local NVFP4 decoding.

Adapted from https://pytorch.org/blog/accelerating-moe-model/ . Unlike
that FP16 example, partial sums live in FP32 scratch (not BF16 atomics),
weights remain packed NVFP4, and all activation boundaries remain BF16.
"""

from typing import TypedDict

import torch
import triton
import triton.language as tl
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    fused_post_expert_norm_weighted_sum,
)
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size


class Weights(TypedDict):
    w13: torch.Tensor          # uint8 [256, 2I, H/2], low nibble first
    w13_sf: torch.Tensor       # float8_e4m3fn [256, 2I, H/16]
    w13_gs: torch.Tensor       # float32 [256]
    w2: torch.Tensor           # uint8 [256, H, I/2]
    w2_sf: torch.Tensor        # float8_e4m3fn [256, H, I/16]
    w2_gs: torch.Tensor        # float32 [256]
    post_norm_w: torch.Tensor  # bfloat16 [H]


class Config(TypedDict):
    bs: int
    d_hidden: int
    d_expert: int
    n_experts: int
    local_start: int
    local_count: int
    top_k: int
    sharding: str


class Inputs(TypedDict):
    hidden_states: torch.Tensor  # bfloat16 [M, H]
    topk_ids: torch.Tensor       # int32 [M, top_k], global ids
    topk_weights: torch.Tensor   # float32 [M, top_k]
    config: Config


class State(TypedDict):
    w13: torch.Tensor
    w13_s: torch.Tensor
    w13_gs: torch.Tensor
    w2: torch.Tensor
    w2_s: torch.Tensor
    w2_gs: torch.Tensor
    expert_map: torch.Tensor
    norm_w: torch.Tensor
    local_start: int
    local_count: int
    d_expert: int


LOCAL_START = 0
LOCAL_COUNT = 32
POST_NORM_EPS = 1e-8


def prepare(weights: Weights) -> State:
    """Keep only the local bank, in the original lossless packed layout."""
    local = slice(LOCAL_START, LOCAL_START + LOCAL_COUNT)
    device = weights["w13"].device
    expert_map = torch.full(
        (weights["w13"].shape[0],), -1, dtype=torch.int32, device=device
    )
    expert_map[local] = torch.arange(LOCAL_COUNT, dtype=torch.int32, device=device)
    return {
        "w13": weights["w13"][local].contiguous(),
        "w13_s": weights["w13_sf"][local].contiguous(),
        "w13_gs": weights["w13_gs"][local].contiguous(),
        "w2": weights["w2"][local].contiguous(),
        "w2_s": weights["w2_sf"][local].contiguous(),
        "w2_gs": weights["w2_gs"][local].contiguous(),
        "expert_map": expert_map,
        "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16),
        "local_start": LOCAL_START,
        "local_count": LOCAL_COUNT,
        "d_expert": weights["w13"].shape[1] // 2,
    }


@triton.jit
def _routed_gemm(
    X, W, SF, GS, Sorted, Experts, Padded, Partial,
    SLOTS: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    TOP_K: tl.constexpr,
    SPLIT_K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
    BK: tl.constexpr,
):
    # Column-major: consecutive CTAs traverse routed M tiles before advancing
    # N, so all token tiles of an expert reuse the same weight columns in L2.
    # Compact the useful CTAs at the beginning of the launch rather than
    # interleaving every N tile with worst-case padding/empty expert blocks.
    active_m_tiles = tl.cdiv(tl.load(Padded), BM)
    if tl.program_id(0) >= active_m_tiles * tl.cdiv(N, BN):
        return
    tile_m = tl.program_id(0) % active_m_tiles
    tile_n = tl.program_id(0) // active_m_tiles
    split = tl.program_id(1)
    expert = tl.load(Experts + tile_m)
    if expert < 0:
        return
    rows = tile_m * BM + tl.arange(0, BM)
    slots = tl.load(Sorted + rows)
    cols = tile_n * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    gs = tl.load(GS + expert)
    acc = tl.full((BM, BN), 0, tl.float32)
    # Disjoint contiguous K partitions. The scratch reduction below avoids
    # nondeterministic atomics and rounds to BF16 only after all splits sum.
    for step in range(triton.cdiv(K, BK * SPLIT_K)):
        k = (split * triton.cdiv(K, BK * SPLIT_K) + step) * BK + ks
        a = tl.load(
            X + (slots[:, None] // TOP_K) * K + k[None, :],
            (slots[:, None] < SLOTS) & (k[None, :] < K), other=0,
        )
        packed = tl.load(
            W + expert * N * (K // 2) + cols[:, None] * (K // 2)
            + k[None, :] // 2,
            (cols[:, None] < N) & (k[None, :] < K), other=0,
        ).to(tl.int32)
        code = (packed >> ((k[None, :] % 2) * 4)) & 15
        mag = code & 7
        # E2M1: 0, .5, 1, 1.5, 2, 3, 4, 6; bit 3 is sign.
        value = tl.where(
            mag < 2, mag.to(tl.float32) * 0.5,
            (((mag // 2 + 126) << 23) | ((mag & 1) << 22))
            .to(tl.float32, bitcast=True),
        )
        value = tl.where((code & 8) != 0, -value, value)
        sf = tl.load(
            SF + expert * N * (K // 16) + cols[:, None] * (K // 16)
            + k[None, :] // 16,
            (cols[:, None] < N) & (k[None, :] < K), other=0.0,
        ).to(tl.float32)
        b = (value * (sf * gs)).to(tl.bfloat16)
        acc = tl.dot(a, tl.trans(b), acc)
    tl.store(
        Partial + split * SLOTS * N + slots[:, None] * N + cols[None, :],
        acc, (slots[:, None] < SLOTS) & (cols[None, :] < N),
    )


@triton.jit
def _reduce_swiglu(
    Partial, LocalIds, Inter,
    SLOTS: tl.constexpr, I: tl.constexpr, SPLIT_K: tl.constexpr,  # noqa: E741
    BLOCK: tl.constexpr,
):
    slot = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    splits = tl.arange(0, SPLIT_K)
    local = tl.load(LocalIds + slot) >= 0
    offset = splits[:, None] * SLOTS * (2 * I) + slot * (2 * I) + cols[None, :]
    gate = tl.sum(tl.load(Partial + offset, local & (cols[None, :] < I), other=0), 0)
    up = tl.sum(tl.load(Partial + offset + I, local & (cols[None, :] < I), other=0), 0)
    # Match both GEMM output rounding and the BF16 SwiGLU boundary.
    gate = gate.to(tl.bfloat16).to(tl.float32)
    up = up.to(tl.bfloat16).to(tl.float32)
    inter = gate / (1.0 + tl.exp(-gate)) * up
    tl.store(Inter + slot * I + cols, inter, cols < I)


@triton.jit
def _reduce_down(
    Partial, LocalIds, Y,
    SLOTS: tl.constexpr, H: tl.constexpr, SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    slot = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    splits = tl.arange(0, SPLIT_K)
    local = tl.load(LocalIds + slot) >= 0
    offset = splits[:, None] * SLOTS * H + slot * H + cols[None, :]
    value = tl.sum(tl.load(Partial + offset, local & (cols[None, :] < H), other=0), 0)
    tl.store(Y + slot * H + cols, value, cols < H)


def _split_k(num_tokens: int) -> int:
    # A conservative shape-only starting point; tune with cold-L2 graph replays.
    return 4 if num_tokens <= 32 else 1


def _launch_gemm(x, state, prefix, sorted_ids, experts, padded, slots, n, k, top_k, split_k):
    bm, bn, bk = 16, 64, 64
    m_tiles = min(slots, triton.cdiv(sorted_ids.shape[0], bm))
    partial = torch.empty((split_k, slots, n), dtype=torch.float32, device=x.device)
    _routed_gemm[(m_tiles * triton.cdiv(n, bn), split_k)](
        x, state[prefix], state[prefix + "_s"], state[prefix + "_gs"],
        sorted_ids, experts, padded, partial,
        SLOTS=slots, N=n, K=k, TOP_K=top_k,
        SPLIT_K=split_k, BM=bm, BN=bn, BK=bk,
        num_warps=4, num_stages=3,
    )
    return partial


def kernel_fn(data: Inputs, state: State) -> torch.Tensor:
    """Fresh device-side routes, two packed GEMMs, then per-expert norm/sum."""
    cfg = data["config"]
    if (cfg["local_start"], cfg["local_count"]) != (
        state["local_start"], state["local_count"]
    ):
        raise ValueError("prepare() packs only the task's EP8 experts 0-31")
    x = data["hidden_states"].contiguous()
    num_tokens, hidden = x.shape
    width = state["d_expert"]
    top_k = cfg["top_k"]
    slots = num_tokens * top_k
    local_ids = state["expert_map"][data["topk_ids"].to(torch.int32)].contiguous()
    weights = data["topk_weights"].to(torch.float32).contiguous()
    # Use the same block height as the tensor-core GEMMs. The aligner skips
    # nonlocal (-1) routes; empty/padded blocks are skipped in the GEMM.
    sorted_ids, experts, padded = moe_align_block_size(local_ids, 16, LOCAL_COUNT)
    split_k = _split_k(num_tokens)
    partial = _launch_gemm(
        x, state, "w13", sorted_ids, experts, padded,
        slots, 2 * width, hidden, top_k, split_k,
    )
    inter = torch.empty((slots, width), device=x.device, dtype=torch.bfloat16)
    _reduce_swiglu[(slots, triton.cdiv(width, 256))](
        partial, local_ids, inter,
        SLOTS=slots, I=width, SPLIT_K=split_k, BLOCK=256,
    )
    down = _launch_gemm(
        inter, state, "w2", sorted_ids, experts, padded,
        slots, hidden, width, 1, split_k,
    )
    y = torch.empty((num_tokens, top_k, hidden), device=x.device, dtype=torch.bfloat16)
    _reduce_down[(slots, triton.cdiv(hidden, 256))](
        down, local_ids, y,
        SLOTS=slots, H=hidden, SPLIT_K=split_k, BLOCK=256,
    )
    return fused_post_expert_norm_weighted_sum(
        y, weights, norm_weight=state["norm_w"], eps=POST_NORM_EPS,
        weight_first=True, slot_expert_ids=local_ids,
    )
