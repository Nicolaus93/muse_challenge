# Column-major Split-K experiment

Branch: `optimize/column-major-splitk`.

The current `submission.py` implements the scheduling ideas from
[Accelerating MoE model inference with Locality-Aware Kernel Design](https://pytorch.org/blog/accelerating-moe-model/).
The Marlin implementation described below is historical and remains available
on `optimize/routed-marlin`.

## Implementation

- Both projections use a custom Triton NVFP4 GEMM. Contiguous CTAs traverse
  expert-grouped M tiles before advancing N (column-major scheduling).
- The device's actual padded route count compacts useful CTAs at the front of
  the launch. Routing and scheduling never read input values on the host.
- Decode-sized batches use four contiguous K partitions; larger batches use
  one. This is a shape-only initial heuristic, not an autotuned configuration.
- Split-K partials accumulate in FP32 scratch and reduce deterministically;
  there are no BF16 atomics or partially rounded GEMM outputs.
- The first reduction fuses SwiGLU, rounding gate/up to BF16 before activation
  and the intermediate to BF16 before the down projection.
- The second reduction rounds to BF16, followed by the existing per-expert
  normalization and routing-weighted sum. Nonlocal slots are masked before
  reading partial scratch and explicitly written as zero by reducers.
- `prepare()` preserves original packed NVFP4 weights and FP8/FP32 scales.
  Only tiles are decoded to BF16 inside the timed GEMM. No expanded weight
  bank or input-derived cache is retained.

## Validation and limitations

Seven CPU wiring/encoding tests pass. They do not execute Triton arithmetic.

Official off-board H200 check `18d326dad5444fc1` passed all five correctness
cases and all six benchmark cases, including CUDA graph replay. Test relative
L2 errors were 0.567–0.579%, below the 1.5% bound. No scored submission was made.
The first check exposed an FP8 masked-load constant typing issue, fixed by
using a floating-point zero.

**This implementation is experimental, not a speedup over Marlin.** The final
check reports aggregate speedup 0.1976 versus SGLang (roughly five times slower).
The saved profile is `profiles/submission.18d326dad5444fc1.txt`; the custom GEMMs
still dominate. Tile decoding/layout, redundant scale loads, launch occupancy,
and the Split-K heuristic need further work before replacing Marlin in a
performance-sensitive deployment. The article compares against older FP16
Triton kernels, not this already optimized NVFP4 Marlin baseline.

# Understanding the starter expert loop

This section explains the **original PyTorch starter** loop in `kernel_fn()`.
The following initial-optimization notes describe the prior Marlin version.

Before the loop:

- M is the number of tokens processed in one call—the batch size (config["bs"]).
- `x`: all token vectors, shape **[M, 4096]**.
- `ids`: each token’s 8 selected expert IDs, shape **[M, 8]**.
- `out`: an initially zero FP32 accumulator, shape **[M, 4096]**.

Each token is routed to all 8 selected experts, not just one. This GPU computes only the selected experts it owns. Its output is a partial sum; cross-GPU summation is outside this task.

## 1. Visit every local expert

```python
for e in range(cfg["local_start"], cfg["local_start"] + cfg["local_count"]):
```

Iterates over the 32 experts owned by this GPU—normally IDs **0–31**.

## 2. Find this expert’s routing weight for every token

```python
g = (data["topk_weights"] * (ids == e)).sum(-1)
```

- `ids == e` produces a Boolean **[M, 8]** mask.
- Multiplication keeps routing weights only where the expert ID equals `e`.
- `.sum(-1)` sums over the 8 slots, producing **[M]**.

For example:

```text
token's expert IDs:  [3,   7,   15,  ...]
routing weights:    [0.2, 0.1, 0.15, ...]
```

For expert `7`, this token gets `g = 0.1`. For an unselected expert, it gets `g = 0`.

## 3. Unpack the gate/up weights

```python
w13 = dequantize(
    weights["w13"][e],
    weights["w13_sf"][e],
    weights["w13_gs"][e],
).to(torch.bfloat16)
```

Turns expert `e`’s packed NVFP4 weights into a BF16 matrix of shape **[8192, 4096]**.

It contains two stacked matrices:

```text
first 4096 rows: gate weights
last 4096 rows:  up weights
```

Block scales (`w13_sf`) and the expert’s global scale (`w13_gs`) reconstruct the weight values.

### Number formats: NVFP4 and BF16

These are formats for storing numbers, with different trade-offs between precision and memory.

**BF16 (Brain Floating Point, 16 bits)** uses **2 bytes per number**:

- 1 sign bit.
- 8 exponent bits, controlling numerical range.
- 7 fraction bits, controlling precision.

BF16 supports roughly the same numerical range as FP32, but with less precision—roughly **2–3 decimal significant digits**. It is widely used for neural-network computation. In this task, token activations are BF16.

**NVFP4 (NVIDIA’s 4-bit floating-point format)** stores each weight’s code in **4 bits**:

- 1 sign bit.
- 2 exponent bits.
- 1 fraction bit.

The code represents one of these values:

```text
±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6
```

To represent useful weight magnitudes, NVFP4 also stores scales:

```text
weight = code_value × block_scale × global_scale
```

In this task, there is one FP8 scale per **16 weights**, plus an FP32 global scale per expert matrix. The FP4 code format is also called **E2M1**: two exponent bits and one fraction bit.

| Format | Storage per value | Role here |
|---|---:|---|
| FP32 | 4 bytes | Reference computation, global scales, output accumulation |
| BF16 | 2 bytes | Activations and intermediate matrix outputs |
| NVFP4 | ½ byte, plus scales | Compressed expert weights |

NVFP4 makes weights much smaller; BF16 provides more precision for computation. **W4A16** means **4-bit weights and 16-bit activations**.

### What dequantization does

`dequantize()` reconstructs numerical weight values from their compressed representation. NVFP4 stores each weight as a **4-bit code**, rather than a full BF16 or FP32 number.

Reconstruction uses:

```text
weight = decoded_4bit_value × block_scale × global_scale
```

For example, if a code decodes to `1.5`, the block scale is `128`, and the global scale is `0.0001`, the reconstructed weight is:

```text
1.5 × 128 × 0.0001 = 0.0192
```

In this code:

- `weights["w13"][e]` contains packed codes: **two weights per byte**.
- `weights["w13_sf"][e]` contains **one scale per group of 16 weights**.
- `weights["w13_gs"][e]` contains **one global scale for this expert’s gate/up matrix**.

`dequantize()` unpacks the codes, decodes them, and applies the scales, returning FP32 values. The following `.to(torch.bfloat16)` converts them to BF16 for matrix multiplication. The same process applies to `w2`.

Compression reduces GPU memory use and data movement. However, the starter expands entire matrices before using them. A faster kernel can reconstruct small tiles during multiplication, avoiding a large expanded matrix in GPU memory.

Dequantization does **not** recover information lost during quantization—it reconstructs the approximation represented by the stored codes and scales.

## 4. Compute gate and up projections together

```python
h = x @ w13.T
```

Matrix multiplication:

```text
[M, 4096] @ [4096, 8192] → [M, 8192]
```

For every token, the first half of `h` is its gate vector, and the second half is its up vector.

**Waste:** this computes expert `e` for every token, even those with `g = 0`.

## 5. Apply SwiGLU

```python
inter = torch.nn.functional.silu(h[:, :d_expert]) * h[:, d_expert:]
```

Equivalent to:

```python
gate = h[:, :4096]
up = h[:, 4096:]
inter = silu(gate) * up
```

Here, `SiLU(z) = z * sigmoid(z)`. The multiplication with `up` is elementwise. Result: **[M, 4096]**.

The gate smoothly controls how strongly each component of `up` passes through. It does not select an expert.

“Up” usually refers to expansion into a larger intermediate width, although the intermediate and input widths are equal in this task.

## 6. Unpack the down-projection weights

```python
w2 = dequantize(
    weights["w2"][e],
    weights["w2_sf"][e],
    weights["w2_gs"][e],
).to(torch.bfloat16)
```

Reconstructs this expert’s down-projection matrix, shape **[4096, 4096]**.

## 7. Project down and normalize

```python
y = post_expert_norm(inter @ w2.T, weights["post_norm_w"])
```

First:

```text
inter @ w2.T → [M, 4096]
```

Then, independently for each token, the starter’s `post_expert_norm()`:

1. Multiplies channels by `1 + post_norm_w` in FP32.
2. Rounds that scaled vector to BF16, then converts it back to FP32.
3. Divides by its root-mean-square magnitude, with a small epsilon for stability.

Conceptually:

```text
z = BF16(y * (1 + post_norm_w))
normalized = FP32(z) / sqrt(mean(FP32(z)²) + 1e-8)
```

The mean is over the 4096 components of each token’s expert output. Normalization happens separately for each expert output, before the weighted sum.

## 8. Add this expert’s weighted contribution

```python
out += g[:, None] * y
```

`g[:, None]` reshapes **[M]** into **[M, 1]**, so each token’s routing weight multiplies all 4096 components of its expert output.

- Selected expert: add its weighted output.
- Unselected expert: multiply by zero and add nothing.

After the loop, the function returns:

```python
return out.to(torch.bfloat16)
```

This produces the GPU’s partial sum as a BF16 **[M, 4096]** tensor. Tokens with no local routes have zero output.

## Where optimization starts

The key inefficiency is that steps 3–7 do substantial work even when step 8 discards the result.

Two main starting points:

1. **Compute only routed local token/expert pairs.** Keep routing on the GPU and compatible with CUDA graph capture; avoid host synchronization or Python branches based on input values.
2. **Avoid fully expanding weights before multiplication.** Use a packed-weight W4A16 kernel that dequantizes tiles internally. Retaining expanded BF16 weights in `prepare()` is prohibited by the task rules.

Study `prepare()` and `kernel_fn()` in `baseline.py` for the existing efficient pipeline:

```text
repack weights once
→ group routes by expert
→ packed-weight gate/up GEMM
→ SwiGLU
→ packed-weight down GEMM
→ fused normalization and weighted sum
```

The timed function includes this entire local computation—not just the final additions. Weight preprocessing in `prepare()` is untimed, and cross-GPU summation is excluded.

## Implemented initial optimization

Work branch: `optimize/routed-marlin`.

`submission.py` now uses installed SGLang/sgl_kernel operators directly. It does not import `baseline.py`, `reference.py`, or a local helper file, so `submission.py` remains the complete submission artifact.

### Changes from the starter

- **Untimed weight repacking:** `prepare()` repacks only experts 0–31 into Marlin's packed NVFP4 layout. It also creates the global-to-local expert map and Marlin workspace. Expanded BF16 expert matrices are not retained.
- **GPU-side routing:** each call maps global IDs to local indices, marking nonlocal routes with `-1`, then groups routes with `moe_align_block_size`. Routing is recomputed for fresh inputs, without host reads.
- **Packed GEMMs:** the gate/up and down projections use `moe_wna16_marlin_gemm`, which reconstructs weight tiles within the kernel and processes routed local pairs rather than every token through every expert.
- **BF16 activation path:** SwiGLU uses the installed masked `silu_and_mul` operator; matrix reductions retain the baseline's FP32 reduction settings.
- **Fused final stage:** `fused_post_expert_norm_weighted_sum` normalizes each expert output, applies routing weights, and sums, while masking nonlocal slots. Routing weights are not applied inside the GEMMs.
- **Scratch reuse:** after SwiGLU consumes the gate/up result, the down projection reuses that storage on the same CUDA stream.

The implementation deliberately follows the supplied SGLang baseline's tiling heuristic and operator configuration. This removes the starter's obvious inefficiencies, but **does not establish a speedup over SGLang**. Further tuning should follow actual CUDA-graph measurements, not guesses based on eager timings.

The prepared bank is intentionally limited to the task's EP8 partition, experts 0–31. Other partitions raise a configuration error rather than silently using incorrect weights.

### Validation performed

```bash
python3 -B -m unittest discover -s muse_moe_w4a16 -p 'test_*.py' -v
ruff check muse_moe_w4a16/submission.py muse_moe_w4a16/test_submission_contract.py
git diff --check
```

Results: **6 contract tests passed**, Ruff passed, and no whitespace errors were reported. All Python files were also compiled from source with Python's `compile()` for syntax validation, without generating bytecode artifacts.

The tests use mocked tensor/operators to check repacking inputs, local routing masks, fresh routing on successive calls, scratch/output dimensions, operator order, FP32 reduction flags, post-normalization weighting, and scored/nearby batch sizes. They also check the submission marker and absence of prohibited reference imports or host tensor reads. **These are wiring tests, not GPU numerical correctness tests.**

### Validation still required

The development environment has Python 3, but no PyTorch, Triton, SGLang, CUDA tooling, or `okc` CLI. No GPU execution, numerical error measurement, CUDA graph capture/replay, or performance benchmark was possible. No official submission was made.

In a configured environment, run:

```bash
cd muse_moe_w4a16
okc check submission.py --json
```

This is the decisive check of the installed operator API, numerical tolerances (including skewed routing), graph compatibility, and performance against SGLang. Inspect its saved profile before choosing the next optimization. No measured speedup is claimed at this stage.
