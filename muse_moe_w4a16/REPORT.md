# Current backend: persistent CUDA

On `optimize/persistent-cuda-moe`, `submission.py` now contains an attributed
persistent CUDA backend. See [PERSISTENT_CUDA.md](PERSISTENT_CUDA.md) for its
provenance, BF16-boundary fixes, limitations and GPU measurements.

The following sections preserve the previous Marlin backend's results. That
backend remains on `optimize/marlin-local-routing`.

# Improved routed-Marlin pipeline (previous backend)

Branch: `optimize/marlin-local-routing`, based on `optimize/routed-marlin`.

## Motivation and scope

The [H200 NVFP4 streaming notes](https://github.com/MrNirmata/h200-decode-notes/blob/main/notes/03-nvfp4-moe-streaming.md)
recommend attacking fixed per-call costs at small batch and tuning consumer
occupancy/tile shapes at larger batch. This change applies those lessons to the
existing Marlin pipeline instead of replacing its optimized GEMMs.

**This is a faster Marlin-based pipeline, not a new Marlin CUDA implementation.**
It does not implement the notes' combined 9216-byte weight/scale stages,
producer-warps, or new dequantization assembly. The installed Marlin Python API
does not expose its internal `thread_k`, `thread_n`, or occupancy parameters.

## Implemented changes

1. **Fused small-batch routing:** for at most 32 tokens, one Triton CTA maps
   global expert IDs, counts routes per local expert, computes padded offsets
   and route ranks, and emits sorted slots, block expert IDs and padded length.
   Nonlocal blocks are omitted. Valid and padding stores are disjoint. Duplicate
   expert routes retain their distinct flattened slot IDs. No host reads of
   route counts or input-derived caching are used.
2. **Fused small-batch post-expert normalization:** below 32 tokens, replace
   SGLang's multi-kernel PyTorch fallback with one masked Triton kernel. Each
   expert is normalized before routing weights are applied; nonlocal scratch
   is not loaded. Larger batches retain the existing fused SGLang norm kernel.
3. **EP8-specific token tiles:** use block height 8 through 32 tokens and 16
   above that, rather than increasing to 48/64 based on all global top-k routes.
   Smaller tiles reduce excess padded work. Tile 8 for every shape performed
   worse at the largest batch, so that configuration was rejected.
4. **Local metadata:** the remaining library alignment path is sized for the
   32 local experts rather than all 256 global experts.

Packed NVFP4 weights and their scales remain lossless in `prepare()`. Marlin
still performs both GEMMs with `use_fp32_reduce=True`, `use_atomic_add=False`,
and BF16 activation/output boundaries. Scratch aliasing is preserved.

## Official scored submission

Submitted commit `aad98b4` as job `6051d2d18d6341d7` on H200. All five
correctness tests and six benchmark cases passed, with no flags or rejections.
Aggregate speedup: **1.31055x**, 95% interval **[1.29090, 1.33021]**.
This agrees with the preceding off-board check. Leaderboard rank at retrieval:
**1025** (provisional).

| Load | SGLang | Submission | Speedup |
|---|---:|---:|---:|
| decode 1 | 79.12 us | 49.87 us | 1.586x |
| decode 16 | 192.27 us | 151.39 us | 1.270x |
| decode 32 | 200.00 us | 195.48 us | 1.023x |
| verify 1 | 147.28 us | 109.96 us | 1.339x |
| verify 16 | 468.75 us | 339.98 us | 1.379x |
| verify 32 | 633.63 us | 476.04 us | 1.331x |

Full profile: `profiles/submission.6051d2d18d6341d7.txt`.

## Official check measurements

Original routed-Marlin check `efcd6c0b0b5b4110` passed and reported aggregate
speedup 1.0016 versus the supplied SGLang baseline (effectively identical).

Final executable implementation: off-board check `9f3a1fbf1b484412` on H200,
all 5 correctness tests and all 6 scored shapes passed. Aggregate speedup:
**1.3119**, 95% interval **[1.2922, 1.3316]**. Correctness-test relative L2 errors
were 0.534–0.557%, below the task's 1.5% bound.

| Tokens | Load | Submission time | Paired speedup vs SGLang |
|---|---|---:|---:|
| 1 | decode 1 | 48.77 us | 1.591x |
| 16 | decode 16 | 149.33 us | 1.266x |
| 32 | decode 32 | 201.11 us | 1.027x |
| 8 | verify 1 | 107.10 us | 1.339x |
| 128 | verify 16 | 334.67 us | 1.385x |
| 256 | verify 32 | 469.04 us | 1.329x |

These are paired cold-L2 CUDA-graph results, not eager timings. Separate check
runs use different routes/weights and may differ in absolute time. The gain
at 32 tokens is comparatively small and should not be overinterpreted.

Additional checks:
- `16aa87685c2247cc`: fused small norm + local metadata, original tile heuristic;
  aggregate speedup 1.1317.
- `d93a9445937145c5`: tile height 8 everywhere; aggregate 1.1317.
- `ae36e309f20744fb`: tile height 16 everywhere; aggregate 1.2286.
- `7da4e11c75084a4a`: 8 for small batches / 32 for large; aggregate 1.2164.
- `1f771d82236342af`: compact small routing with 8/32 tiles; aggregate 1.2731.
- `0509b1571e404ae0`: final 8/16 tiles with non-EP GEMM specialization on the
  compact route path; aggregate 1.3081. No clear benefit versus retaining EP;
  rejected to avoid unnecessary specialization.

Reports are under `profiles/submission.<job>.txt`; a compact final result is
in `results/marlin-local-routing.json`. These checks preceded the scored
submission reported above.

## Local validation

Six CPU wiring tests pass; Ruff and `git diff --check` pass. The five optional
CUDA edge-case tests in `test_marlin_gpu.py` cover empty local routes, duplicate
routes/skew, graph replay with fresh routes, NaN-filled nonlocal scratch, and
normalization arithmetic. **They skip locally because PyTorch/CUDA is absent**;
they are not claimed as executed. Official checks above execute the submission
on GPU, including nearby and skewed test shapes, but are not a substitute for
running every optional edge test in a configured development environment.

## Historical starter and initial implementation

The remaining sections describe the original PyTorch starter and the previous
untuned Marlin implementation, not the current performance results.

# Understanding the starter expert loop

This section explains the **original PyTorch starter** loop in `kernel_fn()`.

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
