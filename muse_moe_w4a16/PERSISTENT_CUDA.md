# Persistent CUDA NVFP4 MoE backend

Branch: `optimize/persistent-cuda-moe`.
Previous working backend: `optimize/marlin-local-routing`, scored submission
`6051d2d18d6341d7` (aggregate speedup 1.31055 versus SGLang).

## Provenance and scope

This is an adaptation of the public Apache-2.0 CUDA kernel from
[alygg77/okc-muse-spark-kernels](https://github.com/alygg77/okc-muse-spark-kernels),
which is built on [vjeux/okc-2026](https://github.com/vjeux/okc-2026).
Most of the scheduling, packing and CUDA implementation is upstream work, not
new work invented in this branch. Original attribution is retained in
`submission.py`; the Apache license and upstream notices are in `licenses/`.

Pinned upstream commit: `6cae49a639ee6d14b929a5fd4a63c627f96d7dc0`.
Imported path: `kernels/muse_moe_w4a16.py`.
Original file SHA-256:
`b33070898927b7b92df6c7a2515f9c02dd12723c43ca63682fa164e915dd4281`.

The H200 [streaming notes](https://github.com/MrNirmata/h200-decode-notes/blob/main/notes/03-nvfp4-moe-streaming.md)
and [Machete article](https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel)
provided motivation. Public X search returned only a logged-out shell, not
verified posts; no X-specific claims were used as implementation evidence.

## Kernel design retained from upstream

- One resident CTA per SM and one CUDA launch per chunk (at most 384 tokens).
- Routing rebuilt from fresh inputs inside every invocation, forming per-expert
  token lists and skinny/fat jobs; original input values are never cached.
- Skinny variants use BF16 `mma.sync`, with K divided across compute warps and
  a helper warp overlapping epilogue work with asynchronous weight copies.
- The larger variant uses register/shared BF16 WGMMA: dequantized weights are
  the register A operand; gathered BF16 activations are the shared B operand.
- Bit/byte-permuted packed codes and scale bytes stream together through
  shared-memory rings. Weights remain NVFP4; several lossless layout copies are
  retained, not expanded BF16 banks.
- KPERM3 permutes K coordinates consistently in codes, scales, activation
  gathering and gate/up row order, improving scale reuse.
- Gate/up plus SwiGLU, down projection plus normalization preparation, and
  expert-weighted output reduction execute within the persistent kernel.
- Global FP32 scratch remains for partial reductions and norm preparation.
  This is not a claim that every intermediate stays in registers/shared memory.

## Integration changes in this branch

1. **Explicit BF16 activation boundaries.** The imported source consumed FP32
   gate/up accumulators directly in SwiGLU and combined down accumulators with
   norm weights before rounding. `swiglu_bf16` now rounds both gate and up to
   BF16 first. Every down epilogue uses `down_norm_bf16`, rounding the complete
   down result before FP32 norm arithmetic, with the effective BF16 norm weight
   as in the SGLang baseline. The intermediate and final output remain BF16.
2. **Empty-route safety.** WGS range construction handles zero local jobs
   without reading an empty job's cost or dividing by zero. All CTAs still
   participate in barriers/reset and the final stage writes zeros.
3. **Single source of truth.** Removed upstream diagnostics, input-generation
   helpers, bandwidth benchmarks and a duplicate alternate CUDA source. The
   MID variant uses the same source as the other variants, without the
   separate helper-warp-rotation experiment.
4. **Launch contract/error handling.** Added CUDA device guards, input shape/
   dtype checks, shared-memory attribute checks, and launch error propagation.
   Hopper SM90 is required explicitly; there is no misleading non-Hopper path.
5. **Self-contained submission.** `load_inline` compiles only declared source;
   there are no local helper imports, network accesses or harness modifications.
   Caches contain compiled extension modules, not input-derived results.

The PRMT scale decode is selected only after checking the entire local weight
bank for positive normal E4M3 scale bytes. Other encodings select the LUT/
`mma.sync` fallback. Host inspection here is weight-only and in `prepare()`.
No FP32 reference computation or activation preparation occurs outside timing.

## Scored submission

Submitted implementation `cabcee2` as job `ab48b9ddea894287` on H200.
All five correctness tests and six benchmark cases passed, with no flags or
rejections. Aggregate speedup versus SGLang: **2.35454x**, 95% interval
**[2.31922, 2.38986]**.

Leaderboard at retrieval: **228 of 1256 ranked teams**, up from **1025** for the
previous Marlin submission (797 places gained). Rankings remain provisional.
Full scored profile: `profiles/submission.ab48b9ddea894287.txt`.

## Verified judge compiler

Check `e46df99945204d27` compiled a host-only metadata function in the CUDA
translation unit and printed its constants once from untimed `prepare()`:

```text
[cuda-toolchain] nvcc=13.0.88 cudart_header=13000
```

Thus the tested judge uses NVCC **13.0.88**, with CUDA runtime **headers 13.0**.
The header macro is not a measurement of the installed driver or loaded runtime.
The function returns `__CUDACC_VER_MAJOR__`, `__CUDACC_VER_MINOR__`,
`__CUDACC_VER_BUILD__`, and `CUDART_VERSION`; it is never called by `kernel_fn`.
All official correctness/benchmark cases passed, with no flags or rejections.
An initial precheck rejected the word `subprocess` in an explanatory comment;
removing only that comment wording resolved the false positive.

## Validation

Final off-board H200 check of implementation commit `cabcee2`:
`0f16573f571040a2`.
- All five correctness tests and all six benchmark shapes passed.
- No rejected or flagged static checks.
- Correctness-test relative L2 error: 0.445–0.458%, below 1.5%.
- Aggregate paired speedup versus SGLang: **2.36131x**.
- 95% interval: **[2.32589, 2.39673]**.
- Untimed preparation/compilation: about 182 seconds (300-second allowance).
- Each scored profile shows one persistent CUDA kernel invocation.

| Tokens | Load | Submission time | Paired speedup vs SGLang |
|---|---|---:|---:|
| 1 | decode 1 | 26.27 us | 3.055x |
| 16 | decode 16 | 95.54 us | 1.974x |
| 32 | decode 32 | 129.69 us | 1.652x |
| 8 | verify 1 | 63.63 us | 2.231x |
| 128 | verify 16 | 178.13 us | 2.639x |
| 256 | verify 32 | 213.07 us | 2.955x |

Full final profile: `profiles/submission.0f16573f571040a2.txt`.

The initial integration also passed all tests and benchmarks in check
`ac6622cf9d104696`: aggregate speedup 2.35177x, interval [2.31649, 2.38705].
The two runs agree within their reported intervals; the small difference is
not evidence that host-side cleanup improved performance. The subsequent scored
submission is recorded above.

Local validation: seven CPU contract/encoding tests pass; four optional Hopper
edge tests in `test_persistent_gpu.py` skip because PyTorch/CUDA is absent.
Those optional tests cover heavy single-expert routing, chunking, empty local
routes, graph replay with changing inputs/routes, and PRMT eligibility. They
are not claimed as executed. Official checks execute GPU correctness and graph
replays, including nearby/skewed cases, but not every optional edge scenario.
Static review is recorded in `reviews/persistent-cuda-static-review.md`.

## Limitations and next work

- Mutable scratch/barrier state is for serial launches on one stream. Concurrent
  graphs/streams need separate prepared state/scratch; sharing it is not safe.
- Width 4096, top-8, and experts 0–31 are explicit task constraints.
- Some shape-selected variants and cost tables were tuned upstream; their
  portability to different GPUs/router distributions is not established here.
- Tiny paths assume valid top-k routing (distinct expert IDs within a token).
- Keep compilation within the setup allowance; do not add many variants blindly.
- Inspect compiled register spills and stage timing before further tuning;
  deeper rings or more CTAs are not assumed to improve performance.
- Changes versus the previous Marlin result were measured in separate runs,
  not a paired direct comparison of those two submissions.
