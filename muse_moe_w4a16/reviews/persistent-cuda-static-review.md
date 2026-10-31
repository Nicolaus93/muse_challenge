# Code Review — Integration Follow-up

**Reviewed:** Current `muse_moe_w4a16/submission.py`, with diagnostics/alternate source removed, BF16 helpers, zero-job guard, and guarded launcher.
**Verdict:** APPROVED for the requested static correctness review; GPU compilation/numerical/replay validation remains outstanding.

## Summary
No additional concrete blocking correctness or compile error was found in the serial H200 task path. This is not a GPU-tested approval: the remote check mentioned by the integrator was not queried, and local PyTorch/CUDA tooling remains unavailable.

## Findings

### [P2] Optional PDL launch branch still ignores launch errors
**File:** `muse_moe_w4a16/submission.py:2900-2903`
**Issue:** Both `cudaLaunchKernelEx` return values remain unchecked and this branch returns before `C10_CUDA_KERNEL_LAUNCH_CHECK()`. The ordinary launch and attribute checks are fixed. `PDL=False` makes this irrelevant to the current check/timing path.
**Suggested Fix:** Check each `cudaLaunchKernelEx` result with `C10_CUDA_CHECK` before returning, if retaining the optional PDL mode.

**Integration follow-up:** The optional PDL launch calls are now wrapped in
`C10_CUDA_CHECK`. Official check `ac6622cf9d104696` subsequently passed GPU
compilation, correctness and graph replay on H200; this review itself remains
static-only. Additional host-side cleanup is covered by the final check noted
in `PERSISTENT_CUDA.md`.

## Requested correctness checks

- **Zero local routes:** The `nj==0` guard at lines 2344-2347 is sufficient for the traced WGS empty-job path: it avoids `wgs_stage_at`/division, both producers still participate in their initialization joins, producer loops and phase loops are empty, and gather has zero rows. Final reduction sees `pos=-1` and writes zeros without accessing stale `ybuf`. TINY/MID helper paths likewise have zero units, but still execute both helper joins. No early return bypasses the final grid barrier/reset.
- **Persistent barrier:** Generation is read before any CTA can complete this launch's grid barrier. Each launch clears the opposite statistics copy/next grid counter; current statistics remain untouched until all final readers complete. Readiness/pool resets are after the grid barrier and before same-stream kernel completion. No new serial cross-replay race was established.
- **Cross-variant state:** TINY/MID and MAIN share the parity scheme and scratch prefix. WGS-only readiness is reset after WGS work completes; non-WGS launches do not consume that readiness. Fixup buffers are distinct from `ybuf` and `ss`. Shared `state` remains unsafe for concurrent graph/stream execution; serialize or give each concurrent execution separate scratch.
- **Rounding edits:** Active WGS norm epilogues now keep `gsc` separate until `down_norm_bf16`; no remaining precombined norm/global-scale occurrence was found in the active paths examined. The base-source-only MID build avoids the old alternate source's divergence.
- **Nearby shapes:** bs4 dispatches TINY, bs33/257 MAIN. Entry capacity remains sufficient; odd token counts still contain a whole number of int4 routing loads because each token has eight slots.
- **Allocation correction to the earlier review:** CUDA `FT_T` is **64**, including WGS; actual CUDA `JMAX` is **88**, not 136. Python's WGS padded-row and `doneAs` allocations use a conservative 136-job bound, which overallocates safely. The earlier report's assertion that WGS CUDA itself uses 136 jobs was incorrect.

## Validation performed
- Python AST parsing: passed.
- Extracted CUDA source preprocessing: passed for MAIN, TINY, TINY2, MID and fallback, each with and without PRMT scales (10 configurations). No unresolved `OKC_*` tokens remained in preprocessed code outside strings. This catches conditional/macro breakage, not CUDA type/PTX compilation errors.
- No implementation files changed; no `okc` invoked.
- GPU compile/replay testing unavailable locally: no PyTorch, `nvcc`, or `nvidia-smi`.

## Next validation priorities
1. Remote check result, especially compile/resource failures after the BF16 conversion additions.
2. On one serial state, alternate populated / all-nonlocal / populated routing, changing hidden states and route weights too, across at least four replays and both generation parities. Test MAIN bs33/257 and TINY bs4; directly check that the all-nonlocal output is zero because the supplied reference checker intentionally rejects wholly nonlocal cases rather than scoring them.
3. Alternate captured bs4, bs16, bs33, bs257 graphs serially on the same state; compare populated results against reference and include per-token zero-local rows.

## What's Good
- Ordinary launch errors and shared-memory attribute failures now fail promptly.
- The zero-job guard preserves barrier participation instead of returning from the persistent kernel early.
- One active CUDA source makes BF16 and synchronization changes apply to all variants.
