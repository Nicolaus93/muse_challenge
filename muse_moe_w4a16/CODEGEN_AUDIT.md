# CUDA code-generation audit (no GPU timing)

Branch: `optimize/cuda-codegen-audit`.
The production `submission.py` is unchanged. This audit does not make an
`okc check` or scored submission and does not change the leaderboard result.

## Compiler and method

- Judge probe: NVCC macros **13.0.88**, runtime-header macro **13000**.
- Compiler Explorer: `nvcc130u1` (label NVCC 13.0.1), independently checked with
  `#pragma message`: its NVCC macros also report **13.0.88**. `nvcc130u2` reports
  the same macros; the label `nvcc130` instead reports **13.0.48**, so the CUDA
  release label alone is insufficient to match this compiler build.
- Target: **sm_90a**, `-O3`, C++17, the submission's active defines, PRMT scales,
  the three-stage settings, and PyTorch's standard explicit CUDA flags.
- Extract `CUDA_SRC` without importing PyTorch. Replace only the PyTorch host
  wrapper with a plain CUDA launcher that instantiates the original kernels.
  Shared-memory structures, launch bounds and kernel bodies are retained.
- Compiler Explorer adds service defaults such as `--keep-device-functions`.
  The resulting code is a reproducible device-code inspection, **not a binary
  dump of the judge's exact extension**. Matching NVCC macros does not prove
  that every toolkit component/host wrapper/environment is identical.
- No kernels are executed by this tool; resource counts and static instruction
  counts are not runtime utilization, latency or speedup measurements.

Reproducer: `tools/inspect_cuda_codegen.py` (Python standard library only).
This explicitly sends public source to Compiler Explorer's third-party API.
Attribution from the submission is retained in newly generated source copies.
No source extraction imports or executes `submission.py`.

```bash
python3 muse_moe_w4a16/tools/inspect_cuda_codegen.py \
  --variant mid --output /tmp/muse-codegen/mid

python3 muse_moe_w4a16/tools/inspect_cuda_codegen.py \
  --variant mid --experiment helper-power2 \
  --output /tmp/muse-codegen/mid-helper-power2
```

## Baseline compilation results

| Active variant | Dispatched token counts | Registers/thread | Spill stores / loads | Status |
|---|---|---:|---:|---|
| tiny | 1–4 | 149 | 0 / 0 bytes | full kernel compiled |
| tiny2 | 5–8 | 149 | 0 / 0 bytes | full kernel compiled |
| mid | 9–16 | 153 | 0 / 0 bytes | full kernel compiled |
| main WGMMA | 17+ | unknown | unknown | service compile timeout |

Both `<false>` and `<true>` entries compile with matching resource counts for
these small variants; their real small-batch launches use `<false>`. Resource
reports show zero stack frames. Compiler Explorer's returned SASS listing may
deduplicate identical kernel bodies; instruction counts below describe that
returned static listing, not sums over executions or repeated loop iterations.

The full main variant exceeded the public service's processing-time limit,
including a retry that instantiated only the `<false>` entry. **No claim is
made about main's register count or spills.** Host-only resource probes for
main compile successfully and return **576 threads**, `sizeof(Smem)=22232`,
and a dynamic shared-memory request of **231168 bytes** (225.75 KiB).
This is a launch-configuration constant, not a full WGMMA assembly result.
The LUT/fallback variant was not compiled in this audit.

Compiler reports are in `codegen_audit/*.json`; representative baseline SASS
is retained as `codegen_audit/{tiny,tiny2,mid}.sass`. Full service responses and
additional experimental SASS are under `/tmp/muse-codegen/` on this machine.
The tool can regenerate them; the large raw JSON responses are not committed.

## Findings and candidate improvements

### 1. No spill problem in the inspected small variants

The compiled small variants have high register counts but **zero spills**.
Do not apply `maxrregcount` blindly. It may add spilling, while the current grid
already launches only one CTA per SM and shared-memory reservations also prevent
a straightforward occupancy increase.

The BF16 conversions compile to native **`F2F.BF16.F32`** and
**`F2FP.BF16.F32.PACK_AB`**, not a software conversion loop. The compute body uses
**`HMMA.16816.F32.BF16`**. Required BF16 rounding must not be removed to save
instructions.

### 2. Confirmed unnecessary integer division in the helper loop

`ws_helper` selects `rbs` dynamically between 256 and 512, then computes:

```cpp
const int j = njF + v / rbs, rb = v - (v / rbs) * rbs;
```

The baseline SASS actually contains **`MUFU.RCP` plus multiply-high/correction
instructions** for this division. The unit index `v` is nonnegative, and the
divisors are exact powers of two. This inspection-only candidate preserves the
same job/block arithmetic without changing synchronization or accumulation:

```cpp
const int j = njF + (v >> (PB ? RB_B_LOG : RB_A_LOG));
const int rb = v & (rbs - 1);
```

Compiled MID candidate (`--experiment helper-power2`):
- helper-loop reciprocal disappears;
- static listing drops from **4741 to 4736 instructions**;
- multiply-high instruction count drops from 4 to 2;
- registers remain 153 and spills remain zero.

See `codegen_audit/mid-helper-power2.sass`. This is a confirmed code-generation
improvement, **not a measured runtime speedup**. Full helper phase-specialization
is another possibility, but increases code size and is not implemented here.

### 3. Small variants reserve a fat pipeline they cannot execute

The host `okc_smem_bytes()` always reserves the maximum of the skinny and fat
pipeline sizes. Yet tiny/tiny2 are skinny-only, and MID splits all jobs into
at-most-eight-route pieces. In addition, their shared routing arrays are sized
for 384 tokens despite their much smaller dispatch bounds.

The inspection-only `compact-tiny` candidate:
- sizes only shared routing arrays for <=8 tokens in tiny/tiny2 and <=16 in MID;
- preserves all `EMAX`-dependent **global scratch offsets, clears and parity**;
- reserves the skinny ring only when `OKC_TINY` is active.

Compiled MID reservation falls from **214784 to 132480 bytes**
(209.75 to 129.375 KiB), with registers still 153 and zero spills. The
instruction count stays 4741. Source-level array accesses stay within the
existing dispatch bounds; no GPU correctness/replay test was run for this
candidate.

This does **not** permit two CTAs/SM as-is (both the reservation and registers
remain too large, and the grid size is unchanged). A smaller reservation might
help shared/L1 partitioning; whether it changes the effective cache carveout
or latency requires real-device measurement. Do not infer speedup from bytes.

### 4. Precompute the effective BF16 norm weight

The current kernel repeatedly forms `1 + post_norm_w` and rounds it to BF16.
A possible weight-only preparation step is to store that effective BF16 norm
weight, as the prior Marlin pipeline already did, and remove redundant adds/
conversions from the CUDA epilogue.

The MID inspection-only `effective-norm` candidate reduces scalar BF16
conversion instructions from 16 to 12 and `FADD` from 118 to 114 in the returned
static listing. Registers stay 153; total instruction count only falls by 3.
It requires the matching `prepare()` change, which has **not** been applied to
the production submission. Expected benefit is modest and must be measured.

### 5. Packed SwiGLU conversion is not an obvious win

Using `__floats2bfloat162_rn(gate, up)` replaces some scalar conversions with
packed conversions, but the MID candidate still has **4741 static instructions,
153 registers and zero spills**. Register unpacking/moves offset the apparent
conversion saving. A shorter-looking intrinsic does not demonstrate a win.

### 6. Several tuning flags are dead

A combined MID build changing `OKC_NACC` from 4 to 2, `OKC_EPI4` from 0 to 1,
and the compiler define `OKC_DQ7` from 1 to 0 produced **identical SASS
instruction text and resource usage**:

```text
d3009f67363d02c3d3f6cbb7be1da45825a9e2eaa03ea11b8a7ca5f67427b49a
```

Source inspection confirms that `OKC_EPI4` is never consumed, `OKC_NACC` only
has an unused default declaration, and DQ7 decode is hardcoded in CUDA.
**Python's `DQ7` switch is different:** it changes weight packing and must not
be disabled just because the compiler define is inert.

Further source-only findings:
- `OKC_CPF` does not affect the active WS path in tiny2/MID.
- MAIN's FAST1 setting is redundant because that optimization requires TINY.
- `OKC_WSBL=2` is boolean; `OKC_WSBLK` is the actual tail-unit count.
- MAIN's WGMMA ring sizes and `OKC_WGSGS` are hardcoded inside the source;
  passing a same-named define is not a reliable way to tune them.

## Recommended next GPU experiments

1. Benchmark **helper power-of-two indexing** independently.
2. Benchmark **compact tiny shared routing/pipeline reservation** independently,
   including bs4/8/16, empty routes, fresh-input graph replays and nearby shapes.
3. Try prepared effective norm weights if profiling still shows epilogue cost.
4. For MAIN, use an unrestricted CUDA compiler/disassembler or export assembly
   and ptxas statistics from the real judge build before claiming spill fixes.

Keep the current scored kernel unchanged until a candidate passes numerical
checks and paired cold-L2 graph timings. None of the inspection candidates has
been promoted into `submission.py` or measured with `okc`.

## Local checks

Four offline extraction/parser/experiment tests pass. Existing submission
contract tests remain unchanged. Ruff and `git diff --check` pass. Tests do not
execute CUDA or call Compiler Explorer.
