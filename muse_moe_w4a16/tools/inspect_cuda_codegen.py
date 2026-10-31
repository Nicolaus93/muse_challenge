"""Compile extracted device code on Compiler Explorer without local CUDA/PyTorch.

The input is sent to a public third-party compilation service. Use only code
that may be shared. This tool never executes the CUDA kernel or changes the
submission. Assembly is for device-code inspection, not runtime benchmarking.
"""

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import urllib.request


VARIANTS = {
    "main": "MAIN_DEFINES",
    "tiny": "TINY_DEFINES",
    "tiny2": "TINY2_DEFINES",
    "mid": "MID_DEFINES",
    "fallback": "MAIN_FALLBACK_DEFINES",
}
COMMON_FLAGS = [
    "-O3", "-std=c++17", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo",
    "-Xptxas=-v,--warn-on-spills",
    "-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__", "-D__CUDA_NO_HALF2_OPERATORS__",
    "--expt-relaxed-constexpr", "-DOKC_NS=3", "-DOKC_NSF=3", "-DOKC_DQ7=1",
]
LAUNCHER = r'''
// Only the PyTorch host wrapper is replaced. Both original kernel templates
// are instantiated, with the same launch bounds, block size and dynamic smem.
extern "C" void inspect_launch(okc::Params p, cudaStream_t stream, int grid) {
  const size_t smem = okc_smem_bytes();
  if (p.M >= OKC_BIGM)
    okc::moe_fused<true><<<grid, okc::NTL, smem, stream>>>(p);
  else
    okc::moe_fused<false><<<grid, okc::NTL, smem, stream>>>(p);
}
'''


def extract(source, variant):
    """Read literal constants only; do not import or execute the submission."""
    literals = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"CUDA_SRC", *VARIANTS.values()}:
                    literals[target.id] = ast.literal_eval(node.value)
    cuda = literals["CUDA_SRC"]
    device = cuda.split("#include <torch/extension.h>", 1)[0]
    smem = cuda.split("static size_t okc_smem_bytes()", 1)[1].split("static int g_okc_pdl", 1)[0]
    notices = []
    for line in source.splitlines()[1:]:
        if line.startswith("#"):
            notices.append("//" + line[1:])
        elif line.strip():
            break
    notice = "\n// Source attribution retained from submission.py:\n" + "\n".join(notices) + "\n"
    return device + "\nstatic size_t okc_smem_bytes()" + smem + LAUNCHER + notice, list(literals[VARIANTS[variant]])


def summarize(result):
    stderr = "\n".join(line.get("text", "") for line in result.get("stderr", []))
    resources, current = [], None
    for line in stderr.splitlines():
        entry = re.search(r"Compiling entry function '([^']+)' for '([^']+)'", line)
        if entry:
            current = {"entry": entry[1], "architecture": entry[2]}
            resources.append(current)
        if current is None:
            continue
        spills = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", line)
        if spills:
            current.update(stack_bytes=int(spills[1]), spill_store_bytes=int(spills[2]), spill_load_bytes=int(spills[3]))
        registers = re.search(r"Used (\d+) registers", line)
        if registers:
            current["registers"] = int(registers[1])
            current["resource_line"] = line.strip()
    assembly = {}
    for device, output in result.get("devices", {}).items():
        if not device.startswith("SASS"):
            continue
        lines, counts = [], Counter()
        for entry in output.get("asm", []):
            if not entry.get("opcodes"):
                continue
            text = entry["text"].strip()
            lines.append(text)
            instruction = re.match(r"(?:@\S+\s+)?([A-Z][A-Z0-9_.]*)\b", text)
            if instruction:
                counts[instruction[1]] += 1
        assembly[device] = {
            "instruction_count_all_functions": len(lines),
            "sha256_instruction_text": hashlib.sha256("\n".join(lines).encode()).hexdigest(),
            "instruction_histogram_all_functions": dict(counts.most_common()),
        }
    host_resources = {}
    probe = None
    for entry in result.get("asm", []):
        text = entry.get("text", "").strip()
        if text.startswith("inspect_") and text.endswith(":"):
            probe = text[:-1]
        immediate = re.search(r"mov[lq]\s+\$(\d+),\s*%[er]ax", text)
        if probe and immediate:
            host_resources[probe] = int(immediate[1])
            probe = None
    return {"host_resources": host_resources,
            "code": result.get("code"), "timed_out": result.get("timedOut"),
            "truncated": result.get("truncated"), "resources": resources, "assembly": assembly,
            "stderr": stderr, "device_outputs": list(result.get("devices", {}))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, default=Path(__file__).resolve().parents[1] / "submission.py")
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--compiler", default="nvcc130u1", help="Compiler Explorer compiler ID")
    parser.add_argument("--define", action="append", default=[], help="Override one macro, e.g. OKC_NACC=2")
    parser.add_argument("--no-prmt", action="store_true", help="Compile LUT scale decoding instead of PRMT")
    parser.add_argument("--entry", choices=("both", "small", "big", "resources"), default="both",
                        help="Select kernel entry, or host-only resource constants")
    parser.add_argument("--experiment", choices=("none", "effective-norm", "packed-swiglu", "compact-tiny", "helper-power2"), default="none",
                        help="Inspect a source candidate without modifying the submission")
    parser.add_argument("--output", type=Path, required=True, help="Directory for source, full response, and summary")
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()
    if args.variant == "main" and args.no_prmt:
        parser.error("main WGMMA requires PRMT; use --variant fallback for the active LUT configuration")
    source = args.submission.read_text()
    cuda, defines = extract(source, args.variant)
    if args.entry == "resources":
        probes = '''
extern "C" size_t inspect_dynamic_smem() { return okc_smem_bytes(); }
extern "C" size_t inspect_static_struct() { return sizeof(okc::Smem); }
extern "C" int inspect_threads() { return okc::NTL; }
'''
        cuda = cuda.replace(LAUNCHER, probes)
    elif args.entry != "both":
        chosen = "true" if args.entry == "big" else "false"
        single = (f'extern "C" void inspect_launch(okc::Params p, cudaStream_t stream, int grid) {{\n'
                  f'  okc::moe_fused<{chosen}><<<grid, okc::NTL, okc_smem_bytes(), stream>>>(p);\n}}\n')
        cuda = cuda.replace(LAUNCHER, single)
    if args.experiment == "effective-norm":
        # Candidate contract: prepare() would supply BF16(1 + post_norm_w).
        # The current code already rounds that effective norm before use.
        cuda = cuda.replace("1.f + __bfloat162float", "__bfloat162float")
        cuda = cuda.replace("1.f + __uint_as_float(lnw0 << 16)", "__uint_as_float(lnw0 << 16)")
        cuda = cuda.replace("1.f + __uint_as_float(lnw1 << 16)", "__uint_as_float(lnw1 << 16)")
        old = "return round_bf16(acc * scale) * round_bf16(norm);"
        assert cuda.count(old) == 1
        cuda = cuda.replace(old, "return round_bf16(acc * scale) * norm;")
    elif args.experiment == "packed-swiglu":
        old = "  gate = round_bf16(gate);\n  up = round_bf16(up);"
        new = ("  const __nv_bfloat162 pair = __floats2bfloat162_rn(gate, up);\n"
               "  gate = __bfloat162float(__low2bfloat16(pair));\n"
               "  up = __bfloat162float(__high2bfloat16(pair));")
        assert cuda.count(old) == 1
        cuda = cuda.replace(old, new)
    elif args.experiment == "compact-tiny":
        # Routing storage only; preserve EMAX-dependent global scratch layout.
        # Small dispatch bounds are <=8 tokens (tiny/tiny2), <=16 (mid).
        for decl in ("signed char eid[EMAX]", "short ent[EMAX]", "short pos[EMAX]"):
            assert cuda.count(decl) == 1
            cuda = cuda.replace(decl, decl.replace("[EMAX]", "[OKC_TINY ? (OKC_SPLIT ? 128 : 64) : EMAX]"))
        # Tiny variants never execute the fat job stream, so don't reserve it.
        old = "  return base + (xa > xb ? xa : xb);"
        assert cuda.count(old) == 1
        cuda = cuda.replace(old, "  return base + (OKC_TINY ? xa : (xa > xb ? xa : xb));")
    elif args.experiment == "helper-power2":
        start = cuda.index("__device__ void ws_helper(")
        end = cuda.index("template <bool BIG, bool PB>", start)
        helper = cuda[start:end]
        old = "const int j = njF + v / rbs, rb = v - (v / rbs) * rbs;"
        assert helper.count(old) == 1
        # v is a nonnegative work-unit index; rbs is exactly 256 or 512.
        helper = helper.replace(old, "const int j = njF + (v >> (PB ? RB_B_LOG : RB_A_LOG)), rb = v & (rbs - 1);")
        cuda = cuda[:start] + helper + cuda[end:]
    if args.entry != "resources":
        cuda += '''
extern "C" size_t inspect_dynamic_smem() { return okc_smem_bytes(); }
extern "C" size_t inspect_static_struct() { return sizeof(okc::Smem); }
extern "C" int inspect_threads() { return okc::NTL; }
'''
    overrides = ["-D" + definition for definition in args.define]
    names = {flag.split("=", 1)[0] for flag in overrides}
    flags = [flag for flag in COMMON_FLAGS + defines if flag.split("=", 1)[0] not in names] + overrides
    if not args.no_prmt and args.variant != "fallback":
        flags.append("-DOKC_PRMTSC=1")
    options = {
        "userArguments": " ".join(flags),
        "filters": {"binary": False, "demangle": True, "directives": True, "labels": True},
        "compilerOptions": {"executorRequest": False},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "device.cu").write_text(cuda)
    (args.output / "request.json").write_text(json.dumps({"compiler": args.compiler, "options": options}, indent=2))
    request = urllib.request.Request(
        f"https://godbolt.org/api/compiler/{args.compiler}/compile",
        data=json.dumps({"source": cuda, "options": options}).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        result = json.load(response)
    (args.output / "response.json").write_text(json.dumps(result))
    summary = summarize(result)
    summary.update(compiler=args.compiler, variant=args.variant, entry=args.entry,
                   experiment=args.experiment, flags=flags,
                   submission_sha256=hashlib.sha256(source.encode()).hexdigest())
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output / "host.asm").write_text("\n".join(entry["text"] for entry in result.get("asm", [])))
    for device, output in result.get("devices", {}).items():
        if device.startswith("SASS"):
            (args.output / "device.sass").write_text("\n".join(entry["text"] for entry in output.get("asm", [])))
        elif device.startswith("PTX"):
            (args.output / "device.ptx").write_text("\n".join(entry["text"] for entry in output.get("asm", [])))
    print(json.dumps({key: value for key, value in summary.items() if key != "assembly"}, indent=2))
    if summary["code"] != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
