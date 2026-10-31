"""Offline extraction/parser tests. No CUDA compiler or network calls."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

TOOL = Path(__file__).with_name("inspect_cuda_codegen.py")
SUBMISSION = TOOL.parents[1] / "submission.py"
spec = importlib.util.spec_from_file_location("codegen_tool", TOOL)
codegen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(codegen)


class CodegenTests(unittest.TestCase):
    def test_extract_has_original_device_code_without_torch_wrapper(self):
        source = SUBMISSION.read_text()
        for variant in codegen.VARIANTS:
            cuda, flags = codegen.extract(source, variant)
            self.assertNotIn("#include <torch/extension.h>", cuda)
            self.assertNotIn("okc_compiler_version", cuda)
            self.assertIn("__global__ void __launch_bounds__(NTL, 1) moe_fused", cuda)
            self.assertIn("static size_t okc_smem_bytes()", cuda)
            self.assertIn("moe_fused<false>", cuda)
            self.assertIn("moe_fused<true>", cuda)
            self.assertTrue(flags)

    def test_resource_parser_and_predicated_sass(self):
        result = {
            "code": 0, "timedOut": False, "truncated": False,
            "stderr": [{"text": line} for line in (
                "ptxas info : Compiling entry function 'entry0' for 'sm_90a'",
                " 0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads",
                "ptxas info : Used 153 registers, used 3 barriers",
                "ptxas info : Compiling entry function 'entry1' for 'sm_90a'",
                " 16 bytes stack frame, 8 bytes spill stores, 4 bytes spill loads",
                "ptxas info : Used 128 registers, used 3 barriers",
            )],
            "devices": {"SASS (sm_90a)": {"asm": [
                {"text": "entry0:"},
                {"text": " @!P0 F2F.BF16.F32 R1, R2", "opcodes": ["123"]},
                {"text": " HMMA.16816.F32.BF16 R0, R1, R2, R3", "opcodes": ["456"]},
            ]}},
            "asm": [{"text": "inspect_dynamic_smem:"}, {"text": "movl $214784, %eax"}],
        }
        summary = codegen.summarize(result)
        self.assertEqual(summary["resources"][0]["registers"], 153)
        self.assertEqual(summary["resources"][0]["spill_load_bytes"], 0)
        self.assertEqual(summary["resources"][1]["spill_load_bytes"], 4)
        self.assertEqual(summary["host_resources"]["inspect_dynamic_smem"], 214784)
        assembly = summary["assembly"]["SASS (sm_90a)"]
        self.assertEqual(assembly["instruction_count_all_functions"], 2)
        self.assertEqual(assembly["instruction_histogram_all_functions"]["F2F.BF16.F32"], 1)

    def test_timeout_is_not_reported_as_successful_assembly(self):
        summary = codegen.summarize({"code": 143, "timedOut": True, "stderr": []})
        self.assertTrue(summary["timed_out"])
        self.assertEqual(summary["resources"], [])
        self.assertEqual(summary["assembly"], {})

    def test_experiments_only_change_uploaded_copy(self):
        original = SUBMISSION.read_text()
        expectations = {
            "effective-norm": "return round_bf16(acc * scale) * norm;",
            "packed-swiglu": "const __nv_bfloat162 pair = __floats2bfloat162_rn(gate, up);",
            "compact-tiny": "[OKC_TINY ? (OKC_SPLIT ? 128 : 64) : EMAX]",
            "helper-power2": "v >> (PB ? RB_B_LOG : RB_A_LOG)",
        }
        for experiment, expected in expectations.items():
            with self.subTest(experiment=experiment), tempfile.TemporaryDirectory() as tmp:
                response = io.BytesIO(json.dumps({"code": 0, "stderr": [], "devices": {}}).encode())
                arguments = [str(TOOL), "--variant", "mid", "--experiment", experiment, "--output", tmp]
                with patch.object(sys, "argv", arguments), patch.object(
                    codegen.urllib.request, "urlopen", return_value=response
                ) as upload, contextlib.redirect_stdout(io.StringIO()):
                    codegen.main()
                request = upload.call_args.args[0]
                data = json.loads(request.data)
                self.assertIn(expected, data["source"])
                self.assertFalse(data["options"]["compilerOptions"]["executorRequest"])
                self.assertEqual(SUBMISSION.read_text(), original)


if __name__ == "__main__":
    unittest.main()
