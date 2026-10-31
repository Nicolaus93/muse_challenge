"""CPU-only contracts for the attributed persistent CUDA backend.

These tests validate dispatch, bit encodings and source structure, not GPU math.
Official GPU checks and the optional test_persistent_gpu.py cover execution.
"""

import ast
import importlib.util
import math
from pathlib import Path
import struct
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).with_name("submission.py")


class Tensor:
    def __init__(self, shape, dtype="bfloat16", pointer=16):
        self.shape, self.dtype, self.pointer = tuple(shape), dtype, pointer
        self.device = SimpleNamespace(index=0)

    def contiguous(self):
        return self

    def to(self, dtype):
        return self if dtype == self.dtype else Tensor(self.shape, dtype, self.pointer)

    def clone(self):
        return Tensor(self.shape, self.dtype, 16)

    def data_ptr(self):
        return self.pointer

    def __getitem__(self, key):
        assert isinstance(key, slice)
        size = len(range(*key.indices(self.shape[0])))
        return Tensor((size, *self.shape[1:]), self.dtype, self.pointer)


def module(name, **attributes):
    result = ModuleType(name)
    result.__dict__.update(attributes)
    return result


class PersistentContractTests(unittest.TestCase):
    def setUp(self):
        self.compiles, self.launches = [], []

        def load_inline(**kwargs):
            self.compiles.append(kwargs)
            return SimpleNamespace(
                okc_set_pdl=lambda on: None,
                okc_run=lambda *args: self.launches.append(args),
            )

        torch = module(
            "torch", Tensor=Tensor, bfloat16="bfloat16", int32="int32", float32="float32",
            empty=lambda shape, **kw: Tensor(shape, kw["dtype"]),
            cuda=SimpleNamespace(get_device_capability=lambda: (9, 0)),
        )
        dependencies = {
            "torch": torch,
            "torch.utils": module("torch.utils"),
            "torch.utils.cpp_extension": module("torch.utils.cpp_extension", load_inline=load_inline),
        }
        with patch.dict(sys.modules, dependencies):
            spec = importlib.util.spec_from_file_location("persistent_under_test", SOURCE)
            self.submission = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.submission)
        self.state = {
            "lo": 0, "cnt": 32, "grid": 132,
            "w": [object() for _ in range(14)], "scratch": [object() for _ in range(3)],
            "sfx": ("-DOKC_PRMTSC=1",),
        }

    def inputs(self, tokens):
        return {
            "hidden_states": Tensor((tokens, 4096)),
            "topk_ids": Tensor((tokens, 8), "int32"),
            "topk_weights": Tensor((tokens, 8), "float32"),
            "config": {"local_start": 0, "local_count": 32, "top_k": 8},
        }

    def test_dispatch_scored_nearby_and_chunked_shapes(self):
        fn = self.submission
        for tokens in (1, 4, 8, 16, 17, 32, 33, 128, 256, 257, 384, 385, 769):
            with self.subTest(tokens=tokens):
                self.launches.clear()
                out = fn.kernel_fn(self.inputs(tokens), self.state)
                self.assertEqual(out.shape, (tokens, 4096))
                self.assertEqual(out.dtype, "bfloat16")
                self.assertEqual(len(self.launches), math.ceil(tokens / 384))
                self.assertEqual(sum(a[2].shape[0] for a in self.launches), tokens)
                for args in self.launches:
                    self.assertIs(args[0], self.state["w"])
                    self.assertIs(args[1], self.state["scratch"])
                    self.assertEqual(args[2].shape[1], 4096)
                    self.assertEqual(args[3].shape, (args[2].shape[0], 8))
                    self.assertEqual(args[4].dtype, "float32")
                    self.assertEqual(args[5].dtype, "bfloat16")
                    self.assertEqual(args[6:], (0, 32, 132))
                flags = self.compiles[-1]["extra_cuda_cflags"]
                if tokens <= 4:
                    self.assertTrue(all(d in flags for d in fn.TINY_DEFINES))
                elif tokens <= 8:
                    self.assertTrue(all(d in flags for d in fn.TINY2_DEFINES))
                elif tokens <= 16:
                    self.assertTrue(all(d in flags for d in fn.MID_DEFINES))
                else:
                    self.assertTrue(all(d in flags for d in fn.MAIN_DEFINES))

    def test_only_compiled_code_is_cached(self):
        fn = self.submission
        data = self.inputs(1)
        first = fn.kernel_fn(data, self.state)
        data["topk_ids"] = Tensor((1, 8), "int32", pointer=32)
        second = fn.kernel_fn(data, self.state)
        self.assertIsNot(first, second)
        self.assertEqual(len(self.compiles), 1)
        self.assertEqual(self.launches[-1][3].data_ptr(), 32)

    def test_alignment_copy_and_scale_fallback(self):
        fn = self.submission
        data = self.inputs(33)
        data["topk_ids"].pointer = 18
        fn.kernel_fn(data, self.state)
        self.assertEqual(self.launches[-1][3].data_ptr(), 16)
        self.assertEqual(fn._main_defines(()), fn.MAIN_FALLBACK_DEFINES)
        self.assertNotIn("-DOKC_WGS=1", fn._main_defines(()))

    def test_rejects_invalid_activation_contract(self):
        fn = self.submission
        for field, value in (("local_start", 32), ("local_count", 16), ("top_k", 4)):
            data = self.inputs(1)
            data["config"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                fn.kernel_fn(data, self.state)
        data = self.inputs(1)
        data["hidden_states"].dtype = "float32"
        with self.assertRaisesRegex(ValueError, "BF16"):
            fn.kernel_fn(data, self.state)

    def test_dq7_bit_permutation_roundtrips_all_codes(self):
        positions = self.submission._stored_pos()
        self.assertEqual(sorted(positions.values()), list(range(32)))
        mask = 0x81C081C0
        def rotl(q, n):
            return ((q << n) | (q >> (32 - n))) & 0xFFFFFFFF
        for reg in range(4):
            for upper in range(2):
                for code in range(16):
                    packed = sum(((code >> bit) & 1) << positions[reg, upper, bit] for bit in range(4))
                    decoded = (
                        packed & mask,
                        (packed << 4) & mask,
                        rotl(packed, 10) & mask,
                        ((packed << 6) & 0x80C080C0) | (rotl(packed, 14) & 0x01000100),
                    )[reg]
                    bits = (decoded >> (16 * upper)) & 0xFFFF
                    scaled = struct.unpack("f", struct.pack("I", bits << 16))[0]
                    expected = (0, 0.5, 1, 1.5, 2, 3, 4, 6)[code & 7]
                    if code & 8:
                        expected = -expected
                    self.assertEqual(math.ldexp(scaled, 126), expected)

    def test_bf16_boundaries_and_empty_routing_guard_are_explicit(self):
        src = self.submission.CUDA_SRC
        self.assertIn("gate = round_bf16(gate);", src)
        self.assertIn("up = round_bf16(up);", src)
        self.assertIn("return round_bf16(acc * scale) * round_bf16(norm);", src)
        self.assertNotIn("SILU_DIV(gt,", src)
        self.assertEqual(src.count("swiglu_bf16(gt, up)"), 5)
        self.assertIn("if (nj == 0)", src)
        self.assertIn("C10_CUDA_KERNEL_LAUNCH_CHECK", src)

    def test_submission_is_self_contained_and_timed_path_has_no_host_reads(self):
        src = SOURCE.read_text()
        self.assertEqual(src.splitlines()[0], "# okc: problem=muse_moe_w4a16")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn(node.module.split(".")[0], {"reference", "baseline", "nvfp4"})
        kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "kernel_fn")
        for node in ast.walk(kernel):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, {"item", "cpu", "tolist", "numpy", "synchronize"})
        self.assertNotIn("_time_graph", src)
        self.assertNotIn("BW_SRC", src)
        self.assertNotIn("ALT_SRCS", src)


if __name__ == "__main__":
    unittest.main()
