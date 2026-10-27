"""CPU-only wiring tests; these do NOT validate GPU arithmetic or performance.

Run: python3 -B -m unittest discover -s muse_moe_w4a16 -p 'test_*.py' -v
The fake operators check dimensions, routing masks, and precision flags without
requiring PyTorch/SGLang in a documentation-only development environment.
"""

import ast
import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).with_name("submission.py")


class Tensor:
    def __init__(self, shape, dtype="bfloat16", values=None):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = "cuda:fake"
        self.values = values
        self.data = self

    def contiguous(self):
        return self

    def to(self, dtype):
        return Tensor(self.shape, dtype, self.values)

    def float(self):
        return self.to("float32")

    def __add__(self, other):
        return Tensor(self.shape, self.dtype)

    def view(self, *shape):
        if -1 in shape:
            known = math.prod(s for s in shape if s != -1)
            shape = tuple(math.prod(self.shape) // known if s == -1 else s for s in shape)
        assert math.prod(shape) == math.prod(self.shape)
        values = self.values
        if values is not None and len(self.shape) == 2 and len(shape) == 1:
            values = [v for row in values for v in row]
        return Tensor(shape, self.dtype, values)

    def __getitem__(self, index):
        if isinstance(index, Tensor):
            assert len(self.shape) == 1
            values = [[self.values[i] for i in row] for row in index.values]
            return Tensor(index.shape, self.dtype, values)
        if isinstance(index, slice):
            size = len(range(*index.indices(self.shape[0])))
            values = None if self.values is None else self.values[index]
            return Tensor((size, *self.shape[1:]), self.dtype, values)
        raise AssertionError(f"unexpected fake index: {index}")

    def __setitem__(self, index, value):
        self.values[index] = value.values


def module(name, **attributes):
    result = ModuleType(name)
    result.__dict__.update(attributes)
    return result


class SubmissionContractTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.packed = {}

        def repack(layer):
            self.calls.append(("repack", layer))
            for name in (
                "w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale",
                "w13_weight_scale_2", "w2_weight_scale_2",
            ):
                self.packed[name] = layer.__dict__[name]

        def align(ids, block_size, n_experts):
            self.calls.append(("align", ids, block_size, n_experts))
            return Tensor((1,), "int32"), Tensor((1,), "int32"), Tensor((1,), "int32")

        def gemm(*args, **kwargs):
            self.calls.append(("gemm", args, kwargs))
            return args[1]

        def silu(h, inter, **kwargs):
            self.calls.append(("silu", h, inter, kwargs))

        def norm(y, weights, **kwargs):
            self.calls.append(("norm", y, weights, kwargs))
            return Tensor((y.shape[0], y.shape[2]))

        torch = module(
            "torch", Tensor=Tensor, bfloat16="bfloat16", int32="int32", float32="float32",
            full=lambda shape, value, **kw: Tensor(shape, kw["dtype"], [value] * shape[0]),
            arange=lambda count, **kw: Tensor((count,), kw["dtype"], list(range(count))),
            empty=lambda shape, **kw: Tensor(shape, kw["dtype"]),
        )
        dependencies = {
            "torch": torch,
            "sgl_kernel.scalar_type": module(
                "sgl_kernel.scalar_type", scalar_types=SimpleNamespace(float4_e2m1f="fp4")
            ),
            "sglang.kernels.ops.activation.activation": module(
                "activation", silu_and_mul=silu
            ),
            "sglang.kernels.ops.moe.moe_wna16_marlin": module(
                "marlin", moe_wna16_marlin_gemm=gemm
            ),
            "sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton": module(
                "norm", fused_post_expert_norm_weighted_sum=norm
            ),
            "sglang.srt.layers.moe.fused_moe_triton": module(
                "align", moe_align_block_size=align
            ),
            "sglang.srt.layers.quantization.marlin_utils": module(
                "workspace", marlin_make_workspace=lambda *a, **kw: Tensor((1,), "int32")
            ),
            "sglang.srt.layers.quantization.marlin_utils_fp4": module(
                "repack", prepare_moe_nvfp4_layer_for_marlin=repack
            ),
        }
        with patch.dict(sys.modules, dependencies):
            spec = importlib.util.spec_from_file_location("submission_under_test", SOURCE)
            self.submission = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.submission)

        self.weights = {
            "w13": Tensor((256, 8192, 2048), "uint8"),
            "w13_sf": Tensor((256, 8192, 256), "float8_e4m3fn"),
            "w13_gs": Tensor((256,), "float32"),
            "w2": Tensor((256, 4096, 2048), "uint8"),
            "w2_sf": Tensor((256, 4096, 256), "float8_e4m3fn"),
            "w2_gs": Tensor((256,), "float32"),
            "post_norm_w": Tensor((4096,)),
        }
        self.state = self.submission.prepare(self.weights)

    def inputs(self, bs):
        return {
            "hidden_states": Tensor((bs, 4096)),
            "topk_ids": Tensor((bs, 8), "int32", [[0, 31, 32, 255, 7, 9, 100, 200]] * bs),
            "topk_weights": Tensor((bs, 8), "float32"),
            "config": {"local_start": 0, "local_count": 32, "top_k": 8},
        }

    def test_prepare_repacks_local_bank_once(self):
        repacks = [c for c in self.calls if c[0] == "repack"]
        self.assertEqual(len(repacks), 1)
        layer = repacks[0][1]
        self.assertEqual(layer.w13_weight.shape, (32, 8192, 2048))
        self.assertEqual(layer.w2_weight.shape, (32, 4096, 2048))
        self.assertEqual(layer.quant_config.group_size, 16)
        self.assertEqual(self.state["expert_map"].values, list(range(32)) + [-1] * 224)
        self.assertIs(self.state["w13"], self.packed["w13_weight"])
        self.assertEqual(self.state["norm_w"].dtype, "bfloat16")

    def test_pipeline_shapes_and_masks_for_scored_and_nearby_batches(self):
        for bs in (1, 4, 8, 16, 32, 33, 128, 256, 257):
            with self.subTest(bs=bs):
                self.calls.clear()
                output = self.submission.kernel_fn(self.inputs(bs), self.state)
                self.assertEqual(output.shape, (bs, 4096))
                self.assertEqual(output.dtype, "bfloat16")
                self.assertEqual([c[0] for c in self.calls], ["align", "gemm", "silu", "gemm", "norm"])
                ids = self.calls[0][1]
                self.assertEqual(ids.values, [[0, 31, -1, -1, 7, 9, -1, -1]] * bs)
                first, second = self.calls[1], self.calls[3]
                self.assertEqual(first[1][0].shape, (bs, 4096))
                self.assertEqual(first[1][1].shape, (bs * 8, 8192))
                self.assertEqual(first[2]["top_k"], 8)
                self.assertEqual(first[2]["size_m"], bs)
                self.assertEqual(second[1][0].shape, (bs * 8, 4096))
                self.assertEqual(second[1][1].shape, (bs * 8, 4096))
                self.assertEqual(second[2]["top_k"], 1)
                self.assertEqual(second[2]["size_m"], bs * 8)
                for call in (first, second):
                    self.assertTrue(call[2]["use_fp32_reduce"])
                    self.assertTrue(call[2]["is_ep"])
                    self.assertFalse(call[2]["mul_topk_weights"])
                    self.assertFalse(call[2]["use_atomic_add"])
                self.assertEqual(self.calls[2][3]["expert_ids"].values, [v for row in ids.values for v in row])
                final = self.calls[-1]
                self.assertIs(final[3]["slot_expert_ids"], ids)
                self.assertTrue(final[3]["weight_first"])
                self.assertEqual(final[3]["eps"], 1e-8)

    def test_routing_is_recomputed_for_each_call(self):
        data = self.inputs(1)
        self.submission.kernel_fn(data, self.state)
        data["topk_ids"].values = [[255] * 8]
        self.submission.kernel_fn(data, self.state)
        self.assertEqual(self.calls[-1][3]["slot_expert_ids"].values, [[-1] * 8])

    def test_rejects_unprepared_expert_partition(self):
        data = self.inputs(1)
        data["config"]["local_start"] = 32
        with self.assertRaisesRegex(ValueError, "EP8"):
            self.submission.kernel_fn(data, self.state)

    def test_tiling_heuristic_uses_only_shape(self):
        for bs in (1, 4, 8, 16, 32, 33, 128, 256, 257, 1024):
            expected = 64
            for block in (8, 16, 32, 48, 64):
                if bs * 8 / 32 / block < 0.9:
                    expected = block
                    break
            self.assertEqual(self.submission._block_size_m(bs, 8, 32), expected)

    def test_no_reference_import_or_host_tensor_reads(self):
        source = SOURCE.read_text()
        self.assertEqual(source.splitlines()[0], "# okc: problem=muse_moe_w4a16")
        tree = ast.parse(source)
        imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        for node in imports:
            names = [node.module] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
            self.assertFalse(any(n.split(".")[0] in {"reference", "baseline", "nvfp4"} for n in names))
        forbidden = {"item", "cpu", "numpy", "tolist", "synchronize", "nonzero"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, forbidden)


if __name__ == "__main__":
    unittest.main()
