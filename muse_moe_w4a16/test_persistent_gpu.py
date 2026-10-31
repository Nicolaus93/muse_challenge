"""Optional end-to-end edge tests for a Hopper development environment.

These are not run by okc. They require CUDA, PyTorch, a C++/CUDA toolchain,
several GB of VRAM, and up to 300 seconds for prepare/compilation.
"""

import unittest

try:
    import torch
except ImportError:
    torch = None

HAS_HOPPER = (torch is not None and torch.cuda.is_available()
              and torch.cuda.get_device_capability() == (9, 0))


@unittest.skipUnless(HAS_HOPPER, "requires Hopper, PyTorch, and CUDA toolchain")
class PersistentGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import submission
        from nvfp4 import random_weights

        cls.fn = submission
        gen = torch.Generator(device="cuda").manual_seed(5374)
        w13, sf13, gs13 = random_weights(8192, 4096, leading=(32,), generator=gen, device="cuda")
        w2, sf2, gs2 = random_weights(4096, 4096, leading=(32,), generator=gen, device="cuda")
        cls.weights = {
            "w13": w13, "w13_sf": sf13, "w13_gs": gs13.expand(32).contiguous(),
            "w2": w2, "w2_sf": sf2, "w2_gs": gs2.expand(32).contiguous(),
            "post_norm_w": torch.randn(4096, device="cuda", dtype=torch.bfloat16) * 0.1,
        }
        cls.state = cls.fn.prepare(cls.weights)
        # Reference matrices live in the test only, never in submission state.
        from nvfp4 import dequantize

        cls.ref13 = dequantize(w13[0], sf13[0], gs13)
        cls.ref2 = dequantize(w2[0], sf2[0], gs2)

    def inputs(self, tokens):
        ids = torch.arange(32, 40, device="cuda", dtype=torch.int32).expand(tokens, 8).clone()
        ids[:, 0] = 0
        return {
            "hidden_states": torch.randn((tokens, 4096), device="cuda", dtype=torch.bfloat16),
            "topk_ids": ids,
            "topk_weights": torch.full((tokens, 8), 0.125, device="cuda"),
            "config": {"local_start": 0, "local_count": 32, "top_k": 8},
        }

    def reference(self, data):
        h = data["hidden_states"].float() @ self.ref13.T
        gate, up = h.chunk(2, dim=-1)
        inter = torch.nn.functional.silu(gate) * up
        y = (inter @ self.ref2.T) * (1 + self.weights["post_norm_w"].float())
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-8)
        return (y * data["topk_weights"][:, :1]).to(torch.bfloat16)

    def test_skewed_single_expert_and_chunking(self):
        for tokens in (1, 4, 8, 16, 33, 257, 385):
            with self.subTest(tokens=tokens):
                data = self.inputs(tokens)
                out = self.fn.kernel_fn(data, self.state)
                ref = self.reference(data)
                error = (out.float() - ref.float()).norm() / ref.float().norm()
                self.assertLess(error.item(), 0.015)

    def test_all_nonlocal_routes_produce_zero(self):
        for tokens in (1, 4, 8, 16, 32, 128, 257):
            with self.subTest(tokens=tokens):
                data = self.inputs(tokens)
                data["topk_ids"][:, 0] = 40
                out = self.fn.kernel_fn(data, self.state)
                torch.testing.assert_close(out, torch.zeros_like(out))

    def test_graph_replay_uses_fresh_routes_and_inputs(self):
        for tokens in (1, 16, 33):
            with self.subTest(tokens=tokens):
                data = self.inputs(tokens)
                self.fn.kernel_fn(data, self.state)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out = self.fn.kernel_fn(data, self.state)
                for iteration in range(8):
                    data["hidden_states"].normal_()
                    data["topk_ids"][:, 0] = 0 if iteration % 2 else 40
                    graph.replay()
                    if iteration % 2:
                        ref = self.reference(data)
                        error = (out.float() - ref.float()).norm() / ref.float().norm()
                        self.assertLess(error.item(), 0.015)
                    else:
                        torch.testing.assert_close(out, torch.zeros_like(out))

    def test_prmt_guard_rejects_subnormal_and_signed_scales(self):
        fn = self.fn
        for code, expected in ((0, False), (1, False), (7, False), (8, True),
                               (0x7E, True), (0x80, False), (0x88, False)):
            sf = torch.tensor([code], device="cuda", dtype=torch.uint8).view(torch.float8_e4m3fn)
            self.assertEqual(fn._scales_prmt_exact(sf), expected)


if __name__ == "__main__":
    unittest.main()
