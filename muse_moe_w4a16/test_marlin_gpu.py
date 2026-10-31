"""Optional CUDA edge-case tests; no expert-weight allocation is required.

Run with the judge's PyTorch/Triton/SGLang environment:
    python3 -B -m unittest discover -s muse_moe_w4a16 -p 'test_*.py' -v
"""

import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires CUDA/PyTorch")
class MarlinRoutingGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import submission

        cls.submission = submission

    def check_routes(self, ids):
        fn = self.submission
        tokens, top_k = ids.shape
        slots = tokens * top_k
        state = {"local_start": 0, "local_count": 32}
        bm = fn._block_size_m(tokens, top_k, 32)
        local, sorted_ids, experts, padded = fn._routes(ids, state, bm)
        expected = torch.where((ids >= 0) & (ids < 32), ids, -1).to(torch.int32)
        torch.testing.assert_close(local, expected)
        length = padded.item()
        self.assertEqual(length % bm, 0)
        self.assertLessEqual(length, sorted_ids.numel())
        ordered = sorted_ids[:length].cpu().tolist()
        block_experts = experts[:length // bm].cpu().tolist()
        flat = expected.flatten().cpu().tolist()
        seen = []
        for block, expert in enumerate(block_experts):
            self.assertTrue(0 <= expert < 32)
            for slot in ordered[block * bm:(block + 1) * bm]:
                if slot == slots:
                    continue
                self.assertTrue(0 <= slot < slots)
                self.assertEqual(flat[slot], expert)
                seen.append(slot)
        self.assertEqual(sorted(seen), [i for i, e in enumerate(flat) if e >= 0])

    def test_empty_local_partition(self):
        for tokens in (1, 4, 16, 32):
            with self.subTest(tokens=tokens):
                self.check_routes(torch.full((tokens, 8), 255, device="cuda", dtype=torch.int32))

    def test_duplicate_routes_and_expert_skew(self):
        for tokens in (1, 4, 16, 32):
            with self.subTest(tokens=tokens):
                ids = torch.zeros((tokens, 8), device="cuda", dtype=torch.int32)
                ids[:, -1] = 31
                self.check_routes(ids)

    def test_random_routes_and_fresh_graph_inputs(self):
        fn = self.submission
        ids = torch.randint(0, 256, (16, 8), device="cuda", dtype=torch.int64)
        self.check_routes(ids)
        fn._routes(ids, {"local_start": 0, "local_count": 32}, 8)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            local, _, _, padded = fn._routes(ids, {"local_start": 0, "local_count": 32}, 8)
        ids.fill_(255)
        graph.replay()
        self.assertEqual(padded.item(), 0)
        self.assertTrue((local == -1).all().item())

    def test_norm_masks_nan_scratch_and_zero_rows(self):
        fn = self.submission
        for tokens in (1, 4, 16):
            with self.subTest(tokens=tokens):
                y = torch.full((tokens, 8, 4096), float("nan"), device="cuda", dtype=torch.bfloat16)
                ids = torch.full((tokens, 8), -1, device="cuda", dtype=torch.int32)
                scores = torch.rand((tokens, 8), device="cuda")
                norm = torch.ones(4096, device="cuda", dtype=torch.bfloat16)
                out = fn._post_norm_sum(y, scores, norm, ids)
                torch.testing.assert_close(out, torch.zeros_like(out))
                ids[:, 0] = 0
                y[:, 0] = 0
                out = fn._post_norm_sum(y, scores, norm, ids)
                torch.testing.assert_close(out, torch.zeros_like(out))

    def test_norm_matches_float32_formula(self):
        fn = self.submission
        y = torch.randn((16, 8, 4096), device="cuda", dtype=torch.bfloat16)
        scores = torch.rand((16, 8), device="cuda")
        ids = torch.randint(-1, 32, (16, 8), device="cuda", dtype=torch.int32)
        norm = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
        z = torch.where((ids >= 0).unsqueeze(-1), y.float(), 0) * norm.float()
        ref = (z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-8)
               * scores.unsqueeze(-1)).sum(1).to(torch.bfloat16)
        out = fn._post_norm_sum(y, scores, norm, ids)
        torch.testing.assert_close(out, ref, rtol=0.01, atol=0.01)


if __name__ == "__main__":
    unittest.main()
