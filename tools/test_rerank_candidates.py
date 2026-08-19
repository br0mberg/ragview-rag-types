from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("rerank_candidates.py")
SPEC = importlib.util.spec_from_file_location("rerank_candidates", MODULE_PATH)
assert SPEC and SPEC.loader
rerank = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = rerank
SPEC.loader.exec_module(rerank)


class RerankCandidatesTest(unittest.TestCase):
    def test_metrics_apply_cutoff(self) -> None:
        metrics = rerank.aggregate([1, 2, 0], 10)

        self.assertAlmostEqual(metrics["hit_rate_at_k"], 2 / 3)
        self.assertAlmostEqual(metrics["mrr_at_k"], 0.5)
        self.assertAlmostEqual(metrics["ndcg_at_k"], (1 + 1 / rerank.math.log2(3)) / 3)

    def test_rank_accepts_any_relevant_chunk(self) -> None:
        rank = rerank.rank_of(["a", "article-chunk-2", "b"], {"article-chunk-1", "article-chunk-2"}, 2)

        self.assertEqual(rank, 2)

    def test_smaller_pool_does_not_use_later_candidates(self) -> None:
        candidates = [
            rerank.Candidate("a", 1, 0.0),
            rerank.Candidate("b", 2, 0.0),
            rerank.Candidate("c", 3, 0.0),
        ]

        ranked = rerank.rerank_at_pool(candidates, [0.1, 0.2, 10.0], 2)

        self.assertEqual([candidate.doc_id for candidate, _ in ranked], ["b", "a"])

    def test_java_candidate_dump_contract(self) -> None:
        record = {
            "qid": "q1",
            "kind": "exact",
            "q": "question",
            "rel": ["d1"],
            "cand": [
                {"rank": 1, "docId": "d1", "score": 0.75},
                {"rank": 2, "docId": "d2", "score": 0.5},
            ],
        }

        candidates = rerank.parse_candidates(record)

        self.assertEqual(candidates[0], rerank.Candidate("d1", 1, 0.75))
        self.assertEqual(candidates[1], rerank.Candidate("d2", 2, 0.5))


if __name__ == "__main__":
    unittest.main()
