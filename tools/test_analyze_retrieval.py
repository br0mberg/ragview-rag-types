from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("analyze_retrieval.py")
SPEC = importlib.util.spec_from_file_location("analyze_retrieval", MODULE_PATH)
assert SPEC and SPEC.loader
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


class AnalyzeRetrievalTest(unittest.TestCase):
    def test_write_report_uses_paired_strategy_queries(self) -> None:
        results = {
            "dense": {
                "e": analysis.QueryRank("e", "exact", 0),
                "s": analysis.QueryRank("s", "semantic", 1),
            },
            "hybrid": {
                "e": analysis.QueryRank("e", "exact", 1),
                "s": analysis.QueryRank("s", "semantic", 1),
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            analysis.write_report(results, output, 100, "fixed")
            deltas = (output / "retrieval_paired_deltas.csv").read_text()

        self.assertIn("dense,hybrid,all,2,hit_rate_at_10,0.500000", deltas)

    def test_loader_rejects_mismatched_qids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            path.write_text(
                "strategy,qid,kind,rank,latency_ms\n"
                "dense,q1,exact,1,1\n"
                "hybrid,q2,exact,1,1\n"
            )
            with self.assertRaisesRegex(ValueError, "same qids"):
                analysis.load_results(path)

    def test_clustered_bootstrap_keeps_exact_semantic_pairs_together(self) -> None:
        rows = [
            analysis.QueryRank("exact-001", "exact", 1),
            analysis.QueryRank("exact-002", "exact", 0),
            analysis.QueryRank("semantic-001", "semantic", 0),
            analysis.QueryRank("semantic-002", "semantic", 1),
        ]

        samples = analysis.bootstrap_qids(rows, 20, "fixed", cluster_pairs=True)

        for sample in samples:
            counts = Counter(qid.removeprefix("exact-").removeprefix("semantic-") for qid in sample)
            self.assertTrue(all(count % 2 == 0 for count in counts.values()))
            self.assertEqual(len(sample), 4)

    def test_fusion_counts_separates_rescues_and_harms(self) -> None:
        results = {
            "dense": {
                "rescue": analysis.QueryRank("rescue", "semantic", 0),
                "harm": analysis.QueryRank("harm", "semantic", 1),
                "stable": analysis.QueryRank("stable", "semantic", 1),
            },
            "bm25": {
                "rescue": analysis.QueryRank("rescue", "semantic", 0),
                "harm": analysis.QueryRank("harm", "semantic", 0),
                "stable": analysis.QueryRank("stable", "semantic", 1),
            },
            "hybrid": {
                "rescue": analysis.QueryRank("rescue", "semantic", 1),
                "harm": analysis.QueryRank("harm", "semantic", 0),
                "stable": analysis.QueryRank("stable", "semantic", 1),
            },
        }

        self.assertEqual(
            analysis.fusion_counts(results, ["rescue", "harm", "stable"]),
            (2, 1, 2, 2, 1, 1),
        )
        self.assertEqual(
            analysis.strategy_transition_counts(
                results["dense"], results["hybrid"], ["rescue", "harm", "stable"]
            ),
            (2, 2, 1, 1),
        )


if __name__ == "__main__":
    unittest.main()
