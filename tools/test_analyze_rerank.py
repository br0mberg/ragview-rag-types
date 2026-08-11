from __future__ import annotations

import importlib.util
import csv
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("analyze_rerank.py")
SPEC = importlib.util.spec_from_file_location("analyze_rerank", MODULE_PATH)
assert SPEC and SPEC.loader
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


class AnalyzeRerankTest(unittest.TestCase):
    def test_contributions(self) -> None:
        self.assertEqual(analysis.contributions(0), (0.0, 0.0, 0.0))
        hit, reciprocal_rank, ndcg = analysis.contributions(2)
        self.assertEqual(hit, 1.0)
        self.assertEqual(reciprocal_rank, 0.5)
        self.assertAlmostEqual(ndcg, 1 / analysis.math.log2(3))

    def test_stratified_bootstrap_preserves_slice_sizes(self) -> None:
        results = [
            analysis.QueryResult("e1", "exact", {10: 1}),
            analysis.QueryResult("e2", "exact", {10: 0}),
            analysis.QueryResult("s1", "semantic", {10: 2}),
        ]
        samples = analysis.bootstrap_indices(results, 5, "fixed")

        self.assertTrue(all(len(sample) == 3 for sample in samples))
        self.assertTrue(all(sum(index < 2 for index in sample) == 2 for sample in samples))

    def test_paired_delta_uses_same_queries(self) -> None:
        results = [
            analysis.QueryResult("a", "exact", {10: 0, 20: 1}),
            analysis.QueryResult("b", "exact", {10: 1, 20: 1}),
        ]
        samples = analysis.bootstrap_indices(results, 200, "fixed")
        point, low, high = analysis.paired_delta_interval(results, samples, 10, 20, 0)

        self.assertEqual(point, 0.5)
        self.assertLessEqual(low, point)
        self.assertGreaterEqual(high, point)

    def test_clustered_bootstrap_keeps_exact_semantic_pairs_together(self) -> None:
        results = [
            analysis.QueryResult("exact-001", "exact", {10: 1}),
            analysis.QueryResult("exact-002", "exact", {10: 0}),
            analysis.QueryResult("semantic-001", "semantic", {10: 0}),
            analysis.QueryResult("semantic-002", "semantic", {10: 1}),
        ]

        samples = analysis.bootstrap_indices(results, 20, "fixed", cluster_pairs=True)

        for sample in samples:
            counts = Counter(
                results[index].qid.removeprefix("exact-").removeprefix("semantic-")
                for index in sample
            )
            self.assertTrue(all(count % 2 == 0 for count in counts.values()))
            self.assertEqual(len(sample), 4)

    def test_write_report_covers_every_adjacent_pool(self) -> None:
        results = [
            analysis.QueryResult("a", "exact", {10: 0, 20: 1, 30: 1}),
            analysis.QueryResult("b", "semantic", {10: 1, 20: 1, 30: 1}),
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "analysis"
            analysis.write_report(results, output, 100, "fixed")
            lines = (output / "rerank_paired_deltas.csv").read_text().splitlines()

        self.assertTrue(any(line.startswith("hybrid,10,20,all") for line in lines))
        self.assertTrue(any(line.startswith("hybrid,20,30,all") for line in lines))

    def test_load_csv_keeps_sources_and_pools_separate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "rerank.csv"
            with source.open("w", encoding="utf-8", newline="") as target:
                writer = csv.writer(target)
                writer.writerow([
                    "source", "qid", "kind", "quality_pool", "scored_pool",
                    "candidate_rank", "rerank_rank", "scored_pool_latency_ms",
                ])
                for candidate_source in ["dense", "hybrid"]:
                    writer.writerow([candidate_source, "q1", "exact", 10, 100, 11, 1, 10])
                    writer.writerow([candidate_source, "q1", "exact", 20, 100, 11, 1, 10])

            results = analysis.load_results(source)

        self.assertEqual({result.source for result in results}, {"dense", "hybrid"})
        self.assertTrue(all(result.ranks == {10: 1, 20: 1} for result in results))

    def test_transition_counts_separates_rescues_and_harms(self) -> None:
        results = [
            analysis.QueryResult("rescue", "exact", {50: 1}, candidate_ranks={50: 11}),
            analysis.QueryResult("harm", "exact", {50: 0}, candidate_ranks={50: 2}),
            analysis.QueryResult("stable", "exact", {50: 1}, candidate_ranks={50: 1}),
        ]

        self.assertEqual(analysis.transition_counts(results, 50), (2, 2, 1, 1))


if __name__ == "__main__":
    unittest.main()
