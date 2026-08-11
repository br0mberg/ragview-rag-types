#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QueryRank:
    qid: str
    kind: str
    rank: int


def contributions(rank: int) -> tuple[float, float, float]:
    if rank <= 0:
        return 0.0, 0.0, 0.0
    return 1.0, 1.0 / rank, 1.0 / math.log2(rank + 1)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def load_results(path: Path) -> dict[str, dict[str, QueryRank]]:
    results: dict[str, dict[str, QueryRank]] = {}
    with path.open(encoding="utf-8") as source:
        for row in csv.DictReader(source):
            strategy = row["strategy"]
            qid = row["qid"]
            result = QueryRank(qid, row["kind"], int(row["rank"]))
            if qid in results.setdefault(strategy, {}):
                raise ValueError(f"duplicate {strategy}/{qid}")
            results[strategy][qid] = result
    qid_sets = {frozenset(items) for items in (strategy_results.keys() for strategy_results in results.values())}
    if len(qid_sets) != 1:
        raise ValueError("strategies do not cover the same qids")
    return results


def paired_qid_groups(rows: list[QueryRank]) -> tuple[list[str], dict[str, dict[str, str]]]:
    kinds = sorted({row.kind for row in rows})
    if len(kinds) < 2:
        raise ValueError("clustered bootstrap requires at least two question kinds")
    groups: dict[str, dict[str, str]] = {}
    for row in rows:
        prefix = f"{row.kind}-"
        if not row.qid.startswith(prefix):
            raise ValueError(f"qid does not encode its kind: {row.qid}")
        group_id = row.qid[len(prefix):]
        group = groups.setdefault(group_id, {})
        if row.kind in group:
            raise ValueError(f"duplicate paired qid group: {group_id}/{row.kind}")
        group[row.kind] = row.qid
    expected = set(kinds)
    incomplete = [group_id for group_id, group in groups.items() if set(group) != expected]
    if incomplete:
        raise ValueError(f"incomplete paired qid groups: {', '.join(sorted(incomplete)[:3])}")
    return kinds, groups


def bootstrap_qids(
    rows: list[QueryRank],
    iterations: int,
    seed: str,
    cluster_pairs: bool = False,
) -> list[list[str]]:
    rng = random.Random(seed)
    if cluster_pairs:
        kinds, groups = paired_qid_groups(rows)
        group_ids = sorted(groups)
        return [
            [
                groups[group_id][kind]
                for group_id in (rng.choice(group_ids) for _ in group_ids)
                for kind in kinds
            ]
            for _ in range(iterations)
        ]
    by_kind: dict[str, list[str]] = {}
    for row in rows:
        by_kind.setdefault(row.kind, []).append(row.qid)
    return [
        [rng.choice(qids) for kind in sorted(by_kind) for qids in [by_kind[kind]] for _ in qids]
        for _ in range(iterations)
    ]


def mean_metric(rows: dict[str, QueryRank], qids: list[str], metric_index: int) -> float:
    return sum(contributions(rows[qid].rank)[metric_index] for qid in qids) / len(qids)


def interval(estimates: list[float], point: float) -> tuple[float, float, float]:
    return point, percentile(estimates, 0.025), percentile(estimates, 0.975)


def fusion_counts(
    results: dict[str, dict[str, QueryRank]],
    qids: list[str],
) -> tuple[int, int, int, int, int, int]:
    required = {"dense", "bm25", "hybrid"}
    if not required.issubset(results):
        raise ValueError("fusion analysis requires dense, bm25, and hybrid")
    dense_hits = sum(0 < results["dense"][qid].rank <= 10 for qid in qids)
    bm25_hits = sum(0 < results["bm25"][qid].rank <= 10 for qid in qids)
    union_hits = sum(
        0 < results["dense"][qid].rank <= 10 or 0 < results["bm25"][qid].rank <= 10
        for qid in qids
    )
    hybrid_hits = sum(0 < results["hybrid"][qid].rank <= 10 for qid in qids)
    rescues = sum(
        results["hybrid"][qid].rank > 0
        and results["dense"][qid].rank == 0
        and results["bm25"][qid].rank == 0
        for qid in qids
    )
    harms = sum(
        results["hybrid"][qid].rank == 0
        and (results["dense"][qid].rank > 0 or results["bm25"][qid].rank > 0)
        for qid in qids
    )
    return dense_hits, bm25_hits, union_hits, hybrid_hits, rescues, harms


def strategy_transition_counts(
    from_rows: dict[str, QueryRank],
    to_rows: dict[str, QueryRank],
    qids: list[str],
) -> tuple[int, int, int, int]:
    before = sum(0 < from_rows[qid].rank <= 10 for qid in qids)
    after = sum(0 < to_rows[qid].rank <= 10 for qid in qids)
    rescues = sum(from_rows[qid].rank == 0 and to_rows[qid].rank > 0 for qid in qids)
    harms = sum(from_rows[qid].rank > 0 and to_rows[qid].rank == 0 for qid in qids)
    return before, after, rescues, harms


def write_report(
    results: dict[str, dict[str, QueryRank]],
    output: Path,
    iterations: int,
    seed: str,
    cluster_pairs: bool = False,
) -> None:
    metric_names = ["hit_rate_at_10", "mrr_at_10", "ndcg_at_10"]
    reference = next(iter(results.values()))
    slice_names = ["all"] + sorted({row.kind for row in reference.values()})
    output.mkdir(parents=True, exist_ok=True)

    ci_path = output / "retrieval_bootstrap_ci.csv"
    with ci_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["strategy", "slice", "queries", "metric", "value", "ci95_low", "ci95_high"])
        for slice_name in slice_names:
            qids = [qid for qid, row in reference.items() if slice_name == "all" or row.kind == slice_name]
            samples = bootstrap_qids(
                [reference[qid] for qid in qids],
                iterations,
                f"{seed}:{slice_name}",
                cluster_pairs and slice_name == "all",
            )
            for strategy, rows in sorted(results.items()):
                for metric_index, metric_name in enumerate(metric_names):
                    point = mean_metric(rows, qids, metric_index)
                    estimates = [mean_metric(rows, sample, metric_index) for sample in samples]
                    value, low, high = interval(estimates, point)
                    writer.writerow([
                        strategy, slice_name, len(qids), metric_name,
                        f"{value:.6f}", f"{low:.6f}", f"{high:.6f}",
                    ])

    delta_path = output / "retrieval_paired_deltas.csv"
    with delta_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow([
            "from_strategy", "to_strategy", "slice", "queries", "metric",
            "delta", "ci95_low", "ci95_high",
        ])
        for slice_name in slice_names:
            qids = [qid for qid, row in reference.items() if slice_name == "all" or row.kind == slice_name]
            samples = bootstrap_qids(
                [reference[qid] for qid in qids],
                iterations,
                f"{seed}:{slice_name}",
                cluster_pairs and slice_name == "all",
            )
            for from_strategy, to_strategy in itertools.combinations(sorted(results), 2):
                for metric_index, metric_name in enumerate(metric_names):
                    point = mean_metric(results[to_strategy], qids, metric_index) - mean_metric(
                        results[from_strategy], qids, metric_index
                    )
                    estimates = [
                        mean_metric(results[to_strategy], sample, metric_index)
                        - mean_metric(results[from_strategy], sample, metric_index)
                        for sample in samples
                    ]
                    value, low, high = interval(estimates, point)
                    writer.writerow([
                        from_strategy, to_strategy, slice_name, len(qids), metric_name,
                        f"{value:.6f}", f"{low:.6f}", f"{high:.6f}",
                    ])

    if {"dense", "bm25", "hybrid"}.issubset(results):
        transition_path = output / "fusion_transitions.csv"
        with transition_path.open("w", encoding="utf-8", newline="") as target:
            writer = csv.writer(target)
            writer.writerow([
                "slice", "queries", "dense_hits", "bm25_hits", "branch_union_hits",
                "hybrid_hits", "rescues", "harms", "net_vs_branch_union",
            ])
            for slice_name in slice_names:
                qids = [
                    qid for qid, row in reference.items()
                    if slice_name == "all" or row.kind == slice_name
                ]
                dense_hits, bm25_hits, union_hits, hybrid_hits, rescues, harms = fusion_counts(
                    results, qids
                )
                writer.writerow([
                    slice_name, len(qids), dense_hits, bm25_hits, union_hits,
                    hybrid_hits, rescues, harms, hybrid_hits - union_hits,
                ])

    transition_path = output / "retrieval_strategy_transitions.csv"
    with transition_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow([
            "from_strategy", "to_strategy", "slice", "queries", "before_hits",
            "after_hits", "rescues", "harms", "net_hits",
        ])
        for slice_name in slice_names:
            qids = [qid for qid, row in reference.items() if slice_name == "all" or row.kind == slice_name]
            for from_strategy, to_strategy in itertools.permutations(sorted(results), 2):
                before, after, rescues, harms = strategy_transition_counts(
                    results[from_strategy], results[to_strategy], qids
                )
                writer.writerow([
                    from_strategy, to_strategy, slice_name, len(qids), before,
                    after, rescues, harms, after - before,
                ])

    (output / "retrieval_bootstrap_manifest.json").write_text(json.dumps({
        "method": (
            "clustered percentile bootstrap over paired question IDs; "
            "paired resampling across strategies"
            if cluster_pairs
            else "stratified percentile bootstrap; paired resampling across strategies"
        ),
        "clusterPairs": cluster_pairs,
        "iterations": iterations,
        "seed": seed,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap confidence intervals for retrieval metrics")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", default="ragview-tax-retrieval-bootstrap-v1")
    parser.add_argument("--cluster-pairs", action="store_true")
    args = parser.parse_args()
    if args.iterations < 100:
        parser.error("iterations must be at least 100")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    write_report(
        load_results(arguments.input),
        arguments.output,
        arguments.iterations,
        arguments.seed,
        arguments.cluster_pairs,
    )
