#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class QueryResult:
    qid: str
    kind: str
    ranks: dict[int, int]
    source: str = "hybrid"
    candidate_ranks: dict[int, int] = field(default_factory=dict)


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


def load_jsonl(path: Path) -> list[QueryResult]:
    results = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            derived = record["derived"]
            results.append(QueryResult(
                qid=str(record["qid"]),
                kind=str(record["kind"]),
                ranks={int(pool): int(value["relevantRank"]) for pool, value in derived.items()},
                source=str(record.get("source", "hybrid")),
                candidate_ranks={
                    int(pool): int(value.get("candidateRank", 0))
                    for pool, value in derived.items()
                },
            ))
    return results


def load_csv(path: Path) -> list[QueryResult]:
    grouped: dict[tuple[str, str], dict[str, object]] = {}
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        required = {"source", "qid", "kind", "quality_pool", "candidate_rank", "rerank_rank"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"rerank CSV is missing columns: {sorted(required - set(reader.fieldnames or []))}")
        for row in reader:
            key = (row["source"], row["qid"])
            item = grouped.setdefault(key, {
                "kind": row["kind"],
                "ranks": {},
                "candidate_ranks": {},
                "scored_pool": row.get("scored_pool", ""),
            })
            if item["kind"] != row["kind"] or item["scored_pool"] != row.get("scored_pool", ""):
                raise ValueError(f"inconsistent rerank rows for {key[0]}/{key[1]}")
            pool = int(row["quality_pool"])
            ranks = item["ranks"]
            candidate_ranks = item["candidate_ranks"]
            assert isinstance(ranks, dict) and isinstance(candidate_ranks, dict)
            if pool in ranks:
                raise ValueError(f"duplicate rerank row for {key[0]}/{key[1]}/pool={pool}")
            ranks[pool] = int(row["rerank_rank"])
            candidate_ranks[pool] = int(row["candidate_rank"])

    results = [
        QueryResult(qid, str(item["kind"]), item["ranks"], source, item["candidate_ranks"])
        for (source, qid), item in sorted(grouped.items())
    ]
    if not results:
        raise ValueError("rerank input is empty")
    source_qids: dict[str, set[str]] = {}
    source_pools: dict[str, set[int]] = {}
    for result in results:
        source_qids.setdefault(result.source, set()).add(result.qid)
        expected = source_pools.setdefault(result.source, set(result.ranks))
        if set(result.ranks) != expected or set(result.candidate_ranks) != expected:
            raise ValueError(f"pool coverage mismatch for {result.source}/{result.qid}")
    if len({frozenset(qids) for qids in source_qids.values()}) != 1:
        raise ValueError("candidate sources do not cover the same qids")
    return results


def load_results(path: Path) -> list[QueryResult]:
    with path.open(encoding="utf-8") as source:
        first = next((character for character in source.read(4096) if not character.isspace()), "")
    return load_jsonl(path) if first == "{" else load_csv(path)


def paired_result_groups(results: list[QueryResult]) -> tuple[list[str], dict[str, dict[str, int]]]:
    kinds = sorted({result.kind for result in results})
    if len(kinds) < 2:
        raise ValueError("clustered bootstrap requires at least two question kinds")
    groups: dict[str, dict[str, int]] = {}
    for index, result in enumerate(results):
        prefix = f"{result.kind}-"
        if not result.qid.startswith(prefix):
            raise ValueError(f"qid does not encode its kind: {result.qid}")
        group_id = result.qid[len(prefix):]
        group = groups.setdefault(group_id, {})
        if result.kind in group:
            raise ValueError(f"duplicate paired qid group: {group_id}/{result.kind}")
        group[result.kind] = index
    expected = set(kinds)
    incomplete = [group_id for group_id, group in groups.items() if set(group) != expected]
    if incomplete:
        raise ValueError(f"incomplete paired qid groups: {', '.join(sorted(incomplete)[:3])}")
    return kinds, groups


def bootstrap_indices(
    results: list[QueryResult],
    iterations: int,
    seed: str,
    cluster_pairs: bool = False,
) -> list[list[int]]:
    rng = random.Random(seed)
    if cluster_pairs:
        kinds, groups = paired_result_groups(results)
        group_ids = sorted(groups)
        return [
            [
                groups[group_id][kind]
                for group_id in (rng.choice(group_ids) for _ in group_ids)
                for kind in kinds
            ]
            for _ in range(iterations)
        ]
    by_kind: dict[str, list[int]] = {}
    for index, result in enumerate(results):
        by_kind.setdefault(result.kind, []).append(index)
    return [
        [rng.choice(indices) for kind in sorted(by_kind) for indices in [by_kind[kind]] for _ in indices]
        for _ in range(iterations)
    ]


def mean_metric(results: list[QueryResult], indices: list[int], pool: int, metric_index: int) -> float:
    return sum(contributions(results[index].ranks[pool])[metric_index] for index in indices) / len(indices)


def confidence_interval(
    results: list[QueryResult],
    samples: list[list[int]],
    pool: int,
    metric_index: int,
) -> tuple[float, float, float]:
    point = mean_metric(results, list(range(len(results))), pool, metric_index)
    estimates = [mean_metric(results, sample, pool, metric_index) for sample in samples]
    return point, percentile(estimates, 0.025), percentile(estimates, 0.975)


def paired_delta_interval(
    results: list[QueryResult],
    samples: list[list[int]],
    lower_pool: int,
    upper_pool: int,
    metric_index: int,
) -> tuple[float, float, float]:
    indices = list(range(len(results)))
    point = mean_metric(results, indices, upper_pool, metric_index) - mean_metric(
        results, indices, lower_pool, metric_index
    )
    estimates = [
        mean_metric(results, sample, upper_pool, metric_index)
        - mean_metric(results, sample, lower_pool, metric_index)
        for sample in samples
    ]
    return point, percentile(estimates, 0.025), percentile(estimates, 0.975)


def transition_counts(results: list[QueryResult], pool: int) -> tuple[int, int, int, int]:
    before = sum(0 < result.candidate_ranks.get(pool, 0) <= 10 for result in results)
    after = sum(0 < result.ranks[pool] <= 10 for result in results)
    rescues = sum(
        (result.candidate_ranks.get(pool, 0) == 0 or result.candidate_ranks[pool] > 10)
        and 0 < result.ranks[pool] <= 10
        for result in results
    )
    harms = sum(
        0 < result.candidate_ranks.get(pool, 0) <= 10
        and (result.ranks[pool] == 0 or result.ranks[pool] > 10)
        for result in results
    )
    return before, after, rescues, harms


def write_report(
    results: list[QueryResult],
    output: Path,
    iterations: int,
    seed: str,
    input_path: Path | None = None,
    cluster_pairs: bool = False,
) -> None:
    metric_names = ["hit_rate_at_10", "mrr_at_10", "ndcg_at_10"]
    output.mkdir(parents=True, exist_ok=True)
    by_source = {
        source: [result for result in results if result.source == source]
        for source in sorted({result.source for result in results})
    }

    ci_path = output / "rerank_bootstrap_ci.csv"
    with ci_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["source", "pool", "slice", "queries", "metric", "value", "ci95_low", "ci95_high"])
        for source, source_results in by_source.items():
            pools = sorted(source_results[0].ranks)
            slices = [("all", source_results)] + [
                (kind, [result for result in source_results if result.kind == kind])
                for kind in sorted({result.kind for result in source_results})
            ]
            for slice_name, subset in slices:
                samples = bootstrap_indices(
                    subset,
                    iterations,
                    f"{seed}:{source}:{slice_name}",
                    cluster_pairs and slice_name == "all",
                )
                for pool in pools:
                    for metric_index, metric_name in enumerate(metric_names):
                        point, low, high = confidence_interval(subset, samples, pool, metric_index)
                        writer.writerow([
                            source, pool, slice_name, len(subset), metric_name,
                            f"{point:.6f}", f"{low:.6f}", f"{high:.6f}",
                        ])

    delta_path = output / "rerank_paired_deltas.csv"
    with delta_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow([
            "source", "from_pool", "to_pool", "slice", "queries", "metric",
            "delta", "ci95_low", "ci95_high",
        ])
        for source, source_results in by_source.items():
            pools = sorted(source_results[0].ranks)
            slices = [("all", source_results)] + [
                (kind, [result for result in source_results if result.kind == kind])
                for kind in sorted({result.kind for result in source_results})
            ]
            for slice_name, subset in slices:
                samples = bootstrap_indices(
                    subset,
                    iterations,
                    f"{seed}:{source}:{slice_name}",
                    cluster_pairs and slice_name == "all",
                )
                for lower_pool, upper_pool in zip(pools, pools[1:]):
                    for metric_index, metric_name in enumerate(metric_names):
                        point, low, high = paired_delta_interval(
                            subset, samples, lower_pool, upper_pool, metric_index
                        )
                        writer.writerow([
                            source, lower_pool, upper_pool, slice_name, len(subset), metric_name,
                            f"{point:.6f}", f"{low:.6f}", f"{high:.6f}",
                        ])

    transition_path = output / "rerank_transitions.csv"
    with transition_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow([
            "source", "pool", "slice", "queries", "before_hits", "after_hits",
            "rescues", "harms", "net_hits",
        ])
        for source, source_results in by_source.items():
            pools = sorted(source_results[0].ranks)
            slices = [("all", source_results)] + [
                (kind, [result for result in source_results if result.kind == kind])
                for kind in sorted({result.kind for result in source_results})
            ]
            for slice_name, subset in slices:
                for pool in pools:
                    before, after, rescues, harms = transition_counts(subset, pool)
                    writer.writerow([
                        source, pool, slice_name, len(subset), before, after,
                        rescues, harms, after - before,
                    ])

    (output / "bootstrap_manifest.json").write_text(json.dumps({
        "method": (
            "clustered percentile bootstrap over paired question IDs; "
            "paired resampling for pool deltas"
            if cluster_pairs
            else "stratified percentile bootstrap; paired resampling for pool deltas"
        ),
        "clusterPairs": cluster_pairs,
        "iterations": iterations,
        "seed": seed,
        "input": str(input_path) if input_path else None,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap confidence intervals for reranker metrics")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", default="ragview-tax-bootstrap-v1")
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
        arguments.input,
        arguments.cluster_pairs,
    )
