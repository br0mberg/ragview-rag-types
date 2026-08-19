#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch==2.7.1",
#   "transformers==4.53.2",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# ///
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


MODEL_ID = "BAAI/bge-reranker-v2-m3"
MODEL_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


@dataclass(frozen=True)
class Candidate:
    doc_id: str
    first_stage_rank: int
    first_stage_score: float


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], p: int) -> float:
    ordered = sorted(values)
    index = math.ceil(p / 100 * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def rank_of(ranked_ids: list[str], relevant_ids: set[str], k: int) -> int:
    for index, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in relevant_ids:
            return index
    return 0


def aggregate(ranks: list[int], k: int) -> dict[str, float]:
    count = len(ranks)
    return {
        "hit_rate_at_k": sum(rank > 0 for rank in ranks) / count,
        "mrr_at_k": sum(1 / rank for rank in ranks if rank > 0) / count,
        "ndcg_at_k": sum(1 / math.log2(rank + 1) for rank in ranks if rank > 0) / count,
    }


def parse_candidates(record: dict[str, object]) -> list[Candidate]:
    parsed: list[Candidate] = []
    for index, item in enumerate(record["cand"], start=1):
        if isinstance(item, str):
            parsed.append(Candidate(item, index, 0.0))
        else:
            parsed.append(Candidate(
                str(item["docId"]),
                int(item.get("rank", index)),
                float(item.get("score", 0.0)),
            ))
    return parsed


def load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def load_corpus(path: Path) -> dict[str, str]:
    documents = json.loads(path.read_text(encoding="utf-8"))
    return {
        document["id"]: f"{document['title']}. {document['text']}"
        for document in documents
    }


def rerank_at_pool(
    candidates: list[Candidate],
    scores: list[float],
    pool: int,
) -> list[tuple[Candidate, float]]:
    selected = list(zip(candidates[:pool], scores[:pool], strict=True))
    return sorted(selected, key=lambda pair: (-pair[1], pair[0].first_stage_rank))


class CrossEncoder:
    def __init__(
        self,
        model_path: str,
        revision: str,
        batch_size: int,
        max_length: int,
        precision: str,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._batch_size = batch_size
        self._max_length = max_length
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if precision == "fp16" and self._device.type != "cuda":
            raise RuntimeError("fp16 reranking requires CUDA; use --precision fp32 on CPU")
        self._precision = precision
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
        dtype = torch.float16 if precision == "fp16" else torch.float32
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            revision=revision,
            torch_dtype=dtype,
        ).to(self._device)
        self._model.eval()

    @property
    def device(self) -> str:
        return str(self._device)

    @property
    def precision(self) -> str:
        return self._precision

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def max_length(self) -> int:
        return self._max_length

    def score(self, query: str, passages: list[str]) -> list[float]:
        scores: list[float] = []
        with self._torch.inference_mode():
            for start in range(0, len(passages), self._batch_size):
                batch = [[query, passage] for passage in passages[start:start + self._batch_size]]
                inputs = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self._max_length,
                    return_tensors="pt",
                ).to(self._device)
                logits = self._model(**inputs, return_dict=True).logits.view(-1).float().cpu()
                scores.extend(logits.tolist())
        return scores


def run(args: argparse.Namespace) -> None:
    records = load_jsonl(args.candidates)
    corpus = load_corpus(args.corpus)
    pools = sorted(set(args.pools))
    reranker = CrossEncoder(
        args.model, args.revision, args.batch_size, args.max_length, args.precision
    )
    args.output.mkdir(parents=True, exist_ok=True)
    raw_path = args.output / "rerank_raw.jsonl"
    measurements: dict[tuple[int, str], list[tuple[int, int, bool]]] = {}
    full_pool_latencies: dict[str, list[float]] = defaultdict(list)

    with raw_path.open("w", encoding="utf-8") as raw:
        for record in records:
            relevant = set(record["rel"] if isinstance(record["rel"], list) else [record["rel"]])
            candidates = parse_candidates(record)[:max(pools)]
            kind = str(record.get("kind", "unknown"))
            passages = [corpus[candidate.doc_id] for candidate in candidates]
            started = time.perf_counter_ns()
            scores = reranker.score(str(record["q"]), passages)
            latency_ms = (time.perf_counter_ns() - started) / 1_000_000
            full_pool_latencies["all"].append(latency_ms)
            full_pool_latencies[kind].append(latency_ms)
            derived: dict[str, object] = {}
            for pool in pools:
                selected = candidates[:pool]
                ranked = rerank_at_pool(candidates, scores, pool)
                ranked_ids = [candidate.doc_id for candidate, _ in ranked]
                rank = rank_of(ranked_ids, relevant, args.top_k)
                candidate_hit = any(candidate.doc_id in relevant for candidate in selected)
                measurement = (rank, len(selected), candidate_hit)
                measurements.setdefault((pool, "all"), []).append(measurement)
                measurements.setdefault((pool, kind), []).append(measurement)
                derived[str(pool)] = {
                    "effectivePool": len(selected),
                    "relevantRank": rank,
                    "candidateHit": candidate_hit,
                    "top10": [candidate.doc_id for candidate, _ in ranked[:args.top_k]],
                }
            raw.write(json.dumps({
                "qid": record["qid"],
                "kind": kind,
                "topK": args.top_k,
                "scoredPool": len(candidates),
                "latencyMs": round(latency_ms, 3),
                "scores": [
                    {
                        "docId": candidate.doc_id,
                        "rerankScore": score,
                        "firstStageRank": candidate.first_stage_rank,
                    }
                    for candidate, score in zip(candidates, scores, strict=True)
                ],
                "derived": derived,
            }, ensure_ascii=False) + "\n")

    metrics_path = args.output / "rerank_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["pool", "slice", "queries", "top_k", "metric", "value"])
        for (pool, slice_name), values in sorted(measurements.items()):
            ranks = [rank for rank, _, _ in values]
            effective_pools = [effective for _, effective, _ in values]
            metrics = aggregate(ranks, args.top_k) | {
                "candidate_hit_rate_at_pool": sum(hit for _, _, hit in values) / len(values),
                "effective_pool_min": min(effective_pools),
                "effective_pool_mean": statistics.fmean(effective_pools),
                "effective_pool_max": max(effective_pools),
            }
            if pool == max(pools):
                latencies = full_pool_latencies[slice_name]
                metrics |= {
                    "scoring_latency_p50_ms": percentile(latencies, 50),
                    "scoring_latency_p95_ms": percentile(latencies, 95),
                    "scoring_latency_mean_ms": statistics.fmean(latencies),
                }
            for name, value in metrics.items():
                writer.writerow([pool, slice_name, len(values), args.top_k, name, f"{value:.6f}"])

    manifest = {
        "model": args.model,
        "revision": args.revision,
        "device": reranker.device,
        "precision": reranker.precision,
        "batchSize": args.batch_size,
        "maxLengthTokens": args.max_length,
        "topK": args.top_k,
        "pools": pools,
        "scoringMode": "score each pair once at max available pool; derive smaller pools by first-stage rank",
        "candidatesSha256": sha256(args.candidates),
        "corpusSha256": sha256(args.corpus),
        "rawSha256": sha256(raw_path),
    }
    (args.output / "rerank_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerank saved retrieval candidates")
    parser.add_argument("--candidates", type=Path, default=Path("results/hybrid_candidates.jsonl"))
    parser.add_argument("--corpus", type=Path, default=Path("data/tax-code-snapshot/corpus.json"))
    parser.add_argument("--output", type=Path, default=Path("results/rerank"))
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--pools", type=int, nargs="+", default=[10, 20, 30, 50, 100])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    args = parser.parse_args()
    if args.top_k <= 0 or args.batch_size <= 0 or args.max_length <= 0:
        parser.error("top-k, batch-size and max-length must be positive")
    if any(pool < args.top_k for pool in args.pools):
        parser.error("every pool must be at least top-k")
    return args


if __name__ == "__main__":
    run(parse_args())
