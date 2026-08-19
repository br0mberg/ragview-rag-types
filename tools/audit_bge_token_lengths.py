#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "transformers==4.53.2",
# ]
# ///

import argparse
import csv
import hashlib
import json
from pathlib import Path


MODEL = "BAAI/bge-reranker-v2-m3"
REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def indexable_text(document: dict) -> str:
    title = (document.get("title") or "").strip()
    text = (document.get("text") or "").strip()
    if not title or text.startswith(title):
        return text
    return f"{title}. {text}"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
    return rows


def require_unique(rows: list[dict], field: str, label: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        value = row.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} contains an invalid {field}")
        if value in result:
            raise ValueError(f"{label} contains duplicate {field}: {value}")
        result[value] = row
    return result


def count_pairs(tokenizer, pairs: list[tuple[str, str]], batch_size: int) -> list[int]:
    lengths = []
    for start in range(0, len(pairs), batch_size):
        batch = [[query, passage] for query, passage in pairs[start:start + batch_size]]
        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )
        lengths.extend(len(input_ids) for input_ids in encoded["input_ids"])
    return lengths


def summarize(lengths: list[int]) -> dict[str, int]:
    if not lengths:
        raise ValueError("Cannot summarize an empty token-length list")
    ordered = sorted(lengths)
    return {
        "pairs": len(ordered),
        "over512": sum(length > 512 for length in ordered),
        "over1024": sum(length > 1024 for length in ordered),
        "min": ordered[0],
        "max": ordered[-1],
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_lengths(path: Path) -> list[int]:
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    try:
        return [int(row["pair_tokens"]) for row in rows]
    except (KeyError, ValueError) as error:
        raise ValueError(f"Invalid pair token length CSV: {path}") from error


def verify_summary(summary_path: Path) -> None:
    summary_path = summary_path.resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for summary_field, artifact_field in (
        ("candidatePairs", "candidatePairTokenLengths"),
        ("targetPairs", "targetPairTokenLengths"),
    ):
        artifact = summary["artifacts"][artifact_field]
        path = summary_path.parent / artifact["filename"]
        if sha256(path) != artifact["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {path}")
        if summarize(read_lengths(path)) != summary[summary_field]:
            raise ValueError(f"Token-length summary mismatch: {summary_field}")


def run(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    questions_path = args.questions.resolve()
    corpus_path = args.corpus.resolve()
    labels_path = args.labels.resolve()
    rankings_path = args.rankings.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")

    questions = require_unique(
        json.loads(questions_path.read_text(encoding="utf-8")), "id", "questions")
    corpus = require_unique(
        json.loads(corpus_path.read_text(encoding="utf-8")), "id", "corpus")
    label_set = json.loads(labels_path.read_text(encoding="utf-8"))
    labels = require_unique(label_set["labels"], "qid", "labels")
    rankings = require_unique(read_jsonl(rankings_path), "qid", "rankings")

    if len(questions) != 240 or len(labels) != 240 or len(rankings) != 240:
        raise ValueError("The frozen frame must contain exactly 240 questions")
    if set(questions) != set(labels) or set(questions) != set(rankings):
        raise ValueError("Question, label and ranking qids differ")

    candidate_keys = []
    candidate_pairs = []
    for qid in sorted(rankings):
        ranking = rankings[qid].get("bge")
        if not isinstance(ranking, list) or len(ranking) != 50:
            raise ValueError(f"Expected 50 BGE candidates for {qid}")
        seen_ranks = set()
        seen_docs = set()
        for candidate in sorted(ranking, key=lambda item: item["firstStageRank"]):
            rank = candidate.get("firstStageRank")
            doc_id = candidate.get("docId")
            if rank in seen_ranks or doc_id in seen_docs:
                raise ValueError(f"Duplicate candidate rank or document for {qid}")
            if doc_id not in corpus:
                raise ValueError(f"Unknown candidate document: {doc_id}")
            seen_ranks.add(rank)
            seen_docs.add(doc_id)
            candidate_keys.append((qid, rank, doc_id))
            candidate_pairs.append((questions[qid]["question"], indexable_text(corpus[doc_id])))
        if seen_ranks != set(range(1, 51)):
            raise ValueError(f"Candidate ranks must cover 1..50 for {qid}")

    target_keys = []
    target_pairs = []
    for qid in sorted(labels):
        label = labels[qid]
        if label.get("status") != "eligible_cited_clause":
            continue
        target_doc_ids = label.get("targetDocIds")
        if not isinstance(target_doc_ids, list) or not target_doc_ids:
            raise ValueError(f"Eligible label has no target documents: {qid}")
        for doc_id in sorted(set(target_doc_ids)):
            if doc_id not in corpus:
                raise ValueError(f"Unknown target document: {doc_id}")
            target_keys.append((qid, doc_id))
            target_pairs.append((questions[qid]["question"], indexable_text(corpus[doc_id])))

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    candidate_lengths = count_pairs(tokenizer, candidate_pairs, args.batch_size)
    target_lengths = count_pairs(tokenizer, target_pairs, args.batch_size)

    candidate_rows = [
        {
            "qid": qid,
            "first_stage_rank": rank,
            "doc_id": doc_id,
            "pair_tokens": length,
            "over_512": str(length > 512).lower(),
            "over_1024": str(length > 1024).lower(),
        }
        for (qid, rank, doc_id), length in zip(candidate_keys, candidate_lengths, strict=True)
    ]
    target_rows = [
        {
            "qid": qid,
            "doc_id": doc_id,
            "pair_tokens": length,
            "over_512": str(length > 512).lower(),
            "over_1024": str(length > 1024).lower(),
        }
        for (qid, doc_id), length in zip(target_keys, target_lengths, strict=True)
    ]

    output.mkdir(parents=True)
    candidate_output = output / "retrieval-pair-token-lengths.csv"
    target_output = output / "target-pair-token-lengths.csv"
    write_csv(
        candidate_output,
        ["qid", "first_stage_rank", "doc_id", "pair_tokens", "over_512", "over_1024"],
        candidate_rows,
    )
    write_csv(
        target_output,
        ["qid", "doc_id", "pair_tokens", "over_512", "over_1024"],
        target_rows,
    )

    summary = {
        "schemaVersion": 1,
        "method": "AutoTokenizer pair encoding with add_special_tokens=true, padding=false, truncation=false",
        "tokenizer": {
            "model": args.model,
            "revision": args.revision,
            "implementation": type(tokenizer).__name__,
            "transformersVersion": __import__("transformers").__version__,
        },
        "inputs": {
            "questionsSha256": sha256(questions_path),
            "corpusSha256": sha256(corpus_path),
            "labelsSha256": sha256(labels_path),
            "rankingsSha256": sha256(rankings_path),
        },
        "candidatePairs": summarize(candidate_lengths),
        "targetPairs": summarize(target_lengths),
        "artifacts": {
            "candidatePairTokenLengths": {
                "filename": candidate_output.name,
                "sha256": sha256(candidate_output),
            },
            "targetPairTokenLengths": {
                "filename": target_output.name,
                "sha256": sha256(target_output),
            },
        },
        "publication": {
            "containsQuestionOrAnswerText": False,
            "containsCorpusText": False,
            "containsStructuralDocumentIdentifiers": True,
        },
    }
    summary_path = output / "token-length-summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-summary", type=Path)
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--rankings", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.verify_summary:
        verify_summary(arguments.verify_summary)
    else:
        required = ("questions", "corpus", "labels", "rankings", "output")
        missing = [name for name in required if getattr(arguments, name) is None]
        if missing:
            raise SystemExit("Missing arguments: " + ", ".join(missing))
        run(arguments)
