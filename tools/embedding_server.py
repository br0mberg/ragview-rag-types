#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "fastapi==0.116.1",
#   "torch==2.7.1",
#   "transformers==4.53.2",
#   "uvicorn==0.35.0",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# ///
import argparse
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ModelProfile:
    alias: str
    model_id: str
    revision: str
    pooling: Literal["mean", "cls"]
    dimensions: int
    max_length: int = 512


PROFILES = {
    "berta": ModelProfile(
        alias="berta",
        model_id="sergeyzh/BERTA",
        revision="914c8c8aed14042ed890fc2c662d5e9e66b2faa7",
        pooling="mean",
        dimensions=768,
    ),
    "frida": ModelProfile(
        alias="frida",
        model_id="ai-forever/FRIDA",
        revision="aed004da8d09c33f6a51240fd9f5bcc625225b67",
        pooling="cls",
        dimensions=1536,
    ),
}

QUERY_PREFIX = "search_query: "
DOCUMENT_PREFIX = "search_document: "


def prefixed(texts: list[str], kind: Literal["query", "document"]) -> list[str]:
    prefix = QUERY_PREFIX if kind == "query" else DOCUMENT_PREFIX
    return [prefix + text for text in texts]


class Encoder:
    def __init__(self, profile: ModelProfile, batch_size: int, device: str) -> None:
        import torch
        import torch.nn.functional as functional
        from transformers import AutoModel, AutoTokenizer, T5EncoderModel

        self.profile = profile
        self.batch_size = batch_size
        self.torch = torch
        self.functional = functional
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(profile.model_id, revision=profile.revision)
        model_type = T5EncoderModel if profile.alias == "frida" else AutoModel
        self.model = model_type.from_pretrained(profile.model_id, revision=profile.revision).to(self.device)
        self.model.eval()

    def encode(self, texts: list[str], kind: Literal["query", "document"]) -> list[list[float]]:
        vectors: list[list[float]] = []
        inputs = prefixed(texts, kind)
        with self.torch.inference_mode():
            for start in range(0, len(inputs), self.batch_size):
                batch = self.tokenizer(
                    inputs[start:start + self.batch_size],
                    max_length=self.profile.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)
                hidden = self.model(**batch).last_hidden_state
                if self.profile.pooling == "cls":
                    pooled = hidden[:, 0]
                else:
                    mask = batch["attention_mask"].unsqueeze(-1).float()
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                normalized = self.functional.normalize(pooled, p=2, dim=1)
                vectors.extend(normalized.float().cpu().tolist())
        if vectors and len(vectors[0]) != self.profile.dimensions:
            raise RuntimeError(
                f"Unexpected vector size: {len(vectors[0])}, expected {self.profile.dimensions}"
            )
        return vectors


def create_app(
    encoder: Encoder,
    reranker=None,
    reranker_model: str | None = None,
    reranker_revision: str | None = None,
):
    from fastapi import FastAPI
    from pydantic import BaseModel, Field

    class EmbeddingRequest(BaseModel):
        texts: list[str] = Field(min_length=1, max_length=256)
        kind: Literal["query", "document"]

    class RerankRequest(BaseModel):
        query: str = Field(min_length=1)
        passages: list[str] = Field(min_length=1, max_length=256)

    app = FastAPI(title="RAGVIEW local embedding server")

    @app.get("/health")
    def health() -> dict[str, object]:
        reranker_health = None
        if reranker is not None:
            reranker_health = {
                "model": reranker_model,
                "revision": reranker_revision,
                "precision": reranker.precision,
                "device": reranker.device,
                "batchSize": reranker.batch_size,
                "maxLength": reranker.max_length,
            }
        return {
            "schemaVersion": 1,
            "status": "ok",
            "model": encoder.profile.model_id,
            "revision": encoder.profile.revision,
            "pooling": encoder.profile.pooling,
            "dimensions": encoder.profile.dimensions,
            "maxLength": encoder.profile.max_length,
            "queryPrefix": QUERY_PREFIX,
            "documentPrefix": DOCUMENT_PREFIX,
            "device": str(encoder.device),
            "reranker": reranker_health,
        }

    @app.post("/embeddings")
    def embeddings(request: EmbeddingRequest) -> dict[str, object]:
        return {
            "vectors": encoder.encode(request.texts, request.kind),
            "model": encoder.profile.model_id,
            "revision": encoder.profile.revision,
        }

    if reranker is not None:
        @app.post("/rerank")
        def rerank(request: RerankRequest) -> dict[str, object]:
            return {
                "scores": reranker.score(request.query, request.passages),
                "model": reranker_model,
                "revision": reranker_revision,
            }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve BERTA or FRIDA embeddings")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="berta")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8077)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--with-reranker", action="store_true")
    parser.add_argument("--rerank-batch-size", type=int, default=64)
    parser.add_argument("--rerank-max-length", type=int, default=1024)
    parser.add_argument("--rerank-precision", choices=["fp16", "fp32"], default="fp16")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.rerank_batch_size <= 0 or args.rerank_max_length <= 0:
        parser.error("batch sizes and max length must be positive")
    return args


def main() -> None:
    import uvicorn

    args = parse_args()
    encoder = Encoder(PROFILES[args.profile], args.batch_size, args.device)
    reranker = None
    reranker_model = None
    reranker_revision = None
    if args.with_reranker:
        from rerank_candidates import CrossEncoder, MODEL_ID, MODEL_REVISION

        reranker = CrossEncoder(
            MODEL_ID,
            MODEL_REVISION,
            args.rerank_batch_size,
            args.rerank_max_length,
            args.rerank_precision,
        )
        reranker_model = MODEL_ID
        reranker_revision = MODEL_REVISION
    uvicorn.run(
        create_app(encoder, reranker, reranker_model, reranker_revision),
        host=args.host,
        port=args.port,
        access_log=False,
    )


if __name__ == "__main__":
    main()
