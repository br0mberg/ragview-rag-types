package ru.brombin.ragview.rerank;

public record RerankQueryResult(
        String source,
        String qid,
        String kind,
        int pool,
        int scoredPool,
        int candidateRank,
        int rerankRank,
        double scoredPoolLatencyMs) {
}
