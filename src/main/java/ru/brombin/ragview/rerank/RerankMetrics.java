package ru.brombin.ragview.rerank;

public record RerankMetrics(
        String source,
        int pool,
        int scoredPool,
        String slice,
        int queries,
        int topK,
        double candidateHitRate,
        double hitRate,
        double mrr,
        double ndcg,
        double scoredPoolLatencyP50Ms,
        double scoredPoolLatencyP95Ms) {
}
