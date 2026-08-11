package ru.brombin.ragview.metrics;

public record StrategyResult(
        String strategy,
        String slice,
        int queryCount,
        int k,
        double hitRateAtK,
        double mrrAtK,
        double ndcgAtK,
        long latencyP50Ms,
        long latencyP95Ms,
        long embeddingRequestsTotal,
        long embeddingInputsTotal,
        long llmCallsTotal,
        long indexEmbeddingRequests,
        long indexEmbeddingInputs,
        long indexLlmCalls) {
}
