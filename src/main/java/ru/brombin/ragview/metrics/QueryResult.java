package ru.brombin.ragview.metrics;

public record QueryResult(
        String strategy,
        String qid,
        String kind,
        int rank,
        long latencyMs) {
}
