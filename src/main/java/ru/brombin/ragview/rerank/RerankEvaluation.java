package ru.brombin.ragview.rerank;

import java.util.List;

public record RerankEvaluation(List<RerankMetrics> metrics, List<RerankQueryResult> queries) {
}
