package ru.brombin.ragview.metrics;

import java.util.List;

public record StrategyEvaluation(
        List<StrategyResult> aggregates,
        List<QueryResult> queries) {
}
