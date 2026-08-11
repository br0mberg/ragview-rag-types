package ru.brombin.ragview;

import ru.brombin.ragview.metrics.CallCounter;
import ru.brombin.ragview.strategy.RagStrategy;

public record StrategySetup(
        RagStrategy strategy,
        CallCounter counter,
        long indexEmbeddingRequests,
        long indexEmbeddingInputs,
        long indexLlmCalls) {
}
