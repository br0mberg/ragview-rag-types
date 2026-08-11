package ru.brombin.ragview.metrics;

import java.util.concurrent.atomic.AtomicLong;

public final class CallCounter {

    private final AtomicLong embeddingRequests = new AtomicLong();
    private final AtomicLong embeddingInputs = new AtomicLong();
    private final AtomicLong llmCalls = new AtomicLong();

    public void countEmbedding() {
        countEmbedding(1);
    }

    public void countEmbedding(int inputs) {
        embeddingRequests.incrementAndGet();
        embeddingInputs.addAndGet(inputs);
    }

    public void countLlm() {
        llmCalls.incrementAndGet();
    }

    public long embeddingRequests() {
        return embeddingRequests.get();
    }

    public long embeddingInputs() {
        return embeddingInputs.get();
    }

    public long llmCalls() {
        return llmCalls.get();
    }

    public void reset() {
        embeddingRequests.set(0);
        embeddingInputs.set(0);
        llmCalls.set(0);
    }
}
