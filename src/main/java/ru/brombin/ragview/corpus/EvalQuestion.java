package ru.brombin.ragview.corpus;

import java.util.List;

public record EvalQuestion(
        String id,
        String question,
        String relevantDocId,
        List<String> relevantDocIds,
        String kind) {

    public String kindOrUnknown() {
        return kind == null || kind.isBlank() ? "unknown" : kind;
    }

    public List<String> relevanceSet() {
        if (relevantDocIds != null && !relevantDocIds.isEmpty()) {
            return List.copyOf(relevantDocIds);
        }
        return relevantDocId == null || relevantDocId.isBlank() ? List.of() : List.of(relevantDocId);
    }
}
