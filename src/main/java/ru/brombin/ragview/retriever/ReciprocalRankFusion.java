package ru.brombin.ragview.retriever;

import ru.brombin.ragview.strategy.RetrievedDoc;

import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

public final class ReciprocalRankFusion {

    public static final int DEFAULT_RRF_K = 60;

    private ReciprocalRankFusion() {
    }

    @SafeVarargs
    public static List<RetrievedDoc> fuse(int topK, List<RetrievedDoc>... rankings) {
        return fuse(topK, DEFAULT_RRF_K, rankings);
    }

    @SafeVarargs
    public static List<RetrievedDoc> fuse(int topK, int rrfK, List<RetrievedDoc>... rankings) {
        if (topK <= 0) {
            throw new IllegalArgumentException("topK must be positive");
        }
        if (rrfK <= 0) {
            throw new IllegalArgumentException("rrfK must be positive");
        }
        Objects.requireNonNull(rankings, "rankings");

        Map<String, Double> fused = new HashMap<>();
        for (List<RetrievedDoc> ranking : rankings) {
            Objects.requireNonNull(ranking, "ranking");
            for (int rank = 0; rank < ranking.size(); rank++) {
                String docId = ranking.get(rank).docId();
                double contribution = 1.0 / (rrfK + rank + 1);
                fused.merge(docId, contribution, Double::sum);
            }
        }
        return fused.entrySet().stream()
                .sorted(Map.Entry.<String, Double>comparingByValue().reversed()
                        .thenComparing(Map.Entry.comparingByKey()))
                .limit(topK)
                .map(item -> new RetrievedDoc(item.getKey(), item.getValue()))
                .toList();
    }
}
