package ru.brombin.ragview.rerank;

import java.util.List;

public interface RerankScorer {

    List<Double> score(String query, List<String> passages);
}
