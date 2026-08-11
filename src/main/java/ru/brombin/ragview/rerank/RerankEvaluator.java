package ru.brombin.ragview.rerank;

import ru.brombin.ragview.corpus.CorpusDocument;
import ru.brombin.ragview.corpus.EvalQuestion;
import ru.brombin.ragview.strategy.RagStrategy;
import ru.brombin.ragview.strategy.RetrievedDoc;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public final class RerankEvaluator {

    public RerankEvaluation evaluate(
            RagStrategy source,
            List<EvalQuestion> questions,
            Map<String, CorpusDocument> corpus,
            List<Integer> pools,
            int topK,
            RerankScorer scorer) {
        List<Integer> normalizedPools = pools.stream().distinct().sorted().toList();
        if (questions.isEmpty() || normalizedPools.isEmpty() || topK <= 0
                || normalizedPools.stream().anyMatch(pool -> pool < topK)) {
            throw new IllegalArgumentException("Invalid rerank evaluation parameters");
        }

        int maxPool = normalizedPools.getLast();
        List<RerankQueryResult> queryResults = new ArrayList<>();
        for (EvalQuestion question : questions) {
            List<RetrievedDoc> candidates = source.retrieve(question.question(), maxPool);
            List<String> passages = candidates.stream()
                    .map(candidate -> requiredDocument(corpus, candidate.docId()).indexableText())
                    .toList();
            if (passages.isEmpty()) {
                for (int pool : normalizedPools) {
                    queryResults.add(new RerankQueryResult(
                            source.name(), question.id(), question.kindOrUnknown(), pool,
                            maxPool, 0, 0, 0));
                }
                continue;
            }

            long started = System.nanoTime();
            List<Double> scores = scorer.score(question.question(), passages);
            double latencyMs = (System.nanoTime() - started) / 1_000_000.0;
            if (scores.size() != candidates.size()) {
                throw new IllegalStateException("Reranker score count differs from candidate count");
            }

            for (int pool : normalizedPools) {
                int effectivePool = Math.min(pool, candidates.size());
                List<ScoredCandidate> selected = new ArrayList<>(effectivePool);
                for (int index = 0; index < effectivePool; index++) {
                    RetrievedDoc candidate = candidates.get(index);
                    selected.add(new ScoredCandidate(candidate.docId(), index + 1, scores.get(index)));
                }
                int candidateRank = rankOfCandidates(selected, question.relevanceSet());
                selected.sort(Comparator.comparingDouble(ScoredCandidate::score).reversed()
                        .thenComparingInt(ScoredCandidate::firstStageRank)
                        .thenComparing(ScoredCandidate::docId));
                int rerankRank = rankOfReranked(selected, question.relevanceSet(), topK);
                queryResults.add(new RerankQueryResult(
                        source.name(), question.id(), question.kindOrUnknown(), pool,
                        maxPool, candidateRank, rerankRank, latencyMs));
            }
        }

        return new RerankEvaluation(
                aggregate(source.name(), normalizedPools, questions, queryResults, topK),
                List.copyOf(queryResults));
    }

    private static List<RerankMetrics> aggregate(
            String source,
            List<Integer> pools,
            List<EvalQuestion> questions,
            List<RerankQueryResult> rows,
            int topK) {
        Set<String> slices = new LinkedHashSet<>();
        slices.add("all");
        questions.forEach(question -> slices.add(question.kindOrUnknown()));
        List<RerankMetrics> metrics = new ArrayList<>();
        for (int pool : pools) {
            for (String slice : slices) {
                List<RerankQueryResult> selected = rows.stream()
                        .filter(row -> row.pool() == pool)
                        .filter(row -> slice.equals("all") || row.kind().equals(slice))
                        .toList();
                metrics.add(metric(source, pool, slice, selected, topK));
            }
        }
        return List.copyOf(metrics);
    }

    private static RerankMetrics metric(
            String source,
            int pool,
            String slice,
            List<RerankQueryResult> rows,
            int topK) {
        int count = rows.size();
        double candidateHits = rows.stream().filter(row -> row.candidateRank() > 0).count();
        double hits = rows.stream().filter(row -> row.rerankRank() > 0).count();
        double mrr = rows.stream().filter(row -> row.rerankRank() > 0)
                .mapToDouble(row -> 1.0 / row.rerankRank()).sum();
        double ndcg = rows.stream().filter(row -> row.rerankRank() > 0)
                .mapToDouble(row -> 1.0 / log2(row.rerankRank() + 1)).sum();
        List<Double> latencies = rows.stream()
                .map(RerankQueryResult::scoredPoolLatencyMs).sorted().toList();
        int scoredPool = rows.stream().mapToInt(RerankQueryResult::scoredPool).distinct()
                .reduce((left, right) -> {
                    throw new IllegalStateException("Mixed scored pools in one metric slice");
                })
                .orElse(0);
        return new RerankMetrics(
                source, pool, scoredPool, slice, count, topK,
                candidateHits / count, hits / count, mrr / count, ndcg / count,
                percentile(latencies, 50), percentile(latencies, 95));
    }

    private static CorpusDocument requiredDocument(Map<String, CorpusDocument> corpus, String docId) {
        CorpusDocument document = corpus.get(docId);
        if (document == null) {
            throw new IllegalArgumentException("Unknown candidate document: " + docId);
        }
        return document;
    }

    private static int rankOfCandidates(List<ScoredCandidate> candidates, List<String> relevant) {
        for (ScoredCandidate candidate : candidates) {
            if (relevant.contains(candidate.docId())) {
                return candidate.firstStageRank();
            }
        }
        return 0;
    }

    private static int rankOfReranked(List<ScoredCandidate> candidates, List<String> relevant, int topK) {
        for (int index = 0; index < Math.min(topK, candidates.size()); index++) {
            if (relevant.contains(candidates.get(index).docId())) {
                return index + 1;
            }
        }
        return 0;
    }

    private static double percentile(List<Double> sorted, int percentile) {
        if (sorted.isEmpty()) {
            return 0;
        }
        int index = (int) Math.ceil(percentile / 100.0 * sorted.size()) - 1;
        return sorted.get(Math.max(0, Math.min(index, sorted.size() - 1)));
    }

    private static double log2(int value) {
        return Math.log(value) / Math.log(2);
    }

    private record ScoredCandidate(String docId, int firstStageRank, double score) {
    }
}
