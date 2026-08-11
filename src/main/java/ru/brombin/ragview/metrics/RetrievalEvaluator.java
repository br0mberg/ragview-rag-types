package ru.brombin.ragview.metrics;

import ru.brombin.ragview.corpus.EvalQuestion;
import ru.brombin.ragview.strategy.RagStrategy;
import ru.brombin.ragview.strategy.RetrievedDoc;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

public class RetrievalEvaluator {

    public List<StrategyResult> evaluate(RagStrategy strategy,
                                         List<EvalQuestion> questions,
                                         int k,
                                         CallCounter counter,
                                         long indexEmbeddingRequests,
                                         long indexEmbeddingInputs,
                                         long indexLlmCalls) {
        return evaluateDetailed(
                strategy, questions, k, counter,
                indexEmbeddingRequests, indexEmbeddingInputs, indexLlmCalls).aggregates();
    }

    public StrategyEvaluation evaluateDetailed(RagStrategy strategy,
                                                List<EvalQuestion> questions,
                                                int k,
                                                CallCounter counter,
                                                long indexEmbeddingRequests,
                                                long indexEmbeddingInputs,
                                                long indexLlmCalls) {
        if (questions.isEmpty()) {
            throw new IllegalArgumentException("Evaluation requires at least one question");
        }

        long embeddingRequestsBefore = counter.embeddingRequests();
        long embeddingInputsBefore = counter.embeddingInputs();
        long llmCallsBefore = counter.llmCalls();
        List<QueryMeasurement> measurements = new ArrayList<>(questions.size());
        List<QueryResult> queryResults = new ArrayList<>(questions.size());
        for (EvalQuestion question : questions) {
            long start = System.nanoTime();
            List<RetrievedDoc> retrieved = strategy.retrieve(question.question(), k);
            long latencyMs = (System.nanoTime() - start) / 1_000_000;
            int rank = rankOf(retrieved, question.relevanceSet());
            measurements.add(new QueryMeasurement(question.kindOrUnknown(), rank, latencyMs));
            queryResults.add(new QueryResult(
                    strategy.name(), question.id(), question.kindOrUnknown(), rank, latencyMs));
        }

        List<StrategyResult> results = new ArrayList<>();
        long embeddingRequestsTotal = indexEmbeddingRequests
                + counter.embeddingRequests() - embeddingRequestsBefore;
        long embeddingInputsTotal = indexEmbeddingInputs
                + counter.embeddingInputs() - embeddingInputsBefore;
        long llmCallsTotal = indexLlmCalls + counter.llmCalls() - llmCallsBefore;
        results.add(aggregate(
                strategy.name(), "all", measurements, k,
                embeddingRequestsTotal, embeddingInputsTotal, llmCallsTotal,
                indexEmbeddingRequests, indexEmbeddingInputs, indexLlmCalls));

        Set<String> kinds = new LinkedHashSet<>();
        questions.forEach(question -> kinds.add(question.kindOrUnknown()));
        for (String kind : kinds) {
            List<QueryMeasurement> slice = measurements.stream()
                    .filter(measurement -> measurement.kind().equals(kind))
                    .toList();
            results.add(aggregate(
                    strategy.name(), kind, slice, k,
                    embeddingRequestsTotal, embeddingInputsTotal, llmCallsTotal,
                    indexEmbeddingRequests, indexEmbeddingInputs, indexLlmCalls));
        }
        return new StrategyEvaluation(List.copyOf(results), List.copyOf(queryResults));
    }

    private static StrategyResult aggregate(String strategy,
                                            String slice,
                                            List<QueryMeasurement> measurements,
                                            int k,
                                            long embeddingRequestsTotal,
                                            long embeddingInputsTotal,
                                            long llmCallsTotal,
                                            long indexEmbeddingRequests,
                                            long indexEmbeddingInputs,
                                            long indexLlmCalls) {
        int hits = 0;
        double reciprocalRankSum = 0.0;
        double ndcgSum = 0.0;
        List<Long> latencies = new ArrayList<>(measurements.size());
        for (QueryMeasurement measurement : measurements) {
            int rank = measurement.rank();
            latencies.add(measurement.latencyMs());
            if (rank > 0 && rank <= k) {
                hits++;
                reciprocalRankSum += 1.0 / rank;
                ndcgSum += 1.0 / log2(rank + 1);
            }
        }
        latencies.sort(Long::compareTo);
        int count = measurements.size();

        return new StrategyResult(
                strategy,
                slice,
                count,
                k,
                (double) hits / count,
                reciprocalRankSum / count,
                ndcgSum / count,
                percentile(latencies, 50),
                percentile(latencies, 95),
                embeddingRequestsTotal,
                embeddingInputsTotal,
                llmCallsTotal,
                indexEmbeddingRequests,
                indexEmbeddingInputs,
                indexLlmCalls);
    }

    private static int rankOf(List<RetrievedDoc> retrieved, List<String> relevantDocIds) {
        for (int i = 0; i < retrieved.size(); i++) {
            if (relevantDocIds.contains(retrieved.get(i).docId())) {
                return i + 1;
            }
        }
        return 0;
    }

    private static double log2(int value) {
        return Math.log(value) / Math.log(2);
    }

    private static long percentile(List<Long> sorted, int percentile) {
        int index = (int) Math.ceil(percentile / 100.0 * sorted.size()) - 1;
        return sorted.get(Math.max(0, Math.min(index, sorted.size() - 1)));
    }

    private record QueryMeasurement(String kind, int rank, long latencyMs) {
    }
}
