package ru.brombin.ragview.metrics;

import lombok.extern.slf4j.Slf4j;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

@Slf4j
public class ComparisonCsvWriter {

    public void write(Path output, List<StrategyResult> results) {
        List<String> lines = new ArrayList<>();
        lines.add("strategy,slice,queries,k,metric,value");
        for (StrategyResult r : results) {
            add(lines, r, "hit_rate_at_k", r.hitRateAtK());
            add(lines, r, "mrr_at_k", r.mrrAtK());
            add(lines, r, "ndcg_at_k", r.ndcgAtK());
            add(lines, r, "latency_p50_ms", r.latencyP50Ms());
            add(lines, r, "latency_p95_ms", r.latencyP95Ms());
            add(lines, r, "embedding_requests_total", r.embeddingRequestsTotal());
            add(lines, r, "embedding_inputs_total", r.embeddingInputsTotal());
            add(lines, r, "llm_calls_total", r.llmCallsTotal());
            add(lines, r, "index_embedding_requests", r.indexEmbeddingRequests());
            add(lines, r, "index_embedding_inputs", r.indexEmbeddingInputs());
            add(lines, r, "index_llm_calls", r.indexLlmCalls());
        }
        try {
            Path absolute = output.toAbsolutePath();
            Files.createDirectories(absolute.getParent());
            Files.write(absolute, lines);
            log.info("comparison written to {}", absolute);
        } catch (IOException e) {
            throw new UncheckedIOException("Cannot write " + output, e);
        }
    }

    public void writePerQuery(Path output, List<QueryResult> results) {
        List<String> lines = new ArrayList<>();
        lines.add("strategy,qid,kind,rank,latency_ms");
        for (QueryResult result : results) {
            lines.add("%s,%s,%s,%d,%d".formatted(
                    result.strategy(), result.qid(), result.kind(), result.rank(), result.latencyMs()));
        }
        try {
            Path absolute = output.toAbsolutePath();
            Files.createDirectories(absolute.getParent());
            Files.write(absolute, lines);
            log.info("per-query results written to {}", absolute);
        } catch (IOException e) {
            throw new UncheckedIOException("Cannot write " + output, e);
        }
    }

    private static void add(List<String> lines, StrategyResult r, String metric, double value) {
        lines.add("%s,%s,%d,%d,%s,%s".formatted(
                r.strategy(), r.slice(), r.queryCount(), r.k(), metric, format(value)));
    }

    private static String format(double value) {
        if (value == Math.rint(value)) {
            return String.valueOf((long) value);
        }
        return String.format(Locale.US, "%.4f", value);
    }
}
