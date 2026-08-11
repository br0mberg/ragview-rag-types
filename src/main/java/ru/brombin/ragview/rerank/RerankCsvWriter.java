package ru.brombin.ragview.rerank;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public final class RerankCsvWriter {

    public void writeMetrics(Path output, List<RerankMetrics> metrics) {
        List<String> lines = new ArrayList<>();
        lines.add("source,quality_pool,scored_pool,slice,queries,top_k,candidate_hit_rate,hit_rate,mrr,ndcg,scored_pool_latency_p50_ms,scored_pool_latency_p95_ms");
        for (RerankMetrics row : metrics) {
            lines.add(String.format(Locale.ROOT,
                    "%s,%d,%d,%s,%d,%d,%.6f,%.6f,%.6f,%.6f,%.3f,%.3f",
                    row.source(), row.pool(), row.scoredPool(), row.slice(), row.queries(), row.topK(),
                    row.candidateHitRate(), row.hitRate(), row.mrr(), row.ndcg(),
                    row.scoredPoolLatencyP50Ms(), row.scoredPoolLatencyP95Ms()));
        }
        write(output, lines);
    }

    public void writeQueries(Path output, List<RerankQueryResult> rows) {
        List<String> lines = new ArrayList<>();
        lines.add("source,qid,kind,quality_pool,scored_pool,candidate_rank,rerank_rank,scored_pool_latency_ms");
        for (RerankQueryResult row : rows) {
            lines.add(String.format(Locale.ROOT, "%s,%s,%s,%d,%d,%d,%d,%.3f",
                    row.source(), row.qid(), row.kind(), row.pool(), row.scoredPool(),
                    row.candidateRank(), row.rerankRank(), row.scoredPoolLatencyMs()));
        }
        write(output, lines);
    }

    private static void write(Path output, List<String> lines) {
        Path absolute = output.toAbsolutePath();
        try {
            Files.createDirectories(absolute.getParent());
            Files.write(absolute, lines);
        } catch (IOException error) {
            throw new UncheckedIOException("Cannot write " + absolute, error);
        }
    }
}
