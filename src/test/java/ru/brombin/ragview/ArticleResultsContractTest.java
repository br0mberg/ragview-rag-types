package ru.brombin.ragview;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

class ArticleResultsContractTest {

    private static final Path RESULTS = Path.of("results", "article");
    private static final ObjectMapper JSON = new ObjectMapper();

    @Test
    void manifestShouldMatchPublishedFiles() throws Exception {
        List<String> lines = Files.readAllLines(RESULTS.resolve("MANIFEST.sha256"), StandardCharsets.UTF_8);

        assertThat(lines).hasSize(5);
        for (String line : lines) {
            Path artifact = RESULTS.resolve(line.substring(66)).normalize();
            assertThat(artifact.startsWith(RESULTS)).isTrue();
            assertThat(sha256(artifact)).isEqualTo(line.substring(0, 64));
        }
    }

    @Test
    void runShouldPinDatasetAndModels() throws Exception {
        JsonNode run = JSON.readTree(RESULTS.resolve("run.json").toFile());

        assertThat(run.at("/dataset/questions").intValue()).isEqualTo(240);
        assertThat(run.at("/dataset/questionsSha256").textValue())
                .isEqualTo(sha256(Path.of("data", "eval", "v2", "questions.json")));
        assertThat(run.at("/configuration/rrfK").intValue()).isEqualTo(60);
        assertThat(run.at("/models/dense").textValue()).startsWith("sergeyzh/BERTA@");
        assertThat(run.at("/models/reranker").textValue()).startsWith("BAAI/bge-reranker-v2-m3@");
        assertThat(run.at("/models/device").textValue()).isEqualTo("cuda");
    }

    @Test
    void metricsShouldMatchReadme() throws Exception {
        Map<String, Map<String, String>> metrics = keyedCsv(
                RESULTS.resolve("metrics.csv"), List.of("stage", "strategy", "metric"));

        assertThat(metrics.get("retrieval|dense|hit_at_10").get("hits")).isEqualTo("213");
        assertThat(metrics.get("retrieval|bm25|hit_at_10").get("hits")).isEqualTo("224");
        assertThat(metrics.get("retrieval|hybrid|hit_at_10").get("hits")).isEqualTo("229");
        assertThat(metrics.get("candidate_pool|hybrid|hit_at_50").get("hits")).isEqualTo("238");
        assertThat(metrics.get("rerank|hybrid|hit_at_10").get("hits")).isEqualTo("230");
        assertThat(metrics.get("rerank|hybrid|ndcg_at_10").get("value")).isEqualTo("0.8827");
        assertThat(metrics.get("routing|routing|hit_at_10").get("hits")).isEqualTo("228");
    }

    @Test
    void examplesAndLatencyShouldMatchArticle() throws Exception {
        Map<String, Map<String, String>> retrieval = keyedCsv(
                RESULTS.resolve("retrieval.csv"), List.of("strategy", "qid"));
        Map<String, Map<String, String>> rerank = keyedCsv(
                RESULTS.resolve("rerank.csv"), List.of("source", "qid", "quality_pool"));
        Map<String, Map<String, String>> metrics = keyedCsv(
                RESULTS.resolve("metrics.csv"), List.of("stage", "strategy", "metric"));

        assertThat(ranks(retrieval, "semantic-101")).containsExactly(3, 3, 1);
        assertThat(ranks(retrieval, "semantic-071")).containsExactly(0, 3, 0);
        assertThat(rerank.get("dense|exact-037|50").get("candidate_rank")).isEqualTo("22");
        assertThat(rerank.get("bm25|exact-037|50").get("candidate_rank")).isEqualTo("1");
        assertThat(rerank.get("hybrid|exact-037|50").get("candidate_rank")).isEqualTo("5");
        assertThat(rerank.get("hybrid|semantic-071|50").get("rerank_rank")).isEqualTo("4");
        assertThat(metrics.get("rerank|bge_pool_20|latency_p50_ms").get("value")).isEqualTo("236.0");
        assertThat(metrics.get("rerank|bge_pool_50|latency_p50_ms").get("value")).isEqualTo("660.0");
    }

    private static int[] ranks(Map<String, Map<String, String>> rows, String qid) {
        return new int[]{
                Integer.parseInt(rows.get("dense|" + qid).get("rank")),
                Integer.parseInt(rows.get("bm25|" + qid).get("rank")),
                Integer.parseInt(rows.get("hybrid|" + qid).get("rank"))
        };
    }

    private static Map<String, Map<String, String>> keyedCsv(Path path, List<String> keyColumns) throws Exception {
        Map<String, Map<String, String>> result = new HashMap<>();
        for (Map<String, String> row : csv(path)) {
            String key = keyColumns.stream().map(row::get).reduce((left, right) -> left + "|" + right).orElseThrow();
            result.put(key, row);
        }
        return Map.copyOf(result);
    }

    private static List<Map<String, String>> csv(Path path) throws Exception {
        List<String> lines = Files.readAllLines(path, StandardCharsets.UTF_8);
        String[] headers = lines.getFirst().split(",", -1);
        return lines.stream().skip(1).map(line -> {
            String[] values = line.split(",", -1);
            Map<String, String> row = new HashMap<>();
            for (int index = 0; index < headers.length; index++) {
                row.put(headers[index], values[index]);
            }
            return Map.copyOf(row);
        }).toList();
    }

    private static String sha256(Path path) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        try (InputStream input = Files.newInputStream(path)) {
            byte[] buffer = new byte[8192];
            int read;
            while ((read = input.read(buffer)) >= 0) {
                digest.update(buffer, 0, read);
            }
        }
        return HexFormat.of().formatHex(digest.digest());
    }
}
