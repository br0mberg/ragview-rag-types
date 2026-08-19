package ru.brombin.ragview;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import ru.brombin.ragview.corpus.CorpusDocument;
import ru.brombin.ragview.eval.FnsFaqBgeLatencyRunner;
import ru.brombin.ragview.eval.FnsFaqBgeCutoffEvaluator;
import ru.brombin.ragview.eval.FnsFaqCandidate;
import ru.brombin.ragview.rerank.AttestedRerankScorer;
import ru.brombin.ragview.rerank.RerankerAttestation;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class FnsFaqBgeLatencyRunnerContractTest {

    @TempDir
    Path tempDirectory;

    ObjectMapper objectMapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);

    @Test
    void run_shouldWarmUpInterleaveCutoffsAndWriteReproducibleLatencyArtifacts() throws Exception {
        Fixture fixture = writeFixture("valid", 3, 50);
        StubScorer firstScorer = new StubScorer(pinnedAttestation());
        var request = new FnsFaqBgeLatencyRunner.LatencyRequest(2, 1, 2, "latency-seed");
        Instant createdAt = Instant.parse("2026-08-16T01:00:00Z");
        Path firstOutput = tempDirectory.resolve("latency-one");
        FnsFaqBgeLatencyRunner firstRunner = new FnsFaqBgeLatencyRunner(
                objectMapper, firstScorer, new FixedDurationClock(5_000_000));

        var manifest = firstRunner.run(
                fixture.questions(),
                fixture.corpus(),
                fixture.bgeManifest(),
                fixture.bgeProvenance(),
                firstOutput,
                request,
                createdAt);

        assertThat(firstScorer.calls).isEqualTo(32);
        for (int cutoff : FnsFaqBgeLatencyRunner.CUTOFFS) {
            assertThat(firstScorer.passageCounts.stream().filter(value -> value == cutoff).count())
                    .isEqualTo(8);
            assertThat(firstScorer.firstPassagesByCount.get(cutoff))
                    .containsExactlyElementsOf(fixture.documents().subList(0, cutoff).stream()
                            .map(CorpusDocument::indexableText)
                            .toList());
        }

        List<String> rawLines = Files.readAllLines(firstOutput.resolve("latency-raw.csv"));
        assertThat(rawLines).hasSize(25);
        assertThat(rawLines.getFirst()).isEqualTo(
                "sequence,repeat,qid,cutoff,candidate_count,elapsed_ns,elapsed_ms");
        assertThat(rawLines.subList(1, rawLines.size()))
                .allMatch(line -> line.endsWith(",5000000,5.000000"));
        assertThat(rawLines.subList(1, rawLines.size()).stream()
                .map(line -> Integer.parseInt(line.split(",")[3])))
                .containsOnly(10, 20, 30, 50);
        for (int start = 1; start < rawLines.size(); start += 4) {
            List<String[]> block = rawLines.subList(start, start + 4).stream()
                    .map(line -> line.split(","))
                    .toList();
            assertThat(block.stream().map(fields -> fields[2])).containsOnly(block.getFirst()[2]);
            assertThat(block.stream().map(fields -> Integer.parseInt(fields[3])))
                    .containsExactlyInAnyOrder(10, 20, 30, 50);
        }

        List<String> summaryLines = Files.readAllLines(firstOutput.resolve("latency-summary.csv"));
        assertThat(summaryLines).containsExactly(
                "cutoff,sample_count,mean_ms,p50_ms,p95_ms",
                "10,6,5.000000,5.000000,5.000000",
                "20,6,5.000000,5.000000,5.000000",
                "30,6,5.000000,5.000000,5.000000",
                "50,6,5.000000,5.000000,5.000000");

        assertThat(manifest.schemaVersion()).isEqualTo(1);
        assertThat(manifest.evalVersion()).isEqualTo("tax-eval-v3-bge-latency-v1");
        assertThat(manifest.questionCount()).isEqualTo(3);
        assertThat(manifest.corpusDocumentCount()).isEqualTo(50);
        assertThat(manifest.frozenRerankerAttestation()).isEqualTo(pinnedAttestation());
        assertThat(manifest.liveRerankerAttestation()).isEqualTo(pinnedAttestation());
        assertThat(manifest.measurement().cutoffs()).containsExactly(10, 20, 30, 50);
        assertThat(manifest.measurement().sampleCount()).isEqualTo(24);
        assertThat(manifest.measurement().warmupQuestionCount()).isEqualTo(2);
        assertThat(manifest.measurement().warmupRepeats()).isEqualTo(1);
        assertThat(manifest.measurement().measuredRepeats()).isEqualTo(2);
        assertThat(manifest.sourceBgeManifest().sha256())
                .isEqualTo(FnsFaqBgeCutoffEvaluator.sha256(
                        Files.readAllBytes(fixture.bgeManifest())));
        assertThat(manifest.sourceBgeSealedProvenance().sha256())
                .isEqualTo(FnsFaqBgeCutoffEvaluator.sha256(
                        Files.readAllBytes(fixture.bgeProvenance())));
        assertThat(manifest.rawLatency().sha256())
                .isEqualTo(FnsFaqBgeCutoffEvaluator.sha256(
                        Files.readAllBytes(firstOutput.resolve("latency-raw.csv"))));
        assertThat(manifest.summary().sha256())
                .isEqualTo(FnsFaqBgeCutoffEvaluator.sha256(
                        Files.readAllBytes(firstOutput.resolve("latency-summary.csv"))));

        StubScorer secondScorer = new StubScorer(pinnedAttestation());
        Path secondOutput = tempDirectory.resolve("latency-two");
        new FnsFaqBgeLatencyRunner(
                objectMapper, secondScorer, new FixedDurationClock(5_000_000)).run(
                fixture.questions(),
                fixture.corpus(),
                fixture.bgeManifest(),
                fixture.bgeProvenance(),
                secondOutput,
                request,
                createdAt);

        assertThat(secondScorer.callSignatures).containsExactlyElementsOf(firstScorer.callSignatures);
        assertThat(Files.readAllBytes(secondOutput.resolve("latency-raw.csv")))
                .isEqualTo(Files.readAllBytes(firstOutput.resolve("latency-raw.csv")));
        assertThat(Files.readAllBytes(secondOutput.resolve("latency-summary.csv")))
                .isEqualTo(Files.readAllBytes(firstOutput.resolve("latency-summary.csv")));
    }

    @Test
    void run_shouldRejectTamperedBgeProvenanceBeforeScoring() throws Exception {
        Fixture fixture = writeFixture("tampered", 2, 50);
        Files.writeString(fixture.bgeProvenance(), "{}\n", StandardCharsets.UTF_8);
        StubScorer scorer = new StubScorer(pinnedAttestation());
        FnsFaqBgeLatencyRunner runner = new FnsFaqBgeLatencyRunner(
                objectMapper, scorer, new FixedDurationClock(1));

        assertThatThrownBy(() -> runFixture(runner, fixture, "tampered-output"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("mismatched SHA-256 for sealedProvenance");
        assertThat(scorer.calls).isZero();
    }

    @Test
    void run_shouldRejectDifferentLiveAttestationBeforeScoring() throws Exception {
        Fixture fixture = writeFixture("attestation", 2, 50);
        RerankerAttestation differentHealth = new RerankerAttestation(
                1,
                FnsFaqBgeCutoffEvaluator.BGE_MODEL,
                FnsFaqBgeCutoffEvaluator.BGE_REVISION,
                FnsFaqBgeCutoffEvaluator.BGE_PRECISION,
                FnsFaqBgeCutoffEvaluator.BGE_DEVICE,
                FnsFaqBgeCutoffEvaluator.BGE_BATCH_SIZE,
                FnsFaqBgeCutoffEvaluator.BGE_MAX_LENGTH,
                "b".repeat(64));
        StubScorer scorer = new StubScorer(differentHealth);
        FnsFaqBgeLatencyRunner runner = new FnsFaqBgeLatencyRunner(
                objectMapper, scorer, new FixedDurationClock(1));

        assertThatThrownBy(() -> runFixture(runner, fixture, "attestation-output"))
                .isInstanceOf(IllegalStateException.class)
                .hasMessage("Live reranker attestation differs from frozen BGE run");
        assertThat(scorer.calls).isZero();
    }

    @Test
    void run_shouldRejectDuplicateFirstStageRankBeforeScoring() throws Exception {
        Fixture fixture = writeFixture("duplicate-rank", 2, 50);
        List<JsonNode> rows = readJsonLines(fixture.bgeProvenance());
        ((ObjectNode) rows.getFirst().path("bge").get(0)).put("firstStageRank", 49);
        writeJsonLines(fixture.bgeProvenance(), rows);
        refreshProvenanceHash(fixture);
        StubScorer scorer = new StubScorer(pinnedAttestation());
        FnsFaqBgeLatencyRunner runner = new FnsFaqBgeLatencyRunner(
                objectMapper, scorer, new FixedDurationClock(1));

        assertThatThrownBy(() -> runFixture(runner, fixture, "duplicate-rank-output"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Duplicate first-stage rank");
        assertThat(scorer.calls).isZero();
    }

    private void runFixture(
            FnsFaqBgeLatencyRunner runner,
            Fixture fixture,
            String outputName) throws Exception {
        runner.run(
                fixture.questions(),
                fixture.corpus(),
                fixture.bgeManifest(),
                fixture.bgeProvenance(),
                tempDirectory.resolve(outputName),
                new FnsFaqBgeLatencyRunner.LatencyRequest(1, 1, 1, "seed"),
                Instant.parse("2026-08-16T01:00:00Z"));
    }

    private Fixture writeFixture(String directory, int questionCount, int documentCount) throws Exception {
        Path root = tempDirectory.resolve(directory);
        Path snapshotDirectory = root.resolve("snapshot");
        Path bgeDirectory = root.resolve("bge");
        Files.createDirectories(snapshotDirectory);
        Files.createDirectories(bgeDirectory);

        List<CorpusDocument> documents = new ArrayList<>();
        for (int index = 1; index <= documentCount; index++) {
            documents.add(new CorpusDocument(
                    "nk-1-article-" + index + "-chunk-001",
                    "Статья " + index,
                    "Текст налоговой нормы " + index));
        }
        Path corpus = root.resolve("corpus.json");
        objectMapper.writerWithDefaultPrettyPrinter().writeValue(corpus.toFile(), documents);
        String corpusSha256 = FnsFaqBgeCutoffEvaluator.sha256(Files.readAllBytes(corpus));

        List<FnsFaqCandidate> questions = new ArrayList<>();
        for (int index = 1; index <= questionCount; index++) {
            questions.add(candidate("q-%03d".formatted(index), "Налоговый вопрос " + index));
        }
        Path candidates = snapshotDirectory.resolve("candidates.json");
        objectMapper.writerWithDefaultPrettyPrinter().writeValue(candidates.toFile(), questions);
        String questionsSha256 = FnsFaqBgeCutoffEvaluator.sha256(Files.readAllBytes(candidates));

        List<Map<String, Object>> provenanceRows = new ArrayList<>();
        for (FnsFaqCandidate question : questions) {
            List<Map<String, Object>> bge = new ArrayList<>();
            for (int index = 0; index < 50; index++) {
                int firstStageRank = 50 - index;
                bge.add(Map.of(
                        "firstStageRank", firstStageRank,
                        "rerankRank", index + 1,
                        "docId", documents.get(firstStageRank - 1).id(),
                        "score", 100.0 - index));
            }
            provenanceRows.add(Map.of("qid", question.id(), "bge", bge));
        }
        Path provenance = bgeDirectory.resolve("pooling-provenance.sealed.jsonl");
        writeJsonLines(provenance, provenanceRows);

        Map<String, Object> manifest = new LinkedHashMap<>();
        manifest.put("schemaVersion", 2);
        manifest.put("evalVersion", "tax-eval-v3-bge-reranker-pool");
        manifest.put("createdAtUtc", "2026-08-13T18:32:42Z");
        manifest.put("sourceEvalVersion", "tax-eval-v3-fns-candidates");
        manifest.put("questionCount", questions.size());
        manifest.put("corpusDocumentCount", documents.size());
        manifest.put("questions", artifact(candidates.getFileName().toString(), questionsSha256));
        manifest.put("snapshotManifest", artifact("manifest.json", "1".repeat(64)));
        manifest.put("corpus", artifact(corpus.getFileName().toString(), corpusSha256));
        manifest.put("sourcePoolManifest", artifact("manifest.json", "2".repeat(64)));
        manifest.put("sourceSealedProvenance", artifact(
                "pooling-provenance.sealed.jsonl", "3".repeat(64)));
        manifest.put("rerankerAttestation", pinnedAttestation());
        manifest.put("pooling", pooling());
        manifest.put("reviewerPoolPolicy", "fixture");
        manifest.put("reviewerShuffle", "fixture");
        manifest.put("reviewerPool", artifact("reviewer-pool.jsonl", "4".repeat(64)));
        manifest.put("sealedProvenance", artifact(
                provenance.getFileName().toString(),
                FnsFaqBgeCutoffEvaluator.sha256(Files.readAllBytes(provenance))));
        Path bgeManifest = bgeDirectory.resolve("manifest.json");
        objectMapper.writerWithDefaultPrettyPrinter().writeValue(bgeManifest.toFile(), manifest);
        return new Fixture(
                candidates,
                corpus,
                bgeManifest,
                provenance,
                List.copyOf(documents));
    }

    private void refreshProvenanceHash(Fixture fixture) throws Exception {
        JsonNode manifest = objectMapper.readTree(fixture.bgeManifest().toFile());
        ((ObjectNode) manifest.path("sealedProvenance")).put(
                "sha256",
                FnsFaqBgeCutoffEvaluator.sha256(Files.readAllBytes(fixture.bgeProvenance())));
        objectMapper.writerWithDefaultPrettyPrinter().writeValue(
                fixture.bgeManifest().toFile(), manifest);
    }

    private void writeJsonLines(Path path, List<?> rows) throws Exception {
        var writer = objectMapper.writer().without(SerializationFeature.INDENT_OUTPUT);
        List<String> lines = rows.stream().map(row -> {
            try {
                return writer.writeValueAsString(row);
            } catch (Exception error) {
                throw new IllegalStateException(error);
            }
        }).toList();
        Files.writeString(path, String.join("\n", lines) + "\n", StandardCharsets.UTF_8);
    }

    private List<JsonNode> readJsonLines(Path path) throws Exception {
        List<JsonNode> rows = new ArrayList<>();
        for (String line : Files.readAllLines(path)) {
            rows.add(objectMapper.readTree(line));
        }
        return rows;
    }

    private static Map<String, Object> pooling() {
        Map<String, Object> pooling = new LinkedHashMap<>();
        pooling.put("topN", 10);
        pooling.put("scoredPool", 50);
        pooling.put("reviewerSeed", "fixture");
        pooling.put("profile", "bge-cross-encoder");
        pooling.put("implementation", "cross-encoder-reranker");
        pooling.put("model", FnsFaqBgeCutoffEvaluator.BGE_MODEL);
        pooling.put("revision", FnsFaqBgeCutoffEvaluator.BGE_REVISION);
        pooling.put("precision", FnsFaqBgeCutoffEvaluator.BGE_PRECISION);
        pooling.put("device", FnsFaqBgeCutoffEvaluator.BGE_DEVICE);
        pooling.put("batchSize", FnsFaqBgeCutoffEvaluator.BGE_BATCH_SIZE);
        pooling.put("maxLength", FnsFaqBgeCutoffEvaluator.BGE_MAX_LENGTH);
        pooling.put("ordering", "score_desc,first_stage_rank_asc,doc_id_asc");
        return pooling;
    }

    private static Map<String, String> artifact(String filename, String sha256) {
        return Map.of("filename", filename, "sha256", sha256);
    }

    private static FnsFaqCandidate candidate(String id, String question) {
        return new FnsFaqCandidate(
                id,
                question,
                "answer",
                "source",
                List.of(),
                List.of(),
                false,
                263,
                "НДФЛ",
                1,
                "https://example.test",
                "0".repeat(64));
    }

    private static RerankerAttestation pinnedAttestation() {
        return new RerankerAttestation(
                1,
                FnsFaqBgeCutoffEvaluator.BGE_MODEL,
                FnsFaqBgeCutoffEvaluator.BGE_REVISION,
                FnsFaqBgeCutoffEvaluator.BGE_PRECISION,
                FnsFaqBgeCutoffEvaluator.BGE_DEVICE,
                FnsFaqBgeCutoffEvaluator.BGE_BATCH_SIZE,
                FnsFaqBgeCutoffEvaluator.BGE_MAX_LENGTH,
                "a".repeat(64));
    }

    private static final class StubScorer implements AttestedRerankScorer {

        final RerankerAttestation attestation;
        int calls;
        List<Integer> passageCounts = new ArrayList<>();
        List<String> callSignatures = new ArrayList<>();
        Map<Integer, List<String>> firstPassagesByCount = new HashMap<>();

        private StubScorer(RerankerAttestation attestation) {
            this.attestation = attestation;
        }

        @Override
        public List<Double> score(String query, List<String> passages) {
            calls++;
            passageCounts.add(passages.size());
            callSignatures.add(query + "\0" + passages.size());
            firstPassagesByCount.putIfAbsent(passages.size(), List.copyOf(passages));
            return java.util.stream.IntStream.range(0, passages.size())
                    .mapToDouble(index -> passages.size() - index)
                    .boxed()
                    .toList();
        }

        @Override
        public RerankerAttestation attestation() {
            return attestation;
        }
    }

    private static final class FixedDurationClock implements FnsFaqBgeLatencyRunner.NanoClock {

        final long duration;
        long calls;

        private FixedDurationClock(long duration) {
            this.duration = duration;
        }

        @Override
        public long nanoTime() {
            long value = ((calls + 1) / 2) * duration;
            calls++;
            return value;
        }
    }

    private record Fixture(
            Path questions,
            Path corpus,
            Path bgeManifest,
            Path bgeProvenance,
            List<CorpusDocument> documents) {
    }
}
