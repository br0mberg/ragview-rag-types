package ru.brombin.ragview;

import com.fasterxml.jackson.databind.json.JsonMapper;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import ru.brombin.ragview.eval.FnsFaqBgeCutoffEvaluator;

import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;

class PublicV3ArtifactsContractTest {

    static final Path RESULTS = Path.of("results/v3-citation-silver");
    static final Path BGE_RESULTS = Path.of("results/v3-bge-cited-clause");

    @Test
    void publicArtifacts_shouldMatchManifestAndContainNoSourceText() throws Exception {
        var manifest = new JsonMapper().readTree(RESULTS.resolve("public-manifest.json").toFile());

        assertThat(manifest.path("evalVersion").asText())
                .isEqualTo("tax-eval-v3-citation-silver-v3");
        assertThat(manifest.path("citationSilver").path("literalCitationQuestions").asInt())
                .isEqualTo(191);
        assertThat(manifest.path("citationSilver").path("metricEligibleQuestions").asInt())
                .isEqualTo(186);
        assertThat(manifest.path("citationSilver").path("provisionalTemporalMismatchCount").asInt())
                .isEqualTo(4);
        assertThat(manifest.path("citationSilver").path("provisionalSourceTypoCount").asInt())
                .isEqualTo(1);
        assertThat(manifest.path("citationSilver").path("evaluationSlices")
                .path("preliminaryFiltered").path("reportingRole").asText())
                .isEqualTo("primary_technical_before_expert_validation");
        assertThat(manifest.path("citationSilver").path("evaluationSlices")
                .path("literalCitationAll").path("reportingRole").asText())
                .isEqualTo("sensitivity");
        assertThat(manifest.path("publication").path("containsQuestionOrAnswerText").asBoolean())
                .isFalse();
        assertThat(manifest.path("publication").path("containsCorpusText").asBoolean())
                .isFalse();
        assertThat(manifest.path("publication").path("containsDocumentIdentifiers").asBoolean())
                .isFalse();

        assertArtifactHash(manifest, "aggregateCsv");
        assertArtifactHash(manifest, "perQueryCsv");
        assertArtifactHash(manifest, "qdrantRepeatSummaryJson");
        assertThat(Files.lines(RESULTS.resolve("aggregate.csv")).count()).isEqualTo(1_189);
        assertThat(Files.lines(RESULTS.resolve("per-query.csv")).count()).isEqualTo(7_921);

        String headers = Files.readAllLines(RESULTS.resolve("aggregate.csv")).getFirst()
                + "," + Files.readAllLines(RESULTS.resolve("per-query.csv")).getFirst();
        assertThat(headers).contains("evaluation_slice", "literal_citation_slice_included");
        assertThat(Files.readAllLines(RESULTS.resolve("aggregate.csv")))
                .anyMatch(line -> line.contains(",literal_citation_all,"))
                .anyMatch(line -> line.contains(",preliminary_filtered,"));
        assertThat(headers).doesNotContain(
                "question", "answer", "rawAnswer", "citedSource", "chunkText", "docId");
    }

    @Test
    void qdrantRepeatSummary_shouldFreezeInputHashesAndExactCounts() throws Exception {
        var summary = new JsonMapper().readTree(RESULTS.resolve("qdrant-repeat-summary.json").toFile());

        assertThat(summary.path("questionCount").asInt()).isEqualTo(240);
        assertThat(summary.path("inputs").path("primary").path("manifestSha256").asText())
                .isEqualTo("276b496e965c9e9123c0538b70eb0a538ee95047c903080ffb482cc651ca5b2a");
        assertThat(summary.path("inputs").path("primary").path("rawRankingsSha256").asText())
                .isEqualTo("a0bc73218b59d851e824819135da28d8283019e6d25afdf6cee02dedcb96f5f5");
        assertThat(summary.path("inputs").path("repeat").path("manifestSha256").asText())
                .isEqualTo("7809aaac0181d1293eb873c93bdc77ceed260c3f9c82af4762b9d011d32db1d6");
        assertThat(summary.path("inputs").path("repeat").path("rawRankingsSha256").asText())
                .isEqualTo("86491891c209572c525fb23b7118c52b7a72fbcbe9e33d11793578eea3df7482");

        var parity = summary.path("primaryClientServerParity");
        assertThat(parity.path("sameFullUnionSetCount").asInt()).isEqualTo(240);
        assertThat(parity.path("sameTop50SetCount").asInt()).isEqualTo(189);
        assertThat(parity.path("sameTop10SetCount").asInt()).isEqualTo(231);
        assertThat(parity.path("scoreEquivalentCount").asInt()).isEqualTo(240);
        assertThat(parity.path("nonTieDifferenceCount").asInt()).isZero();

        assertRepeatability(summary, "denseFullOrder", 240, 0);
        assertRepeatability(summary, "nativeSparseFullOrder", 240, 0);
        assertRepeatability(summary, "clientRrfFullOrder", 240, 0);
        assertRepeatability(summary, "serverRrfFullOrder", 0, 240);
        assertRepeatability(summary, "serverRrfFullSet", 240, 0);
        assertRepeatability(summary, "serverRrfTop50Order", 4, 236);
        assertRepeatability(summary, "serverRrfTop50Set", 198, 42);
        assertRepeatability(summary, "serverRrfTop10Order", 201, 39);
        assertRepeatability(summary, "serverRrfTop10Set", 233, 7);

        var publication = summary.path("publication");
        assertThat(publication.path("containsQuestionOrAnswerText").asBoolean()).isFalse();
        assertThat(publication.path("containsCorpusText").asBoolean()).isFalse();
        assertThat(publication.path("containsDocumentOrQuestionIdentifiers").asBoolean()).isFalse();

        var frame = new JsonMapper().readTree(Path.of("data/eval/v3_frame_manifest.json").toFile());
        assertThat(frame.path("qdrantRrfParity").path("repeatabilitySummarySha256").asText())
                .isEqualTo(sha256(RESULTS.resolve("qdrant-repeat-summary.json")));
    }

    @Test
    void publicChecksumFile_shouldCoverEveryPublishedArtifact() throws Exception {
        var expected = Files.readAllLines(RESULTS.resolve("MANIFEST.public.sha256"));

        assertThat(expected).hasSize(4);
        assertThat(expected.stream().map(line -> line.split("  ", 2)[1]).collect(java.util.stream.Collectors.toSet()))
                .isEqualTo(Set.of(
                        "aggregate.csv",
                        "per-query.csv",
                        "qdrant-repeat-summary.json",
                        "public-manifest.json"));
        for (String line : expected) {
            String[] parts = line.split("  ", 2);
            assertThat(parts).hasSize(2);
            assertThat(sha256(RESULTS.resolve(parts[1]))).isEqualTo(parts[0]);
        }
    }

    @Test
    void publicBgeArtifacts_shouldMatchFrozenV3EvaluationWithoutSourceText() throws Exception {
        var mapper = new JsonMapper();
        var manifest = mapper.readTree(BGE_RESULTS.resolve("public-manifest.json").toFile());

        assertThat(manifest.path("evalVersion").asText())
                .isEqualTo("tax-eval-v3-bge-cutoff-evaluation-v1");
        assertThat(manifest.path("target").path("eligiblePrimaryQuestions").asInt())
                .isEqualTo(79);
        assertThat(manifest.path("target").path("internalLabelsSha256").asText())
                .isEqualTo("f73f2b802395f3f68544bc47e35bedf8fa08680eff6574b1d7095d3f5213f49f");
        assertThat(manifest.path("target").path("labelsCanonicalJsonSha256").asText())
                .isEqualTo("d447a9a347cf0bc22c39bb1799ae159e98bdb77aad005d2a84afff4f13854126");
        assertThat(manifest.path("evaluation").path("primaryCutoff").asInt())
                .isEqualTo(50);
        var primary = manifest.path("evaluation").path("primaryC50");
        assertThat(primary.path("beforeHitAt10").asText()).isEqualTo("67/79");
        assertThat(primary.path("afterHitAt10").asText()).isEqualTo("71/79");
        assertThat(primary.path("beforeNdcgAt10").asDouble()).isEqualTo(0.603152288);
        assertThat(primary.path("afterNdcgAt10").asDouble()).isEqualTo(0.719341514);
        assertThat(manifest.path("evaluation").path("observedC20Sensitivity")
                .path("afterNdcgAt10").asDouble()).isEqualTo(0.726769241);

        var publication = manifest.path("publication");
        assertThat(publication.path("containsQuestionOrAnswerText").asBoolean()).isFalse();
        assertThat(publication.path("containsCitedSource").asBoolean()).isFalse();
        assertThat(publication.path("containsCorpusText").asBoolean()).isFalse();
        assertThat(publication.path("containsAuditTrace").asBoolean()).isFalse();
        assertThat(publication.path("containsStructuralDocumentIdentifiers").asBoolean()).isTrue();
        assertThat(publication.path("containsModelScores").asBoolean()).isTrue();

        for (String artifact : Set.of(
                "aggregateCsv",
                "perQueryCsv",
                "publicBgeManifest",
                "bgeRanking",
                "publicTargetManifest",
                "citedClauseLabels",
                "candidatePairTokenLengths",
                "targetPairTokenLengths",
                "tokenLengthSummary")) {
            assertArtifactHash(BGE_RESULTS, manifest, artifact);
        }
        assertThat(Files.lines(BGE_RESULTS.resolve("aggregate.csv")).count()).isEqualTo(21);
        assertThat(Files.lines(BGE_RESULTS.resolve("per-query.csv")).count()).isEqualTo(961);
        assertThat(Files.lines(BGE_RESULTS.resolve("bge-ranking.jsonl")).count()).isEqualTo(240);

        String headers = Files.readAllLines(BGE_RESULTS.resolve("per-query.csv")).getFirst();
        assertThat(headers).contains("qid", "query_group", "before_ndcg_at_10", "after_ndcg_at_10");
        assertThat(headers).doesNotContain("question", "answer", "citedSource", "chunkText");
        assertThat(Files.readAllLines(BGE_RESULTS.resolve("per-query.csv")))
                .allMatch(line -> line.split(",", -1).length == 22);
        assertNoSourceTextFields(Files.readString(BGE_RESULTS.resolve("cited-clause-labels.json")));
        assertNoSourceTextFields(Files.readString(BGE_RESULTS.resolve("bge-ranking.jsonl")));
    }

    @Test
    void publicTokenLengths_shouldReproduceArticleCounts() throws Exception {
        var mapper = new JsonMapper();
        var publicManifest = mapper.readTree(BGE_RESULTS.resolve("public-manifest.json").toFile());
        var summary = mapper.readTree(BGE_RESULTS.resolve("token-length-summary.json").toFile());

        assertThat(summary.path("tokenizer").path("model").asText())
                .isEqualTo("BAAI/bge-reranker-v2-m3");
        assertThat(summary.path("tokenizer").path("revision").asText())
                .isEqualTo("953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e");
        assertThat(summary.path("tokenizer").path("transformersVersion").asText())
                .isEqualTo("4.53.2");

        var candidateCounts = tokenLengthCounts(
                BGE_RESULTS.resolve("retrieval-pair-token-lengths.csv"));
        var targetCounts = tokenLengthCounts(
                BGE_RESULTS.resolve("target-pair-token-lengths.csv"));
        assertThat(candidateCounts).containsEntry("pairs", 12_000L)
                .containsEntry("over512", 9_681L)
                .containsEntry("over1024", 35L)
                .containsEntry("max", 1_132L);
        assertThat(targetCounts).containsEntry("pairs", 95L)
                .containsEntry("over512", 81L)
                .containsEntry("over1024", 0L)
                .containsEntry("max", 980L);

        assertThat(summary.path("candidatePairs").path("pairs").asLong())
                .isEqualTo(candidateCounts.get("pairs"));
        assertThat(summary.path("candidatePairs").path("over512").asLong())
                .isEqualTo(candidateCounts.get("over512"));
        assertThat(summary.path("candidatePairs").path("over1024").asLong())
                .isEqualTo(candidateCounts.get("over1024"));
        assertThat(summary.path("targetPairs").path("pairs").asLong())
                .isEqualTo(targetCounts.get("pairs"));
        assertThat(summary.path("targetPairs").path("over512").asLong())
                .isEqualTo(targetCounts.get("over512"));
        assertThat(summary.path("targetPairs").path("over1024").asLong())
                .isEqualTo(targetCounts.get("over1024"));

        var audit = publicManifest.path("tokenLengthAudit");
        assertThat(audit.path("candidatePairs").path("over1024").asLong()).isEqualTo(35);
        assertThat(audit.path("targetPairs").path("over512").asLong()).isEqualTo(81);
        assertThat(audit.path("targetPairs").path("over1024").asLong()).isZero();
    }

    @Test
    void publicBgeInputs_shouldRecomputePublishedCsv(@TempDir Path temp) throws Exception {
        var evaluator = new FnsFaqBgeCutoffEvaluator(new JsonMapper());

        evaluator.run(
                BGE_RESULTS.resolve("bge-manifest.json"),
                BGE_RESULTS.resolve("cited-clause-manifest.json"),
                FnsFaqBgeCutoffEvaluator.REQUIRED_CUTOFFS,
                temp.resolve("recomputed"));

        assertThat(sha256(temp.resolve("recomputed/aggregate.csv")))
                .isEqualTo(sha256(BGE_RESULTS.resolve("aggregate.csv")));
        assertThat(sha256(temp.resolve("recomputed/per-query.csv")))
                .isEqualTo(sha256(BGE_RESULTS.resolve("per-query.csv")));
    }

    @Test
    void publicBgeChecksumFile_shouldCoverEveryPublishedArtifact() throws Exception {
        var expected = Files.readAllLines(BGE_RESULTS.resolve("MANIFEST.public.sha256"));

        assertThat(expected).hasSize(10);
        assertThat(expected.stream().map(line -> line.split("  ", 2)[1]).collect(java.util.stream.Collectors.toSet()))
                .isEqualTo(Set.of(
                        "aggregate.csv",
                        "bge-manifest.json",
                        "bge-ranking.jsonl",
                        "cited-clause-labels.json",
                        "cited-clause-manifest.json",
                        "per-query.csv",
                        "public-manifest.json",
                        "retrieval-pair-token-lengths.csv",
                        "target-pair-token-lengths.csv",
                        "token-length-summary.json"));
        for (String line : expected) {
            String[] parts = line.split("  ", 2);
            assertThat(parts).hasSize(2);
            assertThat(sha256(BGE_RESULTS.resolve(parts[1]))).isEqualTo(parts[0]);
        }
    }

    private static void assertRepeatability(
            com.fasterxml.jackson.databind.JsonNode summary,
            String name,
            int repeatable,
            int changed) {
        var metric = summary.path("repeatability").path(name);
        assertThat(metric.path("repeatable").asInt()).isEqualTo(repeatable);
        assertThat(metric.path("changed").asInt()).isEqualTo(changed);
        assertThat(metric.path("repeatable").asInt() + metric.path("changed").asInt())
                .isEqualTo(summary.path("questionCount").asInt());
    }

    private static void assertArtifactHash(com.fasterxml.jackson.databind.JsonNode manifest, String name)
            throws Exception {
        assertArtifactHash(RESULTS, manifest, name);
    }

    private static void assertArtifactHash(
            Path directory,
            com.fasterxml.jackson.databind.JsonNode manifest,
            String name) throws Exception {
        var artifact = manifest.path("artifacts").path(name);
        assertThat(sha256(directory.resolve(artifact.path("filename").asText())))
                .isEqualTo(artifact.path("sha256").asText());
    }

    private static void assertNoSourceTextFields(String content) {
        assertThat(content).doesNotContain(
                "\"question\" :",
                "\"question\":",
                "\"answer\" :",
                "\"answer\":",
                "\"rawAnswer\"",
                "\"citedSource\"",
                "\"chunkText\"",
                "\"sourceText\"");
    }

    private static HashMap<String, Long> tokenLengthCounts(Path path) throws Exception {
        var counts = new HashMap<String, Long>();
        counts.put("pairs", 0L);
        counts.put("over512", 0L);
        counts.put("over1024", 0L);
        counts.put("max", 0L);
        try (var lines = Files.lines(path)) {
            lines.skip(1).forEach(line -> {
                String[] columns = line.split(",", -1);
                long length = Long.parseLong(columns[columns.length - 3]);
                counts.compute("pairs", (key, value) -> value + 1);
                counts.compute("over512", (key, value) -> value + (length > 512 ? 1 : 0));
                counts.compute("over1024", (key, value) -> value + (length > 1024 ? 1 : 0));
                counts.compute("max", (key, value) -> Math.max(value, length));
            });
        }
        return counts;
    }

    private static String sha256(Path path) throws Exception {
        byte[] digest = MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(path));
        return HexFormat.of().formatHex(digest);
    }
}
