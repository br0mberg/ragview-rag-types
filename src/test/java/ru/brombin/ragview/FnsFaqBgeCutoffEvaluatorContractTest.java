package ru.brombin.ragview;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import ru.brombin.ragview.eval.FnsFaqBgeCutoffEvaluator;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class FnsFaqBgeCutoffEvaluatorContractTest {

    @TempDir
    Path tempDirectory;

    ObjectMapper objectMapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);

    @Test
    void run_shouldRecoverEveryCutoffAndKeepFrozenDenominators() throws Exception {
        Fixture fixture = writeFixture("valid");
        FnsFaqBgeCutoffEvaluator evaluator = new FnsFaqBgeCutoffEvaluator(objectMapper);
        Path firstOutput = tempDirectory.resolve("result-one");

        var result = evaluator.run(
                fixture.bgeManifest(),
                fixture.targetsManifest(),
                List.of(50, 10, 30, 20),
                firstOutput);

        Map<String, Map<String, String>> perQuery = csvRows(
                firstOutput.resolve(FnsFaqBgeCutoffEvaluator.PER_QUERY_FILENAME),
                row -> row.get("qid") + "|" + row.get("cutoff"));
        assertThat(perQuery).hasSize(960);
        assertThat(perQuery.get("q-001|10"))
                .containsEntry("candidate_hit", "false")
                .containsEntry("after_hit_at_10", "false");
        assertThat(perQuery.get("q-001|20"))
                .containsEntry("candidate_hit", "true")
                .containsEntry("after_rank_at_10", "1")
                .containsEntry("comparison", "win");
        assertThat(perQuery.get("q-002|10"))
                .containsEntry("before_rank_at_10", "1")
                .containsEntry("after_rank_at_10", "10");
        assertThat(perQuery.get("q-002|20"))
                .containsEntry("after_rank_at_10", "0")
                .containsEntry("comparison", "loss");
        assertThat(perQuery.get("q-003|10"))
                .containsEntry("candidate_recall", "0.500000000")
                .containsEntry("all_targets", "false");
        assertThat(perQuery.get("q-003|30"))
                .containsEntry("candidate_recall", "1.000000000")
                .containsEntry("all_targets", "true");
        assertThat(perQuery.get("q-004|50"))
                .containsEntry("after_rank_at_10", "10")
                .containsEntry("comparison", "tie");
        assertThat(perQuery.get("q-005|50"))
                .containsEntry("candidate_hit", "false")
                .containsEntry("after_hit_at_10", "false");
        assertThat(perQuery.get("q-240|50"))
                .containsEntry("status", "excluded")
                .containsEntry("target_count", "")
                .containsEntry("candidate_hit", "");

        Map<String, Map<String, String>> aggregate = csvRows(
                firstOutput.resolve(FnsFaqBgeCutoffEvaluator.AGGREGATE_FILENAME),
                row -> row.get("slice") + "|" + row.get("cutoff"));
        assertThat(aggregate.get("primary_cited_clause|50"))
                .containsEntry("reporting_role", "primary")
                .containsEntry("eligible_queries", "4")
                .containsEntry("candidate_hit_at_cutoff", "0.750000000")
                .containsEntry("mean_delta_ndcg_at_10", "0.000000000")
                .containsEntry("paired_wins", "1")
                .containsEntry("paired_ties", "2")
                .containsEntry("paired_losses", "1");
        assertThat(aggregate.get("primary_cited_clause|10"))
                .containsEntry("reporting_role", "sensitivity")
                .containsEntry("eligible_queries", "4");
        assertThat(aggregate.get("article_only_single_chunk|50"))
                .containsEntry("reporting_role", "sensitivity")
                .containsEntry("eligible_queries", "1");
        assertThat(aggregate.get("multi_target|10"))
                .containsEntry("candidate_recall_at_cutoff", "0.500000000")
                .containsEntry("all_targets_at_cutoff", "0.000000000");

        assertThat(result.manifest().eligibleQuestionCount()).isEqualTo(5);
        assertThat(result.manifest().cutoffs())
                .extracting(FnsFaqBgeCutoffEvaluator.CutoffDefinition::reportingRole)
                .containsExactly("sensitivity", "sensitivity", "sensitivity", "primary");
        assertThat(result.manifest().bootstrap().resamples()).isEqualTo(10_000);
        assertThat(result.manifestSha256()).isEqualTo(
                FnsFaqBgeCutoffEvaluator.sha256(Files.readAllBytes(
                        firstOutput.resolve(FnsFaqBgeCutoffEvaluator.MANIFEST_FILENAME))));

        Path secondOutput = tempDirectory.resolve("result-two");
        evaluator.run(
                fixture.bgeManifest(),
                fixture.targetsManifest(),
                FnsFaqBgeCutoffEvaluator.REQUIRED_CUTOFFS,
                secondOutput);
        for (String filename : List.of(
                FnsFaqBgeCutoffEvaluator.PER_QUERY_FILENAME,
                FnsFaqBgeCutoffEvaluator.AGGREGATE_FILENAME,
                FnsFaqBgeCutoffEvaluator.MANIFEST_FILENAME,
                FnsFaqBgeCutoffEvaluator.MANIFEST_SHA_FILENAME)) {
            assertThat(Files.readAllBytes(secondOutput.resolve(filename)))
                    .isEqualTo(Files.readAllBytes(firstOutput.resolve(filename)));
        }
    }

    @Test
    void run_shouldRejectTamperedBgeProvenance() throws Exception {
        Fixture fixture = writeFixture("tampered");
        Files.writeString(
                fixture.bgeProvenance(),
                " ",
                StandardCharsets.UTF_8,
                StandardOpenOption.APPEND);

        assertThatThrownBy(() -> run(fixture, "tampered-output"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("BGE sealed provenance SHA-256 differs from manifest");
    }

    @Test
    void run_shouldRejectTamperedTargetLabels() throws Exception {
        Fixture fixture = writeFixture("tampered-targets");
        Files.writeString(
                fixture.targets(),
                " ",
                StandardCharsets.UTF_8,
                StandardOpenOption.APPEND);

        assertThatThrownBy(() -> run(fixture, "tampered-targets-output"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("cited-clause labels SHA-256 differs from manifest");
    }

    @Test
    void run_shouldRejectUnknownTargetQid() throws Exception {
        Fixture fixture = writeFixture("unknown-qid");
        ObjectNode labels = (ObjectNode) objectMapper.readTree(fixture.targets().toFile());
        ((ObjectNode) labels.withArray("labels").get(239)).put("qid", "q-unknown");
        rewriteTargets(fixture, labels);

        assertThatThrownBy(() -> run(fixture, "unknown-qid-output"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Unknown qid in targets: q-unknown");
    }

    @Test
    void run_shouldRejectMalformedTargetDocId() throws Exception {
        Fixture fixture = writeFixture("unknown-doc");
        ObjectNode labels = (ObjectNode) objectMapper.readTree(fixture.targets().toFile());
        ((ObjectNode) labels.withArray("labels").get(0))
                .putArray("targetDocIds")
                .add("unknown-doc");
        rewriteTargets(fixture, labels);

        assertThatThrownBy(() -> run(fixture, "unknown-doc-output"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Invalid target docId for q-001");
    }

    @Test
    void run_shouldRejectDuplicateAndMissingFirstStageRank() throws Exception {
        Fixture fixture = writeFixture("bad-rank");
        List<String> lines = Files.readAllLines(fixture.bgeProvenance(), StandardCharsets.UTF_8);
        ObjectNode first = (ObjectNode) objectMapper.readTree(lines.getFirst());
        ArrayNode candidates = first.withArray("bge");
        int duplicateRank = candidates.get(1).path("firstStageRank").intValue();
        ((ObjectNode) candidates.get(0)).put("firstStageRank", duplicateRank);
        lines.set(0, compact(first));
        Files.write(fixture.bgeProvenance(), lines, StandardCharsets.UTF_8);
        updateArtifactHash(fixture.bgeManifest(), "sealedProvenance", fixture.bgeProvenance());

        assertThatThrownBy(() -> run(fixture, "bad-rank-output"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Invalid BGE firstStageRank for q-001");
    }

    @Test
    void run_shouldRejectSelectiveCutoffList() throws Exception {
        Fixture fixture = writeFixture("selective-cutoffs");

        assertThatThrownBy(() -> new FnsFaqBgeCutoffEvaluator(objectMapper).run(
                fixture.bgeManifest(),
                fixture.targetsManifest(),
                List.of(10, 50),
                tempDirectory.resolve("selective-output")))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessage("Cutoffs must be exactly 10,20,30,50");
    }

    @Test
    void run_shouldRejectLegacyBgeMaxLength() throws Exception {
        Fixture fixture = writeFixture("legacy-max-length");
        ObjectNode manifest = (ObjectNode) objectMapper.readTree(fixture.bgeManifest().toFile());
        ((ObjectNode) manifest.path("rerankerAttestation")).put("maxLength", 512);
        ((ObjectNode) manifest.path("pooling")).put("maxLength", 512);
        objectMapper.writeValue(fixture.bgeManifest().toFile(), manifest);

        assertThatThrownBy(() -> run(fixture, "legacy-max-length-output"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessage("BGE reranker max length differs from the frozen contract");
    }

    private void run(Fixture fixture, String output) throws Exception {
        new FnsFaqBgeCutoffEvaluator(objectMapper).run(
                fixture.bgeManifest(),
                fixture.targetsManifest(),
                FnsFaqBgeCutoffEvaluator.REQUIRED_CUTOFFS,
                tempDirectory.resolve(output));
    }

    private Fixture writeFixture(String name) throws Exception {
        Path root = tempDirectory.resolve(name);
        Path bgeDirectory = root.resolve("bge");
        Path targetsDirectory = root.resolve("targets");
        Files.createDirectories(bgeDirectory);
        Files.createDirectories(targetsDirectory);

        Path provenance = bgeDirectory.resolve("pooling-provenance.sealed.jsonl");
        List<String> provenanceLines = new ArrayList<>();
        for (int question = 1; question <= 240; question++) {
            String qid = qid(question);
            List<CandidateFixture> candidates = new ArrayList<>();
            for (int rank = 1; rank <= 50; rank++) {
                double score = 1_000.0 - rank;
                if (question == 1 && rank == 20) {
                    score = 2_000.0;
                } else if (question == 2 && rank == 1) {
                    score = -100.0;
                } else if (question == 4) {
                    score = 1.0;
                }
                candidates.add(new CandidateFixture(rank, docId(question, rank), score));
            }
            candidates.sort(Comparator.comparingDouble(CandidateFixture::score).reversed()
                    .thenComparingInt(CandidateFixture::firstStageRank)
                    .thenComparing(CandidateFixture::docId));
            ObjectNode row = objectMapper.createObjectNode().put("qid", qid);
            ArrayNode bge = row.putArray("bge");
            for (int index = 0; index < candidates.size(); index++) {
                CandidateFixture candidate = candidates.get(index);
                bge.addObject()
                        .put("firstStageRank", candidate.firstStageRank())
                        .put("rerankRank", index + 1)
                        .put("docId", candidate.docId())
                        .put("score", candidate.score());
            }
            provenanceLines.add(compact(row));
        }
        Files.write(provenance, provenanceLines, StandardCharsets.UTF_8);

        Path bgeManifest = bgeDirectory.resolve("manifest.json");
        ObjectNode bgeRoot = objectMapper.createObjectNode();
        bgeRoot.put("schemaVersion", 2);
        bgeRoot.put("evalVersion", "tax-eval-v3-bge-reranker-pool");
        bgeRoot.put("questionCount", 240);
        bgeRoot.put("corpusDocumentCount", 2074);
        ObjectNode attestation = bgeRoot.putObject("rerankerAttestation");
        attestation.put("healthSchemaVersion", 1);
        attestation.put("model", FnsFaqBgeCutoffEvaluator.BGE_MODEL);
        attestation.put("revision", FnsFaqBgeCutoffEvaluator.BGE_REVISION);
        attestation.put("precision", FnsFaqBgeCutoffEvaluator.BGE_PRECISION);
        attestation.put("device", FnsFaqBgeCutoffEvaluator.BGE_DEVICE);
        attestation.put("batchSize", FnsFaqBgeCutoffEvaluator.BGE_BATCH_SIZE);
        attestation.put("maxLength", FnsFaqBgeCutoffEvaluator.BGE_MAX_LENGTH);
        attestation.put("healthResponseSha256", "a".repeat(64));
        ObjectNode pooling = bgeRoot.putObject("pooling");
        pooling.put("topN", 10);
        pooling.put("scoredPool", 50);
        pooling.put("model", FnsFaqBgeCutoffEvaluator.BGE_MODEL);
        pooling.put("revision", FnsFaqBgeCutoffEvaluator.BGE_REVISION);
        pooling.put("precision", FnsFaqBgeCutoffEvaluator.BGE_PRECISION);
        pooling.put("device", FnsFaqBgeCutoffEvaluator.BGE_DEVICE);
        pooling.put("batchSize", FnsFaqBgeCutoffEvaluator.BGE_BATCH_SIZE);
        pooling.put("maxLength", FnsFaqBgeCutoffEvaluator.BGE_MAX_LENGTH);
        pooling.put("ordering", "score_desc,first_stage_rank_asc,doc_id_asc");
        artifact(bgeRoot.putObject("sealedProvenance"), provenance);
        objectMapper.writeValue(bgeManifest.toFile(), bgeRoot);

        Path labelsPath = targetsDirectory.resolve("cited_clause_labels.json");
        ObjectNode labelSet = objectMapper.createObjectNode();
        labelSet.put("schemaVersion", 1);
        labelSet.put("evalVersion", "tax-eval-v3-cited-clause-silver");
        labelSet.put("freezeStatus", "frozen");
        labelSet.put("ruleVersion", "strict-cited-clause-silver-v1");
        labelSet.put("parserVersion", "fns-cited-clause-parser-v1");
        labelSet.put("questionCount", 240);
        ArrayNode labels = labelSet.putArray("labels");
        for (int question = 1; question <= 240; question++) {
            ObjectNode label = labels.addObject();
            label.put("qid", qid(question));
            label.put("queryGroup", "group-" + question);
            label.put("categoryId", question % 14 + 1);
            if (question == 1) {
                eligible(label, "eligible_cited_clause", "primary_cited_clause", docId(1, 20));
            } else if (question == 2) {
                eligible(label, "eligible_cited_clause", "primary_cited_clause", docId(2, 1));
            } else if (question == 3) {
                eligible(
                        label,
                        "eligible_article_only_single_chunk",
                        "article_only_single_chunk",
                        docId(3, 2),
                        docId(3, 25));
            } else if (question == 4) {
                eligible(label, "eligible_cited_clause", "primary_cited_clause", docId(4, 10));
            } else if (question == 5) {
                eligible(
                        label,
                        "eligible_cited_clause",
                        "primary_cited_clause",
                        "nk-2-article-999-chunk-001");
            } else {
                label.put("status", "excluded");
                label.put("slice", "none");
                label.put("reason", "no_resolved_literal_citation");
                label.putArray("targetDocIds");
            }
        }
        objectMapper.writeValue(labelsPath.toFile(), labelSet);

        Path targetsManifest = targetsDirectory.resolve("manifest.json");
        ObjectNode targetRoot = objectMapper.createObjectNode();
        targetRoot.put("schemaVersion", 1);
        targetRoot.put("evalVersion", "tax-eval-v3-cited-clause-silver");
        targetRoot.put("freezeStatus", "frozen");
        targetRoot.put("ruleVersion", "strict-cited-clause-silver-v1");
        targetRoot.put("parserVersion", "fns-cited-clause-parser-v1");
        targetRoot.put("rulesetSha256", "a".repeat(64));
        targetRoot.put("questionCount", 240);
        targetRoot.put("primaryCitedClauseCount", 4);
        targetRoot.put("articleOnlySingleChunkCount", 1);
        targetRoot.put("mixedLiteralCount", 0);
        targetRoot.put("excludedCount", 235);
        targetRoot.set("statusCounts", objectMapper.valueToTree(Map.of(
                "eligible_cited_clause", 4,
                "eligible_article_only_single_chunk", 1,
                "excluded", 235)));
        targetRoot.set("reasonCounts", objectMapper.valueToTree(Map.of(
                "resolved_literal_citation", 5,
                "no_resolved_literal_citation", 235)));
        artifact(targetRoot.putObject("sourceManifest"), "source-manifest.json", "b".repeat(64));
        artifact(targetRoot.putObject("reviewPacket"), "review_packet.json", "c".repeat(64));
        artifact(targetRoot.putObject("corpus"), "corpus.json", "d".repeat(64));
        artifact(targetRoot.putObject("labels"), labelsPath);
        objectMapper.writeValue(targetsManifest.toFile(), targetRoot);

        return new Fixture(bgeManifest, provenance, targetsManifest, labelsPath);
    }

    private static void eligible(ObjectNode label, String status, String slice, String... docIds) {
        label.put("status", status);
        label.put("slice", slice);
        label.put("reason", "resolved_literal_citation");
        ArrayNode targets = label.putArray("targetDocIds");
        for (String docId : docIds) {
            targets.add(docId);
        }
    }

    private void rewriteTargets(Fixture fixture, ObjectNode labels) throws Exception {
        objectMapper.writeValue(fixture.targets().toFile(), labels);
        updateArtifactHash(fixture.targetsManifest(), "labels", fixture.targets());
    }

    private String compact(JsonNode value) throws Exception {
        return objectMapper.writer().without(SerializationFeature.INDENT_OUTPUT)
                .writeValueAsString(value);
    }

    private void updateArtifactHash(Path manifestPath, String field, Path artifact) throws Exception {
        ObjectNode manifest = (ObjectNode) objectMapper.readTree(manifestPath.toFile());
        ((ObjectNode) manifest.path(field)).put(
                "sha256",
                FnsFaqBgeCutoffEvaluator.sha256(Files.readAllBytes(artifact)));
        objectMapper.writeValue(manifestPath.toFile(), manifest);
    }

    private static void artifact(ObjectNode target, Path path) throws Exception {
        artifact(
                target,
                path.getFileName().toString(),
                FnsFaqBgeCutoffEvaluator.sha256(Files.readAllBytes(path)));
    }

    private static void artifact(ObjectNode target, String filename, String sha256) {
        target.put("filename", filename);
        target.put("sha256", sha256);
    }

    private static String qid(int question) {
        return "q-%03d".formatted(question);
    }

    private static String docId(int question, int rank) {
        return "nk-1-article-%d-chunk-%03d".formatted(question, rank);
    }

    private static Map<String, Map<String, String>> csvRows(
            Path path,
            java.util.function.Function<Map<String, String>, String> key) throws Exception {
        List<String> lines = Files.readAllLines(path, StandardCharsets.UTF_8);
        String[] headers = lines.getFirst().split(",", -1);
        Map<String, Map<String, String>> rows = new LinkedHashMap<>();
        for (String line : lines.subList(1, lines.size())) {
            String[] values = line.split(",", -1);
            assertThat(values).hasSameSizeAs(headers);
            Map<String, String> row = new HashMap<>();
            for (int index = 0; index < headers.length; index++) {
                row.put(headers[index], values[index]);
            }
            rows.put(key.apply(row), row);
        }
        return rows;
    }

    private record CandidateFixture(int firstStageRank, String docId, double score) {
    }

    private record Fixture(
            Path bgeManifest,
            Path bgeProvenance,
            Path targetsManifest,
            Path targets) {
    }
}
