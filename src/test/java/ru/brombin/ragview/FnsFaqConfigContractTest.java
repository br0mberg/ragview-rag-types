package ru.brombin.ragview;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.json.JsonMapper;
import org.junit.jupiter.api.Test;
import ru.brombin.ragview.eval.FnsFaqSourceConfig;

import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.HexFormat;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class FnsFaqConfigContractTest {

    @Test
    void sourceConfig_shouldIncludeFrozenSelectionDedupAndScope() throws Exception {
        var mapper = JsonMapper.builder()
                .enable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES)
                .build();

        FnsFaqSourceConfig config = mapper.readValue(
                Path.of("data/eval/v3_source_config.json").toFile(), FnsFaqSourceConfig.class);

        assertThat(config.selection().targetTotal()).isEqualTo(240);
        assertThat(config.selection().mode()).isEqualTo("balanced_by_category");
        assertThat(config.nearDuplicates().algorithm()).isEqualTo("normalized_levenshtein");
        assertThat(config.nearDuplicates().threshold()).isEqualTo(0.75);
        assertThat(config.scope().level()).isEqualTo("federal");
        assertThat(config.scope().corpusSha256()).hasSize(64);
        assertThat(config.requestDelayMs()).isGreaterThanOrEqualTo(500);
    }

    @Test
    void sourceConfig_shouldRejectUnknownProperties() {
        var mapper = JsonMapper.builder()
                .enable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES)
                .build();
        String json = """
                {
                  "schemaVersion": 1,
                  "source": "https://example.test/kb/",
                  "endpoint": "https://example.test/Ajax.html",
                  "regionId": 446,
                  "selectionSeed": "seed",
                  "pagesPerCategory": 1,
                  "requestDelayMs": 500,
                  "selection": {"unit":"question","targetTotal":1,"mode":"balanced_by_category","pageSelection":"sha256_rank_without_replacement","splitSeed":"split"},
                  "nearDuplicates": {"algorithm":"normalized_levenshtein","threshold":0.75},
                  "scope": {"level":"federal","corpusDate":"2026-07-11","corpusSha256":"0000000000000000000000000000000000000000000000000000000000000000"},
                  "categories": [{"id": 1, "name": "one"}],
                  "unexpected": true
                }
                """;

        assertThatThrownBy(() -> mapper.readValue(json, FnsFaqSourceConfig.class))
                .hasMessageContaining("unexpected");
    }

    @Test
    void publicFrameManifest_shouldFreezeInputsWithoutPublishingFnsContent() throws Exception {
        var mapper = new JsonMapper();
        JsonNode manifest = mapper.readTree(Path.of("data/eval/v3_frame_manifest.json").toFile());

        assertThat(manifest.path("schemaVersion").asInt()).isEqualTo(2);
        assertThat(manifest.path("selection").path("selectedQuestions").asInt()).isEqualTo(240);
        assertThat(manifest.path("selection").path("naturalExactQuestions").asInt()).isEqualTo(3);
        assertThat(manifest.path("selection").path("configSha256").asText())
                .isEqualTo(sha256(Path.of("data/eval/v3_source_config.json")));
        assertThat(manifest.path("reviewSubset").path("selectedQuestions").asInt()).isEqualTo(120);
        assertThat(manifest.path("reviewSubset").path("packetStatus").asText())
                .isEqualTo("superseded_requires_reassembly_with_bge_max_length_1024");
        assertThat(manifest.path("reviewSubset").path("bgeDependency").path("maxLength").asInt())
                .isEqualTo(512);
        assertThat(manifest.path("reviewSubset").path("bgeDependency").path("reassemblyStatus").asText())
                .isEqualTo("pending");
        assertThat(manifest.path("reviewSubset").path("fullAssemblyManifestSha256").asText())
                .isEqualTo("31f4f8a02c183a847e2c28758bc9fde45945920f096db3e7cc4ce6d700a1af05");
        assertThat(manifest.path("reviewSubset").path("subsetAssemblyManifestSha256").asText())
                .isEqualTo("c395299e2ca0d01e8ff51515e8b3433ec62907b6549c5dbff77968198bfb58d5");
        var bge = manifest.path("qrelPooling").path("bgeTop10FromBertaHybridTop50");
        assertThat(bge.path("manifestSchemaVersion").asInt()).isEqualTo(2);
        assertThat(bge.path("manifestSha256").asText())
                .isEqualTo("b238793211ef2a6533b53c4d43855a7ab06b7eac5d5be4a3eee5f89926807a51");
        assertThat(bge.path("reviewerPoolSha256").asText())
                .isEqualTo("1e8131ab7172db5e54ac942ca4de09192577a488545238ecfad2357c557c9364");
        assertThat(bge.path("sealedProvenanceSha256").asText())
                .isEqualTo("92016c8fd914360d9da8c5bedf72af6867657c403fa5ae37d89480cf040bd4cf");
        assertThat(bge.path("healthResponseSha256").asText())
                .isEqualTo("3b0f88209ebd04f615143d8d83ca89b2cc357444826b2d0e6db527a08201c611");
        assertThat(bge.path("maxLength").asInt()).isEqualTo(1024);
        assertThat(manifest.path("evaluation").path("goldQrelsFrozen").asBoolean()).isFalse();
        assertThat(manifest.path("evaluation").path("goldQualityMetricsCalculated").asBoolean()).isFalse();
        assertThat(manifest.path("citationSilver").path("auditSchemaVersion").asInt()).isEqualTo(3);
        assertThat(manifest.path("citationSilver").path("auditDecisionsSha256").asText())
                .isEqualTo("bdeb58b27406e8a3b3933a8fa01286c7390c0f7b4a56d3c288cbff92d7fbe570");
        assertThat(manifest.path("citationSilver").path("evaluationManifestSha256").asText())
                .isEqualTo("2209ab3e8056cdd24dff475229df2f0a21a6c427adc62707e3d8f02f4a728e71");
        assertThat(manifest.path("citationSilver").path("literalCitationQuestions").asInt()).isEqualTo(191);
        assertThat(manifest.path("citationSilver").path("metricEligibleQuestions").asInt()).isEqualTo(186);
        assertThat(manifest.path("citationSilver").path("evaluationSlices")
                .path("preliminaryFiltered").path("reportingRole").asText())
                .isEqualTo("primary_technical_before_expert_validation");
        assertThat(manifest.path("citationSilver").path("evaluationSlices")
                .path("literalCitationAll").path("reportingRole").asText())
                .isEqualTo("sensitivity");
        var citedClause = manifest.path("citedClauseSilver");
        assertThat(citedClause.path("primaryCitedClauseQuestions").asInt()).isEqualTo(79);
        assertThat(citedClause.path("labelsSha256").asText())
                .isEqualTo("f73f2b802395f3f68544bc47e35bedf8fa08680eff6574b1d7095d3f5213f49f");
        assertThat(citedClause.path("bgeEvaluation").path("primaryCutoff").asInt()).isEqualTo(50);
        assertThat(citedClause.path("bgeEvaluation").path("beforeHitAt10").asText()).isEqualTo("67/79");
        assertThat(citedClause.path("bgeEvaluation").path("afterHitAt10").asText()).isEqualTo("71/79");
        assertThat(citedClause.path("bgeEvaluation").path("beforeNdcgAt10").asDouble())
                .isEqualTo(0.603152288);
        assertThat(citedClause.path("bgeEvaluation").path("afterNdcgAt10").asDouble())
                .isEqualTo(0.719341514);
        assertThat(citedClause.path("bgeEvaluation").path("publicManifestSha256").asText())
                .isEqualTo("53682910c4d9e882924774e58360ca62b7f679f80870a4cced1f4906d6a8da0b");
        assertThat(citedClause.path("bgeEvaluation").path("c20OptimalityClaim").asBoolean()).isFalse();
        assertThat(manifest.path("evaluation").path("citedClauseSilverQualityMetricsCalculated").asBoolean())
                .isTrue();

        String serialized = manifest.toString();
        assertThat(serialized).doesNotContain("rawAnswer", "citedSource", "review_packet_A.json", "questionText");
    }

    private static String sha256(Path path) throws Exception {
        byte[] digest = MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(path));
        return HexFormat.of().formatHex(digest);
    }
}
