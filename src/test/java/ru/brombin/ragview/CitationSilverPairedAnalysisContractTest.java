package ru.brombin.ragview;

import org.junit.jupiter.api.Test;
import ru.brombin.ragview.eval.CitationSilverPairedAnalysis;

import java.nio.file.Path;

import static org.assertj.core.api.Assertions.assertThat;

class CitationSilverPairedAnalysisContractTest {

    @Test
    void publicPerQueryArtifact_shouldReproducePairedDiagnostics() throws Exception {
        var results = CitationSilverPairedAnalysis.analyze(
                Path.of("results/v3-citation-silver/per-query.csv"));

        assertThat(results).extracting(
                        CitationSilverPairedAnalysis.PairedResult::comparison,
                        CitationSilverPairedAnalysis.PairedResult::k,
                        CitationSilverPairedAnalysis.PairedResult::rescue,
                        CitationSilverPairedAnalysis.PairedResult::harm,
                        CitationSilverPairedAnalysis.PairedResult::exactTwoSidedP)
                .containsExactly(
                        org.assertj.core.groups.Tuple.tuple(
                                "berta:hybrid_vs_dense", 10, 8, 7, 1.0),
                        org.assertj.core.groups.Tuple.tuple(
                                "berta:hybrid_vs_dense", 50, 6, 0, 0.03125),
                        org.assertj.core.groups.Tuple.tuple(
                                "berta:hybrid_vs_bm25", 50, 6, 1, 0.125),
                        org.assertj.core.groups.Tuple.tuple(
                                "qdrant-native-hybrid-rrf:clientRrf_vs_dense", 10, 7, 9, 0.803619384765625),
                        org.assertj.core.groups.Tuple.tuple(
                                "qdrant-native-hybrid-rrf:clientRrf_vs_dense", 50, 7, 0, 0.015625),
                        org.assertj.core.groups.Tuple.tuple(
                                "qdrant-native-hybrid-rrf:clientRrf_vs_nativeSparse", 50, 8, 0, 0.0078125));
    }

    @Test
    void publicPerQueryArtifact_shouldReproduceLiteralCitationSensitivity() throws Exception {
        var results = CitationSilverPairedAnalysis.analyze(
                Path.of("results/v3-citation-silver/per-query.csv"),
                "literal_citation_all");

        assertThat(results).extracting(
                        CitationSilverPairedAnalysis.PairedResult::comparison,
                        CitationSilverPairedAnalysis.PairedResult::k,
                        CitationSilverPairedAnalysis.PairedResult::rescue,
                        CitationSilverPairedAnalysis.PairedResult::harm)
                .containsExactly(
                        org.assertj.core.groups.Tuple.tuple(
                                "berta:hybrid_vs_dense", 10, 8, 9),
                        org.assertj.core.groups.Tuple.tuple(
                                "berta:hybrid_vs_dense", 50, 6, 0),
                        org.assertj.core.groups.Tuple.tuple(
                                "berta:hybrid_vs_bm25", 50, 6, 1),
                        org.assertj.core.groups.Tuple.tuple(
                                "qdrant-native-hybrid-rrf:clientRrf_vs_dense", 10, 7, 10),
                        org.assertj.core.groups.Tuple.tuple(
                                "qdrant-native-hybrid-rrf:clientRrf_vs_dense", 50, 7, 0),
                        org.assertj.core.groups.Tuple.tuple(
                                "qdrant-native-hybrid-rrf:clientRrf_vs_nativeSparse", 50, 8, 0));
    }
}
