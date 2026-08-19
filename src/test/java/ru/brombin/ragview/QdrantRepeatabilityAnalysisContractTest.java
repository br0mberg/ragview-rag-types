package ru.brombin.ragview;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.json.JsonMapper;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import ru.brombin.ragview.eval.FnsFaqQdrantHybridParityRunner;
import ru.brombin.ragview.eval.FnsFaqQdrantHybridParityRunner.BackendIdentity;
import ru.brombin.ragview.eval.FnsFaqQdrantHybridParityRunner.DenseParameters;
import ru.brombin.ragview.eval.FnsFaqQdrantHybridParityRunner.EmbeddingIdentity;
import ru.brombin.ragview.eval.FnsFaqQdrantHybridParityRunner.InputArtifact;
import ru.brombin.ragview.eval.FnsFaqQdrantHybridParityRunner.OutputArtifact;
import ru.brombin.ragview.eval.FnsFaqQdrantHybridParityRunner.ParityResult;
import ru.brombin.ragview.eval.FnsFaqQdrantHybridParityRunner.ParitySummary;
import ru.brombin.ragview.eval.FnsFaqQdrantHybridParityRunner.QdrantIdentity;
import ru.brombin.ragview.eval.FnsFaqQdrantHybridParityRunner.RankedCandidate;
import ru.brombin.ragview.eval.FnsFaqQdrantHybridParityRunner.RawRankingRow;
import ru.brombin.ragview.eval.FnsFaqQdrantHybridParityRunner.RrfContract;
import ru.brombin.ragview.eval.FnsFaqQdrantHybridParityRunner.RunManifest;
import ru.brombin.ragview.eval.FnsFaqQdrantHybridParityRunner.RunOptions;
import ru.brombin.ragview.eval.FnsFaqQdrantHybridParityRunner.SparseParameters;
import ru.brombin.ragview.eval.QdrantRepeatabilityAnalysis;
import ru.brombin.ragview.strategy.RetrievedDoc;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class QdrantRepeatabilityAnalysisContractTest {

    @TempDir
    Path temp;

    ObjectMapper objectMapper = JsonMapper.builder()
            .enable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES)
            .enable(SerializationFeature.INDENT_OUTPUT)
            .build();

    @Test
    void write_shouldBeDeterministicAndPublishOnlyAggregates() throws Exception {
        Fixture fixture = fixture();
        QdrantRepeatabilityAnalysis analysis = new QdrantRepeatabilityAnalysis(objectMapper);
        Path first = temp.resolve("summary-first.json");
        Path second = temp.resolve("summary-second.json");

        var summary = analysis.write(fixture.primaryManifest(), fixture.repeatManifest(), first);
        analysis.write(fixture.primaryManifest(), fixture.repeatManifest(), second);

        assertThat(Files.readAllBytes(first)).isEqualTo(Files.readAllBytes(second));
        assertThat(summary.questionCount()).isEqualTo(2);
        assertThat(summary.primaryClientServerParity().sameFullUnionSetCount()).isEqualTo(2);
        assertThat(summary.repeatability().clientRrfFullOrder())
                .isEqualTo(new QdrantRepeatabilityAnalysis.CountPair(2, 0));
        assertThat(summary.repeatability().serverRrfFullOrder())
                .isEqualTo(new QdrantRepeatabilityAnalysis.CountPair(1, 1));
        assertThat(summary.repeatability().serverRrfFullSet())
                .isEqualTo(new QdrantRepeatabilityAnalysis.CountPair(2, 0));
        assertThat(summary.inputs().primary().manifestSha256()).hasSize(64);
        assertThat(summary.inputs().primary().rawRankingsSha256()).hasSize(64);

        String published = Files.readString(first);
        assertThat(published)
                .doesNotContain("q-1", "q-2", "doc-a", "doc-b")
                .endsWith("\n");
    }

    @Test
    void analyze_shouldRejectRawRankingsChangedAfterManifestWasWritten() throws Exception {
        Fixture fixture = fixture();
        Files.writeString(
                fixture.primaryManifest().resolveSibling("rankings.jsonl"),
                "{}\n",
                StandardCharsets.UTF_8);

        assertThatThrownBy(() -> new QdrantRepeatabilityAnalysis(objectMapper)
                .analyze(fixture.primaryManifest(), fixture.repeatManifest()))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessage("primary raw rankings do not match the manifest");
    }

    private Fixture fixture() throws Exception {
        Path primary = temp.resolve("primary");
        Path repeat = temp.resolve("repeat");
        Files.createDirectories(primary);
        Files.createDirectories(repeat);

        List<RawRankingRow> primaryRows = List.of(
                row("q-1", List.of("doc-a", "doc-b")),
                row("q-2", List.of("doc-a", "doc-b")));
        List<RawRankingRow> repeatRows = List.of(
                row("q-1", List.of("doc-b", "doc-a")),
                row("q-2", List.of("doc-a", "doc-b")));
        Path primaryManifest = writeRun(primary, primaryRows, 2, 0);
        Path repeatManifest = writeRun(repeat, repeatRows, 1, 1);
        return new Fixture(primaryManifest, repeatManifest);
    }

    private Path writeRun(
            Path directory,
            List<RawRankingRow> rows,
            int identicalOrderCount,
            int tieOnlyDifferenceCount) throws Exception {
        Path rankings = directory.resolve("rankings.jsonl");
        var writer = objectMapper.writer().without(SerializationFeature.INDENT_OUTPUT);
        StringBuilder raw = new StringBuilder();
        for (RawRankingRow row : rows) {
            raw.append(writer.writeValueAsString(row)).append('\n');
        }
        Files.writeString(rankings, raw, StandardCharsets.UTF_8);
        String rawSha256 = FnsFaqQdrantHybridParityRunner.sha256(Files.readAllBytes(rankings));

        RunManifest manifest = new RunManifest(
                1,
                "tax-eval-v3-qdrant-1.19-native-hybrid-rrf-parity",
                Instant.EPOCH.toString(),
                "fixture-source",
                rows.size(),
                2,
                new InputArtifact("questions.json", "a".repeat(64)),
                new InputArtifact("snapshot.json", "b".repeat(64)),
                new InputArtifact("corpus.json", "c".repeat(64)),
                "d".repeat(64),
                "id+question",
                new RunOptions(1, 2, 60, 61),
                backend(),
                new RrfContract(
                        60,
                        61,
                        "sum(1/(60+rank_one_based))",
                        "sum(1/(61+rank_zero_based))",
                        "score_desc,doc_id_asc",
                        "unspecified",
                        1e-7),
                new ParitySummary(
                        rows.size(),
                        identicalOrderCount,
                        rows.size(),
                        rows.size(),
                        tieOnlyDifferenceCount,
                        0,
                        0),
                new OutputArtifact("rankings.jsonl", rawSha256));
        Path manifestPath = directory.resolve("manifest.json");
        objectMapper.writerWithDefaultPrettyPrinter().writeValue(manifestPath.toFile(), manifest);
        return manifestPath;
    }

    private static BackendIdentity backend() {
        return new BackendIdentity(
                new EmbeddingIdentity(
                        1,
                        "fixture-model",
                        "fixture-revision",
                        "mean",
                        2,
                        16,
                        "query: ",
                        "document: ",
                        "cpu",
                        "e".repeat(64)),
                new DenseParameters("dense", 2, "cosine", true),
                new SparseParameters(
                        "bm25",
                        "fixture-bm25",
                        "russian",
                        "multilingual",
                        1.2,
                        0.75,
                        "fixture",
                        1.0,
                        "idf"),
                new QdrantIdentity(
                        "1.19.0",
                        "commit",
                        "client",
                        "collection",
                        "immutable",
                        "uuid-v3",
                        "f".repeat(64)),
                0,
                0);
    }

    private static RawRankingRow row(String qid, List<String> serverOrder) {
        List<RankedCandidate> dense = List.of(new RankedCandidate(1, "doc-a", 0.5));
        List<RankedCandidate> sparse = List.of(new RankedCandidate(1, "doc-b", 1.0));
        List<RankedCandidate> client = List.of(
                new RankedCandidate(1, "doc-a", 0.01),
                new RankedCandidate(2, "doc-b", 0.01));
        List<RankedCandidate> server = List.of(
                new RankedCandidate(1, serverOrder.get(0), 0.01),
                new RankedCandidate(2, serverOrder.get(1), 0.01));
        ParityResult parity = FnsFaqQdrantHybridParityRunner.compare(
                retrieved(client), retrieved(server));
        return new RawRankingRow(qid, dense, sparse, client, server, parity);
    }

    private static List<RetrievedDoc> retrieved(List<RankedCandidate> ranking) {
        return ranking.stream()
                .map(candidate -> new RetrievedDoc(candidate.docId(), candidate.score()))
                .toList();
    }

    private record Fixture(Path primaryManifest, Path repeatManifest) {
    }
}
