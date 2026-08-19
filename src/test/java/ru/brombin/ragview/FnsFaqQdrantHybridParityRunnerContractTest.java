package ru.brombin.ragview;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.json.JsonMapper;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import ru.brombin.ragview.corpus.CorpusDocument;
import ru.brombin.ragview.eval.FnsFaqQdrantHybridParityRunner;
import ru.brombin.ragview.eval.FnsFaqQdrantHybridParityRunner.BackendIdentity;
import ru.brombin.ragview.eval.FnsFaqQdrantHybridParityRunner.DenseParameters;
import ru.brombin.ragview.eval.FnsFaqQdrantHybridParityRunner.EmbeddingIdentity;
import ru.brombin.ragview.eval.FnsFaqQdrantHybridParityRunner.QdrantIdentity;
import ru.brombin.ragview.eval.FnsFaqQdrantHybridParityRunner.QueryRankings;
import ru.brombin.ragview.eval.FnsFaqQdrantHybridParityRunner.RunOptions;
import ru.brombin.ragview.eval.FnsFaqQdrantHybridParityRunner.SparseParameters;
import ru.brombin.ragview.retriever.ReciprocalRankFusion;
import ru.brombin.ragview.strategy.RetrievedDoc;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class FnsFaqQdrantHybridParityRunnerContractTest {

    @TempDir
    Path temp;

    ObjectMapper objectMapper = JsonMapper.builder()
            .enable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES)
            .enable(SerializationFeature.INDENT_OUTPUT)
            .build();

    @Test
    void run_shouldUseOnlyQuestionProjectionAndSealAllRawRanks() throws Exception {
        Fixture fixture = fixture();
        StubBackend backend = new StubBackend();
        FnsFaqQdrantHybridParityRunner runner =
                new FnsFaqQdrantHybridParityRunner(objectMapper, backend);
        Path output = temp.resolve("run");

        var manifest = runner.run(
                fixture.questions(),
                fixture.corpus(),
                output,
                new RunOptions(2, 4, 60, 61),
                Instant.parse("2026-08-13T20:00:00Z"));

        assertThat(backend.seenQuestions).hasSize(240).allMatch(value -> value.startsWith("question-"));
        assertThat(manifest.questionCount()).isEqualTo(240);
        assertThat(manifest.questionProjectionSha256()).hasSize(64);
        assertThat(manifest.backend().sparse().averageLength()).isEqualTo(2.0);
        assertThat(manifest.rrf().clientKOneBased()).isEqualTo(60);
        assertThat(manifest.rrf().qdrantKZeroBased()).isEqualTo(61);
        assertThat(manifest.rrf().clientFormula()).isEqualTo("sum(1/(60+rank_one_based))");
        assertThat(manifest.rrf().qdrantFormula()).isEqualTo("sum(1/(61+rank_zero_based))");
        assertThat(manifest.parity().identicalOrderCount()).isEqualTo(240);
        assertThat(manifest.parity().maximumScoreDelta()).isLessThan(1e-7);

        List<String> lines = Files.readAllLines(
                output.resolve("question-only-rankings.sealed.jsonl"), StandardCharsets.UTF_8);
        assertThat(lines).hasSize(240);
        var first = objectMapper.readTree(lines.getFirst());
        assertThat(first.path("qid").textValue()).isEqualTo("q-000");
        assertThat(first.path("dense")).hasSize(2);
        assertThat(first.path("nativeSparse")).hasSize(2);
        assertThat(first.path("clientRrf")).hasSize(3);
        assertThat(first.path("serverRrf")).hasSize(3);
        assertThat(first.has("question")).isFalse();
        assertThat(first.has("answer")).isFalse();

        String manifestText = Files.readString(output.resolve("manifest.json"));
        assertThat(manifestText).doesNotContain("answer-").doesNotContain("citedSource");
        assertThat(Files.readAllLines(output.resolve("MANIFEST.sha256"))).hasSize(2);
    }

    @Test
    void run_shouldFailWhenSnapshotHashWasChanged() throws Exception {
        Fixture fixture = fixture();
        Files.writeString(fixture.questions(), "[]", StandardCharsets.UTF_8);
        FnsFaqQdrantHybridParityRunner runner =
                new FnsFaqQdrantHybridParityRunner(objectMapper, new StubBackend());

        assertThatThrownBy(() -> runner.run(
                fixture.questions(),
                fixture.corpus(),
                temp.resolve("output"),
                new RunOptions(2, 4, 60, 61),
                Instant.EPOCH))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessage("candidates.json does not match its frozen manifest");
    }

    @Test
    void runOptions_shouldTranslateOneBasedClientKToQdrantZeroBasedK() {
        assertThatThrownBy(() -> new RunOptions(50, 100, 60, 60))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("zero-based k");
    }

    @Test
    void run_shouldWriteConfiguredRrfFormula() throws Exception {
        Fixture fixture = fixture();
        FnsFaqQdrantHybridParityRunner runner =
                new FnsFaqQdrantHybridParityRunner(objectMapper, new StubBackend());

        var manifest = runner.run(
                fixture.questions(),
                fixture.corpus(),
                temp.resolve("run-k20"),
                new RunOptions(2, 4, 20, 21),
                Instant.EPOCH);

        assertThat(manifest.rrf().clientFormula()).isEqualTo("sum(1/(20+rank_one_based))");
        assertThat(manifest.rrf().qdrantFormula()).isEqualTo("sum(1/(21+rank_zero_based))");
    }

    @Test
    void compare_shouldClassifyOnlyEqualScorePermutationAsTieOnly() {
        List<RetrievedDoc> client = List.of(
                new RetrievedDoc("a", 0.02),
                new RetrievedDoc("b", 0.01),
                new RetrievedDoc("c", 0.01));
        List<RetrievedDoc> server = List.of(
                new RetrievedDoc("a", 0.02),
                new RetrievedDoc("c", 0.01),
                new RetrievedDoc("b", 0.01));

        var parity = FnsFaqQdrantHybridParityRunner.compare(client, server);

        assertThat(parity.identicalOrder()).isFalse();
        assertThat(parity.sameSet()).isTrue();
        assertThat(parity.scoreEquivalent()).isTrue();
        assertThat(parity.tieOnlyDifference()).isTrue();
        assertThat(parity.firstDifferenceRank()).isEqualTo(2);
    }

    @Test
    void compare_shouldRejectPermutationAcrossDifferentScores() {
        List<RetrievedDoc> client = List.of(
                new RetrievedDoc("a", 0.02),
                new RetrievedDoc("b", 0.01));
        List<RetrievedDoc> server = List.of(
                new RetrievedDoc("b", 0.01),
                new RetrievedDoc("a", 0.02));

        var parity = FnsFaqQdrantHybridParityRunner.compare(client, server);

        assertThat(parity.scoreEquivalent()).isTrue();
        assertThat(parity.tieOnlyDifference()).isFalse();
    }

    private Fixture fixture() throws Exception {
        Path snapshot = Files.createDirectories(temp.resolve("snapshot"));
        Path questions = snapshot.resolve("candidates.json");
        Path corpus = temp.resolve("corpus.json");
        List<CorpusDocument> documents = List.of(
                new CorpusDocument("d1", "one", "text"),
                new CorpusDocument("d2", "two", "text"),
                new CorpusDocument("d3", "three", "text"));
        objectMapper.writeValue(corpus.toFile(), documents);
        String corpusSha = FnsFaqQdrantHybridParityRunner.sha256(Files.readAllBytes(corpus));

        List<Object> candidates = new ArrayList<>();
        for (int index = 0; index < 240; index++) {
            candidates.add(java.util.Map.of(
                    "id", "q-%03d".formatted(index),
                    "question", "question-%03d".formatted(index),
                    "answer", "answer-%03d".formatted(index),
                    "citedSource", "source-%03d".formatted(index)));
        }
        objectMapper.writeValue(questions.toFile(), candidates);
        String questionsSha = FnsFaqQdrantHybridParityRunner.sha256(Files.readAllBytes(questions));
        objectMapper.writeValue(snapshot.resolve("manifest.json").toFile(), java.util.Map.of(
                "schemaVersion", 1,
                "evalVersion", "test-v3",
                "selectedCandidates", 240,
                "candidatesSha256", questionsSha,
                "scope", java.util.Map.of("corpusSha256", corpusSha)));
        return new Fixture(questions, corpus);
    }

    private static final class StubBackend implements FnsFaqQdrantHybridParityRunner.Backend {

        List<String> seenQuestions = new ArrayList<>();

        @Override
        public BackendIdentity prepare(List<CorpusDocument> corpus, String corpusSha256) {
            return new BackendIdentity(
                    new EmbeddingIdentity(
                            1,
                            FnsFaqQdrantHybridParityRunner.BERTA_MODEL,
                            FnsFaqQdrantHybridParityRunner.BERTA_REVISION,
                            FnsFaqQdrantHybridParityRunner.BERTA_POOLING,
                            FnsFaqQdrantHybridParityRunner.BERTA_DIMENSIONS,
                            512,
                            FnsFaqQdrantHybridParityRunner.BERTA_QUERY_PREFIX,
                            FnsFaqQdrantHybridParityRunner.BERTA_DOCUMENT_PREFIX,
                            "cuda",
                            "a".repeat(64)),
                    new DenseParameters("dense", 768, "cosine", true),
                    new SparseParameters(
                            "bm25", "qdrant/bm25", "russian", "multilingual", 1.2, 0.75,
                            "unicode-whitespace-v1", 2.0, "idf"),
                    new QdrantIdentity(
                            "1.19.0", "commit", "qdrant-java-client-1.19.0", "collection",
                            "create-only", "uuid", "b".repeat(64)),
                    1,
                    corpus.size());
        }

        @Override
        public QueryRankings search(String question, RunOptions options) {
            seenQuestions.add(question);
            List<RetrievedDoc> dense = List.of(
                    new RetrievedDoc("d1", 0.9),
                    new RetrievedDoc("d2", 0.8));
            List<RetrievedDoc> sparse = List.of(
                    new RetrievedDoc("d2", 3.0),
                    new RetrievedDoc("d3", 2.0));
            List<RetrievedDoc> client = ReciprocalRankFusion.fuse(
                    options.fusionLimit(), options.clientRrfKOneBased(), dense, sparse);
            List<RetrievedDoc> server = client.stream()
                    .map(document -> new RetrievedDoc(document.docId(), (float) document.score()))
                    .toList();
            return new QueryRankings(dense, sparse, client, server);
        }
    }

    private record Fixture(Path questions, Path corpus) {
    }
}
