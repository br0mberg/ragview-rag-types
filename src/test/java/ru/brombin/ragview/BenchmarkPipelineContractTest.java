package ru.brombin.ragview;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.EmbeddingRequest;
import org.springframework.ai.embedding.EmbeddingResponse;
import org.springframework.beans.factory.support.StaticListableBeanFactory;
import ru.brombin.ragview.artifact.CandidateDumpWriter;
import ru.brombin.ragview.config.RagProperties;
import ru.brombin.ragview.corpus.CorpusDocument;
import ru.brombin.ragview.corpus.CorpusLoader;
import ru.brombin.ragview.corpus.EvalQuestion;
import ru.brombin.ragview.metrics.ComparisonCsvWriter;
import ru.brombin.ragview.metrics.RetrievalEvaluator;
import ru.brombin.ragview.rerank.RerankCsvWriter;
import ru.brombin.ragview.rerank.RerankEvaluator;
import ru.brombin.ragview.rerank.RerankScorer;
import ru.brombin.ragview.retriever.AsymmetricEmbedding;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
class BenchmarkPipelineContractTest {

    @TempDir
    Path tempDirectory;

    @Test
    void runner_shouldProduceRetrievalAndRerankArtifactsOffline() throws Exception {
        List<CorpusDocument> corpus = List.of(
                new CorpusDocument("d1", "", "акциз"),
                new CorpusDocument("d2", "", "пошлина"),
                new CorpusDocument("d3", "", "налог"));
        List<EvalQuestion> questions = List.of(
                new EvalQuestion("q1", "акциз", "d1", List.of("d1"), "exact"),
                new EvalQuestion("q2", "пошлина", "d2", List.of("d2"), "semantic"));
        Path dataset = tempDirectory.resolve("dataset");
        Files.createDirectories(dataset);
        ObjectMapper objectMapper = new ObjectMapper();
        objectMapper.writeValue(dataset.resolve("corpus.json").toFile(), corpus);
        objectMapper.writeValue(dataset.resolve("questions.json").toFile(), questions);
        Path comparison = tempDirectory.resolve("run/comparison.csv");
        RagProperties properties = new RagProperties(
                1, 3, 3, 60, true, List.of(1, 3), 0,
                List.of("dense", "bm25", "hybrid"), dataset.toString(), comparison.toString());

        CorpusLoader loader = new CorpusLoader(objectMapper, properties);
        RerankScorer scorer = (query, passages) -> passages.stream()
                .map(passage -> passage.contains(query) ? 10.0 : 0.0)
                .toList();
        StaticListableBeanFactory beans = new StaticListableBeanFactory();
        beans.addBean("rerankScorer", scorer);

        BenchmarkRunner runner = new BenchmarkRunner(
                loader,
                new StrategyFactory(new FakeEmbeddingModel(), properties),
                new RetrievalEvaluator(),
                new ComparisonCsvWriter(),
                new CandidateDumpWriter(objectMapper),
                new RerankEvaluator(),
                new RerankCsvWriter(),
                beans.getBeanProvider(RerankScorer.class),
                properties);

        runner.run(null);

        assertThat(comparison).isRegularFile();
        assertThat(tempDirectory.resolve("run/dense_candidates.jsonl")).isRegularFile();
        assertThat(tempDirectory.resolve("run/bm25_candidates.jsonl")).isRegularFile();
        assertThat(tempDirectory.resolve("run/hybrid_candidates.jsonl")).isRegularFile();
        Path rerankMetrics = tempDirectory.resolve("run/rerank_metrics.csv");
        assertThat(rerankMetrics).isRegularFile();
        assertThat(Files.readString(rerankMetrics))
                .contains("source,quality_pool,scored_pool")
                .contains("dense,3,3,all")
                .contains("bm25,3,3,all")
                .contains("hybrid,3,3,all");
    }

    private static final class FakeEmbeddingModel implements EmbeddingModel, AsymmetricEmbedding {

        @Override
        public List<float[]> embedDocuments(List<String> texts) {
            return texts.stream().map(FakeEmbeddingModel::vector).toList();
        }

        @Override
        public float[] embedQuery(String text) {
            return vector(text);
        }

        @Override
        public float[] embed(Document document) {
            return vector(document.getText());
        }

        @Override
        public EmbeddingResponse call(EmbeddingRequest request) {
            throw new UnsupportedOperationException();
        }

        private static float[] vector(String text) {
            if (text.contains("акциз")) {
                return new float[]{1, 0};
            }
            if (text.contains("пошлина")) {
                return new float[]{0, 1};
            }
            return new float[]{0.5f, 0.5f};
        }
    }
}
