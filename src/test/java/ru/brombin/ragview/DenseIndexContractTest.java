package ru.brombin.ragview;

import org.junit.jupiter.api.Test;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.EmbeddingRequest;
import org.springframework.ai.embedding.EmbeddingResponse;
import ru.brombin.ragview.corpus.CorpusDocument;
import ru.brombin.ragview.metrics.CallCounter;
import ru.brombin.ragview.retriever.AsymmetricEmbedding;
import ru.brombin.ragview.retriever.DenseIndex;
import ru.brombin.ragview.strategy.RetrievedDoc;

import java.util.List;
import java.util.function.Function;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class DenseIndexContractTest {

    @Test
    void index_shouldReplacePreviousCorpus() {
        StubEmbeddingModel model = new StubEmbeddingModel(
                texts -> texts.stream().map(this::vectorFor).toList(),
                this::vectorFor);
        DenseIndex index = new DenseIndex(model, new CallCounter());
        index.index(List.of(new CorpusDocument("old", "", "alpha")));

        index.index(List.of(new CorpusDocument("new", "", "beta")));

        assertThat(index.search("beta", 10)).extracting(RetrievedDoc::docId)
                .containsExactly("new");
    }

    @Test
    void index_shouldRejectInconsistentDimensions() {
        StubEmbeddingModel model = new StubEmbeddingModel(
                texts -> List.of(new float[]{1, 0}, new float[]{1, 0, 0}),
                query -> new float[]{1, 0});
        DenseIndex index = new DenseIndex(model, new CallCounter());
        List<CorpusDocument> corpus = List.of(
                new CorpusDocument("a", "", "alpha"),
                new CorpusDocument("b", "", "beta"));

        assertThatThrownBy(() -> index.index(corpus))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessage("document vector dimensions differ: expected 2, got 3");
    }

    @Test
    void index_shouldRejectNonFiniteDocumentVector() {
        StubEmbeddingModel model = new StubEmbeddingModel(
                texts -> List.of(new float[]{Float.NaN, 0}),
                query -> new float[]{1, 0});
        DenseIndex index = new DenseIndex(model, new CallCounter());

        assertThatThrownBy(() -> index.index(List.of(new CorpusDocument("a", "", "alpha"))))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessage("document vector contains a non-finite value");
    }

    @Test
    void search_shouldRejectQueryWithWrongDimensions() {
        StubEmbeddingModel model = new StubEmbeddingModel(
                texts -> List.of(new float[]{1, 0}),
                query -> new float[]{1, 0, 0});
        DenseIndex index = new DenseIndex(model, new CallCounter());
        index.index(List.of(new CorpusDocument("a", "", "alpha")));

        assertThatThrownBy(() -> index.search("query", 1))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessage("query vector dimensions differ: expected 2, got 3");
    }

    @Test
    void search_shouldRejectNonFiniteQueryVector() {
        StubEmbeddingModel model = new StubEmbeddingModel(
                texts -> List.of(new float[]{1, 0}),
                query -> new float[]{Float.POSITIVE_INFINITY, 0});
        DenseIndex index = new DenseIndex(model, new CallCounter());
        index.index(List.of(new CorpusDocument("a", "", "alpha")));

        assertThatThrownBy(() -> index.search("query", 1))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessage("query vector contains a non-finite value");
    }

    @Test
    void search_shouldRejectNonPositiveK() {
        StubEmbeddingModel model = new StubEmbeddingModel(
                texts -> List.of(new float[]{1, 0}),
                query -> new float[]{1, 0});
        DenseIndex index = new DenseIndex(model, new CallCounter());

        assertThatThrownBy(() -> index.search("query", 0))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessage("k must be positive");
    }

    private float[] vectorFor(String text) {
        return text.contains("alpha") ? new float[]{1, 0} : new float[]{0, 1};
    }

    private static final class StubEmbeddingModel implements EmbeddingModel, AsymmetricEmbedding {

        private final Function<List<String>, List<float[]>> documents;
        private final Function<String, float[]> queries;

        private StubEmbeddingModel(
                Function<List<String>, List<float[]>> documents,
                Function<String, float[]> queries) {
            this.documents = documents;
            this.queries = queries;
        }

        @Override
        public List<float[]> embedDocuments(List<String> texts) {
            return documents.apply(texts);
        }

        @Override
        public float[] embedQuery(String text) {
            return queries.apply(text);
        }

        @Override
        public float[] embed(Document document) {
            return embedQuery(document.getText());
        }

        @Override
        public EmbeddingResponse call(EmbeddingRequest request) {
            throw new UnsupportedOperationException();
        }
    }
}
