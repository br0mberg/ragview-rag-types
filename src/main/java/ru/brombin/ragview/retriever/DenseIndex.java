package ru.brombin.ragview.retriever;

import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.EmbeddingOptionsBuilder;
import org.springframework.ai.embedding.EmbeddingRequest;
import ru.brombin.ragview.corpus.CorpusDocument;
import ru.brombin.ragview.metrics.CallCounter;
import ru.brombin.ragview.strategy.RetrievedDoc;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

@RequiredArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE, makeFinal = true)
public class DenseIndex {

    EmbeddingModel embeddingModel;
    CallCounter callCounter;

    List<String> docIds = new ArrayList<>();
    List<float[]> vectors = new ArrayList<>();

    public void index(List<CorpusDocument> corpus) {
        if (corpus == null) {
            throw new NullPointerException("corpus");
        }
        if (corpus.isEmpty()) {
            throw new IllegalArgumentException("Corpus must not be empty");
        }
        docIds.clear();
        vectors.clear();

        AsymmetricEmbedding asym = asymmetric();
        final int batch = 64;
        for (int start = 0; start < corpus.size(); start += batch) {
            List<CorpusDocument> slice = corpus.subList(start, Math.min(start + batch, corpus.size()));
            List<String> texts = slice.stream().map(CorpusDocument::indexableText).toList();
            List<float[]> vecs = asym != null
                    ? asym.embedDocuments(texts)
                    : embeddingModel.call(new EmbeddingRequest(texts, EmbeddingOptionsBuilder.builder().build()))
                            .getResults().stream().map(e -> e.getOutput()).toList();
            validateBatch(vecs, slice.size(), vectors.isEmpty() ? null : vectors.getFirst().length);
            callCounter.countEmbedding(slice.size());
            for (int i = 0; i < slice.size(); i++) {
                docIds.add(slice.get(i).id());
                vectors.add(vecs.get(i));
            }
        }
    }

    private AsymmetricEmbedding asymmetric() {
        return embeddingModel instanceof AsymmetricEmbedding a ? a : null;
    }

    public List<RetrievedDoc> search(String query, int k) {
        if (k <= 0) {
            throw new IllegalArgumentException("k must be positive");
        }
        AsymmetricEmbedding asym = asymmetric();
        float[] q;
        if (asym != null) {
            callCounter.countEmbedding();
            q = asym.embedQuery(query);
        } else {
            q = embed(query);
        }
        validateVector(q, vectors.isEmpty() ? null : vectors.getFirst().length, "query vector");
        List<RetrievedDoc> scored = new ArrayList<>(docIds.size());
        for (int i = 0; i < docIds.size(); i++) {
            scored.add(new RetrievedDoc(docIds.get(i), cosine(q, vectors.get(i))));
        }
        scored.sort(Comparator.comparingDouble(RetrievedDoc::score).reversed()
                .thenComparing(RetrievedDoc::docId));
        return List.copyOf(scored.subList(0, Math.min(k, scored.size())));
    }

    private float[] embed(String text) {
        callCounter.countEmbedding();
        return embeddingModel.embed(text);
    }

    private static double cosine(float[] a, float[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Vector dimensions differ: " + a.length + " != " + b.length);
        }
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        double denom = Math.sqrt(na) * Math.sqrt(nb);
        return denom == 0 ? 0 : dot / denom;
    }

    private static void validateBatch(List<float[]> batch, int expectedSize, Integer expectedDimensions) {
        if (batch == null || batch.size() != expectedSize) {
            throw new IllegalArgumentException("Embedding batch size differs: expected "
                    + expectedSize + ", got " + (batch == null ? "null" : batch.size()));
        }
        Integer dimensions = expectedDimensions;
        for (float[] vector : batch) {
            validateVector(vector, dimensions, "document vector");
            if (dimensions == null) {
                dimensions = vector.length;
            }
        }
    }

    private static void validateVector(float[] vector, Integer expectedDimensions, String label) {
        if (vector == null || vector.length == 0) {
            throw new IllegalArgumentException(label + " must not be empty");
        }
        if (expectedDimensions != null && vector.length != expectedDimensions) {
            throw new IllegalArgumentException(label + " dimensions differ: expected "
                    + expectedDimensions + ", got " + vector.length);
        }
        for (float value : vector) {
            if (!Float.isFinite(value)) {
                throw new IllegalArgumentException(label + " contains a non-finite value");
            }
        }
    }
}
