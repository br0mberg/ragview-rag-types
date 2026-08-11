package ru.brombin.ragview.config;

import lombok.AccessLevel;
import lombok.experimental.FieldDefaults;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.Embedding;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.EmbeddingRequest;
import org.springframework.ai.embedding.EmbeddingResponse;
import org.springframework.http.client.JdkClientHttpRequestFactory;
import org.springframework.web.client.RestClient;
import ru.brombin.ragview.retriever.AsymmetricEmbedding;

import java.net.http.HttpClient;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

@FieldDefaults(level = AccessLevel.PRIVATE, makeFinal = true)
public class PrefixEmbeddingModel implements EmbeddingModel, AsymmetricEmbedding {

    RestClient http;
    String expectedModel;
    String expectedRevision;
    int expectedDimensions;
    String expectedDevice;

    public PrefixEmbeddingModel(
            String baseUrl,
            String expectedModel,
            String expectedRevision,
            String expectedPooling,
            int expectedDimensions,
            String expectedDevice) {
        HttpClient client = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_1_1)
                .connectTimeout(Duration.ofSeconds(5))
                .build();
        JdkClientHttpRequestFactory requestFactory = new JdkClientHttpRequestFactory(client);
        requestFactory.setReadTimeout(Duration.ofMinutes(5));
        http = RestClient.builder()
                .requestFactory(requestFactory)
                .baseUrl(baseUrl)
                .build();
        this.expectedModel = expectedModel;
        this.expectedRevision = expectedRevision;
        this.expectedDimensions = expectedDimensions;
        this.expectedDevice = expectedDevice;
        validateHealth(expectedPooling);
    }

    @Override
    public List<float[]> embedDocuments(List<String> texts) {
        return encode(texts, "document");
    }

    @Override
    public float[] embedQuery(String text) {
        return encode(List.of(text), "query").getFirst();
    }

    @Override
    public float[] embed(Document document) {
        return encode(List.of(document.getText()), "document").getFirst();
    }

    @Override
    public float[] embed(String text) {
        return embedQuery(text);
    }

    @Override
    public EmbeddingResponse call(EmbeddingRequest request) {
        List<float[]> vectors = embedDocuments(request.getInstructions());
        List<Embedding> embeddings = new ArrayList<>(vectors.size());
        for (int index = 0; index < vectors.size(); index++) {
            embeddings.add(new Embedding(vectors.get(index), index));
        }
        return new EmbeddingResponse(embeddings);
    }

    private List<float[]> encode(List<String> texts, String kind) {
        EmbeddingResult result = http.post()
                .uri("/embeddings")
                .body(new EmbeddingPayload(texts, kind))
                .retrieve()
                .body(EmbeddingResult.class);
        if (result == null || result.vectors() == null || result.vectors().size() != texts.size()) {
            throw new IllegalStateException("Embedding server returned an invalid batch");
        }
        if (!expectedModel.equals(result.model()) || !expectedRevision.equals(result.revision())) {
            throw new IllegalStateException("Embedding server identity changed during the run");
        }
        List<float[]> vectors = new ArrayList<>(result.vectors().size());
        Integer dimensions = null;
        for (List<Double> raw : result.vectors()) {
            if (raw == null || raw.isEmpty()) {
                throw new IllegalStateException("Embedding server returned an empty vector");
            }
            if (dimensions == null) {
                dimensions = raw.size();
            } else if (dimensions != raw.size()) {
                throw new IllegalStateException("Embedding server returned inconsistent dimensions");
            }
            float[] vector = new float[raw.size()];
            for (int index = 0; index < raw.size(); index++) {
                Double value = raw.get(index);
                if (value == null || !Double.isFinite(value)) {
                    throw new IllegalStateException("Embedding server returned a non-finite value");
                }
                vector[index] = value.floatValue();
            }
            vectors.add(vector);
        }
        if (dimensions == null || dimensions != expectedDimensions) {
            throw new IllegalStateException("Embedding server returned unexpected dimensions");
        }
        return vectors;
    }

    private void validateHealth(String expectedPooling) {
        HealthResult health = http.get().uri("/health").retrieve().body(HealthResult.class);
        if (health == null
                || health.schemaVersion() != 1
                || !expectedModel.equals(health.model())
                || !expectedRevision.equals(health.revision())
                || !expectedPooling.equals(health.pooling())
                || expectedDimensions != health.dimensions()
                || !"search_query: ".equals(health.queryPrefix())
                || !"search_document: ".equals(health.documentPrefix())
                || health.device() == null
                || !health.device().startsWith(expectedDevice)) {
            throw new IllegalStateException("Embedding server does not match the configured model");
        }
    }

    private record EmbeddingPayload(List<String> texts, String kind) {
    }

    private record EmbeddingResult(List<List<Double>> vectors, String model, String revision) {
    }

    private record HealthResult(
            int schemaVersion,
            String model,
            String revision,
            String pooling,
            int dimensions,
            String queryPrefix,
            String documentPrefix,
            String device) {
    }
}
