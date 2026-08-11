package ru.brombin.ragview.rerank;

import org.springframework.http.client.JdkClientHttpRequestFactory;
import org.springframework.web.client.RestClient;

import java.net.http.HttpClient;
import java.time.Duration;
import java.util.List;

public final class HttpRerankScorer implements RerankScorer {

    private final RestClient http;
    private final String expectedModel;
    private final String expectedRevision;
    private final String expectedPrecision;
    private final String expectedDevice;

    public HttpRerankScorer(
            String baseUrl,
            String expectedModel,
            String expectedRevision,
            String expectedPrecision,
            String expectedDevice) {
        HttpClient client = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_1_1)
                .connectTimeout(Duration.ofSeconds(5))
                .build();
        JdkClientHttpRequestFactory requestFactory = new JdkClientHttpRequestFactory(client);
        requestFactory.setReadTimeout(Duration.ofMinutes(10));
        this.http = RestClient.builder()
                .requestFactory(requestFactory)
                .baseUrl(baseUrl)
                .build();
        this.expectedModel = expectedModel;
        this.expectedRevision = expectedRevision;
        this.expectedPrecision = expectedPrecision;
        this.expectedDevice = expectedDevice;
        validateHealth();
    }

    @Override
    public List<Double> score(String query, List<String> passages) {
        if (query == null || query.isBlank() || passages == null || passages.isEmpty()) {
            throw new IllegalArgumentException("Query and passages must not be empty");
        }
        RerankResponse response = http.post()
                .uri("/rerank")
                .body(new RerankRequest(query, passages))
                .retrieve()
                .body(RerankResponse.class);
        if (response == null
                || !expectedModel.equals(response.model())
                || !expectedRevision.equals(response.revision())
                || response.scores() == null
                || response.scores().size() != passages.size()
                || response.scores().stream().anyMatch(value -> value == null || !Double.isFinite(value))) {
            throw new IllegalStateException("Rerank server returned an invalid response");
        }
        return List.copyOf(response.scores());
    }

    private void validateHealth() {
        HealthResponse health = http.get().uri("/health").retrieve().body(HealthResponse.class);
        if (health == null
                || health.schemaVersion() != 1
                || health.reranker() == null
                || !expectedModel.equals(health.reranker().model())
                || !expectedRevision.equals(health.reranker().revision())
                || !expectedPrecision.equals(health.reranker().precision())
                || health.reranker().device() == null
                || !health.reranker().device().startsWith(expectedDevice)) {
            throw new IllegalStateException("Rerank server does not match the configured model");
        }
    }

    private record RerankRequest(String query, List<String> passages) {
    }

    private record RerankResponse(List<Double> scores, String model, String revision) {
    }

    private record HealthResponse(int schemaVersion, RerankerHealth reranker) {
    }

    private record RerankerHealth(String model, String revision, String precision, String device) {
    }
}
