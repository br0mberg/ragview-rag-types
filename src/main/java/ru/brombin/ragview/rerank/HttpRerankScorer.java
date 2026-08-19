package ru.brombin.ragview.rerank;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.http.client.JdkClientHttpRequestFactory;
import org.springframework.web.client.RestClient;

import java.io.IOException;
import java.net.http.HttpClient;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Duration;
import java.util.HexFormat;
import java.util.List;

public final class HttpRerankScorer implements AttestedRerankScorer {

    private final RestClient http;
    private final ObjectMapper objectMapper;
    private final String expectedModel;
    private final String expectedRevision;
    private final String expectedPrecision;
    private final String expectedDevice;
    private final Integer expectedBatchSize;
    private final Integer expectedMaxLength;
    private final RerankerAttestation attestation;

    public HttpRerankScorer(
            String baseUrl,
            String expectedModel,
            String expectedRevision,
            String expectedPrecision,
            String expectedDevice) {
        this(baseUrl, expectedModel, expectedRevision, expectedPrecision, expectedDevice, null, null);
    }

    public HttpRerankScorer(
            String baseUrl,
            String expectedModel,
            String expectedRevision,
            String expectedPrecision,
            String expectedDevice,
            Integer expectedBatchSize,
            Integer expectedMaxLength) {
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
        this.objectMapper = new ObjectMapper()
                .disable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES);
        this.expectedModel = expectedModel;
        this.expectedRevision = expectedRevision;
        this.expectedPrecision = expectedPrecision;
        this.expectedDevice = expectedDevice;
        this.expectedBatchSize = expectedBatchSize;
        this.expectedMaxLength = expectedMaxLength;
        this.attestation = validateHealth();
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

    @Override
    public RerankerAttestation attestation() {
        return attestation;
    }

    private RerankerAttestation validateHealth() {
        byte[] healthBytes = http.get().uri("/health").retrieve().body(byte[].class);
        HealthResponse health = readHealth(healthBytes);
        if (health == null
                || health.schemaVersion() != 1
                || health.reranker() == null
                || !expectedModel.equals(health.reranker().model())
                || !expectedRevision.equals(health.reranker().revision())
                || !expectedPrecision.equals(health.reranker().precision())
                || health.reranker().device() == null
                || !health.reranker().device().startsWith(expectedDevice)
                || health.reranker().batchSize() == null
                || health.reranker().maxLength() == null
                || health.reranker().batchSize() <= 0
                || health.reranker().maxLength() <= 0
                || expectedBatchSize != null && !expectedBatchSize.equals(health.reranker().batchSize())
                || expectedMaxLength != null && !expectedMaxLength.equals(health.reranker().maxLength())) {
            throw new IllegalStateException("Rerank server does not match the configured model");
        }
        return new RerankerAttestation(
                health.schemaVersion(),
                health.reranker().model(),
                health.reranker().revision(),
                health.reranker().precision(),
                health.reranker().device(),
                health.reranker().batchSize(),
                health.reranker().maxLength(),
                sha256(healthBytes));
    }

    private HealthResponse readHealth(byte[] healthBytes) {
        if (healthBytes == null || healthBytes.length == 0) {
            throw new IllegalStateException("Rerank server does not match the configured model");
        }
        try {
            return objectMapper.readValue(healthBytes, HealthResponse.class);
        } catch (IOException error) {
            throw new IllegalStateException("Rerank server does not match the configured model", error);
        }
    }

    private static String sha256(byte[] value) {
        try {
            return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(value));
        } catch (NoSuchAlgorithmException error) {
            throw new IllegalStateException("SHA-256 is not available", error);
        }
    }

    private record RerankRequest(String query, List<String> passages) {
    }

    private record RerankResponse(List<Double> scores, String model, String revision) {
    }

    private record HealthResponse(int schemaVersion, RerankerHealth reranker) {
    }

    private record RerankerHealth(
            String model,
            String revision,
            String precision,
            String device,
            Integer batchSize,
            Integer maxLength) {
    }
}
