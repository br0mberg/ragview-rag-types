package ru.brombin.ragview.rerank;

import java.util.regex.Pattern;

public record RerankerAttestation(
        int healthSchemaVersion,
        String model,
        String revision,
        String precision,
        String device,
        int batchSize,
        int maxLength,
        String healthResponseSha256) {

    private static final Pattern SHA256 = Pattern.compile("[0-9a-f]{64}");

    public RerankerAttestation {
        requireText(model, "model");
        requireText(revision, "revision");
        requireText(precision, "precision");
        requireText(device, "device");
        if (healthSchemaVersion <= 0 || batchSize <= 0 || maxLength <= 0) {
            throw new IllegalArgumentException("Reranker attestation has invalid numeric fields");
        }
        if (healthResponseSha256 == null || !SHA256.matcher(healthResponseSha256).matches()) {
            throw new IllegalArgumentException("Reranker attestation has invalid health response SHA-256");
        }
    }

    private static void requireText(String value, String field) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException("Reranker attestation has blank " + field);
        }
    }
}
