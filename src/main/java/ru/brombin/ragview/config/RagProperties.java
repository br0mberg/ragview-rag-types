package ru.brombin.ragview.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

import java.util.List;

@ConfigurationProperties(prefix = "ragview")
public record RagProperties(
        int topK,
        int candidatePool,
        int candidateDumpLimit,
        int rrfK,
        boolean rerankEnabled,
        List<Integer> rerankPools,
        int warmupQueries,
        List<String> strategies,
        String datasetPath,
        String outputPath) {

    public RagProperties {
        if (topK <= 0) {
            throw new IllegalArgumentException("ragview.top-k must be positive");
        }
        if (candidatePool < topK) {
            throw new IllegalArgumentException("ragview.candidate-pool must be at least top-k");
        }
        if (candidateDumpLimit < candidatePool) {
            throw new IllegalArgumentException("ragview.candidate-dump-limit must be at least candidate-pool");
        }
        if (rrfK <= 0) {
            throw new IllegalArgumentException("ragview.rrf-k must be positive");
        }
        if (rerankPools == null || rerankPools.isEmpty()) {
            throw new IllegalArgumentException("ragview.rerank-pools must not be empty");
        }
        rerankPools = rerankPools.stream().distinct().sorted().toList();
        if (rerankPools.stream().anyMatch(pool -> pool < topK || pool > candidateDumpLimit)) {
            throw new IllegalArgumentException(
                    "ragview.rerank-pools must be between top-k and candidate-dump-limit");
        }
        if (warmupQueries < 0) {
            throw new IllegalArgumentException("ragview.warmup-queries must not be negative");
        }
        if (datasetPath == null || datasetPath.isBlank()) {
            throw new IllegalArgumentException("ragview.dataset-path must not be blank");
        }
        if (strategies == null || strategies.isEmpty()) {
            throw new IllegalArgumentException("ragview.strategies must not be empty");
        } else {
            strategies = strategies.stream().map(String::strip).map(String::toLowerCase).distinct().toList();
            if (strategies.stream().anyMatch(String::isBlank)) {
                throw new IllegalArgumentException("ragview.strategies must not contain blanks");
            }
        }
        if (outputPath == null || outputPath.isBlank()) {
            throw new IllegalArgumentException("ragview.output-path must not be blank");
        }
    }
}
