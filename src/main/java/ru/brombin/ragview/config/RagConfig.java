package ru.brombin.ragview.config;

import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Primary;
import org.springframework.context.annotation.Profile;
import ru.brombin.ragview.metrics.ComparisonCsvWriter;
import ru.brombin.ragview.metrics.RetrievalEvaluator;
import ru.brombin.ragview.artifact.CandidateDumpWriter;
import ru.brombin.ragview.rerank.HttpRerankScorer;
import ru.brombin.ragview.rerank.RerankCsvWriter;
import ru.brombin.ragview.rerank.RerankEvaluator;
import ru.brombin.ragview.rerank.RerankScorer;

import com.fasterxml.jackson.databind.ObjectMapper;

@Configuration
@EnableConfigurationProperties(RagProperties.class)
public class RagConfig {

    @Bean
    RetrievalEvaluator retrievalEvaluator() {
        return new RetrievalEvaluator();
    }

    @Bean
    ComparisonCsvWriter comparisonCsvWriter() {
        return new ComparisonCsvWriter();
    }

    @Bean
    CandidateDumpWriter candidateDumpWriter(ObjectMapper objectMapper) {
        return new CandidateDumpWriter(objectMapper);
    }

    @Bean
    RerankEvaluator rerankEvaluator() {
        return new RerankEvaluator();
    }

    @Bean
    RerankCsvWriter rerankCsvWriter() {
        return new RerankCsvWriter();
    }

    @Bean
    @ConditionalOnProperty(prefix = "ragview", name = "rerank-enabled", havingValue = "true")
    RerankScorer rerankScorer(
            @Value("${embedding-server.base-url:http://127.0.0.1:8077}") String baseUrl,
            @Value("${embedding-server.reranker-precision:fp16}") String precision,
            @Value("${embedding-server.expected-device:cuda}") String device) {
        return new HttpRerankScorer(
                baseUrl,
                "BAAI/bge-reranker-v2-m3",
                "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
                precision,
                device);
    }

    @Bean
    @Primary
    @Profile("berta")
    EmbeddingModel bertaEmbeddingModel(
            @Value("${embedding-server.base-url:http://127.0.0.1:8077}") String baseUrl,
            @Value("${embedding-server.expected-device:cuda}") String device) {
        return new PrefixEmbeddingModel(
                baseUrl,
                "sergeyzh/BERTA",
                "914c8c8aed14042ed890fc2c662d5e9e66b2faa7",
                "mean",
                768,
                device);
    }

    @Bean
    @Primary
    @Profile("frida")
    EmbeddingModel fridaEmbeddingModel(
            @Value("${embedding-server.base-url:http://127.0.0.1:8077}") String baseUrl,
            @Value("${embedding-server.expected-device:cuda}") String device) {
        return new PrefixEmbeddingModel(
                baseUrl,
                "ai-forever/FRIDA",
                "aed004da8d09c33f6a51240fd9f5bcc625225b67",
                "cls",
                1536,
                device);
    }
}
