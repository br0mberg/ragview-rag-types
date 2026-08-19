package ru.brombin.ragview;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import ru.brombin.ragview.rerank.HttpRerankScorer;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class HttpRerankScorerContractTest {

    private HttpServer server;

    @AfterEach
    void stopServer() {
        if (server != null) {
            server.stop(0);
        }
    }

    @Test
    void client_shouldValidateIdentityAndReturnScores() throws Exception {
        String healthBody = combinedHealth("model-a", "rev-a");
        startServer(
                healthBody,
                "{\"scores\":[-1.5,2.25],\"model\":\"model-a\",\"revision\":\"rev-a\"}");
        HttpRerankScorer scorer = new HttpRerankScorer(
                baseUrl(), "model-a", "rev-a", "fp16", "cuda", 64, 512);

        assertThat(scorer.attestation().healthSchemaVersion()).isEqualTo(1);
        assertThat(scorer.attestation().model()).isEqualTo("model-a");
        assertThat(scorer.attestation().revision()).isEqualTo("rev-a");
        assertThat(scorer.attestation().precision()).isEqualTo("fp16");
        assertThat(scorer.attestation().device()).isEqualTo("cuda");
        assertThat(scorer.attestation().batchSize()).isEqualTo(64);
        assertThat(scorer.attestation().maxLength()).isEqualTo(512);
        assertThat(scorer.attestation().healthResponseSha256())
                .isEqualTo(sha256(healthBody.getBytes(StandardCharsets.UTF_8)));
        assertThat(scorer.score("вопрос", List.of("первый", "второй")))
                .containsExactly(-1.5, 2.25);
    }

    @Test
    void client_shouldRejectUnexpectedRerankerDuringHandshake() throws Exception {
        startServer(health("model-b", "rev-b"), "{}");

        assertThatThrownBy(() -> new HttpRerankScorer(
                baseUrl(), "model-a", "rev-a", "fp16", "cuda"))
                .isInstanceOf(IllegalStateException.class)
                .hasMessage("Rerank server does not match the configured model");
    }

    @Test
    void client_shouldRejectUnexpectedPrecisionOrDevice() throws Exception {
        startServer(health("model-a", "rev-a", "fp32", "cpu"), "{}");

        assertThatThrownBy(() -> new HttpRerankScorer(
                baseUrl(), "model-a", "rev-a", "fp16", "cuda"))
                .isInstanceOf(IllegalStateException.class)
                .hasMessage("Rerank server does not match the configured model");
    }

    @Test
    void client_shouldRejectUnexpectedBatchSizeOrMaxLength() throws Exception {
        startServer(health("model-a", "rev-a", "fp16", "cuda", 32, 256), "{}");

        assertThatThrownBy(() -> new HttpRerankScorer(
                baseUrl(), "model-a", "rev-a", "fp16", "cuda", 64, 512))
                .isInstanceOf(IllegalStateException.class)
                .hasMessage("Rerank server does not match the configured model");
    }

    private void startServer(String healthBody, String rerankBody) throws IOException {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/health", exchange -> respond(exchange, healthBody));
        server.createContext("/rerank", exchange -> respond(exchange, rerankBody));
        server.start();
    }

    private String baseUrl() {
        return "http://127.0.0.1:" + server.getAddress().getPort();
    }

    private static String health(String model, String revision) {
        return health(model, revision, "fp16", "cuda");
    }

    private static String health(String model, String revision, String precision, String device) {
        return health(model, revision, precision, device, 64, 512);
    }

    private static String health(
            String model,
            String revision,
            String precision,
            String device,
            int batchSize,
            int maxLength) {
        return """
                {"schemaVersion":1,"reranker":{"model":"%s","revision":"%s",
                 "precision":"%s","device":"%s","batchSize":%d,"maxLength":%d}}
                """.formatted(model, revision, precision, device, batchSize, maxLength);
    }

    private static String combinedHealth(String rerankerModel, String rerankerRevision) {
        return """
                {"schemaVersion":1,"status":"ok","model":"sergeyzh/BERTA",
                 "revision":"encoder-revision","pooling":"mean","dimensions":768,
                 "maxLength":512,"queryPrefix":"search_query: ",
                 "documentPrefix":"search_document: ","device":"cuda:0",
                 "reranker":{"model":"%s","revision":"%s","precision":"fp16",
                 "device":"cuda","batchSize":64,"maxLength":512}}
                """.formatted(rerankerModel, rerankerRevision);
    }

    private static void respond(HttpExchange exchange, String body) throws IOException {
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().add("Content-Type", "application/json");
        exchange.sendResponseHeaders(200, bytes.length);
        exchange.getResponseBody().write(bytes);
        exchange.close();
    }

    private static String sha256(byte[] value) throws Exception {
        return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(value));
    }
}
