package ru.brombin.ragview;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import ru.brombin.ragview.rerank.HttpRerankScorer;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
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
        startServer(
                health("model-a", "rev-a"),
                "{\"scores\":[-1.5,2.25],\"model\":\"model-a\",\"revision\":\"rev-a\"}");
        HttpRerankScorer scorer = new HttpRerankScorer(
                baseUrl(), "model-a", "rev-a", "fp16", "cuda");

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
        return """
                {"schemaVersion":1,"reranker":{"model":"%s","revision":"%s",
                 "precision":"%s","device":"%s"}}
                """.formatted(model, revision, precision, device);
    }

    private static void respond(HttpExchange exchange, String body) throws IOException {
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().add("Content-Type", "application/json");
        exchange.sendResponseHeaders(200, bytes.length);
        exchange.getResponseBody().write(bytes);
        exchange.close();
    }
}
