package ru.brombin.ragview;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import ru.brombin.ragview.config.PrefixEmbeddingModel;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class PrefixEmbeddingModelContractTest {

    private HttpServer server;

    @AfterEach
    void stopServer() {
        if (server != null) {
            server.stop(0);
        }
    }

    @Test
    void client_shouldValidateIdentityAndReturnEmbedding() throws Exception {
        startServer(
                health("model-a", "rev-a", "mean", 2),
                "{\"vectors\":[[0.25,0.75]],\"model\":\"model-a\",\"revision\":\"rev-a\"}");

        PrefixEmbeddingModel model = new PrefixEmbeddingModel(
                baseUrl(), "model-a", "rev-a", "mean", 2, "cpu");

        assertThat(model.embedQuery("вопрос")).containsExactly(0.25f, 0.75f);
    }

    @Test
    void client_shouldRejectUnexpectedModelDuringHandshake() throws Exception {
        startServer(health("model-b", "rev-b", "mean", 2), "{}");

        assertThatThrownBy(() -> new PrefixEmbeddingModel(
                baseUrl(), "model-a", "rev-a", "mean", 2, "cpu"))
                .isInstanceOf(IllegalStateException.class)
                .hasMessage("Embedding server does not match the configured model");
    }

    @Test
    void client_shouldRejectUnexpectedDevice() throws Exception {
        startServer(health("model-a", "rev-a", "mean", 2, "cpu"), "{}");

        assertThatThrownBy(() -> new PrefixEmbeddingModel(
                baseUrl(), "model-a", "rev-a", "mean", 2, "cuda"))
                .isInstanceOf(IllegalStateException.class)
                .hasMessage("Embedding server does not match the configured model");
    }

    private void startServer(String healthBody, String embeddingsBody) throws IOException {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/health", exchange -> respond(exchange, healthBody));
        server.createContext("/embeddings", exchange -> respond(exchange, embeddingsBody));
        server.start();
    }

    private String baseUrl() {
        return "http://127.0.0.1:" + server.getAddress().getPort();
    }

    private static String health(String model, String revision, String pooling, int dimensions) {
        return health(model, revision, pooling, dimensions, "cpu");
    }

    private static String health(
            String model,
            String revision,
            String pooling,
            int dimensions,
            String device) {
        return """
                {"schemaVersion":1,"status":"ok","model":"%s","revision":"%s",
                 "pooling":"%s","dimensions":%d,"queryPrefix":"search_query: ",
                 "documentPrefix":"search_document: ","device":"%s"}
                """.formatted(model, revision, pooling, dimensions, device);
    }

    private static void respond(HttpExchange exchange, String body) throws IOException {
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().add("Content-Type", "application/json");
        exchange.sendResponseHeaders(200, bytes.length);
        exchange.getResponseBody().write(bytes);
        exchange.close();
    }
}
