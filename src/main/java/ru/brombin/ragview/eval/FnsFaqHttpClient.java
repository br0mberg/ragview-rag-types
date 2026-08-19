package ru.brombin.ragview.eval;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

public final class FnsFaqHttpClient {

    private static final String USER_AGENT = "ragview-fns-eval-collector/1.0 (+https://github.com/br0mberg/ragview-rag-types)";
    private static final String USER_AGENT_TOKEN = "ragview-fns-eval-collector";
    private static final int MAX_ATTEMPTS = 4;
    private static final int MAX_ROBOTS_BYTES = 512 * 1024;

    private final Transport transport;
    private final ObjectMapper objectMapper;
    private final Duration timeout;

    public FnsFaqHttpClient(HttpClient httpClient, ObjectMapper objectMapper, Duration timeout) {
        this((request) -> httpClient.send(request, HttpResponse.BodyHandlers.ofByteArray()),
                objectMapper, timeout);
    }

    FnsFaqHttpClient(Transport transport, ObjectMapper objectMapper, Duration timeout) {
        this.transport = transport;
        this.objectMapper = objectMapper;
        this.timeout = timeout;
    }

    public RawPage fetch(FnsFaqSourceConfig config, FnsFaqSourceConfig.Category category, int page)
            throws IOException, InterruptedException {
        HttpRequest request = buildRequest(config, category.id(), page, timeout);
        for (int attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
            HttpResponse<byte[]> response;
            try {
                response = transport.send(request);
            } catch (IOException networkFailure) {
                if (attempt == MAX_ATTEMPTS) {
                    throw new IOException("FNS request failed after " + MAX_ATTEMPTS
                            + " attempts for category=" + category.id() + ", page=" + page,
                            networkFailure);
                }
                Duration delay = FnsFaqRetryPolicy.delay(null, attempt, Instant.now());
                TimeUnit.MILLISECONDS.sleep(delay.toMillis());
                continue;
            }
            FnsFaqRetryPolicy.ResponseAction action = FnsFaqRetryPolicy.classify(response.statusCode());
            if (action == FnsFaqRetryPolicy.ResponseAction.SUCCESS) {
                return decodePage(response.body(), objectMapper, category.id(), page);
            }
            if (action == FnsFaqRetryPolicy.ResponseAction.STOP_FORBIDDEN) {
                throw new IOException("FNS returned HTTP 403; collector stopped without retry for category="
                        + category.id() + ", page=" + page);
            }
            if (action == FnsFaqRetryPolicy.ResponseAction.FAIL) {
                throw new IOException("FNS returned HTTP " + response.statusCode()
                        + " for category=" + category.id() + ", page=" + page);
            }
            if (attempt == MAX_ATTEMPTS) {
                throw new IOException("FNS returned HTTP " + response.statusCode()
                        + " after " + MAX_ATTEMPTS + " attempts for category="
                        + category.id() + ", page=" + page);
            }
            Duration delay = FnsFaqRetryPolicy.delay(
                    response.headers().firstValue("Retry-After").orElse(null), attempt, Instant.now());
            TimeUnit.MILLISECONDS.sleep(delay.toMillis());
        }
        throw new IllegalStateException("Retry loop ended unexpectedly");
    }

    public RobotsCheck verifyRobots(FnsFaqSourceConfig config) throws IOException, InterruptedException {
        URI robotsUri = URI.create(config.sourceUri().getScheme() + "://" + config.sourceUri().getAuthority()
                + "/robots.txt");
        HttpRequest request = HttpRequest.newBuilder(robotsUri)
                .timeout(timeout)
                .header("Accept", "text/plain")
                .header("User-Agent", USER_AGENT)
                .GET()
                .build();
        HttpResponse<byte[]> response = transport.send(request);
        if (response.statusCode() < 200 || response.statusCode() >= 300) {
            throw new IOException("FNS robots.txt returned HTTP " + response.statusCode()
                    + "; collector stopped before data requests");
        }
        byte[] bytes = response.body();
        if (bytes.length > MAX_ROBOTS_BYTES) {
            throw new IOException("FNS robots.txt exceeds 512 KiB");
        }
        String robots = new String(bytes, StandardCharsets.UTF_8);
        boolean sourceAllowed = FnsFaqRobotsPolicy.isAllowed(robots, USER_AGENT_TOKEN, config.sourceUri());
        boolean endpointAllowed = FnsFaqRobotsPolicy.isAllowed(robots, USER_AGENT_TOKEN, config.endpointUri());
        if (!sourceAllowed || !endpointAllowed) {
            throw new IOException("FNS robots.txt disallows collector paths: source="
                    + sourceAllowed + ", endpoint=" + endpointAllowed);
        }
        return new RobotsCheck(robotsUri.toString(), FnsFaqPageSampler.sha256(bytes), USER_AGENT_TOKEN);
    }

    private static RawPage decodePage(byte[] raw, ObjectMapper objectMapper, int categoryId, int page)
            throws IOException {
        FnsAjaxPayload payload = decode(raw, objectMapper);
        if (!payload.success()) {
            throw new IOException("FNS rejected category=" + categoryId + ", page=" + page
                    + ": " + payload.result());
        }
        if (payload.answer() == null) {
            throw new IOException("FNS returned no Answer for category=" + categoryId + ", page=" + page);
        }
        return new RawPage(page, raw, FnsFaqPageSampler.sha256(raw), payload.answer());
    }

    public static HttpRequest buildRequest(
            FnsFaqSourceConfig config,
            int categoryId,
            int page,
            Duration timeout) {
        if (categoryId <= 0 || page <= 0) {
            throw new IllegalArgumentException("categoryId and page must be positive");
        }
        String body = formBody(categoryId, config.regionId(), page);
        String separator = config.source().contains("?") ? "&" : "?";
        String referer = config.source() + separator + "t1=" + categoryId;
        return HttpRequest.newBuilder(config.endpointUri())
                .timeout(timeout)
                .header("Accept", "application/json, text/javascript, */*; q=0.01")
                .header("Content-Type", "application/x-www-form-urlencoded; charset=UTF-8")
                .header("Referer", referer)
                .header("User-Agent", USER_AGENT)
                .header("X-Requested-With", "XMLHttpRequest")
                .POST(HttpRequest.BodyPublishers.ofString(body, StandardCharsets.UTF_8))
                .build();
    }

    static FnsAjaxPayload decode(byte[] raw, ObjectMapper objectMapper) throws IOException {
        return objectMapper.readValue(raw, FnsAjaxPayload.class);
    }

    private static String formBody(int categoryId, int regionId, int page) {
        Map<String, String> fields = new LinkedHashMap<>();
        fields.put("type", "KbSearch");
        fields.put("t1", Integer.toString(categoryId));
        fields.put("t2", "");
        fields.put("t3", "");
        fields.put("r", Integer.toString(regionId));
        fields.put("cont", "");
        fields.put("ch", "");
        fields.put("pgid", Integer.toString(page));
        return fields.entrySet().stream()
                .map(entry -> encode(entry.getKey()) + "=" + encode(entry.getValue()))
                .collect(Collectors.joining("&"));
    }

    private static String encode(String value) {
        return URLEncoder.encode(value, StandardCharsets.UTF_8);
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    record FnsAjaxPayload(
            @JsonProperty("Success") boolean success,
            @JsonProperty("Result") String result,
            @JsonProperty("WebIndex") String webIndex,
            @JsonProperty("Answer") String answer) {
    }

    public record RawPage(int page, byte[] bytes, String sha256, String answerHtml) {
        public RawPage {
            bytes = bytes.clone();
        }

        @Override
        public byte[] bytes() {
            return bytes.clone();
        }
    }

    public record RobotsCheck(String url, String sha256, String userAgent) {
    }

    @FunctionalInterface
    interface Transport {
        HttpResponse<byte[]> send(HttpRequest request) throws IOException, InterruptedException;
    }
}
