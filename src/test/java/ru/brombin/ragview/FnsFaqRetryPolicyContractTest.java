package ru.brombin.ragview;

import org.junit.jupiter.api.Test;
import ru.brombin.ragview.eval.FnsFaqRetryPolicy;

import java.time.Duration;
import java.time.Instant;

import static org.assertj.core.api.Assertions.assertThat;

class FnsFaqRetryPolicyContractTest {

    private static final Instant NOW = Instant.parse("2026-08-13T12:00:00Z");

    @Test
    void classify_shouldRetryOnlyRateLimitAndServerErrors() {
        assertThat(FnsFaqRetryPolicy.classify(200))
                .isEqualTo(FnsFaqRetryPolicy.ResponseAction.SUCCESS);
        assertThat(FnsFaqRetryPolicy.classify(429))
                .isEqualTo(FnsFaqRetryPolicy.ResponseAction.RETRY);
        assertThat(FnsFaqRetryPolicy.classify(500))
                .isEqualTo(FnsFaqRetryPolicy.ResponseAction.RETRY);
        assertThat(FnsFaqRetryPolicy.classify(503))
                .isEqualTo(FnsFaqRetryPolicy.ResponseAction.RETRY);
        assertThat(FnsFaqRetryPolicy.classify(403))
                .isEqualTo(FnsFaqRetryPolicy.ResponseAction.STOP_FORBIDDEN);
        assertThat(FnsFaqRetryPolicy.classify(400))
                .isEqualTo(FnsFaqRetryPolicy.ResponseAction.FAIL);
    }

    @Test
    void delay_shouldHonorRetryAfterSeconds() {
        assertThat(FnsFaqRetryPolicy.delay("12", 1, NOW)).isEqualTo(Duration.ofSeconds(12));
    }

    @Test
    void delay_shouldHonorRetryAfterHttpDate() {
        assertThat(FnsFaqRetryPolicy.delay("Thu, 13 Aug 2026 12:00:15 GMT", 1, NOW))
                .isEqualTo(Duration.ofSeconds(15));
    }

    @Test
    void delay_shouldUseBoundedExponentialBackoffForMissingOrInvalidHeader() {
        assertThat(FnsFaqRetryPolicy.delay(null, 1, NOW)).isEqualTo(Duration.ofSeconds(1));
        assertThat(FnsFaqRetryPolicy.delay("invalid", 2, NOW)).isEqualTo(Duration.ofSeconds(2));
        assertThat(FnsFaqRetryPolicy.delay(null, 10, NOW)).isEqualTo(Duration.ofSeconds(30));
        assertThat(FnsFaqRetryPolicy.delay("3600", 1, NOW)).isEqualTo(Duration.ofSeconds(30));
    }
}
