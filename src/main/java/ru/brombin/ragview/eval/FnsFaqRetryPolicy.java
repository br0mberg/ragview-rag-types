package ru.brombin.ragview.eval;

import java.time.Duration;
import java.time.Instant;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;

public final class FnsFaqRetryPolicy {

    private static final Duration INITIAL_BACKOFF = Duration.ofSeconds(1);
    private static final Duration MAX_BACKOFF = Duration.ofSeconds(30);

    private FnsFaqRetryPolicy() {
    }

    public static ResponseAction classify(int statusCode) {
        if (statusCode >= 200 && statusCode < 300) {
            return ResponseAction.SUCCESS;
        }
        if (statusCode == 403) {
            return ResponseAction.STOP_FORBIDDEN;
        }
        if (statusCode == 429 || statusCode >= 500 && statusCode < 600) {
            return ResponseAction.RETRY;
        }
        return ResponseAction.FAIL;
    }

    public static Duration delay(String retryAfter, int failedAttempt, Instant now) {
        if (failedAttempt <= 0) {
            throw new IllegalArgumentException("failedAttempt must be positive");
        }
        Duration fallback = exponentialBackoff(failedAttempt);
        Duration requested = parseRetryAfter(retryAfter, now);
        Duration selected = requested.compareTo(fallback) > 0 ? requested : fallback;
        return selected.compareTo(MAX_BACKOFF) > 0 ? MAX_BACKOFF : selected;
    }

    private static Duration exponentialBackoff(int failedAttempt) {
        int shift = Math.min(failedAttempt - 1, 30);
        long seconds = Math.min(MAX_BACKOFF.toSeconds(), INITIAL_BACKOFF.toSeconds() << shift);
        return Duration.ofSeconds(seconds);
    }

    private static Duration parseRetryAfter(String value, Instant now) {
        if (value == null || value.isBlank()) {
            return Duration.ZERO;
        }
        String trimmed = value.trim();
        try {
            long seconds = Long.parseLong(trimmed);
            return seconds <= 0 ? Duration.ZERO : Duration.ofSeconds(seconds);
        } catch (NumberFormatException ignored) {
            try {
                Instant retryAt = ZonedDateTime.parse(trimmed, DateTimeFormatter.RFC_1123_DATE_TIME).toInstant();
                Duration delay = Duration.between(now, retryAt);
                return delay.isNegative() ? Duration.ZERO : delay;
            } catch (DateTimeParseException invalidDate) {
                return Duration.ZERO;
            }
        }
    }

    public enum ResponseAction {
        SUCCESS,
        RETRY,
        STOP_FORBIDDEN,
        FAIL
    }
}
