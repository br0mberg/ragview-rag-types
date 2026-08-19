package ru.brombin.ragview.eval;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Comparator;
import java.util.List;
import java.util.stream.IntStream;

public final class FnsFaqPageSampler {

    private FnsFaqPageSampler() {
    }

    public static List<Integer> select(String seed, int categoryId, int totalPages, int limit) {
        if (seed == null || seed.isBlank()) {
            throw new IllegalArgumentException("seed must not be blank");
        }
        if (categoryId <= 0 || totalPages <= 0 || limit <= 0) {
            throw new IllegalArgumentException("categoryId, totalPages and limit must be positive");
        }
        Comparator<Integer> byStableHash = Comparator
                .comparing((Integer page) -> sha256(seed + ":" + categoryId + ":" + page))
                .thenComparingInt(Integer::intValue);
        return IntStream.rangeClosed(1, totalPages)
                .boxed()
                .sorted(byStableHash)
                .limit(Math.min(limit, totalPages))
                .sorted()
                .toList();
    }

    static String sha256(String value) {
        return sha256(value.getBytes(StandardCharsets.UTF_8));
    }

    static String sha256(byte[] value) {
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256")
                    .digest(value);
            return java.util.HexFormat.of().formatHex(digest);
        } catch (NoSuchAlgorithmException error) {
            throw new IllegalStateException("SHA-256 is unavailable", error);
        }
    }
}
