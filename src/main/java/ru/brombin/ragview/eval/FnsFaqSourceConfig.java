package ru.brombin.ragview.eval;

import java.net.URI;
import java.time.LocalDate;
import java.util.List;
import java.util.regex.Pattern;

public record FnsFaqSourceConfig(
        int schemaVersion,
        String source,
        String endpoint,
        int regionId,
        String selectionSeed,
        int pagesPerCategory,
        long requestDelayMs,
        Selection selection,
        NearDuplicates nearDuplicates,
        Scope scope,
        List<Category> categories) {

    private static final Pattern SHA256 = Pattern.compile("[0-9a-f]{64}");

    public FnsFaqSourceConfig {
        if (schemaVersion != 1) {
            throw new IllegalArgumentException("Unsupported source config schema: " + schemaVersion);
        }
        URI sourceUri = requireHttps(source, "source");
        URI endpointUri = requireHttps(endpoint, "endpoint");
        if (!endpointUri.getPath().endsWith("/Ajax.html")) {
            throw new IllegalArgumentException("FNS endpoint must end with /Ajax.html");
        }
        if (sourceUri.equals(endpointUri)) {
            throw new IllegalArgumentException("Source page and endpoint must differ");
        }
        if (regionId <= 0) {
            throw new IllegalArgumentException("regionId must be positive");
        }
        if (selectionSeed == null || selectionSeed.isBlank()) {
            throw new IllegalArgumentException("selectionSeed must not be blank");
        }
        if (pagesPerCategory <= 0) {
            throw new IllegalArgumentException("pagesPerCategory must be positive");
        }
        if (requestDelayMs < 500) {
            throw new IllegalArgumentException("requestDelayMs must be at least 500");
        }
        if (categories == null || categories.isEmpty()) {
            throw new IllegalArgumentException("categories must not be empty");
        }
        if (selection == null || nearDuplicates == null || scope == null) {
            throw new IllegalArgumentException("selection, nearDuplicates and scope are required");
        }
        categories = List.copyOf(categories);
        if (categories.stream().map(Category::id).distinct().count() != categories.size()) {
            throw new IllegalArgumentException("Category IDs must be unique");
        }
    }

    public URI sourceUri() {
        return URI.create(source);
    }

    public URI endpointUri() {
        return URI.create(endpoint);
    }

    private static URI requireHttps(String value, String field) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(field + " must not be blank");
        }
        URI uri = URI.create(value);
        if (!"https".equalsIgnoreCase(uri.getScheme()) || uri.getHost() == null) {
            throw new IllegalArgumentException(field + " must be an HTTPS URL");
        }
        return uri;
    }

    public record Category(int id, String name) {
        public Category {
            if (id <= 0) {
                throw new IllegalArgumentException("Category ID must be positive");
            }
            if (name == null || name.isBlank()) {
                throw new IllegalArgumentException("Category name must not be blank");
            }
        }
    }

    public record Selection(
            String unit,
            int targetTotal,
            String mode,
            String pageSelection,
            String splitSeed) {

        public Selection {
            if (!"question".equals(unit)) {
                throw new IllegalArgumentException("selection.unit must be question");
            }
            if (targetTotal <= 0) {
                throw new IllegalArgumentException("selection.targetTotal must be positive");
            }
            if (!"balanced_by_category".equals(mode)) {
                throw new IllegalArgumentException("Unsupported selection.mode: " + mode);
            }
            if (!"sha256_rank_without_replacement".equals(pageSelection)) {
                throw new IllegalArgumentException("Unsupported selection.pageSelection: " + pageSelection);
            }
            if (splitSeed == null || splitSeed.isBlank()) {
                throw new IllegalArgumentException("selection.splitSeed must not be blank");
            }
        }
    }

    public record NearDuplicates(String algorithm, double threshold) {
        public NearDuplicates {
            if (!"normalized_levenshtein".equals(algorithm)) {
                throw new IllegalArgumentException("Unsupported near-duplicate algorithm: " + algorithm);
            }
            if (threshold <= 0.0 || threshold > 1.0) {
                throw new IllegalArgumentException("nearDuplicates.threshold must be in (0, 1]");
            }
        }
    }

    public record Scope(String level, String corpusDate, String corpusSha256) {
        public Scope {
            if (!"federal".equals(level)) {
                throw new IllegalArgumentException("scope.level must be federal");
            }
            LocalDate.parse(corpusDate);
            if (corpusSha256 == null || !SHA256.matcher(corpusSha256).matches()) {
                throw new IllegalArgumentException("scope.corpusSha256 must be lowercase SHA-256");
            }
        }
    }
}
