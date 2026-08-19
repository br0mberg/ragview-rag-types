package ru.brombin.ragview.eval;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;

public final class CitationSilverPairedAnalysis {

    static final Path DEFAULT_INPUT = Path.of("results/v3-citation-silver/per-query.csv");
    static final String DEFAULT_SLICE = "preliminary_filtered";
    static final String LITERAL_CITATION_SLICE = "literal_citation_all";

    private CitationSilverPairedAnalysis() {
    }

    public static void main(String[] args) throws Exception {
        Path input = args.length == 0 ? DEFAULT_INPUT : Path.of(args[0]);
        String evaluationSlice = args.length < 2 ? DEFAULT_SLICE : args[1];
        System.out.println("comparison,k,rescue,harm,exact_two_sided_p");
        for (PairedResult result : analyze(input, evaluationSlice)) {
            System.out.printf(
                    Locale.ROOT,
                    "%s,%d,%d,%d,%.8f%n",
                    result.comparison(),
                    result.k(),
                    result.rescue(),
                    result.harm(),
                    result.exactTwoSidedP());
        }
    }

    public static List<PairedResult> analyze(Path input) throws IOException {
        return analyze(input, DEFAULT_SLICE);
    }

    public static List<PairedResult> analyze(Path input, String evaluationSlice) throws IOException {
        String inclusionColumn = switch (evaluationSlice) {
            case DEFAULT_SLICE -> "included";
            case LITERAL_CITATION_SLICE -> "literal_citation_slice_included";
            default -> throw new IllegalArgumentException("Unknown evaluation slice: " + evaluationSlice);
        };
        List<String> lines = Files.readAllLines(input);
        if (lines.size() < 2) {
            throw new IllegalArgumentException("Per-query CSV is empty");
        }
        List<String> header = List.of(lines.getFirst().split(",", -1));
        Map<String, Integer> columns = new HashMap<>();
        for (int index = 0; index < header.size(); index++) {
            columns.put(header.get(index), index);
        }
        for (String required : List.of(
                "source_id", "method", "qid", inclusionColumn, "k_chunks", "cited_article_hit")) {
            if (!columns.containsKey(required)) {
                throw new IllegalArgumentException("Missing CSV column: " + required);
            }
        }

        Map<ObservationKey, Boolean> observations = new HashMap<>();
        for (String line : lines.subList(1, lines.size())) {
            String[] values = line.split(",", -1);
            if (values.length != header.size()) {
                throw new IllegalArgumentException("Invalid CSV row width");
            }
            if (!parseBoolean(value(values, columns, inclusionColumn))) {
                continue;
            }
            ObservationKey key = new ObservationKey(
                    value(values, columns, "source_id"),
                    value(values, columns, "method"),
                    Integer.parseInt(value(values, columns, "k_chunks")),
                    value(values, columns, "qid"));
            Boolean hit = parseBoolean(value(values, columns, "cited_article_hit"));
            if (observations.putIfAbsent(key, hit) != null) {
                throw new IllegalArgumentException("Duplicate per-query observation: " + key);
            }
        }

        List<Comparison> comparisons = List.of(
                new Comparison("berta", "hybrid", "dense", 10),
                new Comparison("berta", "hybrid", "dense", 50),
                new Comparison("berta", "hybrid", "bm25", 50),
                new Comparison("qdrant-native-hybrid-rrf", "clientRrf", "dense", 10),
                new Comparison("qdrant-native-hybrid-rrf", "clientRrf", "dense", 50),
                new Comparison("qdrant-native-hybrid-rrf", "clientRrf", "nativeSparse", 50));
        List<PairedResult> results = new ArrayList<>(comparisons.size());
        for (Comparison comparison : comparisons) {
            results.add(compare(observations, comparison));
        }
        return List.copyOf(results);
    }

    private static PairedResult compare(
            Map<ObservationKey, Boolean> observations,
            Comparison comparison) {
        Map<String, Boolean> left = observationsFor(observations, comparison, comparison.leftMethod());
        Map<String, Boolean> right = observationsFor(observations, comparison, comparison.rightMethod());
        if (!left.keySet().equals(right.keySet()) || left.isEmpty()) {
            throw new IllegalArgumentException("Paired query sets differ: " + comparison.label());
        }
        int rescue = 0;
        int harm = 0;
        for (String qid : left.keySet()) {
            boolean leftHit = left.get(qid);
            boolean rightHit = right.get(qid);
            if (leftHit && !rightHit) {
                rescue++;
            } else if (!leftHit && rightHit) {
                harm++;
            }
        }
        return new PairedResult(
                comparison.label(),
                comparison.k(),
                rescue,
                harm,
                exactTwoSidedBinomial(rescue, harm));
    }

    private static Map<String, Boolean> observationsFor(
            Map<ObservationKey, Boolean> observations,
            Comparison comparison,
            String method) {
        Map<String, Boolean> result = new TreeMap<>();
        observations.forEach((key, hit) -> {
            if (key.sourceId().equals(comparison.sourceId())
                    && key.method().equals(method)
                    && key.k() == comparison.k()) {
                result.put(key.qid(), hit);
            }
        });
        return result;
    }

    static double exactTwoSidedBinomial(int rescue, int harm) {
        int discordant = rescue + harm;
        if (discordant == 0) {
            return 1.0;
        }
        int tail = Math.min(rescue, harm);
        double term = Math.scalb(1.0, -discordant);
        double cumulative = term;
        for (int index = 1; index <= tail; index++) {
            term *= (discordant - index + 1) / (double) index;
            cumulative += term;
        }
        double result = Math.min(1.0, 2.0 * cumulative);
        return Math.abs(result - 1.0) < 1e-15 ? 1.0 : result;
    }

    private static Boolean parseBoolean(String value) {
        if (!"true".equals(value) && !"false".equals(value)) {
            throw new IllegalArgumentException("Expected boolean value");
        }
        return Boolean.valueOf(value);
    }

    private static String value(String[] values, Map<String, Integer> columns, String name) {
        return values[columns.get(name)];
    }

    record ObservationKey(String sourceId, String method, int k, String qid) {
    }

    record Comparison(String sourceId, String leftMethod, String rightMethod, int k) {
        String label() {
            return sourceId + ":" + leftMethod + "_vs_" + rightMethod;
        }
    }

    public record PairedResult(
            String comparison,
            int k,
            int rescue,
            int harm,
            double exactTwoSidedP) {
    }
}
