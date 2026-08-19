package ru.brombin.ragview.eval;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class FnsFaqCandidateSampler {

    private FnsFaqCandidateSampler() {
    }

    public static DeduplicationResult deduplicate(List<FnsFaqCandidate> candidates) {
        Map<String, FnsFaqCandidate> unique = new LinkedHashMap<>();
        for (FnsFaqCandidate candidate : candidates) {
            String questionKey = FnsFaqParser.normalizeQuestionKey(candidate.question());
            FnsFaqCandidate previous = unique.putIfAbsent(questionKey, candidate);
            if (previous != null && !sameAnswer(previous.answer(), candidate.answer())) {
                throw new IllegalArgumentException("Conflicting answers for normalized question: "
                        + previous.id() + " and " + candidate.id());
            }
        }
        return new DeduplicationResult(List.copyOf(unique.values()), candidates.size() - unique.size());
    }

    private static boolean sameAnswer(String left, String right) {
        return FnsFaqParser.normalizeQuestionKey(left)
                .equals(FnsFaqParser.normalizeQuestionKey(right));
    }

    public static List<FnsFaqCandidate> selectBalanced(
            List<FnsFaqCandidate> candidates,
            List<FnsFaqSourceConfig.Category> configuredCategories,
            int targetTotal,
            String seed) {
        if (targetTotal <= 0 || seed == null || seed.isBlank()) {
            throw new IllegalArgumentException("targetTotal must be positive and seed must not be blank");
        }
        if (candidates.size() < targetTotal) {
            throw new IllegalArgumentException("Only " + candidates.size()
                    + " unique questions are available, target is " + targetTotal);
        }

        Map<Integer, List<FnsFaqCandidate>> byCategory = new LinkedHashMap<>();
        for (FnsFaqSourceConfig.Category category : configuredCategories) {
            byCategory.put(category.id(), new ArrayList<>());
        }
        for (FnsFaqCandidate candidate : candidates) {
            List<FnsFaqCandidate> category = byCategory.get(candidate.categoryId());
            if (category == null) {
                throw new IllegalArgumentException("Candidate has unknown category: " + candidate.categoryId());
            }
            category.add(candidate);
        }
        byCategory.forEach((categoryId, values) -> values.sort(Comparator
                .comparing((FnsFaqCandidate candidate) -> FnsFaqPageSampler.sha256(
                        seed + ":question:" + categoryId + ":"
                                + FnsFaqParser.normalizeQuestionKey(candidate.question())))
                .thenComparing(FnsFaqCandidate::id)));

        List<Integer> categoryOrder = configuredCategories.stream()
                .map(FnsFaqSourceConfig.Category::id)
                .sorted(Comparator
                        .comparing((Integer id) -> FnsFaqPageSampler.sha256(seed + ":category:" + id))
                        .thenComparingInt(Integer::intValue))
                .toList();
        Map<Integer, Integer> offsets = new LinkedHashMap<>();
        categoryOrder.forEach(id -> offsets.put(id, 0));

        List<FnsFaqCandidate> selected = new ArrayList<>(targetTotal);
        while (selected.size() < targetTotal) {
            boolean added = false;
            for (int categoryId : categoryOrder) {
                int offset = offsets.get(categoryId);
                List<FnsFaqCandidate> values = byCategory.get(categoryId);
                if (offset < values.size()) {
                    selected.add(values.get(offset));
                    offsets.put(categoryId, offset + 1);
                    added = true;
                    if (selected.size() == targetTotal) {
                        break;
                    }
                }
            }
            if (!added) {
                throw new IllegalStateException("Balanced selection stopped before targetTotal");
            }
        }
        return List.copyOf(selected);
    }

    public record DeduplicationResult(List<FnsFaqCandidate> candidates, int removed) {
    }
}
