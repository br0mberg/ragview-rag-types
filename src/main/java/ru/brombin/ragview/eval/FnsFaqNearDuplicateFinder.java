package ru.brombin.ragview.eval;

import java.util.ArrayList;
import java.util.List;

public final class FnsFaqNearDuplicateFinder {

    private FnsFaqNearDuplicateFinder() {
    }

    public static List<NearDuplicatePair> find(List<FnsFaqCandidate> candidates, double threshold) {
        if (threshold <= 0.0 || threshold > 1.0) {
            throw new IllegalArgumentException("threshold must be in (0, 1]");
        }
        List<NearDuplicatePair> result = new ArrayList<>();
        for (int left = 0; left < candidates.size(); left++) {
            String leftText = FnsFaqParser.normalizeQuestionKey(candidates.get(left).question());
            for (int right = left + 1; right < candidates.size(); right++) {
                String rightText = FnsFaqParser.normalizeQuestionKey(candidates.get(right).question());
                double similarity = similarity(leftText, rightText);
                if (similarity >= threshold) {
                    result.add(new NearDuplicatePair(
                            candidates.get(left).id(), candidates.get(right).id(), similarity));
                }
            }
        }
        return List.copyOf(result);
    }

    static double similarity(String left, String right) {
        if (left.equals(right)) {
            return 1.0;
        }
        int maximum = Math.max(left.length(), right.length());
        if (maximum == 0) {
            return 1.0;
        }
        return 1.0 - (double) levenshtein(left, right) / maximum;
    }

    private static int levenshtein(String left, String right) {
        if (left.length() > right.length()) {
            return levenshtein(right, left);
        }
        int[] previous = new int[left.length() + 1];
        int[] current = new int[left.length() + 1];
        for (int column = 0; column <= left.length(); column++) {
            previous[column] = column;
        }
        for (int row = 1; row <= right.length(); row++) {
            current[0] = row;
            for (int column = 1; column <= left.length(); column++) {
                int substitution = previous[column - 1]
                        + (left.charAt(column - 1) == right.charAt(row - 1) ? 0 : 1);
                current[column] = Math.min(
                        Math.min(current[column - 1] + 1, previous[column] + 1),
                        substitution);
            }
            int[] swap = previous;
            previous = current;
            current = swap;
        }
        return previous[left.length()];
    }

    public record NearDuplicatePair(String leftId, String rightId, double similarity) {
    }
}
