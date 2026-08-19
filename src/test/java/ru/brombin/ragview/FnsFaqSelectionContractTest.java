package ru.brombin.ragview;

import org.junit.jupiter.api.Test;
import ru.brombin.ragview.eval.FnsFaqCandidate;
import ru.brombin.ragview.eval.FnsFaqCandidateSampler;
import ru.brombin.ragview.eval.FnsFaqNearDuplicateFinder;
import ru.brombin.ragview.eval.FnsFaqPageSampler;
import ru.brombin.ragview.eval.FnsFaqSourceConfig;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class FnsFaqSelectionContractTest {

    @Test
    void pageSampler_shouldUseStableHashWithoutReplacement() {
        assertThat(FnsFaqPageSampler.select("ragview-tax-eval-v3-fns-2026-08-13", 911, 23, 8))
                .containsExactly(2, 3, 6, 9, 12, 16, 17, 23);
    }

    @Test
    void selection_shouldDeduplicateThenBalanceAcrossCategories() {
        List<FnsFaqCandidate> input = List.of(
                candidate("a1", "Как платить налог?", 1),
                candidate("a2", " как  платить налог? ", 1),
                candidate("a3", "Когда платить налог?", 1),
                candidate("b1", "Как получить вычет?", 2),
                candidate("b2", "Когда получить вычет?", 2),
                candidate("b3", "Кому положен вычет?", 2));

        var deduplicated = FnsFaqCandidateSampler.deduplicate(input);
        List<FnsFaqCandidate> selected = FnsFaqCandidateSampler.selectBalanced(
                deduplicated.candidates(),
                List.of(
                        new FnsFaqSourceConfig.Category(1, "one"),
                        new FnsFaqSourceConfig.Category(2, "two")),
                4,
                "seed");

        assertThat(deduplicated.removed()).isEqualTo(1);
        assertThat(selected).hasSize(4);
        assertThat(selected).filteredOn(candidate -> candidate.categoryId() == 1).hasSize(2);
        assertThat(selected).filteredOn(candidate -> candidate.categoryId() == 2).hasSize(2);
    }

    @Test
    void nearDuplicateFinder_shouldUseNormalizedLevenshtein() {
        List<FnsFaqCandidate> candidates = List.of(
                candidate("a", "Как получить налоговый вычет?", 1),
                candidate("b", "Как получить налоговые вычеты?", 1),
                candidate("c", "Когда платить НДС?", 1));

        assertThat(FnsFaqNearDuplicateFinder.find(candidates, 0.75))
                .singleElement()
                .satisfies(pair -> {
                    assertThat(pair.leftId()).isEqualTo("a");
                    assertThat(pair.rightId()).isEqualTo("b");
                });
    }

    @Test
    void selection_shouldFailClosedOnSameQuestionWithDifferentAnswers() {
        List<FnsFaqCandidate> input = List.of(
                candidate("a1", "Как платить налог?", "Ответ один", 1),
                candidate("a2", " как  платить налог? ", "Другой ответ", 1));

        assertThatThrownBy(() -> FnsFaqCandidateSampler.deduplicate(input))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("Conflicting answers", "a1", "a2");
    }

    private static FnsFaqCandidate candidate(String id, String question, int categoryId) {
        return candidate(id, question, "answer", categoryId);
    }

    private static FnsFaqCandidate candidate(
            String id,
            String question,
            String answer,
            int categoryId) {
        return new FnsFaqCandidate(
                id, question, answer, "source", List.of(), List.of(), false,
                categoryId, "category-" + categoryId, 1, "https://example.test", "0".repeat(64));
    }
}
