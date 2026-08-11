package ru.brombin.ragview;

import org.junit.jupiter.api.Test;
import ru.brombin.ragview.retriever.ReciprocalRankFusion;
import ru.brombin.ragview.strategy.RetrievedDoc;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class ReciprocalRankFusionContractTest {

    @Test
    void fuse_shouldUseDefaultK60() {
        List<RetrievedDoc> bm25 = List.of(new RetrievedDoc("a", 10));

        List<RetrievedDoc> implicit = ReciprocalRankFusion.fuse(1, bm25);
        List<RetrievedDoc> explicit = ReciprocalRankFusion.fuse(1, 60, bm25);

        assertThat(ReciprocalRankFusion.DEFAULT_RRF_K).isEqualTo(60);
        assertThat(implicit).isEqualTo(explicit);
        assertThat(implicit.getFirst().score()).isEqualTo(1.0 / 61);
    }

    @Test
    void fuse_shouldBreakEqualScoresByDocId() {
        List<RetrievedDoc> bm25 = List.of(
                new RetrievedDoc("bm25-r1", 9),
                new RetrievedDoc("bm25-r2", 8));
        List<RetrievedDoc> dense = List.of(
                new RetrievedDoc("dense-r1", 0.9),
                new RetrievedDoc("dense-r2", 0.8));

        List<RetrievedDoc> result = ReciprocalRankFusion.fuse(4, 60, bm25, dense);

        assertThat(result).extracting(RetrievedDoc::docId)
                .containsExactly("bm25-r1", "dense-r1", "bm25-r2", "dense-r2");
    }

    @Test
    void fuse_shouldIgnoreBranchOrderOnTies() {
        List<RetrievedDoc> bm25 = List.of(new RetrievedDoc("bm25", 9));
        List<RetrievedDoc> dense = List.of(new RetrievedDoc("dense", 0.9));

        List<RetrievedDoc> result = ReciprocalRankFusion.fuse(2, 60, dense, bm25);

        assertThat(result).extracting(RetrievedDoc::docId)
                .containsExactly("bm25", "dense");
    }

    @Test
    void fuse_shouldPromoteDocumentSupportedByBothBranches() {
        List<RetrievedDoc> bm25 = List.of(
                new RetrievedDoc("lexical", 9),
                new RetrievedDoc("shared", 8));
        List<RetrievedDoc> dense = List.of(
                new RetrievedDoc("shared", 0.9),
                new RetrievedDoc("semantic", 0.8));

        List<RetrievedDoc> result = ReciprocalRankFusion.fuse(1, bm25, dense);

        assertThat(result.getFirst().docId()).isEqualTo("shared");
    }

    @Test
    void fuse_canDemoteSpecialistWinnerWhenDistractorHasCrossBranchSupport() {
        List<RetrievedDoc> bm25 = List.of(
                new RetrievedDoc("relevant", 9),
                new RetrievedDoc("distractor", 8));
        List<RetrievedDoc> dense = List.of(
                new RetrievedDoc("distractor", 0.9),
                new RetrievedDoc("other", 0.8));

        List<RetrievedDoc> result = ReciprocalRankFusion.fuse(1, bm25, dense);

        assertThat(result.getFirst().docId()).isEqualTo("distractor");
    }

    @Test
    void fuse_shouldRejectNonPositiveParameters() {
        List<RetrievedDoc> ranking = List.of(new RetrievedDoc("a", 1));

        assertThatThrownBy(() -> ReciprocalRankFusion.fuse(0, ranking))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessage("topK must be positive");
        assertThatThrownBy(() -> ReciprocalRankFusion.fuse(1, 0, ranking))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessage("rrfK must be positive");
    }
}
