package ru.brombin.ragview.qdrant;

import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class QdrantV3HybridParityPoolContractTest {

    @Test
    void whitespaceTokenCount_shouldUseUnicodeWhitespace() {
        assertThat(QdrantV3HybridParityPool.whitespaceTokenCount("  один\tдва\nтри  "))
                .isEqualTo(3);
        assertThat(QdrantV3HybridParityPool.whitespaceTokenCount(" \u2003\u2009 "))
                .isZero();
        assertThat(QdrantV3HybridParityPool.AVERAGE_LENGTH_METHOD)
                .contains("UNICODE_CHARACTER_CLASS");
    }

    @Test
    void pointId_shouldBindDocumentToCorpus() {
        UUID first = QdrantV3HybridParityPool.pointId("a".repeat(64), "doc-1");
        UUID repeated = QdrantV3HybridParityPool.pointId("a".repeat(64), "doc-1");
        UUID otherCorpus = QdrantV3HybridParityPool.pointId("b".repeat(64), "doc-1");

        assertThat(first).isEqualTo(repeated).isNotEqualTo(otherCorpus);
    }

    @Test
    void storedIdentity_shouldRejectIncompleteCollection() {
        UUID pointId = QdrantV3HybridParityPool.pointId("a".repeat(64), "doc-1");
        Map<UUID, String> expected = Map.of(pointId, "doc-1");

        assertThatThrownBy(() -> QdrantV3HybridParityPool.validateStoredIdentity(
                expected,
                Map.of(),
                "a".repeat(64),
                "b".repeat(64)))
                .isInstanceOf(IllegalStateException.class)
                .hasMessage("Qdrant collection identity mismatch: retrieved 0 points, expected 1");
    }

    @Test
    void storedIdentity_shouldRejectForeignPayload() {
        UUID pointId = QdrantV3HybridParityPool.pointId("a".repeat(64), "doc-1");
        Map<UUID, String> expected = Map.of(pointId, "doc-1");
        Map<UUID, QdrantV3HybridParityPool.StoredIdentity> stored = Map.of(
                pointId,
                new QdrantV3HybridParityPool.StoredIdentity(
                        "doc-1", "a".repeat(64), "c".repeat(64)));

        assertThatThrownBy(() -> QdrantV3HybridParityPool.validateStoredIdentity(
                expected,
                stored,
                "a".repeat(64),
                "b".repeat(64)))
                .isInstanceOf(IllegalStateException.class)
                .hasMessage("Qdrant collection identity mismatch at document doc-1");
    }
}
