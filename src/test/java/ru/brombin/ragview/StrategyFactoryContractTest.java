package ru.brombin.ragview;

import org.junit.jupiter.api.Test;
import ru.brombin.ragview.config.RagProperties;
import ru.brombin.ragview.corpus.CorpusDocument;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class StrategyFactoryContractTest {

    private static final List<CorpusDocument> CORPUS = List.of(
            new CorpusDocument("doc-1", "Статья 1", "налоговый кодекс"));

    @Test
    void build_shouldRunBm25WithoutEmbeddingModel() {
        StrategyFactory factory = new StrategyFactory(null, properties(List.of("bm25")));

        List<StrategySetup> strategies = factory.build(CORPUS);

        assertThat(strategies).singleElement()
                .extracting(setup -> setup.strategy().name())
                .isEqualTo("bm25");
    }

    @Test
    void build_shouldExplainMissingModelForDenseStrategy() {
        StrategyFactory factory = new StrategyFactory(null, properties(List.of("dense")));

        assertThatThrownBy(() -> factory.build(CORPUS))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("berta or frida Spring profile");
    }

    private static RagProperties properties(List<String> strategies) {
        return new RagProperties(
                1, 1, 1, 60, false, List.of(1), 0,
                strategies, "unused", "results/test/comparison.csv");
    }
}
