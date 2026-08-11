package ru.brombin.ragview;

import org.junit.jupiter.api.Test;
import ru.brombin.ragview.corpus.CorpusDocument;
import ru.brombin.ragview.retriever.Bm25Index;
import ru.brombin.ragview.strategy.RetrievedDoc;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class Bm25IndexContractTest {

    @Test
    void index_shouldReplacePreviousCorpus() {
        Bm25Index index = new Bm25Index();
        index.index(List.of(new CorpusDocument("old", "", "alpha")));

        index.index(List.of(new CorpusDocument("new", "", "beta")));

        assertThat(index.search("alpha", 10)).isEmpty();
        assertThat(index.search("beta", 10)).extracting(RetrievedDoc::docId)
                .containsExactly("new");
    }

    @Test
    void search_shouldRejectNonPositiveK() {
        Bm25Index index = new Bm25Index();

        assertThatThrownBy(() -> index.search("query", 0))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessage("k must be positive");
    }

    @Test
    void search_shouldMatchRussianInflections() {
        Bm25Index index = new Bm25Index();
        index.index(List.of(
                new CorpusDocument("target", "", "Перечень утверждается налоговым органом"),
                new CorpusDocument("noise", "", "Декларация подается налогоплательщиком")));

        assertThat(index.search("Кто утверждает перечень?", 10))
                .extracting(RetrievedDoc::docId)
                .containsExactly("target");
    }
}
