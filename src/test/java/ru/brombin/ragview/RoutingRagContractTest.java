package ru.brombin.ragview;

import org.junit.jupiter.api.Test;
import ru.brombin.ragview.corpus.CorpusDocument;
import ru.brombin.ragview.strategy.ExplicitArticleReferenceRouter;
import ru.brombin.ragview.strategy.RagStrategy;
import ru.brombin.ragview.strategy.RetrievedDoc;
import ru.brombin.ragview.strategy.RoutingRag;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

class RoutingRagContractTest {

    private static final List<CorpusDocument> CORPUS = List.of(
            new CorpusDocument(
                    "article-93.1", "Статья 93.1 НК РФ",
                    "Документы представляются в течение десяти дней"),
            new CorpusDocument(
                    "article-107", "Статья 107 НК РФ",
                    "Формы вины при совершении налогового правонарушения"));

    @Test
    void retrieve_shouldUseOnlyBm25_whenQueryContainsArticleReference() {
        RecordingStrategy bm25 = new RecordingStrategy();
        RecordingStrategy hybrid = new RecordingStrategy();
        RoutingRag routing = routing(bm25, hybrid);
        routing.index(CORPUS);

        routing.retrieve("По статье 93.1 НК РФ какой срок?", 1);

        assertThat(bm25.retrieveCalls()).isEqualTo(1);
        assertThat(hybrid.retrieveCalls()).isZero();
    }

    @Test
    void retrieve_shouldUseHybrid_whenQueryHasNoArticleReference() {
        RecordingStrategy bm25 = new RecordingStrategy();
        RecordingStrategy hybrid = new RecordingStrategy();
        RoutingRag routing = routing(bm25, hybrid);
        routing.index(CORPUS);

        routing.retrieve("В какой срок представить документы?", 1);

        assertThat(bm25.retrieveCalls()).isZero();
        assertThat(hybrid.retrieveCalls()).isEqualTo(1);
    }

    private static RoutingRag routing(RagStrategy bm25, RagStrategy hybrid) {
        return new RoutingRag(
                bm25, hybrid, new ExplicitArticleReferenceRouter());
    }

    private static final class RecordingStrategy implements RagStrategy {

        int retrieveCalls;

        @Override
        public String name() {
            return "recording";
        }

        @Override
        public void index(List<CorpusDocument> corpus) {
        }

        @Override
        public List<RetrievedDoc> retrieve(String query, int k) {
            retrieveCalls++;
            return List.of(new RetrievedDoc("article-93.1", 0.8));
        }

        private int retrieveCalls() {
            return retrieveCalls;
        }
    }
}
