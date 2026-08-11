package ru.brombin.ragview;

import org.junit.jupiter.api.Test;
import ru.brombin.ragview.corpus.CorpusDocument;
import ru.brombin.ragview.corpus.EvalQuestion;
import ru.brombin.ragview.metrics.CallCounter;
import ru.brombin.ragview.metrics.RetrievalEvaluator;
import ru.brombin.ragview.metrics.StrategyEvaluation;
import ru.brombin.ragview.metrics.StrategyResult;
import ru.brombin.ragview.retriever.Bm25Index;
import ru.brombin.ragview.retriever.ReciprocalRankFusion;
import ru.brombin.ragview.strategy.RagStrategy;
import ru.brombin.ragview.strategy.RetrievedDoc;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

class RetrievalComponentsTest {

    private static final List<CorpusDocument> CORPUS = List.of(
            new CorpusDocument("d01", "Идемпотентность HTTP", "POST не идемпотентен, PUT идемпотентен на ретраях"),
            new CorpusDocument("d02", "Оптимистичная блокировка", "версия записи проверяется при обновлении"),
            new CorpusDocument("d03", "Пул соединений", "слишком большой пул вредит базе"));

    @Test
    void bm25_should_rankExactTermMatchFirst() {
        Bm25Index index = new Bm25Index();
        index.index(CORPUS);

        List<RetrievedDoc> hits = index.search("идемпотентен PUT на ретраях", 3);

        assertThat(hits).isNotEmpty();
        assertThat(hits.get(0).docId()).isEqualTo("d01");
    }

    @Test
    void rrf_should_rewardDocsRankedHighInBothLists() {
        List<RetrievedDoc> lexical = List.of(
                new RetrievedDoc("d02", 5.0), new RetrievedDoc("d01", 1.0));
        List<RetrievedDoc> semantic = List.of(
                new RetrievedDoc("d01", 0.9), new RetrievedDoc("d03", 0.8));

        List<RetrievedDoc> fused = ReciprocalRankFusion.fuse(3, lexical, semantic);

        assertThat(fused.get(0).docId()).isEqualTo("d01");
    }

    @Test
    void evaluator_should_computeRecallAndMrr() {
        RagStrategy perfect = new RagStrategy() {
            public String name() {
                return "perfect";
            }

            public void index(List<CorpusDocument> corpus) {
            }

            public List<RetrievedDoc> retrieve(String query, int k) {
                return List.of(new RetrievedDoc("d01", 1.0));
            }

        };
        List<EvalQuestion> questions = List.of(
                new EvalQuestion("q1", "?", "d01", List.of("d01"), "exact"));

        StrategyResult result = new RetrievalEvaluator()
                .evaluate(perfect, questions, 5, new CallCounter(), 0, 0, 0).getFirst();

        assertThat(result.hitRateAtK()).isEqualTo(1.0);
        assertThat(result.mrrAtK()).isEqualTo(1.0);
    }

    @Test
    void evaluator_should_returnPerQueryRanks() {
        RagStrategy perfect = new RagStrategy() {
            public String name() {
                return "perfect";
            }

            public void index(List<CorpusDocument> corpus) {
            }

            public List<RetrievedDoc> retrieve(String query, int k) {
                return List.of(new RetrievedDoc("d01", 1.0));
            }

        };
        List<EvalQuestion> questions = List.of(
                new EvalQuestion("q1", "?", "d01", List.of("d01"), "exact"));

        StrategyEvaluation evaluation = new RetrievalEvaluator()
                .evaluateDetailed(perfect, questions, 5, new CallCounter(), 0, 0, 0);

        assertThat(evaluation.queries()).hasSize(1);
        assertThat(evaluation.queries().getFirst().qid()).isEqualTo("q1");
        assertThat(evaluation.queries().getFirst().rank()).isEqualTo(1);
    }

    @Test
    void evaluator_should_excludeWarmupFromCallTotals() {
        CallCounter counter = new CallCounter();
        RagStrategy strategy = new RagStrategy() {
            public String name() {
                return "counted";
            }

            public void index(List<CorpusDocument> corpus) {
            }

            public List<RetrievedDoc> retrieve(String query, int k) {
                counter.countEmbedding();
                return List.of(new RetrievedDoc("d01", 1.0));
            }

        };
        strategy.retrieve("warmup", 5);
        List<EvalQuestion> questions = List.of(
                new EvalQuestion("q1", "?", "d01", List.of("d01"), "exact"));

        StrategyResult result = new RetrievalEvaluator()
                .evaluate(strategy, questions, 5, counter, 10, 100, 0).getFirst();

        assertThat(result.embeddingRequestsTotal()).isEqualTo(11);
        assertThat(result.embeddingInputsTotal()).isEqualTo(101);
    }

    @Test
    void evaluator_should_scoreMissingResultAsZero() {
        RagStrategy empty = new RagStrategy() {
            public String name() {
                return "empty";
            }

            public void index(List<CorpusDocument> corpus) {
            }

            public List<RetrievedDoc> retrieve(String query, int k) {
                return List.of();
            }
        };
        List<EvalQuestion> questions = List.of(
                new EvalQuestion("q1", "?", "d01", List.of("d01"), "semantic"));

        StrategyResult result = new RetrievalEvaluator()
                .evaluate(empty, questions, 10, new CallCounter(), 0, 0, 0).getFirst();

        assertThat(result.hitRateAtK()).isZero();
        assertThat(result.mrrAtK()).isZero();
        assertThat(result.ndcgAtK()).isZero();
    }
}
