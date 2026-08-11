package ru.brombin.ragview;

import org.junit.jupiter.api.Test;
import ru.brombin.ragview.corpus.CorpusDocument;
import ru.brombin.ragview.corpus.EvalQuestion;
import ru.brombin.ragview.rerank.RerankEvaluation;
import ru.brombin.ragview.rerank.RerankEvaluator;
import ru.brombin.ragview.rerank.RerankMetrics;
import ru.brombin.ragview.strategy.RagStrategy;
import ru.brombin.ragview.strategy.RetrievedDoc;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class RerankEvaluatorContractTest {

    @Test
    void reranker_shouldPromoteRelevantCandidateOnlyInsidePool() {
        Map<String, CorpusDocument> corpus = corpus("d1", "d2", "relevant");
        RagStrategy source = fixedStrategy("dense", "d1", "d2", "relevant");
        EvalQuestion question = question("relevant");

        RerankEvaluation evaluation = new RerankEvaluator().evaluate(
                source,
                List.of(question),
                corpus,
                List.of(2, 3),
                2,
                (query, passages) -> List.of(0.2, 0.1, 5.0));

        assertThat(evaluation.queries()).hasSize(2);
        assertThat(evaluation.queries().getFirst().candidateRank()).isZero();
        assertThat(evaluation.queries().getFirst().rerankRank()).isZero();
        assertThat(evaluation.queries().getLast().candidateRank()).isEqualTo(3);
        assertThat(evaluation.queries().getLast().rerankRank()).isEqualTo(1);

        RerankMetrics pool3 = evaluation.metrics().stream()
                .filter(metric -> metric.pool() == 3 && metric.slice().equals("all"))
                .findFirst().orElseThrow();
        assertThat(pool3.candidateHitRate()).isEqualTo(1);
        assertThat(pool3.hitRate()).isEqualTo(1);
        assertThat(pool3.scoredPool()).isEqualTo(3);
    }

    @Test
    void reranker_shouldKeepFirstStageOrderWhenScoresTie() {
        Map<String, CorpusDocument> corpus = corpus("z-first", "a-relevant");
        RagStrategy source = fixedStrategy("bm25", "z-first", "a-relevant");

        RerankEvaluation evaluation = new RerankEvaluator().evaluate(
                source,
                List.of(question("a-relevant")),
                corpus,
                List.of(2),
                2,
                (query, passages) -> List.of(1.0, 1.0));

        assertThat(evaluation.queries().getFirst().rerankRank()).isEqualTo(2);
    }

    @Test
    void reranker_shouldRejectWrongScoreCount() {
        Map<String, CorpusDocument> corpus = corpus("d1", "relevant");
        RagStrategy source = fixedStrategy("hybrid", "d1", "relevant");

        assertThatThrownBy(() -> new RerankEvaluator().evaluate(
                source,
                List.of(question("relevant")),
                corpus,
                List.of(2),
                1,
                (query, passages) -> List.of(1.0)))
                .isInstanceOf(IllegalStateException.class)
                .hasMessage("Reranker score count differs from candidate count");
    }

    private static EvalQuestion question(String relevantId) {
        return new EvalQuestion("q1", "вопрос", relevantId, List.of(relevantId), "semantic");
    }

    private static Map<String, CorpusDocument> corpus(String... ids) {
        Map<String, CorpusDocument> result = new LinkedHashMap<>();
        for (String id : ids) {
            result.put(id, new CorpusDocument(id, id, "текст " + id));
        }
        return Map.copyOf(result);
    }

    private static RagStrategy fixedStrategy(String name, String... ids) {
        List<RetrievedDoc> hits = java.util.stream.IntStream.range(0, ids.length)
                .mapToObj(index -> new RetrievedDoc(ids[index], ids.length - index))
                .toList();
        return new RagStrategy() {
            @Override
            public String name() {
                return name;
            }

            @Override
            public void index(List<CorpusDocument> corpus) {
            }

            @Override
            public List<RetrievedDoc> retrieve(String query, int k) {
                return hits.subList(0, Math.min(k, hits.size()));
            }
        };
    }
}
