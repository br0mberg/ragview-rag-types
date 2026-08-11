package ru.brombin.ragview;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import ru.brombin.ragview.corpus.EvalQuestion;
import ru.brombin.ragview.strategy.ExplicitArticleReferenceRouter;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

class RoutingPolicyContractTest {

    private static final Path QUESTIONS = Path.of("data", "eval", "v2", "questions.json");
    private static final Path RETRIEVAL = Path.of(
            "results", "article", "retrieval.csv");
    private static final Path ROUTING_RECHECK = Path.of(
            "results", "article", "routing.csv");

    @Test
    void router_shouldMatchFrozenQuestionKinds() throws Exception {
        List<EvalQuestion> questions = questions();
        ExplicitArticleReferenceRouter router = new ExplicitArticleReferenceRouter();

        long matchedExact = questions.stream()
                .filter(question -> question.kind().equals("exact"))
                .filter(question -> router.matches(question.question()))
                .count();
        long matchedSemantic = questions.stream()
                .filter(question -> question.kind().equals("semantic"))
                .filter(question -> router.matches(question.question()))
                .count();

        assertThat(matchedExact).isEqualTo(120);
        assertThat(matchedSemantic).isZero();
    }

    @Test
    void bm25ThenHybridPolicy_shouldPreserveAllButOneHybridHit() throws Exception {
        List<EvalQuestion> questions = questions();
        Map<Key, Integer> ranks = ranks(RETRIEVAL);
        ExplicitArticleReferenceRouter router = new ExplicitArticleReferenceRouter();

        assertThat(hits(questions, ranks, router, "hybrid")).isEqualTo(228);
        assertThat(hits(questions, ranks, router, "dense")).isEqualTo(223);
    }

    @Test
    void fullRoutingRun_shouldMatchSelectedBranchForEveryQuestion() throws Exception {
        List<EvalQuestion> questions = questions();
        Map<Key, Integer> ranks = ranks(ROUTING_RECHECK);
        ExplicitArticleReferenceRouter router = new ExplicitArticleReferenceRouter();

        int mismatches = 0;
        for (EvalQuestion question : questions) {
            String selected = router.matches(question.question()) ? "bm25" : "hybrid";
            int expected = ranks.get(new Key(selected, question.id()));
            int actual = ranks.get(new Key("routing", question.id()));
            if (actual != expected) {
                mismatches++;
            }
        }

        assertThat(mismatches).isZero();
    }

    private static int hits(
            List<EvalQuestion> questions,
            Map<Key, Integer> ranks,
            ExplicitArticleReferenceRouter router,
            String fallback) {
        int hits = 0;
        for (EvalQuestion question : questions) {
            String strategy = router.matches(question.question()) ? "bm25" : fallback;
            int rank = ranks.get(new Key(strategy, question.id()));
            if (rank > 0 && rank <= 10) {
                hits++;
            }
        }
        return hits;
    }

    private static List<EvalQuestion> questions() throws Exception {
        EvalQuestion[] questions = new ObjectMapper().readValue(QUESTIONS.toFile(), EvalQuestion[].class);
        return Arrays.asList(questions);
    }

    private static Map<Key, Integer> ranks(Path path) throws Exception {
        Map<Key, Integer> ranks = new HashMap<>();
        List<String> lines = Files.readAllLines(path);
        for (String line : lines.subList(1, lines.size())) {
            String[] columns = line.split(",");
            ranks.put(new Key(columns[0], columns[1]), Integer.parseInt(columns[3]));
        }
        return ranks;
    }

    private record Key(String strategy, String qid) {
    }
}
