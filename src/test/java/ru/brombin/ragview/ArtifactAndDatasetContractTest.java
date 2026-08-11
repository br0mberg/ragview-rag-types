package ru.brombin.ragview;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import ru.brombin.ragview.artifact.CandidateDumpWriter;
import ru.brombin.ragview.corpus.CorpusDocument;
import ru.brombin.ragview.corpus.DatasetValidator;
import ru.brombin.ragview.corpus.EvalQuestion;
import ru.brombin.ragview.strategy.RagStrategy;
import ru.brombin.ragview.strategy.RetrievedDoc;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class ArtifactAndDatasetContractTest {

    @TempDir
    Path tempDirectory;

    @Test
    void candidateDump_shouldStayValidJsonUnderRussianLocale() throws Exception {
        Locale previous = Locale.getDefault();
        Locale.setDefault(Locale.forLanguageTag("ru-RU"));
        try {
            RagStrategy strategy = fixedStrategy(List.of(new RetrievedDoc("doc-\"1", 1.25)));
            EvalQuestion question = new EvalQuestion(
                    "q1", "строка \"один\"\nстрока два", "doc-\"1", List.of("doc-\"1"), "exact");
            Path output = tempDirectory.resolve("candidates.jsonl");

            new CandidateDumpWriter(new ObjectMapper())
                    .write(output, strategy, List.of(question), 10);

            JsonNode record = new ObjectMapper().readTree(Files.readString(output).strip());
            assertThat(record.get("q").textValue()).isEqualTo(question.question());
            assertThat(record.at("/cand/0/score").doubleValue()).isEqualTo(1.25);
            assertThat(record.at("/cand/0/docId").textValue()).isEqualTo("doc-\"1");
        } finally {
            Locale.setDefault(previous);
        }
    }

    @Test
    void datasetValidator_shouldRejectUnknownRelevantDocument() {
        List<CorpusDocument> corpus = List.of(new CorpusDocument("known", "", "текст"));
        List<EvalQuestion> questions = List.of(
                new EvalQuestion("q1", "вопрос", "missing", List.of("known", "missing"), "semantic"));

        assertThatThrownBy(() -> DatasetValidator.validate(corpus, questions))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessage("Question has invalid relevance: q1");
    }

    @Test
    void datasetValidator_shouldRejectBlankQuestion() {
        List<CorpusDocument> corpus = List.of(new CorpusDocument("known", "", "текст"));
        List<EvalQuestion> questions = List.of(
                new EvalQuestion("q1", " ", "known", List.of("known"), "semantic"));

        assertThatThrownBy(() -> DatasetValidator.validate(corpus, questions))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessage("Blank question: q1");
    }

    private static RagStrategy fixedStrategy(List<RetrievedDoc> result) {
        return new RagStrategy() {
            @Override
            public String name() {
                return "fixed";
            }

            @Override
            public void index(List<CorpusDocument> corpus) {
            }

            @Override
            public List<RetrievedDoc> retrieve(String query, int k) {
                return result.stream().limit(k).toList();
            }
        };
    }
}
