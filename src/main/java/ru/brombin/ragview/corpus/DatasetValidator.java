package ru.brombin.ragview.corpus;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

public final class DatasetValidator {

    private DatasetValidator() {
    }

    public static void validate(List<CorpusDocument> corpus, List<EvalQuestion> questions) {
        if (corpus.isEmpty() || questions.isEmpty()) {
            throw new IllegalArgumentException("Corpus and questions must not be empty");
        }

        Set<String> documentIds = new HashSet<>();
        for (CorpusDocument document : corpus) {
            if (document.id() == null || document.id().isBlank()) {
                throw new IllegalArgumentException("Document id must not be blank");
            }
            if (!documentIds.add(document.id())) {
                throw new IllegalArgumentException("Duplicate document id: " + document.id());
            }
            if (document.indexableText().isBlank()) {
                throw new IllegalArgumentException("Empty document: " + document.id());
            }
        }

        Set<String> questionIds = new HashSet<>();
        for (EvalQuestion question : questions) {
            if (question.id() == null || question.id().isBlank() || !questionIds.add(question.id())) {
                throw new IllegalArgumentException("Duplicate or blank question id: " + question.id());
            }
            if (question.question() == null || question.question().isBlank()) {
                throw new IllegalArgumentException("Blank question: " + question.id());
            }
            List<String> relevantIds = question.relevanceSet();
            if (relevantIds.isEmpty() || relevantIds.stream().anyMatch(id -> !documentIds.contains(id))) {
                throw new IllegalArgumentException("Question has invalid relevance: " + question.id());
            }
            if (new HashSet<>(relevantIds).size() != relevantIds.size()) {
                throw new IllegalArgumentException("Question has duplicate relevance: " + question.id());
            }
        }
    }
}
