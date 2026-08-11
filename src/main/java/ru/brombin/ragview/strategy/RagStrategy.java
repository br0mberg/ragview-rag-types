package ru.brombin.ragview.strategy;

import ru.brombin.ragview.corpus.CorpusDocument;

import java.util.List;

public interface RagStrategy {

    String name();

    void index(List<CorpusDocument> corpus);

    List<RetrievedDoc> retrieve(String query, int k);

}
