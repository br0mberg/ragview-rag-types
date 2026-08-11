package ru.brombin.ragview.strategy;

import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import ru.brombin.ragview.corpus.CorpusDocument;
import ru.brombin.ragview.retriever.Bm25Index;

import java.util.List;

@RequiredArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE, makeFinal = true)
public class Bm25Rag implements RagStrategy {

    Bm25Index bm25;

    @Override
    public String name() {
        return "bm25";
    }

    @Override
    public void index(List<CorpusDocument> corpus) {
        bm25.index(corpus);
    }

    @Override
    public List<RetrievedDoc> retrieve(String query, int k) {
        return bm25.search(query, k);
    }

}
