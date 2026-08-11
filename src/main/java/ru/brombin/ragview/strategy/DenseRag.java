package ru.brombin.ragview.strategy;

import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import ru.brombin.ragview.corpus.CorpusDocument;
import ru.brombin.ragview.retriever.DenseIndex;

import java.util.List;

@RequiredArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE, makeFinal = true)
public class DenseRag implements RagStrategy {

    DenseIndex dense;

    @Override
    public String name() {
        return "dense";
    }

    @Override
    public void index(List<CorpusDocument> corpus) {
        dense.index(corpus);
    }

    @Override
    public List<RetrievedDoc> retrieve(String query, int k) {
        return dense.search(query, k);
    }

}
