package ru.brombin.ragview.strategy;

import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import ru.brombin.ragview.corpus.CorpusDocument;
import ru.brombin.ragview.retriever.Bm25Index;
import ru.brombin.ragview.retriever.DenseIndex;
import ru.brombin.ragview.retriever.ReciprocalRankFusion;

import java.util.List;

@RequiredArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE, makeFinal = true)
public class HybridRag implements RagStrategy {

    Bm25Index bm25;
    DenseIndex dense;
    int candidatePool;
    int rrfK;

    @Override
    public String name() {
        return "hybrid";
    }

    @Override
    public void index(List<CorpusDocument> corpus) {
        bm25.index(corpus);
        dense.index(corpus);
    }

    @Override
    public List<RetrievedDoc> retrieve(String query, int k) {
        List<RetrievedDoc> lexical = bm25.search(query, candidatePool);
        List<RetrievedDoc> semantic = dense.search(query, candidatePool);
        return ReciprocalRankFusion.fuse(k, rrfK, lexical, semantic);
    }

}
