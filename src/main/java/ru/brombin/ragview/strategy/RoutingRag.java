package ru.brombin.ragview.strategy;

import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import ru.brombin.ragview.corpus.CorpusDocument;

import java.util.List;

@RequiredArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE, makeFinal = true)
public final class RoutingRag implements RagStrategy {

    RagStrategy articleReferenceStrategy;
    RagStrategy fallbackStrategy;
    ExplicitArticleReferenceRouter router;

    @Override
    public String name() {
        return "routing";
    }

    @Override
    public void index(List<CorpusDocument> corpus) {
        articleReferenceStrategy.index(corpus);
        fallbackStrategy.index(corpus);
    }

    @Override
    public List<RetrievedDoc> retrieve(String query, int k) {
        if (router.matches(query)) {
            return articleReferenceStrategy.retrieve(query, k);
        }
        return fallbackStrategy.retrieve(query, k);
    }
}
