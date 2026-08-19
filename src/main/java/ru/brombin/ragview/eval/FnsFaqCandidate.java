package ru.brombin.ragview.eval;

import java.util.List;

public record FnsFaqCandidate(
        String id,
        String question,
        String answer,
        String citedSource,
        List<String> sourceLinks,
        List<String> citedNkArticles,
        boolean naturalExact,
        int categoryId,
        String categoryName,
        int page,
        String sourceUrl,
        String rawResponseSha256) {

    public FnsFaqCandidate {
        sourceLinks = List.copyOf(sourceLinks);
        citedNkArticles = List.copyOf(citedNkArticles);
    }
}
