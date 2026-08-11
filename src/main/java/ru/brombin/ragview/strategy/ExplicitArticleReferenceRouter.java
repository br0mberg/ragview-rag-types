package ru.brombin.ragview.strategy;

import java.util.Objects;
import java.util.regex.Pattern;

public final class ExplicitArticleReferenceRouter {

    private static final Pattern ARTICLE_REFERENCE = Pattern.compile(
            "(?:статья|статье|статьи|статью|ст\\.)\\s*\\d+(?:[.\\-]\\d+)*",
            Pattern.CASE_INSENSITIVE | Pattern.UNICODE_CASE);

    public boolean matches(String query) {
        return ARTICLE_REFERENCE.matcher(Objects.requireNonNull(query, "query")).find();
    }
}
