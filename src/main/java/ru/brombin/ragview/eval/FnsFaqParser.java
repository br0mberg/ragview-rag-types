package ru.brombin.ragview.eval;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.nodes.Node;

import java.text.Normalizer;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class FnsFaqParser {

    public static final String CITATION_PARSER_VERSION = "nk-citation-hints-v2";

    private static final String CITATION_RULESET = String.join("\n",
            CITATION_PARSER_VERSION,
            "article markers: статья|статьи|ст.",
            "tax anchors: НК[РФ]|Налоговый кодекс[РФ]",
            "tax anchors may immediately follow an article number",
            "other-law anchors are exclusion boundaries",
            "single-anchor clause association; multi-anchor nearest-forward fail-closed association",
            "integer and same-prefix decimal ranges expand up to 200 articles");
    private static final Pattern PAGE_CALL = Pattern.compile("SearchKb\\(['\"]?(\\d+)['\"]?\\)");
    private static final Pattern FOUND = Pattern.compile("(?iu)Найдено\\s*:\\s*(\\d+)");
    private static final Pattern SHOWN = Pattern.compile(
            "(?iu)Показано\\s+с\\s+(\\d+)\\s+по\\s+(\\d+)");
    private static final String ARTICLE_NUMBER = "(?:\\d+(?:\\.\\d+)+(?:-\\d+)?|\\d+)";
    private static final String ARTICLE_WORD =
            "(?:статья|статьи|статье|статью|статьёй|статьей|статьями|ст\\.?)";
    private static final Pattern ARTICLE_MARKER = Pattern.compile(
            "(?iu)(?<![\\p{L}])" + ARTICLE_WORD + "(?=\\s*№?\\s*\\d)");
    private static final Pattern ARTICLE_NUMBER_AT_CURSOR = Pattern.compile(
            "\\s*№?\\s*(" + ARTICLE_NUMBER + ")");
    private static final Pattern LIST_SEPARATOR_AT_CURSOR = Pattern.compile(
            "\\s*(?:,|и(?!\\p{L}))\\s*");
    private static final Pattern RANGE_SEPARATOR_AT_CURSOR = Pattern.compile(
            "\\s*[-–—]\\s*");
    private static final Pattern NATURAL_ARTICLE = Pattern.compile(
            "(?iu)(?<![\\p{L}\\p{N}])" + ARTICLE_WORD + "\\s*№?\\s*" + ARTICLE_NUMBER);
    private static final Pattern TAX_CODE = Pattern.compile(
            "(?iu)(?<![\\p{L}])(?:НК(?:\\s*РФ)?|"
                    + "Налогов(?:ый|ого|ом|ому)\\s+кодекс(?:а|е|у|ом)?"
                    + "(?:\\s+Российской\\s+Федерации|\\s+РФ)?)(?![\\p{L}\\p{N}])");
    private static final Pattern OTHER_LAW = Pattern.compile(
            "(?iu)(?<![\\p{L}\\p{N}])(?:"
                    + "(?:ГК|СК|УК|ТК|ЖК|БК|ГПК|АПК)\\s*РФ|"
                    + "КоАП(?:\\s*РФ)?|"
                    + "ФЗ(?:\\s+от)?|"
                    + "Федеральн(?:ый|ого|ом|ому)\\s+закон(?:а|е|у|ом)?|"
                    + "Гражданск(?:ий|ого|ом|ому)\\s+кодекс(?:а|е|у|ом)?(?:\\s+РФ)?|"
                    + "Семейн(?:ый|ого|ом|ому)\\s+кодекс(?:а|е|у|ом)?(?:\\s+РФ)?"
                    + ")(?![\\p{L}\\p{N}])");
    private static final Pattern SOURCE_LABEL = Pattern.compile("(?iu)^Источник\\s*:?$");
    private static final int MAX_RANGE_SIZE = 200;

    public ParsedPage parse(String answerHtml, String baseUri) {
        if (answerHtml == null) {
            throw new IllegalArgumentException("Answer HTML must not be null");
        }
        Document document = Jsoup.parseBodyFragment(answerHtml, baseUri == null ? "" : baseUri);
        List<Element> questionElements = document.select("strong.pointer");
        List<ParsedFaq> entries = new ArrayList<>();
        int skippedQuestionMarkers = 0;
        for (Element questionElement : questionElements) {
            Element questionRow = questionElement.closest("tr");
            Element answerRow = questionRow == null ? null : questionRow.nextElementSibling();
            Element answerElement = answerRow == null ? null : answerRow.selectFirst("div.sp");
            String question = normalizeDisplay(questionElement.text());
            if (question.isBlank() || answerElement == null) {
                skippedQuestionMarkers++;
                continue;
            }
            AnswerParts parts = splitAnswer(answerElement);
            entries.add(new ParsedFaq(
                    question,
                    parts.answer(),
                    parts.source(),
                    parts.sourceLinks(),
                    extractCitedNkArticles(parts.source()),
                    NATURAL_ARTICLE.matcher(question).find()));
        }
        ShownRange shownRange = shownRange(document);
        return new ParsedPage(
                pageCount(document),
                foundCount(document),
                shownRange.from(),
                shownRange.to(),
                questionElements.size(),
                skippedQuestionMarkers,
                List.copyOf(entries));
    }

    public static String normalizeQuestionKey(String question) {
        String normalized = Normalizer.normalize(normalizeDisplay(question), Normalizer.Form.NFKC)
                .toLowerCase(Locale.ROOT)
                .replace('ё', 'е');
        return normalized;
    }

    public static String citationParserRulesetSha256() {
        return FnsFaqPageSampler.sha256(CITATION_RULESET);
    }

    public static List<String> extractCitedNkArticles(String source) {
        String value = normalizeDisplay(source);
        Set<String> result = new LinkedHashSet<>();
        for (TextClause clause : clauses(value)) {
            List<ArticleReference> references = articleReferences(clause.text(), clause.offset());
            if (references.isEmpty()) {
                continue;
            }
            List<LawAnchor> anchors = lawAnchors(clause.text(), clause.offset());
            if (anchors.size() == 1) {
                if (anchors.get(0).taxCode()) {
                    references.forEach(reference -> result.addAll(reference.articles()));
                }
                continue;
            }
            for (ArticleReference reference : references) {
                LawAnchor anchor = resolveAnchor(reference, references, anchors, value);
                if (anchor != null && anchor.taxCode()) {
                    result.addAll(reference.articles());
                }
            }
        }
        return List.copyOf(result);
    }

    private static List<TextClause> clauses(String source) {
        List<TextClause> clauses = new ArrayList<>();
        int start = 0;
        for (int index = 0; index <= source.length(); index++) {
            if (index == source.length() || source.charAt(index) == ';' || source.charAt(index) == '\n') {
                if (index > start) {
                    clauses.add(new TextClause(start, source.substring(start, index)));
                }
                start = index + 1;
            }
        }
        return clauses;
    }

    private static List<ArticleReference> articleReferences(String clause, int offset) {
        List<ArticleReference> references = new ArrayList<>();
        Matcher marker = ARTICLE_MARKER.matcher(clause);
        while (marker.find()) {
            ParsedArticleList parsed = parseArticleList(clause, marker.end());
            if (!parsed.articles().isEmpty()) {
                references.add(new ArticleReference(
                        offset + marker.start(),
                        offset + parsed.end(),
                        parsed.articles()));
            }
        }
        return references;
    }

    private static ParsedArticleList parseArticleList(String value, int start) {
        Matcher first = ARTICLE_NUMBER_AT_CURSOR.matcher(value).region(start, value.length());
        if (!first.lookingAt()) {
            return new ParsedArticleList(start, List.of());
        }
        List<String> articles = new ArrayList<>();
        String previous = first.group(1);
        articles.add(previous);
        int cursor = first.end();
        while (cursor < value.length()) {
            Matcher range = RANGE_SEPARATOR_AT_CURSOR.matcher(value).region(cursor, value.length());
            if (range.lookingAt() && canBeRangeEndpoint(previous, value, range.end())) {
                Matcher endpoint = ARTICLE_NUMBER_AT_CURSOR.matcher(value).region(range.end(), value.length());
                if (!endpoint.lookingAt()) {
                    break;
                }
                String next = endpoint.group(1);
                List<String> expansion = expandRange(previous, next);
                if (expansion.isEmpty()) {
                    break;
                }
                expansion.stream().skip(1).forEach(articles::add);
                previous = next;
                cursor = endpoint.end();
                continue;
            }
            Matcher separator = LIST_SEPARATOR_AT_CURSOR.matcher(value).region(cursor, value.length());
            if (!separator.lookingAt()) {
                break;
            }
            Matcher nextNumber = ARTICLE_NUMBER_AT_CURSOR.matcher(value).region(separator.end(), value.length());
            if (!nextNumber.lookingAt()) {
                Matcher nextMarker = ARTICLE_MARKER.matcher(value).region(separator.end(), value.length());
                if (nextMarker.find() && nextMarker.start() == separator.end()) {
                    ParsedArticleList nextList = parseArticleList(value, nextMarker.end());
                    if (!nextList.articles().isEmpty()) {
                        articles.addAll(nextList.articles());
                        cursor = nextList.end();
                    }
                }
                break;
            }
            previous = nextNumber.group(1);
            articles.add(previous);
            cursor = nextNumber.end();
        }
        return new ParsedArticleList(cursor, List.copyOf(articles));
    }

    private static boolean canBeRangeEndpoint(String previous, String value, int endpointStart) {
        Matcher endpoint = ARTICLE_NUMBER_AT_CURSOR.matcher(value).region(endpointStart, value.length());
        if (!endpoint.lookingAt()) {
            return false;
        }
        String candidate = endpoint.group(1);
        return previous.indexOf('.') == candidate.indexOf('.')
                || previous.contains(".") && candidate.contains(".");
    }

    private static List<String> expandRange(String from, String to) {
        if (from.contains("-") || to.contains("-")) {
            return List.of();
        }
        String[] left = from.split("\\.");
        String[] right = to.split("\\.");
        if (left.length != right.length) {
            return List.of();
        }
        for (int index = 0; index < left.length - 1; index++) {
            if (!left[index].equals(right[index])) {
                return List.of();
            }
        }
        int first = Integer.parseInt(left[left.length - 1]);
        int last = Integer.parseInt(right[right.length - 1]);
        if (last < first || last - first + 1 > MAX_RANGE_SIZE) {
            return List.of();
        }
        String prefix = left.length == 1
                ? ""
                : String.join(".", java.util.Arrays.copyOf(left, left.length - 1)) + ".";
        List<String> result = new ArrayList<>(last - first + 1);
        for (int value = first; value <= last; value++) {
            result.add(prefix + value);
        }
        return List.copyOf(result);
    }

    private static List<LawAnchor> lawAnchors(String clause, int offset) {
        List<LawAnchor> anchors = new ArrayList<>();
        collectAnchors(TAX_CODE.matcher(clause), offset, true, anchors);
        collectAnchors(OTHER_LAW.matcher(clause), offset, false, anchors);
        anchors.sort(java.util.Comparator.comparingInt(LawAnchor::start));
        return List.copyOf(anchors);
    }

    private static void collectAnchors(
            Matcher matcher,
            int offset,
            boolean taxCode,
            List<LawAnchor> anchors) {
        while (matcher.find()) {
            anchors.add(new LawAnchor(offset + matcher.start(), offset + matcher.end(), taxCode));
        }
    }

    private static LawAnchor resolveAnchor(
            ArticleReference reference,
            List<ArticleReference> references,
            List<LawAnchor> anchors,
            String source) {
        LawAnchor previous = null;
        LawAnchor next = null;
        for (LawAnchor anchor : anchors) {
            if (anchor.end() <= reference.start()) {
                previous = anchor;
            } else if (anchor.start() >= reference.end()) {
                next = anchor;
                break;
            }
        }
        if (next == null) {
            return previous;
        }
        if (previous == null) {
            return hasInterveningArticleReference(reference, next, references) ? null : next;
        }
        String betweenPrevious = source.substring(previous.end(), reference.start());
        boolean directReverseCitation = betweenPrevious.matches("[\\s,:()]*")
                && !hasForwardReference(previous, references, source);
        if (directReverseCitation && previous.taxCode() != next.taxCode()) {
            return null;
        }
        return directReverseCitation ? previous : next;
    }

    private static boolean hasForwardReference(
            LawAnchor anchor,
            List<ArticleReference> references,
            String source) {
        ArticleReference nearest = null;
        for (ArticleReference candidate : references) {
            if (candidate.end() <= anchor.start()
                    && (nearest == null || candidate.end() > nearest.end())) {
                nearest = candidate;
            }
        }
        return nearest != null
                && source.substring(nearest.end(), anchor.start()).matches("[\\s,:()]*");
    }

    private static boolean hasInterveningArticleReference(
            ArticleReference reference,
            LawAnchor anchor,
            List<ArticleReference> references) {
        return references.stream().anyMatch(candidate -> candidate.start() > reference.end()
                && candidate.end() <= anchor.start());
    }

    private static int pageCount(Document document) {
        int maximum = 1;
        for (Element link : document.select(".pagination [onclick]")) {
            Matcher matcher = PAGE_CALL.matcher(link.attr("onclick"));
            if (matcher.find()) {
                maximum = Math.max(maximum, Integer.parseInt(matcher.group(1)));
            }
        }
        return maximum;
    }

    private static int foundCount(Document document) {
        for (Element element : document.select(".pages")) {
            Matcher matcher = FOUND.matcher(element.text());
            if (matcher.find()) {
                return Integer.parseInt(matcher.group(1));
            }
        }
        return -1;
    }

    private static ShownRange shownRange(Document document) {
        for (Element element : document.select(".pages")) {
            Matcher matcher = SHOWN.matcher(element.text());
            if (matcher.find()) {
                return new ShownRange(Integer.parseInt(matcher.group(1)), Integer.parseInt(matcher.group(2)));
            }
        }
        return new ShownRange(-1, -1);
    }

    private static AnswerParts splitAnswer(Element original) {
        Element answer = original.clone();
        Element marker = answer.getAllElements().stream()
                .filter(element -> SOURCE_LABEL.matcher(normalizeDisplay(element.ownText())).matches())
                .findFirst()
                .orElse(null);
        if (marker == null) {
            return new AnswerParts(normalizeDisplay(answer.text()), "", List.of());
        }

        Element boundary = marker;
        while (boundary.parent() != null
                && boundary.parent() != answer
                && startsWithSourceLabel(boundary.parent().text())) {
            boundary = boundary.parent();
        }
        Element source = new Element("div", answer.baseUri());
        Node current = boundary;
        while (current != null) {
            Node next = current.nextSibling();
            source.appendChild(current.clone());
            current.remove();
            current = next;
        }
        String sourceText = normalizeDisplay(source.text())
                .replaceFirst("(?iu)^Источник\\s*:?\\s*", "");
        List<String> links = source.select("a[href]").stream()
                .map(link -> resolveLink(answer.baseUri(), link.attr("href")))
                .filter(value -> !value.isBlank())
                .distinct()
                .toList();
        return new AnswerParts(normalizeDisplay(answer.text()), sourceText, links);
    }

    private static boolean startsWithSourceLabel(String value) {
        return normalizeDisplay(value).matches("(?iu)^Источник\\s*:?.*");
    }

    private static String resolveLink(String baseUri, String href) {
        if (href == null || href.isBlank()) {
            return "";
        }
        try {
            return java.net.URI.create(baseUri).resolve(href).toString();
        } catch (IllegalArgumentException ignored) {
            return "";
        }
    }

    private static String normalizeDisplay(String value) {
        if (value == null) {
            return "";
        }
        return value.replace('\u00a0', ' ').trim().replaceAll("\\s+", " ");
    }

    private record AnswerParts(String answer, String source, List<String> sourceLinks) {
    }

    private record TextClause(int offset, String text) {
    }

    private record ParsedArticleList(int end, List<String> articles) {
    }

    private record ArticleReference(int start, int end, List<String> articles) {
    }

    private record LawAnchor(int start, int end, boolean taxCode) {
    }

    private record ShownRange(int from, int to) {
    }

    public record ParsedPage(
            int totalPages,
            int totalFound,
            int shownFrom,
            int shownTo,
            int questionMarkers,
            int skippedQuestionMarkers,
            List<ParsedFaq> entries) {
    }

    public record ParsedFaq(
            String question,
            String answer,
            String source,
            List<String> sourceLinks,
            List<String> citedNkArticles,
            boolean naturalExact) {
    }
}
