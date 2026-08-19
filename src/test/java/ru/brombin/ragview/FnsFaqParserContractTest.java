package ru.brombin.ragview;

import org.junit.jupiter.api.Test;
import ru.brombin.ragview.eval.FnsFaqParser;

import static org.assertj.core.api.Assertions.assertThat;

class FnsFaqParserContractTest {

    private final FnsFaqParser parser = new FnsFaqParser();

    @Test
    void parse_shouldExtractQuestionAnswerSourceAndPagination() {
        String html = """
                <table>
                  <tr><td><strong class="pointer">Будет ли штраф по статье 129.13 НК РФ?<br><br></strong></td></tr>
                  <tr><td><div class="sp" style="display:none">
                    <div>Штраф составит 20 процентов.</div>
                    <div style="padding-top:15px"><b>Источник:</b></div>
                    <div>Ст.129.13 НК РФ; ст. 112 НК РФ.</div>
                    <a href="/documents/example">Документ</a>
                  </div></td></tr>
                </table>
                <div class="pages">Найдено: 230 | Показано с 1 по 10</div>
                <span class="pagination">
                  <a class="active">1</a><a onclick="SearchKb('2')">2</a><a onclick="SearchKb('23')">23</a>
                </span>
                """;

        FnsFaqParser.ParsedPage page = parser.parse(html, "https://www.nalog.gov.ru/rn77/service/kb/");

        assertThat(page.totalPages()).isEqualTo(23);
        assertThat(page.totalFound()).isEqualTo(230);
        assertThat(page.shownFrom()).isEqualTo(1);
        assertThat(page.shownTo()).isEqualTo(10);
        assertThat(page.questionMarkers()).isEqualTo(1);
        assertThat(page.skippedQuestionMarkers()).isZero();
        assertThat(page.entries()).singleElement().satisfies(entry -> {
            assertThat(entry.question()).isEqualTo("Будет ли штраф по статье 129.13 НК РФ?");
            assertThat(entry.answer()).isEqualTo("Штраф составит 20 процентов.");
            assertThat(entry.source()).isEqualTo("Ст.129.13 НК РФ; ст. 112 НК РФ. Документ");
            assertThat(entry.sourceLinks())
                    .containsExactly("https://www.nalog.gov.ru/documents/example");
            assertThat(entry.citedNkArticles()).containsExactly("129.13", "112");
            assertThat(entry.naturalExact()).isTrue();
        });
    }

    @Test
    void extractCitedNkArticles_shouldIgnoreOtherLawsAndBareArticleNumbers() {
        String source = "ч. 5 ст. 4 ФЗ № 17-ФЗ; ст. 75 НК РФ; "
                + "Налогового кодекса Российской Федерации, статья 112; письмо № 03-11-11/8320";

        assertThat(FnsFaqParser.extractCitedNkArticles(source)).containsExactly("75", "112");
    }

    @Test
    void normalizeQuestionKey_shouldNormalizeWhitespaceCaseAndYo() {
        assertThat(FnsFaqParser.normalizeQuestionKey("  Ёлка  ИП? "))
                .isEqualTo(FnsFaqParser.normalizeQuestionKey("елка ИП?"));
    }

    @Test
    void extractCitedNkArticles_shouldPreserveHyphenatedArticleNumber() {
        String source = "Пункт 2 статьи 25.12-1 НК РФ";

        assertThat(FnsFaqParser.extractCitedNkArticles(source)).containsExactly("25.12-1");
    }

    @Test
    void extractCitedNkArticles_shouldReadEveryArticleBeforeSharedTaxCodeAnchor() {
        assertThat(FnsFaqParser.extractCitedNkArticles(
                "Пункт 6 статьи 6.1 и пункт 7 статьи 431 НК"))
                .containsExactly("6.1", "431");
        assertThat(FnsFaqParser.extractCitedNkArticles(
                "п.6 ст.6.1, абз. 3 п.4 ст.31 Налогового кодекса РФ"))
                .containsExactly("6.1", "31");
        assertThat(FnsFaqParser.extractCitedNkArticles("П.1ст.256 НК РФ"))
                .containsExactly("256");
    }

    @Test
    void extractCitedNkArticles_shouldReadTaxCodeAnchorJoinedToArticleNumber() {
        assertThat(FnsFaqParser.extractCitedNkArticles(
                "Пункт 5 статьи 346.9Налогового кодекса Российской Федерации"))
                .containsExactly("346.9");
        assertThat(FnsFaqParser.extractCitedNkArticles(
                "Подпункт 12 пункта 2 статьи 346.43Налогового кодекса РФ"))
                .containsExactly("346.43");
        assertThat(FnsFaqParser.extractCitedNkArticles("П.2 ст.217.1НК РФ"))
                .containsExactly("217.1");
        assertThat(FnsFaqParser.extractCitedNkArticles("П.1ст.219.2НК РФ"))
                .containsExactly("219.2");
    }

    @Test
    void extractCitedNkArticles_shouldKeepTaxArticlesOutOfMixedLawChains() {
        String source = "ст. 75 НК РФ, ст. 10 ГК РФ, ст. 4 СК РФ, ст. 15 ФЗ № 17-ФЗ; "
                + "ст. 346.12 и ст. 346.13 Налогового кодекса РФ";

        assertThat(FnsFaqParser.extractCitedNkArticles(source))
                .containsExactly("75", "346.12", "346.13");
    }

    @Test
    void extractCitedNkArticles_shouldNotAttachUnanchoredArticleAcrossAnotherCitation() {
        String source = "ст. 10; ст. 75 НК РФ";

        assertThat(FnsFaqParser.extractCitedNkArticles(source)).containsExactly("75");
    }

    @Test
    void extractCitedNkArticles_shouldExpandBoundedRangesWithoutSplittingHyphenatedArticles() {
        String source = "статьи 23–25 и 25.12-1 НК РФ; статьи 346.11–346.13 НК РФ";

        assertThat(FnsFaqParser.extractCitedNkArticles(source))
                .containsExactly("23", "24", "25", "25.12-1", "346.11", "346.12", "346.13");
    }

    @Test
    void parse_shouldReportQuestionMarkersThatCouldNotBeParsed() {
        String html = """
                <table>
                  <tr><td><strong class="pointer">Вопрос без ответа</strong></td></tr>
                  <tr><td><div class="unexpected">Другая разметка</div></td></tr>
                </table>
                <div class="pages">Найдено: 1 | Показано с 1 по 1</div>
                """;

        FnsFaqParser.ParsedPage page = parser.parse(html, "https://www.nalog.gov.ru/");

        assertThat(page.questionMarkers()).isEqualTo(1);
        assertThat(page.skippedQuestionMarkers()).isEqualTo(1);
        assertThat(page.entries()).isEmpty();
    }
}
