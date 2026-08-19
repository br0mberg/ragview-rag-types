package ru.brombin.ragview;

import org.junit.jupiter.api.Test;
import ru.brombin.ragview.eval.FnsFaqRobotsPolicy;

import java.net.URI;

import static org.assertj.core.api.Assertions.assertThat;

class FnsFaqRobotsPolicyContractTest {

    @Test
    void isAllowed_shouldAllowFaqAndAjaxUnderCurrentRules() {
        String robots = """
                User-agent: *
                Disallow: /rn*/search/
                Disallow: /rn*/not_found/
                """;

        assertThat(allowed(robots, "/rn77/service/kb/")).isTrue();
        assertThat(allowed(robots, "/Ajax.html")).isTrue();
        assertThat(allowed(robots, "/rn77/search/")).isFalse();
    }

    @Test
    void isAllowed_shouldPreferSpecificAgentAndLongestRule() {
        String robots = """
                User-agent: *
                Disallow: /

                User-agent: ragview-fns-eval-collector
                Disallow: /rn77/service/
                Allow: /rn77/service/kb/
                """;

        assertThat(allowed(robots, "/rn77/service/kb/")).isTrue();
        assertThat(allowed(robots, "/rn77/service/other/")).isFalse();
        assertThat(allowed(robots, "/Ajax.html")).isTrue();
    }

    @Test
    void isAllowed_shouldIgnoreEmptyDisallowAndHonorEndAnchor() {
        String robots = """
                User-agent: *
                Disallow:
                Disallow: /Ajax.html$
                """;

        assertThat(allowed(robots, "/Ajax.html")).isFalse();
        assertThat(allowed(robots, "/Ajax.html/other")).isTrue();
    }

    private static boolean allowed(String robots, String path) {
        return FnsFaqRobotsPolicy.isAllowed(
                robots,
                "ragview-fns-eval-collector",
                URI.create("https://www.nalog.gov.ru" + path));
    }
}
