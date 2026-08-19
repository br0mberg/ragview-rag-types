package ru.brombin.ragview.eval;

import java.net.URI;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.regex.Pattern;

public final class FnsFaqRobotsPolicy {

    private FnsFaqRobotsPolicy() {
    }

    public static boolean isAllowed(String robots, String userAgent, URI target) {
        if (robots == null || userAgent == null || target == null) {
            throw new NullPointerException("robots, userAgent and target are required");
        }
        String product = userAgent.toLowerCase(Locale.ROOT);
        List<Group> groups = parse(robots);
        List<Rule> exact = groups.stream()
                .filter(group -> group.agents().stream().anyMatch(product::equals))
                .flatMap(group -> group.rules().stream())
                .toList();
        List<Rule> rules = exact.isEmpty()
                ? groups.stream()
                .filter(group -> group.agents().contains("*"))
                .flatMap(group -> group.rules().stream())
                .toList()
                : exact;
        String path = target.getRawPath();
        if (target.getRawQuery() != null) {
            path += "?" + target.getRawQuery();
        }
        Rule selected = null;
        for (Rule rule : rules) {
            if (!rule.matches(path)) {
                continue;
            }
            if (selected == null
                    || rule.specificity() > selected.specificity()
                    || rule.specificity() == selected.specificity() && rule.allow() && !selected.allow()) {
                selected = rule;
            }
        }
        return selected == null || selected.allow();
    }

    private static List<Group> parse(String robots) {
        List<Group> groups = new ArrayList<>();
        List<String> agents = new ArrayList<>();
        List<Rule> rules = new ArrayList<>();
        boolean hasRules = false;
        for (String rawLine : robots.split("\\R")) {
            String line = rawLine.split("#", 2)[0].trim();
            int separator = line.indexOf(':');
            if (separator < 0) {
                continue;
            }
            String field = line.substring(0, separator).trim().toLowerCase(Locale.ROOT);
            String value = line.substring(separator + 1).trim();
            if (field.equals("user-agent")) {
                if (hasRules) {
                    groups.add(new Group(List.copyOf(agents), List.copyOf(rules)));
                    agents.clear();
                    rules.clear();
                    hasRules = false;
                }
                agents.add(value.toLowerCase(Locale.ROOT));
            } else if ((field.equals("allow") || field.equals("disallow")) && !agents.isEmpty()) {
                if (field.equals("disallow") && value.isEmpty()) {
                    continue;
                }
                rules.add(new Rule(field.equals("allow"), value));
                hasRules = true;
            }
        }
        if (!agents.isEmpty()) {
            groups.add(new Group(List.copyOf(agents), List.copyOf(rules)));
        }
        return List.copyOf(groups);
    }

    private record Group(List<String> agents, List<Rule> rules) {
    }

    private record Rule(boolean allow, String value) {

        boolean matches(String path) {
            return regex().matcher(path).find();
        }

        int specificity() {
            return value.endsWith("$") ? value.length() - 1 : value.length();
        }

        private Pattern regex() {
            boolean anchoredEnd = value.endsWith("$");
            String pattern = anchoredEnd ? value.substring(0, value.length() - 1) : value;
            StringBuilder result = new StringBuilder("^");
            for (int index = 0; index < pattern.length(); index++) {
                char character = pattern.charAt(index);
                result.append(character == '*' ? ".*" : Pattern.quote(Character.toString(character)));
            }
            if (anchoredEnd) {
                result.append('$');
            }
            return Pattern.compile(result.toString());
        }
    }
}
