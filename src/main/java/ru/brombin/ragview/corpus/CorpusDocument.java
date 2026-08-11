package ru.brombin.ragview.corpus;

public record CorpusDocument(String id, String title, String text) {

    public String indexableText() {
        String normalizedTitle = title == null ? "" : title.strip();
        String normalizedText = text == null ? "" : text.strip();
        if (normalizedTitle.isEmpty() || normalizedText.startsWith(normalizedTitle)) {
            return normalizedText;
        }
        return normalizedTitle + ". " + normalizedText;
    }
}
