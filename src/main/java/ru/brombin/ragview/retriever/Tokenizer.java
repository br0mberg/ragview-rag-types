package ru.brombin.ragview.retriever;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.ru.RussianAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.List;

public final class Tokenizer {

    private static final Analyzer ANALYZER = new RussianAnalyzer();

    private Tokenizer() {
    }

    public static List<String> tokenize(String text) {
        if (text == null || text.isBlank()) {
            return List.of();
        }
        List<String> tokens = new ArrayList<>();
        try (TokenStream stream = ANALYZER.tokenStream("text", text)) {
            CharTermAttribute term = stream.addAttribute(CharTermAttribute.class);
            stream.reset();
            while (stream.incrementToken()) {
                tokens.add(term.toString());
            }
            stream.end();
            return List.copyOf(tokens);
        } catch (IOException error) {
            throw new UncheckedIOException("Cannot analyze text", error);
        }
    }
}
