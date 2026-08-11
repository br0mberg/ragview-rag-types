package ru.brombin.ragview.retriever;

import ru.brombin.ragview.corpus.CorpusDocument;
import ru.brombin.ragview.strategy.RetrievedDoc;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Comparator;
import java.util.Map;
import java.util.Objects;

public final class Bm25Index {

    private static final double K1 = 1.2;
    private static final double B = 0.75;

    private final List<String> docIds = new ArrayList<>();
    private final List<Map<String, Integer>> termFreqs = new ArrayList<>();
    private final List<Integer> docLengths = new ArrayList<>();
    private final Map<String, Integer> docFreq = new HashMap<>();
    private double avgDocLength;

    public void index(List<CorpusDocument> corpus) {
        Objects.requireNonNull(corpus, "corpus");
        if (corpus.isEmpty()) {
            throw new IllegalArgumentException("Corpus must not be empty");
        }
        docIds.clear();
        termFreqs.clear();
        docLengths.clear();
        docFreq.clear();
        avgDocLength = 0;

        for (CorpusDocument doc : corpus) {
            List<String> tokens = Tokenizer.tokenize(doc.indexableText());
            Map<String, Integer> tf = new HashMap<>();
            for (String token : tokens) {
                tf.merge(token, 1, Integer::sum);
            }
            docIds.add(doc.id());
            termFreqs.add(tf);
            docLengths.add(tokens.size());
            for (String term : tf.keySet()) {
                docFreq.merge(term, 1, Integer::sum);
            }
        }
        avgDocLength = docLengths.stream().mapToInt(Integer::intValue).average().orElse(0);
        if (avgDocLength == 0) {
            throw new IllegalArgumentException("Corpus has no searchable terms");
        }
    }

    public List<RetrievedDoc> search(String query, int k) {
        if (k <= 0) {
            throw new IllegalArgumentException("k must be positive");
        }
        List<String> queryTerms = Tokenizer.tokenize(query);
        int totalDocs = docIds.size();

        List<RetrievedDoc> scored = new ArrayList<>(totalDocs);
        for (int i = 0; i < totalDocs; i++) {
            double score = 0.0;
            Map<String, Integer> tf = termFreqs.get(i);
            int docLen = docLengths.get(i);
            for (String term : queryTerms) {
                int freq = tf.getOrDefault(term, 0);
                if (freq == 0) {
                    continue;
                }
                int df = docFreq.getOrDefault(term, 0);
                double idf = Math.log(1 + (totalDocs - df + 0.5) / (df + 0.5));
                double norm = freq * (K1 + 1)
                        / (freq + K1 * (1 - B + B * docLen / avgDocLength));
                score += idf * norm;
            }
            if (score > 0) {
                scored.add(new RetrievedDoc(docIds.get(i), score));
            }
        }
        scored.sort(Comparator.comparingDouble(RetrievedDoc::score).reversed()
                .thenComparing(RetrievedDoc::docId));
        return List.copyOf(scored.subList(0, Math.min(k, scored.size())));
    }
}
