package ru.brombin.ragview.artifact;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import ru.brombin.ragview.corpus.EvalQuestion;
import ru.brombin.ragview.strategy.RagStrategy;
import ru.brombin.ragview.strategy.RetrievedDoc;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

@RequiredArgsConstructor
public class CandidateDumpWriter {

    private final ObjectMapper objectMapper;

    public void write(Path output, RagStrategy strategy, List<EvalQuestion> questions, int limit) {
        Path absolute = output.toAbsolutePath();
        try {
            Files.createDirectories(absolute.getParent());
            try (BufferedWriter writer = Files.newBufferedWriter(absolute)) {
                for (EvalQuestion question : questions) {
                    writer.write(toJson(record(strategy, question, limit)));
                    writer.newLine();
                }
            }
        } catch (IOException error) {
            throw new UncheckedIOException("Cannot write " + absolute, error);
        }
    }

    private CandidateRecord record(RagStrategy strategy, EvalQuestion question, int limit) {
        List<RetrievedDoc> hits = strategy.retrieve(question.question(), limit);
        List<CandidateEntry> candidates = new ArrayList<>(hits.size());
        int relevantRank = 0;
        for (int index = 0; index < hits.size(); index++) {
            RetrievedDoc hit = hits.get(index);
            int rank = index + 1;
            candidates.add(new CandidateEntry(rank, hit.docId(), hit.score()));
            if (relevantRank == 0 && question.relevanceSet().contains(hit.docId())) {
                relevantRank = rank;
            }
        }
        return new CandidateRecord(
                question.id(), question.kindOrUnknown(), question.question(), question.relevanceSet(),
                relevantRank, candidates.size(), List.copyOf(candidates));
    }

    private String toJson(CandidateRecord record) throws JsonProcessingException {
        return objectMapper.writeValueAsString(record);
    }

    public record CandidateEntry(int rank, String docId, double score) {
    }

    public record CandidateRecord(
            String qid,
            String kind,
            String q,
            List<String> rel,
            int relevantRank,
            int candidateCount,
            List<CandidateEntry> cand) {
    }
}
