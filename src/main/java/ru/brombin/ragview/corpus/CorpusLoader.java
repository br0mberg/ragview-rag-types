package ru.brombin.ragview.corpus;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import ru.brombin.ragview.config.RagProperties;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

@Slf4j
@Component
@RequiredArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE, makeFinal = true)
public class CorpusLoader {

    ObjectMapper objectMapper;
    RagProperties properties;

    public List<CorpusDocument> loadDocuments() {
        return read("corpus.json", CorpusDocument[].class);
    }

    public List<EvalQuestion> loadQuestions() {
        return read("questions.json", EvalQuestion[].class);
    }

    private <T> List<T> read(String filename, Class<T[]> type) {
        Path external = Path.of(properties.datasetPath(), filename);
        if (!Files.isRegularFile(external)) {
            throw new IllegalStateException("Dataset file does not exist: " + external.toAbsolutePath());
        }
        try (InputStream in = Files.newInputStream(external)) {
            List<T> parsed = List.of(objectMapper.readValue(in, type));
            log.info("loaded {} entries from {}", parsed.size(), external.toAbsolutePath());
            return parsed;
        } catch (IOException e) {
            throw new IllegalStateException("Cannot read dataset: " + filename, e);
        }
    }
}
