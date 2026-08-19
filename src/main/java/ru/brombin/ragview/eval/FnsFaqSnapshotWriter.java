package ru.brombin.ragview.eval;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public final class FnsFaqSnapshotWriter {

    private final ObjectMapper objectMapper;
    private final Path outputDirectory;

    public FnsFaqSnapshotWriter(ObjectMapper objectMapper, Path outputDirectory) {
        this.objectMapper = objectMapper;
        this.outputDirectory = outputDirectory.toAbsolutePath().normalize();
    }

    public void initialize() throws IOException {
        if (Files.exists(outputDirectory)) {
            try (var children = Files.list(outputDirectory)) {
                if (children.findAny().isPresent()) {
                    throw new IOException("Output directory is not empty: " + outputDirectory);
                }
            }
        }
        Files.createDirectories(outputDirectory);
    }

    public RawArtifact writeRaw(int categoryId, int page, byte[] bytes) throws IOException {
        Path relative = Path.of("raw", "category-%d".formatted(categoryId), "page-%04d.json".formatted(page));
        Path target = outputDirectory.resolve(relative);
        Files.createDirectories(target.getParent());
        Files.write(target, bytes);
        return new RawArtifact(relative.toString().replace('\\', '/'), FnsFaqPageSampler.sha256(bytes));
    }

    public String writeJson(String filename, Object value) throws IOException {
        byte[] bytes = objectMapper.writerWithDefaultPrettyPrinter().writeValueAsBytes(value);
        Files.write(outputDirectory.resolve(filename), bytes);
        return FnsFaqPageSampler.sha256(bytes);
    }

    public Path outputDirectory() {
        return outputDirectory;
    }

    public record RawArtifact(String path, String sha256) {
    }
}
