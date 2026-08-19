package ru.brombin.ragview.eval;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.json.JsonMapper;
import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import ru.brombin.ragview.corpus.CorpusDocument;
import ru.brombin.ragview.rerank.AttestedRerankScorer;
import ru.brombin.ragview.rerank.HttpRerankScorer;
import ru.brombin.ragview.rerank.RerankScorer;
import ru.brombin.ragview.rerank.RerankerAttestation;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

@RequiredArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE, makeFinal = true)
public final class FnsFaqBgeLatencyRunner {

    public static final int MANIFEST_SCHEMA_VERSION = 1;
    public static final int SOURCE_BGE_MANIFEST_SCHEMA_VERSION = 2;
    public static final String EVAL_VERSION = "tax-eval-v3-bge-latency-v1";
    public static final List<Integer> CUTOFFS = List.of(10, 20, 30, 50);
    public static final int DEFAULT_WARMUP_QUESTIONS = 8;
    public static final int DEFAULT_WARMUP_REPEATS = 1;
    public static final int DEFAULT_MEASURED_REPEATS = 3;
    public static final String DEFAULT_ORDER_SEED =
            "ragview-v3-bge-latency-order-2026-08-16";
    public static final String RAW_FILENAME = "latency-raw.csv";
    public static final String SUMMARY_FILENAME = "latency-summary.csv";
    public static final String MANIFEST_FILENAME = "manifest.json";
    public static final String ORDER_ALGORITHM =
            "sha256(seed\\0phase\\0repeat\\0qid), then sha256(seed\\0phase\\0repeat\\0qid\\0cutoff)";
    public static final String PERCENTILE_ALGORITHM = "nearest-rank: ceil(p*n)-1";

    ObjectMapper objectMapper;
    RerankScorer scorer;
    NanoClock nanoClock;

    public static void main(String[] args) throws Exception {
        CliArguments cli = CliArguments.parse(args);
        if (cli.help()) {
            System.out.println("Usage: FnsFaqBgeLatencyRunner --questions PATH --corpus PATH "
                    + "--bge-manifest PATH --bge-provenance PATH --output PATH "
                    + "[--rerank-base-url http://127.0.0.1:8079] "
                    + "[--warmup-questions 8] [--warmup-repeats 1] [--repeats 3] "
                    + "[--order-seed VALUE]");
            return;
        }

        ObjectMapper objectMapper = JsonMapper.builder()
                .enable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES)
                .enable(SerializationFeature.INDENT_OUTPUT)
                .build();
        RerankScorer scorer = new HttpRerankScorer(
                cli.rerankBaseUrl(),
                FnsFaqBgeCutoffEvaluator.BGE_MODEL,
                FnsFaqBgeCutoffEvaluator.BGE_REVISION,
                FnsFaqBgeCutoffEvaluator.BGE_PRECISION,
                FnsFaqBgeCutoffEvaluator.BGE_DEVICE,
                FnsFaqBgeCutoffEvaluator.BGE_BATCH_SIZE,
                FnsFaqBgeCutoffEvaluator.BGE_MAX_LENGTH);
        LatencyRequest request = new LatencyRequest(
                cli.warmupQuestions(),
                cli.warmupRepeats(),
                cli.measuredRepeats(),
                cli.orderSeed());
        new FnsFaqBgeLatencyRunner(objectMapper, scorer, System::nanoTime).run(
                cli.questions(),
                cli.corpus(),
                cli.bgeManifest(),
                cli.bgeProvenance(),
                cli.output(),
                request,
                Instant.now());
        System.out.println(cli.output().toAbsolutePath().normalize());
    }

    public LatencyManifest run(
            Path questionsPath,
            Path corpusPath,
            Path bgeManifestPath,
            Path bgeProvenancePath,
            Path outputDirectory,
            LatencyRequest request,
            Instant createdAt) throws IOException {
        if (request == null || createdAt == null) {
            throw new IllegalArgumentException("Latency request and createdAt are required");
        }
        Path output = validateOutputDirectory(outputDirectory);
        InputBytes questionsInput = readInput(questionsPath, "questions");
        InputBytes corpusInput = readInput(corpusPath, "corpus");
        InputBytes bgeManifestInput = readInput(bgeManifestPath, "BGE manifest");
        InputBytes bgeProvenanceInput = readInput(bgeProvenancePath, "BGE provenance");

        List<FnsFaqCandidate> questions = readQuestions(questionsInput);
        CorpusInput corpus = readCorpus(corpusInput);
        FrozenBgeRun frozenBgeRun = readFrozenBgeRun(
                bgeManifestInput,
                bgeProvenanceInput,
                questionsInput,
                corpusInput,
                questions.size(),
                corpus.documents().size());
        RerankerAttestation liveAttestation = requireMatchingAttestation(
                scorer, frozenBgeRun.rerankerAttestation());
        List<QuestionWork> work = readQuestionWork(
                bgeProvenanceInput, questions, corpus.byId());
        if (request.warmupQuestions() > work.size()) {
            throw new IllegalArgumentException("warmupQuestions exceeds frozen question count");
        }

        runWarmup(work, request);
        List<LatencyMeasurement> measurements = runMeasurements(work, request);
        List<LatencySummary> summaries = summarize(measurements);

        Files.createDirectories(output);
        Path rawPath = output.resolve(RAW_FILENAME);
        Path summaryPath = output.resolve(SUMMARY_FILENAME);
        writeRawCsv(rawPath, measurements);
        writeSummaryCsv(summaryPath, summaries);

        MeasurementOptions measurementOptions = new MeasurementOptions(
                CUTOFFS,
                request.warmupQuestions(),
                request.warmupRepeats(),
                request.measuredRepeats(),
                measurements.size(),
                request.orderSeed(),
                ORDER_ALGORITHM,
                "System.nanoTime",
                "RerankScorer.score(query, first-stage prefix)",
                PERCENTILE_ALGORITHM);
        LatencyManifest manifest = new LatencyManifest(
                MANIFEST_SCHEMA_VERSION,
                EVAL_VERSION,
                createdAt.toString(),
                frozenBgeRun.sourceEvalVersion(),
                questions.size(),
                corpus.documents().size(),
                artifact(questionsInput),
                artifact(corpusInput),
                artifact(bgeManifestInput),
                artifact(bgeProvenanceInput),
                frozenBgeRun.rerankerAttestation(),
                liveAttestation,
                measurementOptions,
                RuntimeEnvironment.current(),
                outputArtifact(rawPath),
                outputArtifact(summaryPath));
        objectMapper.writerWithDefaultPrettyPrinter()
                .writeValue(output.resolve(MANIFEST_FILENAME).toFile(), manifest);
        return manifest;
    }

    private List<FnsFaqCandidate> readQuestions(InputBytes input) throws IOException {
        FnsFaqCandidate[] parsed = objectMapper.readValue(input.bytes(), FnsFaqCandidate[].class);
        if (parsed.length == 0) {
            throw new IllegalArgumentException("Frozen questions must not be empty");
        }
        Set<String> seen = new HashSet<>();
        for (FnsFaqCandidate question : parsed) {
            if (question.id() == null
                    || !question.id().matches("[A-Za-z0-9._-]+")
                    || !seen.add(question.id())) {
                throw new IllegalArgumentException("Question IDs must be safe, non-blank and unique");
            }
            if (question.question() == null || question.question().isBlank()) {
                throw new IllegalArgumentException("Blank question: " + question.id());
            }
        }
        return List.of(parsed);
    }

    private CorpusInput readCorpus(InputBytes input) throws IOException {
        List<CorpusDocument> documents = List.of(
                objectMapper.readValue(input.bytes(), CorpusDocument[].class));
        if (documents.isEmpty()) {
            throw new IllegalArgumentException("Frozen corpus must not be empty");
        }
        Map<String, CorpusDocument> byId = new LinkedHashMap<>();
        for (CorpusDocument document : documents) {
            if (document.id() == null || document.id().isBlank()
                    || document.title() == null || document.title().isBlank()
                    || document.text() == null || document.text().isBlank()
                    || byId.putIfAbsent(document.id(), document) != null) {
                throw new IllegalArgumentException("Corpus documents must be complete and unique");
            }
        }
        return new CorpusInput(documents, Map.copyOf(byId));
    }

    private FrozenBgeRun readFrozenBgeRun(
            InputBytes manifestInput,
            InputBytes provenanceInput,
            InputBytes questionsInput,
            InputBytes corpusInput,
            int questionCount,
            int corpusDocumentCount) throws IOException {
        JsonNode manifest = objectMapper.readTree(manifestInput.bytes());
        requireEqual(SOURCE_BGE_MANIFEST_SCHEMA_VERSION,
                requiredPositiveInt(manifest, "schemaVersion", "BGE manifest"),
                "Unsupported BGE manifest schema");
        requireTextEquals(manifest, "evalVersion", "tax-eval-v3-bge-reranker-pool", "BGE manifest");
        String sourceEvalVersion = requiredText(manifest, "sourceEvalVersion", "BGE manifest");
        if (!"tax-eval-v3-fns-candidates".equals(sourceEvalVersion)) {
            throw new IllegalArgumentException("BGE manifest has unexpected sourceEvalVersion");
        }
        requireEqual(questionCount,
                requiredPositiveInt(manifest, "questionCount", "BGE manifest"),
                "BGE question count differs from questions");
        requireEqual(corpusDocumentCount,
                requiredPositiveInt(manifest, "corpusDocumentCount", "BGE manifest"),
                "BGE corpus count differs from corpus");
        validateArtifact(manifest, "questions", questionsInput, "BGE manifest");
        validateArtifact(manifest, "corpus", corpusInput, "BGE manifest");
        validateArtifact(manifest, "sealedProvenance", provenanceInput, "BGE manifest");
        validateReferencedHash(manifest, "snapshotManifest", "BGE manifest");
        validateReferencedHash(manifest, "sourcePoolManifest", "BGE manifest");
        validateReferencedHash(manifest, "sourceSealedProvenance", "BGE manifest");

        RerankerAttestation attestation = objectMapper.treeToValue(
                requiredObject(manifest, "rerankerAttestation", "BGE manifest"),
                RerankerAttestation.class);
        validatePinnedIdentity(attestation, "BGE manifest attestation");
        JsonNode pooling = requiredObject(manifest, "pooling", "BGE manifest");
        requireEqual(10, requiredPositiveInt(pooling, "topN", "BGE pooling"),
                "BGE output topN differs from frozen contract");
        requireEqual(50, requiredPositiveInt(pooling, "scoredPool", "BGE pooling"),
                "BGE scored pool differs from frozen contract");
        requireTextEquals(pooling, "profile", "bge-cross-encoder", "BGE pooling");
        requireTextEquals(pooling, "implementation", "cross-encoder-reranker", "BGE pooling");
        requireTextEquals(pooling, "model", attestation.model(), "BGE pooling");
        requireTextEquals(pooling, "revision", attestation.revision(), "BGE pooling");
        requireTextEquals(pooling, "precision", attestation.precision(), "BGE pooling");
        requireTextEquals(pooling, "device", attestation.device(), "BGE pooling");
        requireEqual(attestation.batchSize(),
                requiredPositiveInt(pooling, "batchSize", "BGE pooling"),
                "BGE pooling batch size differs from attestation");
        requireEqual(attestation.maxLength(),
                requiredPositiveInt(pooling, "maxLength", "BGE pooling"),
                "BGE pooling max length differs from attestation");
        requireTextEquals(
                pooling,
                "ordering",
                "score_desc,first_stage_rank_asc,doc_id_asc",
                "BGE pooling");
        return new FrozenBgeRun(sourceEvalVersion, attestation);
    }

    private List<QuestionWork> readQuestionWork(
            InputBytes provenance,
            List<FnsFaqCandidate> questions,
            Map<String, CorpusDocument> corpus) throws IOException {
        List<JsonNode> rows = readJsonLines(provenance);
        requireEqual(questions.size(), rows.size(), "BGE provenance line count differs from questions");
        List<QuestionWork> work = new ArrayList<>(questions.size());
        Set<String> qids = new HashSet<>();
        for (int questionIndex = 0; questionIndex < questions.size(); questionIndex++) {
            FnsFaqCandidate question = questions.get(questionIndex);
            JsonNode row = rows.get(questionIndex);
            String qid = requiredText(row, "qid", "BGE provenance row");
            if (!question.id().equals(qid) || !qids.add(qid)) {
                throw new IllegalArgumentException("BGE provenance qids differ from frozen questions");
            }
            JsonNode bge = requiredArray(row, "bge", "BGE provenance row");
            if (bge.size() != 50) {
                throw new IllegalArgumentException("BGE provenance must contain 50 candidates for " + qid);
            }
            List<String> passagesByFirstStageRank = new ArrayList<>(java.util.Collections.nCopies(50, null));
            Set<String> docIds = new HashSet<>();
            double previousScore = Double.POSITIVE_INFINITY;
            int previousFirstStageRank = 0;
            String previousDocId = "";
            for (int index = 0; index < bge.size(); index++) {
                JsonNode candidate = bge.get(index);
                int firstStageRank = requiredPositiveInt(
                        candidate, "firstStageRank", "BGE candidate");
                int rerankRank = requiredPositiveInt(candidate, "rerankRank", "BGE candidate");
                String docId = requiredText(candidate, "docId", "BGE candidate");
                double score = requiredFiniteDouble(candidate, "score", "BGE candidate");
                if (firstStageRank > 50 || rerankRank != index + 1
                        || !docIds.add(docId) || !corpus.containsKey(docId)) {
                    throw new IllegalArgumentException("Invalid BGE candidate for " + qid);
                }
                if (score > previousScore
                        || Double.compare(score, previousScore) == 0
                        && (firstStageRank < previousFirstStageRank
                        || firstStageRank == previousFirstStageRank
                        && docId.compareTo(previousDocId) < 0)) {
                    throw new IllegalArgumentException("BGE provenance ordering is invalid for " + qid);
                }
                if (passagesByFirstStageRank.set(
                        firstStageRank - 1, corpus.get(docId).indexableText()) != null) {
                    throw new IllegalArgumentException("Duplicate first-stage rank for " + qid);
                }
                previousScore = score;
                previousFirstStageRank = firstStageRank;
                previousDocId = docId;
            }
            if (passagesByFirstStageRank.stream().anyMatch(java.util.Objects::isNull)) {
                throw new IllegalArgumentException("Incomplete first-stage ranks for " + qid);
            }
            work.add(new QuestionWork(
                    qid, question.question(), List.copyOf(passagesByFirstStageRank)));
        }
        return List.copyOf(work);
    }

    private void runWarmup(List<QuestionWork> work, LatencyRequest request) {
        List<QuestionWork> warmupQuestions = orderedQuestions(
                work, request.orderSeed(), "warmup-sample", 0).stream()
                .limit(request.warmupQuestions())
                .toList();
        for (int repeat = 1; repeat <= request.warmupRepeats(); repeat++) {
            for (QuestionWork question : orderedQuestions(
                    warmupQuestions, request.orderSeed(), "warmup", repeat)) {
                for (int cutoff : orderedCutoffs(
                        request.orderSeed(), "warmup", repeat, question.qid())) {
                    score(question, cutoff);
                }
            }
        }
    }

    private List<LatencyMeasurement> runMeasurements(
            List<QuestionWork> work,
            LatencyRequest request) {
        List<LatencyMeasurement> measurements = new ArrayList<>(
                work.size() * CUTOFFS.size() * request.measuredRepeats());
        int sequence = 0;
        for (int repeat = 1; repeat <= request.measuredRepeats(); repeat++) {
            for (QuestionWork question : orderedQuestions(
                    work, request.orderSeed(), "measure", repeat)) {
                for (int cutoff : orderedCutoffs(
                        request.orderSeed(), "measure", repeat, question.qid())) {
                    List<String> passages = question.passagesByFirstStageRank().subList(0, cutoff);
                    long startedAt = nanoClock.nanoTime();
                    List<Double> scores = scorer.score(question.question(), passages);
                    long elapsedNanos = nanoClock.nanoTime() - startedAt;
                    validateScores(question.qid(), cutoff, scores);
                    if (elapsedNanos < 0) {
                        throw new IllegalStateException("Nano clock moved backwards");
                    }
                    measurements.add(new LatencyMeasurement(
                            ++sequence,
                            repeat,
                            question.qid(),
                            cutoff,
                            cutoff,
                            elapsedNanos,
                            elapsedNanos / 1_000_000.0));
                }
            }
        }
        return List.copyOf(measurements);
    }

    private void score(QuestionWork question, int cutoff) {
        List<Double> scores = scorer.score(
                question.question(), question.passagesByFirstStageRank().subList(0, cutoff));
        validateScores(question.qid(), cutoff, scores);
    }

    private static void validateScores(String qid, int cutoff, List<Double> scores) {
        if (scores == null || scores.size() != cutoff
                || scores.stream().anyMatch(score -> score == null || !Double.isFinite(score))) {
            throw new IllegalStateException("BGE returned invalid scores for " + qid + " at cutoff " + cutoff);
        }
    }

    private static List<QuestionWork> orderedQuestions(
            List<QuestionWork> questions,
            String seed,
            String phase,
            int repeat) {
        return questions.stream()
                .sorted(Comparator
                        .comparing((QuestionWork question) -> FnsFaqBgeCutoffEvaluator.sha256(
                                (seed + "\0" + phase + "\0" + repeat + "\0" + question.qid())
                                        .getBytes(StandardCharsets.UTF_8)))
                        .thenComparing(QuestionWork::qid))
                .toList();
    }

    private static List<Integer> orderedCutoffs(
            String seed,
            String phase,
            int repeat,
            String qid) {
        return CUTOFFS.stream()
                .sorted(Comparator
                        .comparing((Integer cutoff) -> FnsFaqBgeCutoffEvaluator.sha256(
                                (seed + "\0" + phase + "\0" + repeat + "\0" + qid + "\0" + cutoff)
                                        .getBytes(StandardCharsets.UTF_8)))
                        .thenComparingInt(Integer::intValue))
                .toList();
    }

    private static List<LatencySummary> summarize(List<LatencyMeasurement> measurements) {
        List<LatencySummary> summaries = new ArrayList<>(CUTOFFS.size());
        for (int cutoff : CUTOFFS) {
            List<Long> values = measurements.stream()
                    .filter(measurement -> measurement.cutoff() == cutoff)
                    .map(LatencyMeasurement::elapsedNanos)
                    .sorted()
                    .toList();
            if (values.isEmpty()) {
                throw new IllegalStateException("No latency measurements for cutoff " + cutoff);
            }
            double meanNanos = values.stream().mapToLong(Long::longValue).average().orElseThrow();
            summaries.add(new LatencySummary(
                    cutoff,
                    values.size(),
                    meanNanos / 1_000_000.0,
                    nearestRank(values, 0.50) / 1_000_000.0,
                    nearestRank(values, 0.95) / 1_000_000.0));
        }
        return List.copyOf(summaries);
    }

    private static long nearestRank(List<Long> sortedValues, double percentile) {
        int index = Math.max(0, (int) Math.ceil(percentile * sortedValues.size()) - 1);
        return sortedValues.get(index);
    }

    private static void writeRawCsv(Path path, List<LatencyMeasurement> measurements) throws IOException {
        try (BufferedWriter writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            writer.write("sequence,repeat,qid,cutoff,candidate_count,elapsed_ns,elapsed_ms\n");
            for (LatencyMeasurement measurement : measurements) {
                writer.write(String.format(
                        Locale.ROOT,
                        "%d,%d,%s,%d,%d,%d,%.6f%n",
                        measurement.sequence(),
                        measurement.repeat(),
                        measurement.qid(),
                        measurement.cutoff(),
                        measurement.candidateCount(),
                        measurement.elapsedNanos(),
                        measurement.elapsedMillis()));
            }
        }
    }

    private static void writeSummaryCsv(Path path, List<LatencySummary> summaries) throws IOException {
        try (BufferedWriter writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            writer.write("cutoff,sample_count,mean_ms,p50_ms,p95_ms\n");
            for (LatencySummary summary : summaries) {
                writer.write(String.format(
                        Locale.ROOT,
                        "%d,%d,%.6f,%.6f,%.6f%n",
                        summary.cutoff(),
                        summary.sampleCount(),
                        summary.meanMillis(),
                        summary.p50Millis(),
                        summary.p95Millis()));
            }
        }
    }

    private List<JsonNode> readJsonLines(InputBytes input) throws IOException {
        List<JsonNode> rows = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new StringReader(
                new String(input.bytes(), StandardCharsets.UTF_8)))) {
            String line;
            int lineNumber = 0;
            while ((line = reader.readLine()) != null) {
                lineNumber++;
                if (line.isBlank()) {
                    throw new IllegalArgumentException(
                            "Blank JSONL row at " + input.filename() + ":" + lineNumber);
                }
                try {
                    rows.add(objectMapper.readTree(line));
                } catch (IOException error) {
                    throw new IllegalArgumentException(
                            "Invalid JSONL row at " + input.filename() + ":" + lineNumber, error);
                }
            }
        }
        return List.copyOf(rows);
    }

    private static RerankerAttestation requireMatchingAttestation(
            RerankScorer scorer,
            RerankerAttestation frozenAttestation) {
        if (!(scorer instanceof AttestedRerankScorer attestedScorer)) {
            throw new IllegalStateException("BGE latency run requires an attested rerank scorer");
        }
        RerankerAttestation liveAttestation = attestedScorer.attestation();
        validatePinnedIdentity(liveAttestation, "Live reranker attestation");
        if (!frozenAttestation.equals(liveAttestation)) {
            throw new IllegalStateException("Live reranker attestation differs from frozen BGE run");
        }
        return liveAttestation;
    }

    private static void validatePinnedIdentity(RerankerAttestation attestation, String label) {
        if (attestation == null
                || attestation.healthSchemaVersion() != 1
                || !FnsFaqBgeCutoffEvaluator.BGE_MODEL.equals(attestation.model())
                || !FnsFaqBgeCutoffEvaluator.BGE_REVISION.equals(attestation.revision())
                || !FnsFaqBgeCutoffEvaluator.BGE_PRECISION.equals(attestation.precision())
                || !FnsFaqBgeCutoffEvaluator.BGE_DEVICE.equals(attestation.device())
                || FnsFaqBgeCutoffEvaluator.BGE_BATCH_SIZE != attestation.batchSize()
                || FnsFaqBgeCutoffEvaluator.BGE_MAX_LENGTH != attestation.maxLength()) {
            throw new IllegalStateException(label + " does not match the pinned BGE identity");
        }
    }

    private static void validateArtifact(
            JsonNode manifest,
            String field,
            InputBytes input,
            String label) {
        JsonNode artifact = requiredObject(manifest, field, label);
        String filename = requiredText(artifact, "filename", label + " " + field);
        if (!filename.equals(input.filename())) {
            throw new IllegalArgumentException(label + " has unexpected filename for " + field);
        }
        String expectedSha256 = requiredSha256(artifact, "sha256", label + " " + field);
        if (!expectedSha256.equals(input.sha256())) {
            throw new IllegalArgumentException(label + " has mismatched SHA-256 for " + field);
        }
    }

    private static void validateReferencedHash(JsonNode manifest, String field, String label) {
        JsonNode artifact = requiredObject(manifest, field, label);
        requiredText(artifact, "filename", label + " " + field);
        requiredSha256(artifact, "sha256", label + " " + field);
    }

    private static Path validateOutputDirectory(Path outputDirectory) throws IOException {
        if (outputDirectory == null) {
            throw new IllegalArgumentException("output directory is required");
        }
        Path output = outputDirectory.toAbsolutePath().normalize();
        if (Files.exists(output)) {
            if (!Files.isDirectory(output)) {
                throw new IOException("Output path is not a directory: " + output);
            }
            try (var children = Files.list(output)) {
                if (children.findAny().isPresent()) {
                    throw new IOException("Output directory is not empty: " + output);
                }
            }
        }
        return output;
    }

    private static InputBytes readInput(Path path, String label) throws IOException {
        if (path == null || !Files.isRegularFile(path)) {
            throw new IllegalArgumentException(label + " file does not exist: " + path);
        }
        Path normalized = path.toAbsolutePath().normalize();
        byte[] bytes = Files.readAllBytes(normalized);
        return new InputBytes(
                normalized,
                normalized.getFileName().toString(),
                bytes,
                FnsFaqBgeCutoffEvaluator.sha256(bytes));
    }

    private static JsonNode requiredObject(JsonNode parent, String field, String label) {
        JsonNode value = parent.path(field);
        if (!value.isObject()) {
            throw new IllegalArgumentException(label + " has invalid " + field);
        }
        return value;
    }

    private static JsonNode requiredArray(JsonNode parent, String field, String label) {
        JsonNode value = parent.path(field);
        if (!value.isArray()) {
            throw new IllegalArgumentException(label + " has invalid " + field);
        }
        return value;
    }

    private static String requiredText(JsonNode parent, String field, String label) {
        JsonNode value = parent.path(field);
        if (!value.isTextual() || value.textValue().isBlank()) {
            throw new IllegalArgumentException(label + " has invalid " + field);
        }
        return value.textValue();
    }

    private static String requiredSha256(JsonNode parent, String field, String label) {
        String value = requiredText(parent, field, label);
        if (!value.matches("[0-9a-f]{64}")) {
            throw new IllegalArgumentException(label + " has invalid " + field);
        }
        return value;
    }

    private static int requiredPositiveInt(JsonNode parent, String field, String label) {
        JsonNode value = parent.path(field);
        if (!value.canConvertToInt() || value.intValue() <= 0) {
            throw new IllegalArgumentException(label + " has invalid " + field);
        }
        return value.intValue();
    }

    private static double requiredFiniteDouble(JsonNode parent, String field, String label) {
        JsonNode value = parent.path(field);
        if (!value.isNumber() || !Double.isFinite(value.doubleValue())) {
            throw new IllegalArgumentException(label + " has invalid " + field);
        }
        return value.doubleValue();
    }

    private static void requireTextEquals(
            JsonNode parent,
            String field,
            String expected,
            String label) {
        String actual = requiredText(parent, field, label);
        if (!expected.equals(actual)) {
            throw new IllegalArgumentException(label + " has unexpected " + field + ": " + actual);
        }
    }

    private static void requireEqual(int expected, int actual, String message) {
        if (expected != actual) {
            throw new IllegalArgumentException(
                    message + ": expected=" + expected + ", actual=" + actual);
        }
    }

    private static InputArtifact artifact(InputBytes input) {
        return new InputArtifact(input.filename(), input.sha256());
    }

    private static OutputArtifact outputArtifact(Path path) throws IOException {
        return new OutputArtifact(
                path.getFileName().toString(),
                FnsFaqBgeCutoffEvaluator.sha256(Files.readAllBytes(path)));
    }

    @FunctionalInterface
    public interface NanoClock {
        long nanoTime();
    }

    public record LatencyRequest(
            int warmupQuestions,
            int warmupRepeats,
            int measuredRepeats,
            String orderSeed) {

        public LatencyRequest {
            if (warmupQuestions <= 0 || warmupRepeats <= 0 || measuredRepeats <= 0) {
                throw new IllegalArgumentException("Warmup and measured repeat counts must be positive");
            }
            if (orderSeed == null || orderSeed.isBlank()) {
                throw new IllegalArgumentException("orderSeed must not be blank");
            }
        }
    }

    public record LatencyMeasurement(
            int sequence,
            int repeat,
            String qid,
            int cutoff,
            int candidateCount,
            long elapsedNanos,
            double elapsedMillis) {
    }

    public record LatencySummary(
            int cutoff,
            int sampleCount,
            double meanMillis,
            double p50Millis,
            double p95Millis) {
    }

    public record InputArtifact(String filename, String sha256) {
    }

    public record OutputArtifact(String filename, String sha256) {
    }

    public record MeasurementOptions(
            List<Integer> cutoffs,
            int warmupQuestionCount,
            int warmupRepeats,
            int measuredRepeats,
            int sampleCount,
            String orderSeed,
            String orderAlgorithm,
            String timer,
            String measuredOperation,
            String percentileAlgorithm) {

        public MeasurementOptions {
            cutoffs = List.copyOf(cutoffs);
        }
    }

    public record RuntimeEnvironment(
            String javaVersion,
            String javaVm,
            String osName,
            String osVersion,
            String osArch,
            int availableProcessors) {

        static RuntimeEnvironment current() {
            return new RuntimeEnvironment(
                    System.getProperty("java.version"),
                    System.getProperty("java.vm.name"),
                    System.getProperty("os.name"),
                    System.getProperty("os.version"),
                    System.getProperty("os.arch"),
                    Runtime.getRuntime().availableProcessors());
        }
    }

    public record LatencyManifest(
            int schemaVersion,
            String evalVersion,
            String createdAtUtc,
            String sourceEvalVersion,
            int questionCount,
            int corpusDocumentCount,
            InputArtifact questions,
            InputArtifact corpus,
            InputArtifact sourceBgeManifest,
            InputArtifact sourceBgeSealedProvenance,
            RerankerAttestation frozenRerankerAttestation,
            RerankerAttestation liveRerankerAttestation,
            MeasurementOptions measurement,
            RuntimeEnvironment runtime,
            OutputArtifact rawLatency,
            OutputArtifact summary) {
    }

    private record InputBytes(Path path, String filename, byte[] bytes, String sha256) {
    }

    private record CorpusInput(
            List<CorpusDocument> documents,
            Map<String, CorpusDocument> byId) {
    }

    private record FrozenBgeRun(
            String sourceEvalVersion,
            RerankerAttestation rerankerAttestation) {
    }

    private record QuestionWork(
            String qid,
            String question,
            List<String> passagesByFirstStageRank) {
    }

    record CliArguments(
            Path questions,
            Path corpus,
            Path bgeManifest,
            Path bgeProvenance,
            Path output,
            String rerankBaseUrl,
            int warmupQuestions,
            int warmupRepeats,
            int measuredRepeats,
            String orderSeed,
            boolean help) {

        static CliArguments parse(String[] args) {
            Path questions = null;
            Path corpus = null;
            Path bgeManifest = null;
            Path bgeProvenance = null;
            Path output = null;
            String rerankBaseUrl = environment(
                    "RAGVIEW_RERANK_BASE_URL", "http://127.0.0.1:8079");
            int warmupQuestions = DEFAULT_WARMUP_QUESTIONS;
            int warmupRepeats = DEFAULT_WARMUP_REPEATS;
            int measuredRepeats = DEFAULT_MEASURED_REPEATS;
            String orderSeed = DEFAULT_ORDER_SEED;
            boolean help = false;

            for (int index = 0; index < args.length; index++) {
                String argument = args[index];
                if ("--help".equals(argument) || "-h".equals(argument)) {
                    help = true;
                } else if (argument.startsWith("--questions=")) {
                    questions = Path.of(argument.substring("--questions=".length()));
                } else if ("--questions".equals(argument)) {
                    questions = Path.of(requireValue(args, ++index, "--questions"));
                } else if (argument.startsWith("--corpus=")) {
                    corpus = Path.of(argument.substring("--corpus=".length()));
                } else if ("--corpus".equals(argument)) {
                    corpus = Path.of(requireValue(args, ++index, "--corpus"));
                } else if (argument.startsWith("--bge-manifest=")) {
                    bgeManifest = Path.of(argument.substring("--bge-manifest=".length()));
                } else if ("--bge-manifest".equals(argument)) {
                    bgeManifest = Path.of(requireValue(args, ++index, "--bge-manifest"));
                } else if (argument.startsWith("--bge-provenance=")) {
                    bgeProvenance = Path.of(argument.substring("--bge-provenance=".length()));
                } else if ("--bge-provenance".equals(argument)) {
                    bgeProvenance = Path.of(requireValue(args, ++index, "--bge-provenance"));
                } else if (argument.startsWith("--output=")) {
                    output = Path.of(argument.substring("--output=".length()));
                } else if ("--output".equals(argument)) {
                    output = Path.of(requireValue(args, ++index, "--output"));
                } else if (argument.startsWith("--rerank-base-url=")) {
                    rerankBaseUrl = argument.substring("--rerank-base-url=".length());
                } else if ("--rerank-base-url".equals(argument)) {
                    rerankBaseUrl = requireValue(args, ++index, "--rerank-base-url");
                } else if (argument.startsWith("--warmup-questions=")) {
                    warmupQuestions = parsePositiveInt(
                            argument.substring("--warmup-questions=".length()), "--warmup-questions");
                } else if ("--warmup-questions".equals(argument)) {
                    warmupQuestions = parsePositiveInt(
                            requireValue(args, ++index, "--warmup-questions"), "--warmup-questions");
                } else if (argument.startsWith("--warmup-repeats=")) {
                    warmupRepeats = parsePositiveInt(
                            argument.substring("--warmup-repeats=".length()), "--warmup-repeats");
                } else if ("--warmup-repeats".equals(argument)) {
                    warmupRepeats = parsePositiveInt(
                            requireValue(args, ++index, "--warmup-repeats"), "--warmup-repeats");
                } else if (argument.startsWith("--repeats=")) {
                    measuredRepeats = parsePositiveInt(
                            argument.substring("--repeats=".length()), "--repeats");
                } else if ("--repeats".equals(argument)) {
                    measuredRepeats = parsePositiveInt(
                            requireValue(args, ++index, "--repeats"), "--repeats");
                } else if (argument.startsWith("--order-seed=")) {
                    orderSeed = argument.substring("--order-seed=".length());
                } else if ("--order-seed".equals(argument)) {
                    orderSeed = requireValue(args, ++index, "--order-seed");
                } else {
                    throw new IllegalArgumentException("Unknown argument: " + argument);
                }
            }

            if (!help && (questions == null || corpus == null || bgeManifest == null
                    || bgeProvenance == null || output == null)) {
                throw new IllegalArgumentException(
                        "--questions, --corpus, --bge-manifest, --bge-provenance and --output are required");
            }
            if (rerankBaseUrl.isBlank() || orderSeed.isBlank()) {
                throw new IllegalArgumentException("Rerank URL and order seed must not be blank");
            }
            return new CliArguments(
                    questions,
                    corpus,
                    bgeManifest,
                    bgeProvenance,
                    output,
                    rerankBaseUrl,
                    warmupQuestions,
                    warmupRepeats,
                    measuredRepeats,
                    orderSeed,
                    help);
        }

        private static int parsePositiveInt(String value, String option) {
            try {
                int parsed = Integer.parseInt(value);
                if (parsed <= 0) {
                    throw new NumberFormatException();
                }
                return parsed;
            } catch (NumberFormatException error) {
                throw new IllegalArgumentException(option + " must be a positive integer", error);
            }
        }

        private static String requireValue(String[] args, int index, String option) {
            if (index >= args.length || args[index].startsWith("--")) {
                throw new IllegalArgumentException("Missing value for " + option);
            }
            return args[index];
        }

        private static String environment(String name, String fallback) {
            String value = System.getenv(name);
            return value == null || value.isBlank() ? fallback : value;
        }
    }
}
