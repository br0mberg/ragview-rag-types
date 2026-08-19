package ru.brombin.ragview.eval;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.json.JsonMapper;
import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import ru.brombin.ragview.config.PrefixEmbeddingModel;
import ru.brombin.ragview.corpus.CorpusDocument;
import ru.brombin.ragview.qdrant.QdrantV3HybridParityPool;
import ru.brombin.ragview.retriever.ReciprocalRankFusion;
import ru.brombin.ragview.strategy.RetrievedDoc;

import java.io.BufferedWriter;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

@RequiredArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE, makeFinal = true)
public final class FnsFaqQdrantHybridParityRunner {

    public static final int EXPECTED_QUESTION_COUNT = 240;
    public static final int DEFAULT_BRANCH_DEPTH = 50;
    public static final int DEFAULT_CLIENT_RRF_K_ONE_BASED = 60;
    public static final int DEFAULT_QDRANT_RRF_K_ZERO_BASED = 61;
    public static final double SCORE_TOLERANCE = 1e-7;
    public static final String BERTA_MODEL = "sergeyzh/BERTA";
    public static final String BERTA_REVISION =
            "914c8c8aed14042ed890fc2c662d5e9e66b2faa7";
    public static final String BERTA_POOLING = "mean";
    public static final int BERTA_DIMENSIONS = 768;
    public static final String BERTA_QUERY_PREFIX = "search_query: ";
    public static final String BERTA_DOCUMENT_PREFIX = "search_document: ";

    static final String RANKINGS_FILENAME = "question-only-rankings.sealed.jsonl";
    static final String MANIFEST_FILENAME = "manifest.json";
    static final String CHECKSUM_FILENAME = "MANIFEST.sha256";
    static final String QUERY_PROJECTION = "id+question; answers, citations and source hints ignored";
    static final String CLIENT_TIE_BREAK = "score_desc,doc_id_asc";
    static final String SERVER_TIE_POLICY =
            "Qdrant 1.19 API does not specify a secondary key; equal-score permutations are tie-only";

    ObjectMapper objectMapper;
    Backend backend;

    public static void main(String[] args) throws Exception {
        CliArguments cli = CliArguments.parse(args);
        if (cli.help()) {
            System.out.println("Usage: FnsFaqQdrantHybridParityRunner "
                    + "--questions PATH --corpus PATH --output PATH --collection NAME "
                    + "[--embedding-base-url http://127.0.0.1:8077] "
                    + "[--expected-device cuda] [--qdrant-host 127.0.0.1] "
                    + "[--qdrant-grpc-port 6334] [--qdrant-rest-port 6333] "
                    + "[--branch-depth 50] [--client-rrf-k 60] [--upsert-batch-size 64]");
            return;
        }

        ObjectMapper objectMapper = JsonMapper.builder()
                .enable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES)
                .enable(SerializationFeature.INDENT_OUTPUT)
                .build();
        EmbeddingIdentity embedding = readEmbeddingIdentity(
                objectMapper, cli.embeddingBaseUrl(), cli.expectedDevice());
        PrefixEmbeddingModel model = new PrefixEmbeddingModel(
                cli.embeddingBaseUrl(),
                embedding.model(),
                embedding.revision(),
                embedding.pooling(),
                embedding.dimensions(),
                cli.expectedDevice());

        try (Backend backend = new QdrantV3HybridParityPool(
                objectMapper,
                model,
                embedding,
                cli.qdrantHost(),
                cli.qdrantGrpcPort(),
                cli.qdrantRestPort(),
                cli.collection(),
                cli.upsertBatchSize())) {
            RunOptions options = new RunOptions(
                    cli.branchDepth(),
                    cli.branchDepth() * 2,
                    cli.clientRrfK(),
                    cli.clientRrfK() + 1);
            new FnsFaqQdrantHybridParityRunner(objectMapper, backend).run(
                    cli.questions(), cli.corpus(), cli.output(), options, Instant.now());
        }
        System.out.println(cli.output().toAbsolutePath().normalize());
    }

    public RunManifest run(
            Path questionsPath,
            Path corpusPath,
            Path outputDirectory,
            RunOptions options,
            Instant createdAt) throws IOException {
        if (options == null || createdAt == null) {
            throw new IllegalArgumentException("Run options and createdAt are required");
        }
        validateOutputDirectory(outputDirectory);
        FrozenQuestions questions = readFrozenQuestions(questionsPath);
        CorpusInput corpus = readCorpus(corpusPath, questions.expectedCorpusSha256());
        BackendIdentity backendIdentity = backend.prepare(corpus.documents(), corpus.sha256());

        List<RawRankingRow> rows = new ArrayList<>(questions.questions().size());
        ParityAccumulator parity = new ParityAccumulator();
        int branchSize = Math.min(options.branchDepth(), corpus.documents().size());
        int maximumFusionSize = Math.min(options.fusionLimit(), branchSize * 2);
        for (Question question : questions.questions()) {
            QueryRankings result = backend.search(question.question(), options);
            validateBranch(question.qid(), "dense", result.dense(), branchSize, corpus.documentIds());
            validateBranch(question.qid(), "sparse", result.sparse(), branchSize, corpus.documentIds());
            validateFusion(
                    question.qid(), result, options, maximumFusionSize, corpus.documentIds());
            ParityResult comparison = compare(result.clientRrf(), result.serverRrf());
            parity.add(comparison);
            rows.add(new RawRankingRow(
                    question.qid(),
                    ranked(result.dense()),
                    ranked(result.sparse()),
                    ranked(result.clientRrf()),
                    ranked(result.serverRrf()),
                    comparison));
        }

        Path output = initializeOutputDirectory(outputDirectory);
        Path rankingsPath = output.resolve(RANKINGS_FILENAME);
        writeJsonLines(rankingsPath, rows);
        OutputArtifact rankingsArtifact = new OutputArtifact(
                RANKINGS_FILENAME, sha256(Files.readAllBytes(rankingsPath)));

        RrfContract rrf = new RrfContract(
                options.clientRrfKOneBased(),
                options.qdrantRrfKZeroBased(),
                "sum(1/(" + options.clientRrfKOneBased() + "+rank_one_based))",
                "sum(1/(" + options.qdrantRrfKZeroBased() + "+rank_zero_based))",
                CLIENT_TIE_BREAK,
                SERVER_TIE_POLICY,
                SCORE_TOLERANCE);
        RunManifest manifest = new RunManifest(
                1,
                "tax-eval-v3-qdrant-1.19-native-hybrid-rrf-parity",
                createdAt.toString(),
                questions.sourceEvalVersion(),
                questions.questions().size(),
                corpus.documents().size(),
                new InputArtifact(questions.questionsPath().getFileName().toString(), questions.questionsSha256()),
                new InputArtifact(
                        questions.snapshotManifestPath().getFileName().toString(),
                        questions.snapshotManifestSha256()),
                new InputArtifact(corpus.path().getFileName().toString(), corpus.sha256()),
                questions.questionProjectionSha256(),
                QUERY_PROJECTION,
                options,
                backendIdentity,
                rrf,
                parity.summary(),
                rankingsArtifact);
        Path manifestPath = output.resolve(MANIFEST_FILENAME);
        objectMapper.writerWithDefaultPrettyPrinter().writeValue(manifestPath.toFile(), manifest);
        writeChecksums(output, rankingsArtifact, manifestPath);
        return manifest;
    }

    private FrozenQuestions readFrozenQuestions(Path questionsPath) throws IOException {
        Path questions = requireRegularFile(questionsPath, "questions");
        Path snapshotManifest = requireRegularFile(
                questions.resolveSibling("manifest.json"), "snapshot manifest");
        byte[] questionBytes = Files.readAllBytes(questions);
        byte[] manifestBytes = Files.readAllBytes(snapshotManifest);
        String questionsSha256 = sha256(questionBytes);
        JsonNode manifest = objectMapper.readTree(manifestBytes);
        if (requireInt(manifest, "schemaVersion") != 1) {
            throw new IllegalArgumentException("Unsupported snapshot manifest schema");
        }
        if (!questionsSha256.equals(requireText(manifest, "candidatesSha256"))) {
            throw new IllegalArgumentException("candidates.json does not match its frozen manifest");
        }

        JsonNode array = objectMapper.readTree(questionBytes);
        if (!array.isArray()
                || array.size() != requireInt(manifest, "selectedCandidates")
                || array.size() != EXPECTED_QUESTION_COUNT) {
            throw new IllegalArgumentException("Expected exactly 240 frozen questions");
        }
        List<Question> projected = new ArrayList<>(array.size());
        Set<String> qids = new HashSet<>();
        StringBuilder projectionMaterial = new StringBuilder();
        for (JsonNode node : array) {
            String qid = requireText(node, "id");
            String question = requireText(node, "question");
            if (!qids.add(qid)) {
                throw new IllegalArgumentException("Duplicate question ID: " + qid);
            }
            projected.add(new Question(qid, question));
            projectionMaterial.append(qid).append('\0').append(question).append('\n');
        }
        return new FrozenQuestions(
                List.copyOf(projected),
                questions,
                questionsSha256,
                snapshotManifest,
                sha256(manifestBytes),
                requireText(manifest.path("scope"), "corpusSha256"),
                requireText(manifest, "evalVersion"),
                sha256(projectionMaterial.toString()));
    }

    private CorpusInput readCorpus(Path corpusPath, String expectedSha256) throws IOException {
        Path corpus = requireRegularFile(corpusPath, "corpus");
        byte[] bytes = Files.readAllBytes(corpus);
        String actualSha256 = sha256(bytes);
        if (!actualSha256.equals(expectedSha256)) {
            throw new IllegalArgumentException("Corpus does not match the frozen snapshot scope");
        }
        List<CorpusDocument> documents = List.of(objectMapper.readValue(bytes, CorpusDocument[].class));
        if (documents.isEmpty()) {
            throw new IllegalArgumentException("Corpus must not be empty");
        }
        Set<String> documentIds = new HashSet<>();
        for (CorpusDocument document : documents) {
            if (document.id() == null || document.id().isBlank() || !documentIds.add(document.id())) {
                throw new IllegalArgumentException("Corpus document IDs must be non-blank and unique");
            }
            if (document.indexableText().isBlank()) {
                throw new IllegalArgumentException("Corpus document text must not be blank: " + document.id());
            }
        }
        return new CorpusInput(documents, Set.copyOf(documentIds), corpus, actualSha256);
    }

    private static void validateBranch(
            String qid,
            String label,
            List<RetrievedDoc> ranking,
            int expectedSize,
            Set<String> corpusIds) {
        validateRanking(qid, label, ranking, expectedSize, corpusIds);
    }

    private static void validateFusion(
            String qid,
            QueryRankings result,
            RunOptions options,
            int maximumFusionSize,
            Set<String> corpusIds) {
        LinkedHashSet<String> union = new LinkedHashSet<>();
        result.dense().stream().map(RetrievedDoc::docId).forEach(union::add);
        result.sparse().stream().map(RetrievedDoc::docId).forEach(union::add);
        int expectedSize = Math.min(maximumFusionSize, union.size());
        validateRanking(qid, "clientRrf", result.clientRrf(), expectedSize, corpusIds);
        validateRanking(qid, "serverRrf", result.serverRrf(), expectedSize, corpusIds);

        Set<String> expectedSet = Set.copyOf(union);
        Set<String> clientSet = docIdSet(result.clientRrf());
        Set<String> serverSet = docIdSet(result.serverRrf());
        if (!expectedSet.equals(clientSet)) {
            throw new IllegalStateException("Client RRF does not contain the complete branch union for " + qid);
        }
        if (!expectedSet.equals(serverSet)) {
            throw new IllegalStateException(
                    "Server RRF prefetch differs from the standalone branch lists for " + qid);
        }

        List<RetrievedDoc> recomputed = ReciprocalRankFusion.fuse(
                options.fusionLimit(),
                options.clientRrfKOneBased(),
                result.dense(),
                result.sparse());
        if (!sameRanking(recomputed, result.clientRrf(), 1e-12)) {
            throw new IllegalStateException("Client RRF cannot be reproduced from branch ranks for " + qid);
        }
    }

    private static void validateRanking(
            String qid,
            String label,
            List<RetrievedDoc> ranking,
            int expectedSize,
            Set<String> corpusIds) {
        if (ranking == null || ranking.size() != expectedSize) {
            throw new IllegalStateException(label + " ranking for " + qid + " has "
                    + (ranking == null ? 0 : ranking.size()) + " results, expected " + expectedSize);
        }
        Set<String> seen = new HashSet<>();
        for (RetrievedDoc document : ranking) {
            if (document == null
                    || document.docId() == null
                    || !corpusIds.contains(document.docId())
                    || !seen.add(document.docId())) {
                throw new IllegalStateException("Invalid " + label + " ranking for " + qid);
            }
            if (!Double.isFinite(document.score())) {
                throw new IllegalStateException("Non-finite " + label + " score for " + qid);
            }
        }
    }

    public static ParityResult compare(List<RetrievedDoc> client, List<RetrievedDoc> server) {
        boolean identicalOrder = docIds(client).equals(docIds(server));
        boolean sameSet = docIdSet(client).equals(docIdSet(server));
        Map<String, Double> clientScores = new HashMap<>();
        client.forEach(document -> clientScores.put(document.docId(), document.score()));
        double maxScoreDelta = 0;
        boolean scoreEquivalent = sameSet;
        for (RetrievedDoc document : server) {
            Double expected = clientScores.get(document.docId());
            double delta = expected == null
                    ? Double.POSITIVE_INFINITY
                    : Math.abs(document.score() - expected);
            maxScoreDelta = Math.max(maxScoreDelta, delta);
            scoreEquivalent &= delta <= SCORE_TOLERANCE;
        }

        boolean preservesScoreOrder = sameSet && scoreEquivalent;
        for (int index = 1; index < server.size() && preservesScoreOrder; index++) {
            double previous = clientScores.get(server.get(index - 1).docId());
            double current = clientScores.get(server.get(index).docId());
            preservesScoreOrder = previous + SCORE_TOLERANCE >= current;
        }
        int firstDifferenceRank = 0;
        for (int index = 0; index < Math.min(client.size(), server.size()); index++) {
            if (!client.get(index).docId().equals(server.get(index).docId())) {
                firstDifferenceRank = index + 1;
                break;
            }
        }
        if (firstDifferenceRank == 0 && client.size() != server.size()) {
            firstDifferenceRank = Math.min(client.size(), server.size()) + 1;
        }
        boolean tieOnlyDifference = !identicalOrder && preservesScoreOrder;
        return new ParityResult(
                identicalOrder,
                sameSet,
                scoreEquivalent,
                tieOnlyDifference,
                firstDifferenceRank,
                maxScoreDelta);
    }

    private static boolean sameRanking(
            List<RetrievedDoc> expected,
            List<RetrievedDoc> actual,
            double tolerance) {
        if (expected.size() != actual.size()) {
            return false;
        }
        for (int index = 0; index < expected.size(); index++) {
            RetrievedDoc left = expected.get(index);
            RetrievedDoc right = actual.get(index);
            if (!left.docId().equals(right.docId())
                    || Math.abs(left.score() - right.score()) > tolerance) {
                return false;
            }
        }
        return true;
    }

    private static List<RankedCandidate> ranked(List<RetrievedDoc> ranking) {
        List<RankedCandidate> rows = new ArrayList<>(ranking.size());
        for (int index = 0; index < ranking.size(); index++) {
            RetrievedDoc document = ranking.get(index);
            rows.add(new RankedCandidate(index + 1, document.docId(), document.score()));
        }
        return List.copyOf(rows);
    }

    private static List<String> docIds(List<RetrievedDoc> ranking) {
        return ranking.stream().map(RetrievedDoc::docId).toList();
    }

    private static Set<String> docIdSet(List<RetrievedDoc> ranking) {
        return Set.copyOf(docIds(ranking));
    }

    private static Path requireRegularFile(Path path, String label) {
        if (path == null || !Files.isRegularFile(path)) {
            throw new IllegalArgumentException(label + " file does not exist: " + path);
        }
        return path.toAbsolutePath().normalize();
    }

    private static int requireInt(JsonNode parent, String field) {
        JsonNode value = parent == null ? null : parent.get(field);
        if (value == null || !value.canConvertToInt()) {
            throw new IllegalArgumentException("Invalid or missing integer field: " + field);
        }
        return value.intValue();
    }

    private static String requireText(JsonNode parent, String field) {
        JsonNode value = parent == null ? null : parent.get(field);
        if (value == null || !value.isTextual() || value.textValue().isBlank()) {
            throw new IllegalArgumentException("Invalid or missing text field: " + field);
        }
        return value.textValue();
    }

    private static void validateOutputDirectory(Path path) throws IOException {
        if (path == null) {
            throw new IllegalArgumentException("Output directory is required");
        }
        if (Files.exists(path)) {
            if (!Files.isDirectory(path)) {
                throw new IOException("Output path is not a directory: " + path);
            }
            try (var children = Files.list(path)) {
                if (children.findAny().isPresent()) {
                    throw new IOException("Output directory is not empty: " + path);
                }
            }
        }
    }

    private static Path initializeOutputDirectory(Path outputDirectory) throws IOException {
        Path output = outputDirectory.toAbsolutePath().normalize();
        Files.createDirectories(output);
        return output;
    }

    private void writeJsonLines(Path path, List<?> rows) throws IOException {
        var writer = objectMapper.writer().without(SerializationFeature.INDENT_OUTPUT);
        try (BufferedWriter output = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            for (Object row : rows) {
                output.write(writer.writeValueAsString(row));
                output.write('\n');
            }
        }
    }

    private static void writeChecksums(
            Path output,
            OutputArtifact rankings,
            Path manifestPath) throws IOException {
        String checksums = rankings.sha256() + "  " + rankings.filename() + "\n"
                + sha256(Files.readAllBytes(manifestPath)) + "  " + MANIFEST_FILENAME + "\n";
        Files.writeString(output.resolve(CHECKSUM_FILENAME), checksums, StandardCharsets.UTF_8);
    }

    static EmbeddingIdentity readEmbeddingIdentity(
            ObjectMapper objectMapper,
            String baseUrl,
            String expectedDevice) throws IOException, InterruptedException {
        if (baseUrl == null || baseUrl.isBlank() || expectedDevice == null || expectedDevice.isBlank()) {
            throw new IllegalArgumentException("Embedding endpoint and expected device are required");
        }
        HttpClient client = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(5))
                .build();
        HttpRequest request = HttpRequest.newBuilder(
                        URI.create(baseUrl.replaceAll("/+$", "") + "/health"))
                .timeout(Duration.ofSeconds(30))
                .GET()
                .build();
        HttpResponse<byte[]> response = client.send(request, HttpResponse.BodyHandlers.ofByteArray());
        if (response.statusCode() != 200) {
            throw new IllegalStateException("Embedding health returned HTTP " + response.statusCode());
        }
        JsonNode health = objectMapper.readTree(response.body());
        EmbeddingIdentity identity = new EmbeddingIdentity(
                requireInt(health, "schemaVersion"),
                requireText(health, "model"),
                requireText(health, "revision"),
                requireText(health, "pooling"),
                requireInt(health, "dimensions"),
                requireInt(health, "maxLength"),
                requireText(health, "queryPrefix"),
                requireText(health, "documentPrefix"),
                requireText(health, "device"),
                sha256(response.body()));
        if (identity.schemaVersion() != 1
                || !BERTA_MODEL.equals(identity.model())
                || !BERTA_REVISION.equals(identity.revision())
                || !BERTA_POOLING.equals(identity.pooling())
                || BERTA_DIMENSIONS != identity.dimensions()
                || !BERTA_QUERY_PREFIX.equals(identity.queryPrefix())
                || !BERTA_DOCUMENT_PREFIX.equals(identity.documentPrefix())
                || !identity.device().toLowerCase(Locale.ROOT)
                .startsWith(expectedDevice.toLowerCase(Locale.ROOT))) {
            throw new IllegalStateException("Embedding server is not the pinned BERTA runtime");
        }
        return identity;
    }

    public static String sha256(String value) {
        return sha256(value.getBytes(StandardCharsets.UTF_8));
    }

    public static String sha256(byte[] value) {
        try {
            return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(value));
        } catch (NoSuchAlgorithmException error) {
            throw new IllegalStateException("SHA-256 is unavailable", error);
        }
    }

    public interface Backend extends AutoCloseable {

        BackendIdentity prepare(List<CorpusDocument> corpus, String corpusSha256);

        QueryRankings search(String question, RunOptions options);

        @Override
        default void close() {
        }
    }

    public record RunOptions(
            int branchDepth,
            int fusionLimit,
            int clientRrfKOneBased,
            int qdrantRrfKZeroBased) {

        public RunOptions {
            if (branchDepth <= 0 || fusionLimit != branchDepth * 2) {
                throw new IllegalArgumentException("Fusion limit must equal twice the positive branch depth");
            }
            if (clientRrfKOneBased <= 0
                    || qdrantRrfKZeroBased != clientRrfKOneBased + 1) {
                throw new IllegalArgumentException(
                        "Qdrant zero-based k must equal the client one-based k plus one");
            }
        }
    }

    public record QueryRankings(
            List<RetrievedDoc> dense,
            List<RetrievedDoc> sparse,
            List<RetrievedDoc> clientRrf,
            List<RetrievedDoc> serverRrf) {

        public QueryRankings {
            dense = List.copyOf(dense);
            sparse = List.copyOf(sparse);
            clientRrf = List.copyOf(clientRrf);
            serverRrf = List.copyOf(serverRrf);
        }
    }

    public record EmbeddingIdentity(
            int schemaVersion,
            String model,
            String revision,
            String pooling,
            int dimensions,
            int maxLength,
            String queryPrefix,
            String documentPrefix,
            String device,
            String healthResponseSha256) {
    }

    public record DenseParameters(String vectorName, int dimensions, String distance, boolean exact) {
    }

    public record SparseParameters(
            String vectorName,
            String model,
            String language,
            String tokenizer,
            double k,
            double b,
            String averageLengthMethod,
            double averageLength,
            String modifier) {
    }

    public record QdrantIdentity(
            String serverVersion,
            String serverCommit,
            String javaClientVersion,
            String collection,
            String collectionCreationPolicy,
            String pointIdMapping,
            String collectionIdentitySha256) {
    }

    public record BackendIdentity(
            EmbeddingIdentity embedding,
            DenseParameters dense,
            SparseParameters sparse,
            QdrantIdentity qdrant,
            int indexEmbeddingRequests,
            int indexEmbeddingInputs) {
    }

    public record RankedCandidate(int rank, String docId, double score) {
    }

    public record ParityResult(
            boolean identicalOrder,
            boolean sameSet,
            boolean scoreEquivalent,
            boolean tieOnlyDifference,
            int firstDifferenceRank,
            double maxScoreDelta) {
    }

    public record RawRankingRow(
            String qid,
            List<RankedCandidate> dense,
            List<RankedCandidate> nativeSparse,
            List<RankedCandidate> clientRrf,
            List<RankedCandidate> serverRrf,
            ParityResult parity) {

        public RawRankingRow {
            dense = List.copyOf(dense);
            nativeSparse = List.copyOf(nativeSparse);
            clientRrf = List.copyOf(clientRrf);
            serverRrf = List.copyOf(serverRrf);
        }
    }

    public record InputArtifact(String filename, String sha256) {
    }

    public record OutputArtifact(String filename, String sha256) {
    }

    public record RrfContract(
            int clientKOneBased,
            int qdrantKZeroBased,
            String clientFormula,
            String qdrantFormula,
            String clientTieBreak,
            String qdrantTiePolicy,
            double scoreTolerance) {
    }

    public record ParitySummary(
            int questionCount,
            int identicalOrderCount,
            int sameSetCount,
            int scoreEquivalentCount,
            int tieOnlyDifferenceCount,
            int nonTieDifferenceCount,
            double maximumScoreDelta) {
    }

    public record RunManifest(
            int schemaVersion,
            String evalVersion,
            String createdAtUtc,
            String sourceEvalVersion,
            int questionCount,
            int corpusDocumentCount,
            InputArtifact questions,
            InputArtifact snapshotManifest,
            InputArtifact corpus,
            String questionProjectionSha256,
            String queryProjection,
            RunOptions search,
            BackendIdentity backend,
            RrfContract rrf,
            ParitySummary parity,
            OutputArtifact rawRankings) {
    }

    private record Question(String qid, String question) {
    }

    private record FrozenQuestions(
            List<Question> questions,
            Path questionsPath,
            String questionsSha256,
            Path snapshotManifestPath,
            String snapshotManifestSha256,
            String expectedCorpusSha256,
            String sourceEvalVersion,
            String questionProjectionSha256) {
    }

    private record CorpusInput(
            List<CorpusDocument> documents,
            Set<String> documentIds,
            Path path,
            String sha256) {
    }

    private static final class ParityAccumulator {

        int questionCount;
        int identicalOrderCount;
        int sameSetCount;
        int scoreEquivalentCount;
        int tieOnlyDifferenceCount;
        int nonTieDifferenceCount;
        double maximumScoreDelta;

        void add(ParityResult result) {
            questionCount++;
            identicalOrderCount += result.identicalOrder() ? 1 : 0;
            sameSetCount += result.sameSet() ? 1 : 0;
            scoreEquivalentCount += result.scoreEquivalent() ? 1 : 0;
            tieOnlyDifferenceCount += result.tieOnlyDifference() ? 1 : 0;
            nonTieDifferenceCount += !result.identicalOrder() && !result.tieOnlyDifference() ? 1 : 0;
            maximumScoreDelta = Math.max(maximumScoreDelta, result.maxScoreDelta());
        }

        ParitySummary summary() {
            return new ParitySummary(
                    questionCount,
                    identicalOrderCount,
                    sameSetCount,
                    scoreEquivalentCount,
                    tieOnlyDifferenceCount,
                    nonTieDifferenceCount,
                    maximumScoreDelta);
        }
    }

    record CliArguments(
            Path questions,
            Path corpus,
            Path output,
            String embeddingBaseUrl,
            String expectedDevice,
            String qdrantHost,
            int qdrantGrpcPort,
            int qdrantRestPort,
            String collection,
            int branchDepth,
            int clientRrfK,
            int upsertBatchSize,
            boolean help) {

        static CliArguments parse(String[] args) {
            Path questions = null;
            Path corpus = null;
            Path output = null;
            String embeddingBaseUrl = environment(
                    "RAGVIEW_EMBEDDING_BASE_URL", "http://127.0.0.1:8077");
            String expectedDevice = environment("RAGVIEW_EXPECTED_DEVICE", "cuda");
            String qdrantHost = environment("RAGVIEW_QDRANT_HOST", "127.0.0.1");
            int qdrantGrpcPort = positiveInt(
                    environment("RAGVIEW_QDRANT_GRPC_PORT", "6334"), "qdrant-grpc-port");
            int qdrantRestPort = positiveInt(
                    environment("RAGVIEW_QDRANT_REST_PORT", "6333"), "qdrant-rest-port");
            String collection = null;
            int branchDepth = DEFAULT_BRANCH_DEPTH;
            int clientRrfK = DEFAULT_CLIENT_RRF_K_ONE_BASED;
            int upsertBatchSize = 64;
            boolean help = false;

            for (int index = 0; index < args.length; index++) {
                String argument = args[index];
                if ("--help".equals(argument) || "-h".equals(argument)) {
                    help = true;
                } else if (argument.startsWith("--questions=")) {
                    questions = Path.of(argument.substring("--questions=".length()));
                } else if ("--questions".equals(argument)) {
                    questions = Path.of(requireValue(args, ++index, argument));
                } else if (argument.startsWith("--corpus=")) {
                    corpus = Path.of(argument.substring("--corpus=".length()));
                } else if ("--corpus".equals(argument)) {
                    corpus = Path.of(requireValue(args, ++index, argument));
                } else if (argument.startsWith("--output=")) {
                    output = Path.of(argument.substring("--output=".length()));
                } else if ("--output".equals(argument)) {
                    output = Path.of(requireValue(args, ++index, argument));
                } else if (argument.startsWith("--embedding-base-url=")) {
                    embeddingBaseUrl = argument.substring("--embedding-base-url=".length());
                } else if ("--embedding-base-url".equals(argument)) {
                    embeddingBaseUrl = requireValue(args, ++index, argument);
                } else if (argument.startsWith("--expected-device=")) {
                    expectedDevice = argument.substring("--expected-device=".length());
                } else if ("--expected-device".equals(argument)) {
                    expectedDevice = requireValue(args, ++index, argument);
                } else if (argument.startsWith("--qdrant-host=")) {
                    qdrantHost = argument.substring("--qdrant-host=".length());
                } else if ("--qdrant-host".equals(argument)) {
                    qdrantHost = requireValue(args, ++index, argument);
                } else if (argument.startsWith("--qdrant-grpc-port=")) {
                    qdrantGrpcPort = positiveInt(
                            argument.substring("--qdrant-grpc-port=".length()), argument);
                } else if ("--qdrant-grpc-port".equals(argument)) {
                    qdrantGrpcPort = positiveInt(requireValue(args, ++index, argument), argument);
                } else if (argument.startsWith("--qdrant-rest-port=")) {
                    qdrantRestPort = positiveInt(
                            argument.substring("--qdrant-rest-port=".length()), argument);
                } else if ("--qdrant-rest-port".equals(argument)) {
                    qdrantRestPort = positiveInt(requireValue(args, ++index, argument), argument);
                } else if (argument.startsWith("--collection=")) {
                    collection = argument.substring("--collection=".length());
                } else if ("--collection".equals(argument)) {
                    collection = requireValue(args, ++index, argument);
                } else if (argument.startsWith("--branch-depth=")) {
                    branchDepth = positiveInt(argument.substring("--branch-depth=".length()), argument);
                } else if ("--branch-depth".equals(argument)) {
                    branchDepth = positiveInt(requireValue(args, ++index, argument), argument);
                } else if (argument.startsWith("--client-rrf-k=")) {
                    clientRrfK = positiveInt(argument.substring("--client-rrf-k=".length()), argument);
                } else if ("--client-rrf-k".equals(argument)) {
                    clientRrfK = positiveInt(requireValue(args, ++index, argument), argument);
                } else if (argument.startsWith("--upsert-batch-size=")) {
                    upsertBatchSize = positiveInt(
                            argument.substring("--upsert-batch-size=".length()), argument);
                } else if ("--upsert-batch-size".equals(argument)) {
                    upsertBatchSize = positiveInt(requireValue(args, ++index, argument), argument);
                } else {
                    throw new IllegalArgumentException("Unknown argument: " + argument);
                }
            }
            if (!help && (questions == null || corpus == null || output == null || collection == null)) {
                throw new IllegalArgumentException(
                        "--questions, --corpus, --output and --collection are required");
            }
            if (embeddingBaseUrl.isBlank()
                    || expectedDevice.isBlank()
                    || qdrantHost.isBlank()
                    || (collection != null && collection.isBlank())) {
                throw new IllegalArgumentException("Runtime endpoints and collection must not be blank");
            }
            return new CliArguments(
                    questions,
                    corpus,
                    output,
                    embeddingBaseUrl,
                    expectedDevice,
                    qdrantHost,
                    qdrantGrpcPort,
                    qdrantRestPort,
                    collection,
                    branchDepth,
                    clientRrfK,
                    upsertBatchSize,
                    help);
        }

        private static String requireValue(String[] args, int index, String option) {
            if (index >= args.length || args[index].startsWith("--")) {
                throw new IllegalArgumentException("Missing value for " + option);
            }
            return args[index];
        }

        private static int positiveInt(String value, String option) {
            try {
                int parsed = Integer.parseInt(value);
                if (parsed <= 0 || parsed > 65535 && option.contains("port")) {
                    throw new IllegalArgumentException(option + " must be a positive valid value");
                }
                return parsed;
            } catch (NumberFormatException error) {
                throw new IllegalArgumentException(option + " must be an integer", error);
            }
        }

        private static String environment(String name, String fallback) {
            String value = System.getenv(name);
            return value == null || value.isBlank() ? fallback : value;
        }
    }
}
