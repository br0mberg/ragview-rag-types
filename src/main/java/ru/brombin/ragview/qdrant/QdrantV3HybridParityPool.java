package ru.brombin.ragview.qdrant;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.qdrant.client.QdrantClient;
import io.qdrant.client.QdrantGrpcClient;
import io.qdrant.client.grpc.Collections.CreateCollection;
import io.qdrant.client.grpc.Collections.Distance;
import io.qdrant.client.grpc.Collections.Modifier;
import io.qdrant.client.grpc.Collections.SparseVectorConfig;
import io.qdrant.client.grpc.Collections.SparseVectorParams;
import io.qdrant.client.grpc.Collections.VectorParams;
import io.qdrant.client.grpc.Collections.VectorParamsMap;
import io.qdrant.client.grpc.Collections.VectorsConfig;
import io.qdrant.client.grpc.Points.PointStruct;
import io.qdrant.client.grpc.Points.QueryPoints;
import io.qdrant.client.grpc.Points.ReadConsistency;
import io.qdrant.client.grpc.Points.ReadConsistencyType;
import io.qdrant.client.grpc.Points.RetrievedPoint;
import io.qdrant.client.grpc.Points.ScoredPoint;
import io.qdrant.client.grpc.Points.UpsertPoints;
import lombok.AccessLevel;
import lombok.experimental.FieldDefaults;
import lombok.experimental.NonFinal;
import ru.brombin.ragview.corpus.CorpusDocument;
import ru.brombin.ragview.eval.FnsFaqQdrantHybridParityRunner;
import ru.brombin.ragview.eval.FnsFaqQdrantHybridParityRunner.BackendIdentity;
import ru.brombin.ragview.eval.FnsFaqQdrantHybridParityRunner.DenseParameters;
import ru.brombin.ragview.eval.FnsFaqQdrantHybridParityRunner.EmbeddingIdentity;
import ru.brombin.ragview.eval.FnsFaqQdrantHybridParityRunner.QdrantIdentity;
import ru.brombin.ragview.eval.FnsFaqQdrantHybridParityRunner.QueryRankings;
import ru.brombin.ragview.eval.FnsFaqQdrantHybridParityRunner.RunOptions;
import ru.brombin.ragview.eval.FnsFaqQdrantHybridParityRunner.SparseParameters;
import ru.brombin.ragview.retriever.AsymmetricEmbedding;
import ru.brombin.ragview.retriever.ReciprocalRankFusion;
import ru.brombin.ragview.strategy.RetrievedDoc;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.regex.Pattern;

import static io.qdrant.client.PointIdFactory.id;
import static io.qdrant.client.ValueFactory.value;
import static io.qdrant.client.VectorFactory.vector;
import static io.qdrant.client.VectorsFactory.namedVectors;

@FieldDefaults(level = AccessLevel.PRIVATE, makeFinal = true)
public final class QdrantV3HybridParityPool implements FnsFaqQdrantHybridParityRunner.Backend {

    public static final String CLIENT_VERSION = "1.19.0";
    public static final String REQUIRED_SERVER_VERSION = "1.19.0";
    public static final String COLLECTION_PREFIX = "ragview_benchmark_v3_hybrid_";
    public static final String COLLECTION_POLICY =
            "create-if-absent; verify complete immutable identity if present; never delete";
    public static final String POINT_ID_MAPPING = "uuid-v3(corpusSha256\\0docId)";
    public static final String AVERAGE_LENGTH_METHOD =
            "unicode-whitespace-v1:String.strip+Pattern.UNICODE_CHARACTER_CLASS(\\s+)";

    static final String CORPUS_SHA_PAYLOAD = "corpus_sha256";
    static final String DOC_ID_PAYLOAD = "doc_id";
    static final String IDENTITY_SHA_PAYLOAD = "collection_identity_sha256";
    static final int IDENTITY_BATCH_SIZE = 128;
    static final Pattern UNICODE_WHITESPACE = Pattern.compile(
            "\\s+", Pattern.UNICODE_CHARACTER_CLASS);
    static final int EMBEDDING_BATCH_SIZE = 64;

    ObjectMapper objectMapper;
    AsymmetricEmbedding embeddingModel;
    EmbeddingIdentity embeddingIdentity;
    QdrantClient client;
    HttpClient http;
    String restBaseUrl;
    String collection;
    int upsertBatchSize;
    String serverVersion;
    String serverCommit;
    Set<String> corpusDocIds = new HashSet<>();

    @NonFinal
    double averageLength;
    @NonFinal
    String corpusSha256;
    @NonFinal
    String collectionIdentitySha256;
    @NonFinal
    int indexEmbeddingRequests;
    @NonFinal
    int indexEmbeddingInputs;

    public QdrantV3HybridParityPool(
            ObjectMapper objectMapper,
            AsymmetricEmbedding embeddingModel,
            EmbeddingIdentity embeddingIdentity,
            String host,
            int grpcPort,
            int restPort,
            String collection,
            int upsertBatchSize) {
        if (objectMapper == null || embeddingModel == null || embeddingIdentity == null) {
            throw new IllegalArgumentException("Mapper, embedding model and identity are required");
        }
        if (host == null || host.isBlank()
                || grpcPort <= 0 || grpcPort > 65535
                || restPort <= 0 || restPort > 65535) {
            throw new IllegalArgumentException("Qdrant endpoint is invalid");
        }
        if (collection == null || !collection.startsWith(COLLECTION_PREFIX)) {
            throw new IllegalArgumentException(
                    "Qdrant collection must start with " + COLLECTION_PREFIX);
        }
        if (upsertBatchSize <= 0) {
            throw new IllegalArgumentException("Qdrant upsert batch size must be positive");
        }
        this.objectMapper = objectMapper;
        this.embeddingModel = embeddingModel;
        this.embeddingIdentity = embeddingIdentity;
        this.collection = collection;
        this.upsertBatchSize = upsertBatchSize;
        this.http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();
        this.restBaseUrl = "http://" + host + ":" + restPort;
        this.client = new QdrantClient(
                QdrantGrpcClient.newBuilder(host, grpcPort, false, false)
                        .withTimeout(Duration.ofMinutes(5))
                        .build());
        this.serverVersion = await(client.healthCheckAsync()).getVersion();
        ServerBuild serverBuild = readServerBuild();
        if (!REQUIRED_SERVER_VERSION.equals(serverVersion)
                || !REQUIRED_SERVER_VERSION.equals(serverBuild.version())) {
            close();
            throw new IllegalStateException(
                    "Qdrant 1.19.0 is required, got " + serverVersion + "/" + serverBuild.version());
        }
        this.serverCommit = serverBuild.commit();
    }

    @Override
    public BackendIdentity prepare(List<CorpusDocument> corpus, String expectedCorpusSha256) {
        validateCorpus(corpus, expectedCorpusSha256);
        corpusSha256 = expectedCorpusSha256;
        averageLength = corpus.stream()
                .map(CorpusDocument::indexableText)
                .mapToInt(QdrantV3HybridParityPool::whitespaceTokenCount)
                .average()
                .orElseThrow();
        collectionIdentitySha256 = FnsFaqQdrantHybridParityRunner.sha256(
                identityMaterial(corpus.size(), corpusSha256, averageLength));
        corpusDocIds.clear();
        corpus.forEach(document -> corpusDocIds.add(document.id()));

        boolean exists = await(client.collectionExistsAsync(collection));
        if (exists) {
            verifyExistingCollection(corpus);
        } else {
            createCollection();
            index(corpus);
            verifyExistingCollection(corpus);
        }
        return identity();
    }

    @Override
    public QueryRankings search(String question, RunOptions options) {
        requirePrepared(question, options);
        float[] queryVector = embeddingModel.embedQuery(question);
        validateVector(queryVector, embeddingIdentity.dimensions());

        QueryPoints denseRequest = QdrantRequestFactory.dense(
                collection, queryVector, options.branchDepth());
        QueryPoints sparseRequest = QdrantRequestFactory.bm25(
                collection, question, options.branchDepth(), averageLength);
        QueryPoints serverRrfRequest = QdrantRequestFactory.hybrid(
                collection,
                queryVector,
                question,
                options.branchDepth(),
                options.fusionLimit(),
                options.clientRrfKOneBased(),
                averageLength);

        Future<List<ScoredPoint>> denseFuture = client.queryAsync(denseRequest);
        Future<List<ScoredPoint>> sparseFuture = client.queryAsync(sparseRequest);
        Future<List<ScoredPoint>> serverFuture = client.queryAsync(serverRrfRequest);
        List<RetrievedDoc> dense = convert(await(denseFuture));
        List<RetrievedDoc> sparse = convert(await(sparseFuture));
        List<RetrievedDoc> clientRrf = ReciprocalRankFusion.fuse(
                options.fusionLimit(), options.clientRrfKOneBased(), dense, sparse);
        List<RetrievedDoc> serverRrf = convert(await(serverFuture));
        return new QueryRankings(dense, sparse, clientRrf, serverRrf);
    }

    private void requirePrepared(String question, RunOptions options) {
        if (collectionIdentitySha256 == null) {
            throw new IllegalStateException("Qdrant hybrid pool is not prepared");
        }
        if (question == null || question.isBlank() || options == null) {
            throw new IllegalArgumentException("Question and run options are required");
        }
    }

    private void createCollection() {
        VectorParams dense = VectorParams.newBuilder()
                .setSize(embeddingIdentity.dimensions())
                .setDistance(Distance.Cosine)
                .build();
        CreateCollection request = CreateCollection.newBuilder()
                .setCollectionName(collection)
                .setVectorsConfig(VectorsConfig.newBuilder()
                        .setParamsMap(VectorParamsMap.newBuilder()
                                .putMap(QdrantRequestFactory.DENSE_VECTOR, dense)
                                .build())
                        .build())
                .setSparseVectorsConfig(SparseVectorConfig.newBuilder()
                        .putMap(QdrantRequestFactory.BM25_VECTOR,
                                SparseVectorParams.newBuilder().setModifier(Modifier.Idf).build())
                        .build())
                .build();
        if (!await(client.createCollectionAsync(request)).getResult()) {
            throw new IllegalStateException("Qdrant did not create collection " + collection);
        }
    }

    private void index(List<CorpusDocument> corpus) {
        for (int start = 0; start < corpus.size(); start += EMBEDDING_BATCH_SIZE) {
            int end = Math.min(start + EMBEDDING_BATCH_SIZE, corpus.size());
            List<CorpusDocument> batch = corpus.subList(start, end);
            List<float[]> vectors = embeddingModel.embedDocuments(
                    batch.stream().map(CorpusDocument::indexableText).toList());
            validateEmbeddingBatch(vectors, batch.size(), embeddingIdentity.dimensions());
            indexEmbeddingRequests++;
            indexEmbeddingInputs += batch.size();

            for (int upsertStart = 0; upsertStart < batch.size(); upsertStart += upsertBatchSize) {
                int upsertEnd = Math.min(upsertStart + upsertBatchSize, batch.size());
                List<PointStruct> points = new ArrayList<>(upsertEnd - upsertStart);
                for (int localIndex = upsertStart; localIndex < upsertEnd; localIndex++) {
                    CorpusDocument document = batch.get(localIndex);
                    points.add(PointStruct.newBuilder()
                            .setId(id(pointId(corpusSha256, document.id())))
                            .setVectors(namedVectors(Map.of(
                                    QdrantRequestFactory.DENSE_VECTOR, vector(vectors.get(localIndex)),
                                    QdrantRequestFactory.BM25_VECTOR,
                                    vector(QdrantRequestFactory.bm25Document(
                                            document.indexableText(), averageLength)))))
                            .putPayload(DOC_ID_PAYLOAD, value(document.id()))
                            .putPayload(CORPUS_SHA_PAYLOAD, value(corpusSha256))
                            .putPayload(IDENTITY_SHA_PAYLOAD, value(collectionIdentitySha256))
                            .build());
                }
                await(client.upsertAsync(UpsertPoints.newBuilder()
                        .setCollectionName(collection)
                        .setWait(true)
                        .addAllPoints(points)
                        .build()));
            }
        }
    }

    private void verifyExistingCollection(List<CorpusDocument> corpus) {
        var info = await(client.getCollectionInfoAsync(collection));
        var params = info.getConfig().getParams();
        if (!params.getVectorsConfig().hasParamsMap()
                || params.getVectorsConfig().getParamsMap().getMapCount() != 1
                || !params.getVectorsConfig().getParamsMap().containsMap(QdrantRequestFactory.DENSE_VECTOR)) {
            throw identityMismatch("missing single named dense vector");
        }
        var dense = params.getVectorsConfig().getParamsMap()
                .getMapOrThrow(QdrantRequestFactory.DENSE_VECTOR);
        if (dense.getSize() != embeddingIdentity.dimensions() || dense.getDistance() != Distance.Cosine) {
            throw identityMismatch("dense vector parameters differ");
        }
        if (!params.hasSparseVectorsConfig()
                || params.getSparseVectorsConfig().getMapCount() != 1
                || !params.getSparseVectorsConfig().containsMap(QdrantRequestFactory.BM25_VECTOR)) {
            throw identityMismatch("missing single native sparse vector");
        }
        var sparse = params.getSparseVectorsConfig().getMapOrThrow(QdrantRequestFactory.BM25_VECTOR);
        if (!sparse.hasModifier() || sparse.getModifier() != Modifier.Idf) {
            throw identityMismatch("native sparse vector does not use IDF");
        }
        long pointCount = await(client.countAsync(collection));
        if (pointCount != corpus.size()) {
            throw identityMismatch("point count is " + pointCount + ", expected " + corpus.size());
        }

        Map<UUID, String> expected = new HashMap<>();
        corpus.forEach(document -> expected.put(pointId(corpusSha256, document.id()), document.id()));
        Map<UUID, StoredIdentity> stored = new HashMap<>();
        List<UUID> ids = List.copyOf(expected.keySet());
        for (int start = 0; start < ids.size(); start += IDENTITY_BATCH_SIZE) {
            List<io.qdrant.client.grpc.Common.PointId> pointIds = ids.subList(
                            start, Math.min(start + IDENTITY_BATCH_SIZE, ids.size()))
                    .stream()
                    .map(io.qdrant.client.PointIdFactory::id)
                    .toList();
            List<RetrievedPoint> points = await(client.retrieveAsync(
                    collection,
                    pointIds,
                    true,
                    false,
                    ReadConsistency.newBuilder().setType(ReadConsistencyType.All).build()));
            for (RetrievedPoint point : points) {
                UUID pointId = UUID.fromString(point.getId().getUuid());
                stored.put(pointId, new StoredIdentity(
                        requirePayload(point.getPayloadMap(), DOC_ID_PAYLOAD),
                        requirePayload(point.getPayloadMap(), CORPUS_SHA_PAYLOAD),
                        requirePayload(point.getPayloadMap(), IDENTITY_SHA_PAYLOAD)));
            }
        }
        validateStoredIdentity(expected, stored, corpusSha256, collectionIdentitySha256);
    }

    static void validateStoredIdentity(
            Map<UUID, String> expected,
            Map<UUID, StoredIdentity> stored,
            String expectedCorpusSha256,
            String expectedIdentitySha256) {
        if (stored.size() != expected.size()) {
            throw new IllegalStateException(
                    "Qdrant collection identity mismatch: retrieved " + stored.size()
                            + " points, expected " + expected.size());
        }
        for (Map.Entry<UUID, String> entry : expected.entrySet()) {
            StoredIdentity actual = stored.get(entry.getKey());
            if (actual == null
                    || !entry.getValue().equals(actual.docId())
                    || !expectedCorpusSha256.equals(actual.corpusSha256())
                    || !expectedIdentitySha256.equals(actual.collectionIdentitySha256())) {
                throw new IllegalStateException(
                        "Qdrant collection identity mismatch at document " + entry.getValue());
            }
        }
    }

    private List<RetrievedDoc> convert(List<ScoredPoint> points) {
        List<RetrievedDoc> result = new ArrayList<>(points.size());
        Set<String> seen = new HashSet<>();
        for (ScoredPoint point : points) {
            String docId = requirePayload(point.getPayloadMap(), DOC_ID_PAYLOAD);
            if (!corpusDocIds.contains(docId) || !seen.add(docId)) {
                throw new IllegalStateException("Qdrant returned an invalid document ID: " + docId);
            }
            if (!Float.isFinite(point.getScore())) {
                throw new IllegalStateException("Qdrant returned a non-finite score");
            }
            result.add(new RetrievedDoc(docId, point.getScore()));
        }
        return List.copyOf(result);
    }

    private BackendIdentity identity() {
        return new BackendIdentity(
                embeddingIdentity,
                new DenseParameters(
                        QdrantRequestFactory.DENSE_VECTOR,
                        embeddingIdentity.dimensions(),
                        "cosine",
                        true),
                new SparseParameters(
                        QdrantRequestFactory.BM25_VECTOR,
                        QdrantRequestFactory.BM25_MODEL,
                        "russian",
                        "multilingual",
                        1.2,
                        0.75,
                        AVERAGE_LENGTH_METHOD,
                        averageLength,
                        "idf"),
                new QdrantIdentity(
                        serverVersion,
                        serverCommit,
                        "qdrant-java-client-" + CLIENT_VERSION,
                        collection,
                        COLLECTION_POLICY,
                        POINT_ID_MAPPING,
                        collectionIdentitySha256),
                indexEmbeddingRequests,
                indexEmbeddingInputs);
    }

    private ServerBuild readServerBuild() {
        try {
            HttpRequest request = HttpRequest.newBuilder(URI.create(restBaseUrl + "/"))
                    .timeout(Duration.ofSeconds(30))
                    .GET()
                    .build();
            HttpResponse<byte[]> response = http.send(request, HttpResponse.BodyHandlers.ofByteArray());
            if (response.statusCode() != 200) {
                throw new IllegalStateException("Qdrant REST health returned HTTP " + response.statusCode());
            }
            JsonNode root = objectMapper.readTree(response.body());
            String version = requireText(root, "version");
            String commit = requireText(root, "commit");
            return new ServerBuild(version, commit);
        } catch (IOException error) {
            throw new IllegalStateException("Cannot read Qdrant server build", error);
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Qdrant REST health was interrupted", error);
        }
    }

    private static String identityMaterial(int count, String sha256, double avgLen) {
        return String.join("\n",
                "qdrant-v3-hybrid-native-bm25",
                "embedding_model=" + FnsFaqQdrantHybridParityRunner.BERTA_MODEL,
                "embedding_revision=" + FnsFaqQdrantHybridParityRunner.BERTA_REVISION,
                "embedding_pooling=" + FnsFaqQdrantHybridParityRunner.BERTA_POOLING,
                "embedding_dimensions=" + FnsFaqQdrantHybridParityRunner.BERTA_DIMENSIONS,
                "dense_vector=" + QdrantRequestFactory.DENSE_VECTOR,
                "dense_distance=cosine",
                "dense_exact_query=true",
                "sparse_model=" + QdrantRequestFactory.BM25_MODEL,
                "sparse_vector=" + QdrantRequestFactory.BM25_VECTOR,
                "language=russian",
                "tokenizer=multilingual",
                "k=1.2",
                "b=0.75",
                "avg_len_method=" + AVERAGE_LENGTH_METHOD,
                "avg_len=" + Double.toHexString(avgLen),
                "modifier=idf",
                "point_id=" + POINT_ID_MAPPING,
                "corpus_sha256=" + sha256,
                "documents=" + count);
    }

    private static void validateCorpus(List<CorpusDocument> corpus, String sha256) {
        if (corpus == null || corpus.isEmpty()) {
            throw new IllegalArgumentException("Corpus must not be empty");
        }
        if (sha256 == null || !sha256.matches("[0-9a-f]{64}")) {
            throw new IllegalArgumentException("Corpus SHA-256 is invalid");
        }
        Set<String> ids = new HashSet<>();
        for (CorpusDocument document : corpus) {
            if (document.id() == null || document.id().isBlank() || !ids.add(document.id())) {
                throw new IllegalArgumentException("Corpus document IDs must be non-blank and unique");
            }
        }
    }

    private static void validateEmbeddingBatch(List<float[]> vectors, int count, int dimensions) {
        if (vectors == null || vectors.size() != count) {
            throw new IllegalStateException("Embedding server returned an invalid batch");
        }
        vectors.forEach(vector -> validateVector(vector, dimensions));
    }

    private static void validateVector(float[] vector, int dimensions) {
        if (vector == null || vector.length != dimensions) {
            throw new IllegalStateException("Embedding vector dimensions differ from the pinned runtime");
        }
        for (float value : vector) {
            if (!Float.isFinite(value)) {
                throw new IllegalStateException("Embedding vector contains a non-finite value");
            }
        }
    }

    static int whitespaceTokenCount(String text) {
        if (text == null) {
            return 0;
        }
        String stripped = text.strip();
        return stripped.isEmpty() ? 0 : UNICODE_WHITESPACE.split(stripped).length;
    }

    static UUID pointId(String corpusSha256, String docId) {
        return UUID.nameUUIDFromBytes(
                (corpusSha256 + "\0" + docId).getBytes(StandardCharsets.UTF_8));
    }

    private static String requirePayload(
            Map<String, io.qdrant.client.grpc.JsonWithInt.Value> payload,
            String key) {
        var value = payload.get(key);
        if (value == null || !value.hasStringValue() || value.getStringValue().isBlank()) {
            throw new IllegalStateException("Qdrant point is missing payload " + key);
        }
        return value.getStringValue();
    }

    private static String requireText(JsonNode node, String field) {
        JsonNode value = node == null ? null : node.get(field);
        if (value == null || !value.isTextual() || value.textValue().isBlank()) {
            throw new IllegalStateException("Qdrant health is missing " + field);
        }
        return value.textValue();
    }

    private static IllegalStateException identityMismatch(String detail) {
        return new IllegalStateException("Qdrant collection identity mismatch: " + detail);
    }

    private static <T> T await(Future<T> future) {
        try {
            return future.get(5, TimeUnit.MINUTES);
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Qdrant request was interrupted", error);
        } catch (ExecutionException error) {
            throw new IllegalStateException("Qdrant request failed", error.getCause());
        } catch (TimeoutException error) {
            throw new IllegalStateException("Qdrant request timed out", error);
        }
    }

    @Override
    public void close() {
        client.close();
    }

    record StoredIdentity(String docId, String corpusSha256, String collectionIdentitySha256) {
    }

    private record ServerBuild(String version, String commit) {
    }
}
