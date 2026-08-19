package ru.brombin.ragview.eval;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.json.JsonMapper;
import ru.brombin.ragview.eval.FnsFaqQdrantHybridParityRunner.BackendIdentity;
import ru.brombin.ragview.eval.FnsFaqQdrantHybridParityRunner.ParityResult;
import ru.brombin.ragview.eval.FnsFaqQdrantHybridParityRunner.ParitySummary;
import ru.brombin.ragview.eval.FnsFaqQdrantHybridParityRunner.RankedCandidate;
import ru.brombin.ragview.eval.FnsFaqQdrantHybridParityRunner.RawRankingRow;
import ru.brombin.ragview.eval.FnsFaqQdrantHybridParityRunner.RunManifest;
import ru.brombin.ragview.strategy.RetrievedDoc;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Set;

public final class QdrantRepeatabilityAnalysis {

    public static final String EVAL_VERSION =
            "tax-eval-v3-qdrant-1.19-native-hybrid-rrf-repeatability-v1";

    private final ObjectMapper objectMapper;

    public QdrantRepeatabilityAnalysis(ObjectMapper objectMapper) {
        if (objectMapper == null) {
            throw new IllegalArgumentException("Object mapper is required");
        }
        this.objectMapper = objectMapper;
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 3) {
            System.out.println("Usage: QdrantRepeatabilityAnalysis "
                    + "PRIMARY_MANIFEST REPEAT_MANIFEST OUTPUT_JSON");
            return;
        }
        ObjectMapper objectMapper = JsonMapper.builder()
                .enable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES)
                .enable(SerializationFeature.INDENT_OUTPUT)
                .build();
        new QdrantRepeatabilityAnalysis(objectMapper).write(
                Path.of(args[0]), Path.of(args[1]), Path.of(args[2]));
    }

    public Summary write(Path primaryManifestPath, Path repeatManifestPath, Path outputPath)
            throws IOException {
        Summary summary = analyze(primaryManifestPath, repeatManifestPath);
        if (outputPath == null) {
            throw new IllegalArgumentException("Output path is required");
        }
        Path output = outputPath.toAbsolutePath().normalize();
        Path parent = output.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        String json = objectMapper.writerWithDefaultPrettyPrinter().writeValueAsString(summary);
        Files.writeString(output, json + "\n", StandardCharsets.UTF_8);
        return summary;
    }

    public Summary analyze(Path primaryManifestPath, Path repeatManifestPath) throws IOException {
        RunInput primary = readRun(primaryManifestPath, "primary");
        RunInput repeat = readRun(repeatManifestPath, "repeat");
        validateSameExperiment(primary.manifest(), repeat.manifest());
        if (!primary.rows().keySet().equals(repeat.rows().keySet())) {
            throw new IllegalArgumentException("Primary and repeat qids differ");
        }

        RepeatabilityAccumulator repeatability = new RepeatabilityAccumulator();
        PrimaryParityAccumulator primaryParity = new PrimaryParityAccumulator();
        for (String qid : primary.rows().keySet()) {
            RawRankingRow first = primary.rows().get(qid);
            RawRankingRow second = repeat.rows().get(qid);
            repeatability.add(first, second);
            primaryParity.add(first);
        }
        int questionCount = primary.manifest().questionCount();
        Repeatability repeatabilitySummary = repeatability.summary(questionCount);
        PrimaryClientServerParity paritySummary = primaryParity.summary(questionCount);
        validateInvariants(questionCount, repeatabilitySummary);

        return new Summary(
                1,
                EVAL_VERSION,
                new Inputs(primary.artifacts(), repeat.artifacts()),
                questionCount,
                paritySummary,
                repeatabilitySummary,
                new Publication(false, false, false));
    }

    private RunInput readRun(Path manifestPath, String label) throws IOException {
        Path manifestFile = requireRegularFile(manifestPath, label + " manifest");
        byte[] manifestBytes = Files.readAllBytes(manifestFile);
        RunManifest manifest = objectMapper.readValue(manifestBytes, RunManifest.class);
        if (manifest.schemaVersion() != 1
                || manifest.questionCount() <= 0
                || manifest.rawRankings() == null
                || manifest.rawRankings().filename() == null
                || manifest.rawRankings().filename().isBlank()) {
            throw new IllegalArgumentException("Invalid " + label + " manifest");
        }
        Path rawPath = requireRegularFile(
                manifestFile.resolveSibling(manifest.rawRankings().filename()), label + " raw rankings");
        byte[] rawBytes = Files.readAllBytes(rawPath);
        String rawSha256 = FnsFaqQdrantHybridParityRunner.sha256(rawBytes);
        if (!rawSha256.equals(manifest.rawRankings().sha256())) {
            throw new IllegalArgumentException(label + " raw rankings do not match the manifest");
        }
        LinkedHashMap<String, RawRankingRow> rows = readRows(rawBytes, manifest, label);
        validateManifestParity(manifest, rows.values(), label);
        return new RunInput(
                manifest,
                rows,
                new RunArtifacts(
                        FnsFaqQdrantHybridParityRunner.sha256(manifestBytes),
                        rawSha256));
    }

    private LinkedHashMap<String, RawRankingRow> readRows(
            byte[] rawBytes,
            RunManifest manifest,
            String label) throws IOException {
        LinkedHashMap<String, RawRankingRow> rows = new LinkedHashMap<>();
        String raw = new String(rawBytes, StandardCharsets.UTF_8);
        for (String line : raw.split("\\R")) {
            if (line.isBlank()) {
                continue;
            }
            RawRankingRow row = objectMapper.readValue(line, RawRankingRow.class);
            validateRow(row, manifest.search().branchDepth(), label);
            if (rows.putIfAbsent(row.qid(), row) != null) {
                throw new IllegalArgumentException("Duplicate qid in " + label + " rankings");
            }
        }
        if (rows.size() != manifest.questionCount()) {
            throw new IllegalArgumentException(label + " rankings count does not match the manifest");
        }
        return rows;
    }

    private static void validateRow(RawRankingRow row, int branchDepth, String label) {
        if (row == null || row.qid() == null || row.qid().isBlank() || row.parity() == null) {
            throw new IllegalArgumentException("Invalid row in " + label + " rankings");
        }
        validateRanking(row.dense(), branchDepth, label + " dense");
        validateRanking(row.nativeSparse(), branchDepth, label + " native sparse");
        validateRanking(row.clientRrf(), -1, label + " client RRF");
        validateRanking(row.serverRrf(), row.clientRrf().size(), label + " server RRF");

        Set<String> branchUnion = new HashSet<>();
        row.dense().forEach(candidate -> branchUnion.add(candidate.docId()));
        row.nativeSparse().forEach(candidate -> branchUnion.add(candidate.docId()));
        if (!branchUnion.equals(docIdSet(row.clientRrf()))
                || !branchUnion.equals(docIdSet(row.serverRrf()))) {
            throw new IllegalArgumentException("RRF ranking differs from the branch union in " + label);
        }
        ParityResult recomputed = FnsFaqQdrantHybridParityRunner.compare(
                retrieved(row.clientRrf()), retrieved(row.serverRrf()));
        if (!recomputed.equals(row.parity())) {
            throw new IllegalArgumentException("Stored parity differs from rankings in " + label);
        }
    }

    private static void validateRanking(List<RankedCandidate> ranking, int expectedSize, String label) {
        if (ranking == null || ranking.isEmpty() || expectedSize >= 0 && ranking.size() != expectedSize) {
            throw new IllegalArgumentException("Invalid " + label + " ranking size");
        }
        Set<String> docIds = new HashSet<>();
        for (int index = 0; index < ranking.size(); index++) {
            RankedCandidate candidate = ranking.get(index);
            if (candidate == null
                    || candidate.rank() != index + 1
                    || candidate.docId() == null
                    || candidate.docId().isBlank()
                    || !docIds.add(candidate.docId())
                    || !Double.isFinite(candidate.score())) {
                throw new IllegalArgumentException("Invalid " + label + " ranking");
            }
        }
    }

    private static void validateManifestParity(
            RunManifest manifest,
            Iterable<RawRankingRow> rows,
            String label) {
        int identicalOrder = 0;
        int sameSet = 0;
        int scoreEquivalent = 0;
        int tieOnlyDifference = 0;
        int nonTieDifference = 0;
        double maximumScoreDelta = 0;
        for (RawRankingRow row : rows) {
            ParityResult parity = row.parity();
            identicalOrder += parity.identicalOrder() ? 1 : 0;
            sameSet += parity.sameSet() ? 1 : 0;
            scoreEquivalent += parity.scoreEquivalent() ? 1 : 0;
            tieOnlyDifference += parity.tieOnlyDifference() ? 1 : 0;
            nonTieDifference += !parity.identicalOrder() && !parity.tieOnlyDifference() ? 1 : 0;
            maximumScoreDelta = Math.max(maximumScoreDelta, parity.maxScoreDelta());
        }
        ParitySummary recomputed = new ParitySummary(
                manifest.questionCount(),
                identicalOrder,
                sameSet,
                scoreEquivalent,
                tieOnlyDifference,
                nonTieDifference,
                maximumScoreDelta);
        if (!recomputed.equals(manifest.parity())) {
            throw new IllegalArgumentException(label + " parity summary does not match raw rankings");
        }
    }

    private static void validateSameExperiment(RunManifest primary, RunManifest repeat) {
        if (!primary.evalVersion().equals(repeat.evalVersion())
                || !primary.sourceEvalVersion().equals(repeat.sourceEvalVersion())
                || primary.questionCount() != repeat.questionCount()
                || primary.corpusDocumentCount() != repeat.corpusDocumentCount()
                || !primary.questions().equals(repeat.questions())
                || !primary.snapshotManifest().equals(repeat.snapshotManifest())
                || !primary.corpus().equals(repeat.corpus())
                || !primary.questionProjectionSha256().equals(repeat.questionProjectionSha256())
                || !primary.queryProjection().equals(repeat.queryProjection())
                || !primary.search().equals(repeat.search())
                || !sameBackendIdentity(primary.backend(), repeat.backend())
                || !primary.rrf().equals(repeat.rrf())) {
            throw new IllegalArgumentException("Primary and repeat manifests describe different experiments");
        }
    }

    private static boolean sameBackendIdentity(BackendIdentity first, BackendIdentity second) {
        return first.embedding().equals(second.embedding())
                && first.dense().equals(second.dense())
                && first.sparse().equals(second.sparse())
                && first.qdrant().equals(second.qdrant());
    }

    private static void validateInvariants(int questionCount, Repeatability repeatability) {
        for (CountPair metric : repeatability.metrics()) {
            if (metric.repeatable() < 0
                    || metric.changed() < 0
                    || metric.repeatable() + metric.changed() != questionCount) {
                throw new IllegalStateException("Repeatability counts do not add up to questionCount");
            }
        }
    }

    private static Path requireRegularFile(Path path, String label) {
        if (path == null || !Files.isRegularFile(path)) {
            throw new IllegalArgumentException(label + " file does not exist: " + path);
        }
        return path.toAbsolutePath().normalize();
    }

    private static List<RetrievedDoc> retrieved(List<RankedCandidate> ranking) {
        return ranking.stream()
                .map(candidate -> new RetrievedDoc(candidate.docId(), candidate.score()))
                .toList();
    }

    private static List<String> docIds(List<RankedCandidate> ranking, int limit) {
        return ranking.stream().limit(limit).map(RankedCandidate::docId).toList();
    }

    private static Set<String> docIdSet(List<RankedCandidate> ranking) {
        return Set.copyOf(docIds(ranking, ranking.size()));
    }

    private static boolean sameOrder(List<RankedCandidate> first, List<RankedCandidate> second, int limit) {
        return docIds(first, limit).equals(docIds(second, limit));
    }

    private static boolean sameSet(List<RankedCandidate> first, List<RankedCandidate> second, int limit) {
        return Set.copyOf(docIds(first, limit)).equals(Set.copyOf(docIds(second, limit)));
    }

    private record RunInput(
            RunManifest manifest,
            LinkedHashMap<String, RawRankingRow> rows,
            RunArtifacts artifacts) {
    }

    public record RunArtifacts(String manifestSha256, String rawRankingsSha256) {
    }

    public record Inputs(RunArtifacts primary, RunArtifacts repeat) {
    }

    public record CountPair(int repeatable, int changed) {

        static CountPair of(int repeatable, int questionCount) {
            return new CountPair(repeatable, questionCount - repeatable);
        }
    }

    public record PrimaryClientServerParity(
            int sameFullUnionSetCount,
            int sameTop50SetCount,
            int sameTop10SetCount,
            int scoreEquivalentCount,
            int nonTieDifferenceCount) {
    }

    public record Repeatability(
            CountPair denseFullOrder,
            CountPair nativeSparseFullOrder,
            CountPair clientRrfFullOrder,
            CountPair serverRrfFullOrder,
            CountPair serverRrfFullSet,
            CountPair serverRrfTop50Order,
            CountPair serverRrfTop50Set,
            CountPair serverRrfTop10Order,
            CountPair serverRrfTop10Set) {

        List<CountPair> metrics() {
            return List.of(
                    denseFullOrder,
                    nativeSparseFullOrder,
                    clientRrfFullOrder,
                    serverRrfFullOrder,
                    serverRrfFullSet,
                    serverRrfTop50Order,
                    serverRrfTop50Set,
                    serverRrfTop10Order,
                    serverRrfTop10Set);
        }
    }

    public record Publication(
            boolean containsQuestionOrAnswerText,
            boolean containsCorpusText,
            boolean containsDocumentOrQuestionIdentifiers) {
    }

    public record Summary(
            int schemaVersion,
            String evalVersion,
            Inputs inputs,
            int questionCount,
            PrimaryClientServerParity primaryClientServerParity,
            Repeatability repeatability,
            Publication publication) {
    }

    private static final class PrimaryParityAccumulator {

        int sameFullSet;
        int sameTop50Set;
        int sameTop10Set;
        int scoreEquivalent;
        int nonTieDifference;

        void add(RawRankingRow row) {
            sameFullSet += sameSet(row.clientRrf(), row.serverRrf(), Integer.MAX_VALUE) ? 1 : 0;
            sameTop50Set += sameSet(row.clientRrf(), row.serverRrf(), 50) ? 1 : 0;
            sameTop10Set += sameSet(row.clientRrf(), row.serverRrf(), 10) ? 1 : 0;
            scoreEquivalent += row.parity().scoreEquivalent() ? 1 : 0;
            nonTieDifference += !row.parity().identicalOrder() && !row.parity().tieOnlyDifference() ? 1 : 0;
        }

        PrimaryClientServerParity summary(int questionCount) {
            if (sameFullSet > questionCount
                    || sameTop50Set > questionCount
                    || sameTop10Set > questionCount
                    || scoreEquivalent > questionCount
                    || nonTieDifference > questionCount) {
                throw new IllegalStateException("Primary parity counts exceed questionCount");
            }
            return new PrimaryClientServerParity(
                    sameFullSet,
                    sameTop50Set,
                    sameTop10Set,
                    scoreEquivalent,
                    nonTieDifference);
        }
    }

    private static final class RepeatabilityAccumulator {

        int denseFullOrder;
        int nativeSparseFullOrder;
        int clientRrfFullOrder;
        int serverRrfFullOrder;
        int serverRrfFullSet;
        int serverRrfTop50Order;
        int serverRrfTop50Set;
        int serverRrfTop10Order;
        int serverRrfTop10Set;
        int questionCount;

        void add(RawRankingRow primary, RawRankingRow repeat) {
            questionCount++;
            denseFullOrder += sameOrder(primary.dense(), repeat.dense(), Integer.MAX_VALUE) ? 1 : 0;
            nativeSparseFullOrder += sameOrder(
                    primary.nativeSparse(), repeat.nativeSparse(), Integer.MAX_VALUE) ? 1 : 0;
            clientRrfFullOrder += sameOrder(
                    primary.clientRrf(), repeat.clientRrf(), Integer.MAX_VALUE) ? 1 : 0;
            serverRrfFullOrder += sameOrder(
                    primary.serverRrf(), repeat.serverRrf(), Integer.MAX_VALUE) ? 1 : 0;
            serverRrfFullSet += sameSet(
                    primary.serverRrf(), repeat.serverRrf(), Integer.MAX_VALUE) ? 1 : 0;
            serverRrfTop50Order += sameOrder(primary.serverRrf(), repeat.serverRrf(), 50) ? 1 : 0;
            serverRrfTop50Set += sameSet(primary.serverRrf(), repeat.serverRrf(), 50) ? 1 : 0;
            serverRrfTop10Order += sameOrder(primary.serverRrf(), repeat.serverRrf(), 10) ? 1 : 0;
            serverRrfTop10Set += sameSet(primary.serverRrf(), repeat.serverRrf(), 10) ? 1 : 0;
        }

        Repeatability summary(int expectedQuestionCount) {
            if (questionCount != expectedQuestionCount) {
                throw new IllegalStateException("Repeatability row count does not match questionCount");
            }
            return new Repeatability(
                    CountPair.of(denseFullOrder, questionCount),
                    CountPair.of(nativeSparseFullOrder, questionCount),
                    CountPair.of(clientRrfFullOrder, questionCount),
                    CountPair.of(serverRrfFullOrder, questionCount),
                    CountPair.of(serverRrfFullSet, questionCount),
                    CountPair.of(serverRrfTop50Order, questionCount),
                    CountPair.of(serverRrfTop50Set, questionCount),
                    CountPair.of(serverRrfTop10Order, questionCount),
                    CountPair.of(serverRrfTop10Set, questionCount));
        }
    }
}
