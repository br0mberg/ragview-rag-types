package ru.brombin.ragview.eval;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.json.JsonMapper;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import java.util.function.Predicate;
import java.util.regex.Pattern;

public final class FnsFaqBgeCutoffEvaluator {

    public static final int QUESTION_COUNT = 240;
    public static final int SCORED_POOL = 50;
    public static final int TOP_K = 10;
    public static final List<Integer> REQUIRED_CUTOFFS = List.of(10, 20, 30, 50);
    public static final String EVAL_VERSION = "tax-eval-v3-bge-cutoff-evaluation-v1";
    public static final String BOOTSTRAP_SEED = "ragview-v3-cited-clause-bootstrap-2026-08-16";
    public static final int BOOTSTRAP_RESAMPLES = 10_000;
    public static final String BGE_MODEL = "BAAI/bge-reranker-v2-m3";
    public static final String BGE_REVISION =
            "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e";
    public static final String BGE_PRECISION = "fp16";
    public static final String BGE_DEVICE = "cuda";
    public static final int BGE_BATCH_SIZE = 64;
    public static final int BGE_MAX_LENGTH = 1024;
    public static final String PER_QUERY_FILENAME = "per-query.csv";
    public static final String AGGREGATE_FILENAME = "aggregate.csv";
    public static final String MANIFEST_FILENAME = "manifest.json";
    public static final String MANIFEST_SHA_FILENAME = "MANIFEST.sha256";

    private static final String PRIMARY_STATUS = "eligible_cited_clause";
    private static final String ARTICLE_STATUS = "eligible_article_only_single_chunk";
    private static final String MIXED_STATUS = "eligible_mixed_literal";
    private static final String EXCLUDED_STATUS = "excluded";
    private static final String PRIMARY_SLICE = "primary_cited_clause";
    private static final String ARTICLE_SLICE = "article_only_single_chunk";
    private static final String MIXED_SLICE = "mixed_clause_and_article";
    private static final Pattern SHA256 = Pattern.compile("^[0-9a-f]{64}$");
    private static final Pattern DOC_ID = Pattern.compile(
            "^nk-\\d+-article-\\d+(?:[.-]\\d+)*-chunk-\\d+$");
    private static final Set<String> TARGET_FIELDS = Set.of(
            "qid", "queryGroup", "categoryId", "status", "slice", "reason", "targetDocIds");
    private static final Comparator<Candidate> RERANK_ORDER = Comparator
            .comparingDouble(Candidate::score).reversed()
            .thenComparingInt(Candidate::firstStageRank)
            .thenComparing(Candidate::docId);
    private static final double EPSILON = 1e-12;

    private final ObjectMapper objectMapper;

    public FnsFaqBgeCutoffEvaluator(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
    }

    public static void main(String[] args) throws Exception {
        CliArguments cli = CliArguments.parse(args);
        if (cli.help()) {
            System.out.println("Usage: FnsFaqBgeCutoffEvaluator "
                    + "--bge-manifest PATH --targets-manifest PATH "
                    + "--cutoffs 10,20,30,50 --output PATH");
            return;
        }

        ObjectMapper objectMapper = JsonMapper.builder()
                .enable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES)
                .enable(MapperFeature.SORT_PROPERTIES_ALPHABETICALLY)
                .enable(SerializationFeature.ORDER_MAP_ENTRIES_BY_KEYS)
                .enable(SerializationFeature.INDENT_OUTPUT)
                .build();
        EvaluationResult result = new FnsFaqBgeCutoffEvaluator(objectMapper).run(
                cli.bgeManifest(), cli.targetsManifest(), cli.cutoffs(), cli.output());
        System.out.println(result.outputDirectory().toAbsolutePath().normalize());
    }

    public EvaluationResult run(
            Path bgeManifestPath,
            Path targetsManifestPath,
            List<Integer> cutoffs,
            Path outputDirectory) throws IOException {
        List<Integer> normalizedCutoffs = normalizeCutoffs(cutoffs);
        BgeInput bge = readBge(bgeManifestPath);
        TargetInput targets = readTargets(targetsManifestPath, bge.byQid().keySet());
        List<PerQueryMetric> perQuery = evaluate(bge, targets, normalizedCutoffs);
        List<AggregateMetric> aggregate = aggregate(perQuery, normalizedCutoffs);

        Path output = initializeOutputDirectory(outputDirectory);
        Path perQueryPath = output.resolve(PER_QUERY_FILENAME);
        Path aggregatePath = output.resolve(AGGREGATE_FILENAME);
        writePerQuery(perQueryPath, perQuery);
        writeAggregate(aggregatePath, aggregate);

        EvaluationManifest manifest = new EvaluationManifest(
                1,
                EVAL_VERSION,
                targets.evalVersion(),
                QUESTION_COUNT,
                targets.eligibleCount(),
                TOP_K,
                normalizedCutoffs.stream()
                        .map(cutoff -> new CutoffDefinition(
                                cutoff,
                                cutoff == SCORED_POOL ? "primary" : "sensitivity"))
                        .toList(),
                new MetricContract(
                        "Binary gain; DCG=sum(rel/log2(rank+1)); IDCG uses unique targetDocIds",
                        "Macro mean over eligible qids with a fixed denominator at every cutoff",
                        "CandidateRecall and AllTargets use the first-stage prefix",
                        "Before is original first-stage order; after is BGE score order within the prefix",
                        "Article-only and mixed labels are reported only as separate sensitivity slices"),
                new BootstrapContract(
                        BOOTSTRAP_SEED,
                        BOOTSTRAP_RESAMPLES,
                        "queryGroup",
                        "resample clusters with replacement; macro qid mean delta nDCG@10",
                        "nearest-rank 2.5% and 97.5% quantiles"),
                bge.identity(),
                artifact(bge.manifest()),
                artifact(bge.provenance()),
                artifact(targets.manifest()),
                artifact(targets.targets()),
                targets.corpus(),
                new OutputArtifact(PER_QUERY_FILENAME, sha256(Files.readAllBytes(perQueryPath))),
                new OutputArtifact(AGGREGATE_FILENAME, sha256(Files.readAllBytes(aggregatePath))));

        ObjectMapper deterministicMapper = objectMapper.copy()
                .enable(MapperFeature.SORT_PROPERTIES_ALPHABETICALLY)
                .enable(SerializationFeature.ORDER_MAP_ENTRIES_BY_KEYS)
                .enable(SerializationFeature.INDENT_OUTPUT);
        Path manifestPath = output.resolve(MANIFEST_FILENAME);
        deterministicMapper.writeValue(manifestPath.toFile(), manifest);
        String manifestSha256 = sha256(Files.readAllBytes(manifestPath));
        Files.writeString(
                output.resolve(MANIFEST_SHA_FILENAME),
                manifestSha256 + "  " + MANIFEST_FILENAME + "\n",
                StandardCharsets.UTF_8);
        return new EvaluationResult(output, manifest, manifestSha256);
    }

    private BgeInput readBge(Path manifestPath) throws IOException {
        InputBytes manifestInput = readInput(manifestPath, "BGE manifest");
        JsonNode manifest = readObject(manifestInput.bytes(), "BGE manifest");
        requireEqual(2, requiredInt(manifest, "schemaVersion", "BGE manifest"),
                "Unsupported BGE manifest schema");
        requireTextEquals(manifest, "evalVersion", "tax-eval-v3-bge-reranker-pool", "BGE manifest");
        requireEqual(QUESTION_COUNT, requiredInt(manifest, "questionCount", "BGE manifest"),
                "BGE manifest question count differs from the frozen frame");
        requiredPositiveInt(manifest, "corpusDocumentCount", "BGE manifest");

        JsonNode attestation = requiredObject(manifest, "rerankerAttestation", "BGE manifest");
        requireEqual(1, requiredPositiveInt(
                attestation, "healthSchemaVersion", "BGE reranker attestation"),
                "Unsupported BGE reranker health schema");
        requireTextEquals(
                attestation, "model", BGE_MODEL, "BGE reranker attestation");
        requireTextEquals(
                attestation, "revision", BGE_REVISION, "BGE reranker attestation");
        requireTextEquals(
                attestation, "precision", BGE_PRECISION, "BGE reranker attestation");
        requireTextEquals(
                attestation, "device", BGE_DEVICE, "BGE reranker attestation");
        requireEqual(BGE_BATCH_SIZE, requiredPositiveInt(
                attestation, "batchSize", "BGE reranker attestation"),
                "BGE reranker batch size differs from the frozen contract");
        requireEqual(BGE_MAX_LENGTH, requiredPositiveInt(
                attestation, "maxLength", "BGE reranker attestation"),
                "BGE reranker max length differs from the frozen contract");
        String healthResponseSha256 = requiredText(
                attestation, "healthResponseSha256", "BGE reranker attestation");
        if (!SHA256.matcher(healthResponseSha256).matches()) {
            throw new IllegalArgumentException("BGE reranker attestation has invalid health response SHA-256");
        }

        JsonNode pooling = requiredObject(manifest, "pooling", "BGE manifest");
        requireEqual(TOP_K, requiredInt(pooling, "topN", "BGE pooling"),
                "BGE output depth must be 10");
        requireEqual(SCORED_POOL, requiredInt(pooling, "scoredPool", "BGE pooling"),
                "BGE scored pool must be 50");
        for (String field : List.of("model", "revision", "precision", "device")) {
            requireTextEquals(pooling, field, requiredText(
                    attestation, field, "BGE reranker attestation"), "BGE pooling");
        }
        requireEqual(requiredPositiveInt(attestation, "batchSize", "BGE reranker attestation"),
                requiredPositiveInt(pooling, "batchSize", "BGE pooling"),
                "BGE pooling batch size differs from attestation");
        requireEqual(requiredPositiveInt(attestation, "maxLength", "BGE reranker attestation"),
                requiredPositiveInt(pooling, "maxLength", "BGE pooling"),
                "BGE pooling max length differs from attestation");
        requireTextEquals(
                pooling,
                "ordering",
                "score_desc,first_stage_rank_asc,doc_id_asc",
                "BGE pooling");

        ArtifactReference provenanceReference = readArtifact(
                manifest, "sealedProvenance", "BGE manifest");
        InputBytes provenanceInput = readSiblingArtifact(
                manifestInput.path(), provenanceReference, "BGE sealed provenance");
        List<JsonNode> rows = readJsonLines(provenanceInput, "BGE sealed provenance");
        requireEqual(QUESTION_COUNT, rows.size(),
                "BGE provenance must contain exactly 240 qids");

        Map<String, List<Candidate>> byQid = new LinkedHashMap<>();
        for (JsonNode row : rows) {
            String qid = requiredText(row, "qid", "BGE provenance row");
            if (byQid.containsKey(qid)) {
                throw new IllegalArgumentException("Duplicate qid in BGE provenance: " + qid);
            }
            JsonNode candidatesNode = requiredArray(row, "bge", "BGE provenance row " + qid);
            requireEqual(SCORED_POOL, candidatesNode.size(),
                    "BGE provenance row must contain exactly 50 candidates: " + qid);
            List<Candidate> candidates = new ArrayList<>(SCORED_POOL);
            Set<Integer> firstStageRanks = new HashSet<>();
            Set<String> docIds = new HashSet<>();
            for (int index = 0; index < candidatesNode.size(); index++) {
                JsonNode candidateNode = candidatesNode.get(index);
                int firstStageRank = requiredPositiveInt(
                        candidateNode, "firstStageRank", "BGE candidate " + qid);
                int rerankRank = requiredPositiveInt(
                        candidateNode, "rerankRank", "BGE candidate " + qid);
                if (rerankRank != index + 1) {
                    throw new IllegalArgumentException("Non-contiguous BGE rerankRank for " + qid);
                }
                if (firstStageRank > SCORED_POOL || !firstStageRanks.add(firstStageRank)) {
                    throw new IllegalArgumentException("Invalid BGE firstStageRank for " + qid);
                }
                String docId = requiredDocId(candidateNode, "docId", "BGE candidate " + qid);
                if (!docIds.add(docId)) {
                    throw new IllegalArgumentException("Duplicate BGE candidate for " + qid + ": " + docId);
                }
                candidates.add(new Candidate(
                        firstStageRank,
                        rerankRank,
                        docId,
                        requiredFiniteDouble(candidateNode, "score", "BGE candidate " + qid)));
            }
            for (int rank = 1; rank <= SCORED_POOL; rank++) {
                if (!firstStageRanks.contains(rank)) {
                    throw new IllegalArgumentException("Missing BGE firstStageRank " + rank + " for " + qid);
                }
            }
            List<Candidate> canonical = candidates.stream().sorted(RERANK_ORDER).toList();
            if (!canonical.equals(candidates)) {
                throw new IllegalArgumentException("BGE provenance is not in canonical score order: " + qid);
            }
            byQid.put(qid, List.copyOf(candidates));
        }

        BgeIdentity identity = new BgeIdentity(
                requiredText(pooling, "model", "BGE pooling"),
                requiredText(pooling, "revision", "BGE pooling"),
                requiredText(pooling, "precision", "BGE pooling"),
                requiredText(pooling, "device", "BGE pooling"),
                requiredPositiveInt(pooling, "batchSize", "BGE pooling"),
                requiredPositiveInt(pooling, "maxLength", "BGE pooling"),
                requiredText(pooling, "ordering", "BGE pooling"));
        return new BgeInput(
                manifestInput,
                provenanceInput,
                identity,
                Map.copyOf(byQid));
    }

    private TargetInput readTargets(Path manifestPath, Set<String> expectedQids) throws IOException {
        InputBytes manifestInput = readInput(manifestPath, "targets manifest");
        JsonNode manifest = readObject(manifestInput.bytes(), "targets manifest");
        requireEqual(1, requiredInt(manifest, "schemaVersion", "targets manifest"),
                "Unsupported targets manifest schema");
        String evalVersion = requiredText(manifest, "evalVersion", "targets manifest");
        requireTextEquals(manifest, "freezeStatus", "frozen", "targets manifest");
        String ruleVersion = requiredText(manifest, "ruleVersion", "targets manifest");
        String parserVersion = requiredText(manifest, "parserVersion", "targets manifest");
        requireEqual(QUESTION_COUNT, requiredInt(manifest, "questionCount", "targets manifest"),
                "Targets source question count differs from the frozen frame");
        int primaryCount = requiredNonNegativeInt(
                manifest, "primaryCitedClauseCount", "targets manifest");
        int articleCount = requiredNonNegativeInt(
                manifest, "articleOnlySingleChunkCount", "targets manifest");
        int mixedCount = requiredNonNegativeInt(
                manifest, "mixedLiteralCount", "targets manifest");
        int excludedCount = requiredNonNegativeInt(
                manifest, "excludedCount", "targets manifest");
        requireEqual(QUESTION_COUNT, primaryCount + articleCount + mixedCount + excludedCount,
                "Targets status counts differ from the frozen frame");
        int expectedEligible = primaryCount + articleCount + mixedCount;
        if (expectedEligible == 0) {
            throw new IllegalArgumentException("Targets manifest has no eligible questions");
        }
        ArtifactReference corpus = readArtifact(manifest, "corpus", "targets manifest");
        ArtifactReference targetsReference = readArtifact(manifest, "labels", "targets manifest");
        InputBytes targetsInput = readSiblingArtifact(
                manifestInput.path(), targetsReference, "cited-clause labels");
        JsonNode labelSet = readObject(targetsInput.bytes(), "cited-clause labels");
        requireEqual(1, requiredInt(labelSet, "schemaVersion", "cited-clause labels"),
                "Unsupported cited-clause label schema");
        requireTextEquals(labelSet, "evalVersion", evalVersion, "cited-clause labels");
        requireTextEquals(labelSet, "freezeStatus", "frozen", "cited-clause labels");
        requireTextEquals(labelSet, "ruleVersion", ruleVersion, "cited-clause labels");
        requireTextEquals(labelSet, "parserVersion", parserVersion, "cited-clause labels");
        requireEqual(QUESTION_COUNT, requiredInt(labelSet, "questionCount", "cited-clause labels"),
                "Cited-clause label count differs from the frozen frame");
        JsonNode rows = requiredArray(labelSet, "labels", "cited-clause labels");
        requireEqual(QUESTION_COUNT, rows.size(),
                "Cited-clause labels must contain exactly 240 qids");

        Map<String, TargetRow> byQid = new LinkedHashMap<>();
        int eligible = 0;
        Map<String, Integer> observedStatuses = new HashMap<>();
        for (JsonNode row : rows) {
            requireExactFields(row, TARGET_FIELDS, "target row");
            String qid = requiredText(row, "qid", "target row");
            if (!expectedQids.contains(qid)) {
                throw new IllegalArgumentException("Unknown qid in targets: " + qid);
            }
            if (byQid.containsKey(qid)) {
                throw new IllegalArgumentException("Duplicate qid in targets: " + qid);
            }
            String status = requiredText(row, "status", "target row " + qid);
            if (!Set.of(PRIMARY_STATUS, ARTICLE_STATUS, MIXED_STATUS, EXCLUDED_STATUS).contains(status)) {
                throw new IllegalArgumentException("Unknown target status for " + qid + ": " + status);
            }
            observedStatuses.merge(status, 1, Integer::sum);
            String reasonCode = requiredText(row, "reason", "target row " + qid);
            String queryGroup = requiredText(row, "queryGroup", "target row " + qid);
            int categoryId = requiredPositiveInt(row, "categoryId", "target row " + qid);
            String slice = requiredText(row, "slice", "target row " + qid);
            JsonNode docIdsNode = requiredArray(row, "targetDocIds", "target row " + qid);
            LinkedHashSet<String> targetDocIds = new LinkedHashSet<>();
            for (JsonNode docIdNode : docIdsNode) {
                if (!docIdNode.isTextual() || !DOC_ID.matcher(docIdNode.textValue()).matches()) {
                    throw new IllegalArgumentException("Invalid target docId for " + qid);
                }
                if (!targetDocIds.add(docIdNode.textValue())) {
                    throw new IllegalArgumentException("Duplicate target docId for " + qid);
                }
            }
            if (!EXCLUDED_STATUS.equals(status)) {
                if (targetDocIds.isEmpty()) {
                    throw new IllegalArgumentException("Eligible target row has no documents: " + qid);
                }
                requireStatusSlice(status, slice, qid);
                eligible++;
            } else {
                if (!"none".equals(slice)) {
                    throw new IllegalArgumentException("Excluded target row has a metric slice: " + qid);
                }
                if (!targetDocIds.isEmpty()) {
                    throw new IllegalArgumentException("Excluded target row has documents: " + qid);
                }
            }
            byQid.put(qid, new TargetRow(
                    qid,
                    status,
                    reasonCode,
                    queryGroup,
                    categoryId,
                    slice,
                    List.copyOf(targetDocIds)));
        }
        if (!byQid.keySet().equals(expectedQids)) {
            throw new IllegalArgumentException("Targets qids differ from BGE provenance");
        }
        requireEqual(expectedEligible, eligible,
                "Targets eligible count differs from targets manifest");
        requireEqual(primaryCount, observedStatuses.getOrDefault(PRIMARY_STATUS, 0),
                "Primary target count differs from targets manifest");
        requireEqual(articleCount, observedStatuses.getOrDefault(ARTICLE_STATUS, 0),
                "Article-only target count differs from targets manifest");
        requireEqual(mixedCount, observedStatuses.getOrDefault(MIXED_STATUS, 0),
                "Mixed target count differs from targets manifest");
        requireEqual(excludedCount, observedStatuses.getOrDefault(EXCLUDED_STATUS, 0),
                "Excluded target count differs from targets manifest");
        return new TargetInput(
                manifestInput,
                targetsInput,
                evalVersion,
                expectedEligible,
                new InputArtifact(corpus.filename(), corpus.sha256()),
                Map.copyOf(byQid));
    }

    private static void requireStatusSlice(String status, String slice, String qid) {
        String expected = switch (status) {
            case PRIMARY_STATUS -> PRIMARY_SLICE;
            case ARTICLE_STATUS -> ARTICLE_SLICE;
            case MIXED_STATUS -> MIXED_SLICE;
            default -> throw new IllegalArgumentException("Status is not eligible: " + qid);
        };
        if (!expected.equals(slice)) {
            throw new IllegalArgumentException("Target status and slice differ for " + qid);
        }
    }

    private static List<PerQueryMetric> evaluate(
            BgeInput bge,
            TargetInput targets,
            List<Integer> cutoffs) {
        List<PerQueryMetric> result = new ArrayList<>(QUESTION_COUNT * cutoffs.size());
        List<String> qids = bge.byQid().keySet().stream().sorted().toList();
        for (String qid : qids) {
            TargetRow target = targets.byQid().get(qid);
            List<Candidate> frozen = bge.byQid().get(qid);
            List<Candidate> firstStage = frozen.stream()
                    .sorted(Comparator.comparingInt(Candidate::firstStageRank))
                    .toList();
            for (int cutoff : cutoffs) {
                String cutoffRole = cutoff == SCORED_POOL ? "primary" : "sensitivity";
                if (!target.eligible()) {
                    result.add(PerQueryMetric.excluded(target, cutoff, cutoffRole));
                    continue;
                }
                Set<String> targetDocs = Set.copyOf(target.targetDocIds());
                List<Candidate> prefix = firstStage.subList(0, cutoff);
                List<Candidate> reranked = prefix.stream().sorted(RERANK_ORDER).toList();
                CandidateMetrics candidate = candidateMetrics(prefix, targetDocs);
                RankingMetrics before = rankingMetrics(firstStage, targetDocs);
                RankingMetrics after = rankingMetrics(reranked, targetDocs);
                double delta = after.ndcg() - before.ndcg();
                result.add(new PerQueryMetric(
                        qid,
                        target.status(),
                        target.reasonCode(),
                        target.queryGroup(),
                        target.categoryId(),
                        target.slice(),
                        cutoff,
                        cutoffRole,
                        targetDocs.size(),
                        candidate.hit(),
                        candidate.recall(),
                        candidate.allTargets(),
                        before.rank(),
                        before.hit(),
                        before.reciprocalRank(),
                        before.ndcg(),
                        after.rank(),
                        after.hit(),
                        after.reciprocalRank(),
                        after.ndcg(),
                        delta,
                        comparison(delta)));
            }
        }
        return List.copyOf(result);
    }

    private static CandidateMetrics candidateMetrics(
            List<Candidate> candidates,
            Set<String> targets) {
        long found = candidates.stream().map(Candidate::docId).filter(targets::contains).count();
        return new CandidateMetrics(
                found > 0,
                (double) found / targets.size(),
                found == targets.size());
    }

    private static RankingMetrics rankingMetrics(
            List<Candidate> candidates,
            Set<String> targets) {
        int limit = Math.min(TOP_K, candidates.size());
        int firstRank = 0;
        double dcg = 0.0;
        for (int index = 0; index < limit; index++) {
            if (targets.contains(candidates.get(index).docId())) {
                int rank = index + 1;
                if (firstRank == 0) {
                    firstRank = rank;
                }
                dcg += discount(rank);
            }
        }
        double idealDcg = 0.0;
        for (int rank = 1; rank <= Math.min(targets.size(), TOP_K); rank++) {
            idealDcg += discount(rank);
        }
        return new RankingMetrics(
                firstRank,
                firstRank > 0,
                firstRank == 0 ? 0.0 : 1.0 / firstRank,
                dcg / idealDcg);
    }

    private static double discount(int rank) {
        return 1.0 / (Math.log(rank + 1.0) / Math.log(2.0));
    }

    private static String comparison(double delta) {
        if (delta > EPSILON) {
            return "win";
        }
        if (delta < -EPSILON) {
            return "loss";
        }
        return "tie";
    }

    private static List<AggregateMetric> aggregate(
            List<PerQueryMetric> perQuery,
            List<Integer> cutoffs) {
        List<Slice> slices = new ArrayList<>();
        slices.add(new Slice("all_eligible", ignored -> true));
        perQuery.stream()
                .filter(PerQueryMetric::eligible)
                .map(PerQueryMetric::slice)
                .distinct()
                .sorted()
                .forEach(slice -> slices.add(new Slice(slice, row -> slice.equals(row.slice()))));
        slices.add(new Slice("multi_target", row -> row.targetCount() > 1));

        List<AggregateMetric> result = new ArrayList<>();
        for (int cutoff : cutoffs) {
            List<PerQueryMetric> eligibleAtCutoff = perQuery.stream()
                    .filter(PerQueryMetric::eligible)
                    .filter(row -> row.cutoff() == cutoff)
                    .toList();
            for (Slice slice : slices) {
                List<PerQueryMetric> rows = eligibleAtCutoff.stream()
                        .filter(slice.predicate())
                        .toList();
                if (rows.isEmpty()) {
                    continue;
                }
                double meanDelta = mean(rows, PerQueryMetric::deltaNdcgAt10);
                ConfidenceInterval interval = clusterBootstrap(rows, cutoff, slice.name());
                String reportingRole;
                if (cutoff == SCORED_POOL && PRIMARY_SLICE.equals(slice.name())) {
                    reportingRole = "primary";
                } else if (cutoff == SCORED_POOL && "multi_target".equals(slice.name())) {
                    reportingRole = "secondary";
                } else {
                    reportingRole = "sensitivity";
                }
                result.add(new AggregateMetric(
                        cutoff,
                        reportingRole,
                        slice.name(),
                        rows.size(),
                        rows.stream().map(PerQueryMetric::queryGroup).distinct().count(),
                        mean(rows, row -> row.candidateHit() ? 1.0 : 0.0),
                        mean(rows, PerQueryMetric::candidateRecall),
                        mean(rows, row -> row.allTargets() ? 1.0 : 0.0),
                        mean(rows, row -> row.beforeHitAt10() ? 1.0 : 0.0),
                        mean(rows, row -> row.afterHitAt10() ? 1.0 : 0.0),
                        mean(rows, PerQueryMetric::beforeReciprocalRankAt10),
                        mean(rows, PerQueryMetric::afterReciprocalRankAt10),
                        mean(rows, PerQueryMetric::beforeNdcgAt10),
                        mean(rows, PerQueryMetric::afterNdcgAt10),
                        meanDelta,
                        interval.low(),
                        interval.high(),
                        rows.stream().filter(row -> "win".equals(row.comparison())).count(),
                        rows.stream().filter(row -> "tie".equals(row.comparison())).count(),
                        rows.stream().filter(row -> "loss".equals(row.comparison())).count()));
            }
        }
        return List.copyOf(result);
    }

    private static ConfidenceInterval clusterBootstrap(
            List<PerQueryMetric> rows,
            int cutoff,
            String slice) {
        Map<String, List<Double>> byGroup = new TreeMap<>();
        for (PerQueryMetric row : rows) {
            byGroup.computeIfAbsent(row.queryGroup(), ignored -> new ArrayList<>())
                    .add(row.deltaNdcgAt10());
        }
        List<List<Double>> groups = List.copyOf(byGroup.values());
        if (groups.size() == 1) {
            double value = groups.getFirst().stream().mapToDouble(Double::doubleValue).average().orElseThrow();
            return new ConfidenceInterval(value, value);
        }

        Random random = new Random(seedFor(cutoff, slice));
        List<Double> estimates = new ArrayList<>(BOOTSTRAP_RESAMPLES);
        for (int replicate = 0; replicate < BOOTSTRAP_RESAMPLES; replicate++) {
            double sum = 0.0;
            int count = 0;
            for (int draw = 0; draw < groups.size(); draw++) {
                List<Double> sampled = groups.get(random.nextInt(groups.size()));
                for (double value : sampled) {
                    sum += value;
                    count++;
                }
            }
            estimates.add(sum / count);
        }
        estimates.sort(Double::compareTo);
        return new ConfidenceInterval(
                nearestRank(estimates, 0.025),
                nearestRank(estimates, 0.975));
    }

    private static long seedFor(int cutoff, String slice) {
        byte[] digest = digest((BOOTSTRAP_SEED + "\0" + cutoff + "\0" + slice)
                .getBytes(StandardCharsets.UTF_8));
        return ByteBuffer.wrap(digest, 0, Long.BYTES).getLong();
    }

    private static double nearestRank(List<Double> sorted, double quantile) {
        int index = (int) Math.ceil(quantile * sorted.size()) - 1;
        return sorted.get(Math.max(0, Math.min(index, sorted.size() - 1)));
    }

    private static double mean(
            List<PerQueryMetric> rows,
            java.util.function.ToDoubleFunction<PerQueryMetric> value) {
        return rows.stream().mapToDouble(value).average().orElseThrow();
    }

    private static void writePerQuery(Path path, List<PerQueryMetric> rows) throws IOException {
        List<String> lines = new ArrayList<>(rows.size() + 1);
        lines.add("qid,status,reason_code,query_group,category_id,slice,cutoff,cutoff_role,target_count,"
                + "candidate_hit,candidate_recall,all_targets,before_rank_at_10,before_hit_at_10,"
                + "before_rr_at_10,before_ndcg_at_10,after_rank_at_10,after_hit_at_10,"
                + "after_rr_at_10,after_ndcg_at_10,delta_ndcg_at_10,comparison");
        for (PerQueryMetric row : rows) {
            lines.add(csvLine(
                    row.qid(),
                    row.status(),
                    row.reasonCode(),
                    row.queryGroup(),
                    row.categoryId(),
                    row.slice(),
                    row.cutoff(),
                    row.cutoffRole(),
                    row.targetCount(),
                    row.candidateHit(),
                    row.candidateRecall(),
                    row.allTargets(),
                    row.beforeRankAt10(),
                    row.beforeHitAt10(),
                    row.beforeReciprocalRankAt10(),
                    row.beforeNdcgAt10(),
                    row.afterRankAt10(),
                    row.afterHitAt10(),
                    row.afterReciprocalRankAt10(),
                    row.afterNdcgAt10(),
                    row.deltaNdcgAt10(),
                    row.comparison()));
        }
        Files.write(path, lines, StandardCharsets.UTF_8);
    }

    private static void writeAggregate(Path path, List<AggregateMetric> rows) throws IOException {
        List<String> lines = new ArrayList<>(rows.size() + 1);
        lines.add("cutoff,reporting_role,slice,eligible_queries,query_groups,candidate_hit_at_cutoff,"
                + "candidate_recall_at_cutoff,all_targets_at_cutoff,before_hit_at_10,after_hit_at_10,"
                + "before_mrr_at_10,after_mrr_at_10,before_ndcg_at_10,after_ndcg_at_10,"
                + "mean_delta_ndcg_at_10,delta_ci95_low,delta_ci95_high,paired_wins,paired_ties,paired_losses");
        for (AggregateMetric row : rows) {
            lines.add(csvLine(
                    row.cutoff(),
                    row.reportingRole(),
                    row.slice(),
                    row.eligibleQueries(),
                    row.queryGroups(),
                    row.candidateHitAtCutoff(),
                    row.candidateRecallAtCutoff(),
                    row.allTargetsAtCutoff(),
                    row.beforeHitAt10(),
                    row.afterHitAt10(),
                    row.beforeMrrAt10(),
                    row.afterMrrAt10(),
                    row.beforeNdcgAt10(),
                    row.afterNdcgAt10(),
                    row.meanDeltaNdcgAt10(),
                    row.deltaCi95Low(),
                    row.deltaCi95High(),
                    row.pairedWins(),
                    row.pairedTies(),
                    row.pairedLosses()));
        }
        Files.write(path, lines, StandardCharsets.UTF_8);
    }

    private static String csvLine(Object... values) {
        List<String> fields = new ArrayList<>(values.length);
        for (Object value : values) {
            String text;
            if (value == null) {
                text = "";
            } else if (value instanceof Double number) {
                text = String.format(Locale.ROOT, "%.9f", number);
            } else {
                text = value.toString();
            }
            if (text.indexOf(',') >= 0 || text.indexOf('"') >= 0
                    || text.indexOf('\n') >= 0 || text.indexOf('\r') >= 0) {
                text = '"' + text.replace("\"", "\"\"") + '"';
            }
            fields.add(text);
        }
        return String.join(",", fields);
    }

    private static Path initializeOutputDirectory(Path outputDirectory) throws IOException {
        if (outputDirectory == null) {
            throw new IllegalArgumentException("Output directory is required");
        }
        Path output = outputDirectory.toAbsolutePath().normalize();
        if (Files.exists(output)) {
            if (!Files.isDirectory(output)) {
                throw new IllegalArgumentException("Output path is not a directory: " + output);
            }
            try (var entries = Files.list(output)) {
                if (entries.findAny().isPresent()) {
                    throw new IllegalArgumentException("Output directory must be empty: " + output);
                }
            }
        }
        Files.createDirectories(output);
        return output;
    }

    private static List<Integer> normalizeCutoffs(List<Integer> cutoffs) {
        if (cutoffs == null) {
            throw new IllegalArgumentException("Cutoffs are required");
        }
        List<Integer> normalized = cutoffs.stream().distinct().sorted().toList();
        if (!REQUIRED_CUTOFFS.equals(normalized)) {
            throw new IllegalArgumentException("Cutoffs must be exactly 10,20,30,50");
        }
        return normalized;
    }

    private JsonNode readObject(byte[] bytes, String description) throws IOException {
        JsonNode root = objectMapper.readTree(bytes);
        if (root == null || !root.isObject()) {
            throw new IllegalArgumentException(description + " must be a JSON object");
        }
        return root;
    }

    private List<JsonNode> readJsonLines(InputBytes input, String description) throws IOException {
        List<JsonNode> rows = new ArrayList<>();
        int lineNumber = 0;
        for (String line : Files.readAllLines(input.path(), StandardCharsets.UTF_8)) {
            lineNumber++;
            if (line.isBlank()) {
                throw new IllegalArgumentException(description + " contains a blank line at " + lineNumber);
            }
            JsonNode row = objectMapper.readTree(line);
            if (row == null || !row.isObject()) {
                throw new IllegalArgumentException(description + " contains a non-object row at " + lineNumber);
            }
            rows.add(row);
        }
        return List.copyOf(rows);
    }

    private static InputBytes readInput(Path path, String description) throws IOException {
        if (path == null) {
            throw new IllegalArgumentException(description + " path is required");
        }
        Path normalized = path.toAbsolutePath().normalize();
        if (!Files.isRegularFile(normalized)) {
            throw new IllegalArgumentException(description + " is not a file: " + normalized);
        }
        byte[] bytes = Files.readAllBytes(normalized);
        return new InputBytes(normalized, normalized.getFileName().toString(), bytes, sha256(bytes));
    }

    private static InputBytes readSiblingArtifact(
            Path manifestPath,
            ArtifactReference reference,
            String description) throws IOException {
        if (!Path.of(reference.filename()).getFileName().toString().equals(reference.filename())) {
            throw new IllegalArgumentException(description + " filename must not contain a path");
        }
        InputBytes input = readInput(manifestPath.resolveSibling(reference.filename()), description);
        if (!reference.sha256().equals(input.sha256())) {
            throw new IllegalArgumentException(description + " SHA-256 differs from manifest");
        }
        return input;
    }

    private static ArtifactReference readArtifact(
            JsonNode root,
            String field,
            String description) {
        JsonNode artifact = requiredObject(root, field, description);
        String filename = requiredText(artifact, "filename", description + " " + field);
        String sha256 = requiredText(artifact, "sha256", description + " " + field);
        if (!SHA256.matcher(sha256).matches()) {
            throw new IllegalArgumentException(description + " has invalid SHA-256 for " + field);
        }
        return new ArtifactReference(filename, sha256);
    }

    private static InputArtifact artifact(InputBytes input) {
        return new InputArtifact(input.filename(), input.sha256());
    }

    private static JsonNode requiredObject(JsonNode root, String field, String description) {
        JsonNode value = root.path(field);
        if (!value.isObject()) {
            throw new IllegalArgumentException(description + " has invalid object field: " + field);
        }
        return value;
    }

    private static JsonNode requiredArray(JsonNode root, String field, String description) {
        JsonNode value = root.path(field);
        if (!value.isArray()) {
            throw new IllegalArgumentException(description + " has invalid array field: " + field);
        }
        return value;
    }

    private static String requiredText(JsonNode root, String field, String description) {
        JsonNode value = root.path(field);
        if (!value.isTextual() || value.textValue().isBlank()) {
            throw new IllegalArgumentException(description + " has invalid text field: " + field);
        }
        return value.textValue();
    }

    private static String requiredDocId(JsonNode root, String field, String description) {
        String value = requiredText(root, field, description);
        if (!DOC_ID.matcher(value).matches()) {
            throw new IllegalArgumentException(description + " has invalid docId: " + value);
        }
        return value;
    }

    private static int requiredInt(JsonNode root, String field, String description) {
        JsonNode value = root.path(field);
        if (!value.isIntegralNumber()) {
            throw new IllegalArgumentException(description + " has invalid integer field: " + field);
        }
        return value.intValue();
    }

    private static int requiredPositiveInt(JsonNode root, String field, String description) {
        int value = requiredInt(root, field, description);
        if (value <= 0) {
            throw new IllegalArgumentException(description + " has non-positive field: " + field);
        }
        return value;
    }

    private static int requiredNonNegativeInt(JsonNode root, String field, String description) {
        int value = requiredInt(root, field, description);
        if (value < 0) {
            throw new IllegalArgumentException(description + " has negative field: " + field);
        }
        return value;
    }

    private static double requiredFiniteDouble(JsonNode root, String field, String description) {
        JsonNode value = root.path(field);
        if (!value.isNumber() || !Double.isFinite(value.doubleValue())) {
            throw new IllegalArgumentException(description + " has invalid number field: " + field);
        }
        return value.doubleValue();
    }

    private static void requireTextEquals(
            JsonNode root,
            String field,
            String expected,
            String description) {
        String actual = requiredText(root, field, description);
        if (!expected.equals(actual)) {
            throw new IllegalArgumentException(description + " has unexpected " + field + ": " + actual);
        }
    }

    private static void requireEqual(int expected, int actual, String message) {
        if (expected != actual) {
            throw new IllegalArgumentException(message);
        }
    }

    private static void requireExactFields(JsonNode root, Set<String> expected, String description) {
        Set<String> actual = new HashSet<>();
        root.fieldNames().forEachRemaining(actual::add);
        if (!actual.equals(expected)) {
            throw new IllegalArgumentException(description + " fields differ from schema");
        }
    }

    public static String sha256(byte[] value) {
        return HexFormat.of().formatHex(digest(value));
    }

    private static byte[] digest(byte[] value) {
        try {
            return MessageDigest.getInstance("SHA-256").digest(value);
        } catch (NoSuchAlgorithmException error) {
            throw new IllegalStateException("SHA-256 is not available", error);
        }
    }

    public record EvaluationResult(
            Path outputDirectory,
            EvaluationManifest manifest,
            String manifestSha256) {
    }

    public record EvaluationManifest(
            int schemaVersion,
            String evalVersion,
            String targetEvalVersion,
            int sourceQuestionCount,
            int eligibleQuestionCount,
            int topK,
            List<CutoffDefinition> cutoffs,
            MetricContract metrics,
            BootstrapContract bootstrap,
            BgeIdentity bge,
            InputArtifact bgeManifest,
            InputArtifact bgeProvenance,
            InputArtifact targetsManifest,
            InputArtifact targets,
            InputArtifact corpus,
            OutputArtifact perQuery,
            OutputArtifact aggregate) {

        public EvaluationManifest {
            cutoffs = List.copyOf(cutoffs);
        }
    }

    public record CutoffDefinition(int cutoff, String reportingRole) {
    }

    public record MetricContract(
            String binaryNdcgAt10,
            String aggregation,
            String candidateCoverage,
            String rankingComparison,
            String articleOnlyPolicy) {
    }

    public record BootstrapContract(
            String seed,
            int resamples,
            String clusterUnit,
            String statistic,
            String interval) {
    }

    public record BgeIdentity(
            String model,
            String revision,
            String precision,
            String device,
            int batchSize,
            int maxLength,
            String ordering) {
    }

    public record InputArtifact(String filename, String sha256) {
    }

    public record OutputArtifact(String filename, String sha256) {
    }

    public record PerQueryMetric(
            String qid,
            String status,
            String reasonCode,
            String queryGroup,
            int categoryId,
            String slice,
            int cutoff,
            String cutoffRole,
            Integer targetCount,
            Boolean candidateHit,
            Double candidateRecall,
            Boolean allTargets,
            Integer beforeRankAt10,
            Boolean beforeHitAt10,
            Double beforeReciprocalRankAt10,
            Double beforeNdcgAt10,
            Integer afterRankAt10,
            Boolean afterHitAt10,
            Double afterReciprocalRankAt10,
            Double afterNdcgAt10,
            Double deltaNdcgAt10,
            String comparison) {

        static PerQueryMetric excluded(TargetRow target, int cutoff, String cutoffRole) {
            return new PerQueryMetric(
                    target.qid(), target.status(), target.reasonCode(), target.queryGroup(),
                    target.categoryId(), target.slice(), cutoff, cutoffRole,
                    null, null, null, null, null, null, null, null, null, null, null, null, null, null);
        }

        boolean eligible() {
            return status.startsWith("eligible_");
        }
    }

    public record AggregateMetric(
            int cutoff,
            String reportingRole,
            String slice,
            int eligibleQueries,
            long queryGroups,
            double candidateHitAtCutoff,
            double candidateRecallAtCutoff,
            double allTargetsAtCutoff,
            double beforeHitAt10,
            double afterHitAt10,
            double beforeMrrAt10,
            double afterMrrAt10,
            double beforeNdcgAt10,
            double afterNdcgAt10,
            double meanDeltaNdcgAt10,
            double deltaCi95Low,
            double deltaCi95High,
            long pairedWins,
            long pairedTies,
            long pairedLosses) {
    }

    record CliArguments(
            Path bgeManifest,
            Path targetsManifest,
            List<Integer> cutoffs,
            Path output,
            boolean help) {

        static CliArguments parse(String[] args) {
            Path bgeManifest = null;
            Path targetsManifest = null;
            List<Integer> cutoffs = null;
            Path output = null;
            boolean help = false;
            for (int index = 0; index < args.length; index++) {
                String argument = args[index];
                if ("--help".equals(argument) || "-h".equals(argument)) {
                    help = true;
                } else if (argument.startsWith("--bge-manifest=")) {
                    bgeManifest = Path.of(argument.substring("--bge-manifest=".length()));
                } else if ("--bge-manifest".equals(argument)) {
                    bgeManifest = Path.of(requireValue(args, ++index, "--bge-manifest"));
                } else if (argument.startsWith("--targets-manifest=")) {
                    targetsManifest = Path.of(argument.substring("--targets-manifest=".length()));
                } else if ("--targets-manifest".equals(argument)) {
                    targetsManifest = Path.of(requireValue(args, ++index, "--targets-manifest"));
                } else if (argument.startsWith("--cutoffs=")) {
                    cutoffs = parseCutoffs(argument.substring("--cutoffs=".length()));
                } else if ("--cutoffs".equals(argument)) {
                    cutoffs = parseCutoffs(requireValue(args, ++index, "--cutoffs"));
                } else if (argument.startsWith("--output=")) {
                    output = Path.of(argument.substring("--output=".length()));
                } else if ("--output".equals(argument)) {
                    output = Path.of(requireValue(args, ++index, "--output"));
                } else {
                    throw new IllegalArgumentException("Unknown argument: " + argument);
                }
            }
            if (!help && (bgeManifest == null || targetsManifest == null
                    || cutoffs == null || output == null)) {
                throw new IllegalArgumentException(
                        "--bge-manifest, --targets-manifest, --cutoffs and --output are required");
            }
            if (cutoffs != null) {
                cutoffs = normalizeCutoffs(cutoffs);
            }
            return new CliArguments(bgeManifest, targetsManifest, cutoffs, output, help);
        }

        private static List<Integer> parseCutoffs(String value) {
            try {
                return java.util.Arrays.stream(value.split(","))
                        .map(String::strip)
                        .map(Integer::parseInt)
                        .toList();
            } catch (NumberFormatException error) {
                throw new IllegalArgumentException("Invalid --cutoffs value: " + value, error);
            }
        }

        private static String requireValue(String[] args, int index, String option) {
            if (index >= args.length || args[index].startsWith("--")) {
                throw new IllegalArgumentException("Missing value for " + option);
            }
            return args[index];
        }
    }

    private record Candidate(
            int firstStageRank,
            int rerankRank,
            String docId,
            double score) {
    }

    private record TargetRow(
            String qid,
            String status,
            String reasonCode,
            String queryGroup,
            int categoryId,
            String slice,
            List<String> targetDocIds) {

        boolean eligible() {
            return status.startsWith("eligible_");
        }
    }

    private record CandidateMetrics(boolean hit, double recall, boolean allTargets) {
    }

    private record RankingMetrics(int rank, boolean hit, double reciprocalRank, double ndcg) {
    }

    private record ConfidenceInterval(double low, double high) {
    }

    private record Slice(String name, Predicate<PerQueryMetric> predicate) {
    }

    private record BgeInput(
            InputBytes manifest,
            InputBytes provenance,
            BgeIdentity identity,
            Map<String, List<Candidate>> byQid) {
    }

    private record TargetInput(
            InputBytes manifest,
            InputBytes targets,
            String evalVersion,
            int eligibleCount,
            InputArtifact corpus,
            Map<String, TargetRow> byQid) {
    }

    private record ArtifactReference(String filename, String sha256) {
    }

    private record InputBytes(Path path, String filename, byte[] bytes, String sha256) {
    }
}
