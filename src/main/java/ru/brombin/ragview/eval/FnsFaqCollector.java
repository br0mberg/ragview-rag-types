package ru.brombin.ragview.eval;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.json.JsonMapper;

import java.io.IOException;
import java.net.http.HttpClient;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

public final class FnsFaqCollector {

    private static final Path DEFAULT_CONFIG = Path.of("data/eval/v3_source_config.json");
    private static final DateTimeFormatter SNAPSHOT_NAME = DateTimeFormatter
            .ofPattern("yyyyMMdd'T'HHmmss'Z'")
            .withZone(ZoneOffset.UTC);

    private final FnsFaqHttpClient client;
    private final FnsFaqParser parser;
    private final ObjectMapper objectMapper;

    public FnsFaqCollector(FnsFaqHttpClient client, FnsFaqParser parser, ObjectMapper objectMapper) {
        this.client = client;
        this.parser = parser;
        this.objectMapper = objectMapper;
    }

    public static void main(String[] args) throws Exception {
        CliArguments cli = CliArguments.parse(args);
        if (cli.help()) {
            System.out.println("Usage: FnsFaqCollector [--config PATH] [--output PATH]");
            return;
        }

        ObjectMapper objectMapper = JsonMapper.builder()
                .enable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES)
                .enable(SerializationFeature.INDENT_OUTPUT)
                .build();
        HttpClient httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(20))
                .followRedirects(HttpClient.Redirect.NORMAL)
                .build();
        FnsFaqCollector collector = new FnsFaqCollector(
                new FnsFaqHttpClient(httpClient, objectMapper, Duration.ofSeconds(30)),
                new FnsFaqParser(),
                objectMapper);
        Path output = cli.output() == null
                ? Path.of("data/local/eval-v3-fns", "snapshot-" + SNAPSHOT_NAME.format(Instant.now()))
                : cli.output();
        collector.collect(cli.config(), output, Instant.now());
        System.out.println(output.toAbsolutePath().normalize());
    }

    public SnapshotManifest collect(Path configPath, Path outputPath, Instant collectedAt)
            throws IOException, InterruptedException {
        byte[] configBytes = Files.readAllBytes(configPath);
        FnsFaqSourceConfig config = objectMapper.readValue(configBytes, FnsFaqSourceConfig.class);
        String configSha256 = FnsFaqPageSampler.sha256(configBytes);
        FnsFaqHttpClient.RobotsCheck robotsCheck = client.verifyRobots(config);
        FnsFaqSnapshotWriter writer = new FnsFaqSnapshotWriter(objectMapper, outputPath);
        writer.initialize();

        List<FnsFaqCandidate> collected = new ArrayList<>();
        List<CategoryManifest> categoryManifests = new ArrayList<>();
        RequestPacer pacer = new RequestPacer(config.requestDelayMs());
        for (FnsFaqSourceConfig.Category category : config.categories()) {
            FnsFaqHttpClient.RawPage first = fetch(config, category, 1, pacer);
            FnsFaqParser.ParsedPage firstParsed = parser.parse(first.answerHtml(), config.source());
            PaginationContract pagination = validateFirstPage(category, firstParsed);
            List<Integer> selectedPages = FnsFaqPageSampler.select(
                    config.selectionSeed(), category.id(), firstParsed.totalPages(), config.pagesPerCategory());
            List<PageManifest> fetchedPages = new ArrayList<>();
            int categoryCandidates = 0;

            PageCollection firstCollection = persistAndParse(
                    writer, config, category, first, selectedPages.contains(1));
            validatePage(category, 1, firstCollection.parsed(), pagination);
            fetchedPages.add(firstCollection.manifest());
            if (selectedPages.contains(1)) {
                collected.addAll(firstCollection.candidates());
                categoryCandidates += firstCollection.candidates().size();
            }

            for (int page : selectedPages) {
                if (page == 1) {
                    continue;
                }
                FnsFaqHttpClient.RawPage rawPage = fetch(config, category, page, pacer);
                PageCollection pageCollection = persistAndParse(writer, config, category, rawPage, true);
                validatePage(category, page, pageCollection.parsed(), pagination);
                fetchedPages.add(pageCollection.manifest());
                collected.addAll(pageCollection.candidates());
                categoryCandidates += pageCollection.candidates().size();
            }
            fetchedPages.sort(java.util.Comparator.comparingInt(PageManifest::page));
            categoryManifests.add(new CategoryManifest(
                    category.id(),
                    category.name(),
                    firstParsed.totalPages(),
                    firstParsed.totalFound(),
                    pagination.pageSize(),
                    selectedPages,
                    categoryCandidates,
                    List.copyOf(fetchedPages)));
        }

        FnsFaqCandidateSampler.DeduplicationResult deduplicated = FnsFaqCandidateSampler.deduplicate(collected);
        List<FnsFaqCandidate> selected = FnsFaqCandidateSampler.selectBalanced(
                deduplicated.candidates(),
                config.categories(),
                config.selection().targetTotal(),
                config.selectionSeed());
        List<FnsFaqNearDuplicateFinder.NearDuplicatePair> nearDuplicates = FnsFaqNearDuplicateFinder.find(
                selected, config.nearDuplicates().threshold());

        String candidatesSha256 = writer.writeJson("candidates.json", selected);
        String nearDuplicatesSha256 = writer.writeJson("near_duplicates.json", nearDuplicates);
        Map<Integer, Long> selectedByCategory = selected.stream().collect(Collectors.groupingBy(
                FnsFaqCandidate::categoryId,
                LinkedHashMap::new,
                Collectors.counting()));
        SnapshotManifest manifest = new SnapshotManifest(
                3,
                "tax-eval-v3-fns-candidates",
                collectedAt.toString(),
                config.source(),
                config.endpoint(),
                config.regionId(),
                robotsCheck,
                configSha256,
                config.selectionSeed(),
                config.pagesPerCategory(),
                config.requestDelayMs(),
                config.selection(),
                config.nearDuplicates(),
                config.scope(),
                new CitationExtraction(
                        FnsFaqParser.CITATION_PARSER_VERSION,
                        FnsFaqParser.citationParserRulesetSha256(),
                        "hint_unresolved",
                        "citedNkArticles are conservative hints; citedSource and raw responses require human review"),
                collected.size(),
                deduplicated.candidates().size(),
                deduplicated.removed(),
                selected.size(),
                selected.stream().filter(FnsFaqCandidate::naturalExact).count(),
                selected.stream().filter(candidate -> !candidate.citedNkArticles().isEmpty()).count(),
                nearDuplicates.size(),
                selectedByCategory,
                candidatesSha256,
                nearDuplicatesSha256,
                List.copyOf(categoryManifests));
        writer.writeJson("manifest.json", manifest);
        return manifest;
    }

    private FnsFaqHttpClient.RawPage fetch(
            FnsFaqSourceConfig config,
            FnsFaqSourceConfig.Category category,
            int page,
            RequestPacer pacer) throws IOException, InterruptedException {
        pacer.beforeRequest();
        return client.fetch(config, category, page);
    }

    private PageCollection persistAndParse(
            FnsFaqSnapshotWriter writer,
            FnsFaqSourceConfig config,
            FnsFaqSourceConfig.Category category,
            FnsFaqHttpClient.RawPage rawPage,
            boolean selected) throws IOException {
        FnsFaqSnapshotWriter.RawArtifact artifact = writer.writeRaw(
                category.id(), rawPage.page(), rawPage.bytes());
        if (!artifact.sha256().equals(rawPage.sha256())) {
            throw new IOException("Raw response changed while writing category="
                    + category.id() + ", page=" + rawPage.page());
        }
        FnsFaqParser.ParsedPage parsed = parser.parse(rawPage.answerHtml(), config.source());
        List<FnsFaqCandidate> candidates = selected
                ? parsed.entries().stream()
                .map(entry -> candidate(config, category, rawPage, entry))
                .toList()
                : List.of();
        return new PageCollection(
                candidates,
                new PageManifest(
                        rawPage.page(),
                        selected,
                        parsed.totalPages(),
                        parsed.totalFound(),
                        parsed.shownFrom(),
                        parsed.shownTo(),
                        parsed.questionMarkers(),
                        parsed.skippedQuestionMarkers(),
                        parsed.entries().size(),
                        artifact.path(),
                        artifact.sha256()),
                parsed);
    }

    static PaginationContract validateFirstPage(
            FnsFaqSourceConfig.Category category,
            FnsFaqParser.ParsedPage page) throws IOException {
        if (page.totalFound() <= 0) {
            throw htmlDrift(category, 1, "missing or non-positive totalFound");
        }
        if (page.totalPages() <= 0) {
            throw htmlDrift(category, 1, "missing or non-positive totalPages");
        }
        if (page.shownFrom() != 1 || page.shownTo() < page.shownFrom()) {
            throw htmlDrift(category, 1, "invalid first-page shown range "
                    + page.shownFrom() + ".." + page.shownTo());
        }
        int pageSize = page.shownTo() - page.shownFrom() + 1;
        int expectedPages = (page.totalFound() + pageSize - 1) / pageSize;
        if (page.totalPages() != expectedPages) {
            throw htmlDrift(category, 1, "pagination says " + page.totalPages()
                    + " pages, expected " + expectedPages + " from totalFound="
                    + page.totalFound() + " and pageSize=" + pageSize);
        }
        PaginationContract contract = new PaginationContract(
                page.totalPages(), page.totalFound(), pageSize);
        validatePage(category, 1, page, contract);
        return contract;
    }

    static void validatePage(
            FnsFaqSourceConfig.Category category,
            int requestedPage,
            FnsFaqParser.ParsedPage page,
            PaginationContract expected) throws IOException {
        if (requestedPage <= 0 || requestedPage > expected.totalPages()) {
            throw htmlDrift(category, requestedPage, "requested page is outside frozen pagination");
        }
        if (page.totalPages() != expected.totalPages() || page.totalFound() != expected.totalFound()) {
            throw htmlDrift(category, requestedPage, "pagination changed during collection: pages="
                    + page.totalPages() + ", totalFound=" + page.totalFound());
        }
        int expectedFrom = (requestedPage - 1) * expected.pageSize() + 1;
        int expectedTo = Math.min(expected.totalFound(), requestedPage * expected.pageSize());
        int expectedMarkers = expectedTo - expectedFrom + 1;
        if (page.shownFrom() != expectedFrom || page.shownTo() != expectedTo) {
            throw htmlDrift(category, requestedPage, "shown range is " + page.shownFrom()
                    + ".." + page.shownTo() + ", expected " + expectedFrom + ".." + expectedTo);
        }
        if (page.questionMarkers() != expectedMarkers
                || page.entries().size() != expectedMarkers
                || page.skippedQuestionMarkers() != 0) {
            throw htmlDrift(category, requestedPage, "question markers=" + page.questionMarkers()
                    + ", parsed=" + page.entries().size()
                    + ", skipped=" + page.skippedQuestionMarkers()
                    + ", expected=" + expectedMarkers);
        }
    }

    private static IOException htmlDrift(
            FnsFaqSourceConfig.Category category,
            int page,
            String detail) {
        return new IOException("FNS HTML contract drift for category=" + category.id()
                + ", page=" + page + ": " + detail);
    }

    private static FnsFaqCandidate candidate(
            FnsFaqSourceConfig config,
            FnsFaqSourceConfig.Category category,
            FnsFaqHttpClient.RawPage rawPage,
            FnsFaqParser.ParsedFaq entry) {
        String questionKey = FnsFaqParser.normalizeQuestionKey(entry.question());
        return new FnsFaqCandidate(
                "fns-" + FnsFaqPageSampler.sha256(questionKey),
                entry.question(),
                entry.answer(),
                entry.source(),
                entry.sourceLinks(),
                entry.citedNkArticles(),
                entry.naturalExact(),
                category.id(),
                category.name(),
                rawPage.page(),
                sourceUrl(config.source(), category.id()),
                rawPage.sha256());
    }

    private static String sourceUrl(String source, int categoryId) {
        return source + (source.contains("?") ? "&" : "?") + "t1=" + categoryId;
    }

    private record PageCollection(
            List<FnsFaqCandidate> candidates,
            PageManifest manifest,
            FnsFaqParser.ParsedPage parsed) {
    }

    private static final class RequestPacer {
        private final long delayMs;
        private boolean first = true;

        private RequestPacer(long delayMs) {
            this.delayMs = delayMs;
        }

        private void beforeRequest() throws InterruptedException {
            if (first) {
                first = false;
                return;
            }
            TimeUnit.MILLISECONDS.sleep(delayMs);
        }
    }

    public record PageManifest(
            int page,
            boolean selected,
            int totalPages,
            int totalFound,
            int shownFrom,
            int shownTo,
            int questionMarkers,
            int skippedQuestionMarkers,
            int parsedQuestions,
            String rawPath,
            String rawSha256) {
    }

    public record CategoryManifest(
            int id,
            String name,
            int availablePages,
            int totalFound,
            int pageSize,
            List<Integer> selectedPages,
            int candidatesBeforeGlobalDedup,
            List<PageManifest> fetchedPages) {
    }

    public record SnapshotManifest(
            int schemaVersion,
            String evalVersion,
            String collectedAtUtc,
            String source,
            String endpoint,
            int regionId,
            FnsFaqHttpClient.RobotsCheck robots,
            String configSha256,
            String selectionSeed,
            int pagesPerCategory,
            long requestDelayMs,
            FnsFaqSourceConfig.Selection selection,
            FnsFaqSourceConfig.NearDuplicates nearDuplicates,
            FnsFaqSourceConfig.Scope scope,
            CitationExtraction citationExtraction,
            int candidatesBeforeDedup,
            int uniqueCandidates,
            int exactDuplicatesRemoved,
            int selectedCandidates,
            long naturalExactCandidates,
            long candidatesWithCitedNkArticles,
            int nearDuplicatePairs,
            Map<Integer, Long> selectedByCategory,
            String candidatesSha256,
            String nearDuplicatesSha256,
            List<CategoryManifest> categories) {
    }

    public record CitationExtraction(
            String parserVersion,
            String rulesetSha256,
            String status,
            String policy) {
    }

    record PaginationContract(int totalPages, int totalFound, int pageSize) {
    }

    record CliArguments(Path config, Path output, boolean help) {
        static CliArguments parse(String[] args) {
            Path config = DEFAULT_CONFIG;
            Path output = null;
            boolean help = false;
            for (int index = 0; index < args.length; index++) {
                String argument = args[index];
                if ("--help".equals(argument) || "-h".equals(argument)) {
                    help = true;
                } else if (argument.startsWith("--config=")) {
                    config = Path.of(argument.substring("--config=".length()));
                } else if ("--config".equals(argument)) {
                    config = Path.of(requireValue(args, ++index, "--config"));
                } else if (argument.startsWith("--output=")) {
                    output = Path.of(argument.substring("--output=".length()));
                } else if ("--output".equals(argument)) {
                    output = Path.of(requireValue(args, ++index, "--output"));
                } else {
                    throw new IllegalArgumentException("Unknown argument: " + argument);
                }
            }
            return new CliArguments(config, output, help);
        }

        private static String requireValue(String[] args, int index, String option) {
            if (index >= args.length || args[index].startsWith("--")) {
                throw new IllegalArgumentException("Missing value for " + option);
            }
            return args[index];
        }
    }
}
