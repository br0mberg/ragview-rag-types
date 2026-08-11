package ru.brombin.ragview;

import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.stereotype.Component;
import ru.brombin.ragview.artifact.CandidateDumpWriter;
import ru.brombin.ragview.config.RagProperties;
import ru.brombin.ragview.corpus.CorpusDocument;
import ru.brombin.ragview.corpus.CorpusLoader;
import ru.brombin.ragview.corpus.DatasetValidator;
import ru.brombin.ragview.corpus.EvalQuestion;
import ru.brombin.ragview.metrics.ComparisonCsvWriter;
import ru.brombin.ragview.metrics.QueryResult;
import ru.brombin.ragview.metrics.RetrievalEvaluator;
import ru.brombin.ragview.metrics.StrategyEvaluation;
import ru.brombin.ragview.metrics.StrategyResult;
import ru.brombin.ragview.rerank.RerankCsvWriter;
import ru.brombin.ragview.rerank.RerankEvaluation;
import ru.brombin.ragview.rerank.RerankEvaluator;
import ru.brombin.ragview.rerank.RerankMetrics;
import ru.brombin.ragview.rerank.RerankQueryResult;
import ru.brombin.ragview.rerank.RerankScorer;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

@Slf4j
@Component
@RequiredArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE, makeFinal = true)
public class BenchmarkRunner implements ApplicationRunner {

    CorpusLoader corpusLoader;
    StrategyFactory strategyFactory;
    RetrievalEvaluator evaluator;
    ComparisonCsvWriter csvWriter;
    CandidateDumpWriter candidateDumpWriter;
    RerankEvaluator rerankEvaluator;
    RerankCsvWriter rerankCsvWriter;
    ObjectProvider<RerankScorer> rerankScorerProvider;
    RagProperties properties;

    @Override
    public void run(ApplicationArguments args) {
        List<CorpusDocument> corpus = corpusLoader.loadDocuments();
        List<EvalQuestion> questions = corpusLoader.loadQuestions();
        DatasetValidator.validate(corpus, questions);
        int k = properties.topK();

        List<StrategySetup> setups = strategyFactory.build(corpus);
        List<StrategyResult> results = new ArrayList<>();
        List<QueryResult> queryResults = new ArrayList<>();
        for (StrategySetup setup : setups) {
            warmup(setup, questions, k);
            StrategyEvaluation evaluation = evaluator.evaluateDetailed(
                    setup.strategy(), questions, k, setup.counter(),
                    setup.indexEmbeddingRequests(), setup.indexEmbeddingInputs(), setup.indexLlmCalls());
            results.addAll(evaluation.aggregates());
            queryResults.addAll(evaluation.queries());
        }

        printTable(results);
        Path comparisonPath = Path.of(properties.outputPath()).toAbsolutePath();
        for (StrategySetup setup : setups) {
            Path output = comparisonPath.getParent()
                    .resolve(setup.strategy().name() + "_candidates.jsonl");
            candidateDumpWriter.write(
                    output, setup.strategy(), questions, properties.candidateDumpLimit());
            log.info("{} candidates written to {}", setup.strategy().name(), output);
        }
        csvWriter.write(comparisonPath, results);
        csvWriter.writePerQuery(comparisonPath.getParent().resolve("retrieval_per_query.csv"), queryResults);
        runRerank(setups, questions, corpus, comparisonPath.getParent());
    }

    private void runRerank(
            List<StrategySetup> setups,
            List<EvalQuestion> questions,
            List<CorpusDocument> corpus,
            Path outputDirectory) {
        if (!properties.rerankEnabled()) {
            return;
        }
        RerankScorer scorer = rerankScorerProvider.getIfAvailable();
        if (scorer == null) {
            throw new IllegalStateException("Rerank is enabled, but no scorer is configured");
        }
        Map<String, CorpusDocument> documents = corpus.stream()
                .collect(Collectors.toUnmodifiableMap(CorpusDocument::id, Function.identity()));
        List<RerankMetrics> metrics = new ArrayList<>();
        List<RerankQueryResult> rows = new ArrayList<>();
        for (StrategySetup setup : setups) {
            RerankEvaluation evaluation = rerankEvaluator.evaluate(
                    setup.strategy(), questions, documents, properties.rerankPools(),
                    properties.topK(), scorer);
            metrics.addAll(evaluation.metrics());
            rows.addAll(evaluation.queries());
        }
        rerankCsvWriter.writeMetrics(outputDirectory.resolve("rerank_metrics.csv"), metrics);
        rerankCsvWriter.writeQueries(outputDirectory.resolve("rerank_per_query.csv"), rows);
    }

    private void warmup(StrategySetup setup, List<EvalQuestion> questions, int k) {
        int count = Math.min(properties.warmupQueries(), questions.size());
        for (int index = 0; index < count; index++) {
            setup.strategy().retrieve(questions.get(index).question(), k);
        }
    }

    private void printTable(List<StrategyResult> results) {
        log.info("=== Сравнение видов RAG (k={}) ===", properties.topK());
        log.info(String.format("%-10s %-9s %5s %8s %8s %8s %10s %10s",
                "strategy", "slice", "n", "hit@k", "mrr@k", "ndcg@k", "p50(ms)", "p95(ms)"));
        for (StrategyResult r : results) {
            log.info(String.format("%-10s %-9s %5d %8.3f %8.3f %8.3f %10d %10d",
                    r.strategy(), r.slice(), r.queryCount(), r.hitRateAtK(), r.mrrAtK(), r.ndcgAtK(),
                    r.latencyP50Ms(), r.latencyP95Ms()));
        }
    }

}
