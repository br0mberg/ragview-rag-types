package ru.brombin.ragview;

import lombok.AccessLevel;
import lombok.experimental.FieldDefaults;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.lang.Nullable;
import org.springframework.stereotype.Component;
import ru.brombin.ragview.config.RagProperties;
import ru.brombin.ragview.corpus.CorpusDocument;
import ru.brombin.ragview.metrics.CallCounter;
import ru.brombin.ragview.retriever.Bm25Index;
import ru.brombin.ragview.retriever.DenseIndex;
import ru.brombin.ragview.strategy.Bm25Rag;
import ru.brombin.ragview.strategy.DenseRag;
import ru.brombin.ragview.strategy.ExplicitArticleReferenceRouter;
import ru.brombin.ragview.strategy.HybridRag;
import ru.brombin.ragview.strategy.RagStrategy;
import ru.brombin.ragview.strategy.RoutingRag;

import java.util.ArrayList;
import java.util.List;

@Component
@FieldDefaults(level = AccessLevel.PRIVATE, makeFinal = true)
public class StrategyFactory {

    EmbeddingModel embeddingModel;
    RagProperties properties;

    public StrategyFactory(@Nullable EmbeddingModel embeddingModel, RagProperties properties) {
        this.embeddingModel = embeddingModel;
        this.properties = properties;
    }

    public List<StrategySetup> build(List<CorpusDocument> corpus) {
        List<StrategySetup> setups = new ArrayList<>();
        for (String name : properties.strategies()) {
            StrategyWithCounter strategy = switch (name) {
                case "dense" -> dense();
                case "bm25" -> bm25();
                case "hybrid" -> hybrid();
                case "routing" -> routing();
                default -> throw new IllegalArgumentException("Unsupported strategy: " + name);
            };
            setups.add(setup(strategy, corpus));
        }
        return List.copyOf(setups);
    }

    private StrategyWithCounter dense() {
        CallCounter counter = new CallCounter();
        DenseIndex denseIdx = new DenseIndex(requireEmbeddingModel(), counter);
        return new StrategyWithCounter(new DenseRag(denseIdx), counter);
    }

    private StrategyWithCounter bm25() {
        CallCounter counter = new CallCounter();
        return new StrategyWithCounter(new Bm25Rag(new Bm25Index()), counter);
    }

    private StrategySetup setup(StrategyWithCounter swc, List<CorpusDocument> corpus) {
        swc.strategy().index(corpus);
        return new StrategySetup(
                swc.strategy(), swc.counter(),
                swc.counter().embeddingRequests(), swc.counter().embeddingInputs(), swc.counter().llmCalls());
    }

    private record StrategyWithCounter(RagStrategy strategy, CallCounter counter) {
    }

    private StrategyWithCounter hybrid() {
        CallCounter counter = new CallCounter();
        DenseIndex dense = new DenseIndex(requireEmbeddingModel(), counter);
        return new StrategyWithCounter(
                new HybridRag(new Bm25Index(), dense, properties.candidatePool(), properties.rrfK()),
                counter);
    }

    private StrategyWithCounter routing() {
        CallCounter counter = new CallCounter();
        DenseIndex dense = new DenseIndex(requireEmbeddingModel(), counter);
        return new StrategyWithCounter(
                new RoutingRag(
                        new Bm25Rag(new Bm25Index()),
                        new HybridRag(
                                new Bm25Index(), dense,
                                properties.candidatePool(), properties.rrfK()),
                        new ExplicitArticleReferenceRouter()),
                counter);
    }

    private EmbeddingModel requireEmbeddingModel() {
        if (embeddingModel == null) {
            throw new IllegalStateException(
                    "Dense, hybrid and routing strategies require the berta or frida Spring profile");
        }
        return embeddingModel;
    }
}
