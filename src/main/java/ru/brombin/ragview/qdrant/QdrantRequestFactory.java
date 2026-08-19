package ru.brombin.ragview.qdrant;

import io.qdrant.client.grpc.Points.Document;
import io.qdrant.client.grpc.Points.PrefetchQuery;
import io.qdrant.client.grpc.Points.QueryPoints;
import io.qdrant.client.grpc.Points.Rrf;
import io.qdrant.client.grpc.Points.SearchParams;

import static io.qdrant.client.QueryFactory.nearest;
import static io.qdrant.client.QueryFactory.rrf;
import static io.qdrant.client.ValueFactory.value;
import static io.qdrant.client.WithPayloadSelectorFactory.enable;

public final class QdrantRequestFactory {

    static final String DENSE_VECTOR = "dense";
    static final String BM25_VECTOR = "bm25";
    static final String BM25_MODEL = "qdrant/bm25";

    private QdrantRequestFactory() {
    }

    public static QueryPoints dense(String collection, float[] vector, int limit) {
        return QueryPoints.newBuilder()
                .setCollectionName(collection)
                .setQuery(nearest(vector))
                .setUsing(DENSE_VECTOR)
                .setParams(exactSearch())
                .setLimit(limit)
                .setWithPayload(enable(true))
                .build();
    }

    public static QueryPoints bm25(String collection, String query, int limit, double averageLength) {
        return QueryPoints.newBuilder()
                .setCollectionName(collection)
                .setQuery(nearest(bm25Document(query, averageLength)))
                .setUsing(BM25_VECTOR)
                .setLimit(limit)
                .setWithPayload(enable(true))
                .build();
    }

    public static QueryPoints hybrid(
            String collection,
            float[] vector,
            String query,
            int candidateLimit,
            int resultLimit,
            int oneBasedRrfK,
            double averageLength) {
        PrefetchQuery dense = PrefetchQuery.newBuilder()
                .setQuery(nearest(vector))
                .setUsing(DENSE_VECTOR)
                .setParams(exactSearch())
                .setLimit(candidateLimit)
                .build();
        PrefetchQuery bm25 = PrefetchQuery.newBuilder()
                .setQuery(nearest(bm25Document(query, averageLength)))
                .setUsing(BM25_VECTOR)
                .setLimit(candidateLimit)
                .build();
        return QueryPoints.newBuilder()
                .setCollectionName(collection)
                .addPrefetch(dense)
                .addPrefetch(bm25)
                .setQuery(rrf(Rrf.newBuilder().setK(oneBasedRrfK + 1).build()))
                .setLimit(resultLimit)
                .setWithPayload(enable(true))
                .build();
    }

    static Document bm25Document(String text, double averageLength) {
        return Document.newBuilder()
                .setText(text)
                .setModel(BM25_MODEL)
                .putOptions("language", value("russian"))
                .putOptions("tokenizer", value("multilingual"))
                .putOptions("k", value(1.2))
                .putOptions("b", value(0.75))
                .putOptions("avg_len", value(averageLength))
                .build();
    }

    private static SearchParams exactSearch() {
        return SearchParams.newBuilder().setExact(true).build();
    }
}
