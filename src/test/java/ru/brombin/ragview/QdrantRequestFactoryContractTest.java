package ru.brombin.ragview;

import io.qdrant.client.grpc.Points.Query.VariantCase;
import org.junit.jupiter.api.Test;
import ru.brombin.ragview.qdrant.QdrantRequestFactory;

import static org.assertj.core.api.Assertions.assertThat;

class QdrantRequestFactoryContractTest {

    @Test
    void dense_shouldForceExactSearch() {
        var request = QdrantRequestFactory.dense("tax", new float[]{1, 0}, 50);

        assertThat(request.getUsing()).isEqualTo("dense");
        assertThat(request.getLimit()).isEqualTo(50);
        assertThat(request.getParams().getExact()).isTrue();
        assertThat(request.getWithPayload().getEnable()).isTrue();
    }

    @Test
    void bm25_shouldUseRussianTextProcessing() {
        var request = QdrantRequestFactory.bm25("tax", "статья 107", 50, 120.5);
        var document = request.getQuery().getNearest().getDocument();

        assertThat(request.getUsing()).isEqualTo("bm25");
        assertThat(request.getWithPayload().getEnable()).isTrue();
        assertThat(document.getModel()).isEqualTo("qdrant/bm25");
        assertThat(document.getOptionsOrThrow("language").getStringValue()).isEqualTo("russian");
        assertThat(document.getOptionsOrThrow("tokenizer").getStringValue()).isEqualTo("multilingual");
        assertThat(document.getOptionsOrThrow("avg_len").getDoubleValue()).isEqualTo(120.5);
    }

    @Test
    void hybrid_shouldUseSameBranchesAndTranslateOneBasedK() {
        var request = QdrantRequestFactory.hybrid(
                "tax", new float[]{1, 0}, "статья 107", 50, 10, 60, 120.5);

        assertThat(request.getPrefetchList()).hasSize(2);
        assertThat(request.getPrefetch(0).getUsing()).isEqualTo("dense");
        assertThat(request.getPrefetch(0).getParams().getExact()).isTrue();
        assertThat(request.getPrefetch(0).getLimit()).isEqualTo(50);
        assertThat(request.getPrefetch(1).getUsing()).isEqualTo("bm25");
        assertThat(request.getPrefetch(1).getLimit()).isEqualTo(50);
        assertThat(request.getQuery().getVariantCase()).isEqualTo(VariantCase.RRF);
        assertThat(request.getQuery().getRrf().getK()).isEqualTo(61);
        assertThat(request.getLimit()).isEqualTo(10);
        assertThat(request.getWithPayload().getEnable()).isTrue();
    }
}
