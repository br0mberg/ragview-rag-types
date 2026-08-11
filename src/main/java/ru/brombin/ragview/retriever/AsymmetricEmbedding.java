package ru.brombin.ragview.retriever;

import java.util.List;

public interface AsymmetricEmbedding {

    List<float[]> embedDocuments(List<String> texts);

    float[] embedQuery(String text);
}
