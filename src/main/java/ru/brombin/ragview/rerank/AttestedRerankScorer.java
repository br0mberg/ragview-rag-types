package ru.brombin.ragview.rerank;

public interface AttestedRerankScorer extends RerankScorer {

    RerankerAttestation attestation();
}
