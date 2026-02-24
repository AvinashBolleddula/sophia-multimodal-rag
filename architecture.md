# Architecture Write-Up: Multimodal RAG System

## Embedding Strategy: Separate Spaces

I use **two separate embedding models** rather than a single unified model:

- **Text → Text retrieval**: `all-MiniLM-L6-v2` (sentence-transformers) — optimized for semantic text similarity. Outperforms CLIP's text encoder for text-to-text matching because it was trained specifically on sentence pairs, not image-text pairs.
- **Text → Image retrieval**: `openai/clip-vit-base-patch32` — projects text and images into a shared embedding space via contrastive learning. The query is encoded with CLIP's text encoder and compared against CLIP-encoded images, enabling cross-modal search.

**Why not a single model?** CLIP handles cross-modal alignment well but underperforms dedicated text models for text-to-text similarity. Using the right tool for each retrieval path improves both precision and recall.

## Model Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Text embeddings | all-MiniLM-L6-v2 | Fast, 384-dim, strong text similarity, runs on CPU |
| Image embeddings | CLIP ViT-B/32 | Industry standard for cross-modal retrieval, open-source |
| Vector store | ChromaDB | Zero-config persistent storage, built-in HNSW, metadata filtering |
| Generator | Claude (Anthropic) | Strong instruction following for citation format, native multimodal input |

## Preventing Hallucinated Citations

Three layers of protection:

1. **Prompt constraint**: The system prompt explicitly lists all valid citation IDs and instructs the model to cite only those IDs.
2. **Post-generation validation**: Regex extraction of all `[cited_id]` patterns, checked against the set of retrieved IDs. Any mismatch is flagged as hallucinated.
3. **Insufficient evidence handling**: If all retrieval scores fall below a confidence threshold, the prompt warns the model, and the system expects an explicit "insufficient evidence" response.

## Cross-Modal Pair Enrichment

A key design decision: separate embedding spaces (sentence-transformers vs CLIP) naturally retrieve different documents. Text search might find `ai2d_0042_txt` while image search finds `ai2d_0069_img` — no group_id overlap.

**Solution**: After vector search, the system performs a direct ID lookup to fetch the paired entry for each result via `group_id`. This guarantees cross-modal pairs are always available for grounded generation, while keeping the initial retrieval honest (scores reflect true vector similarity). Enriched entries are marked with `score: -1.0` to distinguish them from vector-retrieved results.

This pattern mirrors production RAG systems where entity linking (e.g., joining equipment records with maintenance docs via equipment_id) supplements semantic search with structured lookups.

## Scaling to 1M+ Documents

| Bottleneck | Current (500 docs) | At 1M+ |
|-----------|-------------------|--------|
| Vector store | ChromaDB (in-process) | Qdrant or Milvus (distributed, sharded) |
| Ingestion | Sequential | GPU-parallel batch embedding, async writes |
| Retrieval latency | ~10ms | Pre-filter by metadata (source, type) → narrower vector search |
| Image storage | Local filesystem | Object storage (S3/GCS) with CDN |
| Index type | HNSW (already used) | HNSW with product quantization for memory efficiency |

Additional optimizations: cache frequent query embeddings, pre-compute group_id cross-references at ingest time, use approximate nearest neighbor with re-ranking for precision-critical queries. For domains with complex entity relationships (e.g., equipment → parts → safety protocols), introduce a knowledge graph layer (GraphRAG) for multi-hop retrieval alongside vector search.

## Evaluation Plan

- **Retrieval quality**: Recall@5 and MRR using ground-truth Q&A pairs from AI2D and ChartQA as relevance labels. Measure separately for text and image retrieval.
- **Citation accuracy**: Automated check — % of generated citations that map to actually-retrieved IDs (target: 100%). % of answers that include at least one citation (target: >95%).
- **Answer faithfulness**: Sample 50 query-answer pairs, manually verify claims are supported by cited evidence. Automate with LLM-as-judge using RAGAS faithfulness metric.
- **Cross-modal coherence**: For queries requiring both text and image evidence, measure how often the system retrieves paired entries (shared group_id).
- **Failure detection**: Measure false-positive rate of the insufficient-evidence detector — does it trigger when it should, and stay silent when evidence is strong?
