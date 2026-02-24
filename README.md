# Sophia Spatial AI — Multimodal RAG System

A minimal but production-aware multimodal Retrieval-Augmented Generation system that retrieves both text and image evidence to generate grounded, cited answers.

## Architecture Overview

```
Query (text)
    │
    ├──→ sentence-transformers ──→ mmrag_text (ChromaDB) ──→ Top-k text results
    │
    └──→ CLIP text encoder ──→ mmrag_image (ChromaDB) ──→ Top-k image results
                                                                │
                        ┌───────────────────────────────────────┘
                        ▼
              Print IDs + scores
                        │
                        ▼
              Claude API (multimodal)
              - Text evidence (as text)
              - Image evidence (as images)
              - Grounded prompt with valid ID list
                        │
                        ▼
              Answer with citations → Validate citations
```

**Two embedding spaces:** `all-MiniLM-L6-v2` for text-to-text retrieval, `CLIP ViT-B/32` for text-to-image cross-modal retrieval.

**Datasets:** AI2D (science diagrams + QA) and ChartQA (charts + QA).

## Setup

### Prerequisites
- Python 3.10+
- Anthropic API key

### Installation

```bash
git clone <repo-url>
cd sophia-multimodal-rag

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Set API Key

```bash
export ANTHROPIC_API_KEY="your-key-here"
```

### Ingest Data

Downloads AI2D and ChartQA datasets, embeds text and images, stores in ChromaDB:

```bash
python ingest.py
```

This takes approximately 5-10 minutes on an M2 MacBook (CPU). Run once; data persists in `chroma_db/`.

## Usage

### Single Query

```bash
python app.py --query "Find a chart about comparison across categories"
```

Options:
- `--k_text N` — number of text results (default: 5)
- `--k_image N` — number of image results (default: 5)
- `--save_log` — save output to `logs/` directory

### Run All 6 Test Queries

```bash
python run_test_queries.py
```

Runs all required test queries and saves consolidated output to `logs/`.

## Project Structure

```
sophia-multimodal-rag/
├── app.py                  # Main entrypoint
├── ingest.py               # Dataset download + embedding + ChromaDB storage
├── retriever.py            # Text and image retrieval with cross-modal pairs
├── generator.py            # Anthropic Claude generation + citation validation
├── config.py               # All configuration constants
├── run_test_queries.py     # Runs all 6 required test queries
├── requirements.txt        # Python dependencies
├── architecture.md         # 1-page architecture write-up
├── README.md               # This file
├── data/                   # Downloaded images (created by ingest.py)
│   ├── ai2d/
│   └── chartqa/
├── chroma_db/              # Persistent vector store (created by ingest.py)
└── logs/                   # Query output logs
```

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Separate embedding spaces | sentence-transformers + CLIP | Better quality per modality vs single model |
| ChromaDB | Persistent, zero-config | Right tool for 4-hour prototype; HNSW built-in |
| Claude multimodal API | Sends actual images | Model can see chart/diagram content directly |
| Citation validation | Post-generation regex check | Catches hallucinated citations automatically |
| group_id linking | Shared prefix for text/image pairs | Enables cross-modal pair verification |
