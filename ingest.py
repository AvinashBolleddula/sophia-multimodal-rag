"""
Dataset ingestion pipeline.

Downloads AI2D and ChartQA datasets, creates text and image entries,
embeds them with separate models, and stores in ChromaDB.

Usage:
    python ingest.py
"""

import os
import time
import warnings
import logging
from datasets import load_dataset
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import chromadb
import torch

from config import (
    AI2D_DIR, CHARTQA_DIR, CHROMA_DIR, DATA_DIR,
    AI2D_DATASET, CHARTQA_DATASET, MAX_SAMPLES_PER_DATASET,
    TEXT_EMBEDDING_MODEL, CLIP_MODEL,
    TEXT_COLLECTION, IMAGE_COLLECTION,
)

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

def ensure_dirs():
    """Create data directories if they don't exist."""
    for d in [AI2D_DIR, CHARTQA_DIR, CHROMA_DIR]:
        os.makedirs(d, exist_ok=True)


def load_ai2d(max_samples: int) -> list[dict]:
    """
    Load AI2D dataset and extract text + image entries.
    
    AI2D contains science diagrams with multiple-choice QA pairs.
    Each sample becomes one text entry and one image entry linked by group_id.
    """
    print(f"Loading AI2D dataset (max {max_samples} samples)...")
    ds = load_dataset(AI2D_DATASET, split="test")

    entries = []
    count = 0
    for idx, sample in enumerate(ds):
        if count >= max_samples:
            break

        # Extract fields
        image: Image.Image = sample["image"]
        question = sample.get("question", "")
        options = sample.get("options", [])
        answer_idx = sample.get("answer", None)

        if not question or not image:
            continue

        # Build answer text
        if options and answer_idx is not None:
            try:
                answer_text = options[int(answer_idx)]
            except (IndexError, ValueError):
                answer_text = str(answer_idx)
        else:
            answer_text = str(answer_idx) if answer_idx is not None else "N/A"

        # Format options string
        options_str = ""
        if options:
            options_str = " | ".join(
                f"{'→ ' if i == int(answer_idx) else ''}{opt}"
                for i, opt in enumerate(options)
            )

        group_id = f"ai2d_{idx:04d}"
        text_id = f"{group_id}_txt"
        image_id = f"{group_id}_img"

        # Save image to disk
        img_path = os.path.join(AI2D_DIR, f"{group_id}.png")
        if not os.path.exists(img_path):
            image.save(img_path)

        # Text entry
        text_content = (
            f"Question: {question}\n"
            f"Options: {options_str}\n"
            f"Correct Answer: {answer_text}"
        )

        entries.append({
            "text_id": text_id,
            "image_id": image_id,
            "group_id": group_id,
            "source": "ai2d",
            "text": text_content,
            "image_path": img_path,
            "question": question,
            "answer": answer_text,
        })
        count += 1

    print(f"  Loaded {len(entries)} AI2D entries.")
    return entries


def load_chartqa(max_samples: int) -> list[dict]:
    """
    Load ChartQA dataset and extract text + image entries.
    
    ChartQA contains chart images with questions that require reading values.
    Each sample becomes one text entry and one image entry linked by group_id.
    """
    print(f"Loading ChartQA dataset (max {max_samples} samples)...")
    ds = load_dataset(CHARTQA_DATASET, split="test")

    entries = []
    count = 0
    for idx, sample in enumerate(ds):
        if count >= max_samples:
            break

        image: Image.Image = sample.get("image", None)
        question = sample.get("query", "") or sample.get("question", "")
        answer = sample.get("label", "") or sample.get("answer", "")

        if not question or not image:
            continue

        group_id = f"chartqa_{idx:04d}"
        text_id = f"{group_id}_txt"
        image_id = f"{group_id}_img"

        # Save image to disk
        img_path = os.path.join(CHARTQA_DIR, f"{group_id}.png")
        if not os.path.exists(img_path):
            image.save(img_path)

        # Text entry
        text_content = (
            f"Question: {question}\n"
            f"Answer: {answer}"
        )

        entries.append({
            "text_id": text_id,
            "image_id": image_id,
            "group_id": group_id,
            "source": "chartqa",
            "text": text_content,
            "image_path": img_path,
            "question": question,
            "answer": str(answer),
        })
        count += 1

    print(f"  Loaded {len(entries)} ChartQA entries.")
    return entries


def embed_texts(entries: list[dict], model: SentenceTransformer) -> list[list[float]]:
    """Embed text entries using sentence-transformers."""
    print(f"  Embedding {len(entries)} text entries...")
    texts = [e["text"] for e in entries]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    return [emb.tolist() for emb in embeddings]


def embed_images(entries: list[dict], clip_model: CLIPModel, clip_processor: CLIPProcessor) -> list[list[float]]:
    """Embed images using CLIP vision encoder."""
    print(f"  Embedding {len(entries)} images with CLIP...")
    embeddings = []
    batch_size = 32

    for i in range(0, len(entries), batch_size):
        batch = entries[i:i + batch_size]
        images = []
        for e in batch:
            try:
                img = Image.open(e["image_path"]).convert("RGB")
                images.append(img)
            except Exception as ex:
                print(f"    Warning: could not load {e['image_path']}: {ex}")
                # Use a blank image as fallback
                images.append(Image.new("RGB", (224, 224), (128, 128, 128)))

        inputs = clip_processor(images=images, return_tensors="pt", padding=True)
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
            if not isinstance(image_features, torch.Tensor):
                image_features = image_features.pooler_output
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        for feat in image_features:
            embeddings.append(feat.cpu().tolist())

        if (i // batch_size) % 5 == 0:
            print(f"    Processed {min(i + batch_size, len(entries))}/{len(entries)} images")

    return embeddings


def store_in_chromadb(entries: list[dict], text_embeddings: list, image_embeddings: list):
    """Store text and image entries in separate ChromaDB collections."""
    print("Storing in ChromaDB...")

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collections if they exist (for clean re-ingestion)
    for name in [TEXT_COLLECTION, IMAGE_COLLECTION]:
        try:
            client.delete_collection(name)
        except Exception:
            pass

    text_col = client.create_collection(
        name=TEXT_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    image_col = client.create_collection(
        name=IMAGE_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    # Prepare text collection data
    text_ids = [e["text_id"] for e in entries]
    text_docs = [e["text"] for e in entries]
    text_metas = [
        {
            "group_id": e["group_id"],
            "source": e["source"],
            "type": "text",
            "question": e["question"],
            "answer": e["answer"],
            "image_path": e["image_path"],  # cross-reference to paired image
        }
        for e in entries
    ]

    # Prepare image collection data
    image_ids = [e["image_id"] for e in entries]
    image_docs = [f"Image for {e['source']} - {e['question']}" for e in entries]
    image_metas = [
        {
            "group_id": e["group_id"],
            "source": e["source"],
            "type": "image",
            "image_path": e["image_path"],
            "question": e["question"],
        }
        for e in entries
    ]

    # Insert in batches (ChromaDB has batch size limits)
    batch_size = 100
    for i in range(0, len(entries), batch_size):
        end = min(i + batch_size, len(entries))
        text_col.add(
            ids=text_ids[i:end],
            embeddings=text_embeddings[i:end],
            documents=text_docs[i:end],
            metadatas=text_metas[i:end],
        )
        image_col.add(
            ids=image_ids[i:end],
            embeddings=image_embeddings[i:end],
            documents=image_docs[i:end],
            metadatas=image_metas[i:end],
        )

    print(f"  Stored {len(entries)} text entries in '{TEXT_COLLECTION}'")
    print(f"  Stored {len(entries)} image entries in '{IMAGE_COLLECTION}'")


def main():
    ensure_dirs()
    start = time.time()

    # --- Load datasets ---
    ai2d_entries = load_ai2d(MAX_SAMPLES_PER_DATASET)
    chartqa_entries = load_chartqa(MAX_SAMPLES_PER_DATASET)
    all_entries = ai2d_entries + chartqa_entries
    print(f"\nTotal entries: {len(all_entries)}")

    # --- Load embedding models ---
    print("\nLoading text embedding model...")
    text_model = SentenceTransformer(TEXT_EMBEDDING_MODEL)

    print("Loading CLIP model...")
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    clip_model.eval()

    # --- Embed ---
    print("\n--- Embedding Text ---")
    text_embeddings = embed_texts(all_entries, text_model)

    print("\n--- Embedding Images ---")
    image_embeddings = embed_images(all_entries, clip_model, clip_processor)

    # --- Store ---
    print("\n--- Storing in ChromaDB ---")
    store_in_chromadb(all_entries, text_embeddings, image_embeddings)

    elapsed = time.time() - start
    print(f"\nIngestion complete in {elapsed:.1f}s")
    print(f"  Text collection: {TEXT_COLLECTION}")
    print(f"  Image collection: {IMAGE_COLLECTION}")
    print(f"  ChromaDB path: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
