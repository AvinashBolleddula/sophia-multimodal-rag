"""
Retrieval module for multimodal RAG.

Two retrieval paths:
  - Text retrieval: query → sentence-transformers embedding → search mmrag_text
  - Image retrieval: query → CLIP text embedding → search mmrag_image

Also supports cross-modal pair detection via group_id linking.
"""
import warnings
import logging
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import chromadb
import torch

from config import (
    CHROMA_DIR, TEXT_COLLECTION, IMAGE_COLLECTION,
    TEXT_EMBEDDING_MODEL, CLIP_MODEL, LOW_CONFIDENCE_THRESHOLD,
)

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

@dataclass
class TextResult:
    id: str
    score: float
    text: str
    group_id: str
    source: str
    question: str
    answer: str
    image_path: str  # cross-reference to paired image


@dataclass
class ImageResult:
    id: str
    score: float
    image_path: str
    group_id: str
    source: str
    question: str


class Retriever:
    """
    Multimodal retriever with separate embedding spaces.

    - sentence-transformers for text-to-text retrieval (better text similarity)
    - CLIP for text-to-image retrieval (cross-modal alignment)
    """

    def __init__(self):
        print("Loading retrieval models...")

        # Text embedding model
        self.text_model = SentenceTransformer(TEXT_EMBEDDING_MODEL)

        # CLIP model for cross-modal retrieval
        self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL)
        self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
        self.clip_model.eval()

        # ChromaDB client
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.text_col = self.client.get_collection(TEXT_COLLECTION)
        self.image_col = self.client.get_collection(IMAGE_COLLECTION)

        print(f"  Text collection: {self.text_col.count()} documents")
        print(f"  Image collection: {self.image_col.count()} documents")
        print("  Retriever ready.\n")

    def retrieve_text(self, query: str, k: int = 5) -> list[TextResult]:
        """
        Retrieve top-k text documents using sentence-transformers embedding.

        Uses cosine similarity in the text embedding space.
        """
        query_embedding = self.text_model.encode(query).tolist()

        results = self.text_col.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        text_results = []
        for i in range(len(results["ids"][0])):
            # ChromaDB returns distances; for cosine space, distance = 1 - similarity
            distance = results["distances"][0][i]
            score = 1.0 - distance  # Convert to similarity score

            meta = results["metadatas"][0][i]
            text_results.append(TextResult(
                id=results["ids"][0][i],
                score=round(score, 4),
                text=results["documents"][0][i],
                group_id=meta.get("group_id", ""),
                source=meta.get("source", ""),
                question=meta.get("question", ""),
                answer=meta.get("answer", ""),
                image_path=meta.get("image_path", ""),
            ))

        return text_results

    def retrieve_images(self, query: str, k: int = 5) -> list[ImageResult]:
        """
        Retrieve top-k images using CLIP text encoder.

        The query text is encoded with CLIP's text encoder into the same
        embedding space as the CLIP-encoded images, enabling cross-modal search.
        """
        # Encode query with CLIP text encoder
        inputs = self.clip_processor(text=[query], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
            if not isinstance(text_features, torch.Tensor):
                text_features = text_features.pooler_output
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        query_embedding = text_features[0].cpu().tolist()

        results = self.image_col.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        image_results = []
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            score = 1.0 - distance

            meta = results["metadatas"][0][i]
            image_results.append(ImageResult(
                id=results["ids"][0][i],
                score=round(score, 4),
                image_path=meta.get("image_path", ""),
                group_id=meta.get("group_id", ""),
                source=meta.get("source", ""),
                question=meta.get("question", ""),
            ))

        return image_results

    def find_cross_modal_pairs(
        self,
        text_results: list[TextResult],
        image_results: list[ImageResult],
    ) -> list[dict]:
        """
        Find entries that appear in both text and image results via group_id.

        This is critical for queries that ask to link a diagram with its
        paired text — proving cross-modal retrieval coherence.
        """
        text_groups = {r.group_id: r for r in text_results}
        image_groups = {r.group_id: r for r in image_results}

        shared_groups = set(text_groups.keys()) & set(image_groups.keys())

        pairs = []
        for gid in shared_groups:
            pairs.append({
                "group_id": gid,
                "text": text_groups[gid],
                "image": image_groups[gid],
            })

        return pairs

    def enrich_with_pairs(
        self,
        text_results: list[TextResult],
        image_results: list[ImageResult],
    ) -> tuple[list[TextResult], list[ImageResult]]:
        """
        Enrich retrieval results by force-fetching paired entries.

        For each top text result, if its paired image isn't already in
        image_results, fetch it directly by ID. And vice versa.
        This guarantees cross-modal pairs exist for grounded generation.
        """
        existing_image_ids = {r.id for r in image_results}
        existing_text_ids = {r.id for r in text_results}

        # For top text results, fetch their paired images
        for tr in text_results:
            paired_img_id = f"{tr.group_id}_img"
            if paired_img_id not in existing_image_ids:
                try:
                    result = self.image_col.get(
                        ids=[paired_img_id],
                        include=["documents", "metadatas"],
                    )
                    if result["ids"]:
                        meta = result["metadatas"][0]
                        image_results.append(ImageResult(
                            id=paired_img_id,
                            score=-1.0,  # -1 indicates forced pair, not vector search
                            image_path=meta.get("image_path", ""),
                            group_id=meta.get("group_id", ""),
                            source=meta.get("source", ""),
                            question=meta.get("question", ""),
                        ))
                        existing_image_ids.add(paired_img_id)
                except Exception:
                    pass

        # For top image results, fetch their paired texts
        for ir in image_results:
            if ir.score == -1.0:
                continue  # Skip already-forced entries
            paired_txt_id = f"{ir.group_id}_txt"
            if paired_txt_id not in existing_text_ids:
                try:
                    result = self.text_col.get(
                        ids=[paired_txt_id],
                        include=["documents", "metadatas"],
                    )
                    if result["ids"]:
                        meta = result["metadatas"][0]
                        text_results.append(TextResult(
                            id=paired_txt_id,
                            score=-1.0,
                            text=result["documents"][0],
                            group_id=meta.get("group_id", ""),
                            source=meta.get("source", ""),
                            question=meta.get("question", ""),
                            answer=meta.get("answer", ""),
                            image_path=meta.get("image_path", ""),
                        ))
                        existing_text_ids.add(paired_txt_id)
                except Exception:
                    pass

        return text_results, image_results


    def check_evidence_quality(
        self,
        text_results: list[TextResult],
        image_results: list[ImageResult],
    ) -> bool:
        """
        Check if retrieval quality is sufficient for grounded generation.

        Returns False if the best scores are below the confidence threshold,
        indicating the system should flag insufficient evidence.
        """
        best_text_score = max((r.score for r in text_results), default=0.0)
        best_image_score = max((r.score for r in image_results), default=0.0)
        best_overall = max(best_text_score, best_image_score)

        return best_overall >= LOW_CONFIDENCE_THRESHOLD
