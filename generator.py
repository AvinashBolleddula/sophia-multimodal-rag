"""
Grounded answer generation using Anthropic Claude.

Sends retrieved text evidence and image evidence to Claude,
generates answers with citations, and validates that all citations
reference actually-retrieved document IDs.
"""

import os
import re
import base64
from pathlib import Path

import anthropic

from config import ANTHROPIC_MODEL, MAX_GENERATION_TOKENS
from retriever import TextResult, ImageResult


SYSTEM_PROMPT = """You are a grounded multimodal QA system. Your job is to answer questions 
using ONLY the evidence provided below. You must follow these rules strictly:

1. ONLY use information from the provided text evidence and image evidence.
2. CITE every claim using the exact document ID in square brackets, e.g., [ai2d_0042_txt] or [chartqa_0021_img].
3. You may cite multiple sources for a single claim: [chartqa_0021_img] [chartqa_0021_txt].
4. If text and image share the same group (same prefix before _txt/_img), mention that linkage.
5. If the provided evidence is insufficient to answer the question, explicitly state:
   "Insufficient evidence in retrieved documents to answer this question."
6. NEVER invent or hallucinate citations. Only cite IDs from the lists below.
7. When describing image content, base your description on the image you can see AND the paired text evidence.
8. Be concise but thorough. Prefer 2-4 sentences."""


def _encode_image_to_base64(image_path: str) -> tuple[str, str]:
    """Read an image file and return (base64_data, media_type)."""
    path = Path(image_path)
    suffix = path.suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_types.get(suffix, "image/png")

    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return data, media_type


def _build_prompt(
    query: str,
    text_results: list[TextResult],
    image_results: list[ImageResult],
    evidence_sufficient: bool,
) -> list[dict]:
    """
    Build a multimodal message for Claude.

    Includes text evidence as text and image evidence as actual images
    so Claude can see and reason about visual content.
    """
    content_parts = []

    # --- Text evidence section ---
    text_evidence_str = "AVAILABLE TEXT EVIDENCE:\n"
    for r in text_results:
        text_evidence_str += f"\n[{r.id}] (score: {r.score}, source: {r.source}, group: {r.group_id})\n"
        text_evidence_str += f"{r.text}\n"
    text_evidence_str += "\n"

    content_parts.append({"type": "text", "text": text_evidence_str})

    # --- Image evidence section ---
    image_evidence_header = "AVAILABLE IMAGE EVIDENCE:\n"
    for r in image_results:
        image_evidence_header += f"[{r.id}] (score: {r.score}, source: {r.source}, group: {r.group_id}, path: {r.image_path})\n"

    content_parts.append({"type": "text", "text": image_evidence_header})

    # Add actual images for Claude to see (top 3 to manage token usage)
    images_added = 0
    for r in image_results[:3]:
        if os.path.exists(r.image_path):
            try:
                b64_data, media_type = _encode_image_to_base64(r.image_path)
                content_parts.append({
                    "type": "text",
                    "text": f"\nImage [{r.id}]:",
                })
                content_parts.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64_data,
                    },
                })
                images_added += 1
            except Exception as e:
                content_parts.append({
                    "type": "text",
                    "text": f"\n[{r.id}]: (image could not be loaded: {e})",
                })

    # --- Sufficiency warning ---
    if not evidence_sufficient:
        content_parts.append({
            "type": "text",
            "text": "\n⚠️ WARNING: Retrieval scores are low. Evidence may be insufficient. "
                    "If you cannot confidently answer from the evidence above, say so explicitly.\n",
        })

    # --- Query ---
    valid_ids = [r.id for r in text_results] + [r.id for r in image_results]
    content_parts.append({
        "type": "text",
        "text": f"\nVALID CITATION IDS (only cite these): {valid_ids}\n\n"
                f"QUERY: {query}\n\n"
                f"Answer with citations:",
    })

    return content_parts


def validate_citations(answer: str, text_results: list[TextResult], image_results: list[ImageResult]) -> dict:
    """
    Extract all citations from the answer and check they match retrieved IDs.

    Returns a dict with:
      - citations_found: list of cited IDs
      - valid_citations: list of IDs that were actually retrieved
      - hallucinated_citations: list of IDs NOT in retrieval results
    """
    # Extract all [xxx] citations
    cited_ids = re.findall(r'\[([a-zA-Z0-9_]+)\]', answer)

    valid_ids = set(r.id for r in text_results) | set(r.id for r in image_results)

    valid = [c for c in cited_ids if c in valid_ids]
    hallucinated = [c for c in cited_ids if c not in valid_ids]

    return {
        "citations_found": cited_ids,
        "valid_citations": valid,
        "hallucinated_citations": hallucinated,
    }


def generate_answer(
    query: str,
    text_results: list[TextResult],
    image_results: list[ImageResult],
    evidence_sufficient: bool = True,
) -> dict:
    """
    Generate a grounded answer using Anthropic Claude.

    Args:
        query: The user's question
        text_results: Retrieved text documents
        image_results: Retrieved image documents
        evidence_sufficient: Whether retrieval quality meets threshold

    Returns:
        dict with 'answer', 'citation_report', and 'model' fields
    """
    client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var

    content_parts = _build_prompt(query, text_results, image_results, evidence_sufficient)

    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=MAX_GENERATION_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": content_parts},
        ],
    )

    answer = response.content[0].text

    # Validate citations
    citation_report = validate_citations(answer, text_results, image_results)

    return {
        "answer": answer,
        "citation_report": citation_report,
        "model": ANTHROPIC_MODEL,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
