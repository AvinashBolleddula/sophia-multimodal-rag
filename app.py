"""
Sophia Spatial AI — Multimodal RAG System

Entrypoint for querying the multimodal RAG system.
Retrieves text and image documents, displays results with scores,
then generates a grounded answer with citations.

Usage:
    python app.py --query "Find a chart about comparison across categories"
    python app.py --query "..." --k_text 5 --k_image 5
"""

import argparse
import os
from datetime import datetime

from config import DEFAULT_K_TEXT, DEFAULT_K_IMAGE, LOGS_DIR
from retriever import Retriever
from generator import generate_answer


def format_text_results(results) -> str:
    """Format text retrieval results for display."""
    lines = []
    for i, r in enumerate(results, 1):
        snippet = r.text[:120].replace('\n', ' ')
        lines.append(f"  [{i}] {r.id:<22} score: {r.score:.4f}  \"{snippet}...\"")
    return "\n".join(lines)


def format_image_results(results) -> str:
    """Format image retrieval results for display."""
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"  [{i}] {r.id:<22} score: {r.score:.4f}  {r.image_path}")
    return "\n".join(lines)


def format_pairs(pairs) -> str:
    """Format cross-modal pairs for display."""
    if not pairs:
        return "  (no cross-modal pairs found in current results)"
    lines = []
    for p in pairs:
        lines.append(
            f"  group_id: {p['group_id']}  →  "
            f"text: {p['text'].id} (score: {p['text'].score:.4f})  |  "
            f"image: {p['image'].id} (score: {p['image'].score:.4f})"
        )
    return "\n".join(lines)


def run_query(retriever: Retriever, query: str, k_text: int, k_image: int) -> str:
    """Run a single query through the full pipeline and return formatted output."""
    output_lines = []

    separator = "=" * 72
    output_lines.append(separator)
    output_lines.append(f"QUERY: {query}")
    output_lines.append(f"Parameters: k_text={k_text}, k_image={k_image}")
    output_lines.append(separator)

    # --- Step 1: Retrieve ---
    text_results = retriever.retrieve_text(query, k=k_text)
    image_results = retriever.retrieve_images(query, k=k_image)

    # --- Step 2: Print retrieval results BEFORE generation ---
    output_lines.append("\n📄 TEXT RETRIEVAL RESULTS:")
    output_lines.append(format_text_results(text_results))

    output_lines.append("\n🖼️  IMAGE RETRIEVAL RESULTS:")
    output_lines.append(format_image_results(image_results))

    # --- Step 3: Cross-modal pairs (natural overlap) ---
    pairs = retriever.find_cross_modal_pairs(text_results, image_results)
    output_lines.append("\n🔗 CROSS-MODAL PAIRS (shared group_id):")
    output_lines.append(format_pairs(pairs))

    # --- Step 3b: Enrich with forced pairs ---
    text_results, image_results = retriever.enrich_with_pairs(text_results, image_results)
    pairs_after = retriever.find_cross_modal_pairs(text_results, image_results)
    new_pairs = [p for p in pairs_after if p not in pairs]
    if new_pairs:
        output_lines.append("\n🔗 ENRICHED PAIRS (fetched via group_id lookup):")
        output_lines.append(format_pairs(new_pairs))

    # --- Step 4: Evidence quality check ---
    evidence_ok = retriever.check_evidence_quality(text_results, image_results)
    if not evidence_ok:
        output_lines.append("\n⚠️  LOW CONFIDENCE: Best retrieval scores are below threshold.")

    # --- Step 5: Generate grounded answer ---
    # Reorder image results: paired images first (most relevant to text evidence)
    text_group_ids = {r.group_id for r in text_results}
    paired_images = [r for r in image_results if r.group_id in text_group_ids]
    unpaired_images = [r for r in image_results if r.group_id not in text_group_ids]
    reordered_images = paired_images + unpaired_images

    output_lines.append("\n💬 GENERATING GROUNDED ANSWER...")
    result = generate_answer(query, text_results, reordered_images, evidence_ok)

    output_lines.append(f"\n{'─' * 72}")
    output_lines.append("ANSWER:")
    output_lines.append(result["answer"])
    output_lines.append(f"{'─' * 72}")

    # --- Step 6: Citation validation ---
    cr = result["citation_report"]
    output_lines.append(f"\n📋 CITATION REPORT:")
    output_lines.append(f"  Citations found:       {cr['citations_found']}")
    output_lines.append(f"  Valid citations:       {cr['valid_citations']}")
    if cr["hallucinated_citations"]:
        output_lines.append(f"  ❌ HALLUCINATED:       {cr['hallucinated_citations']}")
    else:
        output_lines.append(f"  ✅ No hallucinated citations detected.")

    output_lines.append(f"\n  Model: {result['model']}")
    output_lines.append(f"  Tokens: {result['input_tokens']} in / {result['output_tokens']} out")
    output_lines.append(separator + "\n")

    return "\n".join(output_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Sophia Spatial AI — Multimodal RAG System"
    )
    parser.add_argument("--query", type=str, required=True, help="Query to search")
    parser.add_argument("--k_text", type=int, default=DEFAULT_K_TEXT, help="Top-k text results")
    parser.add_argument("--k_image", type=int, default=DEFAULT_K_IMAGE, help="Top-k image results")
    parser.add_argument("--save_log", action="store_true", help="Save output to logs/ directory")
    args = parser.parse_args()

    # Initialize retriever (loads models + connects to ChromaDB)
    retriever = Retriever()

    # Run query
    output = run_query(retriever, args.query, args.k_text, args.k_image)

    # Print to console
    print(output)

    # Optionally save to log file
    if args.save_log:
        os.makedirs(LOGS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(LOGS_DIR, f"query_{timestamp}.txt")
        with open(log_file, "w") as f:
            f.write(output)
        print(f"Log saved to: {log_file}")


if __name__ == "__main__":
    main()
