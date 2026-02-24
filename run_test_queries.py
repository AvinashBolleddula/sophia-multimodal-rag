"""
Run all 6 required test queries and save consolidated output.

Usage:
    python run_test_queries.py
"""

import os
from datetime import datetime

from config import LOGS_DIR
from retriever import Retriever
from app import run_query


# The 6 required test queries from the assignment
TEST_QUERIES = [
    # Diagram-Oriented (AI2D)
    {
        "id": 1,
        "category": "Diagram-Oriented",
        "query": "What does the diagram question say about the correct choice? Return the answer and cite the diagram.",
    },
    {
        "id": 2,
        "category": "Diagram-Oriented",
        "query": "Find a relevant AI2D diagram about a process or cycle. Summarize what it shows in 2 sentences. Cite.",
    },
    {
        "id": 3,
        "category": "Diagram-Oriented",
        "query": "Retrieve a diagram and its paired text with the same group_id. Explain how you linked them. Cite both.",
    },
    # Chart-Oriented (ChartQA)
    {
        "id": 4,
        "category": "Chart-Oriented",
        "query": "Find a chart about comparison across categories. State which category is highest. Cite the chart.",
    },
    {
        "id": 5,
        "category": "Chart-Oriented",
        "query": "Find a chart question that requires reading values. Answer it and cite chart + text evidence.",
    },
    {
        "id": 6,
        "category": "Chart-Oriented",
        "query": "Retrieve a chart and produce a one-sentence operator summary (what a technician would do next). Cite.",
    },
]


def main():
    os.makedirs(LOGS_DIR, exist_ok=True)

    print("Initializing retriever...")
    retriever = Retriever()

    all_output = []
    all_output.append("=" * 72)
    all_output.append("SOPHIA SPATIAL AI — MULTIMODAL RAG TEST QUERIES")
    all_output.append(f"Timestamp: {datetime.now().isoformat()}")
    all_output.append("=" * 72)
    all_output.append("")

    for tq in TEST_QUERIES:
        header = f"TEST QUERY {tq['id']} ({tq['category']})"
        print(f"\n{'=' * 60}")
        print(f"Running {header}...")
        print(f"{'=' * 60}")

        all_output.append(f"\n{'#' * 72}")
        all_output.append(f"# {header}")
        all_output.append(f"{'#' * 72}\n")

        output = run_query(retriever, tq["query"], k_text=5, k_image=5)
        print(output)
        all_output.append(output)

    # Save consolidated log
    consolidated = "\n".join(all_output)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOGS_DIR, f"test_queries_{timestamp}.txt")
    with open(log_file, "w") as f:
        f.write(consolidated)

    print(f"\n{'=' * 60}")
    print(f"All 6 test queries complete.")
    print(f"Consolidated log saved to: {log_file}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
