"""
Configuration constants for the Multimodal RAG system.
"""
import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
AI2D_DIR = os.path.join(DATA_DIR, "ai2d")
CHARTQA_DIR = os.path.join(DATA_DIR, "chartqa")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# --- Dataset ---
AI2D_DATASET = "lmms-lab/ai2d"
CHARTQA_DATASET = "HuggingFaceM4/ChartQA"
MAX_SAMPLES_PER_DATASET = 250  # Keep prototype manageable

# --- Embedding Models ---
TEXT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # sentence-transformers, fast + good quality
CLIP_MODEL = "openai/clip-vit-base-patch32"  # CLIP for cross-modal text↔image

# --- Vector Store ---
TEXT_COLLECTION = "mmrag_text"
IMAGE_COLLECTION = "mmrag_image"

# --- Generation ---
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
MAX_GENERATION_TOKENS = 1024

# --- Retrieval ---
DEFAULT_K_TEXT = 5
DEFAULT_K_IMAGE = 5
LOW_CONFIDENCE_THRESHOLD = 0.3  # Below this, flag as insufficient evidence
