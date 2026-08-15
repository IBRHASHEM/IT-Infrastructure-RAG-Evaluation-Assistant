import os

from dotenv import load_dotenv

load_dotenv()


# =========================================================
# Generation Model
# =========================================================
MODEL_PATH = r"D:\Models\Qwen2.5-0.5B-Instruct"
MODEL_NAME = "Qwen2.5-0.5B-Instruct (Local)"
MODEL_NAME = r"D:\Models\Qwen2.5-0.5B-Instruct"


# =========================================================
# Local Embedding Model
# =========================================================

EMBEDDING_MODEL_PATH = r"D:\Models\bge-small-en-v1.5"


# =========================================================
# ChromaDB
# =========================================================

CHROMA_PATH = "chroma_db"

COLLECTION_NAME = "it_infrastructure"