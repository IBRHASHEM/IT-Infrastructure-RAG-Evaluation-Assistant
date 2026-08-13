import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# LLM
MODEL_NAME = "gemini-3.1-flash-lite"

# Local embedding model
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "it_infrastructure"