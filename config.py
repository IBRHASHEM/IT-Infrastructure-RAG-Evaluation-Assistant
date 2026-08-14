import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# LLM
MODEL_NAME = "gemini-3.6-flash"

# Local embedding model
EMBEDDING_MODEL = r"D:\Models\bge-small-en-v1.5"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "it_infrastructure"