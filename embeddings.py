from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_PATH


class EmbeddingGenerator:

    def __init__(self):

        print("Loading local BGE model...")
        print(f"Model path: {EMBEDDING_MODEL_PATH}")

        self.model = SentenceTransformer(
            EMBEDDING_MODEL_PATH,
            local_files_only=True,
            device="cpu"
        )

        print("Local BGE model loaded.")

    def embed(self, text):

        return self.model.encode(
            text,
            normalize_embeddings=True
        ).tolist()

    def embed_batch(self, texts):

        return self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False
        ).tolist()