import chromadb

from config import (
    CHROMA_PATH,
    COLLECTION_NAME
)


class VectorStore:

    def __init__(self):

        self.client = chromadb.PersistentClient(
            path=CHROMA_PATH
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks, embeddings):

        self.collection.add(
            ids=[c["id"] for c in chunks],

            documents=[c["text"] for c in chunks],

            embeddings=embeddings,

            metadatas=[
                {
                    "source": c["source"],
                    "page": c["page"]
                }
                for c in chunks
            ]
        )

    def search(self, embedding, k=5):

        return self.collection.query(
            query_embeddings=[embedding],
            n_results=k
        )