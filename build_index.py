from tqdm import tqdm
import time
from document_loader import DocumentLoader
from chunker import TextChunker
from embeddings import EmbeddingGenerator
from vector_store import VectorStore


BATCH_SIZE = 64


def main():

    print("=" * 60)
    print("Loading documents...")
    print("=" * 60)

    loader = DocumentLoader("data")
    documents = loader.load()

    print(f"\nDocuments : {len(documents)}")

    print("\nChunking...")

    chunker = TextChunker()

    chunks = chunker.split(documents)

    print(f"Chunks : {len(chunks)}")

    embedder = EmbeddingGenerator()
    vector_db = VectorStore()

    print("\nCreating embeddings...\n")

    for i in tqdm(range(0, len(chunks), BATCH_SIZE)):

        batch = chunks[i:i + BATCH_SIZE]

        texts = [
            chunk["text"]
            for chunk in batch
        ]

        embeddings = embedder.embed_batch(texts)
        vector_db.add_chunks(batch, embeddings)
         
    print("\nDone!")

    print(f"Indexed {len(chunks)} chunks.")


if __name__ == "__main__":
    main()