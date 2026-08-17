from tqdm import tqdm

from document_loader import DocumentLoader
from chunker import TextChunker
from embeddings import EmbeddingGenerator
from vector_store import VectorStore


# =========================================================
# Configuration
# =========================================================

DATA_PATH = "data"
BATCH_SIZE = 64


# =========================================================
# Ingestion Pipeline
# =========================================================

def ingest():

    print("=" * 70)
    print("IT INFRASTRUCTURE RAG - INGESTION PIPELINE")
    print("=" * 70)

    # -----------------------------------------------------
    # 1. Load PDF documents
    # -----------------------------------------------------

    print("\n[1/4] Loading PDF documents...")

    loader = DocumentLoader(
        DATA_PATH
    )

    documents = loader.load()

    if not documents:

        raise RuntimeError(
            "No PDF documents were found in the data directory."
        )

    print(
        f"Loaded {len(documents)} pages with text."
    )

    # -----------------------------------------------------
    # 2. Chunk documents
    # -----------------------------------------------------

    print("\n[2/4] Chunking documents...")

    chunker = TextChunker(
        chunk_size=800,
        overlap=150
    )

    chunks = chunker.split(
        documents
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    # -----------------------------------------------------
    # 3. Generate embeddings
    # -----------------------------------------------------

    print("\n[3/4] Loading embedding model...")

    embedder = EmbeddingGenerator()

    # -----------------------------------------------------
    # 4. Store in ChromaDB
    # -----------------------------------------------------

    print("\n[4/4] Creating ChromaDB index...")

    vector_db = VectorStore()

    # Rebuild the collection from the current PDF dataset.
    vector_db.reset_collection()

    print(
        f"\nGenerating embeddings in batches of {BATCH_SIZE}..."
    )

    for start in tqdm(
        range(
            0,
            len(chunks),
            BATCH_SIZE
        ),
        desc="Embedding"
    ):

        batch = chunks[
            start:start + BATCH_SIZE
        ]

        texts = [
            chunk["text"]
            for chunk in batch
        ]

        embeddings = embedder.embed_batch(
            texts
        )

        vector_db.add_chunks(
            batch,
            embeddings
        )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("INGESTION COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        f"Documents/pages : {len(documents)}"
    )

    print(
        f"Chunks           : {len(chunks)}"
    )

    print(
        "Embedding model  : D:\Models\bge-small-en-v1.5"
    )

    print(
        "Vector database  : ChromaDB"
    )

    print("=" * 70)


# =========================================================
# Entry Point
# =========================================================

if __name__ == "__main__":

    ingest()