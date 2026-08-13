from document_loader import DocumentLoader
from chunker import TextChunker

loader = DocumentLoader("data")
documents = loader.load()

chunker = TextChunker(
    chunk_size=800,
    overlap=150
)

chunks = chunker.split(documents)

print(f"\nDocuments : {len(documents)}")
print(f"Chunks    : {len(chunks)}")

print("\nFirst Chunk\n")
print("=" * 80)
print(chunks[0]["text"][:500])
print("=" * 80)