import os

from embeddings import EmbeddingGenerator


print("=" * 70)
print("OFFLINE EMBEDDING TEST")
print("=" * 70)

# ---------------------------------------------------------
# Disable Hugging Face network access for this test
# ---------------------------------------------------------

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

print("\nOffline mode enabled.")
print("HF_HUB_OFFLINE =", os.environ["HF_HUB_OFFLINE"])
print("TRANSFORMERS_OFFLINE =", os.environ["TRANSFORMERS_OFFLINE"])

# ---------------------------------------------------------
# Load local embedding model
# ---------------------------------------------------------

print("\nLoading embedding generator...")

embedder = EmbeddingGenerator()

# ---------------------------------------------------------
# Test single embedding
# ---------------------------------------------------------

text = "What is VMware vMotion?"

print("\nGenerating single embedding...")

embedding = embedder.embed(text)

print("Embedding generated successfully.")
print("Embedding type:", type(embedding))
print("Embedding dimension:", len(embedding))

# ---------------------------------------------------------
# Test batch embedding
# ---------------------------------------------------------

texts = [
    "What is VMware vMotion?",
    "What is an IBM PowerVM logical partition?",
    "What is a Virtual I/O Server?",
]

print("\nGenerating batch embeddings...")

embeddings = embedder.embed_batch(texts)

print("Batch embeddings generated successfully.")
print("Number of embeddings:", len(embeddings))
print("Embedding dimension:", len(embeddings[0]))

# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

assert len(embedding) == 384
assert len(embeddings) == 3
assert len(embeddings[0]) == 384

print("\n" + "=" * 70)
print("OFFLINE TEST PASSED")
print("=" * 70)

print("\nResults:")
print("Single embedding dimension : 384")
print("Batch embeddings           : 3")
print("Model                     : Local BGE")
print("Internet                  : NOT REQUIRED")
print("=" * 70)