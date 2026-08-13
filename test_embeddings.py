from embeddings import EmbeddingGenerator

embedder = EmbeddingGenerator()

vector = embedder.embed(
    "What is Active Directory?"
)

print(type(vector))
print(len(vector))
print(vector[:10])