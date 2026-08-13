from embeddings import EmbeddingGenerator

embedder = EmbeddingGenerator()

texts = [
    "Hello world",
    "This is a test",
    "IBM PowerVM",
]

try:
    vectors = embedder.embed_batch(texts)

    print("Success")
    print(len(vectors))
    print(len(vectors[0]))

except Exception as e:
    import traceback
    traceback.print_exc()