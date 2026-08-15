from embeddings import EmbeddingGenerator
from vector_store import VectorStore
from qwen_generator import QwenGenerator


class RAG:

    def __init__(self):

        print("Initializing RAG...")

        # Local embedding model
        self.embedder = EmbeddingGenerator()

        # ChromaDB vector store
        self.vector_db = VectorStore()

        # Local Qwen generation model
        self.generator = QwenGenerator()

        print("RAG initialized successfully.")

    # -------------------------------------------------
    # Search
    # -------------------------------------------------

    def search(self, question: str, k: int = 5):

        query_embedding = self.embedder.embed(question)

        results = self.vector_db.search(
            query_embedding,
            k=k
        )

        return results

    # -------------------------------------------------
    # Ask
    # -------------------------------------------------

    def ask(self, question: str):

        results = self.search(question, k=5)

        # No search results
        if (
            not results.get("documents")
            or not results["documents"][0]
        ):
            return {
                "answer": "I don't know based on the indexed documentation.",
                "sources": []
            }

        # -------------------------------------------------
        # Retrieved documents
        # -------------------------------------------------

        print("=" * 80)
        print("Retrieved Documents")
        print("=" * 80)

        for i, meta in enumerate(
            results["metadatas"][0],
            start=1
        ):

            distance = results["distances"][0][i - 1]

            print(
                f"{i}. {meta['source']} "
                f"(Page {meta['page']}) "
                f"Distance={distance:.3f}"
            )

        # -------------------------------------------------
        # Build context
        # -------------------------------------------------

        context = "\n\n".join(
            results["documents"][0]
        )

        # -------------------------------------------------
        # Local Qwen
        # -------------------------------------------------

        print("=" * 80)
        print("Generating answer using local Qwen...")
        print("=" * 80)

        answer = self.generator.generate(
            question=question,
            context=context
        )

        # -------------------------------------------------
        # Sources
        # -------------------------------------------------

        sources = []

        for i, meta in enumerate(results["metadatas"][0]):

            distance = results["distances"][0][i] if "distances" in results and len(results["distances"][0]) > i else 0.0

            item = {
                "source": meta["source"],
                "page": meta["page"],
                "text": results["documents"][0][i] if "documents" in results and len(results["documents"][0]) > i else "",
                "score": float(distance),
                "vector_rank": i + 1,
                "bm25_rank": None
            }

            if item not in sources:
                sources.append(item)

        return {
            "answer": answer,
            "sources": sources
        } 