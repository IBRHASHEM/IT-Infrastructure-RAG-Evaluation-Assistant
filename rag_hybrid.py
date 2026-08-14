from hybrid_search import HybridSearch
from qwen_generator import QwenGenerator


class HybridRAG:

    def __init__(self):

        print("Initializing Hybrid RAG...")

        # Hybrid retrieval:
        # Vector Search + BM25 + RRF
        self.searcher = HybridSearch(
            vector_k=50,
            bm25_k=50,
            final_k=5
        )

        # Local Qwen 2.5 3B generation model
        self.generator = QwenGenerator()

        print("Hybrid RAG initialized successfully.")

    # -------------------------------------------------
    # Search
    # -------------------------------------------------

    def search(self, question: str, top_k: int = 5):

        results = self.searcher.search(question)

        if not results:
            return []

        return results[:top_k]

    # -------------------------------------------------
    # Ask
    # -------------------------------------------------

    def ask(self, question: str):

        results = self.search(
            question,
            top_k=5
        )

        if not results:

            return {
                "answer": "I don't know based on the indexed documentation.",
                "sources": []
            }

        print("=" * 80)
        print("Retrieved Documents")
        print("=" * 80)

        for result in results:

            print(
                f"{result['rank']}. "
                f"{result['source']} "
                f"(Page {result['page']}) "
                f"RRF={result['score']:.6f}"
            )

        # -------------------------------------------------
        # Build context
        # -------------------------------------------------

        context_parts = []

        for result in results:

            context_parts.append(
                f"Source: {result['source']}\n"
                f"Page: {result['page']}\n"
                f"Content:\n{result['text']}"
            )

        context = "\n\n".join(context_parts)

        # -------------------------------------------------
        # Generate answer
        # -------------------------------------------------

        print("=" * 80)
        print("Generating answer using local Qwen2.5-3B-Instruct...")
        print("=" * 80)

        answer = self.generator.generate(
            question=question,
            context=context
        )

        # -------------------------------------------------
        # Sources
        # -------------------------------------------------

        sources = []

        for result in results:

            item = {
                "source": result["source"],
                "page": result["page"]
            }

            if item not in sources:
                sources.append(item)

        return {
            "answer": answer,
            "sources": sources
        }


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 80)
    print("HYBRID RAG TEST")
    print("=" * 80)

    rag = HybridRAG()

    question = "What is vSphere HA?"

    result = rag.ask(question)

    print("\n")
    print("=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)

    print(result["answer"])

    print("\n")
    print("=" * 80)
    print("SOURCES")
    print("=" * 80)

    for source in result["sources"]:

        print(
            f"- {source['source']} "
            f"(Page {source['page']})"
        )
