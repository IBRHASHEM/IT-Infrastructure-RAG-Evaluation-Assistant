import time

from hybrid_search import HybridSearch
from qwen_generator import QwenGenerator


class HybridRAG:
    """
    Hybrid RAG pipeline.

    Retrieval:
        Vector Search + BM25 + RRF

    Generation:
        Local Qwen model

    Monitoring:
        Retrieval time
        Generation time
        Total time
        Candidate chunks
        Retrieved chunks
        Unique sources
    """

    FALLBACK_ANSWER = (
        "I don't know based on the indexed documentation."
    )

    def __init__(self):

        print("Initializing Hybrid RAG...")

        # -------------------------------------------------
        # Hybrid Retrieval
        # -------------------------------------------------

        self.searcher = HybridSearch(
            vector_k=50,
            bm25_k=50,
            final_k=5,
        )

        # -------------------------------------------------
        # Local Generation
        # -------------------------------------------------

        self.generator = QwenGenerator()

        print("Hybrid RAG initialized successfully.")

    # =====================================================
    # RETRIEVAL
    # =====================================================

    def search(self, question: str, top_k: int = 5):
        """
        Retrieve the most relevant documents.

        Search implementation remains inside HybridSearch.
        """

        results = self.searcher.search(question)

        if not results:
            return []

        return results[:top_k]

    # =====================================================
    # CONTEXT
    # =====================================================

    def _build_context(self, results):
        """
        Convert retrieved documents into the context
        sent to the generation model.
        """

        context_parts = []

        for result in results:

            source = result.get(
                "source",
                "Unknown source"
            )

            page = result.get(
                "page",
                "Unknown page"
            )

            text = result.get(
                "text",
                ""
            )

            if not text:
                continue

            context_parts.append(
                f"Source: {source}\n"
                f"Page: {page}\n"
                f"Content:\n{text}"
            )

        return "\n\n".join(context_parts)

    # =====================================================
    # SOURCES
    # =====================================================

    def _build_sources(self, results):
        """
        Build a unique source list for the UI.
        """

        sources = []

        for result in results:

            source = {
                "source": result.get(
                    "source",
                    "Unknown"
                ),
                "page": result.get(
                    "page",
                    "Unknown"
                ),
            }

            if source not in sources:
                sources.append(source)

        return sources

    # =====================================================
    # ASK
    # =====================================================

    def ask(self, question: str):
        """
        Run the complete RAG pipeline.
        """

        total_start = time.perf_counter()

        # -------------------------------------------------
        # Retrieval
        # -------------------------------------------------

        retrieval_start = time.perf_counter()

        results = self.search(
            question,
            top_k=5
        )

        retrieval_time = (
            time.perf_counter()
            - retrieval_start
        )

        candidate_chunks = getattr(
            self.searcher,
            "last_candidate_count",
            None
        )

        # -------------------------------------------------
        # No results
        # -------------------------------------------------

        if not results:

            total_time = (
                time.perf_counter()
                - total_start
            )

            metrics = {
                "retrieval_time": retrieval_time,
                "generation_time": 0.0,
                "total_time": total_time,
                "candidate_chunks": (
                    candidate_chunks
                    if candidate_chunks is not None
                    else 0
                ),
                "retrieved_chunks": 0,
                "unique_sources": 0,
            }

            self._print_monitoring(metrics)

            return {
                "answer": self.FALLBACK_ANSWER,
                "sources": [],
                "metrics": metrics,
            }

        # -------------------------------------------------
        # Build Context
        # -------------------------------------------------

        context = self._build_context(
            results
        )

        # -------------------------------------------------
        # Generation
        # -------------------------------------------------

        generation_start = time.perf_counter()

        answer = self.generator.generate(
            question=question,
            context=context,
        )

        generation_time = (
            time.perf_counter()
            - generation_start
        )

        # -------------------------------------------------
        # Safety fallback
        # -------------------------------------------------

        if not answer:
            answer = self.FALLBACK_ANSWER

        # -------------------------------------------------
        # Sources
        # -------------------------------------------------

        sources = self._build_sources(
            results
        )

        # -------------------------------------------------
        # Total time
        # -------------------------------------------------

        total_time = (
            time.perf_counter()
            - total_start
        )

        # -------------------------------------------------
        # Monitoring
        # -------------------------------------------------

        metrics = {
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "total_time": total_time,
            "candidate_chunks": (
                candidate_chunks
                if candidate_chunks is not None
                else 10
            ),
            "retrieved_chunks": len(results),
            "unique_sources": len(sources),
        }

        self._print_monitoring(
            metrics
        )

        # -------------------------------------------------
        # Final Result
        # -------------------------------------------------

        return {
            "answer": answer,
            "sources": sources,
            "metrics": metrics,
        }

    # =====================================================
    # MONITORING
    # =====================================================

    @staticmethod
    def _print_monitoring(metrics):
        """
        Print pipeline performance metrics.
        """

        print("=" * 80)
        print("MONITORING")
        print("=" * 80)

        print(
            f"Retrieval time : "
            f"{metrics['retrieval_time']:.2f}s"
        )

        print(
            f"Generation time: "
            f"{metrics['generation_time']:.2f}s"
        )

        print(
            f"Total time     : "
            f"{metrics['total_time']:.2f}s"
        )

        print(
            f"Candidates     : "
            f"{metrics['candidate_chunks']}"
        )

        print(
            f"Chunks used    : "
            f"{metrics['retrieved_chunks']}"
        )

        print(
            f"Unique sources : "
            f"{metrics['unique_sources']}"
        )

        print("=" * 80)


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 80)
    print("HYBRID RAG TEST")
    print("=" * 80)

    rag = HybridRAG()

    test_questions = [
        "What is vSphere HA?",
        "What is VMware vMotion?",
        "What is Windows Server management?",
        "What is an authoritative restore of SYSVOL?",
        "What is a Virtual I/O Server (VIOS)?",
    ]

    for question in test_questions:

        print("\n")
        print("=" * 80)
        print(f"QUESTION: {question}")
        print("=" * 80)

        result = rag.ask(question)

        print("\n")
        print("=" * 80)
        print("FINAL ANSWER")
        print("=" * 80)

        print(
            result["answer"]
        )

        print("\n")
        print("=" * 80)
        print("SOURCES")
        print("=" * 80)

        for source in result["sources"]:

            print(
                f"- {source['source']} "
                f"(Page {source['page']})"
            )

