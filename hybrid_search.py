from rank_bm25 import BM25Okapi
import hashlib

from embeddings import EmbeddingGenerator
from vector_store import VectorStore


class HybridSearch:

    def __init__(self, vector_k=50, bm25_k=50, final_k=10):

        self.vector_k = vector_k
        self.bm25_k = bm25_k
        self.final_k = final_k

        print("Loading vector store...")
        self.vector_db = VectorStore()

        print("Loading embedding model...")
        self.embedder = EmbeddingGenerator()

        print("Loading documents from Chroma...")

        data = self.vector_db.collection.get(
            include=["documents", "metadatas"]
        )

        self.documents = data["documents"]
        self.metadatas = data["metadatas"]
        self.ids = data["ids"]

        print(
            f"Loaded {len(self.documents)} documents from Chroma."
        )

        # ---------------------------------------------------------
        # Prepare BM25 corpus
        # ---------------------------------------------------------

        tokenized_documents = [
            self.tokenize(text)
            for text in self.documents
        ]

        print("Building BM25 index...")

        self.bm25 = BM25Okapi(
            tokenized_documents
        )

        print("Hybrid search initialized.")

    # -------------------------------------------------------------
    # Tokenization
    # -------------------------------------------------------------

    @staticmethod
    def tokenize(text):

        return text.lower().split()

    # -------------------------------------------------------------
    # Vector Search
    # -------------------------------------------------------------

    def vector_search(self, query):

        query_embedding = self.embedder.embed_batch(
            [query]
        )[0]

        results = self.vector_db.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.vector_k,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        vector_results = []

        for rank, (
            doc_id,
            document,
            metadata,
            distance
        ) in enumerate(
            zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ),
            start=1
        ):

            vector_results.append(
                {
                    "id": doc_id,
                    "document": document,
                    "metadata": metadata,
                    "distance": distance,
                    "rank": rank
                }
            )

        return vector_results

    # -------------------------------------------------------------
    # BM25 Search
    # -------------------------------------------------------------

    def bm25_search(self, query):

        query_tokens = self.tokenize(query)

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        results = []

        for rank, index in enumerate(
            ranked_indexes[:self.bm25_k],
            start=1
        ):

            results.append(
                {
                    "id": self.ids[index],
                    "document": self.documents[index],
                    "metadata": self.metadatas[index],
                    "score": float(scores[index]),
                    "rank": rank
                }
            )

        return results

    # -------------------------------------------------------------
    # Content Hash
    # -------------------------------------------------------------

    @staticmethod
    def content_hash(text):
        """
        Create a stable hash for document content.
        Used to identify identical chunks.
        """

        normalized = text.strip()

        return hashlib.md5(
            normalized.encode("utf-8")
        ).hexdigest()

    # -------------------------------------------------------------
    # Reciprocal Rank Fusion
    # -------------------------------------------------------------

    def reciprocal_rank_fusion(
        self,
        vector_results,
        bm25_results,
        rrf_k=20
    ):
        """
        Combine Vector Search and BM25 results
        using Reciprocal Rank Fusion.

        Scores are aggregated at source level.
        The best-ranked chunk from each source
        is retained as the representative result.
        """

        source_scores = {}

        source_results = {}

        # ---------------------------------------------------------
        # Vector results
        # ---------------------------------------------------------

        for result in vector_results:

            source = result["metadata"]["source"]

            score = 1 / (
                rrf_k + result["rank"]
            )

            source_scores[source] = (
                source_scores.get(source, 0)
                + score
            )

            # Keep the best-ranked chunk
            # from this source

            if (
                source not in source_results
                or result["rank"]
                < source_results[source]["rank"]
            ):

                source_results[source] = result

        # ---------------------------------------------------------
        # BM25 results
        # ---------------------------------------------------------

        for result in bm25_results:

            source = result["metadata"]["source"]

            score = 1 / (
                rrf_k + result["rank"]
            )

            source_scores[source] = (
                source_scores.get(source, 0)
                + score
            )

            # Keep the best-ranked chunk
            # from this source

            if (
                source not in source_results
                or result["rank"]
                < source_results[source]["rank"]
            ):

                source_results[source] = result

        # ---------------------------------------------------------
        # Rank sources
        # ---------------------------------------------------------

        ranked_sources = sorted(
            source_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # ---------------------------------------------------------
        # Build final results
        # ---------------------------------------------------------

        final_results = []

        seen_content = set()

        for source, source_score in ranked_sources:

            result = source_results[source]

            # -----------------------------------------------------
            # Remove identical chunks
            # -----------------------------------------------------

            content_hash = self.content_hash(
                result["document"]
            )

            if content_hash in seen_content:
                continue

            seen_content.add(
                content_hash
            )

            final_results.append(
                {
                    "rank": len(final_results) + 1,
                    "id": result["id"],
                    "source": source,
                    "page": result["metadata"]["page"],
                    "score": source_score,
                    "text": result["document"]
                }
            )

            if len(final_results) >= self.final_k:
                break

        return final_results

    # -------------------------------------------------------------
    # Hybrid Search
    # -------------------------------------------------------------

    def search(self, query):

        vector_results = self.vector_search(
            query
        )

        bm25_results = self.bm25_search(
            query
        )

        return self.reciprocal_rank_fusion(
            vector_results,
            bm25_results
        )


# =================================================================
# Test
# =================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("HYBRID SEARCH TEST")
    print("=" * 70)

    searcher = HybridSearch()

    queries = [
        "What is VMware vMotion?",
        "What is a logical partition (LPAR) in IBM PowerVM?",
        "What is an authoritative restore of SYSVOL?",
        "What is a Virtual I/O Server (VIOS)?",
    ]

    for query in queries:

        print("\n" + "=" * 70)
        print(f"Question: {query}")
        print("=" * 70)

        results = searcher.search(
            query
        )

        for result in results:

            print(
                f"{result['rank']}. "
                f"{result['source']} "
                f"(Page {result['page']}) "
                f"RRF={result['score']:.6f}"
            )