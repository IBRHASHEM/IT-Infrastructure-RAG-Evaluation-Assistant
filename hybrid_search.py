# from rank_bm25 import BM25Okapi
# import hashlib

# from embeddings import EmbeddingGenerator
# from vector_store import VectorStore


# class HybridSearch:

#     def __init__(self, vector_k=50, bm25_k=50, final_k=10):

#         self.vector_k = vector_k
#         self.bm25_k = bm25_k
#         self.final_k = final_k

#         print("Loading vector store...")
#         self.vector_db = VectorStore()

#         print("Loading embedding model...")
#         self.embedder = EmbeddingGenerator()

#         print("Loading documents from Chroma...")

#         data = self.vector_db.collection.get(
#             include=["documents", "metadatas"]
#         )

#         self.documents = data["documents"]
#         self.metadatas = data["metadatas"]
#         self.ids = data["ids"]

#         print(
#             f"Loaded {len(self.documents)} documents from Chroma."
#         )

#         # ---------------------------------------------------------
#         # Prepare BM25 corpus
#         # ---------------------------------------------------------

#         tokenized_documents = [
#             self.tokenize(text)
#             for text in self.documents
#         ]

#         print("Building BM25 index...")

#         self.bm25 = BM25Okapi(
#             tokenized_documents
#         )

#         print("Hybrid search initialized.")

#     # -------------------------------------------------------------
#     # Tokenization
#     # -------------------------------------------------------------

#     @staticmethod
#     def tokenize(text):

#         return text.lower().split()

#     # -------------------------------------------------------------
#     # Vector Search
#     # -------------------------------------------------------------

#     def vector_search(self, query):

#         query_embedding = self.embedder.embed_batch(
#             [query]
#         )[0]

#         results = self.vector_db.collection.query(
#             query_embeddings=[query_embedding],
#             n_results=self.vector_k,
#             include=[
#                 "documents",
#                 "metadatas",
#                 "distances"
#             ]
#         )

#         vector_results = []

#         for rank, (
#             doc_id,
#             document,
#             metadata,
#             distance
#         ) in enumerate(
#             zip(
#                 results["ids"][0],
#                 results["documents"][0],
#                 results["metadatas"][0],
#                 results["distances"][0]
#             ),
#             start=1
#         ):

#             vector_results.append(
#                 {
#                     "id": doc_id,
#                     "document": document,
#                     "metadata": metadata,
#                     "distance": distance,
#                     "rank": rank
#                 }
#             )

#         return vector_results

#     # -------------------------------------------------------------
#     # BM25 Search
#     # -------------------------------------------------------------

#     def bm25_search(self, query):

#         query_tokens = self.tokenize(query)

#         scores = self.bm25.get_scores(
#             query_tokens
#         )

#         ranked_indexes = sorted(
#             range(len(scores)),
#             key=lambda i: scores[i],
#             reverse=True
#         )

#         results = []

#         for rank, index in enumerate(
#             ranked_indexes[:self.bm25_k],
#             start=1
#         ):

#             results.append(
#                 {
#                     "id": self.ids[index],
#                     "document": self.documents[index],
#                     "metadata": self.metadatas[index],
#                     "score": float(scores[index]),
#                     "rank": rank
#                 }
#             )

#         return results

#     # -------------------------------------------------------------
#     # Content Hash
#     # -------------------------------------------------------------

#     @staticmethod
#     def content_hash(text):
#         """
#         Create a stable hash for document content.
#         Used to identify identical chunks.
#         """

#         normalized = text.strip()

#         return hashlib.md5(
#             normalized.encode("utf-8")
#         ).hexdigest()

#     # -------------------------------------------------------------
#     # Reciprocal Rank Fusion
#     # -------------------------------------------------------------

#     def reciprocal_rank_fusion(
#         self,
#         vector_results,
#         bm25_results,
#         rrf_k=20
#     ):
#         """
#         Combine Vector Search and BM25 results
#         using Reciprocal Rank Fusion.

#         Scores are aggregated at source level.
#         The best-ranked chunk from each source
#         is retained as the representative result.
#         """

#         source_scores = {}

#         source_results = {}

#         # ---------------------------------------------------------
#         # Vector results
#         # ---------------------------------------------------------

#         for result in vector_results:

#             source = result["metadata"]["source"]

#             score = 1 / (
#                 rrf_k + result["rank"]
#             )

#             source_scores[source] = (
#                 source_scores.get(source, 0)
#                 + score
#             )

#             # Keep the best-ranked chunk
#             # from this source

#             if (
#                 source not in source_results
#                 or result["rank"]
#                 < source_results[source]["rank"]
#             ):

#                 source_results[source] = result

#         # ---------------------------------------------------------
#         # BM25 results
#         # ---------------------------------------------------------

#         for result in bm25_results:

#             source = result["metadata"]["source"]

#             score = 1 / (
#                 rrf_k + result["rank"]
#             )

#             source_scores[source] = (
#                 source_scores.get(source, 0)
#                 + score
#             )

#             # Keep the best-ranked chunk
#             # from this source

#             if (
#                 source not in source_results
#                 or result["rank"]
#                 < source_results[source]["rank"]
#             ):

#                 source_results[source] = result

#         # ---------------------------------------------------------
#         # Rank sources
#         # ---------------------------------------------------------

#         ranked_sources = sorted(
#             source_scores.items(),
#             key=lambda x: x[1],
#             reverse=True
#         )

#         # ---------------------------------------------------------
#         # Build final results
#         # ---------------------------------------------------------

#         final_results = []

#         seen_content = set()

#         for source, source_score in ranked_sources:

#             result = source_results[source]

#             # -----------------------------------------------------
#             # Remove identical chunks
#             # -----------------------------------------------------

#             content_hash = self.content_hash(
#                 result["document"]
#             )

#             if content_hash in seen_content:
#                 continue

#             seen_content.add(
#                 content_hash
#             )

#             final_results.append(
#                 {
#                     "rank": len(final_results) + 1,
#                     "id": result["id"],
#                     "source": source,
#                     "page": result["metadata"]["page"],
#                     "score": source_score,
#                     "text": result["document"]
#                 }
#             )

#             if len(final_results) >= self.final_k:
#                 break

#         return final_results

#     # -------------------------------------------------------------
#     # Hybrid Search
#     # -------------------------------------------------------------

#     def search(self, query):

#         vector_results = self.vector_search(
#             query
#         )

#         bm25_results = self.bm25_search(
#             query
#         )

#         return self.reciprocal_rank_fusion(
#             vector_results,
#             bm25_results
#         )


# # =================================================================
# # Test
# # =================================================================

# if __name__ == "__main__":

#     print("=" * 70)
#     print("HYBRID SEARCH TEST")
#     print("=" * 70)

#     searcher = HybridSearch()

#     queries = [
#         "What is VMware vMotion?",
#         "What is a logical partition (LPAR) in IBM PowerVM?",
#         "What is an authoritative restore of SYSVOL?",
#         "What is a Virtual I/O Server (VIOS)?",
#     ]

#     for query in queries:

#         print("\n" + "=" * 70)
#         print(f"Question: {query}")
#         print("=" * 70)

#         results = searcher.search(
#             query
#         )

#         for result in results:

#             print(
#                 f"{result['rank']}. "
#                 f"{result['source']} "
#                 f"(Page {result['page']}) "
#                 f"RRF={result['score']:.6f}"
#             )

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
        # Build BM25 index
        # ---------------------------------------------------------

        tokenized_documents = [
            self.tokenize(text)
            for text in self.documents
        ]

        print("Building BM25 index...")

        self.bm25 = BM25Okapi(tokenized_documents)

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
        Combine Vector Search and BM25 results using
        Reciprocal Rank Fusion.

        Identical chunks are deduplicated BEFORE RRF scoring.

        If the same chunk exists under multiple source filenames,
        it is treated as one document instead of multiple documents.
        """

        content_groups = {}

        # ---------------------------------------------------------
        # Collect Vector results
        # ---------------------------------------------------------

        for result in vector_results:

            content_hash = self.content_hash(
                result["document"]
            )

            if content_hash not in content_groups:

                content_groups[content_hash] = {
                    "document": result["document"],
                    "metadata": result["metadata"],
                    "id": result["id"],
                    "vector_rank": result["rank"],
                    "bm25_rank": None
                }

            else:

                # Keep the best vector rank
                if result["rank"] < content_groups[content_hash]["vector_rank"]:
                    content_groups[content_hash]["vector_rank"] = result["rank"]
                    content_groups[content_hash]["metadata"] = result["metadata"]
                    content_groups[content_hash]["id"] = result["id"]

        # ---------------------------------------------------------
        # Collect BM25 results
        # ---------------------------------------------------------

        for result in bm25_results:

            content_hash = self.content_hash(
                result["document"]
            )

            if content_hash not in content_groups:

                content_groups[content_hash] = {
                    "document": result["document"],
                    "metadata": result["metadata"],
                    "id": result["id"],
                    "vector_rank": None,
                    "bm25_rank": result["rank"]
                }

            else:

                # Keep the best BM25 rank
                if (
                    content_groups[content_hash]["bm25_rank"] is None
                    or result["rank"]
                    < content_groups[content_hash]["bm25_rank"]
                ):

                    content_groups[content_hash]["bm25_rank"] = result["rank"]

        # ---------------------------------------------------------
        # Calculate RRF
        # ---------------------------------------------------------

        fused_results = []

        for content_hash, item in content_groups.items():

            score = 0.0

            # Vector contribution
            if item["vector_rank"] is not None:

                score += 1 / (
                    rrf_k + item["vector_rank"]
                )

            # BM25 contribution
            if item["bm25_rank"] is not None:

                score += 1 / (
                    rrf_k + item["bm25_rank"]
                )

            fused_results.append(
                {
                    "id": item["id"],
                    "source": item["metadata"]["source"],
                    "page": item["metadata"]["page"],
                    "text": item["document"],
                    "score": score,
                    "vector_rank": item["vector_rank"],
                    "bm25_rank": item["bm25_rank"],
                    "content_hash": content_hash
                }
            )

        # ---------------------------------------------------------
        # Sort by RRF score
        # ---------------------------------------------------------

        fused_results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        # ---------------------------------------------------------
        # Final ranking
        # ---------------------------------------------------------

        final_results = []

        for rank, result in enumerate(
            fused_results[:self.final_k],
            start=1
        ):

            result["rank"] = rank

            # Remove internal field
            result.pop("content_hash", None)

            final_results.append(result)

        return final_results

    # -------------------------------------------------------------
    # Search
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


# -----------------------------------------------------------------
# Standalone Test
# -----------------------------------------------------------------

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

        results = searcher.search(query)

        for result in results:

            print(
                f"{result['rank']}. "
                f"{result['source']} "
                f"(Page {result['page']}) "
                f"RRF={result['score']:.6f} "
                f"VectorRank={result['vector_rank']} "
                f"BM25Rank={result['bm25_rank']}"
            )
