from sentence_transformers import CrossEncoder


class DocumentReranker:

    def __init__(self):
        print("Loading BGE reranker...")

        self.model = CrossEncoder(
            "BAAI/bge-reranker-base",
            cache_folder=r"D:\Models"
        )

        print("Reranker loaded.")

    def rerank(self, query, results, top_k=10):

        if not results:
            return []

        pairs = [
            [query, result["text"]]
            for result in results
        ]

        scores = self.model.predict(pairs)

        reranked = []

        for result, score in zip(results, scores):
            item = result.copy()
            item["reranker_score"] = float(score)
            reranked.append(item)

        reranked.sort(
            key=lambda x: x["reranker_score"],
            reverse=True
        )

        for rank, result in enumerate(
            reranked[:top_k],
            start=1
        ):
            result["rank"] = rank

        return reranked[:top_k]