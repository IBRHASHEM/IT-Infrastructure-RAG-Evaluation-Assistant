import json

from hybrid_search import HybridSearch
from reranker import DocumentReranker


DATASET_FILE = "evaluation_dataset.json"
OUTPUT_FILE = "reranker_results.json"


def evaluate():

    print("=" * 80)
    print("HYBRID + RERANKER EVALUATION")
    print("=" * 80)

    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    searcher = HybridSearch(
        vector_k=50,
        bm25_k=50,
        final_k=30
    )

    reranker = DocumentReranker()

    results = []

    for item in dataset:

        question = item["question"]
        expected_source = item.get("expected_source")
        question_type = item.get("type", "Positive")

        print()
        print(f"Question: {question}")

        # Hybrid retrieval
        hybrid_results = searcher.search(question)

        # Reranking
        reranked_results = reranker.rerank(
            question,
            hybrid_results,
            top_k=10
        )

        rank = None

        if question_type != "Negative":

            for result in reranked_results:

                if result["source"] == expected_source:
                    rank = result["rank"]
                    break

            if rank is not None:
                print(
                    f"Expected source found at rank {rank}"
                )
            else:
                print(
                    "Expected source NOT found"
                )

        else:

            print("Type: Negative")

        results.append(
            {
                "id": item.get("id"),
                "question": question,
                "expected_source": expected_source,
                "type": question_type,
                "rank": rank,
                "retrieved": [
                    {
                        "rank": result["rank"],
                        "source": result["source"],
                        "page": result["page"],
                        "reranker_score": result["reranker_score"]
                    }
                    for result in reranked_results
                ]
            }
        )

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    positive_results = [
        r for r in results
        if r["type"] != "Negative"
    ]

    total = len(positive_results)

    hit_1 = sum(
        1 for r in positive_results
        if r["rank"] is not None and r["rank"] <= 1
    )

    hit_3 = sum(
        1 for r in positive_results
        if r["rank"] is not None and r["rank"] <= 3
    )

    hit_5 = sum(
        1 for r in positive_results
        if r["rank"] is not None and r["rank"] <= 5
    )

    hit_10 = sum(
        1 for r in positive_results
        if r["rank"] is not None and r["rank"] <= 10
    )

    reciprocal_ranks = []

    for r in positive_results:

        if r["rank"] is not None:
            reciprocal_ranks.append(
                1 / r["rank"]
            )
        else:
            reciprocal_ranks.append(0)

    mrr = sum(reciprocal_ranks) / total

    metrics = {
        "hit_at_1": hit_1 / total,
        "hit_at_3": hit_3 / total,
        "hit_at_5": hit_5 / total,
        "hit_at_10": hit_10 / total,
        "mrr": mrr
    }

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(f"Hit@1:  {metrics['hit_at_1'] * 100:.2f}%")
    print(f"Hit@3:  {metrics['hit_at_3'] * 100:.2f}%")
    print(f"Hit@5:  {metrics['hit_at_5'] * 100:.2f}%")
    print(f"Hit@10: {metrics['hit_at_10'] * 100:.2f}%")
    print(f"MRR:    {metrics['mrr']:.4f}")

    output = {
        "method": "hybrid_reranker",
        "metrics": metrics,
        "results": results
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print(
        f"Results saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    evaluate()