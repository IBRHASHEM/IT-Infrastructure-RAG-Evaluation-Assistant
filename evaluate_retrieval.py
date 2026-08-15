import json
from pathlib import Path

from hybrid_search import HybridSearch


# =========================================================
# Configuration
# =========================================================

DATASET_PATH = Path("evaluation_dataset.json")
RESULTS_PATH = Path("retrieval_evaluation.json")

K_VALUES = [1, 3, 5, 10]

# Same retrieval configuration used by the application,
# except final_k=10 so we can evaluate up to Hit@10.
VECTOR_K = 50
BM25_K = 50
HYBRID_FINAL_K = 10


# =========================================================
# Dataset
# =========================================================

def load_dataset():

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# =========================================================
# Utility
# =========================================================

def calculate_rank(results, expected_source):

    matching_ranks = [
        result["rank"]
        for result in results
        if result["source"] == expected_source
    ]

    if matching_ranks:
        return min(matching_ranks)

    return None


def calculate_metrics(results, k_values):

    total_positive = len(results)

    metrics = {}

    if total_positive == 0:
        for k in k_values:
            metrics[f"Hit@{k}"] = 0.0

        metrics["MRR"] = 0.0

        return metrics

    # -----------------------------------------------------
    # Hit Rate
    # -----------------------------------------------------

    for k in k_values:

        hits = sum(
            1
            for result in results
            if result["rank"] is not None
            and result["rank"] <= k
        )

        metrics[f"Hit@{k}"] = (
            hits / total_positive
        )

    # -----------------------------------------------------
    # MRR
    # -----------------------------------------------------

    reciprocal_ranks = []

    for result in results:

        rank = result["rank"]

        if rank is None:
            reciprocal_ranks.append(0.0)
        else:
            reciprocal_ranks.append(
                1.0 / rank
            )

    metrics["MRR"] = (
        sum(reciprocal_ranks)
        / total_positive
    )

    return metrics


# =========================================================
# Evaluate one retrieval method
# =========================================================

def evaluate_method(
    dataset,
    searcher,
    method
):

    positive_results = []
    negative_results = []

    print("\n" + "=" * 80)
    print(f"{method.upper()} RETRIEVAL")
    print("=" * 80)

    for item in dataset:

        question = item["question"]
        expected_source = item["expected_source"]

        print(f"\nQuestion: {question}")

        # -------------------------------------------------
        # Retrieval
        # -------------------------------------------------

        if method == "vector":

            raw_results = searcher.vector_search(
                question
            )

            retrieved_results = []

            for result in raw_results:

                retrieved_results.append(
                    {
                        "rank": result["rank"],
                        "source": result["metadata"]["source"],
                        "page": result["metadata"]["page"],
                        "score": None,
                        "distance": result["distance"]
                    }
                )

        elif method == "hybrid":

            raw_results = searcher.search(
                question
            )

            retrieved_results = []

            for result in raw_results:

                retrieved_results.append(
                    {
                        "rank": result["rank"],
                        "source": result["source"],
                        "page": result["page"],
                        "score": result["score"],
                        "vector_rank": result["vector_rank"],
                        "bm25_rank": result["bm25_rank"]
                    }
                )

        else:

            raise ValueError(
                f"Unknown retrieval method: {method}"
            )

        # -------------------------------------------------
        # Negative question
        # -------------------------------------------------

        if expected_source is None:

            negative_results.append(
                {
                    "id": item["id"],
                    "question": question,
                    "expected_source": None,
                    "retrieved": retrieved_results
                }
            )

            print("Type: Negative")

            continue

        # -------------------------------------------------
        # Positive question
        # -------------------------------------------------

        first_rank = calculate_rank(
            retrieved_results,
            expected_source
        )

        if first_rank is not None:

            print(
                f"Expected source found at rank {first_rank}"
            )

        else:

            print(
                "Expected source NOT found"
            )

        positive_results.append(
            {
                "id": item["id"],
                "question": question,
                "expected_source": expected_source,
                "rank": first_rank,
                "retrieved": retrieved_results
            }
        )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    metrics = calculate_metrics(
        positive_results,
        K_VALUES
    )

    return {
        "method": method,
        "positive_questions": len(
            positive_results
        ),
        "negative_questions": len(
            negative_results
        ),
        "metrics": metrics,
        "positive_results": positive_results,
        "negative_results": negative_results
    }


# =========================================================
# Compare methods
# =========================================================

def determine_winner(
    vector_metrics,
    hybrid_metrics
):

    # Primary metric: MRR
    if hybrid_metrics["MRR"] > vector_metrics["MRR"]:
        return "Hybrid"

    if vector_metrics["MRR"] > hybrid_metrics["MRR"]:
        return "Vector"

    # Secondary metric: Hit@5
    if (
        hybrid_metrics["Hit@5"]
        > vector_metrics["Hit@5"]
    ):
        return "Hybrid"

    if (
        vector_metrics["Hit@5"]
        > hybrid_metrics["Hit@5"]
    ):
        return "Vector"

    return "Tie"


# =========================================================
# Main Evaluation
# =========================================================

def evaluate():

    dataset = load_dataset()

    print("=" * 80)
    print("RETRIEVAL EVALUATION")
    print("Vector Search vs Hybrid Search")
    print("=" * 80)

    print(
        f"\nDataset questions: {len(dataset)}"
    )

    print(
        "\nInitializing HybridSearch..."
    )

    # One searcher is enough.
    # It contains both Vector Search and BM25,
    # and uses the same RRF implementation
    # as the production application.
    searcher = HybridSearch(
        vector_k=VECTOR_K,
        bm25_k=BM25_K,
        final_k=HYBRID_FINAL_K
    )

    # -----------------------------------------------------
    # Vector evaluation
    # -----------------------------------------------------

    vector_results = evaluate_method(
        dataset,
        searcher,
        "vector"
    )

    # -----------------------------------------------------
    # Hybrid evaluation
    # -----------------------------------------------------

    hybrid_results = evaluate_method(
        dataset,
        searcher,
        "hybrid"
    )

    # -----------------------------------------------------
    # Winner
    # -----------------------------------------------------

    winner = determine_winner(
        vector_results["metrics"],
        hybrid_results["metrics"]
    )

    # -----------------------------------------------------
    # Comparison
    # -----------------------------------------------------

    comparison = {}

    for k in K_VALUES:

        vector_score = vector_results[
            "metrics"
        ][f"Hit@{k}"]

        hybrid_score = hybrid_results[
            "metrics"
        ][f"Hit@{k}"]

        comparison[f"Hit@{k}"] = {
            "vector": vector_score,
            "hybrid": hybrid_score,
            "difference": (
                hybrid_score
                - vector_score
            )
        }

    comparison["MRR"] = {
        "vector": vector_results[
            "metrics"
        ]["MRR"],
        "hybrid": hybrid_results[
            "metrics"
        ]["MRR"],
        "difference": (
            hybrid_results["metrics"]["MRR"]
            - vector_results["metrics"]["MRR"]
        )
    }

    # -----------------------------------------------------
    # Final output
    # -----------------------------------------------------

    output = {
        "evaluation": "Vector vs Hybrid Retrieval",
        "dataset": str(DATASET_PATH),
        "total_questions": len(dataset),
        "k_values": K_VALUES,

        "configuration": {
            "vector_k": VECTOR_K,
            "bm25_k": BM25_K,
            "hybrid_final_k": HYBRID_FINAL_K,
            "rrf": True
        },

        "vector_retrieval": vector_results,

        "hybrid_retrieval": hybrid_results,

        "comparison": comparison,

        "winner": winner
    }

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    # =====================================================
    # Print final results
    # =====================================================

    print("\n")
    print("=" * 80)
    print("FINAL RETRIEVAL COMPARISON")
    print("=" * 80)

    print(
        f"\n{'Metric':<12}"
        f"{'Vector':<15}"
        f"{'Hybrid':<15}"
        f"{'Difference':<15}"
    )

    print("-" * 57)

    for k in K_VALUES:

        vector_score = vector_results[
            "metrics"
        ][f"Hit@{k}"]

        hybrid_score = hybrid_results[
            "metrics"
        ][f"Hit@{k}"]

        difference = (
            hybrid_score
            - vector_score
        )

        print(
            f"{'Hit@' + str(k):<12}"
            f"{vector_score:<15.2%}"
            f"{hybrid_score:<15.2%}"
            f"{difference:+.2%}"
        )

    print(
        f"{'MRR':<12}"
        f"{vector_results['metrics']['MRR']:<15.4f}"
        f"{hybrid_results['metrics']['MRR']:<15.4f}"
        f"{comparison['MRR']['difference']:+.4f}"
    )

    print("-" * 57)

    print(
        f"\nWINNER: {winner}"
    )

    print(
        f"\nResults saved to: {RESULTS_PATH}"
    )

    print("=" * 80)


# =========================================================
# Entry Point
# =========================================================

if __name__ == "__main__":
    evaluate()