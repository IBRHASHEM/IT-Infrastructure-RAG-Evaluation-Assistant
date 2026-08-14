import json
from pathlib import Path

from embeddings import EmbeddingGenerator
from vector_store import VectorStore


DATASET_PATH = Path("evaluation_dataset.json")
RESULTS_PATH = Path("baseline_results.json")

K_VALUES = [1, 3, 5, 10]


def load_dataset():

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate():

    dataset = load_dataset()

    embedder = EmbeddingGenerator()
    vector_db = VectorStore()

    positive_results = []
    negative_results = []

    reciprocal_ranks = []

    print("=" * 80)
    print("BASELINE RETRIEVAL EVALUATION")
    print("=" * 80)

    for item in dataset:

        question = item["question"]
        expected_source = item["expected_source"]

        print(f"\nQuestion: {question}")

        embedding = embedder.embed_batch([question])[0]

        retrieved = vector_db.search(
            embedding,
            k=max(K_VALUES)
        )

        metadatas = retrieved["metadatas"][0]
        distances = retrieved["distances"][0]

        retrieved_results = []

        for rank, (metadata, distance) in enumerate(
            zip(metadatas, distances),
            start=1
        ):
            retrieved_results.append(
                {
                    "rank": rank,
                    "source": metadata["source"],
                    "page": metadata["page"],
                    "distance": distance
                }
            )

        # -----------------------------
        # Negative question
        # -----------------------------

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

        # -----------------------------
        # Positive question
        # -----------------------------

        matching_ranks = [
            result["rank"]
            for result in retrieved_results
            if result["source"] == expected_source
        ]

        if matching_ranks:

            first_rank = min(matching_ranks)

            reciprocal_rank = 1 / first_rank

            reciprocal_ranks.append(reciprocal_rank)

            print(
                f"Expected source found at rank {first_rank}"
            )

        else:

            first_rank = None

            reciprocal_ranks.append(0)

            print("Expected source NOT found")

        positive_results.append(
            {
                "id": item["id"],
                "question": question,
                "expected_source": expected_source,
                "rank": first_rank,
                "retrieved": retrieved_results
            }
        )

    # --------------------------------
    # Metrics
    # --------------------------------

    total_positive = len(positive_results)

    metrics = {}

    for k in K_VALUES:

        hits = sum(
            1
            for result in positive_results
            if result["rank"] is not None
            and result["rank"] <= k
        )

        score = (
            hits / total_positive
            if total_positive
            else 0
        )

        metrics[f"Hit@{k}"] = score

    metrics["MRR"] = (
        sum(reciprocal_ranks) / total_positive
        if total_positive
        else 0
    )

    # --------------------------------
    # Save results
    # --------------------------------

    output = {
        "evaluation": "Baseline Vector Retrieval",
        "retrieval_method": "BGE + ChromaDB",
        "total_questions": len(dataset),
        "positive_questions": len(positive_results),
        "negative_questions": len(negative_results),
        "metrics": metrics,
        "positive_results": positive_results,
        "negative_results": negative_results
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

    # --------------------------------
    # Print results
    # --------------------------------

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    for k in K_VALUES:

        print(
            f"Hit@{k}: "
            f"{metrics[f'Hit@{k}']:.2%}"
        )

    print(
        f"MRR: {metrics['MRR']:.4f}"
    )

    print("=" * 80)

    print(
        f"\nResults saved to: {RESULTS_PATH}"
    )


if __name__ == "__main__":
    evaluate()