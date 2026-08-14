import json

from hybrid_search import HybridSearch


# Treat these two filenames as the same source because
# their extracted contents were verified to be identical.
SOURCE_ALIASES = {
    "Microsoft.pdf": {
        "Microsoft.pdf",
        "Microsoft Windows, Windows Server, Azure Stack Administrative Guide (22H2).pdf",
    }
}

TOP_K = 10


def source_matches(expected, retrieved):
    """Return True when the retrieved source matches the expected source."""

    if expected == retrieved:
        return True

    aliases = SOURCE_ALIASES.get(expected)

    if aliases and retrieved in aliases:
        return True

    return False


def main():

    print("=" * 80)
    print("HYBRID RETRIEVAL EVALUATION")
    print("=" * 80)

    # Load evaluation dataset
    with open(
        "evaluation_dataset.json",
        "r",
        encoding="utf-8"
    ) as f:
        questions = json.load(f)

    hybrid = HybridSearch(final_k=TOP_K)
    results = []

    reciprocal_ranks = []

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    hits_at_10 = 0

    for item in questions:

        question = item["question"]
        expected_source = item["expected_source"]
        question_type = item.get("type", "Positive")

        print(f"\nQuestion: {question}")

        retrieved = hybrid.search(question)
        
        retrieved_results = []

        rank_found = None

        # Process retrieved results
        for rank, result in enumerate(retrieved, start=1):

            source = result["source"]
            page = result["page"]

            rrf_score = result.get(
                "rrf_score",
                result.get("score", 0)
            )

            retrieved_results.append(
                {
                    "rank": rank,
                    "source": source,
                    "page": page,
                    "score": rrf_score
                }
            )

            # Check source using aliases
            if (
                rank_found is None
                and source_matches(
                    expected_source,
                    source
                )
            ):
                rank_found = rank

        # ---------------------------------------------------------
        # Negative questions
        # ---------------------------------------------------------

        if question_type == "Negative":

            # For negative questions, the expected source
            # should NOT appear in the retrieved results.
            passed = all(
                not source_matches(
                    expected_source,
                    r["source"]
                )
                for r in retrieved_results
            )

            if passed:
                print("PASS - no expected source")
            else:
                print("FAIL - unexpected source retrieved")

        # ---------------------------------------------------------
        # Positive questions
        # ---------------------------------------------------------

        else:

            if rank_found is not None:

                print(
                    f"Expected source found at rank {rank_found}"
                )

                reciprocal_ranks.append(
                    1 / rank_found
                )

                if rank_found <= 1:
                    hits_at_1 += 1

                if rank_found <= 3:
                    hits_at_3 += 1

                if rank_found <= 5:
                    hits_at_5 += 1

                if rank_found <= 10:
                    hits_at_10 += 1

            else:

                print("Expected source NOT found")

                reciprocal_ranks.append(0)

        # Save individual result
        results.append(
            {
                "id": item.get("id"),
                "question": question,
                "expected_source": expected_source,
                "type": question_type,
                "rank": rank_found,
                "retrieved": retrieved_results,
            }
        )

    # -------------------------------------------------------------
    # Calculate metrics
    # -------------------------------------------------------------

    positive_questions = [
        q
        for q in questions
        if q.get("type", "Positive") != "Negative"
    ]

    total = len(positive_questions)

    hit_at_1 = hits_at_1 / total
    hit_at_3 = hits_at_3 / total
    hit_at_5 = hits_at_5 / total
    hit_at_10 = hits_at_10 / total

    mrr = sum(reciprocal_ranks) / total

    # -------------------------------------------------------------
    # Print results
    # -------------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(f"Hit@1:  {hit_at_1:.2%}")
    print(f"Hit@3:  {hit_at_3:.2%}")
    print(f"Hit@5:  {hit_at_5:.2%}")
    print(f"Hit@10: {hit_at_10:.2%}")
    print(f"MRR:    {mrr:.4f}")

    # -------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------

    output = {
        "method": "hybrid",
        "top_k": TOP_K,
        "metrics": {
            "hit_at_1": hit_at_1,
            "hit_at_3": hit_at_3,
            "hit_at_5": hit_at_5,
            "hit_at_10": hit_at_10,
            "mrr": mrr,
        },
        "results": results,
    }

    with open(
        "hybrid_results.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\nResults saved to: hybrid_results.json")


if __name__ == "__main__":
    main()