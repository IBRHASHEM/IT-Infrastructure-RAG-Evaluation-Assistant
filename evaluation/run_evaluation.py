import json

from evaluation.evaluator import Evaluator
from rag_hybrid import HybridRAG


def main():

    with open("evaluation/dataset.json", encoding="utf-8") as file:
        dataset = json.load(file)

    evaluator = Evaluator()
    assistant = HybridRAG()

    scores = []

    for item in dataset:

        print("\n" + "=" * 80)
        print("Question:", item["question"])

        response = assistant.ask(item["question"])

        answer = response["answer"]

        print("\nAnswer:")
        print(answer)

        result = evaluator.evaluate(
            answer,
            item["expected_keywords"]
        )

        print("\nEvaluation:")
        print(result)

        scores.append(
            result["final_score"]
        )

    print("\n" + "=" * 80)
    print(
        "Average Score:",
        round(sum(scores) / len(scores), 2)
    )


if __name__ == "__main__":
    main()