import json

from evaluation.evaluator import Evaluator

from assistant import GeminiAssistant


def main():


    with open(
        "evaluation/dataset.json"
    ) as file:

        dataset = json.load(file)



    evaluator = Evaluator()
    assistant = GeminiAssistant()

    scores = []


    for item in dataset:


        print("\n================")

        print(
            "Question:",
            item["question"]
        )

        response = assistant.ask(
        item["question"]
            )

        answer = response["answer"]


        print(
                "\nAnswer:",
                answer[:300]
            )

        print(
            "\nResponse Type:",
            type(answer)
        )

        print(
            "\nFull Response:",
            answer
        )
        print(
        "\nAnswer:",
            answer[:300]
        )


        result = evaluator.evaluate(
            answer,
            item["expected_keywords"]
        )


        print(
            "\nEvaluation:",
            result
        )


        scores.append(
            result["final_score"]
        )



    print(
        "\n================"
    )

    print(
        "Average Score:",
        round(
            sum(scores)/len(scores),
            2
        )
    )



if __name__ == "__main__":
    main()