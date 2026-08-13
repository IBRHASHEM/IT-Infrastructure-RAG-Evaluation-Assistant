from evaluation.metrics import (
    keyword_coverage,
    response_length_score,
    final_score
)


class Evaluator:


    def evaluate(
        self,
        answer,
        keywords
    ):

        keyword = keyword_coverage(
            answer,
            keywords
        )


        length = response_length_score(
            answer
        )


        return {

            "keyword_score": keyword,

            "length_score": length,

            "final_score":
                final_score(
                    keyword,
                    length
                )
        }