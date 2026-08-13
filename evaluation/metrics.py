def keyword_coverage(answer, keywords):
    """
    Measure how many expected keywords
    exist in the generated answer
    """

    answer = answer.lower()

    matched = 0

    for keyword in keywords:
        if keyword.lower() in answer:
            matched += 1

    if not keywords:
        return 0

    return matched / len(keywords)



def response_length_score(answer):
    """
    Simple quality check based on answer size
    """

    words = len(answer.split())


    if words < 20:
        return 0.5


    if words > 400:
        return 0.7


    return 1.0



def final_score(keyword_score, length_score):

    return round(
        (keyword_score + length_score) / 2,
        2
    )