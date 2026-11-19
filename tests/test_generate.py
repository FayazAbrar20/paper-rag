from generate import REFUSAL, build_prompt, cited_sources, is_refusal


def make_hits():
    return [
        {"paper": "Sarcasm", "page": 3, "score": 0.81234, "text": "F1 0.72"},
        {"paper": "Sarcasm", "page": 3, "score": 0.70111, "text": "same page again"},
        {"paper": "Bitcoin", "page": 7, "score": 0.65555, "text": "accuracy 65.45%"},
        {"paper": "Emotion", "page": 5, "score": 0.60001, "text": "BERT 94.2%"},
    ]


def test_exact_refusal_sentence_is_a_refusal():
    assert is_refusal(REFUSAL)


def test_refusal_check_ignores_case():
    assert is_refusal("I DON'T KNOW based on the papers.")


def test_normal_answer_is_not_a_refusal():
    assert not is_refusal("The best F1 was 0.72 [S1].")


def test_refusal_has_no_sources():
    assert cited_sources(REFUSAL, make_hits()) == []


def test_only_cited_sources_are_returned():
    sources = cited_sources("Accuracy was 65.45% [S3].", make_hits())
    assert sources == [{"paper": "Bitcoin", "page": 7, "score": 0.656}]


def test_same_page_is_listed_once():
    sources = cited_sources("See [S1][S2].", make_hits())
    assert sources == [{"paper": "Sarcasm", "page": 3, "score": 0.812}]


def test_labels_outside_the_retrieved_sources_are_ignored():
    sources = cited_sources("See [S4] and [S9].", make_hits())
    assert sources == [{"paper": "Emotion", "page": 5, "score": 0.6}]


def test_paper_reference_markers_are_not_treated_as_sources():
    sources = cited_sources("As shown in [5], F1 was 0.72 [S1].", make_hits())
    assert [s["paper"] for s in sources] == ["Sarcasm"]


def test_answer_without_citations_falls_back_to_all_distinct_pages():
    sources = cited_sources("F1 was 0.72.", make_hits())
    assert [(s["paper"], s["page"]) for s in sources] == [("Sarcasm", 3), ("Bitcoin", 7), ("Emotion", 5)]


def test_prompt_labels_sources_s1_to_s4_with_paper_and_page():
    prompt = build_prompt("What was the best F1?", make_hits())
    for n, hit in enumerate(make_hits(), start=1):
        assert f"Source S{n} — {hit['paper']}, page {hit['page']}" in prompt
    assert prompt.endswith("Question: What was the best F1?\nAnswer:")
