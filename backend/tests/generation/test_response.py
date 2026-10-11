import json

import pytest

from askact.generation.llm import LLMBadOutputError
from askact.generation.response import ModelReply, parse_reply

GOOD = {"answer": "Real-time remote biometric identification is prohibited with narrow exceptions.",
        "citations": ["article:5"], "covered": True}


def reply(**overrides) -> str:
    return json.dumps({**GOOD, **overrides})


def bad(text: str) -> str:
    with pytest.raises(LLMBadOutputError) as info:
        parse_reply(text)
    return str(info.value)


# --- valid replies ---------------------------------------------------------------------

def test_a_valid_reply_is_read_into_its_three_fields():
    parsed = parse_reply(reply())
    assert parsed == ModelReply(answer=GOOD["answer"], citations=["article:5"], covered=True)


def test_surrounding_whitespace_is_ignored():
    assert parse_reply("\n  " + reply() + "  \n\n").covered is True


@pytest.mark.parametrize("opening, closing", [
    ("```json\n", "\n```"),
    ("```\n", "\n```"),
    ("```JSON\n", "\n```"),
    ("```json", "```"),             # no line breaks at all
    ("  ```json\r\n", "\r\n```  "),  # Windows line endings and outer whitespace
])
def test_a_code_fence_around_the_json_is_tolerated(opening, closing):
    assert parse_reply(opening + reply() + closing) == parse_reply(reply())


def test_pretty_printed_json_over_many_lines_is_read_with_or_without_a_fence():
    pretty = json.dumps(GOOD, indent=2)
    assert pretty.count("\n") >= 4
    assert parse_reply(pretty) == parse_reply(reply())
    assert parse_reply(f"```json\n{pretty}\n```") == parse_reply(reply())
    assert parse_reply(f"```\n\n{pretty}\n\n```") == parse_reply(reply())


def test_a_space_between_the_fence_and_the_language_tag_is_tolerated():
    assert parse_reply("``` json\n" + reply() + "\n```") == parse_reply(reply())


def test_backticks_inside_the_answer_do_not_end_the_fence_early():
    answer = "Use ```code``` carefully."
    assert parse_reply(f"```json\n{reply(answer=answer)}\n```").answer == answer


def test_several_citations_are_kept_in_order():
    assert parse_reply(reply(citations=["article:5", "annex:III", "recital:48"])).citations == [
        "article:5", "annex:III", "recital:48"]


def test_citations_are_not_checked_here():
    # Dropping invalid or unretrieved citations is the answer pipeline's job; here they are just strings.
    assert parse_reply(reply(citations=["Art. 5(1)", "article:9999", ""])).citations == ["Art. 5(1)", "article:9999", ""]


def test_not_covered_may_come_with_an_empty_answer_and_no_citations():
    parsed = parse_reply(reply(answer="", citations=[], covered=False))
    assert (parsed.answer, parsed.citations, parsed.covered) == ("", [], False)


def test_extra_fields_are_ignored_rather_than_failing_a_good_reply():
    assert parse_reply(reply(reasoning="because", confidence=0.9)).answer == GOOD["answer"]


def test_non_ascii_text_survives():
    text = "Hochrisiko-KI-Systeme müssen „angemessen“ überwacht werden — siehe Artikel 14."
    assert parse_reply(reply(answer=text)).answer == text


# --- replies that are not the required structure ---------------------------------------

@pytest.mark.parametrize("text", [
    "",
    "   \n",
    "I think the answer is Article 5.",
    "Here is the JSON you asked for: " + json.dumps(GOOD),
    json.dumps(GOOD) + "\nHope this helps!",
    "```\nnot json at all\n```",
    "```json\n" + json.dumps(GOOD) + "\n```\n```json\n" + json.dumps(GOOD) + "\n```",  # two blocks
    json.dumps(GOOD)[:-10],  # cut off by the output-token limit
    "```json\n" + json.dumps(GOOD),  # a fence that is never closed: malformed, so not tolerated
    json.dumps(GOOD) + "\n```",  # a closing fence with no opening one
    "{'answer': 'single quotes', 'citations': [], 'covered': false}",
    '{"answer": "x", "citations": [], "covered": false,}',  # trailing comma
])
def test_text_that_is_not_one_json_object_is_rejected(text):
    assert "not a single JSON object" in bad(text)


@pytest.mark.parametrize("text", ["[]", '["article:5"]', '"just a string"', "42", "null", "true"])
def test_json_that_is_not_an_object_is_rejected(text):
    assert "not an object" in bad(text)


@pytest.mark.parametrize("field", ["answer", "citations", "covered"])
def test_each_missing_field_is_rejected_and_named(field):
    data = dict(GOOD)
    del data[field]
    message = bad(json.dumps(data))
    assert f"{field}: Field required" in message


@pytest.mark.parametrize("overrides, field", [
    ({"answer": 5}, "answer"),
    ({"answer": None}, "answer"),
    ({"answer": ["a"]}, "answer"),
    ({"citations": "article:5"}, "citations"),
    ({"citations": None}, "citations"),
    ({"citations": [5]}, "citations.0"),
    ({"citations": ["article:5", None]}, "citations.1"),
    ({"citations": [["article:5"]]}, "citations.0"),
    ({"covered": "true"}, "covered"),
    ({"covered": 1}, "covered"),
    ({"covered": None}, "covered"),
    ({"covered": "yes"}, "covered"),
])
def test_a_field_of_the_wrong_type_is_rejected_and_named(overrides, field):
    assert f"{field}: " in bad(reply(**overrides))


def test_covered_with_no_answer_is_rejected():
    assert "covered but gives no answer" in bad(reply(answer="", covered=True))
    assert "covered but gives no answer" in bad(reply(answer="  \n ", covered=True))


# --- what the error says ---------------------------------------------------------------

MARKER = "SECRET-MODEL-OUTPUT-7f3a"


@pytest.mark.parametrize("text", [
    f"Here you go: {MARKER}",
    json.dumps({"answer": MARKER, "citations": MARKER, "covered": MARKER}),
    json.dumps({"answer": MARKER, "citations": [MARKER, 5], "covered": True}),
    json.dumps([MARKER]),
])
def test_an_error_never_repeats_what_the_model_said(text):
    # Model output is untrusted and must not reach a user or a log through an error message.
    assert MARKER not in bad(text)


def test_the_error_carries_the_api_error_code():
    with pytest.raises(LLMBadOutputError) as info:
        parse_reply("nope")
    assert info.value.code == "llm_bad_output"


def test_all_the_problems_with_the_fields_are_listed_together():
    message = bad(json.dumps({"answer": 5, "citations": "x"}))
    assert "answer: " in message and "citations: " in message and "covered: Field required" in message
