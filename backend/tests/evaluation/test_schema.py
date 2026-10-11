import re
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from askact.evaluation import schema
from askact.evaluation.schema import EvalSetError, Question, load_questions
from askact.models import SectionId


def valid_entry(**overrides):
    entry = {
        "id": "q001",
        "question": "Which AI practices are banned?",
        "split": "dev",
        "category": "direct",
        "expected_ids": ["article:5"],
    }
    entry.update(overrides)
    return entry


def write_entries(tmp_path: Path, *entries) -> Path:
    path = tmp_path / "questions.yaml"
    path.write_text(yaml.safe_dump(list(entries), sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def write_text(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "questions.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def load_error(path: Path) -> str:
    with pytest.raises(EvalSetError) as info:
        load_questions(path)
    return str(info.value)


# --- a valid file ----------------------------------------------------------------------

def test_a_valid_file_loads_in_file_order(tmp_path):
    path = write_entries(
        tmp_path,
        valid_entry(id="q001"),
        valid_entry(id="q002", question="Rephrased", split="test", category="paraphrase", expected_ids=["article:5"]),
        valid_entry(id="q003", category="multi", expected_ids=["article:9", "annex:III", "recital:48"]),
        valid_entry(id="q004", question="What is the capital of France?", category="out_of_scope", expected_ids=[]),
    )
    questions = load_questions(path)
    assert [q.id for q in questions] == ["q001", "q002", "q003", "q004"]
    assert [q.split for q in questions] == ["dev", "test", "dev", "dev"]
    assert [q.category for q in questions] == ["direct", "paraphrase", "multi", "out_of_scope"]
    assert questions[0].question == "Which AI practices are banned?"
    assert questions[2].expected_ids == [
        SectionId("article", "9"), SectionId("annex", "III"), SectionId("recital", "48"),
    ]
    assert questions[3].expected_ids == []
    assert all(isinstance(q, Question) for q in questions)


def test_expected_ids_are_normalised_to_section_ids(tmp_path):
    path = write_entries(tmp_path, valid_entry(expected_ids=["Article 5", "annex:iii"]))
    assert [str(i) for i in load_questions(path)[0].expected_ids] == ["article:5", "annex:III"]


def test_non_ascii_question_text_is_kept(tmp_path):
    text = "Was gilt für „Hochrisiko“-KI — Artikel 6?"
    assert load_questions(write_entries(tmp_path, valid_entry(question=text)))[0].question == text


def test_a_quoted_numeric_id_is_fine(tmp_path):
    assert load_questions(write_entries(tmp_path, valid_entry(id="007")))[0].id == "007"


def test_the_default_location_is_eval_questions_yaml():
    assert schema.QUESTIONS_PATH == Path("eval/questions.yaml")


# --- the file itself -------------------------------------------------------------------

def test_missing_file_is_an_error_that_names_the_file(tmp_path):
    message = load_error(tmp_path / "nope.yaml")
    assert "nope.yaml" in message and "not found" in message


def test_a_directory_is_an_error(tmp_path):
    assert "cannot read" in load_error(tmp_path)


def test_a_file_that_is_not_utf8_is_an_error(tmp_path):
    path = tmp_path / "questions.yaml"
    path.write_bytes(b"- id: q1\n  question: \xff\xfe\n")
    assert "cannot read" in load_error(path)


def test_malformed_yaml_names_the_file_and_line(tmp_path):
    path = write_text(tmp_path, "- id: q1\n  question: [unclosed\n")
    message = load_error(path)
    assert "not valid YAML" in message
    assert f'in "{path}", line' in message  # the YAML error itself names the file, not "<unicode string>"
    assert "\n" not in message  # folded onto one line


@pytest.mark.parametrize("text", ["", "   \n", "# only a comment\n", "[]\n"])
def test_a_file_without_questions_is_an_error(tmp_path, text):
    assert "no questions" in load_error(write_text(tmp_path, text))


@pytest.mark.parametrize("text", ["a: 1\n", "just text\n", "42\n"])
def test_a_top_level_that_is_not_a_list_is_an_error(tmp_path, text):
    assert "must contain a list" in load_error(write_text(tmp_path, text))


def test_an_entry_that_is_not_a_mapping_is_named_by_position_and_line(tmp_path):
    text = yaml.safe_dump([valid_entry()], sort_keys=False) + "- just a string\n"
    message = load_error(write_text(tmp_path, text))
    assert "entry 2 (line 7)" in message and "must be a mapping" in message


def test_a_key_written_twice_is_an_error_not_last_one_wins(tmp_path):
    # PyYAML alone would keep `split: test` and say nothing.
    text = (
        "- id: q1\n  question: Q\n  split: dev\n  split: test\n"
        "  category: direct\n  expected_ids: [article:5]\n"
    )
    message = load_error(write_text(tmp_path, text))
    assert "'split' appears twice" in message
    assert "line 4" in message


def test_a_repeated_key_is_caught_whatever_its_type(tmp_path):
    assert "appears twice" in load_error(write_text(tmp_path, "- 1: a\n  1: b\n"))


def test_the_same_key_in_different_entries_is_fine(tmp_path):
    assert len(load_questions(write_entries(tmp_path, valid_entry(id="a"), valid_entry(id="b")))) == 2


# --- one entry breaks the schema: the message names the entry and the field ------------

def test_each_required_field_is_required(tmp_path):
    for field in ("id", "question", "split", "category", "expected_ids"):
        entry = valid_entry()
        del entry[field]
        message = load_error(write_entries(tmp_path, entry))
        assert f"{field}: Field required" in message, field
        assert "(got" not in message, field  # there is no rejected value to show


@pytest.mark.parametrize("overrides, field, fragment", [
    ({"split": "train"}, "split", "'dev' or 'test'"),
    ({"split": "DEV"}, "split", "'dev' or 'test'"),
    ({"split": None}, "split", "'dev' or 'test'"),
    ({"category": "easy"}, "category", "'direct', 'paraphrase', 'multi' or 'out_of_scope'"),
    ({"category": "out-of-scope"}, "category", "'out_of_scope'"),
    ({"question": ""}, "question", "must not be blank"),
    ({"question": "  \n"}, "question", "must not be blank"),
    ({"question": 42}, "question", "valid string"),
    ({"id": ""}, "id", "non-empty"),
    ({"id": "q 1"}, "id", "without spaces"),
    ({"id": 7}, "id", "put it in quotes"),
    ({"id": False}, "id", "put it in quotes"),
    ({"expected_ids": "article:5"}, "expected_ids", "valid list"),
    ({"expected_ids": None}, "expected_ids", "valid list"),
    ({"expected_ids": [5]}, "expected_ids", "section id must be a string"),
    ({"expected_ids": ["chapter:3"]}, "expected_ids", "invalid section id"),
    ({"expected_ids": ["article:0"]}, "expected_ids", "positive integer"),
    ({"expected_ids": ["article:5(1)"]}, "expected_ids", "positive integer"),  # sub-references are not sections
    ({"expected_ids": ["annex:IIII"]}, "expected_ids", "Roman numeral"),
    ({"expected_ids": ["article:5", ""]}, "expected_ids", "invalid section id"),
    ({"unexpected": 1}, "unexpected", "Extra inputs are not permitted"),
    ({"expected_id": ["article:5"]}, "expected_id", "Extra inputs are not permitted"),
])
def test_a_bad_field_is_reported_with_entry_and_field(tmp_path, overrides, field, fragment):
    message = load_error(write_entries(tmp_path, valid_entry(**overrides)))
    assert ": 1 problem:" in message  # exactly the one fault that was injected
    assert re.search(r"entry 1 \(line 1[,)]", message)  # `line 1` followed by `,` or `)`, so not `line 10`
    assert re.search(rf"\b{field}(\[\d+\])?: ", message)
    assert fragment in message


def test_an_empty_id_is_not_used_to_name_the_entry(tmp_path):
    message = load_error(write_entries(tmp_path, valid_entry(id="")))
    assert "entry 1 (line 1): id: " in message  # not "(line 1, id '')"


def test_the_message_names_the_entry_by_id(tmp_path):
    message = load_error(write_entries(tmp_path, valid_entry(id="q-bad", split="train")))
    assert "entry 1 (line 1, id 'q-bad')" in message


def test_messages_do_not_carry_pydantics_value_error_prefix(tmp_path):
    message = load_error(write_entries(tmp_path, valid_entry(question="", expected_ids=["chapter:3"])))
    assert "Value error" not in message
    assert "must not be blank" in message and "invalid section id" in message


def test_a_loaded_question_cannot_be_changed(tmp_path):
    question = load_questions(write_entries(tmp_path, valid_entry()))[0]
    with pytest.raises(ValidationError):
        question.split = "test"


def test_the_message_shows_the_value_that_was_rejected(tmp_path):
    assert "(got 'train')" in load_error(write_entries(tmp_path, valid_entry(split="train")))


def test_a_very_long_rejected_value_is_shortened(tmp_path):
    message = load_error(write_entries(tmp_path, valid_entry(split="x" * 500)))
    assert "x" * 500 not in message
    assert "...)" in message


def test_an_unknown_key_is_reported_without_a_rejected_value(tmp_path):
    message = load_error(write_entries(tmp_path, valid_entry(unexpected=1)))
    assert "unexpected: Extra inputs are not permitted" in message
    assert "(got" not in message


def test_a_key_that_cannot_be_a_key_is_an_error_not_a_crash(tmp_path):
    text = "- ? [a, b]\n  : 1\n"  # a list used as a mapping key
    assert "not valid YAML" in load_error(write_text(tmp_path, text))


def test_a_chunk_id_is_rejected_with_a_hint_to_use_the_section_id(tmp_path):
    message = load_error(write_entries(tmp_path, valid_entry(expected_ids=["article:5#2"])))
    assert "looks like a chunk id" in message
    assert "(for example 'article:5')" in message  # the suggested section id


def test_a_chunk_id_is_rejected_even_next_to_valid_ids(tmp_path):
    message = load_error(write_entries(tmp_path, valid_entry(expected_ids=["article:9", "annex:III#1"])))
    assert "'annex:III#1' looks like a chunk id" in message


# --- expected_ids must be empty exactly for out_of_scope -------------------------------

def test_out_of_scope_with_expected_ids_is_an_error(tmp_path):
    entry = valid_entry(id="q9", category="out_of_scope", expected_ids=["article:5"])
    message = load_error(write_entries(tmp_path, entry))
    assert "entry 1 (line 1, id 'q9')" in message
    assert "out_of_scope question must have expected_ids: []" in message
    assert "article:5" in message


@pytest.mark.parametrize("category", ["direct", "paraphrase", "multi"])
def test_an_in_scope_question_without_expected_ids_is_an_error(tmp_path, category):
    message = load_error(write_entries(tmp_path, valid_entry(category=category, expected_ids=[])))
    assert f"a {category} question needs at least one expected id" in message


def test_out_of_scope_with_empty_expected_ids_is_valid(tmp_path):
    path = write_entries(tmp_path, valid_entry(category="out_of_scope", expected_ids=[]))
    assert load_questions(path)[0].expected_ids == []


def test_a_section_listed_twice_is_an_error(tmp_path):
    # "Article 5" and "article:5" are the same section once normalised.
    message = load_error(write_entries(tmp_path, valid_entry(expected_ids=["article:5", "Article 5"])))
    assert "lists a section twice" in message


# --- unique ids ------------------------------------------------------------------------

def test_duplicate_ids_are_an_error_naming_both_lines(tmp_path):
    text = yaml.safe_dump([valid_entry(id="q1"), valid_entry(id="q2"), valid_entry(id="q1")], sort_keys=False)
    message = load_error(write_text(tmp_path, text))
    # Compared as a whole line: "...on line 1" would also match a wrong "...on line 101".
    assert "  - id 'q1' (line 13): already used by the entry on line 1" in message.splitlines()


def test_ids_that_differ_only_by_case_are_different(tmp_path):
    assert len(load_questions(write_entries(tmp_path, valid_entry(id="Q1"), valid_entry(id="q1")))) == 2


def test_a_duplicate_id_is_reported_even_if_the_entry_has_another_problem(tmp_path):
    message = load_error(write_entries(tmp_path, valid_entry(id="q1"), valid_entry(id="q1", split="train")))
    assert "already used" in message
    assert "split:" in message


# --- every problem is listed, nothing is partially loaded ------------------------------

def test_all_problems_are_reported_together(tmp_path):
    message = load_error(write_entries(
        tmp_path,
        valid_entry(id="a", split="train"),
        valid_entry(id="b"),
        valid_entry(id="c", category="easy"),
        valid_entry(id="d", expected_ids=["article:5#1"]),
    ))
    assert "3 problems" in message
    for name in ("id 'a'", "id 'c'", "id 'd'"):
        assert name in message
    assert "id 'b'" not in message  # the valid entry is not blamed


def test_one_bad_entry_means_no_questions_are_returned(tmp_path):
    path = write_entries(tmp_path, valid_entry(id="a"), valid_entry(id="b", split="train"))
    with pytest.raises(EvalSetError):
        load_questions(path)


def test_an_entry_without_a_usable_id_is_named_by_position_and_line(tmp_path):
    entry = valid_entry()
    del entry["id"]
    text = yaml.safe_dump([valid_entry(id="ok"), entry], sort_keys=False)
    assert "entry 2 (line 7)" in load_error(write_text(tmp_path, text))


def test_the_error_message_is_one_line_per_problem(tmp_path):
    message = load_error(write_entries(tmp_path, valid_entry(split="train", question="")))
    lines = message.splitlines()
    assert lines[0].endswith("2 problems:")
    assert len(lines) == 3 and all(line.startswith("  - ") for line in lines[1:])
