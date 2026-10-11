"""The evaluation question set, `eval/questions.yaml`, and its strict loader (R5.1-R5.4).

Every score the harness prints comes from this file, so it is checked completely before anything is
scored: a typo that silently changed a question's split, or an expected id that can never match,
would give a wrong number that looks like a right one. The loader therefore either returns every
question or raises `EvalSetError` listing the problems it found in every entry (not just the first),
and never returns a partial list. One limit: the rule tying `category` to `expected_ids` is only
checked on an entry whose own fields are all valid, so it can appear on the run after the others are fixed.

An entry looks like this (all five keys are required, and no others are allowed):

    - id: q001
      question: "Which AI practices are banned?"
      split: dev                   # dev | test
      category: direct             # direct | paraphrase | multi | out_of_scope
      expected_ids: [article:5]    # section ids, [] exactly for out_of_scope
"""

from collections.abc import Hashable
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from askact.models import SectionId

QUESTIONS_PATH = Path("eval/questions.yaml")

Split = Literal["dev", "test"]
Category = Literal["direct", "paraphrase", "multi", "out_of_scope"]


class EvalSetError(Exception):
    """The question file is missing, malformed or breaks the schema. The message names every problem."""


class Question(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")  # a misspelt key is an error, not ignored

    id: str  # unique in the file; a YAML number such as `id: 7` is rejected, so quote it
    question: str
    split: Split
    category: Category
    # Section ids (`article:5`), not chunk ids (`article:5#2`). Every listed section counts as relevant,
    # not as an alternative: a hit needs any one of them, and citation recall is the share that is cited.
    expected_ids: list[SectionId]

    @field_validator("id", mode="before")
    @classmethod
    def _id_is_one_word(cls, value: object) -> object:
        if not isinstance(value, str):
            # YAML turns `id: 007` into the number 7 and `id: no` into False.
            raise ValueError('must be a string: put it in quotes, for example id: "007"')
        if not value or any(c.isspace() for c in value):
            raise ValueError("must be a non-empty string without spaces")
        return value

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("expected_ids", mode="before")
    @classmethod
    def _not_chunk_ids(cls, value: object) -> object:
        # `article:5#2` is what a retrieved chunk is called. Without this check it would still be
        # rejected, but with the less helpful "article number must be a positive integer".
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and "#" in item:
                    raise ValueError(
                        f"{item!r} looks like a chunk id; use the section id instead "
                        f"(for example {item.split('#')[0].strip()!r})"
                    )
        return value

    @model_validator(mode="after")
    def _expected_ids_match_category(self) -> "Question":
        ids = [str(i) for i in self.expected_ids]
        if len(set(ids)) != len(ids):
            raise ValueError(f"expected_ids lists a section twice: {ids}")
        if self.category == "out_of_scope" and ids:
            raise ValueError(f"an out_of_scope question must have expected_ids: [], but has {ids}")
        if self.category != "out_of_scope" and not ids:
            raise ValueError(f"a {self.category} question needs at least one expected id")
        return self


class _UniqueKeyLoader(yaml.SafeLoader):
    """YAML's safe loader, except that a key written twice in one entry is an error.

    PyYAML otherwise keeps the last one without a word, so a duplicated `split:` line would quietly
    move a question between dev and test, which is exactly what R5.10 must rule out.
    """

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Hashable, Any]:
        seen: set[Hashable] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=True)
            if isinstance(key, Hashable):  # an unhashable key is reported by the parent class
                if key in seen:
                    raise yaml.constructor.ConstructorError(
                        None, None, f"the key {key!r} appears twice in one entry", key_node.start_mark
                    )
                seen.add(key)
        return super().construct_mapping(node, deep)


def load_questions(path: Path = QUESTIONS_PATH) -> list[Question]:
    """Read and validate the question file. Returns the questions in file order, or raises `EvalSetError`."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise EvalSetError(f"{path}: file not found. Write the questions there; the format is in askact/evaluation/schema.py") from None
    except (OSError, UnicodeDecodeError) as exc:
        raise EvalSetError(f"{path}: cannot read the file as UTF-8 text ({exc})") from exc

    loader = _UniqueKeyLoader(text)
    loader.name = str(path)  # so a YAML error says `in "eval/questions.yaml", line 7` rather than `<unicode string>`
    try:
        root = loader.get_single_node()
        if root is None:  # nothing but whitespace or comments
            raise EvalSetError(f"{path}: the file contains no questions")
        if not isinstance(root, yaml.SequenceNode):
            raise EvalSetError(f"{path}: the file must contain a list of questions, each starting with `- id: ...`")
        entries = [(node.start_mark.line + 1, loader.construct_object(node, deep=True)) for node in root.value]
    except yaml.YAMLError as exc:
        raise EvalSetError(f"{path}: not valid YAML: {_one_line(exc)}") from exc
    if not entries:
        raise EvalSetError(f"{path}: the file contains no questions")

    questions: list[Question] = []
    problems: list[str] = []
    for number, (line, raw) in enumerate(entries, start=1):
        label = _label(number, line, raw)
        if not isinstance(raw, dict):
            problems.append(f"{label}: must be a mapping with id, question, split, category and expected_ids")
            continue
        try:
            questions.append(Question.model_validate(raw))
        except ValidationError as exc:
            problems.extend(f"{label}: {message}" for message in _messages(exc))

    first_line: dict[str, int] = {}
    for line, raw in entries:
        entry_id = raw.get("id") if isinstance(raw, dict) else None
        if isinstance(entry_id, str) and entry_id:
            if entry_id in first_line:
                problems.append(f"id {entry_id!r} (line {line}): already used by the entry on line {first_line[entry_id]}")
            else:
                first_line[entry_id] = line

    if problems:
        noun = "problem" if len(problems) == 1 else "problems"
        raise EvalSetError(f"{path}: {len(problems)} {noun}:\n" + "\n".join(f"  - {p}" for p in problems))
    return questions


def _label(number: int, line: int, raw: object) -> str:
    """How an error names its entry: its position and line always, and its id when it has a usable one."""
    entry_id = raw.get("id") if isinstance(raw, dict) else None
    named = f", id {entry_id!r}" if isinstance(entry_id, str) and entry_id.strip() else ""
    return f"entry {number} (line {line}{named})"


def _messages(exc: ValidationError) -> list[str]:
    messages = []
    for error in exc.errors():
        where = ""
        for part in error["loc"]:  # ("expected_ids", 1) reads best as expected_ids[1]
            where += f"[{part}]" if isinstance(part, int) else (f".{part}" if where else str(part))
        text = error["msg"].removeprefix("Value error, ")
        if where and error["type"] not in ("missing", "extra_forbidden"):
            text += f" (got {_short(error['input'])})"
        messages.append(f"{where}: {text}" if where else text)
    return messages


def _short(value: object, limit: int = 60) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _one_line(exc: yaml.YAMLError) -> str:
    return " ".join(str(exc).split())
