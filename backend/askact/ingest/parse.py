"""Ingestion step 2: turn the Official Journal XHTML into recitals, articles and annexes (R1.5-R1.7).

The document marks its structure with ids: `rct_N` (recital), `art_N` (article) and `anx_ROMAN`
(annex). Lists of points such as "(a) ..." are two-column tables. The text of each section is
flattened to plain lines, one block per top-level unit, for the chunker.

The parser is deliberately strict. Markup it has not seen raises `ParseError` naming the section,
because silently dropping an element would silently delete part of the law from the index.
Section counts alone would not catch that.
"""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from bs4 import BeautifulSoup
from bs4.element import Comment, NavigableString, PageElement, Tag

from askact.ingest.fetch import OJ_DOCUMENT_ID
from askact.models import Kind, SectionId

_SECTION_TAG_ID = re.compile(r"(rct|art|anx)_([A-Za-z0-9]+)")
_KIND_BY_PREFIX: dict[str, Kind] = {"rct": "recital", "art": "article", "anx": "annex"}
# A bare ";" or "." that follows a block belongs to the end of the previous line.
_TRAILING_PUNCTUATION = re.compile(r"[;,.:]+")


class ParseError(Exception):
    """The document is not the expected Official Journal text, or its markup is not understood."""


@dataclass(frozen=True)
class Counts:
    recitals: int
    articles: int
    annexes: int


# The number of sections in the original Official Journal text of Regulation (EU) 2024/1689.
# Unset until confirmed by the project owner against the parser's output on the real file:
# until then `verify` refuses to run, rather than pin a number nobody has checked.
EXPECTED_COUNTS: Counts | None = None


@dataclass(frozen=True)
class Section:
    section_id: SectionId
    title: str | None  # articles and annexes; None for recitals
    # One entry per top-level unit of the section (a numbered paragraph with its sub-points, a
    # definition, an annex heading, ...), each possibly several lines. The chunker packs these.
    blocks: tuple[str, ...]

    @property
    def text(self) -> str:
        return "\n".join(self.blocks)


@dataclass(frozen=True)
class ParsedAct:
    document_title: str
    recitals: tuple[Section, ...]
    articles: tuple[Section, ...]
    annexes: tuple[Section, ...]

    @property
    def sections(self) -> tuple[Section, ...]:
        return self.recitals + self.articles + self.annexes

    @property
    def counts(self) -> Counts:
        return Counts(len(self.recitals), len(self.articles), len(self.annexes))


class _Markup(Exception):
    """Internal: markup that is not understood. Wrapped into ParseError with the section id."""


# --- inline text ------------------------------------------------------------------------

def _classes(tag: Tag) -> set[str]:
    # With the XML parser a class attribute is one string, not a list.
    value = tag.get("class") or ""
    return set(value.split()) if isinstance(value, str) else set(value)


def _clean(text: str) -> str:
    """Collapse whitespace, including the non-breaking spaces the document uses everywhere."""
    return re.sub(r"\s+", " ", text).strip()


def _is_footnote_ref(tag: Tag) -> bool:
    return tag.name == "a" and any("oj-note-tag" in _classes(span) for span in tag.find_all("span"))


def _inline_nodes(nodes: Iterable[PageElement]) -> str:
    """The text of inline content: text, spans and links, with two document-specific rules.

    * A footnote reference "(1)" is dropped, and the space before it, so no stray number or
      " ," is left in the sentence. The footnote texts themselves are outside the sections.
    * A superscript is an exponent and is written with "^": the compute threshold in Article 51
      is 10<sup>25</sup>, and plain text extraction would turn it into the number 1025.
    """
    parts: list[str] = []

    def drop_trailing_space() -> None:
        while parts and not parts[-1].strip():
            parts.pop()
        if parts:
            parts[-1] = parts[-1].rstrip()

    def walk(children: Iterable[PageElement], in_link: bool) -> None:
        for child in children:
            if isinstance(child, Comment):
                continue
            if isinstance(child, NavigableString):
                parts.append(str(child))
            elif not isinstance(child, Tag):
                continue
            elif _is_footnote_ref(child):
                drop_trailing_space()
            elif child.name in ("span", "a"):
                # Inside a link the superscript is the "(*)" note marker of a quoted text, not an exponent.
                if child.name == "span" and "oj-super" in _classes(child) and not in_link:
                    parts.append("^")
                walk(child.children, in_link or child.name == "a")
            else:
                raise _Markup(f"unexpected <{child.name}> inside a paragraph")

    walk(nodes, False)
    return "".join(parts)


def _inline_raw(paragraph: Tag) -> str:
    return _inline_nodes(paragraph.children)


# --- blocks of lines ---------------------------------------------------------------------

def _is_inline_paragraph(tag: Tag) -> bool:
    """Annex VI sets a number and its text as two paragraphs styled to share one line."""
    return tag.name == "p" and "display:inline" in re.sub(r"\s+", "", str(tag.get("style") or ""))


def _lines(nodes: Sequence[PageElement]) -> list[str]:
    """Flatten document-order nodes (a section body, a div, or a table cell) into text lines."""
    lines: list[str] = []
    run: list[str] = []  # inline content waiting to become one line

    def flush() -> None:
        text = _clean("".join(run))
        run.clear()
        if not text:
            return
        if lines and _TRAILING_PUNCTUATION.fullmatch(text):
            lines[-1] += text
        else:
            lines.append(text)

    for node in nodes:
        if isinstance(node, Comment):
            continue
        if isinstance(node, NavigableString):
            run.append(str(node))
        elif not isinstance(node, Tag) or node.name in ("col", "colgroup"):
            continue
        elif node.name in ("span", "a"):
            run.append(_inline_nodes([node]))  # the node itself, so a bare superscript is seen too
        elif _is_inline_paragraph(node):
            run.append(_inline_raw(node))
        elif node.name == "p":
            flush()
            text = _clean(_inline_raw(node))
            if text:
                lines.append(text)
        elif node.name == "div":
            flush()
            lines += _lines(list(node.children))
        elif node.name == "table":
            flush()
            lines += _table_lines(node)
        else:
            raise _Markup(f"unexpected <{node.name}> in the text")
    flush()
    return lines


def _table_lines(table: Tag) -> list[str]:
    """One line per row, "(a) text"; a cell that holds more (a sub-list) continues on later lines."""
    rows: list[Tag] = []
    for child in table.children:
        if not isinstance(child, Tag) or child.name in ("col", "colgroup"):
            continue  # whitespace, comments, column widths
        if child.name == "tbody":
            rows += [row for row in child.children if isinstance(row, Tag)]
        elif child.name == "tr":
            rows.append(child)
        else:
            raise _Markup(f"unexpected <{child.name}> in a table")

    lines: list[str] = []
    for row in rows:
        cells = [cell for cell in row.children if isinstance(cell, Tag)]
        if row.name != "tr" or any(cell.name != "td" for cell in cells):
            raise _Markup("a table row that is not made of <td> cells")
        # Annex rows sometimes carry an empty spacer cell; what is left must be label + content.
        filled = [cell_lines for cell_lines in (_lines(list(cell.children)) for cell in cells) if cell_lines]
        if len(filled) != 2:
            raise _Markup(f"a table row with {len(filled)} non-empty cells, expected a label and its content")
        label, content = filled
        if len(label) != 1:
            raise _Markup("a table row whose label spans several lines")
        lines.append(f"{label[0]} {content[0]}")
        lines += content[1:]
    return lines


def _blocks(body: Sequence[PageElement]) -> tuple[str, ...]:
    blocks = ("\n".join(_lines([node])) for node in body)
    return tuple(block for block in blocks if block)


# --- sections ----------------------------------------------------------------------------

def _only(children: list[Tag], tag: str, css_class: str, what: str) -> Tag:
    found = [c for c in children if c.name == tag and css_class in _classes(c)]
    if len(found) != 1:
        raise _Markup(f"expected exactly one {what} (<{tag} class={css_class!r}>), found {len(found)}")
    return found[0]


def _heading_text(tag: Tag, what: str) -> str:
    text = _clean(_inline_raw(tag))
    if not text:
        raise _Markup(f"empty {what}")
    return text


def _parse_recital(element: Tag, section_id: SectionId) -> Section:
    lines = _lines(list(element.children))
    label = f"({section_id.number}) "
    if not lines or not lines[0].startswith(label):
        raise _Markup(f"the text does not start with its label {label.strip()!r}")
    lines[0] = lines[0][len(label):]
    if not lines[0]:
        raise _Markup("the recital has no text")
    return Section(section_id, None, ("\n".join(lines),))


def _check_heading_number(heading: str, section_id: SectionId) -> None:
    try:
        shown = SectionId.parse(heading)
    except ValueError as exc:
        raise _Markup(f"heading {heading!r} is not a section number: {exc}") from exc
    if shown != section_id:
        raise _Markup(f"heading {heading!r} does not match the id {section_id}")


def _parse_article(element: Tag, section_id: SectionId) -> Section:
    children = [c for c in element.children if isinstance(c, Tag)]
    number = _only(children, "p", "oj-ti-art", "article number")
    title_div = _only(children, "div", "eli-title", "title")
    title_p = [c for c in title_div.children if isinstance(c, Tag)]
    _check_heading_number(_heading_text(number, "article number"), section_id)
    title = _heading_text(_only(title_p, "p", "oj-sti-art", "title text"), "title")
    body = [c for c in element.children if c is not number and c is not title_div]
    return Section(section_id, title, _blocks(body))


def _parse_annex(element: Tag, section_id: SectionId) -> Section:
    children = [c for c in element.children if isinstance(c, Tag)]
    headings = [c for c in children if c.name == "p" and "oj-doc-ti" in _classes(c)]
    # Identity, not ==: bs4 compares tags by content. The headings must be the first two elements.
    if len(headings) != 2 or any(a is not b for a, b in zip(children, headings)):
        raise _Markup(f"expected the annex number and title as the first two <p class='oj-doc-ti'>, found {len(headings)}")
    _check_heading_number(_heading_text(headings[0], "annex number"), section_id)
    title = _heading_text(headings[1], "annex title")
    body = [c for c in element.children if c is not headings[0] and c is not headings[1]]
    return Section(section_id, title, _blocks(body))


_PARSERS = {"recital": _parse_recital, "article": _parse_article, "annex": _parse_annex}


def _find_section_elements(soup: BeautifulSoup) -> list[tuple[SectionId, Tag]]:
    found: list[tuple[SectionId, Tag]] = []
    # A function, not a regex: bs4 would use re.search and also match the title divs "art_1.tit_1".
    for element in soup.find_all("div", id=lambda value: bool(value and _SECTION_TAG_ID.fullmatch(value))):
        match = _SECTION_TAG_ID.fullmatch(str(element["id"]))
        assert match is not None  # guaranteed by the filter above
        prefix, number = match.groups()
        kind = _KIND_BY_PREFIX[prefix]
        try:
            section_id = SectionId(kind, number.upper() if kind == "annex" else str(int(number)))
        except ValueError as exc:
            raise ParseError(f"{element['id']}: not a valid section id ({exc})") from exc
        found.append((section_id, element))
    return found


def parse_act(source: bytes | str) -> ParsedAct:
    """Parse the Official Journal XHTML of Regulation (EU) 2024/1689 into sections.

    Raises `ParseError` if the document is not that text or contains markup that is not understood.
    The pinned section counts are checked separately, by `verify`.
    """
    soup = BeautifulSoup(source, "xml")
    title = _clean(soup.title.get_text()) if soup.title else ""
    if not title.startswith(OJ_DOCUMENT_ID):
        # A consolidated version has a different file id (02024R1689-...) and later amendments.
        raise ParseError(
            f"the document title is {title!r}, expected it to start with {OJ_DOCUMENT_ID!r}: "
            "this is not the original Official Journal text of Regulation (EU) 2024/1689"
        )

    sections: dict[Kind, list[Section]] = {"recital": [], "article": [], "annex": []}
    seen: set[SectionId] = set()
    for section_id, element in _find_section_elements(soup):
        if section_id in seen:
            raise ParseError(f"{section_id} appears more than once")
        seen.add(section_id)
        try:
            sections[section_id.kind].append(_PARSERS[section_id.kind](element, section_id))
        except _Markup as exc:
            raise ParseError(f"{section_id}: {exc}") from exc

    if not any(sections.values()):
        raise ParseError("no recitals, articles or annexes found: the markup is not the expected one")
    return ParsedAct(title, tuple(sections["recital"]), tuple(sections["article"]), tuple(sections["annex"]))


# --- the pin -----------------------------------------------------------------------------

def _roman(n: int) -> str:
    out = ""
    for value, symbol in ((10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while n >= value:
            out, n = out + symbol, n - value
    return out


def verify(act: ParsedAct, expected: Counts | None = None) -> None:
    """Check the parsed counts against the pinned ones and that the numbering has no gaps.

    `expected` defaults to EXPECTED_COUNTS. Any mismatch aborts the build (R1.13): it means the
    document is not the pinned one, or the parser lost or invented a section.
    """
    expected = expected or EXPECTED_COUNTS
    if expected is None:
        raise ParseError("EXPECTED_COUNTS has not been set yet, so the parsed counts cannot be verified")
    if act.counts != expected:
        raise ParseError(
            f"parsed {act.counts.recitals} recitals, {act.counts.articles} articles and "
            f"{act.counts.annexes} annexes, but the pinned text has {expected.recitals}, "
            f"{expected.articles} and {expected.annexes}"
        )
    for kind, sections in (("recital", act.recitals), ("article", act.articles), ("annex", act.annexes)):
        want = [str(n) if kind != "annex" else _roman(n) for n in range(1, len(sections) + 1)]
        got = [s.section_id.number for s in sections]
        if got != want:
            gap = next((g, w) for g, w in zip(got, want) if g != w)
            raise ParseError(f"the {kind}s are not numbered 1..{len(sections)} in order: found {gap[0]} where {gap[1]} was expected")
