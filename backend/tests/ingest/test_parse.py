import copy
import re
from pathlib import Path

import httpx
import pytest
from bs4 import BeautifulSoup

from askact.config import Settings
from askact.ingest import parse
from askact.ingest.fetch import get_source
from askact.ingest.parse import Counts, ParseError, parse_act, verify
from askact.models import SectionId

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "mini_act.html"
REAL = Path(__file__).resolve().parents[3] / "data" / "raw" / "ai-act-oj-2024-1689.html"
MANUAL = REAL.with_name("ai-act-oj-2024-1689.manual.html")
FIXTURE_COUNTS = Counts(recitals=3, articles=2, annexes=1)


@pytest.fixture(scope="module")
def mini():
    return parse_act(FIXTURE.read_bytes())


def section(act, text: str):
    sid = SectionId.parse(text)
    return next(s for s in act.sections if s.section_id == sid)


def doc(*sections: str, title: str = "L_202401689EN.000101.fmx.xml") -> str:
    """A minimal document around hand-written sections, for testing one quirk at a time."""
    return (
        '<?xml version="1.0" encoding="UTF-8"?><html xmlns="http://www.w3.org/1999/xhtml">'
        f"<head><title>{title}</title></head><body>{''.join(sections)}</body></html>"
    )


def row(label: str, content: str) -> str:
    return (
        '<table><col width="4%"/><col width="96%"/><tbody><tr>'
        f'<td valign="top"><p class="oj-normal">{label}</p></td>'
        f'<td valign="top"><p class="oj-normal">{content}</p></td></tr></tbody></table>'
    )


def article(number: int, title: str, body: str) -> str:
    return (
        f'<div class="eli-subdivision" id="art_{number}"><p class="oj-ti-art">Article {number}</p>'
        f'<div class="eli-title" id="art_{number}.tit_1"><p class="oj-sti-art">{title}</p></div>{body}</div>'
    )


def one_article_text(body: str) -> str:
    return parse_act(doc(article(1, "T", body))).articles[0].text


# --- the fixture (a trimmed copy of the real page) -----------------------------------------

def test_fixture_counts(mini):
    assert mini.counts == FIXTURE_COUNTS
    assert [str(s.section_id) for s in mini.recitals] == ["recital:1", "recital:8", "recital:23"]
    assert [str(s.section_id) for s in mini.articles] == ["article:1", "article:5"]
    assert [str(s.section_id) for s in mini.annexes] == ["annex:VI"]


def test_sections_come_back_in_document_order(mini):
    assert [str(s.section_id) for s in mini.sections] == [
        "recital:1", "recital:8", "recital:23", "article:1", "article:5", "annex:VI",
    ]


def test_content_outside_the_sections_is_ignored(mini):
    everything = " ".join(s.text for s in mini.sections)
    assert "outside any section" not in everything
    assert "Footnote text that sits outside" not in everything
    assert "THE EUROPEAN PARLIAMENT AND THE COUNCIL" not in everything


# --- article number and title (R1.7) ------------------------------------------------------

def test_article_number_and_title_are_captured(mini):
    a5 = section(mini, "article:5")
    assert a5.section_id == SectionId("article", "5")
    assert a5.title == "Prohibited AI practices"


def test_titles_are_kept_exactly_as_published_including_a_typo_in_the_source(mini):
    # The Official Journal text itself has a stray backtick in Article 1's title. We do not edit law.
    assert section(mini, "article:1").title == "Subject matter`"


def test_annex_number_and_title_are_captured(mini):
    annex = section(mini, "annex:VI")
    assert annex.section_id == SectionId("annex", "VI")
    assert annex.title == "Conformity assessment procedure based on internal control"


def test_recitals_have_no_title_and_no_leading_label(mini):
    r1 = section(mini, "recital:1")
    assert r1.title is None
    assert r1.text.startswith("The purpose of this Regulation is to improve the functioning of the internal market")
    assert not r1.text.startswith("(1)")
    assert len(r1.blocks) == 1


# --- table flattening (lists of points) ---------------------------------------------------

def test_a_list_of_points_becomes_one_line_per_row(mini):
    lines = section(mini, "article:1").text.split("\n")
    assert lines[1] == "2. This Regulation lays down:"
    assert lines[2].startswith("(a) harmonised rules for the placing on the market")
    assert lines[3] == "(b) prohibitions of certain AI practices;"
    assert lines[-1].startswith("(g) measures to support innovation")


def test_nested_sub_points_stay_with_their_parent_in_order(mini):
    lines = section(mini, "article:5").text.split("\n")
    point_c = next(i for i, line in enumerate(lines) if line.startswith("(c) the placing on the market"))
    assert lines[point_c + 1].startswith("(i) detrimental or unfavourable treatment")
    assert lines[point_c + 2].startswith("(ii) detrimental or unfavourable treatment")
    assert lines[point_c + 3].startswith("(d) ")


def test_blocks_are_the_top_level_units_of_a_section(mini):
    a5 = section(mini, "article:5")
    assert len(a5.blocks) == 8  # the eight numbered paragraphs of Article 5
    assert a5.blocks[0].startswith("1. The following AI practices shall be prohibited:")
    assert a5.blocks[0].count("\n") > 5  # paragraph 1 carries its (a), (b), ... points
    assert a5.text == "\n".join(a5.blocks)
    assert len(section(mini, "article:1").blocks) == 2


def test_inline_paragraph_pairs_form_one_line(mini):
    lines = section(mini, "annex:VI").text.split("\n")
    assert lines[0].startswith("1. The conformity assessment procedure based on internal control")
    assert lines[1].startswith("2. The provider verifies")


# --- the document's quirks, one at a time ---------------------------------------------------

def test_footnote_references_are_dropped_with_the_space_before_them(mini):
    text = section(mini, "recital:8").text
    assert "European Council, and it ensures" in text   # was "Council\u00a0(5), and"
    assert not re.search(r"\s[,.;]", text)               # no space left before punctuation
    assert not re.search(r"Council\s*\(\d", text)


def test_a_footnote_reference_followed_by_a_word_leaves_one_space():
    html = doc(article(1, "T", '<p class="oj-normal">the Committee\u00a0<a href="#n"><span class="oj-super oj-note-tag">1</span></a> and the Bank</p>'))
    assert parse_act(html).articles[0].text == "the Committee and the Bank"


def test_a_superscript_exponent_is_written_with_a_caret():
    """Plain text extraction would turn 10 to the 25 into the number 1025."""
    text = one_article_text('<p class="oj-normal">greater than 10<span class="oj-super">25</span>.</p>')
    assert text == "greater than 10^25."


def test_the_note_marker_inside_a_quotation_link_is_not_an_exponent():
    text = one_article_text(
        '<p class="oj-note"><a class="oj-quotation" href="#x">(<span class="oj-super">*</span>)</a> Regulation (EU) 2024/1689</p>'
    )
    assert text == "(*) Regulation (EU) 2024/1689"


def test_a_bare_semicolon_after_a_block_joins_the_previous_line():
    """Article 108 ends a quoted paragraph with a ';' that sits in the cell after the <div>."""
    html = (
        '<table><tbody><tr><td><p class="oj-normal">(1)</p></td><td>'
        '<p class="oj-normal">in Article 17, the following paragraph is added:</p>'
        '<div><p class="oj-normal">‘3. The requirements shall be taken into account.’</p></div>;'
        "</td></tr></tbody></table>"
    )
    assert one_article_text(html) == (
        "(1) in Article 17, the following paragraph is added:\n"
        "‘3. The requirements shall be taken into account.’;"
    )


def test_links_keep_their_text_but_not_their_url():
    text = one_article_text('<p class="oj-normal">Directive 2006/42/EC (<a href="http://example.org/x">OJ L 157, 9.6.2006, p. 24</a>);</p>')
    assert text == "Directive 2006/42/EC (OJ L 157, 9.6.2006, p. 24);"
    assert "http" not in text


def test_non_breaking_spaces_become_ordinary_spaces():
    assert one_article_text('<p class="oj-normal">EUR\u00a035\u00a0000\u00a0000</p>') == "EUR 35 000 000"


def test_annex_rows_with_an_empty_spacer_cell_still_give_label_and_content():
    annex = (
        '<div class="eli-container" id="anx_I"><p class="oj-doc-ti">ANNEX I</p><p class="oj-doc-ti">List</p>'
        '<table><tbody><tr><td valign="top"></td><td><p class="oj-normal">1.</p></td>'
        "<td><span>Directive 2006/42/EC;</span></td></tr></tbody></table></div>"
    )
    assert parse_act(doc(annex)).annexes[0].text == "1. Directive 2006/42/EC;"


def test_a_cell_with_a_sub_list_continues_on_the_following_lines():
    nested = row("(a)", "first") + row("(b)", "second")
    html = (
        '<table><tbody><tr><td><p class="oj-normal">(61)</p></td>'
        f'<td><p class="oj-normal">‘widespread infringement’ means:</p>{nested}</td></tr></tbody></table>'
    )
    assert one_article_text(html) == "(61) ‘widespread infringement’ means:\n(a) first\n(b) second"


def test_comments_are_ignored():
    assert one_article_text('<!-- a comment --><p class="oj-normal">text</p>') == "text"


# --- fail loudly on markup we do not understand ----------------------------------------------

@pytest.mark.parametrize("body, message", [
    ('<p class="oj-normal">a <b>bold</b> word</p>', r"article:1: unexpected <b> inside a paragraph"),
    ("<ul><li>x</li></ul>", r"article:1: unexpected <ul> in the text"),
    ("<table><tbody><tr><td><p>only one cell</p></td></tr></tbody></table>", r"1 non-empty cells"),
    (
        "<table><tbody><tr><td><p>a</p></td><td><p>b</p></td><td><p>c</p></td></tr></tbody></table>",
        r"3 non-empty cells",
    ),
    ("<table><tbody><tr><th>(a)</th><td><p>x</p></td></tr></tbody></table>", r"not made of <td> cells"),
    ("<table><thead></thead></table>", r"unexpected <thead> in a table"),
])
def test_unknown_markup_is_an_error_that_names_the_section(body, message):
    with pytest.raises(ParseError, match=message):
        parse_act(doc(article(1, "T", body)))


def test_a_label_spanning_several_lines_is_an_error():
    html = '<table><tbody><tr><td><p>(a)</p><p>(b)</p></td><td><p>x</p></td></tr></tbody></table>'
    with pytest.raises(ParseError, match="label spans several lines"):
        parse_act(doc(article(1, "T", html)))


@pytest.mark.parametrize("section_html, message", [
    # article: heading disagrees with the id
    ('<div id="art_2"><p class="oj-ti-art">Article 3</p><div class="eli-title"><p class="oj-sti-art">T</p></div></div>', "does not match the id article:2"),
    # article: no title
    ('<div id="art_2"><p class="oj-ti-art">Article 2</p></div>', "expected exactly one title"),
    # article: empty title
    ('<div id="art_2"><p class="oj-ti-art">Article 2</p><div class="eli-title"><p class="oj-sti-art"> </p></div></div>', "empty title"),
    # article: heading is not a number
    ('<div id="art_2"><p class="oj-ti-art">Chapter II</p><div class="eli-title"><p class="oj-sti-art">T</p></div></div>', "is not a section number"),
    # recital: label disagrees with id
    ('<div id="rct_4">' + row("(5)", "text") + "</div>", r"does not start with its label '\(4\)'"),
    # recital: no text after the label
    ('<div id="rct_4"><table><tbody><tr><td><p>(4)</p></td><td><p> </p></td></tr></tbody></table></div>', "non-empty cells"),
    # annex: number disagrees with id
    ('<div id="anx_II"><p class="oj-doc-ti">ANNEX III</p><p class="oj-doc-ti">T</p></div>', "does not match the id annex:II"),
    # annex: missing title
    ('<div id="anx_II"><p class="oj-doc-ti">ANNEX II</p></div>', "found 1"),
    # annex: headings not first
    ('<div id="anx_II"><p class="oj-normal">x</p><p class="oj-doc-ti">ANNEX II</p><p class="oj-doc-ti">T</p></div>', "first two"),
])
def test_structure_that_does_not_add_up_is_an_error(section_html, message):
    with pytest.raises(ParseError, match=message):
        parse_act(doc(section_html))


def test_duplicate_sections_are_an_error():
    with pytest.raises(ParseError, match="article:1 appears more than once"):
        parse_act(doc(article(1, "A", ""), article(1, "B", "")))


def test_a_document_without_any_sections_is_an_error():
    with pytest.raises(ParseError, match="no recitals, articles or annexes found"):
        parse_act(doc('<p class="oj-normal">hello</p>'))


def test_title_divs_are_not_mistaken_for_sections():
    """Titles have ids like art_1.tit_1, which a substring match on 'art_1' would also catch."""
    act = parse_act(doc(article(1, "T", "")))
    assert act.counts == Counts(0, 1, 0)


# --- the pin check (R1.5) -----------------------------------------------------------------------

@pytest.mark.parametrize("title", [
    "02024R1689-20260727",                    # the consolidated version
    "L_202401744EN.000101.fmx.xml",           # a different act
    "",
    "something else",
])
def test_a_document_that_is_not_the_original_text_is_rejected(title):
    with pytest.raises(ParseError, match="not the original Official Journal text"):
        parse_act(doc(article(1, "T", ""), title=title))


def test_a_document_without_a_title_is_rejected():
    with pytest.raises(ParseError, match="not the original Official Journal text"):
        parse_act('<html xmlns="http://www.w3.org/1999/xhtml"><body></body></html>')


def test_verify_accepts_matching_counts(mini):
    # The fixture keeps the real ids (1, 8, 23, ...), so it is not contiguous; use a contiguous act.
    act = parse_act(doc(article(1, "A", ""), article(2, "B", "")))
    verify(act, Counts(0, 2, 0))


@pytest.mark.parametrize("wrong", [Counts(1, 2, 0), Counts(0, 3, 0), Counts(0, 2, 1), Counts(0, 1, 0)])
def test_verify_rejects_a_wrong_count_and_says_what_it_found(wrong):
    act = parse_act(doc(article(1, "A", ""), article(2, "B", "")))
    with pytest.raises(ParseError, match=r"parsed 0 recitals, 2 articles and 0 annexes"):
        verify(act, wrong)


def test_verify_rejects_a_gap_in_the_numbering():
    act = parse_act(doc(article(1, "A", ""), article(3, "C", "")))
    with pytest.raises(ParseError, match="found 3 where 2 was expected"):
        verify(act, Counts(0, 2, 0))


def test_verify_checks_annex_numbering_in_roman_numerals():
    def annex(numeral):
        return f'<div id="anx_{numeral}"><p class="oj-doc-ti">ANNEX {numeral}</p><p class="oj-doc-ti">T</p></div>'

    verify(parse_act(doc(annex("I"), annex("II"), annex("III"), annex("IV"))), Counts(0, 0, 4))
    with pytest.raises(ParseError, match="found IV where III was expected"):
        verify(parse_act(doc(annex("I"), annex("II"), annex("IV"))), Counts(0, 0, 3))


def test_verify_refuses_to_run_until_the_expected_counts_are_confirmed(monkeypatch):
    monkeypatch.setattr(parse, "EXPECTED_COUNTS", None)
    act = parse_act(doc(article(1, "A", "")))
    with pytest.raises(ParseError, match="EXPECTED_COUNTS has not been set"):
        verify(act)


def test_verify_uses_the_pinned_counts_by_default(monkeypatch):
    monkeypatch.setattr(parse, "EXPECTED_COUNTS", Counts(0, 1, 0))
    verify(parse_act(doc(article(1, "A", ""))))
    monkeypatch.setattr(parse, "EXPECTED_COUNTS", Counts(0, 2, 0))
    with pytest.raises(ParseError):
        verify(parse_act(doc(article(1, "A", ""))))


# --- the pin check also applies to a manually downloaded file (R1.4, R1.5) -------------------------

def manual_source(tmp_path: Path, content: bytes):
    """Run Task 3's get_source with a failing fetch, so the manual file is what comes back."""
    manual = tmp_path / "manual.html"
    manual.write_bytes(content)
    failing = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(503)))
    settings = Settings(_env_file=None, source_fallback_path=manual)
    source = get_source(settings, raw_dir=tmp_path / "raw", client=failing)
    assert source.origin == "manual"
    return source


def test_a_manual_file_with_the_wrong_title_is_rejected_by_the_parser(tmp_path):
    # It mentions the Official Journal id in its text, so the fetch step's sanity check lets it
    # through; the parser's title check is the real pin and catches that it is another document.
    consolidated = doc(article(1, "T", '<p class="oj-normal">see L_202401689EN</p>'), title="02024R1689-20260727")
    source = manual_source(tmp_path, consolidated.encode())
    with pytest.raises(ParseError, match="not the original Official Journal text"):
        parse_act(source.path.read_bytes())


def test_a_manual_file_with_the_wrong_counts_is_rejected_by_the_pin_check(tmp_path):
    source = manual_source(tmp_path, FIXTURE.read_bytes())
    act = parse_act(source.path.read_bytes())
    assert act.counts == FIXTURE_COUNTS
    with pytest.raises(ParseError, match="parsed 3 recitals, 2 articles and 1 annexes"):
        verify(act, Counts(recitals=180, articles=113, annexes=13))


# --- the real document (R1.12): runs only when the file is present ----------------------------------

real_source = pytest.mark.skipif(
    not REAL.exists(), reason=f"the full source is not present at {REAL} (ingestion downloads it there)"
)


@pytest.fixture(scope="module")
def real_bytes() -> bytes:
    return REAL.read_bytes()


@pytest.fixture(scope="module")
def real_act(real_bytes):
    return parse_act(real_bytes)


@real_source
def test_real_source_numbering_has_no_gaps(real_act):
    verify(real_act, real_act.counts)  # counts trivially match; this checks the 1..n / I..n numbering


@real_source
def test_real_source_has_the_pinned_counts(real_act):
    if parse.EXPECTED_COUNTS is None:
        pytest.skip("EXPECTED_COUNTS has not been confirmed yet")
    assert real_act.counts == parse.EXPECTED_COUNTS
    verify(real_act)


@pytest.mark.skipif(not MANUAL.exists(), reason=f"no manually downloaded copy at {MANUAL}")
def test_a_manually_downloaded_copy_goes_through_the_same_parser_and_pin_check():
    act = parse_act(MANUAL.read_bytes())
    verify(act, act.counts)  # numbering and structure; the pinned counts are checked in the test above


@real_source
def test_real_source_every_section_has_text_and_every_article_and_annex_a_title(real_act):
    assert all(s.blocks for s in real_act.sections)
    assert all(s.title for s in real_act.articles + real_act.annexes)
    assert all(s.title is None for s in real_act.recitals)


@real_source
def test_real_source_known_facts(real_act):
    by = {str(s.section_id): s for s in real_act.sections}
    assert by["article:5"].title == "Prohibited AI practices"
    assert by["article:3"].title == "Definitions"
    assert by["annex:III"].title == "High-risk AI systems referred to in Article 6(2)"
    assert "greater than 10^25." in by["article:51"].text   # the exponent, not "1025"


@real_source
def test_real_source_text_matches_an_independent_reading_of_the_markup(real_bytes, real_act):
    """Nothing is lost, doubled or reordered: for each section, the parser's text equals a plain
    text read of the raw element (footnote references removed, exponents marked), ignoring only
    whitespace. This catches a silently dropped paragraph, which a section count never would.
    The reading below deliberately shares no code with the parser."""
    soup = BeautifulSoup(real_bytes, "xml")

    def has(token):  # the XML parser gives class as one string
        return lambda value: bool(value) and token in value.split()

    def plain(element, kind, number):
        element = copy.copy(element)
        for link in element.find_all("a"):
            if link.find("span", class_=has("oj-note-tag")):
                link.decompose()
        for sup in element.find_all("span", class_=has("oj-super")):
            if not sup.find_parent("a"):
                sup.insert_before("^")
        if kind == "article":
            element.find("p", class_=has("oj-ti-art")).decompose()
            element.find("div", class_=has("eli-title")).decompose()
        if kind == "annex":
            for heading in element.find_all("p", class_=has("oj-doc-ti"))[:2]:
                heading.decompose()
        text = re.sub(r"\s+", "", element.get_text(""))
        if kind == "recital":
            assert text.startswith(f"({number})")
            text = text[len(f"({number})"):]
        return text

    prefix = {"recital": "rct", "article": "art", "annex": "anx"}
    mismatches = []
    for s in real_act.sections:
        element = soup.find("div", id=f"{prefix[s.section_id.kind]}_{s.section_id.number}")
        if re.sub(r"\s+", "", "".join(s.blocks)) != plain(element, s.section_id.kind, s.section_id.number):
            mismatches.append(str(s.section_id))
    assert mismatches == []
