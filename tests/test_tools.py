"""Tools: BM25 corpus search, safe calculator, SSRF guard. No API key required."""

from __future__ import annotations

import pytest

from agent.tools.calculator import CalculatorError, evaluate
from agent.tools.corpus_search import CorpusIndex, load_chunks, search_corpus
from agent.tools.fetch_url import FetchError, _assert_public, _strip_html


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "retention.md").write_text(
        "Customer retention reached 84 percent in the third quarter.\n\n"
        "Churn among enterprise accounts fell to 3 percent.\n",
        encoding="utf-8",
    )
    (tmp_path / "hiring.txt").write_text(
        "The team grew from twelve to nineteen engineers during the year.\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("Drop documents here.\n", encoding="utf-8")
    (tmp_path / "ignored.pdf").write_bytes(b"%PDF-1.4")
    return tmp_path


def test_only_text_documents_are_indexed(corpus):
    sources = {c.source for c in load_chunks(corpus)}
    assert sources == {"retention.md", "hiring.txt"}  # README and PDF excluded


def test_paragraphs_become_separate_chunks(corpus):
    chunks = [c for c in load_chunks(corpus) if c.source == "retention.md"]
    assert len(chunks) == 2


def test_search_ranks_the_relevant_chunk_first(corpus):
    index = CorpusIndex(load_chunks(corpus))
    hits = index.search("retention quarter", k=2)
    assert hits
    assert "retention" in hits[0][0].text.lower()


def test_search_output_names_its_sources(corpus):
    out = search_corpus("engineers", corpus_dir=str(corpus))
    assert "hiring.txt" in out
    assert "nineteen engineers" in out


def test_empty_corpus_says_so_rather_than_failing(tmp_path):
    assert "CORPUS EMPTY" in search_corpus("anything", corpus_dir=str(tmp_path))


def test_no_match_is_reported_plainly(corpus):
    assert "No corpus passage matched" in search_corpus(
        "zzzzqqqq", corpus_dir=str(corpus)
    )


# -- calculator ---------------------------------------------------------


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("2 + 2", 4),
        ("(1240 - 890) / 890 * 100", pytest.approx(39.3258, rel=1e-4)),
        ("-3 ** 2", -9),
        ("7 // 2", 3),
        ("2 ** 10", 1024),
    ],
)
def test_arithmetic(expression, expected):
    assert evaluate(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('rm -rf /')",
        "open('/etc/passwd').read()",
        "(1).__class__",
        "[x for x in range(10)]",
        "9 ** 9 ** 9",
        "1 / 0",
        "'a' * 5",
        "x + 1",
    ],
)
def test_non_arithmetic_input_is_refused(expression):
    with pytest.raises(CalculatorError):
        evaluate(expression)


def test_overlong_expression_is_refused():
    with pytest.raises(CalculatorError, match="too long"):
        evaluate("1+" * 200 + "1")


# -- fetch_url ----------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    ["http://localhost:8080/x", "http://127.0.0.1/", "http://169.254.169.254/latest/"],
)
def test_private_addresses_are_refused(url):
    with pytest.raises(FetchError):
        _assert_public(url)


def test_non_http_schemes_are_refused():
    with pytest.raises(FetchError, match="scheme"):
        _assert_public("file:///etc/passwd")


def test_html_is_stripped_to_text():
    html = "<html><script>evil()</script><p>Hello &amp; welcome</p></html>"
    text = _strip_html(html)
    assert "evil()" not in text
    assert "Hello & welcome" in text
