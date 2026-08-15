"""BM25 keyword search over the local ``corpus/`` folder.

Retrieval here is deliberately a *tool*, not the architecture. There is no
vector store and no embedding call: the point of this project is orchestration,
and a dependency-light BM25 index keeps setup at ``pip install`` while still
giving the researcher node something real to call.

Documents are chunked by paragraph so a hit returns a quotable passage rather
than a whole file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from langchain_core.tools import tool

CORPUS_DIR = Path("corpus")
SUFFIXES = {".txt", ".md"}
MAX_CHUNK_CHARS = 1200


@dataclass(frozen=True)
class Chunk:
    source: str
    index: int
    text: str


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _split(text: str) -> list[str]:
    chunks: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        while len(para) > MAX_CHUNK_CHARS:
            cut = para.rfind(" ", 0, MAX_CHUNK_CHARS)
            cut = cut if cut > 0 else MAX_CHUNK_CHARS
            chunks.append(para[:cut].strip())
            para = para[cut:].strip()
        if para:
            chunks.append(para)
    return chunks


def load_chunks(corpus_dir: Path | str = CORPUS_DIR) -> list[Chunk]:
    corpus_dir = Path(corpus_dir)
    chunks: list[Chunk] = []
    if not corpus_dir.exists():
        return chunks
    for path in sorted(corpus_dir.rglob("*")):
        if path.suffix.lower() not in SUFFIXES or not path.is_file():
            continue
        if path.name.lower() == "readme.md":
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, chunk in enumerate(_split(text)):
            chunks.append(Chunk(source=path.name, index=i, text=chunk))
    return chunks


class CorpusIndex:
    """BM25 index, with a graceful degradation to term overlap.

    If ``rank_bm25`` is unavailable the index still answers queries using a
    plain overlap score. Worse ranking, but the graph keeps running and the
    degradation is visible in the tool's own output.
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self._corpus = [_tokenize(c.text) for c in chunks]
        self._bm25 = None
        self.backend = "overlap"
        if chunks:
            try:
                from rank_bm25 import BM25Okapi

                self._bm25 = BM25Okapi(self._corpus)
                self.backend = "bm25"
            except ImportError:
                pass

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        terms = _tokenize(query)
        if not terms:
            return []
        if self._bm25 is not None:
            scores = list(self._bm25.get_scores(terms))
        else:
            wanted = set(terms)
            scores = [
                len(wanted & set(doc)) / len(wanted) if wanted else 0.0
                for doc in self._corpus
            ]
        pairs = zip(self.chunks, scores, strict=False)
        ranked = sorted(pairs, key=lambda p: p[1], reverse=True)
        return [(chunk, float(score)) for chunk, score in ranked[:k] if score > 0]


@lru_cache(maxsize=4)
def get_index(corpus_dir: str = str(CORPUS_DIR)) -> CorpusIndex:
    return CorpusIndex(load_chunks(corpus_dir))


def search_corpus(query: str, k: int = 4, corpus_dir: str | None = None) -> str:
    """Plain-function core, so tests can call it without the LangChain wrapper."""
    index = get_index(corpus_dir or str(CORPUS_DIR))
    if not index.chunks:
        return (
            "CORPUS EMPTY: no .txt or .md documents found in corpus/. "
            "No evidence is available from local sources."
        )
    hits = index.search(query, k=k)
    if not hits:
        return f"No corpus passage matched the query: {query!r}"

    lines = [f"{len(hits)} passage(s) for {query!r} (backend={index.backend}):"]
    for chunk, score in hits:
        lines.append(
            f"\n--- source: {chunk.source} #chunk{chunk.index} "
            f"(score {score:.3f}) ---\n{chunk.text}"
        )
    return "\n".join(lines)


@tool("corpus_search")
def corpus_search(query: str, k: int = 4) -> str:
    """Search the local document corpus for passages relevant to a query.

    Use this first for any factual question. Returns ranked passages with their
    source filename so they can be quoted as evidence.
    """
    return search_corpus(query, k=k)
