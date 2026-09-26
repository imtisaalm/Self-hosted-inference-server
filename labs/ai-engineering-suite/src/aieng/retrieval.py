"""Offset-preserving chunks, BM25, exact vector search, RRF, and reranking."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from .core import number, positive_int


def terms(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), re.UNICODE)


def unit(vector: Sequence[float]) -> tuple[float, ...]:
    data = tuple(float(x) for x in vector)
    if not data or not all(math.isfinite(x) for x in data):
        raise ValueError("vector must be nonempty and finite")
    norm = math.hypot(*data)
    if norm == 0 or not math.isfinite(norm):
        raise ValueError("vector must have a finite, nonzero norm")
    return tuple(x / norm for x in data)


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimension mismatch")
    return max(-1.0, min(1.0, sum(a * b for a, b in zip(unit(left), unit(right)))))


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    text: str
    start: int
    end: int


def chunk_document(source: str, text: str, size: int = 160, overlap: int = 24) -> list[Chunk]:
    positive_int(size, "size")
    if not isinstance(overlap, int) or isinstance(overlap, bool) or not 0 <= overlap < size:
        raise ValueError("overlap must be an integer in [0, size)")
    if not source:
        raise ValueError("source must be nonempty")
    tokens = list(re.finditer(r"\S+", text))
    output = []
    for start in range(0, len(tokens), size - overlap):
        stop = min(start + size, len(tokens))
        begin, end = tokens[start].start(), tokens[stop - 1].end()
        output.append(Chunk(f"{source}:{begin}:{end}", source, text[begin:end], begin, end))
        if stop == len(tokens):
            break
    return output


class BM25:
    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        number(k1, "k1", 0.000001)
        if not 0 <= number(b, "b") <= 1:
            raise ValueError("b must be <= 1")
        self.chunks, self.k1, self.b = tuple(chunks), k1, b
        self.counts = [Counter(terms(chunk.text)) for chunk in chunks]
        self.lengths = [sum(count.values()) for count in self.counts]
        self.avg = sum(self.lengths) / len(chunks) if chunks else 1
        self.df = Counter(term for count in self.counts for term in count)

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        positive_int(k, "k")
        scores = []
        for i, count in enumerate(self.counts):
            score = 0.0
            for term in set(terms(query)):
                freq = count[term]
                if freq:
                    idf = math.log1p((len(self.chunks) - self.df[term] + 0.5) / (self.df[term] + 0.5))
                    norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg)
                    score += idf * freq * (self.k1 + 1) / (freq + norm)
            if score > 0:
                scores.append((i, score))
        return sorted(scores, key=lambda value: (-value[1], value[0]))[:k]


class DenseIndex:
    """Exact cosine index over supplied embeddings; no hidden model downloads."""

    def __init__(self, vectors: Iterable[Sequence[float]]) -> None:
        self.vectors = tuple(unit(vector) for vector in vectors)
        if len({len(vector) for vector in self.vectors}) > 1:
            raise ValueError("inconsistent vector dimensions")

    def search(self, query: Sequence[float], k: int = 10) -> list[tuple[int, float]]:
        positive_int(k, "k")
        normalized = unit(query)
        results = [(i, cosine(normalized, vector)) for i, vector in enumerate(self.vectors)]
        return sorted(results, key=lambda value: (-value[1], value[0]))[:k]


class TfidfEncoder:
    """Offline lexical vector baseline, explicitly NOT a neural semantic encoder."""

    def __init__(self, texts: Sequence[str]) -> None:
        counts = [set(terms(text)) for text in texts]
        self.vocab = sorted(set().union(*counts)) if counts else []
        df = Counter(word for count in counts for word in count)
        self.idf = {word: math.log((1 + len(texts)) / (1 + df[word])) + 1 for word in self.vocab}

    def __call__(self, text: str) -> tuple[float, ...]:
        count = Counter(terms(text))
        # A tiny explicit OOV coordinate keeps empty/unseen queries defined.
        return tuple(count[word] * self.idf[word] for word in self.vocab) + (1e-12,)


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    fusion_score: float
    rerank_score: float


def coverage_reranker(query: str, text: str) -> float:
    query_terms, text_terms = set(terms(query)), set(terms(text))
    if not query_terms:
        return 0.0
    phrase = " ".join(terms(query)) in " ".join(terms(text))
    return len(query_terms & text_terms) / len(query_terms) + 0.25 * phrase


class RetrievalStack:
    def __init__(
        self, chunks: Sequence[Chunk], encoder: Callable[[str], Sequence[float]],
        reranker: Callable[[str, str], float] = coverage_reranker,
    ) -> None:
        if len({chunk.id for chunk in chunks}) != len(chunks):
            raise ValueError("chunk IDs must be unique")
        self.chunks, self.encoder, self.reranker = tuple(chunks), encoder, reranker
        self.lexical = BM25(chunks)
        self.dense = DenseIndex(encoder(chunk.text) for chunk in chunks)

    def search(self, query: str, k: int = 5, candidate_k: int = 20) -> list[Hit]:
        positive_int(k, "k")
        positive_int(candidate_k, "candidate_k")
        if not terms(query) or not self.chunks:
            return []
        candidate_k = max(k, candidate_k)
        rankings = [self.lexical.search(query, candidate_k), self.dense.search(self.encoder(query), candidate_k)]
        fusion: dict[int, float] = {}
        for ranking in rankings:
            for rank, (index, _) in enumerate(ranking, 1):
                fusion[index] = fusion.get(index, 0) + 1 / (60 + rank)
        hits = []
        for index, score in fusion.items():
            rerank = self.reranker(query, self.chunks[index].text)
            if not math.isfinite(rerank):
                raise ValueError("reranker produced a nonfinite score")
            hits.append(Hit(self.chunks[index], score, rerank))
        return sorted(hits, key=lambda hit: (-hit.rerank_score, -hit.fusion_score, hit.chunk.id))[:k]
