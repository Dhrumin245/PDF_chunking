from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .chunker import Chunk
from .embeddings import generate_query_embedding
from .token_tracker import count_tokens
from .vector_store import VectorStore, search_index


_WORD_PATTERN = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]*")
_TOC_DOT_LINE = re.compile(r"^[.\s]{5,}$")
_PAGE_NUMBER_LINE = re.compile(r"^\d{1,4}$")
_DEFINITION_PATTERNS = (
    " refers to ",
    " fundamentally ",
    " means ",
    " provides ",
    " enables ",
    " allows ",
    " consists of ",
    " responsible for ",
    " used to ",
)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


@dataclass(frozen=True)
class RetrievalResult:
    chunk: Chunk
    score: float
    vector_score: float = 0.0
    keyword_score: float = 0.0
    quality_score: float = 1.0


def retrieve_top_k(
    query: str,
    chunks: list[Chunk],
    vector_store: VectorStore,
    top_k: int = 5,
    context_budget: int = 900,
    embedding_dimensions: int = 384,
) -> list[RetrievalResult]:
    """Retrieve relevant chunks with hybrid scoring and token-budget packing."""

    if not chunks:
        return []

    candidate_limit = min(len(chunks), max(top_k * 12, 50))
    query_embedding = generate_query_embedding(query, dimensions=embedding_dimensions)
    vector_scores = {
        index: max(0.0, score)
        for index, score in search_index(vector_store, query_embedding, candidate_limit)
    }
    keyword_scores = _bm25_scores(query, chunks)
    candidate_indexes = _candidate_indexes(vector_scores, keyword_scores, candidate_limit)
    focus_phrase = _focus_phrase(query)
    answer_heading_indexes = _answer_heading_indexes(chunks, focus_phrase)

    candidates: list[RetrievalResult] = []
    for index in candidate_indexes:
        chunk = chunks[index]
        vector_score = vector_scores.get(index, 0.0)
        keyword_score = keyword_scores.get(index, 0.0)
        quality_score = _quality_score(chunk, query)
        if focus_phrase and index - 1 in answer_heading_indexes and focus_phrase in _compact(chunk.text):
            quality_score *= 1.4
        if quality_score <= 0.0:
            continue
        score = _combined_score(vector_score, keyword_score, quality_score)
        if score > 0:
            candidates.append(
                RetrievalResult(
                    chunk=chunk,
                    score=score,
                    vector_score=vector_score,
                    keyword_score=keyword_score,
                    quality_score=quality_score,
                )
            )
    candidates.sort(key=lambda item: item.score, reverse=True)

    packed: list[RetrievalResult] = []
    used_tokens = 0
    seen_signatures: set[str] = set()
    best_score = candidates[0].score if candidates else 0.0

    for candidate in candidates:
        if len(packed) >= top_k:
            break
        if packed and candidate.score < best_score * 0.40:
            continue
        if packed and candidate.quality_score < 0.5:
            continue
        signature = " ".join(candidate.chunk.text.lower().split()[:32])
        if signature in seen_signatures:
            continue
        token_count = candidate.chunk.token_count or count_tokens(candidate.chunk.text)
        if used_tokens + token_count > context_budget:
            continue
        packed.append(candidate)
        seen_signatures.add(signature)
        used_tokens += token_count

    return packed


def _candidate_indexes(
    vector_scores: dict[int, float],
    keyword_scores: dict[int, float],
    candidate_limit: int,
) -> list[int]:
    vector_top = sorted(vector_scores, key=vector_scores.get, reverse=True)[:candidate_limit]
    keyword_top = sorted(keyword_scores, key=keyword_scores.get, reverse=True)[:candidate_limit]
    ordered: list[int] = []
    seen: set[int] = set()
    for index in vector_top + keyword_top:
        if index not in seen:
            ordered.append(index)
            seen.add(index)
    return ordered


def _combined_score(vector_score: float, keyword_score: float, quality_score: float) -> float:
    normalized_keyword = keyword_score / (keyword_score + 1.0) if keyword_score > 0 else 0.0
    base_score = (0.45 * vector_score) + (0.55 * normalized_keyword)
    return base_score * quality_score


def _bm25_scores(query: str, chunks: list[Chunk]) -> dict[int, float]:
    query_terms = _tokenize(query)
    if not query_terms:
        return {}

    documents = [_tokenize(chunk.text) for chunk in chunks]
    doc_lengths = [len(document) for document in documents]
    avg_doc_length = sum(doc_lengths) / max(1, len(doc_lengths))
    document_frequency: Counter[str] = Counter()
    term_counts = [Counter(document) for document in documents]

    for document in documents:
        document_frequency.update(set(document))

    scores: dict[int, float] = {}
    doc_count = len(documents)
    k1 = 1.5
    b = 0.75

    for index, counts in enumerate(term_counts):
        score = 0.0
        doc_length = doc_lengths[index] or 1
        for term in query_terms:
            frequency = counts.get(term, 0)
            if frequency == 0:
                continue
            df = document_frequency.get(term, 0)
            idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
            numerator = frequency * (k1 + 1)
            denominator = frequency + k1 * (1 - b + b * doc_length / avg_doc_length)
            score += idf * numerator / denominator
        if score > 0:
            scores[index] = score
    return scores


def _quality_score(chunk: Chunk, query: str) -> float:
    text = chunk.text.strip()
    lowered = f" {text.lower()} "
    compact = _compact(text)
    score = 1.0
    focus_phrase = _focus_phrase(query)

    if _is_toc_like(text):
        score *= 0.08
    if chunk.token_count < 40:
        score *= 0.35
    if chunk.token_count < 20:
        score *= 0.15
    if _looks_like_index_or_reference(text):
        score *= 0.35
    if not any(mark in text for mark in ".:;"):
        score *= 0.75

    query_terms = set(_tokenize(query))
    text_terms = set(_tokenize(text))
    overlap = query_terms & text_terms
    if query_terms and overlap:
        score *= 1.0 + min(0.25, len(overlap) / len(query_terms) * 0.25)
    if any(pattern in lowered for pattern in _DEFINITION_PATTERNS):
        score *= 1.2
    if focus_phrase:
        if f" what is {focus_phrase}" in compact or f" what are {focus_phrase}" in compact:
            score *= 1.7
        if focus_phrase in compact and _has_definition_language(compact, focus_phrase):
            score *= 1.45
        if _is_neighboring_section(compact, focus_phrase):
            score *= 0.55
        if _is_table_or_figure_heavy(compact):
            score *= 0.65

    return score


def _is_toc_like(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 4:
        return False

    dot_lines = sum(1 for line in lines if _TOC_DOT_LINE.match(line))
    page_number_lines = sum(1 for line in lines if _PAGE_NUMBER_LINE.match(line))
    short_lines = sum(1 for line in lines if len(line) <= 24)
    heading_with_page = sum(1 for line in lines if re.search(r"\b\d{1,4}$", line))
    punctuation_light = sum(1 for line in lines if not re.search(r"[.!?;:]", line))

    return (
        dot_lines >= 3
        or page_number_lines >= 3
        or (short_lines / len(lines) > 0.65 and heading_with_page >= 2)
        or (punctuation_light / len(lines) > 0.8 and heading_with_page >= 3)
    )


def _looks_like_index_or_reference(text: str) -> bool:
    lowered = text.lower()
    if "table of contents" in lowered or "index" in lowered:
        return True
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    comma_page_refs = sum(1 for line in lines if re.search(r",\s*\d{1,4}(,\s*\d{1,4})+", line))
    return comma_page_refs >= 3


def _focus_phrase(query: str) -> str:
    terms = _tokenize(query)
    if not terms:
        return ""
    return " ".join(terms)


def _answer_heading_indexes(chunks: list[Chunk], focus_phrase: str) -> set[int]:
    if not focus_phrase:
        return set()
    return {
        index
        for index, chunk in enumerate(chunks)
        for compact_text in [_compact(chunk.text)]
        if (
            (f" what is {focus_phrase}" in compact_text
            or f" what are {focus_phrase}" in compact_text)
            and _has_definition_language(compact_text, focus_phrase)
            and not _is_toc_like(chunk.text)
        )
    }


def _compact(text: str) -> str:
    return f" {' '.join(text.lower().split())} "


def _has_definition_language(lowered_text: str, focus_phrase: str) -> bool:
    focus_pattern = re.escape(focus_phrase)
    quoted_focus_pattern = rf"[\"'“”‘’]?\s*{focus_pattern}\s*[\"'“”‘’]?"
    definition_patterns = (
        rf"{quoted_focus_pattern}\s+is\s+(a|an|the)\s+",
        rf"{quoted_focus_pattern}\s+refers\s+to\s+",
        rf"fundamentally,\s+{quoted_focus_pattern}\s+is\s+",
        rf"{quoted_focus_pattern}\s+means\s+",
        rf"{quoted_focus_pattern}\s+is\s+used\s+to\s+",
    )
    return any(re.search(pattern, lowered_text) for pattern in definition_patterns)


def _is_neighboring_section(lowered_text: str, focus_phrase: str) -> bool:
    if focus_phrase not in lowered_text:
        return False
    neighboring_markers = (
        f" where is a {focus_phrase}",
        f" where is {focus_phrase}",
        f" how does a {focus_phrase}",
        f" how does {focus_phrase}",
        f" why use a {focus_phrase}",
        f" why use {focus_phrase}",
        f" selecting a {focus_phrase}",
        f" selecting {focus_phrase}",
        f" checklist: selecting a {focus_phrase}",
        f" {focus_phrase} taxonomy ",
        f" {focus_phrase} as gateway ",
        f" {focus_phrase} as esb ",
    )
    return any(marker in lowered_text for marker in neighboring_markers)


def _is_table_or_figure_heavy(lowered_text: str) -> bool:
    table_markers = lowered_text.count(" table ")
    figure_markers = lowered_text.count(" figure ")
    page_markers = len(re.findall(r"\|\s*\d{1,4}\s*(\||$)", lowered_text))
    return table_markers + figure_markers >= 2 or page_markers >= 3


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in (match.group(0).lower() for match in _WORD_PATTERN.finditer(text))
        if token not in _STOPWORDS and len(token) > 1
    ]
