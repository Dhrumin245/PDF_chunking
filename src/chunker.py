from __future__ import annotations

import re
from dataclasses import dataclass

from .token_tracker import count_tokens, decode_tokens, encode_tokens, tail_tokens


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    token_count: int
    chunk_size: int


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int | None = None,
    encoding_name: str = "cl100k_base",
) -> list[Chunk]:
    """Split text into retrieval chunks sized between 128 and 512 tokens."""

    if chunk_size not in {128, 256, 512}:
        raise ValueError("chunk_size must be one of: 128, 256, 512")

    overlap = overlap if overlap is not None else min(48, max(12, chunk_size // 10))
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks: list[Chunk] = []
    buffer: list[str] = []

    for segment in _split_text(text):
        if count_tokens(segment, encoding_name) > chunk_size:
            _flush(chunks, buffer, chunk_size, encoding_name)
            _append_token_windows(chunks, segment, chunk_size, overlap, encoding_name)
            continue

        candidate = "\n".join(buffer + [segment]).strip()
        if buffer and count_tokens(candidate, encoding_name) > chunk_size:
            _flush(chunks, buffer, chunk_size, encoding_name)
            if chunks and overlap:
                overlap_text = tail_tokens(chunks[-1].text, overlap, encoding_name)
                if count_tokens(f"{overlap_text}\n{segment}", encoding_name) <= chunk_size:
                    buffer.append(overlap_text)

        buffer.append(segment)

    _flush(chunks, buffer, chunk_size, encoding_name)
    return chunks


def _split_text(text: str) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    segments: list[str] = []
    for paragraph in paragraphs:
        if count_tokens(paragraph) <= 160:
            segments.append(paragraph)
        else:
            sentences = [part.strip() for part in _SENTENCE_BOUNDARY.split(paragraph) if part.strip()]
            segments.extend(sentences or [paragraph])
    return segments


def _flush(
    chunks: list[Chunk],
    buffer: list[str],
    chunk_size: int,
    encoding_name: str,
) -> None:
    text = "\n".join(buffer).strip()
    buffer.clear()
    if not text:
        return
    chunks.append(
        Chunk(
            id=f"chunk-{len(chunks):04d}",
            text=text,
            token_count=count_tokens(text, encoding_name),
            chunk_size=chunk_size,
        )
    )


def _append_token_windows(
    chunks: list[Chunk],
    text: str,
    chunk_size: int,
    overlap: int,
    encoding_name: str,
) -> None:
    tokens = encode_tokens(text, encoding_name)
    step = chunk_size - overlap
    for start in range(0, len(tokens), step):
        window = tokens[start : start + chunk_size]
        if not window:
            break
        chunk_text_value = decode_tokens(window, encoding_name).strip()
        chunks.append(
            Chunk(
                id=f"chunk-{len(chunks):04d}",
                text=chunk_text_value,
                token_count=len(window),
                chunk_size=chunk_size,
            )
        )
        if start + chunk_size >= len(tokens):
            break
