from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache


_FALLBACK_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@lru_cache(maxsize=4)
def _get_encoding(encoding_name: str):
    try:
        import tiktoken

        return tiktoken.get_encoding(encoding_name)
    except Exception:
        return None


def encode_tokens(text: str, encoding_name: str = "cl100k_base") -> list[int] | list[str]:
    encoding = _get_encoding(encoding_name)
    if encoding is not None:
        return encoding.encode(text)
    return _FALLBACK_TOKEN_PATTERN.findall(text)


def decode_tokens(tokens: list[int] | list[str], encoding_name: str = "cl100k_base") -> str:
    encoding = _get_encoding(encoding_name)
    if encoding is not None and tokens and isinstance(tokens[0], int):
        return encoding.decode(tokens)

    output: list[str] = []
    for token in tokens:
        value = str(token)
        if not output:
            output.append(value)
        elif re.match(r"[^\w\s]", value):
            output[-1] += value
        else:
            output.append(" " + value)
    return "".join(output)


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    return len(encode_tokens(text, encoding_name))


def truncate_tokens(text: str, max_tokens: int, encoding_name: str = "cl100k_base") -> str:
    if max_tokens <= 0:
        return ""
    tokens = encode_tokens(text, encoding_name)
    if len(tokens) <= max_tokens:
        return text
    return decode_tokens(tokens[:max_tokens], encoding_name).strip()


def tail_tokens(text: str, max_tokens: int, encoding_name: str = "cl100k_base") -> str:
    if max_tokens <= 0:
        return ""
    tokens = encode_tokens(text, encoding_name)
    if len(tokens) <= max_tokens:
        return text
    return decode_tokens(tokens[-max_tokens:], encoding_name).strip()


def tokenizer_is_exact(encoding_name: str = "cl100k_base") -> bool:
    return _get_encoding(encoding_name) is not None


@dataclass
class TokenTracker:
    encoding_name: str = "cl100k_base"
    usage: dict[str, int] = field(default_factory=dict)

    def add(self, label: str, text: str) -> int:
        tokens = count_tokens(text, self.encoding_name)
        self.usage[label] = self.usage.get(label, 0) + tokens
        return tokens

    def set(self, label: str, tokens: int) -> None:
        self.usage[label] = tokens

    def report(self, answer_budget: int = 200) -> dict[str, int | str | bool]:
        prompt_tokens = self.usage.get("prompt_tokens", 0)
        return {
            "tokenizer": self.encoding_name,
            "exact_tokenizer": tokenizer_is_exact(self.encoding_name),
            **self.usage,
            "answer_budget_tokens": answer_budget,
            "estimated_total_tokens": prompt_tokens + answer_budget,
        }
