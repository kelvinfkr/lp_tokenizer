"""
Shared utilities for tokenizer experiments.

- write_datafile()     : write tokenized data to llm.c binary format
- write_tokenizer_bin(): write tokenizer vocab to llm.c binary format
- TokenizerMetrics     : compression ratio, fertility, encoding speed
"""

from __future__ import annotations

import struct
import time
import numpy as np
from pathlib import Path
from typing import Protocol, runtime_checkable


# ──────────────────────────────────────────────
#  llm.c binary data file format
# ──────────────────────────────────────────────
# Header: 256 × int32 (1024 bytes)
#   [0] = 20240520  (magic, GPT-2 style)
#   [1] = 1         (version)
#   [2] = num_tokens
# Body: token stream as uint16 (if vocab_size <= 65535) or uint32

MAGIC_GPT2   = 20240520
VERSION      = 1
HEADER_SIZE  = 256  # int32s


def write_datafile(tokens: list[int] | np.ndarray, path: str | Path,
                   vocab_size: int = 50257) -> None:
    """
    Write a token array to llm.c's binary data format.

    Chooses uint16 if vocab_size <= 65535, otherwise uint32
    (matches the convention in llm.c/dev/data/data_common.py).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tokens_arr = np.asarray(tokens, dtype=np.uint16 if vocab_size <= 65535
                            else np.uint32)
    num_tokens = len(tokens_arr)

    header = np.zeros(HEADER_SIZE, dtype=np.int32)
    header[0] = MAGIC_GPT2
    header[1] = VERSION
    header[2] = num_tokens

    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens_arr.tobytes())

    print(f"[write_datafile] {path}  tokens={num_tokens:,}  "
          f"dtype={'uint16' if vocab_size <= 65535 else 'uint32'}")


def read_datafile(path: str | Path) -> tuple[np.ndarray, dict]:
    """Read a llm.c binary data file. Returns (tokens, header_info)."""
    path = Path(path)
    with open(path, "rb") as f:
        header_raw = f.read(HEADER_SIZE * 4)
        header = np.frombuffer(header_raw, dtype=np.int32)
        assert header[0] == MAGIC_GPT2, f"bad magic: {header[0]}"
        num_tokens = header[2]
        # Try uint16 first; if any value > 65535 reread as uint32
        body = f.read()

    tokens = np.frombuffer(body, dtype=np.uint16)
    if len(tokens) != num_tokens:
        tokens = np.frombuffer(body, dtype=np.uint32)

    info = {"magic": int(header[0]), "version": int(header[1]),
            "num_tokens": int(header[2])}
    return tokens, info


# ──────────────────────────────────────────────
#  Tokenizer protocol (for duck-typing)
# ──────────────────────────────────────────────

@runtime_checkable
class TokenizerLike(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> bytes: ...
    @property
    def n_vocab(self) -> int: ...
    @property
    def eot_token(self) -> int: ...


# ──────────────────────────────────────────────
#  Tokenizer evaluation metrics
# ──────────────────────────────────────────────

class TokenizerMetrics:
    """
    Compute standard tokenizer evaluation metrics.

    Metrics:
      - compression_ratio : bytes / token  (higher = better compression)
      - fertility         : tokens / word  (lower = better)
      - encode_speed_mbs  : MB/s encoding throughput
      - vocab_coverage    : fraction of test text covered by non-byte tokens
    """

    @staticmethod
    def compression_ratio(text: str, tokenizer: TokenizerLike) -> float:
        """Average bytes per token on the given text."""
        data = text.encode("utf-8", errors="replace")
        ids  = tokenizer.encode(text)
        return len(data) / max(len(ids), 1)

    @staticmethod
    def fertility(text: str, tokenizer: TokenizerLike) -> float:
        """Average tokens per whitespace-split word."""
        import re
        words = re.findall(r"\S+", text)
        if not words:
            return 0.0
        total_tokens = sum(len(tokenizer.encode(w)) for w in words)
        return total_tokens / len(words)

    @staticmethod
    def encode_speed(text: str, tokenizer: TokenizerLike,
                     repeats: int = 3) -> float:
        """Return encoding throughput in MB/s."""
        data_bytes = len(text.encode("utf-8", errors="replace"))
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            tokenizer.encode(text)
            times.append(time.perf_counter() - t0)
        best_time = min(times)
        return (data_bytes / 1e6) / best_time  # MB/s

    @staticmethod
    def vocab_coverage(text: str, tokenizer: TokenizerLike) -> float:
        """
        Fraction of encoded tokens that are multi-byte (non-fallback).
        A higher value means the vocabulary is being used effectively.
        """
        ids = tokenizer.encode(text)
        if not ids:
            return 0.0
        # Assume tokens 0-255 are single-byte fallbacks
        multi = sum(1 for i in ids if i > 255)
        return multi / len(ids)

    @classmethod
    def evaluate(cls, text: str, tokenizer: TokenizerLike,
                 name: str = "tokenizer") -> dict:
        """Run all metrics and return a summary dict."""
        cr   = cls.compression_ratio(text, tokenizer)
        fert = cls.fertility(text, tokenizer)
        spd  = cls.encode_speed(text, tokenizer)
        cov  = cls.vocab_coverage(text, tokenizer)
        n_tokens = len(tokenizer.encode(text))

        result = {
            "name": name,
            "vocab_size": tokenizer.n_vocab,
            "n_tokens": n_tokens,
            "compression_ratio_bytes_per_token": round(cr, 3),
            "fertility_tokens_per_word": round(fert, 3),
            "encode_speed_mbs": round(spd, 2),
            "vocab_coverage": round(cov, 4),
        }
        print(f"\n── {name} ──")
        for k, v in result.items():
            if k != "name":
                print(f"  {k}: {v}")
        return result


# ──────────────────────────────────────────────
#  Bits-per-byte (BPB) computation
# ──────────────────────────────────────────────

def bits_per_byte(
    log_probs: np.ndarray,   # shape (N,), per-token log2 probabilities
    token_byte_lengths: np.ndarray,  # shape (N,), bytes covered by each token
) -> float:
    """
    Compute bits-per-byte (BPB), the canonical cross-tokenizer metric.

    BPB = -sum(log2_probs) / total_bytes
    Lower is better. Equivalent to per-byte perplexity in log2 space.

    This normalises out tokenizer compression ratio so results from
    different tokenizers are directly comparable.
    """
    total_nats  = -np.sum(log_probs)            # negative log-likelihood
    total_bytes = np.sum(token_byte_lengths)
    return float(total_nats / total_bytes)


if __name__ == "__main__":
    # Smoke test write/read roundtrip
    tokens = list(range(1000))
    write_datafile(tokens, "/tmp/test_tokens.bin", vocab_size=50257)
    arr, info = read_datafile("/tmp/test_tokens.bin")
    assert list(arr) == tokens, "roundtrip mismatch"
    print("✓ write_datafile / read_datafile roundtrip OK")
    print(f"  info: {info}")
