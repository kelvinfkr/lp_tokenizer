"""
BPE baseline tokenizer wrapper (tiktoken gpt2).

Wraps tiktoken's GPT-2 encoding to match the TokenizerLike protocol
defined in tokenizer_utils.py, so it can be used interchangeably with
ClusteringTokenizer in data preparation and evaluation scripts.
"""

from __future__ import annotations


class BPETokenizer:
    """
    Thin wrapper around tiktoken GPT-2 encoding.

    Matches the TokenizerLike protocol:
      encode(text) → list[int]
      decode(ids)  → bytes
      n_vocab      → int
      eot_token    → int
    """

    def __init__(self):
        import tiktoken
        self._enc = tiktoken.get_encoding("gpt2")

    def encode(self, text: str | bytes) -> list[int]:
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        return self._enc.encode(text, allowed_special={"<|endoftext|>"})

    def encode_with_eot(self, text: str | bytes) -> list[int]:
        """Prepend EOT token (for document boundaries, matching llm.c convention)."""
        return [self.eot_token] + self.encode(text)

    def decode(self, ids: list[int]) -> bytes:
        return self._enc.decode_bytes(ids)

    def decode_str(self, ids: list[int]) -> str:
        return self._enc.decode(ids)

    @property
    def n_vocab(self) -> int:
        return self._enc.n_vocab   # 50257

    @property
    def eot_token(self) -> int:
        return self._enc.eot_token  # 50256


if __name__ == "__main__":
    tok = BPETokenizer()
    text = "Hello, world! This is the BPE baseline."
    ids = tok.encode(text)
    decoded = tok.decode_str(ids)
    print(f"encode: {ids}")
    print(f"decode: {decoded!r}")
    assert decoded == text
    print(f"✓ BPETokenizer OK, vocab_size={tok.n_vocab}, eot={tok.eot_token}")
