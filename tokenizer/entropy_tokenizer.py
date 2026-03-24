"""
EntropyTokenizer
================
Boundary-based tokenizer grounded in the information-theoretic formulation:

    min_{b ∈ {0,1}^{N-1}}  ∑_j H(t_j | t_{<j})  +  λ|V(b)|  +  μ·L(b)

Key insight
-----------
The total byte-level entropy ∑_i h_i  is invariant to the segmentation b
(chain rule of information).  The objective therefore reduces to minimising
the *modelling cost* a finite-capacity LM pays for the chosen segmentation.

The optimal per-position greedy proxy is:

    b_i = 1   iff   h_i = -log P(x_{i+1} | x_{≤i}) > τ

This is *entropy patching* (BLT, Pagnoni et al. 2024), but derived here
from the combinatorial optimisation formulation rather than introduced as
an empirical heuristic.

The threshold τ is chosen as the p-th percentile of {h_i}, where p =
`boundary_percentile` (default 80 → 20 % of positions become boundaries
→ avg token length ≈ 5 bytes, comparable to GPT-2 BPE at vocab=50 K).

PMI vocabulary filter
---------------------
After segmentation, candidate multi-byte tokens are scored by sequential
PMI:

    PMI(t = x_1…x_n) ≈ Σ_{i=1}^{n-1} [log P(x_i|x_{i-1}) - log P(x_i)]

Tokens with PMI < pmi_min (default 0 = positive association) are removed
before the frequency-based top-K selection.  This discards tokens whose
internal bytes are not genuinely associated — e.g. accidental n-grams that
straddle two semantically unrelated words.

Differences from ClusteringTokenizer
-------------------------------------
* Uses semantic signal (byte-level surprise) instead of surface similarity
  (MinHash Jaccard ≥ 0.8) — which was shown to be a near-no-op.
* No GPU dependency: entire training is CPU-only numpy.
* Principled threshold instead of greedy top-K by raw frequency.
* PMI filter discards spurious co-occurrences.

Binary file format (llm.c-compatible)
--------------------------------------
  Header : 256 × int32 (little-endian)
    [0]  magic   = 20240328
    [1]  version = 2
    [2]  vocab_size
    [3]  eot_token_id
    [4…255] reserved zeros
  Tokens : for id 0 … vocab_size-1
    1 byte : token length L  (0 for special tokens)
    L bytes: token bytes
"""

from __future__ import annotations

import math
import struct
import time
from collections import Counter
from typing import Dict, List, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Binary format constants (must match llmc/tokenizer.h and ClusteringTokenizer)
# ─────────────────────────────────────────────────────────────────────────────
_MAGIC   = 20240328
_VERSION = 2
_HEADER_INTS = 256          # 256 × int32 = 1024 bytes


# ─────────────────────────────────────────────────────────────────────────────
#  Byte n-gram language model
# ─────────────────────────────────────────────────────────────────────────────

class ByteNgramLM:
    """
    Dense byte-level n-gram LM (order 1–3) for per-position surprise.

    Uses Laplace-smoothed counts with longest-available-context fallback:
      order 3  →  P(x_{i+1} | x_{i-1}, x_i)  (trigram, position i ≥ 1)
               +  P(x_1    | x_0)              (bigram,  position i = 0)
      order 2  →  P(x_{i+1} | x_i)
      order 1  →  P(x_{i+1})

    Memory:
      unigram : 256 × int32           =    1 KB
      bigram  : 256² × int32          =  256 KB
      trigram : 256³ × int32          =   64 MB
    """

    def __init__(self, order: int = 3) -> None:
        if order not in (1, 2, 3):
            raise ValueError(f"order must be 1, 2, or 3; got {order}")
        self.order = order
        self._trained = False

    # ── Training ──────────────────────────────────────────────────────────

    def train(self, data: bytes, verbose: bool = False) -> None:
        """Count n-gram statistics from raw byte corpus."""
        t0 = time.time()
        d  = np.frombuffer(data, dtype=np.uint8)
        N  = len(d)
        if verbose:
            print(f"  [ByteNgramLM] counting {self.order}-gram stats "
                  f"on {N/1e6:.1f} MB ...", flush=True)

        # ── Unigram ───────────────────────────────────────────────────────
        self.uni: np.ndarray = np.bincount(
            d.astype(np.int64), minlength=256
        ).astype(np.int32)
        self.uni_total: int = int(N)

        # ── Bigram ────────────────────────────────────────────────────────
        if self.order >= 2:
            CHUNK = 50_000_000
            flat_bi = np.zeros(256 * 256, dtype=np.int32)
            for s in range(0, N - 1, CHUNK):
                e   = min(s + CHUNK, N - 1)
                idx = (d[s:e].astype(np.int32) * 256
                       + d[s + 1:e + 1].astype(np.int32))
                flat_bi += np.bincount(
                    idx, minlength=256 * 256
                ).astype(np.int32)
            self.bi: np.ndarray       = flat_bi.reshape(256, 256)
            self.bi_rowsum: np.ndarray = self.bi.sum(axis=1)   # (256,)

        # ── Trigram ───────────────────────────────────────────────────────
        if self.order >= 3:
            CHUNK = 20_000_000
            flat_tri = np.zeros(256 ** 3, dtype=np.int32)
            for s in range(0, N - 2, CHUNK):
                e   = min(s + CHUNK, N - 2)
                a   = d[s:e].astype(np.int32)
                b_  = d[s + 1:e + 1].astype(np.int32)
                c   = d[s + 2:e + 2].astype(np.int32)
                idx = (a << 16) | (b_ << 8) | c
                flat_tri += np.bincount(
                    idx, minlength=256 ** 3
                ).astype(np.int32)
            self.tri: np.ndarray          = flat_tri.reshape(256, 256, 256)
            self.tri_ctx_sum: np.ndarray  = self.tri.sum(axis=2)  # (256, 256)

        self._trained = True
        if verbose:
            print(f"  [ByteNgramLM] done in {time.time() - t0:.1f}s", flush=True)

    # ── Inference ─────────────────────────────────────────────────────────

    def surprise(self, data: bytes) -> np.ndarray:
        """
        Return h[i] = -log P(x[i+1] | context) for i = 0 … N-2.
        Output dtype: float32, shape (N-1,).
        """
        assert self._trained, "call train() first"
        d = np.frombuffer(data, dtype=np.uint8)
        N = len(d)
        if N < 2:
            return np.zeros(0, dtype=np.float32)

        nxt = d[1:]

        # ── Unigram baseline (all positions) ──────────────────────────────
        p = ((self.uni[nxt].astype(np.float64) + 1.0)
             / (self.uni_total + 256.0))
        h = -np.log(p)

        # ── Bigram (all positions, overwrites unigram) ────────────────────
        if self.order >= 2:
            ctx      = d[:-1]
            rowsum   = self.bi_rowsum[ctx].astype(np.float64)
            nxt_cnt  = self.bi[ctx, nxt].astype(np.float64)
            h = -np.log((nxt_cnt + 1.0) / (rowsum + 256.0))

        # ── Trigram (positions 1…N-2, overwrites bigram for those slots) ──
        if self.order >= 3 and N >= 3:
            c2       = d[:-2]
            c1       = d[1:-1]
            nx3      = d[2:]
            ctx_sum  = self.tri_ctx_sum[c2, c1].astype(np.float64)
            nxt_cnt3 = self.tri[c2, c1, nx3].astype(np.float64)
            h[1:]    = -np.log((nxt_cnt3 + 1.0) / (ctx_sum + 256.0))

        return h.astype(np.float32)

    # ── Helpers used by EntropyTokenizer ──────────────────────────────────

    def log_unigram_probs(self) -> np.ndarray:
        """log P(x) for x ∈ [0,255], shape (256,), float64."""
        assert self._trained
        return np.log(
            (self.uni.astype(np.float64) + 1.0)
            / (self.uni_total + 256.0)
        )

    def log_bigram_cond(self, ctx: int, nxt: int) -> float:
        """log P(nxt | ctx) using bigram counts (Laplace-smoothed)."""
        assert self._trained and self.order >= 2
        c   = float(self.bi[ctx, nxt]) + 1.0
        tot = float(self.bi_rowsum[ctx]) + 256.0
        return math.log(c / tot)


# ─────────────────────────────────────────────────────────────────────────────
#  Entropy tokenizer
# ─────────────────────────────────────────────────────────────────────────────

class EntropyTokenizer:
    """
    Entropy-patching tokenizer with PMI vocabulary filtering.

    Interface is compatible with ClusteringTokenizer:
      train(data_bytes, verbose)
      encode(text | bytes)  →  list[int]
      encode_with_eot(text) →  list[int]
      decode(ids)           →  bytes
      decode_str(ids)       →  str
      save(path) / load(path)
      .eot_token            →  int
      .n_vocab              →  int
      .vocab_size           →  int
    """

    _EOT_BYTES = b"<|endoftext|>"
    _PAD_BYTES = b"<|pad|>"

    def __init__(
        self,
        vocab_size: int          = 50257,
        max_token_len: int       = 16,
        ngram_order: int         = 3,
        boundary_percentile: int = 80,
        use_pmi: bool            = True,
        pmi_min: float           = 0.0,
        min_count: int           = 10,
    ) -> None:
        """
        Args
        ----
        vocab_size           : Target vocabulary size (incl. 256 byte
                               fallbacks and EOT).
        max_token_len        : Maximum multi-byte token length in bytes.
        ngram_order          : Order for ByteNgramLM (1–3).
        boundary_percentile  : τ = percentile of h_i at which to cut.
                               Higher → fewer boundaries → longer tokens.
                               80 ≈ 5 byte avg length (GPT-2 range).
        use_pmi              : Apply sequential PMI filter on candidates.
        pmi_min              : Minimum PMI to include a token (default 0 =
                               positive association only).
        min_count            : Minimum corpus frequency to enter vocabulary.
        """
        self.vocab_size          = vocab_size
        self.max_token_len       = max_token_len
        self.ngram_order         = ngram_order
        self.boundary_percentile = boundary_percentile
        self.use_pmi             = use_pmi
        self.pmi_min             = pmi_min
        self.min_count           = min_count

        self._lm: Optional[ByteNgramLM]     = None
        self._vocab: Optional[List[bytes]]  = None   # id → bytes
        self._token2id: Optional[Dict[bytes, int]] = None  # bytes → id
        self._eot_token: Optional[int]      = None
        self._trained: bool                 = False

    # ── Public training API ───────────────────────────────────────────────

    def train(self, data: bytes | str, verbose: bool = True) -> None:
        """
        Train on raw byte corpus.

        `data` may be a bytes object or a str (UTF-8 encoded internally).
        For large corpora, pass bytes directly to avoid an extra copy.
        """
        if isinstance(data, str):
            data = data.encode("utf-8", errors="replace")

        t0 = time.time()
        N  = len(data)
        if verbose:
            print(f"[EntropyTokenizer] training on {N / 1e6:.1f} MB corpus",
                  flush=True)

        # 1. Train byte LM ─────────────────────────────────────────────────
        if verbose:
            print("[1/5] Training byte n-gram LM ...", flush=True)
        self._lm = ByteNgramLM(order=self.ngram_order)
        self._lm.train(data, verbose=verbose)

        # 2. Compute per-position surprise ─────────────────────────────────
        if verbose:
            print("[2/5] Computing byte-level surprise h_i ...", flush=True)
        t1 = time.time()
        h  = self._lm.surprise(data)
        if verbose:
            print(
                f"  h_i: mean={h.mean():.3f}  std={h.std():.3f}  "
                f"p50={np.percentile(h, 50):.3f}  "
                f"p80={np.percentile(h, 80):.3f}  "
                f"p95={np.percentile(h, 95):.3f}  "
                f"({time.time() - t1:.1f}s)",
                flush=True,
            )

        # 3. Choose threshold τ ────────────────────────────────────────────
        if verbose:
            print(f"[3/5] Setting τ at p{self.boundary_percentile} "
                  f"of surprise distribution ...", flush=True)
        tau = float(np.percentile(h, self.boundary_percentile))
        boundaries = h > tau
        frac = float(boundaries.mean())
        avg_len = 1.0 / (frac + 1e-9)
        if verbose:
            print(
                f"  τ = {tau:.4f}  →  {frac * 100:.1f}% boundaries  "
                f"(avg token ≈ {avg_len:.1f} bytes)",
                flush=True,
            )

        # 4. Segment corpus and count tokens ───────────────────────────────
        if verbose:
            print("[4/5] Segmenting corpus and counting tokens ...", flush=True)
        t2 = time.time()
        token_counts = self._count_tokens(data, boundaries)
        n_types  = len(token_counts)
        n_freq   = sum(1 for c in token_counts.values() if c >= self.min_count)
        if verbose:
            print(
                f"  {n_types:,} distinct types  |  "
                f"{n_freq:,} with freq ≥ {self.min_count}  "
                f"({time.time() - t2:.1f}s)",
                flush=True,
            )

        # 5. Build vocabulary ──────────────────────────────────────────────
        if verbose:
            print("[5/5] Building vocabulary ...", flush=True)
        self._build_vocab(token_counts)

        self._trained = True
        n_multi = sum(1 for t in self._vocab if len(t) > 1
                      and t not in (self._EOT_BYTES, self._PAD_BYTES))
        if verbose:
            print(
                f"[EntropyTokenizer] done in {time.time() - t0:.1f}s  |  "
                f"vocab_size={len(self._vocab)}  multi-byte tokens={n_multi}",
                flush=True,
            )

    # ── Encoding / Decoding ───────────────────────────────────────────────

    def encode(self, text: str | bytes) -> List[int]:
        """Greedy longest-match encoding, O(n). Falls back to byte tokens."""
        assert self._trained, "call train() first"
        if isinstance(text, str):
            data = text.encode("utf-8", errors="replace")
        else:
            data = text
        return self._encode_bytes(data)

    def encode_with_eot(self, text: str | bytes) -> List[int]:
        """Encode and prepend EOT token (document boundary convention)."""
        return [self._eot_token] + self.encode(text)

    def _encode_bytes(self, data: bytes) -> List[int]:
        ids: List[int] = []
        i   = 0
        N   = len(data)
        ml  = self.max_token_len
        while i < N:
            matched = False
            for length in range(min(ml, N - i), 1, -1):
                chunk = data[i: i + length]
                tid   = self._token2id.get(chunk)
                if tid is not None:
                    ids.append(tid)
                    i += length
                    matched = True
                    break
            if not matched:
                # Single-byte fallback: always in vocab as IDs 0-255
                ids.append(data[i])
                i += 1
        return ids

    def decode(self, ids: List[int]) -> bytes:
        """Decode token IDs to raw bytes (skips EOT / PAD tokens)."""
        assert self._trained
        parts = []
        for tid in ids:
            if 0 <= tid < len(self._vocab):
                tok = self._vocab[tid]
                if tok not in (self._EOT_BYTES, self._PAD_BYTES):
                    parts.append(tok)
        return b"".join(parts)

    def decode_str(self, ids: List[int]) -> str:
        return self.decode(ids).decode("utf-8", errors="replace")

    # ── Save / Load ───────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Save in llm.c-compatible binary format (same as ClusteringTokenizer).
        """
        assert self._trained
        header = [0] * _HEADER_INTS
        header[0] = _MAGIC
        header[1] = _VERSION
        header[2] = len(self._vocab)
        header[3] = self._eot_token

        with open(path, "wb") as f:
            f.write(struct.pack(f"{_HEADER_INTS}i", *header))
            for tok in self._vocab:
                if tok in (self._EOT_BYTES, self._PAD_BYTES):
                    f.write(struct.pack("B", 0))
                else:
                    b = tok[:255]           # truncate if somehow longer
                    f.write(struct.pack("B", len(b)))
                    f.write(b)

        print(f"[EntropyTokenizer] saved to {path}", flush=True)

    def load(self, path: str) -> None:
        """Load from binary file (inverse of save())."""
        with open(path, "rb") as f:
            hdr_bytes = f.read(_HEADER_INTS * 4)
            header    = struct.unpack(f"{_HEADER_INTS}i", hdr_bytes)

            assert header[0] == _MAGIC,   f"bad magic {header[0]}"
            assert header[1] == _VERSION, f"unsupported version {header[1]}"
            vocab_size = header[2]
            eot_token  = header[3]

            vocab: List[bytes] = []
            for i in range(vocab_size):
                length = struct.unpack("B", f.read(1))[0]
                if length == 0:
                    vocab.append(
                        self._EOT_BYTES if i == eot_token else self._PAD_BYTES
                    )
                else:
                    vocab.append(f.read(length))

        self.vocab_size  = vocab_size
        self._vocab      = vocab
        self._token2id   = {t: i for i, t in enumerate(vocab)}
        self._eot_token  = eot_token
        self._trained    = True

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def eot_token(self) -> int:
        return self._eot_token

    @property
    def n_vocab(self) -> int:
        return len(self._vocab)

    # ── Private helpers ───────────────────────────────────────────────────

    def _count_tokens(
        self, data: bytes, boundaries: np.ndarray
    ) -> Counter:
        """
        Segment `data` at every True position in `boundaries` and count
        occurrences of each resulting byte sequence.

        Tokens longer than max_token_len are split at the length limit
        rather than discarded, so no bytes are lost.
        """
        N   = len(data)
        ml  = self.max_token_len
        cnt: Counter = Counter()

        # Positions where a new token begins (0 is always a start)
        cuts = np.where(boundaries)[0] + 1   # shape: (n_cuts,)

        prev = 0
        for cut in cuts.tolist():
            _add_span(data, prev, cut, ml, cnt)
            prev = cut
        _add_span(data, prev, N, ml, cnt)

        return cnt

    def _pmi(self, tok: bytes) -> float:
        """
        Sequential PMI ≈ Σ_{i=1}^{n-1} [log P(tok[i] | tok[i-1]) - log P(tok[i])]

        Positive ↔ bytes within tok co-occur more than independent chance.
        Returns 0.0 for single-byte tokens (undefined by convention).
        """
        n = len(tok)
        if n <= 1:
            return 0.0
        assert self._lm is not None and self._lm.order >= 2
        log_uni = self._lm.log_unigram_probs()
        pmi     = 0.0
        for i in range(1, n):
            pmi += (self._lm.log_bigram_cond(tok[i - 1], tok[i])
                    - log_uni[tok[i]])
        return pmi

    def _build_vocab(self, token_counts: Counter) -> None:
        """
        Construct final vocabulary list (indexed by token ID):
          IDs   0-255            : single-byte fallbacks
          IDs 256 .. V-2         : multi-byte tokens (freq + PMI filtered)
          ID  V-1                : EOT = b"<|endoftext|>"
        """
        n_multi = self.vocab_size - 257     # slots for multi-byte tokens

        # ── Score and filter multi-byte candidates ─────────────────────────
        candidates: List[tuple] = []        # (freq, pmi, tok)
        for tok, freq in token_counts.items():
            if len(tok) <= 1:
                continue                    # handled by byte fallbacks
            if freq < self.min_count:
                continue
            if self.use_pmi and self._lm is not None and self._lm.order >= 2:
                pmi = self._pmi(tok)
                if pmi < self.pmi_min:
                    continue
            else:
                pmi = 0.0
            candidates.append((freq, pmi, tok))

        # Sort: frequency first, PMI as tie-breaker
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        selected = [tok for _, _, tok in candidates[:n_multi]]

        # ── Build ordered vocab list ──────────────────────────────────────
        vocab: List[bytes] = []
        for i in range(256):
            vocab.append(bytes([i]))        # single-byte tokens: IDs 0-255
        for tok in selected:
            vocab.append(tok)
        # Pad to vocab_size - 1 if we have fewer multi-byte tokens than slots
        while len(vocab) < self.vocab_size - 1:
            vocab.append(self._PAD_BYTES)
        vocab.append(self._EOT_BYTES)       # EOT always at ID vocab_size - 1

        self._vocab     = vocab
        self._token2id  = {t: i for i, t in enumerate(vocab)}
        self._eot_token = len(vocab) - 1


# ─────────────────────────────────────────────────────────────────────────────
#  Module-level helper (keeps _count_tokens tight)
# ─────────────────────────────────────────────────────────────────────────────

def _add_span(
    data: bytes, start: int, end: int, max_len: int, cnt: Counter
) -> None:
    """Add data[start:end] to cnt, splitting at max_len if necessary."""
    length = end - start
    if length <= 0:
        return
    if length <= max_len:
        cnt[data[start:end]] += 1
    else:
        for s in range(start, end, max_len):
            cnt[data[s: s + max_len]] += 1
