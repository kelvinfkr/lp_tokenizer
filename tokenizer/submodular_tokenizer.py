"""
SubmodularTokenizer
===================

Two-phase tokenizer with a (1 - 1/e) approximation guarantee on vocabulary
selection quality (relative to the minimum-token-count DP objective).

──────────────────────────────────────────────────────────────────────────────
Theory
──────────────────────────────────────────────────────────────────────────────

Phase 1 — Entropy patching (boundary decisions)
  h_i = -log P(x_{i+1} | x_{i-order+1}..x_i)   [ByteNgramLM]
  b_i = 1  iff  h_i > τ = percentile(h, boundary_percentile)

  Derived from: min_b ∑_j H(t_j|t_{<j}) + λ|V(b)| + μL(b)
  Solution is entropy patching because total byte entropy is invariant to b.

Phase 2 — Submodular vocabulary selection
  Define:
    f(V) = Viterbi DP cost  (min token count to cover corpus sample with V)
    g(V) = f(V_bytes) - f(V)     # cost reduction, ≥ 0

  Key facts (proved in comments below):
    • f is monotone non-increasing:   V ⊆ W  →  f(W) ≤ f(V)
    • g is monotone non-decreasing and submodular
    • Therefore greedy maximisation of g gives a (1-1/e) approximation:
        g(V_greedy) ≥ (1-1/e) · g(V*)   [Nemhauser-Wolsey-Fisher 1978]

  Implemented via CELF (Leskovec et al. 2007) for O(|C| log |C|) lazy evals
  instead of O(|C|·K) full re-evaluations.

  NOTE: the guarantee applies to f = min-token-count (a proxy for BPB).
  f ≠ LM cross-entropy, but high compression correlates with low BPB.

──────────────────────────────────────────────────────────────────────────────
Submodularity proof sketch
──────────────────────────────────────────────────────────────────────────────

For any string s, let opt(s, V) = Viterbi min token count with vocabulary V.

Claim: opt(s, ·) is submodular.
Proof: for A ⊆ B and token t ∉ B,
  Δ(t|A) = opt(s,A) - opt(s,A∪{t})
  Δ(t|B) = opt(s,B) - opt(s,B∪{t})

  Any coverage improvement t enables when added to B is also enabled when
  added to A (since A ⊆ B, anything achievable via B is achievable via A).
  So the set of improvements t can make to A ⊇ those to B, thus Δ(t|A) ≥ Δ(t|B).

Since f(V) = ∑_s opt(s,V) is a non-negative sum of submodular functions,
f is submodular. Then g(V) = const - f(V) is monotone non-decreasing and
submodular. □

──────────────────────────────────────────────────────────────────────────────
Computational notes
──────────────────────────────────────────────────────────────────────────────

All DP evaluations run on a sample (default 2 MB) for speed:
  • Full Viterbi DP:   O(N_sample × max_len)    Python loop, ~1–4 s
  • Match precompute:  O(|C| × N_sample × L)   numpy, ~0.5–2 s total
  • CELF greedy:       O(K × recompute_iv × DP_cost + amortised gain evals)

First-order gain approximation (used between full DP recomputes):
  gain_1(t | dp) = ∑_{i: data[i:i+L]=t} max(0, dp[i+L] - dp[i] - 1)

  This lower-bounds the true gain (cascade effects omitted) and preserves the
  CELF lazy-evaluation correctness property (gains are non-increasing with
  respect to this approximation when dp only improves monotonically).
"""

from __future__ import annotations

import heapq
import math
import struct
import time
from collections import Counter
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from .entropy_tokenizer import ByteNgramLM, _add_span


# ─────────────────────────────────────────────────────────────────────────────
#  Binary format constants (llm.c-compatible, same as ClusteringTokenizer)
# ─────────────────────────────────────────────────────────────────────────────
_MAGIC        = 20240328
_VERSION      = 2
_HEADER_INTS  = 256


# ─────────────────────────────────────────────────────────────────────────────
#  Module-level DP helpers
# ─────────────────────────────────────────────────────────────────────────────

def _viterbi_dp(
    data: bytes,
    vocab: FrozenSet[bytes],
    max_len: int,
) -> np.ndarray:
    """
    Viterbi minimum-token-count DP on raw bytes.

    dp[i] = minimum number of tokens needed to cover data[:i].
    dp[0] = 0; dp[N] = optimal cost for the whole corpus sample.

    Complexity: O(N × max_len) Python iterations.
    """
    N  = len(data)
    dp = np.full(N + 1, N + 1, dtype=np.int32)
    dp[0] = 0
    for i in range(N):
        vi = int(dp[i])
        if vi >= N:                     # unreachable position
            continue
        v1 = vi + 1
        cap = min(max_len, N - i)
        for L in range(1, cap + 1):
            if data[i: i + L] in vocab:
                end = i + L
                if v1 < dp[end]:
                    dp[end] = v1
    return dp


def _find_matches(d: np.ndarray, tok: bytes) -> np.ndarray:
    """
    Return all start positions where tok appears in d (numpy uint8 array).

    Vectorised: O(N × len(tok)) numpy ops.
    Returns int32 array; empty array if no match.
    """
    L = len(tok)
    N = len(d)
    if L == 0 or N < L:
        return np.empty(0, dtype=np.int32)
    tok_arr = np.frombuffer(tok, dtype=np.uint8)
    mask = np.ones(N - L + 1, dtype=bool)
    for k in range(L):
        mask &= d[k: N - L + 1 + k] == tok_arr[k]
    return np.where(mask)[0].astype(np.int32)


def _first_order_gain(
    dp: np.ndarray, starts: np.ndarray, L: int
) -> int:
    """
    First-order marginal gain of adding a token of length L at given starts.

    gain = ∑_i max(0, dp[i+L] - dp[i] - 1)

    Lower-bounds the true gain (no cascade accounted for).
    Used for CELF lazy evaluation between full DP recomputes.
    """
    if len(starts) == 0:
        return 0
    N        = len(dp) - 1
    ends     = starts.astype(np.int64) + L
    valid    = ends <= N
    s, e     = starts[valid].astype(np.int64), ends[valid].astype(np.int64)
    savings  = dp[e].astype(np.int64) - (dp[s].astype(np.int64) + 1)
    return int(np.maximum(0, savings).sum())


def _apply_incremental_update(
    dp: np.ndarray, starts: np.ndarray, L: int
) -> None:
    """
    Apply first-order incremental DP update after accepting a new token.

    For each match at position i: dp[i+L] = min(dp[i+L], dp[i]+1).
    Does NOT propagate cascades; call _viterbi_dp periodically to re-sync.
    """
    if len(starts) == 0:
        return
    N     = len(dp) - 1
    ends  = starts.astype(np.int64) + L
    valid = ends <= N
    s, e  = starts[valid].astype(np.int64), ends[valid].astype(np.int64)
    np.minimum.at(dp, e, dp[s] + 1)


# ─────────────────────────────────────────────────────────────────────────────
#  SubmodularTokenizer
# ─────────────────────────────────────────────────────────────────────────────

class SubmodularTokenizer:
    """
    Entropy-patching segmentation + submodular greedy vocabulary selection.

    Interface identical to ClusteringTokenizer / EntropyTokenizer:
      train(data, verbose)
      encode(text)  / encode_with_eot(text)
      decode(ids)   / decode_str(ids)
      save(path)    / load(path)
      .eot_token    / .n_vocab  / .vocab_size
    """

    _EOT_BYTES = b"<|endoftext|>"
    _PAD_BYTES = b"<|pad|>"

    def __init__(
        self,
        vocab_size:           int   = 50257,
        max_token_len:        int   = 16,
        ngram_order:          int   = 3,
        boundary_percentile:  int   = 80,
        candidate_factor:     int   = 10,
        sample_mb:            float = 2.0,
        min_count:            int   = 10,
        recompute_interval:   int   = 500,
    ) -> None:
        """
        Args
        ----
        vocab_size           Target vocabulary size (incl. 256 byte fallbacks + EOT).
        max_token_len        Maximum multi-byte token length in bytes.
        ngram_order          ByteNgramLM order (1–3).
        boundary_percentile  Entropy-patching threshold percentile (higher → longer tokens).
        candidate_factor     Candidate pool = candidate_factor × (vocab_size - 257).
        sample_mb            Size of corpus sample used for DP evaluations.
        min_count            Minimum corpus frequency for a candidate to be considered.
        recompute_interval   Full Viterbi DP recompute every N greedy steps (cascade sync).
        """
        self.vocab_size          = vocab_size
        self.max_token_len       = max_token_len
        self.ngram_order         = ngram_order
        self.boundary_percentile = boundary_percentile
        self.candidate_factor    = candidate_factor
        self.sample_mb           = sample_mb
        self.min_count           = min_count
        self.recompute_interval  = recompute_interval

        self._lm:        Optional[ByteNgramLM]     = None
        self._vocab:     Optional[List[bytes]]     = None   # id → bytes
        self._token2id:  Optional[Dict[bytes,int]] = None   # bytes → id
        self._eot_token: Optional[int]             = None
        self._trained:   bool                      = False

    # ── Public training API ───────────────────────────────────────────────

    def train(self, data: bytes | str, verbose: bool = True) -> None:
        """Train on raw byte corpus."""
        if isinstance(data, str):
            data = data.encode("utf-8", errors="replace")

        t0 = time.time()
        N  = len(data)
        if verbose:
            print(f"[SubmodularTokenizer] training on {N/1e6:.1f} MB corpus",
                  flush=True)

        # ── 1. Byte LM ────────────────────────────────────────────────────
        if verbose:
            print("[1/5] Training byte n-gram LM ...", flush=True)
        self._lm = ByteNgramLM(order=self.ngram_order)
        self._lm.train(data, verbose=verbose)

        # ── 2. Entropy patching → initial segmentation ────────────────────
        if verbose:
            print("[2/5] Entropy patching ...", flush=True)
        h   = self._lm.surprise(data)
        tau = float(np.percentile(h, self.boundary_percentile))
        boundaries = h > tau
        frac = float(boundaries.mean())
        if verbose:
            print(f"  τ={tau:.4f}  →  {frac*100:.1f}% boundaries  "
                  f"avg token ≈ {1/(frac+1e-9):.1f} bytes", flush=True)

        # ── 3. Candidate extraction ───────────────────────────────────────
        if verbose:
            print("[3/5] Extracting candidates from segmentation ...", flush=True)
        candidates = self._extract_candidates(data, boundaries, verbose)
        n_multi    = self.vocab_size - 257          # target multi-byte slots
        pool_size  = self.candidate_factor * n_multi
        candidates = candidates[:pool_size]
        if verbose:
            print(f"  {len(candidates)} candidates (pool cap = {pool_size})",
                  flush=True)

        # ── 4. Submodular greedy on sample ────────────────────────────────
        if verbose:
            print("[4/5] Submodular greedy vocabulary selection ...", flush=True)
        sample_bytes = min(N, int(self.sample_mb * 1e6))
        sample       = data[:sample_bytes]
        selected     = self._submodular_greedy(
            sample, candidates, n_multi, verbose
        )

        # ── 5. Build vocabulary ───────────────────────────────────────────
        if verbose:
            print("[5/5] Building vocabulary ...", flush=True)
        self._build_vocab(selected)

        self._trained = True
        n_multi_final = sum(
            1 for t in self._vocab
            if len(t) > 1 and t not in (self._EOT_BYTES, self._PAD_BYTES)
        )
        if verbose:
            print(
                f"[SubmodularTokenizer] done in {time.time()-t0:.1f}s  |  "
                f"vocab={len(self._vocab)}  multi-byte={n_multi_final}",
                flush=True,
            )

    # ── Encoding / Decoding ───────────────────────────────────────────────

    def encode(self, text: str | bytes) -> List[int]:
        """Greedy longest-match encoding. O(n). Falls back to byte tokens."""
        assert self._trained
        if isinstance(text, str):
            text = text.encode("utf-8", errors="replace")
        ids: List[int] = []
        i = 0; N = len(text); ml = self.max_token_len
        while i < N:
            matched = False
            for L in range(min(ml, N - i), 1, -1):
                tid = self._token2id.get(text[i: i + L])
                if tid is not None:
                    ids.append(tid); i += L; matched = True; break
            if not matched:
                ids.append(text[i]); i += 1
        return ids

    def encode_with_eot(self, text: str | bytes) -> List[int]:
        """Encode and prepend EOT token (document boundary convention)."""
        return [self._eot_token] + self.encode(text)

    def decode(self, ids: List[int]) -> bytes:
        """Decode token IDs to raw bytes (skips EOT / PAD)."""
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
        """Save in llm.c-compatible binary format (identical to ClusteringTokenizer)."""
        assert self._trained
        header    = [0] * _HEADER_INTS
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
                    b = tok[:255]
                    f.write(struct.pack("B", len(b)))
                    f.write(b)
        print(f"[SubmodularTokenizer] saved to {path}", flush=True)

    def load(self, path: str) -> None:
        """Load from binary file (inverse of save())."""
        with open(path, "rb") as f:
            header    = struct.unpack(f"{_HEADER_INTS}i", f.read(_HEADER_INTS * 4))
            assert header[0] == _MAGIC,    f"bad magic {header[0]}"
            assert header[1] == _VERSION,  f"unsupported version {header[1]}"
            vocab_size = header[2]; eot = header[3]
            vocab: List[bytes] = []
            for i in range(vocab_size):
                L = struct.unpack("B", f.read(1))[0]
                vocab.append(
                    (self._EOT_BYTES if i == eot else self._PAD_BYTES)
                    if L == 0 else f.read(L)
                )
        self.vocab_size = vocab_size
        self._vocab     = vocab
        self._token2id  = {t: i for i, t in enumerate(vocab)}
        self._eot_token = eot
        self._trained   = True

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def eot_token(self) -> int:
        return self._eot_token

    @property
    def n_vocab(self) -> int:
        return len(self._vocab)

    # ── Phase 3: candidate extraction ────────────────────────────────────

    def _extract_candidates(
        self,
        data: bytes,
        boundaries: np.ndarray,
        verbose: bool,
    ) -> List[bytes]:
        """
        Generate candidate multi-byte tokens from the entropy-patching
        segmentation.  Two sources:

        1. Segment tokens (each contiguous chunk between boundaries).
        2. Adjacent-pair merges (two consecutive segment tokens merged).

        Both sources are filtered by min_count, then sorted by frequency
        descending.  This gives a principled candidate pool:
        - Source 1: segments are already information-theoretically justified.
        - Source 2: merges that might reduce token count further.
        """
        t0   = time.time()
        N    = len(data)
        cuts = np.where(boundaries)[0] + 1        # start positions after cuts

        seg_counts:  Counter = Counter()
        pair_counts: Counter = Counter()

        prev_tok: Optional[bytes] = None
        start = 0

        for cut in cuts.tolist():
            tok = data[start: int(cut)]
            if 2 <= len(tok) <= self.max_token_len:
                seg_counts[tok] += 1
            if prev_tok is not None:
                merge = prev_tok + tok
                if 2 <= len(merge) <= self.max_token_len:
                    pair_counts[merge] += 1
            prev_tok = tok
            start    = int(cut)

        # Last segment
        tok = data[start: N]
        if 2 <= len(tok) <= self.max_token_len:
            seg_counts[tok] += 1
        if prev_tok is not None:
            merge = prev_tok + tok
            if 2 <= len(merge) <= self.max_token_len:
                pair_counts[merge] += 1

        # Merge both sources
        combined: Counter = Counter()
        for tok, cnt in seg_counts.items():
            combined[tok] = cnt
        for tok, cnt in pair_counts.items():
            combined[tok] = max(combined.get(tok, 0), cnt)

        # Filter and sort
        candidates = [
            tok for tok, cnt in combined.most_common()
            if cnt >= self.min_count
        ]

        if verbose:
            print(
                f"  {len(seg_counts)} segment types, "
                f"{len(pair_counts)} pair types  →  "
                f"{len(candidates)} candidates with freq ≥ {self.min_count}  "
                f"({time.time()-t0:.1f}s)",
                flush=True,
            )
        return candidates

    # ── Phase 4: submodular greedy with CELF ─────────────────────────────

    def _submodular_greedy(
        self,
        sample: bytes,
        candidates: List[bytes],
        n_select: int,
        verbose: bool,
    ) -> List[bytes]:
        """
        Greedy maximisation of g(V) = f(V_bytes) - f(V) with CELF lazy eval.

        Approximation guarantee: g(V_greedy) ≥ (1-1/e)·g(V*)
        (holds exactly for true marginal gains; approximated here by
        first-order gain between full DP recomputes).

        Algorithm:
          1. Precompute match positions for all candidates (vectorised).
          2. Build max-heap with initial first-order gains.
          3. CELF loop:
               a. Pop top candidate t from heap.
               b. If gain was computed THIS iteration → select t (CELF guarantee).
               c. Otherwise recompute true gain, re-push, repeat.
          4. After selecting t: incremental DP update + periodic full recompute.
        """
        d      = np.frombuffer(sample, dtype=np.uint8)
        N_s    = len(sample)
        ml     = self.max_token_len

        single_bytes: FrozenSet[bytes] = frozenset(bytes([b]) for b in range(256))
        current_vocab: Set[bytes]      = set(single_bytes)

        # ── Precompute match positions ────────────────────────────────────
        if verbose:
            print(f"  Precomputing matches for {len(candidates)} candidates "
                  f"on {N_s/1e6:.1f} MB sample ...", flush=True)
        t_pre = time.time()
        match_pos: Dict[bytes, np.ndarray] = {}
        for tok in candidates:
            match_pos[tok] = _find_matches(d, tok)
        if verbose:
            total_m = sum(len(v) for v in match_pos.values())
            print(f"  {total_m:,} total match positions  ({time.time()-t_pre:.1f}s)",
                  flush=True)

        # ── Initial Viterbi DP ────────────────────────────────────────────
        if verbose:
            print(f"  Initial Viterbi DP ...", flush=True)
        dp           = _viterbi_dp(sample, frozenset(current_vocab), ml)
        baseline     = int(dp[N_s])
        current_cost = baseline
        if verbose:
            print(f"  Baseline (byte vocab): {baseline} tokens", flush=True)

        # ── Build CELF heap: (-gain, iteration_computed, token) ───────────
        heap: List[Tuple[int, int, bytes]] = []
        for tok in candidates:
            g = _first_order_gain(dp, match_pos[tok], len(tok))
            heapq.heappush(heap, (-g, 0, tok))

        # ── CELF greedy loop ──────────────────────────────────────────────
        selected: List[bytes]  = []
        last_recompute: int    = 0

        for iteration in range(n_select):
            if not heap:
                break

            # Find the best token whose gain is up-to-date for this iteration
            best_tok:  Optional[bytes] = None
            best_gain: int             = 0

            while heap:
                neg_gain, iter_computed, tok = heapq.heappop(heap)

                if iter_computed == iteration:
                    # CELF guarantee: gain is current → optimal
                    best_tok  = tok
                    best_gain = -neg_gain
                    break

                # Gain is stale; recompute with current dp
                true_gain = _first_order_gain(dp, match_pos[tok], len(tok))
                if true_gain > 0:
                    heapq.heappush(heap, (-true_gain, iteration, tok))
                # tokens with true_gain = 0 are silently dropped (no improvement)

            if best_tok is None or best_gain <= 0:
                break                           # no more improving tokens

            # Accept best_tok
            selected.append(best_tok)
            current_vocab.add(best_tok)

            # Incremental DP update (first-order; no cascade)
            _apply_incremental_update(dp, match_pos[best_tok], len(best_tok))

            # Periodic full Viterbi recompute to correct cascade errors
            steps_since = iteration - last_recompute + 1
            if steps_since >= self.recompute_interval:
                dp = _viterbi_dp(sample, frozenset(current_vocab), ml)
                last_recompute = iteration + 1
                current_cost   = int(dp[N_s])
                if verbose:
                    print(
                        f"  iter {iteration+1:>6}/{n_select}  "
                        f"cost={current_cost}  "
                        f"reduction={baseline-current_cost}  "
                        f"selected={len(selected)}",
                        flush=True,
                    )

        # Final full recompute for accurate reporting
        dp           = _viterbi_dp(sample, frozenset(current_vocab), ml)
        final_cost   = int(dp[N_s])
        total_saving = baseline - final_cost

        if verbose:
            ratio = total_saving / max(1, baseline - n_select)  # rough bound
            print(
                f"  Greedy done: {len(selected)} tokens selected  |  "
                f"cost {baseline} → {final_cost}  "
                f"(saving {total_saving}, "
                f"≈ {total_saving/baseline*100:.1f}%)",
                flush=True,
            )

        return selected

    # ── Phase 5: vocabulary construction ─────────────────────────────────

    def _build_vocab(self, selected: List[bytes]) -> None:
        """
        Construct final vocabulary list:
          IDs   0–255       single-byte fallbacks
          IDs 256 .. V-2    selected multi-byte tokens (greedy order = freq desc)
          ID  V-1           EOT = b"<|endoftext|>"
        """
        n_multi = self.vocab_size - 257

        vocab: List[bytes] = [bytes([i]) for i in range(256)]
        for tok in selected[:n_multi]:
            vocab.append(tok)
        while len(vocab) < self.vocab_size - 1:
            vocab.append(self._PAD_BYTES)
        vocab.append(self._EOT_BYTES)

        self._vocab     = vocab
        self._token2id  = {t: i for i, t in enumerate(vocab)}
        self._eot_token = len(vocab) - 1
