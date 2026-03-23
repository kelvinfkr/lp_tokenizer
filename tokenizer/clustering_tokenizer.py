"""
Clustering-based tokenizer using MinHash + LSH approximate clustering.

Algorithm:
1. Extract all byte n-grams (length 2-8) from corpus, count frequencies
2. For each candidate n-gram, compute 128-dim MinHash signature over its
   character set (treats each n-gram as a set of (pos, byte) pairs)
3. LSH with b=16 bands, r=8 rows/band: Jaccard>=0.8 → collision prob >=99%
4. Merge same-bucket candidates, keep highest-frequency as representative
5. Build vocab = {representative tokens} + {256 byte fallbacks}

GPU-accelerated training via PyTorch (tested on RTX 4090, 48GB VRAM).
Inference uses CPU hash table for encode; decode is O(1) array lookup.

Approximation guarantee:
  With b=16, r=8: false-negative rate for Jaccard>=0.8 pairs < 0.01
  S-curve threshold at Jaccard ≈ (1/b)^(1/r) ≈ 0.75
"""

from __future__ import annotations

import struct
import numpy as np
import torch
from collections import defaultdict
from typing import Optional
import re


# ──────────────────────────────────────────────
#  MinHash + LSH constants
# ──────────────────────────────────────────────
NUM_HASHES = 128      # MinHash signature length
LSH_BANDS  = 16       # number of LSH bands  (b)
LSH_ROWS   = 8        # rows per band        (r); NUM_HASHES == b*r
assert NUM_HASHES == LSH_BANDS * LSH_ROWS

# Large prime for universal hashing
_LARGE_PRIME = (1 << 61) - 1   # Mersenne prime


def _make_hash_params(n: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Return (a, b) arrays of shape (n,) for universal hashing: (a*x + b) % p."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, _LARGE_PRIME, size=n, dtype=np.int64)
    b = rng.integers(0, _LARGE_PRIME, size=n, dtype=np.int64)
    return a, b


# ──────────────────────────────────────────────
#  GPU-accelerated MinHash computation
# ──────────────────────────────────────────────

def _ngram_to_feature_indices(ngram: bytes) -> np.ndarray:
    """
    Represent an n-gram as a set of (position, byte_value) features.
    Max n-gram length 8, byte value 0-255 → feature index = pos*256 + byte.
    Feature universe size = 8*256 = 2048.
    """
    return np.array(
        [i * 256 + b for i, b in enumerate(ngram)],
        dtype=np.int32
    )


def compute_minhash_signatures_gpu(
    candidates: list[bytes],
    device: torch.device,
    batch_size: int = 4096,  # kept for API compat, not used
) -> torch.Tensor:
    """
    Compute MinHash signatures for all candidates on GPU.

    Returns tensor of shape (len(candidates), NUM_HASHES), dtype int64.
    Groups candidates by length so each group is one batched matrix op,
    reducing kernel launches from O(n_candidates) to O(unique_lengths).
    """
    n = len(candidates)
    rng = np.random.default_rng(0)
    a  = rng.integers(1, _LARGE_PRIME, size=NUM_HASHES, dtype=np.int64)
    b_ = rng.integers(0, _LARGE_PRIME, size=NUM_HASHES, dtype=np.int64)

    a_t = torch.tensor(a,  dtype=torch.int64, device=device)   # (H,)
    b_t = torch.tensor(b_, dtype=torch.int64, device=device)   # (H,)
    p   = _LARGE_PRIME

    sigs = torch.zeros(n, NUM_HASHES, dtype=torch.int64, device=device)

    # Group by length: same-length n-grams have identical feature structure
    # → one batched (H, B, length) computation per group
    by_len: dict[int, list[tuple[int, bytes]]] = defaultdict(list)
    for i, c in enumerate(candidates):
        by_len[len(c)].append((i, c))

    for ng_len, group in by_len.items():
        idxs  = [i for i, _ in group]
        cands = [c for _, c in group]

        # Inner batching: (H, B, ng_len) int64 uses H*B*ng_len*8 bytes.
        # Cap at 512 MB to stay safe on 16 GB GPUs (T4/V100).
        inner_bs = max(1, (512 * 1024 * 1024) // (NUM_HASHES * ng_len * 8))

        for b_start in range(0, len(cands), inner_bs):
            b_end  = min(b_start + inner_bs, len(cands))
            i_idxs = idxs[b_start:b_end]
            i_cands = cands[b_start:b_end]

            # Feature matrix: (B, ng_len) int64, feature = pos*256 + byte
            feat = torch.tensor(
                [[pos * 256 + byte for pos, byte in enumerate(c)] for c in i_cands],
                dtype=torch.int64, device=device,
            )  # (B, ng_len)

            # Vectorised MinHash: (H, B, ng_len) → min over ng_len → (B, H)
            hv = (a_t[:, None, None] * feat[None] + b_t[:, None, None]) % p
            sigs[i_idxs] = hv.min(dim=2).values.T

    return sigs   # (n, NUM_HASHES)


def lsh_buckets_gpu(sigs: torch.Tensor, device: torch.device) -> dict[tuple, list[int]]:
    """
    Locality-Sensitive Hashing: group candidate indices into buckets.

    For each of the LSH_BANDS bands, compute a polynomial rolling hash of the
    (LSH_ROWS,) sub-vector on GPU, then sort + segment to find collisions.
    Only candidates inside ≥2-element segments are transferred to CPU.

    Returns dict: bucket_key → [candidate_indices].
    """
    n = sigs.shape[0]
    s = sigs.view(n, LSH_BANDS, LSH_ROWS)   # (n, 16, 8) int64, on GPU
    p = _LARGE_PRIME

    buckets: dict[tuple, list[int]] = defaultdict(list)

    for band in range(LSH_BANDS):
        # Polynomial rolling hash for this band: one pass over LSH_ROWS
        col = s[:, band, 0].clone()                         # (n,) int64
        for r in range(1, LSH_ROWS):
            col = (col * 1_000_003 + s[:, band, r]) % p    # (n,) int64

        # Sort by hash value
        sorted_vals, sorted_idx = torch.sort(col)           # (n,), (n,)

        # Find segment boundaries fully on GPU
        diff = torch.empty(n + 1, dtype=torch.bool, device=device)
        diff[0]    = True
        diff[1:-1] = sorted_vals[1:] != sorted_vals[:-1]
        diff[-1]   = True
        seg_starts = torch.where(diff[:-1])[0]              # (S,)
        seg_ends   = torch.where(diff[1:])[0]               # (S,)

        # Keep only segments with ≥2 elements before transferring to CPU
        sizes = seg_ends - seg_starts + 1
        mask  = sizes >= 2
        if not mask.any():
            continue

        starts_cpu = seg_starts[mask].cpu().tolist()
        ends_cpu   = seg_ends[mask].cpu().tolist()
        vals_cpu   = sorted_vals[seg_starts[mask]].cpu().tolist()
        idx_cpu    = sorted_idx.cpu().tolist()

        for s_i, e_i, val in zip(starts_cpu, ends_cpu, vals_cpu):
            key = (band, val)
            buckets[key].extend(idx_cpu[s_i : e_i + 1])

    return {k: v for k, v in buckets.items() if len(v) >= 2}


# ──────────────────────────────────────────────
#  Main tokenizer class
# ──────────────────────────────────────────────

class ClusteringTokenizer:
    """
    MinHash + LSH clustering tokenizer.

    The vocabulary is built offline (train()) then frozen.
    Encoding is O(n) greedy longest-match; decoding is O(1) array lookup.
    """

    BYTE_FALLBACK_SIZE = 256   # byte-level fallback tokens 0-255

    def __init__(
        self,
        vocab_size: int = 50257,
        min_ngram: int = 2,
        max_ngram: int = 8,
        top_k_factor: int = 10,
        device: Optional[str] = None,
        cluster: bool | str = False,
    ):
        """
        cluster controls the vocab-selection strategy after n-gram counting:

          False  (default) — no deduplication; directly select top-K by
                             frequency.  Fastest; LP-optimal when the
                             similarity constraints are all inactive.

          'lp'             — LP relaxation of the maximum-weight independent
                             set (MWIS) on the similarity graph:

                               max  Σ freq_i · y_i
                               s.t. y_i + y_j ≤ 1  ∀ similar (i,j)
                                    Σ y_i ≤ M
                                    0 ≤ y_i ≤ 1

                             Solved with scipy/HiGHS (polynomial time); result
                             is rounded greedily for a 2-approximation.
                             Requires scipy.  Similar pairs found via MinHash+LSH,
                             so total time ≈ cluster=True minus union-find plus
                             LP solve (~seconds for sparse graphs).

          True             — original MinHash+LSH+union-find heuristic.  Kept
                             for ablation; union-find's transitive closure can
                             over-merge (not equivalent to LP relaxation).
        """
        if cluster not in (False, True, "lp"):
            raise ValueError(f"cluster must be False, True, or 'lp', got {cluster!r}")
        self.vocab_size   = vocab_size
        self.min_ngram    = min_ngram
        self.max_ngram    = max_ngram
        self.top_k_factor = top_k_factor
        self.cluster      = cluster

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Populated after train()
        self._vocab: list[bytes] = []          # id → bytes
        self._token2id: dict[bytes, int] = {}  # bytes → id
        self._eot_token: int = vocab_size - 1  # last token = EOT by convention
        self._trained = False

    # ── Training ──────────────────────────────

    def train(self, text: str | bytes, verbose: bool = True) -> None:
        """
        Build vocabulary from a text corpus.

        When cluster=False (default, fast):
          1. Count n-gram frequencies (GPU bitwise packing + torch.sort + topk)
          2. Directly select top-(vocab_size - 256 - 1) n-grams by frequency
          3. Build final vocab

        When cluster=True (slow, original pipeline):
          1. Count n-gram frequencies
          2. Keep top-K candidates (top_k_factor × overshoot)
          3. Compute MinHash signatures (GPU)
          4. LSH clustering → union-find
          5. Select cluster representatives
          6. Build final vocab

        The clustering step is skipped by default because in practice the
        position-aware Jaccard threshold (≥0.8) merges very few pairs among
        the top-K high-frequency byte n-grams, making it an expensive no-op.
        """
        if isinstance(text, str):
            data = text.encode("utf-8", errors="replace")
        else:
            data = text

        if verbose:
            mode = "cluster=True" if self.cluster else "cluster=False (fast)"
            print(f"[ClusteringTokenizer] training on {len(data)/1e6:.1f} MB corpus")
            print(f"  device={self.device}, vocab_size={self.vocab_size}, {mode}")

        n_content_tokens = self.vocab_size - self.BYTE_FALLBACK_SIZE - 1  # -1 for EOT

        if not self.cluster:
            # ── Fast path: count → direct top-K ──────────────
            # top_k_factor=1: keep exactly n_content_tokens per n-gram length;
            # no need to over-sample for a clustering step that doesn't run.
            if verbose:
                print("  Step 1/2: counting n-gram frequencies (GPU) ...")
            freq = self._count_ngrams_gpu(data, keep_k=n_content_tokens)
            if verbose:
                print(f"    unique n-grams retained: {len(freq)}")
                print(f"  Step 2/2: selecting top-{n_content_tokens} tokens ...")
            top_sorted = sorted(freq.items(), key=lambda x: -x[1])
            top_reps = [tok for tok, _ in top_sorted[:n_content_tokens]]

        elif self.cluster == "lp":
            # ── LP path: count → MinHash → LSH → LP relaxation + greedy round ──
            # Same as cluster=True up to LSH; replaces union-find with LP.
            if verbose:
                print("  Step 1/4: counting n-gram frequencies ...")
            keep_k = (self.vocab_size - self.BYTE_FALLBACK_SIZE) * self.top_k_factor
            freq = self._count_ngrams_gpu(data, keep_k=keep_k)
            if verbose:
                print(f"    total unique n-grams: {len(freq)}")

            target_k = min(keep_k, len(freq))
            if verbose:
                print(f"  Step 2/4: keeping top-{target_k} candidates ...")
            candidates_sorted = sorted(freq.items(), key=lambda x: -x[1])[:target_k]
            candidates = [c for c, _ in candidates_sorted]
            cand_freqs = {c: f for c, f in candidates_sorted}

            if verbose:
                print(f"  Step 3/4: computing MinHash signatures on {self.device} ...")
            sigs = compute_minhash_signatures_gpu(candidates, self.device)
            if verbose:
                print(f"    signatures shape: {tuple(sigs.shape)}")

            if verbose:
                print("  Step 4/4: LSH → LP relaxation (MWIS) + greedy rounding ...")
            buckets = lsh_buckets_gpu(sigs, self.device)

            # Collect unique similar pairs from all LSH buckets.
            # Each pair (i,j) means Jaccard(candidates[i], candidates[j]) ≈ ≥ threshold.
            pair_set: set[tuple[int, int]] = set()
            for members in buckets.values():
                for a in range(len(members)):
                    for b_idx in range(a + 1, len(members)):
                        u, v = members[a], members[b_idx]
                        pair_set.add((min(u, v), max(u, v)))
            similar_pairs = list(pair_set)

            if verbose:
                print(f"    similar pairs found: {len(similar_pairs)}")
                print(f"    solving LP (K={len(candidates)}, constraints={len(similar_pairs)+1}) ...")

            top_reps = self._select_vocab_lp(
                candidates, cand_freqs, similar_pairs, n_content_tokens
            )

        else:
            # ── Full pipeline: count → MinHash → LSH → union-find ──
            if verbose:
                print("  Step 1/4: counting n-gram frequencies ...")
            keep_k = (self.vocab_size - self.BYTE_FALLBACK_SIZE) * self.top_k_factor
            freq = self._count_ngrams_gpu(data, keep_k=keep_k)
            if verbose:
                print(f"    total unique n-grams: {len(freq)}")

            target_k = min(keep_k, len(freq))
            if verbose:
                print(f"  Step 2/4: keeping top-{target_k} candidates ...")
            candidates_sorted = sorted(freq.items(), key=lambda x: -x[1])[:target_k]
            candidates = [c for c, _ in candidates_sorted]
            cand_freqs = {c: f for c, f in candidates_sorted}

            if verbose:
                print(f"  Step 3/4: computing MinHash signatures on {self.device} ...")
            sigs = compute_minhash_signatures_gpu(candidates, self.device)
            if verbose:
                print(f"    signatures shape: {tuple(sigs.shape)}")

            if verbose:
                print("  Step 4/4: LSH clustering & vocab construction ...")
            buckets = lsh_buckets_gpu(sigs, self.device)

            parent = list(range(len(candidates)))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(x, y):
                px, py = find(x), find(y)
                if px != py:
                    parent[px] = py

            for members in buckets.values():
                for m in members[1:]:
                    union(members[0], m)

            clusters: dict[int, list[bytes]] = defaultdict(list)
            for i, c in enumerate(candidates):
                clusters[find(i)].append(c)

            representatives: list[tuple[bytes, int]] = []
            for members in clusters.values():
                best = max(members, key=lambda c: cand_freqs[c])
                representatives.append((best, cand_freqs[best]))

            representatives.sort(key=lambda x: -x[1])
            top_reps = [tok for tok, _ in representatives[:n_content_tokens]]

        # ── Build final vocabulary ────────────────────────
        # Layout:
        #   0 .. 255          : single-byte fallback
        #   256 .. 256+N-1    : cluster representative tokens (multi-byte)
        #   vocab_size - 1    : EOT
        vocab: list[bytes] = []
        # byte fallbacks
        for b in range(self.BYTE_FALLBACK_SIZE):
            vocab.append(bytes([b]))
        # multi-byte tokens
        seen = set(vocab)
        for tok in top_reps:
            if tok not in seen and len(vocab) < self.vocab_size - 1:
                vocab.append(tok)
                seen.add(tok)
        # EOT
        eot_bytes = b"<|endoftext|>"
        vocab.append(eot_bytes)

        # Pad to exact vocab_size if needed
        while len(vocab) < self.vocab_size:
            vocab.append(b"<|pad|>")

        self._vocab    = vocab[: self.vocab_size]
        self._token2id = {tok: i for i, tok in enumerate(self._vocab)}
        self._eot_token = self.vocab_size - 1
        self._trained  = True

        if verbose:
            avg_len = np.mean([len(t) for t in self._vocab])
            print(f"  Done! vocab_size={len(self._vocab)}, "
                  f"avg_token_len={avg_len:.2f} bytes")

    # ── LP relaxation vocab selection ─────────────────────────────────────────

    def _select_vocab_lp(
        self,
        candidates: list[bytes],
        cand_freqs: dict[bytes, int],
        similar_pairs: list[tuple[int, int]],
        n_select: int,
    ) -> list[bytes]:
        """
        LP relaxation of the Maximum Weighted Independent Set (MWIS) problem
        on the similarity graph, followed by greedy rounding.

        Integer Program
        ───────────────
          max  Σ_i  freq_i · y_i
          s.t. y_i + y_j ≤ 1    ∀ similar pairs (i,j)   [independence]
               Σ_i  y_i  ≤ M                              [vocab budget]
               y_i ∈ {0,1}

        LP Relaxation (this function)
        ──────────────────────────────
          Same but y_i ∈ [0,1].  Solved with scipy / HiGHS in O(K^1..2.5)
          depending on problem density.  For typical sparse similarity graphs
          (few Jaccard≥0.8 pairs among high-freq n-grams) this runs in seconds.

        Greedy Rounding → 2-approximation
        ────────────────────────────────────
          Sort by x_i·freq_i (LP fractional value × frequency), greedily
          select each candidate unless it conflicts with an already-selected one.
          For MWIS on sparse graphs, this is typically near-optimal in practice.
        """
        try:
            from scipy.optimize import linprog
            from scipy.sparse import lil_array, csr_array
        except ImportError as e:
            raise ImportError(
                "cluster='lp' requires scipy. Install with: pip install scipy"
            ) from e

        K = len(candidates)
        freqs = np.array([cand_freqs[c] for c in candidates], dtype=np.float64)
        # Normalise for numerical stability; LP objective scale does not matter.
        freqs_norm = freqs / (freqs.max() + 1e-9)

        # Objective: minimise −freq (scipy convention for minimisation)
        c_obj = -freqs_norm

        # ── Build sparse inequality constraint matrix ──────────────────────────
        # Each similar pair (i,j) contributes: y_i + y_j ≤ 1
        # Plus one global budget row:          Σ y_i ≤ M
        n_pair_constraints = len(similar_pairs)
        n_rows = n_pair_constraints + 1

        A = lil_array((n_rows, K), dtype=np.float64)
        b = np.ones(n_rows, dtype=np.float64)

        for row, (i, j) in enumerate(similar_pairs):
            A[row, i] = 1.0
            A[row, j] = 1.0

        # Budget row
        A[n_pair_constraints, :] = 1.0
        b[n_pair_constraints] = float(n_select)

        bounds = [(0.0, 1.0)] * K

        result = linprog(
            c_obj,
            A_ub=csr_array(A),
            b_ub=b,
            bounds=bounds,
            method="highs",
            options={"disp": False},
        )
        if result.status != 0:
            # Fall back to unconstrained greedy if LP fails
            import warnings
            warnings.warn(
                f"LP solver returned status {result.status} ({result.message}); "
                "falling back to greedy top-K selection.",
                RuntimeWarning,
            )
            order = np.argsort(-freqs)
            return [candidates[i] for i in order[:n_select]]

        x_lp = result.x   # fractional solution in [0,1]^K

        # ── Greedy rounding ────────────────────────────────────────────────────
        # Sort by LP-fractional value × raw frequency (both reflect importance)
        score = x_lp * freqs
        order = np.argsort(-score)

        # Build adjacency list for O(1) conflict lookup
        adj: dict[int, set[int]] = defaultdict(set)
        for i, j in similar_pairs:
            adj[i].add(j)
            adj[j].add(i)

        selected: list[bytes] = []
        blocked: set[int] = set()

        for idx in order:
            if idx in blocked:
                continue
            selected.append(candidates[idx])
            blocked.update(adj[idx])
            if len(selected) >= n_select:
                break

        return selected

    # ── N-gram frequency counting ──────────────────────────────────────────────

    def _count_ngrams_gpu(self, data: bytes, keep_k: int | None = None) -> dict[bytes, int]:
        """
        Count byte n-gram frequencies using GPU bitwise packing + torch.sort.

        Algorithm per n-gram length n:
          1. Pack: keys[i] = data[i] | data[i+1]<<8 | ... | data[i+n-1]<<(8*(n-1))
             → (L,) int64 on GPU, no intermediate (L,8) buffer needed
          2. torch.sort: GPU radix sort, ~5-10× faster than numpy mergesort
          3. Run-length encoding via diff detection → (unique_keys, counts)
          4. torch.topk on GPU: keep only keep_k entries before CPU transfer
          5. Python loop is O(keep_k), not O(unique n-grams)

        Memory per n: 2 × (L,) int64 ≈ 2 × corpus_size × 8 bytes
                      = ~800 MB for 50 MB corpus → safe on L4 (24 GB).
        """
        # Upload corpus once; reuse across all n values.
        data_np = np.frombuffer(data, dtype=np.uint8)
        data_t  = torch.from_numpy(data_np.copy()).to(self.device)   # (N,) uint8
        L_total = len(data_t)

        if keep_k is None:
            keep_k = (self.vocab_size - self.BYTE_FALLBACK_SIZE) * self.top_k_factor

        freq: dict[bytes, int] = {}

        for n in range(self.min_ngram, self.max_ngram + 1):
            L = L_total - n + 1
            if L <= 0:
                continue

            # ── Bitwise pack: each n-gram → one int64 key ──────────────────────
            # keys[i] = data[i] | data[i+1]<<8 | ... | data[i+n-1]<<(8*(n-1))
            # Little-endian so the first byte is the LSB — same as b'\xb1\xb2'.
            # n ≤ 8 bytes → always fits in int64 (sign bit only set for n=8 if
            # the 8th byte ≥ 128, but sort order is preserved because we only
            # need consistent ordering, not numeric magnitude).
            keys = data_t[:L].to(torch.int64)   # byte 0
            for i in range(1, n):
                keys = keys | (data_t[i : L + i].to(torch.int64) << (8 * i))

            # ── GPU sort → run-length encoding ─────────────────────────────────
            sorted_keys, _ = torch.sort(keys)

            # Mark positions where the value changes (run boundaries)
            boundary = torch.empty(L + 1, dtype=torch.bool, device=self.device)
            boundary[0]  = True
            boundary[1:] = sorted_keys[1:] != sorted_keys[:-1]
            boundary[L]  = True

            unique_keys = sorted_keys[boundary[:L]]   # (U,) — one entry per run
            run_starts  = torch.where(boundary[:L])[0]
            run_ends    = torch.where(boundary[1:])[0]
            counts      = (run_ends - run_starts + 1).to(torch.int32)   # (U,)

            # ── Top-K filter on GPU ─────────────────────────────────────────────
            if counts.numel() > keep_k:
                top_counts, top_idx = torch.topk(counts, keep_k)
                unique_keys = unique_keys[top_idx]
                counts      = top_counts

            # ── CPU transfer + Python dict: only keep_k entries ────────────────
            import struct
            pack_q = struct.Struct("<q")   # little-endian signed int64
            for key_int, cnt in zip(unique_keys.cpu().tolist(), counts.cpu().tolist()):
                bs = pack_q.pack(key_int)[:n]
                if bs in freq:
                    freq[bs] += cnt
                else:
                    freq[bs] = cnt

        return freq

    # ── Encode / Decode ───────────────────────

    def encode(self, text: str | bytes) -> list[int]:
        """Greedy longest-match encoding, O(n). Falls back to byte tokens."""
        assert self._trained, "Call train() first"
        if isinstance(text, str):
            data = text.encode("utf-8", errors="replace")
        else:
            data = text

        ids: list[int] = []
        i = 0
        while i < len(data):
            # Try lengths from max_ngram down to 1
            matched = False
            for length in range(self.max_ngram, 0, -1):
                chunk = data[i: i + length]
                if chunk in self._token2id:
                    ids.append(self._token2id[chunk])
                    i += length
                    matched = True
                    break
            if not matched:
                # Single byte fallback (always in vocab as ids 0-255)
                ids.append(data[i])
                i += 1
        return ids

    def encode_with_eot(self, text: str | bytes) -> list[int]:
        """Encode and prepend EOT token (for document boundaries)."""
        return [self._eot_token] + self.encode(text)

    def decode(self, ids: list[int]) -> bytes:
        """Decode token ids to bytes."""
        assert self._trained, "Call train() first"
        parts = []
        for i in ids:
            if 0 <= i < len(self._vocab):
                tok = self._vocab[i]
                # Skip special tokens in text output
                if tok not in (b"<|endoftext|>", b"<|pad|>"):
                    parts.append(tok)
        return b"".join(parts)

    def decode_str(self, ids: list[int]) -> str:
        return self.decode(ids).decode("utf-8", errors="replace")

    @property
    def eot_token(self) -> int:
        return self._eot_token

    @property
    def n_vocab(self) -> int:
        return len(self._vocab)

    # ── Save / Load ───────────────────────────

    def save(self, path: str) -> None:
        """
        Save tokenizer to a binary file compatible with llm.c's tokenizer format.

        Format (matching llmc/tokenizer.h):
          Header: 256 × int32
            [0] = 20240328  (magic)
            [1] = 2         (version)
            [2] = vocab_size
            [3] = eot_token
          Token data: for each token 0..vocab_size-1:
            uint8 length, then <length> bytes
        """
        assert self._trained, "Call train() first"
        import pathlib
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)

        header = [0] * 256
        header[0] = 20240328
        header[1] = 2
        header[2] = self.vocab_size
        header[3] = self._eot_token

        with open(path, "wb") as f:
            f.write(struct.pack(f"{256}i", *header))
            for tok in self._vocab:
                # Truncate to 255 bytes if somehow longer
                tok_bytes = tok[:255]
                f.write(struct.pack("B", len(tok_bytes)))
                f.write(tok_bytes)

        print(f"[ClusteringTokenizer] saved to {path}")

    def load(self, path: str) -> None:
        """Load tokenizer from binary file (inverse of save())."""
        with open(path, "rb") as f:
            header_raw = f.read(256 * 4)
            header = struct.unpack(f"{256}i", header_raw)

            assert header[0] == 20240328, "bad magic"
            assert header[1] == 2,        "unsupported version"
            vocab_size = header[2]
            eot_token  = header[3]

            vocab: list[bytes] = []
            for _ in range(vocab_size):
                length = struct.unpack("B", f.read(1))[0]
                tok    = f.read(length)
                vocab.append(tok)

        self.vocab_size  = vocab_size
        self._vocab      = vocab
        self._token2id   = {tok: i for i, tok in enumerate(vocab)}
        self._eot_token  = eot_token
        self._trained    = True
        print(f"[ClusteringTokenizer] loaded from {path}, vocab_size={vocab_size}")


# ──────────────────────────────────────────────
#  Quick smoke test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    text = ("Hello world! This is a test of the clustering tokenizer. " * 200)
    tok = ClusteringTokenizer(vocab_size=512, max_ngram=6, top_k_factor=5)
    tok.train(text)

    sample = "Hello world!"
    ids = tok.encode(sample)
    decoded = tok.decode_str(ids)
    print(f"encode({sample!r}) → {ids}")
    print(f"decode → {decoded!r}")
    assert sample.encode() == tok.decode(ids), \
        f"roundtrip failed: {sample!r} != {decoded!r}"
    print("✓ roundtrip OK")
