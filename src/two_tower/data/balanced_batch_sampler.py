from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from torch.utils.data import Sampler


@dataclass(frozen=True)
class BalancedBatchSpec:
    """Per-batch quotas derived from global client proportions + per-client neg/pos ratio."""

    # client_id -> total rows from this client per batch
    per_client_total: dict[object, int]
    # (client_id, is_pos) -> rows per batch
    per_bucket: dict[tuple[object, bool], int]


def _largest_remainder_split(total: int, weights: dict[object, float]) -> dict[object, int]:
    """Allocate integer counts summing to total, proportional to weights (>=0)."""
    if total < 0:
        raise ValueError("total must be >= 0")
    if not weights:
        return {}
    keys = list(weights.keys())
    w = np.asarray([float(weights[k]) for k in keys], dtype=np.float64)
    w = np.clip(w, 0.0, None)
    s = float(w.sum())
    if s <= 0.0:
        # fallback: uniform
        w = np.ones_like(w, dtype=np.float64)
        s = float(w.sum())
    raw = (w / s) * float(total)
    base = np.floor(raw).astype(np.int64)
    rem = raw - base
    out = {k: int(v) for k, v in zip(keys, base)}
    missing = int(total - base.sum())
    if missing > 0:
        order = np.argsort(-rem)  # descending remainder
        for i in range(missing):
            out[keys[int(order[i % len(order)])]] += 1
    elif missing < 0:
        # Remove from smallest remainders first (ties don't matter much).
        order = np.argsort(rem)  # ascending remainder
        need = -missing
        for i in range(need):
            k = keys[int(order[i % len(order)])]
            if out[k] > 0:
                out[k] -= 1
    # final guard
    delta = total - sum(out.values())
    if delta != 0:
        # adjust arbitrarily but deterministically
        k0 = keys[0]
        out[k0] += int(delta)
    return out


def build_balanced_batch_spec(
    df: pd.DataFrame,
    *,
    client_id_col: str,
    label_col: str,
    batch_size: int,
    neg_per_pos: int = 3,
) -> BalancedBatchSpec:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if neg_per_pos < 0:
        raise ValueError("neg_per_pos must be >= 0")
    if client_id_col not in df.columns:
        raise KeyError(f"missing client_id_col {client_id_col!r} in training frame")
    if label_col not in df.columns:
        raise KeyError(f"missing label_col {label_col!r} in training frame")

    client_counts = df[client_id_col].value_counts(dropna=False)
    weights = {cid: float(n) for cid, n in client_counts.items()}
    per_client_total = _largest_remainder_split(batch_size, weights)

    # neg:pos = neg_per_pos:1 => pos_frac = 1/(neg_per_pos+1)
    denom = int(neg_per_pos + 1)
    per_bucket: dict[tuple[object, bool], int] = {}
    for cid, n_total in per_client_total.items():
        n_pos = int(round(float(n_total) / float(denom)))
        n_pos = min(max(n_pos, 0), n_total)
        n_neg = int(n_total - n_pos)
        per_bucket[(cid, True)] = n_pos
        per_bucket[(cid, False)] = n_neg

    # Ensure totals still sum to batch_size.
    if sum(per_client_total.values()) != int(batch_size):
        raise AssertionError("per_client_total does not sum to batch_size")
    if sum(per_bucket.values()) != int(batch_size):
        # Fix any rounding drift by adjusting the largest client.
        delta = int(batch_size - sum(per_bucket.values()))
        if delta != 0 and per_client_total:
            cid0 = max(per_client_total.keys(), key=lambda c: per_client_total[c])
            # adjust positives by delta, clamp to [0, per_client_total[cid0]]
            cur_pos = per_bucket[(cid0, True)]
            cur_neg = per_bucket[(cid0, False)]
            new_pos = min(max(cur_pos + delta, 0), per_client_total[cid0])
            per_bucket[(cid0, True)] = new_pos
            per_bucket[(cid0, False)] = per_client_total[cid0] - new_pos
    if sum(per_bucket.values()) != int(batch_size):
        raise AssertionError("per_bucket does not sum to batch_size after adjustment")

    return BalancedBatchSpec(per_client_total=per_client_total, per_bucket=per_bucket)


class BalancedClientLabelBatchSampler(Sampler[list[int]]):
    """
    BatchSampler that yields indices so each batch matches:

    - global client distribution (based on df[client_id_col] row counts)
    - within each client, a fixed neg:pos ratio (neg_per_pos:1)

    Uses wrap+reshuffle when a bucket exhausts, so it can run for arbitrary steps/epoch (option B).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        client_id_col: str,
        label_col: str,
        batch_size: int,
        neg_per_pos: int = 3,
        seed: int = 42,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.client_id_col = str(client_id_col)
        self.label_col = str(label_col)
        self.batch_size = int(batch_size)
        self.neg_per_pos = int(neg_per_pos)
        self.seed = int(seed)

        self.spec = build_balanced_batch_spec(
            self.df,
            client_id_col=self.client_id_col,
            label_col=self.label_col,
            batch_size=self.batch_size,
            neg_per_pos=self.neg_per_pos,
        )

        y = self.df[self.label_col].to_numpy()
        # treat >0 as positive; supports float labels (0/1) and bool/int.
        is_pos = (y.astype(np.float64) > 0.0)
        cids = self.df[self.client_id_col].to_numpy(dtype=object)

        self._rng = np.random.default_rng(self.seed)
        self._buckets: dict[tuple[object, bool], np.ndarray] = {}
        self._ptr: dict[tuple[object, bool], int] = {}

        for cid in self.spec.per_client_total.keys():
            for pos_flag in (True, False):
                m = (cids == cid) & (is_pos if pos_flag else (~is_pos))
                idx = np.nonzero(m)[0].astype(np.int64)
                if idx.size == 0 and self.spec.per_bucket[(cid, pos_flag)] > 0:
                    raise ValueError(
                        f"Cannot satisfy batch quota: client={cid!r} pos={pos_flag} "
                        f"requires {self.spec.per_bucket[(cid, pos_flag)]}/batch but dataset has 0 rows."
                    )
                self._rng.shuffle(idx)
                self._buckets[(cid, pos_flag)] = idx
                self._ptr[(cid, pos_flag)] = 0

    def __iter__(self):
        while True:
            out: list[int] = []
            for (cid, pos_flag), k in self.spec.per_bucket.items():
                if k <= 0:
                    continue
                arr = self._buckets[(cid, pos_flag)]
                if arr.size == 0:
                    continue
                ptr = self._ptr[(cid, pos_flag)]
                # Wrap + reshuffle when needed
                if ptr + k <= arr.size:
                    take = arr[ptr : ptr + k]
                    self._ptr[(cid, pos_flag)] = ptr + k
                    out.extend(take.tolist())
                else:
                    # take remainder, reshuffle, then take from start
                    rem = arr[ptr:].tolist()
                    self._rng.shuffle(arr)
                    need = k - (arr.size - ptr)
                    head = arr[:need].tolist()
                    self._ptr[(cid, pos_flag)] = need
                    out.extend(rem)
                    out.extend(head)

            if len(out) != self.batch_size:
                raise RuntimeError(f"balanced sampler internal error: got {len(out)} != batch_size {self.batch_size}")
            # Optional: shuffle within batch so model doesn't see grouped blocks.
            out = self._rng.permutation(np.asarray(out, dtype=np.int64)).tolist()
            yield out

    def __len__(self) -> int:
        # This sampler is intentionally "infinite" under option B; length is undefined.
        # Returning 0 avoids misleading progress bars / schedulers relying on __len__.
        return 0

