from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CatVocab:
    token_to_id: dict[str, int]
    n_frequent: int
    num_oov_buckets: int

    @property
    def size(self) -> int:
        # 0 reserved for missing/padding
        return int(self.n_frequent + self.num_oov_buckets + 1)

    def encode_scalar(self, raw) -> int:
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            return 0
        s = str(raw).strip()
        if s == "" or s.lower() == "nan" or s == "__NA__":
            return 0
        tid = self.token_to_id.get(s)
        if tid is not None:
            return int(tid)
        h = int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)
        return int(self.n_frequent + 1 + (h % self.num_oov_buckets))


def build_cat_vocab(series, min_count: int, num_oov_buckets: int) -> CatVocab:
    s = series.fillna("__NA__").astype(str)
    vc = s.value_counts()
    frequent = vc[vc > int(min_count)]
    tokens = list(frequent.index)
    token_to_id = {tok: i + 1 for i, tok in enumerate(tokens)}
    return CatVocab(token_to_id=token_to_id, n_frequent=len(tokens), num_oov_buckets=int(num_oov_buckets))


def parse_multi_cell(val) -> list[str]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if isinstance(val, (list, tuple)):
        return [str(x).strip() for x in val if str(x).strip() and str(x).strip().lower() != "nan"]
    if isinstance(val, np.ndarray):
        return [
            str(x).strip()
            for x in val.flatten().tolist()
            if str(x).strip() and str(x).strip().lower() != "nan"
        ]
    s = str(val).strip()
    if not s or s == "__NA__" or s.lower() == "nan":
        return []
    parts = re.split(r"[\|,;]+|\s+", s)
    return [p for p in parts if p]


def build_multi_token_vocab(series, min_count: int, num_oov_buckets: int) -> CatVocab:
    ctr: Counter[str] = Counter()
    for v in series:
        for tok in parse_multi_cell(v):
            ctr[tok] += 1
    tokens = sorted([t for t, n in ctr.items() if n > int(min_count)], key=lambda t: (-ctr[t], t))
    token_to_id = {tok: i + 1 for i, tok in enumerate(tokens)}
    return CatVocab(token_to_id=token_to_id, n_frequent=len(tokens), num_oov_buckets=int(num_oov_buckets))


def vocab_to_dict(v: CatVocab) -> dict:
    return {"token_to_id": dict(v.token_to_id), "n_frequent": int(v.n_frequent), "num_oov_buckets": int(v.num_oov_buckets)}


def vocab_from_dict(d: dict) -> CatVocab:
    return CatVocab(token_to_id=dict(d["token_to_id"]), n_frequent=int(d["n_frequent"]), num_oov_buckets=int(d["num_oov_buckets"]))

