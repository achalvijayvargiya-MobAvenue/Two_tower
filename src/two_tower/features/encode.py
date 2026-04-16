from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from two_tower.features.vocab import parse_multi_cell


@dataclass(frozen=True)
class TwoTowerBatch:
    user_cat: torch.Tensor
    user_num: torch.Tensor
    user_multi: torch.Tensor
    client_cat: torch.Tensor
    client_num: torch.Tensor
    client_multi: torch.Tensor
    label: torch.Tensor


def encode_cats(df: pd.DataFrame, cat_cols: list[str], vocabs: dict) -> torch.Tensor:
    if not cat_cols:
        return torch.zeros((len(df), 0), dtype=torch.long)
    out = []
    for c in cat_cols:
        vocab = vocabs[c]
        ids = df[c].map(lambda x: vocab.encode_scalar(x)).astype("int64").to_numpy()
        out.append(torch.from_numpy(ids))
    return torch.stack(out, dim=1)


def encode_multi_matrix(
    df: pd.DataFrame, cols: list[str], vocabs: dict, max_tokens: int
) -> torch.Tensor:
    """Shape (N, len(cols), max_tokens); 0 = pad / empty slot."""
    n = len(df)
    f = len(cols)
    mt = int(max_tokens)
    if f == 0:
        return torch.zeros((n, 0, max(1, mt)), dtype=torch.long)
    out = np.zeros((n, f, mt), dtype=np.int64)
    for j, c in enumerate(cols):
        vcb = vocabs[c]
        for i, val in enumerate(df[c]):
            toks = parse_multi_cell(val)[:mt]
            for k, tok in enumerate(toks):
                tid = vcb.token_to_id.get(tok)
                if tid is not None:
                    out[i, j, k] = tid
                else:
                    h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
                    out[i, j, k] = int(vcb.n_frequent + 1 + (h % vcb.num_oov_buckets))
    return torch.from_numpy(out)


def encode_nums(df: pd.DataFrame, num_cols: list[str]) -> torch.Tensor:
    if not num_cols:
        return torch.zeros((len(df), 0), dtype=torch.float32)
    x = df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32").to_numpy()
    return torch.from_numpy(x)


def collate_fn(batch) -> TwoTowerBatch:
    uc, un, um, cc, cn, cm, y = zip(*batch)
    return TwoTowerBatch(
        user_cat=torch.stack(uc),
        user_num=torch.stack(un),
        user_multi=torch.stack(um),
        client_cat=torch.stack(cc),
        client_num=torch.stack(cn),
        client_multi=torch.stack(cm),
        label=torch.stack(y),
    )
