from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset

from two_tower.features.encode import encode_cats, encode_multi_matrix, encode_nums
from two_tower.features.prepare import FeatureArtifacts


class TwoTowerDataset(Dataset):
    """Pre-encodes the full frame once (same as reference notebook)."""

    def __init__(self, df: pd.DataFrame, fa: FeatureArtifacts, *, label_col: str, multi_max_tokens: int):
        df = df.reset_index(drop=True)
        self.user_cat = encode_cats(df, fa.user_cat_cols, fa.user_vocabs)
        self.user_num = encode_nums(df, fa.user_num_cols)
        self.user_multi = encode_multi_matrix(df, fa.user_multi_cols, fa.user_multi_vocabs, multi_max_tokens)
        self.client_cat = encode_cats(df, fa.client_cat_cols, fa.client_vocabs)
        self.client_num = encode_nums(df, fa.client_num_cols)
        self.client_multi = encode_multi_matrix(
            df, fa.client_multi_cols, fa.client_multi_vocabs, multi_max_tokens
        )
        self.y = torch.from_numpy(df[label_col].astype("float32").to_numpy()).unsqueeze(1)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        return (
            self.user_cat[idx],
            self.user_num[idx],
            self.user_multi[idx],
            self.client_cat[idx],
            self.client_num[idx],
            self.client_multi[idx],
            self.y[idx],
        )
