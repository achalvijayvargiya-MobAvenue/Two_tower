from __future__ import annotations

from typing import Iterable


def intersect_feature_cols(df, cols: list[str], label: str) -> list[str]:
    """Keep only columns present in the dataframe.

    Mirrors notebook helper `_intersect_feature_cols`.
    """
    ok = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        tail = " ..." if len(missing) > 8 else ""
        print(f"{label}: skipping {len(missing)} columns not in data: {missing[:8]}{tail}")
    return ok


def split_cat_num_multi(df, cols: list[str], multi_cols: set[str]) -> tuple[list[str], list[str], list[str]]:
    """Split into scalar categorical, numeric, multi-valued categorical."""
    import pandas as pd

    cat: list[str] = []
    num: list[str] = []
    multi: list[str] = []
    for c in cols:
        if c in multi_cols:
            multi.append(c)
            continue
        dt = df[c].dtype
        if pd.api.types.is_string_dtype(dt) or pd.api.types.is_object_dtype(dt):
            cat.append(c)
        else:
            num.append(c)
    return cat, num, multi


def ensure_columns_present(df, required: Iterable[str], *, where: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        tail = " ..." if len(missing) > 10 else ""
        raise KeyError(f"{where}: missing {len(missing)} column(s): {missing[:10]}{tail}")

