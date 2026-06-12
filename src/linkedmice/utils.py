from typing import List

import pandas as pd

from .feature_eng import DEFAULT_BPP
from .mice import SkipDep


def normalize_categorical_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert object-dtype categorical columns to string-only categories.

    PyArrow (used by to_parquet) rejects CategoricalDtype whose categories have
    object dtype mixing Python int and str values.  This maps every category
    label to str so the column serialises cleanly.
    """
    df = df.copy()
    for col in df.select_dtypes("category").columns:
        cats = df[col].cat.categories
        if cats.dtype == object:
            new_cats = cats.map(str)
            df[col] = df[col].cat.rename_categories(new_cats)
    return df


def repair_parent_child_nan(
    df: pd.DataFrame,
    deps: List[SkipDep],
    bpp_value: str = DEFAULT_BPP,
) -> pd.DataFrame:
    """Reset child BPP rows to NaN when the skip condition cannot be confirmed.

    When at least one parent is NaN and the combined skip predicate does not
    return ``True`` for the row (because NaN parent comparisons evaluate to
    ``False`` in pandas), the skip condition is unresolvable and the child must
    be NaN rather than BPP.  If another parent already triggers the skip
    condition (predicate returns ``True`` despite some NaN parents), the child
    correctly stays BPP.

    Call this before building skip / missing masks.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame; a copy is returned.
    deps : List[SkipDep]
        Skip dependency triples ``(parent_cols, combined_predicate, child_col)``
        in the same format used by :func:`refresh_dependent_skips`.
    bpp_value : str, optional
        BPP sentinel string.  Default :data:`DEFAULT_BPP`.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with affected child cells reset from BPP to NaN.
    """
    df = df.copy()
    for parents, pred, child in deps:
        missing_cols = [p for p in parents if p not in df.columns]
        if missing_cols or child not in df.columns:
            continue
        child_bpp = df[child] == bpp_value
        any_parent_nan = pd.concat([df[p].isna() for p in parents], axis=1).any(axis=1)
        # pred returns False for NaN parents, so ~pred identifies rows where
        # the skip is not confirmed by the currently observed parent values.
        skip_not_confirmed = ~pred(df)
        mask = child_bpp & any_parent_nan & skip_not_confirmed
        n_fixed = int(mask.sum())
        if n_fixed:
            df.loc[mask, child] = pd.NA
            print(
                f"Repaired {child}: {n_fixed:,} rows reset from BPP → NaN "
                f"(parent(s) {parents} NaN, skip not confirmed)"
            )
    return df
