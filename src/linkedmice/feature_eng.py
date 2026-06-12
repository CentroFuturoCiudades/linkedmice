from typing import List

import pandas as pd

# Default structural-skip sentinel.  Inherited from the Mexico Census 2020
# extended questionnaire ("Blanco por pase" = blank due to questionnaire skip
# pattern); every function that consumes it accepts a ``bpp_value`` override.
DEFAULT_BPP = "Blanco por pase"


def broadcast_household_attrs(
    person_df: pd.DataFrame,
    household_df: pd.DataFrame,
    cols_to_broadcast: List[str],
    hh_key: str = "ID_VIV",
) -> pd.DataFrame:
    """Merge selected household columns into the person DataFrame.

    Parameters
    ----------
    person_df : pd.DataFrame
        Person DataFrame indexed by ``(hh_key, person_id)``.
    household_df : pd.DataFrame
        Household DataFrame indexed by ``hh_key``.
    cols_to_broadcast : List[str]
        Household columns to merge in (columns absent from ``household_df``
        are silently skipped).
    hh_key : str, optional
        Name of the household-key index level.  Default ``"ID_VIV"``.

    Returns
    -------
    pd.DataFrame
        ``person_df`` with the selected household columns appended and
        suffixed with ``'_hh'``.
    """
    available = [c for c in cols_to_broadcast if c in household_df.columns]
    hh_vals = household_df[available].copy()
    hh_vals.columns = [f"{c}_hh" for c in available]

    hh_ids = person_df.index.get_level_values(hh_key)
    # Look up one row per person, reset to person MultiIndex
    broadcast = hh_vals.loc[hh_ids]
    broadcast.index = person_df.index
    return pd.concat([person_df, broadcast], axis=1)


def leave_one_out_category_counts(
    df: pd.DataFrame,
    col: str,
    hh_key: str = "ID_VIV",
    bpp_value: str = DEFAULT_BPP,
) -> pd.DataFrame:
    """Count how many *other* household members hold each category of ``col``.

    Structural-skip (``bpp_value``) and NaN values are excluded from the
    household tally.

    Parameters
    ----------
    df : pd.DataFrame
        Person DataFrame indexed by ``(hh_key, person_id)``.
    col : str
        Column whose category distribution is tallied within each household.
    hh_key : str, optional
        Name of the household-key index level.  Default ``"ID_VIV"``.
    bpp_value : str, optional
        Structural-skip sentinel string.  Default :data:`DEFAULT_BPP`.

    Returns
    -------
    pd.DataFrame
        ``int16`` columns named ``n_other_{col}_{category}``, one per category
        of ``col``, with the same index as ``df``.
    """
    hh_ids = df.index.get_level_values(hh_key)

    # Treat BPP and NaN as unknown — do not count them
    valid_vals = df[col].where(df[col].notna() & (df[col] != bpp_value))
    dummies = pd.get_dummies(
        valid_vals, prefix=f"n_other_{col}", dummy_na=False, dtype="int16"
    )

    # Sum per household, then subtract own row
    hh_totals = dummies.groupby(hh_ids).transform("sum")
    return (hh_totals - dummies).astype("int16")
