from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_DEFAULT_SEED = 42


def create_validation_mask(
    df: pd.DataFrame,
    targets: List[str],
    missing_mask_dict: Dict[str, pd.Series],
    skip_mask_dict: Dict[str, pd.Series],
    frac: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.Series], Dict[str, pd.Series]]:
    """Artificially mask a fraction of observed values for held-out validation.

    Observed rows for each target are those where
    ``~missing_mask_dict[col] & ~skip_mask_dict[col]``.  A random ``frac``
    of those rows are removed from the returned DataFrame (set to ``pd.NA``)
    so the imputer can be evaluated against ground-truth values.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame.
    targets : List[str]
        Columns to mask.
    missing_mask_dict : Dict[str, pd.Series]
        Original NaN masks (columns absent from ``df`` are skipped).
    skip_mask_dict : Dict[str, pd.Series]
        BPP masks.
    frac : float, optional
        Fraction of observed values to hold out per column.  Default ``0.05``.
    rng : np.random.Generator, optional
        Random number generator.  Defaults to ``np.random.default_rng(42)``.

    Returns
    -------
    df_masked : pd.DataFrame
        Copy of ``df`` with held-out cells set to ``pd.NA``.
    missing_mask_masked : Dict[str, pd.Series]
        Updated masks: original missing rows plus newly held-out rows.
    held_out_values : Dict[str, pd.Series]
        ``{col: Series of true values at held-out indices}``.
    """
    if rng is None:
        rng = np.random.default_rng(_DEFAULT_SEED)

    df_masked = df.copy()
    missing_mask_masked = {col: mm.copy() for col, mm in missing_mask_dict.items()}
    held_out_values: Dict[str, pd.Series] = {}

    for col in targets:
        if col not in df.columns:
            continue
        mm = missing_mask_dict.get(col, pd.Series(False, index=df.index))
        sm = skip_mask_dict.get(col, pd.Series(False, index=df.index))

        obs_idx = df.index[~mm & ~sm]
        if len(obs_idx) == 0:
            continue

        n_holdout = max(1, int(len(obs_idx) * frac))
        chosen = rng.choice(len(obs_idx), size=n_holdout, replace=False)
        held_idx = obs_idx[chosen]

        held_out_values[col] = df.loc[held_idx, col].copy()
        df_masked.loc[held_idx, col] = pd.NA

        new_mm = missing_mask_masked[col].copy()
        new_mm.loc[held_idx] = True
        missing_mask_masked[col] = new_mm

    return df_masked, missing_mask_masked, held_out_values


def evaluate_imputation(
    imputed_df: pd.DataFrame,
    held_out: Dict[str, pd.Series],
    weights: pd.Series,
) -> Dict[str, dict]:
    """Evaluate imputation quality against held-out ground-truth values.

    Parameters
    ----------
    imputed_df : DataFrame containing the imputed column values
    held_out   : {col: Series of true values at held-out indices}
    weights    : sample weights aligned to imputed_df.index

    Returns
    -------
    Dict[str, {"accuracy": float, "tv": float}]
    """
    results: Dict[str, dict] = {}
    for col, true_vals in held_out.items():
        if col not in imputed_df.columns:
            continue
        imp_vals = imputed_df.loc[true_vals.index, col]
        w = weights.reindex(true_vals.index).fillna(1.0)

        match = (imp_vals == true_vals).astype(float)
        weighted_acc = float(np.average(match.values, weights=w.values))

        true_w_counts = true_vals.groupby(true_vals, observed=True).apply(
            lambda s: w.loc[s.index].sum()
        )
        imp_w_counts = imp_vals.groupby(imp_vals, observed=True).apply(
            lambda s: w.loc[s.index].sum()
        )
        all_cats = true_w_counts.index.union(imp_w_counts.index)
        true_prop = true_w_counts.reindex(all_cats, fill_value=0.0) / true_w_counts.sum()
        imp_prop = imp_w_counts.reindex(all_cats, fill_value=0.0) / imp_w_counts.sum()
        tv = float(0.5 * (true_prop - imp_prop).abs().sum())

        results[col] = {"accuracy": weighted_acc, "tv": tv}

    return results


def print_bakeoff_summary(
    metrics: Dict[str, dict],
    label: str,
    *,
    top_n: int = 10,
) -> None:
    """Print a ranked summary table from :func:`evaluate_imputation` output.

    Parameters
    ----------
    metrics : Dict[str, dict]
        Output of :func:`evaluate_imputation` — ``{col: {"accuracy": float, "tv": float}}``.
    label : str
        Header label for the summary block (e.g. the backend name).
    top_n : int, optional
        Number of lowest-accuracy columns to display.  Default ``10``.
    """
    if not metrics:
        print(f"{label}: no metrics.")
        return
    rows = sorted(metrics.items(), key=lambda kv: kv[1]["accuracy"])
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"  mean accuracy: {np.mean([v['accuracy'] for v in metrics.values()]):.4f}")
    print(f"  mean TV dist : {np.mean([v['tv'] for v in metrics.values()]):.4f}")
    print(f"{'='*55}")
    print(f"  {'column':<38}  {'accuracy':>8}  {'tv':>8}")
    print(f"  {'-'*56}")
    for col, vals in rows[:top_n]:
        print(f"  {col:<38}  {vals['accuracy']:>8.4f}  {vals['tv']:>8.4f}")
    if len(rows) > top_n:
        print(f"  ... ({len(rows) - top_n} more columns)")
