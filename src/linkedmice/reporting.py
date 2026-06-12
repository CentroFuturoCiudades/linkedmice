from typing import Dict, List, Tuple

import pandas as pd

from .mice import SkipDep


def missing_report(df: pd.DataFrame, cols: List[str], label: str) -> pd.DataFrame:
    """Print and return a per-column NaN count report.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to inspect.
    cols : List[str]
        Columns to report on (columns absent from ``df`` are silently skipped).
    label : str
        Section header printed above the report.

    Returns
    -------
    pd.DataFrame
        Table with columns ``n_missing`` and ``pct_missing``, indexed by
        column name, sorted by ``n_missing`` descending.
    """
    present = [c for c in cols if c in df.columns]
    missing_n = df[present].isna().sum().sort_values(ascending=False)
    missing_pct = (missing_n / len(df) * 100).round(3)
    report = pd.DataFrame({"n_missing": missing_n, "pct_missing": missing_pct})
    has_missing = report[report["n_missing"] > 0]
    print(f"\n=== {label}: {len(has_missing)}/{len(present)} columns have NaN ===")
    print(report.to_string() if len(report) else "  (none)")
    return report


def missing_report_both(
    df_viv: pd.DataFrame,
    df_per: pd.DataFrame,
    hh_impute_targets: List[str],
    person_impute_targets: List[str],
) -> None:
    """Print missing-value reports for both the household and person tables.

    Parameters
    ----------
    df_viv : pd.DataFrame
        Household DataFrame.
    df_per : pd.DataFrame
        Person DataFrame.
    hh_impute_targets : List[str]
        Household columns to report on.
    person_impute_targets : List[str]
        Person columns to report on.
    """
    hh_miss_report = missing_report(df_viv, hh_impute_targets, "Household targets")
    per_miss_report = missing_report(df_per, person_impute_targets, "Person targets")

    # Columns with actual missing values (will be actual imputation targets)
    hh_cols_with_nan = hh_miss_report[hh_miss_report["n_missing"] > 0].index.tolist()
    per_cols_with_nan = per_miss_report[per_miss_report["n_missing"] > 0].index.tolist()
    print(f"\nHH columns requiring imputation  : {len(hh_cols_with_nan)}")
    print(f"Per columns requiring imputation : {len(per_cols_with_nan)}")


def cross_validate_skip(
    label: str,
    bpp_mask: pd.Series,
    cond_mask: pd.Series,
    n_total: int,
) -> None:
    """Print a forward/backward consistency check for one skip condition.

    Parameters
    ----------
    label : str
        Human-readable name for the condition being checked.
    bpp_mask : pd.Series
        Boolean Series, ``True`` where the column value is BPP.
    cond_mask : pd.Series
        Boolean Series, ``True`` where the expected skip condition is satisfied.
    n_total : int
        Total number of rows (used for percentage formatting).
    """
    bpp_no_cond = (bpp_mask & ~cond_mask).sum()
    cond_no_bpp = (~bpp_mask & cond_mask).sum()
    ok_fwd = "✓" if bpp_no_cond == 0 else "⚠"
    ok_bwd = "✓" if cond_no_bpp == 0 else "⚠"
    print(f"{label}")
    print(
        f"  BPP rows              : {bpp_mask.sum():>8,}  ({bpp_mask.sum() / n_total * 100:.2f}%)"
    )
    print(
        f"  {ok_fwd} BPP w/o condition  : {bpp_no_cond:>8,}  (BPP rows that don't match the expected condition)"
    )
    print(
        f"  {ok_bwd} Condition w/o BPP  : {cond_no_bpp:>8,}  (rows matching condition that aren't BPP)"
    )


def cross_validate_all_skips(
    deps: List[SkipDep],
    df: pd.DataFrame,
    table_label: str,
    bpp_value: str,
) -> bool:
    """Cross-validate BPP ↔ expected skip condition for a list of skip dependencies.

    Two checks are run per dependency:

    - **Forward** (BPP → condition): every BPP row in the child column must
      satisfy the combined parent predicate.
    - **Backward** (condition → BPP): every row satisfying the predicate must
      hold the BPP value in the child column.

    A compact table is printed with one row per dependency.  Failing entries
    are expanded with detail lines.

    Note: the forward check may report failures when a child column has
    multiple independent reasons to be BPP (e.g. a base-skip parent not
    captured in ``deps``).  This is expected and not a bug.

    Parameters
    ----------
    deps : List[SkipDep]
        Skip dependency triples ``(parent_cols, combined_predicate, child_col)``
        in the same format used by :func:`~popsynth.imputer.mice.integrated_mice`.
    df : pd.DataFrame
        DataFrame to validate.
    table_label : str
        Header string identifying the table (household or person).
    bpp_value : str, optional
        BPP sentinel string.

    Returns
    -------
    bool
        ``True`` when all checks pass, ``False`` otherwise.
    """
    col_w = 36
    print(f"\n=== {table_label} — {len(deps)} skip dep(s) ===")
    print(
        f"  {'child column':<{col_w}} {'n_bpp':>8}  {'bpp\\cond':>9}  {'cond\\bpp':>9}  fwd bwd"
    )
    print("  " + "-" * (col_w + 38))

    failures: List[dict] = []

    for parents, pred, child in deps:
        if child not in df.columns:
            print(f"  {'(skipped — ' + child + ' not in df)':<{col_w + 40}}")
            continue

        cond_label = f"{', '.join(parents)} → {child}"

        if hasattr(df[child], "cat") and bpp_value in df[child].cat.categories:
            bpp_mask = df[child] == bpp_value
        else:
            bpp_mask = pd.Series(False, index=df.index)

        cond_mask = pred(df)

        bpp_no_cond = int((bpp_mask & ~cond_mask).sum())
        cond_no_bpp = int((~bpp_mask & cond_mask).sum())
        n_bpp = int(bpp_mask.sum())

        ok_fwd = "✓" if bpp_no_cond == 0 else "⚠"
        ok_bwd = "✓" if cond_no_bpp == 0 else "⚠"
        print(
            f"  {child:<{col_w}} {n_bpp:>8,}  {bpp_no_cond:>9,}  {cond_no_bpp:>9,}   {ok_fwd}   {ok_bwd}"
        )

        if bpp_no_cond > 0 or cond_no_bpp > 0:
            failures.append(
                {
                    "child": child,
                    "label": cond_label,
                    "bpp_no_cond": bpp_no_cond,
                    "cond_no_bpp": cond_no_bpp,
                }
            )

    print()
    if not failures:
        print(f"  All {len(deps)} dep(s) passed.")
        return True

    print(f"  ⚠ {len(failures)} dep(s) failed:")
    for f in failures:
        if f["bpp_no_cond"] > 0:
            print(
                f"    {f['child']}: {f['bpp_no_cond']:,} BPP row(s) don't satisfy '{f['label']}'"
            )
        if f["cond_no_bpp"] > 0:
            print(
                f"    {f['child']}: {f['cond_no_bpp']:,} row(s) satisfy '{f['label']}' but aren't BPP"
            )
    return False


def initial_fill_report(
    hh_work: pd.DataFrame,
    per_work: pd.DataFrame,
    hh_missing_masks: Dict[str, pd.Series],
    person_missing_masks: Dict[str, pd.Series],
    hh_skip_masks: Dict[str, pd.Series],
    person_skip_masks: Dict[str, pd.Series],
    hh_impute_targets: List[str],
    person_impute_targets: List[str],
    bpp_value: str,
) -> None:
    """Verify :func:`initial_fill` output: no unfilled targets, no BPP overwritten.

    Prints a post-fill NaN check, a BPP-preservation check, and total cell
    counts filled for each table.

    Parameters
    ----------
    hh_work : pd.DataFrame
        Household DataFrame after :func:`initial_fill`.
    per_work : pd.DataFrame
        Person DataFrame after :func:`initial_fill`.
    hh_missing_masks : Dict[str, pd.Series]
        Original household NaN masks (before filling).
    person_missing_masks : Dict[str, pd.Series]
        Original person NaN masks.
    hh_skip_masks : Dict[str, pd.Series]
        Household BPP masks.
    person_skip_masks : Dict[str, pd.Series]
        Person BPP masks.
    hh_impute_targets : List[str]
        Household columns that were filled.
    person_impute_targets : List[str]
        Person columns that were filled.
    bpp_value : str, optional
        BPP sentinel string.
    """
    _tables = [
        ("HH ", hh_work, hh_impute_targets, hh_missing_masks, hh_skip_masks),
        (
            "Per",
            per_work,
            person_impute_targets,
            person_missing_masks,
            person_skip_masks,
        ),
    ]

    problems = []
    violations = []
    for label, df, targets, mm, sm in _tables:
        for col in targets:
            if col not in df.columns:
                continue
            still_nan = mm[col] & df[col].isna()
            if still_nan.any():
                problems.append(f"  {label} {col}: {still_nan.sum()} NaN remaining")
            if not hasattr(df[col], "cat"):
                continue
            changed = sm[col] & (df[col] != bpp_value)
            if changed.any():
                violations.append(
                    f"  {label} {col}: {changed.sum()} BPP rows overwritten"
                )

    print("Post-fill NaN check:")
    if problems:
        print("  WARNING — unfilled targets (no observed pool):")
        for p in problems:
            print(p)
    else:
        print("  All target NaN cells filled.")

    print("\nBPP preservation check:")
    if violations:
        print("  ERROR — BPP rows modified by initial_fill:")
        for v in violations:
            print(v)
    else:
        print("  All BPP rows preserved.")

    n_hh = sum(
        hh_missing_masks[c].sum() for c in hh_impute_targets if c in hh_work.columns
    )
    n_per = sum(
        person_missing_masks[c].sum()
        for c in person_impute_targets
        if c in per_work.columns
    )
    print(f"\nTotal cells filled — HH: {n_hh:,}  |  Person: {n_per:,}")


# ── Post-MICE sanity checks ──────────────────────────────────────────────────


def post_mice_sanity_checks(
    hh_df: pd.DataFrame,
    per_df: pd.DataFrame,
    hh_mm_in: Dict[str, pd.Series],
    per_mm_in: Dict[str, pd.Series],
    hh_sm_final: Dict[str, pd.Series],
    per_sm_final: Dict[str, pd.Series],
    hh_impute_targets: List[str],
    person_impute_targets: List[str],
    bpp_value: str,
) -> None:
    """Assert that the MICE loop produced no residual NaN and preserved all BPP rows.

    Two checks are run:

    1. **No residual NaN**: every cell that was originally missing (in the input
       missing masks) must have been filled.  Cells dynamically promoted to skip
       status during the loop are excluded because they are now BPP, not NaN.
    2. **BPP preservation**: every row in the *final* skip masks (which include
       rows promoted dynamically during the loop) must still hold ``bpp_value``.

    Raises ``AssertionError`` on the first violation found.  Prints a one-line
    confirmation for each passing check.

    Parameters
    ----------
    hh_df : pd.DataFrame
        Imputed household DataFrame.
    per_df : pd.DataFrame
        Imputed person DataFrame.
    hh_mm_in : Dict[str, pd.Series]
        Household missing masks as passed *into* :func:`integrated_mice`
        (identifies originally-missing cells).
    per_mm_in : Dict[str, pd.Series]
        Person missing masks as passed into the loop.
    hh_sm_final : Dict[str, pd.Series]
        Household BPP masks *returned* by the loop (includes dynamically promoted rows).
    per_sm_final : Dict[str, pd.Series]
        Person BPP masks returned by the loop.
    hh_impute_targets : List[str]
        Household columns to check.
    person_impute_targets : List[str]
        Person columns to check.
    bpp_value : str
        BPP sentinel string (e.g. ``"Blanco por pase"``).
    """
    for label, df, targets, mm in [
        ("HH", hh_df, hh_impute_targets, hh_mm_in),
        ("Per", per_df, person_impute_targets, per_mm_in),
    ]:
        for col in targets:
            if col not in df.columns:
                continue
            still_nan = mm[col] & df[col].isna()
            assert not still_nan.any(), (
                f"{label} {col}: {int(still_nan.sum())} NaN remaining after imputation"
            )
    print("post_mice_sanity_checks: ✓ No residual NaN in imputed cells")

    for label, df, targets, sm in [
        ("HH", hh_df, hh_impute_targets, hh_sm_final),
        ("Per", per_df, person_impute_targets, per_sm_final),
    ]:
        for col in targets:
            if col not in df.columns or not hasattr(df[col], "cat"):
                continue
            was_bpp = sm[col]
            if not was_bpp.any():
                continue
            assert (df.loc[was_bpp, col] == bpp_value).all(), (
                f"{label} {col}: BPP rows modified"
            )
    print("post_mice_sanity_checks: ✓ All BPP rows preserved")


# ── Post-imputation validation ────────────────────────────────────────────────


def validate_marginals(
    df: pd.DataFrame,
    col: str,
    weights: pd.Series,
    missing_mask: pd.Series,
    skip_mask: pd.Series,
    bpp_value: str,
) -> Tuple[pd.DataFrame, float]:
    """Compare weighted distributions of observed vs. imputed cells for one column.

    Observed pool : ``~missing_mask & ~skip_mask`` (and value != BPP for categoricals).
    Imputed pool  : ``missing_mask`` rows.

    Parameters
    ----------
    df : pd.DataFrame
        Imputed DataFrame.
    col : str
        Column to validate.
    weights : pd.Series
        Sample weights aligned to ``df.index``.
    missing_mask : pd.Series
        Boolean Series, ``True`` for originally-missing (now imputed) rows.
    skip_mask : pd.Series
        Boolean Series, ``True`` for BPP rows.
    bpp_value : str, optional
        BPP sentinel string.
    Returns
    -------
    Tuple[pd.DataFrame, float]
        ``(result_df, tv_distance)`` where ``result_df`` has columns
        ``obs_weighted``, ``imp_weighted``, and ``diff``; and ``tv_distance``
        is the total-variation distance between the two distributions.
    """
    obs_mask = ~missing_mask & ~skip_mask
    if hasattr(df[col], "cat"):
        obs_mask = obs_mask & (df[col] != bpp_value)

    def _weighted_dist(vals: pd.Series, w: pd.Series) -> pd.Series:
        counts = w.groupby(vals, observed=True).sum()
        total = counts.sum()
        return counts / total if total > 0 else pd.Series(dtype=float)

    obs_dist = (
        _weighted_dist(df.loc[obs_mask, col], weights.loc[obs_mask])
        if obs_mask.any()
        else pd.Series(dtype=float)
    )

    imp_vals = df.loc[missing_mask, col]
    if hasattr(imp_vals, "cat"):
        imp_vals = imp_vals[imp_vals != bpp_value]
    imp_dist = (
        _weighted_dist(imp_vals, weights.loc[imp_vals.index])
        if missing_mask.any()
        else pd.Series(dtype=float)
    )

    all_cats = obs_dist.index.union(imp_dist.index)
    result_df = pd.DataFrame(
        {
            "obs_weighted": obs_dist.reindex(all_cats, fill_value=0.0),
            "imp_weighted": imp_dist.reindex(all_cats, fill_value=0.0),
        }
    )
    result_df["diff"] = result_df["imp_weighted"] - result_df["obs_weighted"]
    tv = float(0.5 * result_df["diff"].abs().sum())
    return result_df, tv


def run_validation_report(
    hh_df: pd.DataFrame,
    per_df: pd.DataFrame,
    hh_mm: Dict[str, pd.Series],
    per_mm: Dict[str, pd.Series],
    hh_sm: Dict[str, pd.Series],
    per_sm: Dict[str, pd.Series],
    hh_impute_targets: List[str],
    person_impute_targets: List[str],
    hh_weight_col: str = "FACTOR",
    per_weight_col: str = "FACTOR",
    bpp_value: str = "Blanco por pase",
) -> Dict[str, float]:
    """Run marginal-distribution validation for all imputed targets.

    Computes :func:`validate_marginals` for every column that has at least one
    imputed row, prints a summary table sorted by TV distance descending, and
    flags columns where TV > 0.05.

    Parameters
    ----------
    hh_df : pd.DataFrame
        Imputed household DataFrame.
    per_df : pd.DataFrame
        Imputed person DataFrame.
    hh_mm : Dict[str, pd.Series]
        Household missing masks (identifies originally-missing rows).
    per_mm : Dict[str, pd.Series]
        Person missing masks.
    hh_sm : Dict[str, pd.Series]
        Household BPP masks.
    per_sm : Dict[str, pd.Series]
        Person BPP masks.
    hh_impute_targets : List[str]
        Household columns to validate.
    person_impute_targets : List[str]
        Person columns to validate.
    hh_weight_col : str, optional
        Household sample-weight column name.  Default ``"FACTOR"``.
    per_weight_col : str, optional
        Person sample-weight column name.  Default ``"FACTOR"``.

    Returns
    -------
    Dict[str, float]
        ``{"table_column": tv_distance}`` for every validated column.
    """
    hh_weights = hh_df[hh_weight_col].fillna(1.0).astype(float)
    per_weights = per_df[per_weight_col].fillna(1.0).astype(float)

    rows = []
    for col in hh_impute_targets:
        if col not in hh_mm or col not in hh_df.columns or not hh_mm[col].any():
            continue
        sm = hh_sm.get(col, pd.Series(False, index=hh_df.index))
        _, tv = validate_marginals(hh_df, col, hh_weights, hh_mm[col], sm, bpp_value)
        rows.append(
            {"table": "hh", "column": col, "n_imputed": int(hh_mm[col].sum()), "tv": tv}
        )

    for col in person_impute_targets:
        if col not in per_mm or col not in per_df.columns or not per_mm[col].any():
            continue
        sm = per_sm.get(col, pd.Series(False, index=per_df.index))
        _, tv = validate_marginals(per_df, col, per_weights, per_mm[col], sm, bpp_value)
        rows.append(
            {
                "table": "per",
                "column": col,
                "n_imputed": int(per_mm[col].sum()),
                "tv": tv,
            }
        )

    if not rows:
        print("No imputed columns — nothing to validate.")
        return {}

    summary = pd.DataFrame(rows).sort_values("tv", ascending=False)
    print("\nValidation Report — Marginal TV Distances:")
    print(f"{'table':<6}{'column':<40}{'n_imputed':>10}{'tv_distance':>12}  flag")
    print("-" * 74)
    for _, r in summary.iterrows():
        flag = "⚠ large" if r["tv"] > 0.05 else ""
        print(
            f"{r['table']:<6}{r['column']:<40}{r['n_imputed']:>10}{r['tv']:>12.4f}  {flag}"
        )

    large = summary[summary["tv"] > 0.05]
    msg = (
        "All within acceptable threshold (<= 0.05)."
        if large.empty
        else f"{len(large)} column(s) flagged."
    )
    print(f"\n{msg}")

    return {f"{r['table']}_{r['column']}": r["tv"] for _, r in summary.iterrows()}
