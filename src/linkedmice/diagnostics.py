from typing import List

import pandas as pd


def analyze_convergence(diagnostics_list: List[dict], threshold: float = 0.01) -> pd.DataFrame:
    """Compute TV distance between consecutive MICE iteration snapshots.

    Each dict in diagnostics_list is produced by integrated_mice and contains
    keys like 'hh_ABA_AGUA_ENTU' or 'per_CONACT_CAT' (prefix is exactly 'hh_'
    or 'per_'; remainder is the column name).

    Parameters
    ----------
    diagnostics_list : list of dicts from integrated_mice
    threshold        : TV distance above which a variable is flagged as not converged

    Returns
    -------
    pd.DataFrame with columns: iteration, table, column, tv_distance
    """
    records = []
    for i in range(1, len(diagnostics_list)):
        prev = diagnostics_list[i - 1]
        curr = diagnostics_list[i]
        iter_label = f"{i}->{i + 1}"

        keys = [k for k in curr if k.startswith("hh_") or k.startswith("per_")]
        for key in keys:
            if key.startswith("hh_"):
                table, col = "hh", key[3:]
            else:
                table, col = "per", key[4:]

            dist_curr = curr.get(key, {})
            dist_prev = prev.get(key, {})
            if not isinstance(dist_curr, dict) or not isinstance(dist_prev, dict):
                continue

            all_cats = set(dist_curr) | set(dist_prev)
            tv = 0.5 * sum(
                abs(dist_curr.get(c, 0.0) - dist_prev.get(c, 0.0)) for c in all_cats
            )
            records.append({"iteration": iter_label, "table": table, "column": col, "tv_distance": tv})

    if not records:
        print("No consecutive iteration pairs found in diagnostics_list.")
        return pd.DataFrame(columns=["iteration", "table", "column", "tv_distance"])

    df_conv = pd.DataFrame(records)

    pivot = df_conv.pivot_table(
        index=["table", "column"],
        columns="iteration",
        values="tv_distance",
        aggfunc="first",
    )
    pivot["max_tv"] = pivot.max(axis=1)
    pivot = pivot.sort_values("max_tv", ascending=False)

    print("\nConvergence TV distances (sorted by max TV across iterations):")
    print(pivot.to_string())

    last_iter = df_conv["iteration"].iloc[-1]
    not_converged = df_conv[
        (df_conv["iteration"] == last_iter) & (df_conv["tv_distance"] > threshold)
    ]
    if not_converged.empty:
        print(f"\nAll variables converged (TV <= {threshold}) in the last iteration gap.")
    else:
        print(f"\nVariables NOT converged (TV > {threshold}) in last gap ({last_iter}):")
        print(not_converged[["table", "column", "tv_distance"]].to_string(index=False))

    return df_conv
