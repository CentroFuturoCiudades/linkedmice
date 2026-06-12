#!/usr/bin/env python3
"""Worked-example pilot: 10% subsample, 2 MICE iterations. LightGBM vs XGBoost bake-off.

Requires the census and xgboost extras: uv sync --extra census --extra xgboost

Run from the repo root:
    python scripts/imputation_pilot.py             # fetch Jalisco data from the
                                                   # mxcensus mirror
    python scripts/imputation_pilot.py --data DIR  # local parquets

Outputs written to outputs/:
    pilot_convergence.csv   TV distances between iterations per variable
    pilot_bakeoff.csv       Weighted accuracy and TV distance per column per backend
    pilot_bakeoff_summary.json  Mean accuracy / TV across all columns per backend
"""

import argparse
import json
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from census_mx import (
    load_census_tables,
    refresh_hh_agg_features,
    refresh_per_position,
    refresh_per_role,
)
from linkedmice import (
    SkipDep,
    analyze_convergence,
    build_missing_masks,
    build_skip_masks,
    create_validation_mask,
    cross_validate_all_skips,
    evaluate_imputation,
    initial_fill,
    initial_fill_report,
    integrated_mice,
    missing_report_both,
    print_bakeoff_summary,
    repair_parent_child_nan,
)
from mxcensus import constraints_personas, constraints_viviendas

_parser = argparse.ArgumentParser()
_parser.add_argument(
    "--data",
    type=Path,
    default=None,
    help="Local dir with Viviendas14.parquet/Personas14.parquet; "
    "omit to fetch Jalisco (state=14) from the mxcensus mirror.",
)
_args = _parser.parse_args()
DATA: Path | None = _args.data

BPP = "Blanco por pase"
ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

RANDOM_SEED = 42
pd.set_option("display.max_columns", None)

# ── Column configuration ──────────────────────────────────────────────────────

col_ignore_viv = ["DUE1_NUM", "DUE2_NUM", "ENT", "JEFE_EDAD"]
col_ignore_per = [
    "ENT",
    "IDENT_MADRE",
    "IDENT_PADRE",
    "IDENT_PAREJA",
    "IDENT_HIJO",
    "EDAD_MORIR_D",
    "EDAD_MORIR_M",
    "EDAD_MORIR_A",
    "EDAD_MORIR_TD",
    "FECHA_NAC_M",
    "QDIALECT_INALI",
    "NUMPER",
    "CAU_VER",
    "CAU_OIR",
    "CAU_CAMINAR",
    "CAU_RECORDAR",
    "CAU_BANARSE",
    "CAU_HABLAR",
    "CAU_MENTAL",
]
num_cols_viv = ["CUADORM", "TOTCUART"]
num_cols_per = [
    "FECHA_NAC_A",
    "ESCOACUM",
    "HIJOS_NAC_VIVOS",
    "HIJOS_FALLECIDOS",
    "HIJOS_SOBREVIV",
]
cat_cols_viv = ["MUN", "LOC50K", "UPM"]
cat_cols_per = ["MUN", "LOC50K", "UPM", "ENT_PAIS_NAC"]

# ── Data loading ──────────────────────────────────────────────────────────────

print("Loading data...")
df_viv, df_per = load_census_tables(DATA, state=14)
df_viv = df_viv.drop(columns=col_ignore_viv).replace("No especificado", np.nan)
df_per = df_per.drop(columns=col_ignore_per).replace("No especificado", np.nan)

for col in num_cols_viv:
    df_viv[col] = df_viv[col].astype("Int64")
for col in num_cols_per:
    df_per[col] = df_per[col].astype("Int64")

for col in cat_cols_viv:
    df_viv[col] = df_viv[col].astype("category")
for col in cat_cols_per:
    df_per[col] = df_per[col].astype("category")

for col in df_viv.select_dtypes("category").columns:
    df_viv[col] = df_viv[col].cat.remove_unused_categories()
for col in df_per.select_dtypes("category").columns:
    df_per[col] = df_per[col].cat.remove_unused_categories()

df_viv = df_viv.drop(columns=[c for c in df_viv.columns if "FINANCIAMIENTO_" in c])
df_per = df_per.drop(columns=[c for c in df_per.columns if "MED_TRASLADO_ESC_" in c])
df_per = df_per.drop(columns=[c for c in df_per.columns if "MED_TRASLADO_TRAB_" in c])
df_per = df_per.drop(columns=[c for c in df_per.columns if c.startswith("DHSERSAL_")])
df_per = df_per.drop(columns=["DIS_CON", "DIS_LIMI", "EDUC"])

print(f"Loaded {len(df_viv):,} households and {len(df_per):,} persons.")

# ── Variable inventory ────────────────────────────────────────────────────────

constraints_viv = constraints_viviendas()
constraints_per = constraints_personas()

per_cols: set = set()
for v in constraints_per.values():
    per_cols = per_cols.union(v)
viv_cols: set = set()
for v in constraints_viv.values():
    viv_cols = viv_cols.union(v)

hh_impute_targets: List[str] = sorted(viv_cols)

dhsersal_derived = {c for c in per_cols if c.startswith("DHSERSAL_")}
educ_derived = {"EDUC"}
dis_derived = {"DIS_CON", "DIS_LIMI"}
to_add = {"DHSERSAL1", "DHSERSAL2", "NIVACAD", "ESCOLARI"}
person_impute_targets: List[str] = sorted(
    (per_cols - dhsersal_derived - educ_derived - dis_derived) | to_add
)

print(f"Household imputation targets : {len(hh_impute_targets)}")
print(f"Person  imputation targets   : {len(person_impute_targets)}")
assert set(person_impute_targets).issubset(set(df_per.columns))
assert set(hh_impute_targets).issubset(set(df_viv.columns))

# ── Skip dependency definitions ───────────────────────────────────────────────

_AGE_0_4 = ["0-2", "3-4"]
_AGE_0_11 = ["0-2", "3-4", "5", "6-7", "8-11"]

_CLAVIVP_SKIP_COLS: List[str] = [
    c
    for c in hh_impute_targets
    if c not in {"CLAVIVP_CAT", "JEFE_SEXO", "ABA_AGUA_ENTU", "CONAGUA"}
]

HH_SKIP_DEPS: List[SkipDep] = [
    *[
        (["CLAVIVP_CAT"], lambda df: df["CLAVIVP_CAT"] == "Otro", child)
        for child in _CLAVIVP_SKIP_COLS
    ],
    (
        ["CLAVIVP_CAT", "AGUA_ENTUBADA"],
        lambda df: (df["CLAVIVP_CAT"] == "Otro") | (df["AGUA_ENTUBADA"] == "No tiene"),
        "ABA_AGUA_ENTU",
    ),
    (
        ["CLAVIVP_CAT", "SERSAN"],
        lambda df: (
            (df["CLAVIVP_CAT"] == "Otro")
            | (df["SERSAN"] == "No tienen taza de baño ni letrina.")
        ),
        "CONAGUA",
    ),
]

PER_SKIP_DEPS: List[SkipDep] = [
    (["EDAD_CAT"], lambda df: df["EDAD_CAT"] == "0-2", "HLENGUA"),
    (["EDAD_CAT"], lambda df: df["EDAD_CAT"] == "0-2", "ESCOLARI"),
    (["EDAD_CAT"], lambda df: df["EDAD_CAT"] == "0-2", "NIVACAD"),
    (["EDAD_CAT"], lambda df: df["EDAD_CAT"] == "0-2", "ASISTEN"),
    (["EDAD_CAT"], lambda df: df["EDAD_CAT"].isin(_AGE_0_4), "ALFABET"),
    (["EDAD_CAT"], lambda df: df["EDAD_CAT"].isin(_AGE_0_4), "ENT_PAIS_RES_CAT"),
    (["EDAD_CAT"], lambda df: df["EDAD_CAT"].isin(_AGE_0_11), "CONACT_CAT"),
    (["EDAD_CAT"], lambda df: df["EDAD_CAT"].isin(_AGE_0_11), "SITUA_CONYUGAL_CAT"),
    (
        ["HLENGUA", "EDAD_CAT"],
        lambda df: (df["HLENGUA"] == "No") | (df["EDAD_CAT"] == "0-2"),
        "HESPANOL",
    ),
]

# ── Loop configuration ────────────────────────────────────────────────────────

person_high_priority_targets: List[str] = [
    "ESCOLARI",
    "NIVACAD",
    "CONACT_CAT",
    "SITUA_CONYUGAL_CAT",
]

HH_BROADCAST_COLS: List[str] = [
    "INGTRHOG_CAT",
    "TIPOHOG",
    "CLAVIVP_CAT",
    "NUMPERS",
    "COMBUSTIBLE",
    "ALIMENTACION",
    "MCONMIG",
    "TENENCIA",
    "SERSAN",
    "DRENAJE_CAT",
]

_HH_EXCLUDE = {"FACTOR"}
_PER_EXCLUDE = {"FACTOR"}

# ── Initialization ────────────────────────────────────────────────────────────

rng = np.random.default_rng(RANDOM_SEED)

df_viv = repair_parent_child_nan(df_viv, HH_SKIP_DEPS, BPP)
df_per = repair_parent_child_nan(df_per, PER_SKIP_DEPS, BPP)

missing_report_both(df_viv, df_per, hh_impute_targets, person_impute_targets)

assert cross_validate_all_skips(HH_SKIP_DEPS, df_viv, "Household table", BPP)
assert cross_validate_all_skips(PER_SKIP_DEPS, df_per, "Person table", BPP)

person_skip_masks = build_skip_masks(df_per, person_impute_targets, BPP)
hh_skip_masks = build_skip_masks(df_viv, hh_impute_targets, BPP)
person_missing_masks = build_missing_masks(df_per, person_impute_targets, person_skip_masks)
hh_missing_masks = build_missing_masks(df_viv, hh_impute_targets, hh_skip_masks)

hh_work = initial_fill(df_viv, hh_impute_targets, hh_missing_masks, hh_skip_masks, rng)
per_work = initial_fill(df_per, person_impute_targets, person_missing_masks, person_skip_masks, rng)

initial_fill_report(
    hh_work, per_work,
    hh_missing_masks, person_missing_masks,
    hh_skip_masks, person_skip_masks,
    hh_impute_targets, person_impute_targets,
    BPP,
)

# ── Pilot subsample (10%) ─────────────────────────────────────────────────────

_rng_pilot = np.random.default_rng(99)
_all_hh_ids = hh_work.index.values
_n_pilot = max(1, int(len(_all_hh_ids) * 0.10))
_pilot_hh_ids = set(_rng_pilot.choice(_all_hh_ids, size=_n_pilot, replace=False))

hh_pilot = hh_work.loc[sorted(_pilot_hh_ids)].copy()
_pilot_per_mask = per_work.index.get_level_values("ID_VIV").isin(_pilot_hh_ids)
per_pilot = per_work.loc[_pilot_per_mask].copy()

hh_mm_pilot = {
    c: hh_missing_masks[c].loc[hh_pilot.index]
    for c in hh_impute_targets
    if c in hh_work.columns
}
per_mm_pilot = {
    c: person_missing_masks[c].loc[per_pilot.index]
    for c in person_impute_targets
    if c in per_work.columns
}
hh_sm_pilot = {
    c: hh_skip_masks[c].loc[hh_pilot.index]
    for c in hh_impute_targets
    if c in hh_work.columns
}
per_sm_pilot = {
    c: person_skip_masks[c].loc[per_pilot.index]
    for c in person_impute_targets
    if c in per_work.columns
}

print(f"\nPilot subsample: {len(hh_pilot):,} households  |  {len(per_pilot):,} persons")
print(
    f"Missing cells — HH: {sum(v.sum() for v in hh_mm_pilot.values()):,}  "
    f"Per: {sum(v.sum() for v in per_mm_pilot.values()):,}"
)

# ── Pilot run (2 iterations, lightgbm) ───────────────────────────────────────

print("\n" + "=" * 60)
print("Pilot run: 2 iterations, LightGBM")
print("=" * 60)

(
    hh_pilot_out,
    per_pilot_out,
    hh_mm_pilot_out,
    per_mm_pilot_out,
    hh_sm_pilot_out,
    per_sm_pilot_out,
    pilot_diag,
) = integrated_mice(
    hh_df=hh_pilot,
    per_df=per_pilot,
    hh_missing_mask_in=hh_mm_pilot,
    person_missing_mask_in=per_mm_pilot,
    hh_skip_masks_in=hh_sm_pilot,
    person_skip_masks_in=per_sm_pilot,
    hh_impute_targets=hh_impute_targets,
    person_impute_targets=person_impute_targets,
    hh_exclude=_HH_EXCLUDE,
    per_exclude=_PER_EXCLUDE,
    hh_skip_deps=HH_SKIP_DEPS,
    per_skip_deps=PER_SKIP_DEPS,
    person_high_priority_targets=person_high_priority_targets,
    hh_broadcast_cols=HH_BROADCAST_COLS,
    hh_feature_hooks=[refresh_hh_agg_features],
    per_feature_hooks=[refresh_per_role, refresh_per_position],
    n_iterations=2,
    backend="lightgbm",
    rng=np.random.default_rng(RANDOM_SEED),
    verbose=True,
)

# ── Convergence diagnostics ───────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Convergence diagnostics")
print("=" * 60)

conv_df = analyze_convergence(pilot_diag, threshold=0.01)
conv_path = OUTPUTS / "pilot_convergence.csv"
conv_df.reset_index().to_csv(conv_path, index=False)
print(f"\nConvergence table saved to {conv_path}")

# ── Backend bake-off: LightGBM vs XGBoost ────────────────────────────────────

print("\n" + "=" * 60)
print("Backend bake-off: LightGBM vs XGBoost (3 iterations, 5% hold-out)")
print("=" * 60)

hh_masked, hh_mm_masked, hh_held = create_validation_mask(
    hh_pilot,
    hh_impute_targets,
    hh_mm_pilot,
    hh_sm_pilot,
    frac=0.05,
    rng=np.random.default_rng(RANDOM_SEED),
)
per_masked, per_mm_masked, per_held = create_validation_mask(
    per_pilot,
    person_impute_targets,
    per_mm_pilot,
    per_sm_pilot,
    frac=0.05,
    rng=np.random.default_rng(RANDOM_SEED + 1),
)

bakeoff_rows = []
bakeoff_summary = {}

for _backend in ["lightgbm", "xgboost"]:
    print(f"\n{'=' * 50}\nBackend: {_backend}\n{'=' * 50}")
    _t0 = time.perf_counter()
    (
        _hh_out, _per_out,
        _hh_mm_out, _per_mm_out,
        _hh_sm_out, _per_sm_out,
        _diag,
    ) = integrated_mice(
        hh_df=hh_masked,
        per_df=per_masked,
        hh_missing_mask_in=hh_mm_masked,
        person_missing_mask_in=per_mm_masked,
        hh_skip_masks_in=hh_sm_pilot,
        person_skip_masks_in=per_sm_pilot,
        hh_impute_targets=hh_impute_targets,
        person_impute_targets=person_impute_targets,
        hh_exclude=_HH_EXCLUDE,
        per_exclude=_PER_EXCLUDE,
        hh_skip_deps=HH_SKIP_DEPS,
        per_skip_deps=PER_SKIP_DEPS,
        person_high_priority_targets=person_high_priority_targets,
        hh_broadcast_cols=HH_BROADCAST_COLS,
        hh_feature_hooks=[refresh_hh_agg_features],
        per_feature_hooks=[refresh_per_role, refresh_per_position],
        n_iterations=3,
        backend=_backend,
        rng=np.random.default_rng(RANDOM_SEED),
        verbose=False,
    )
    _elapsed = time.perf_counter() - _t0
    print(f"Backend {_backend} finished in {_elapsed:.1f}s ({_elapsed/60:.1f} min)")
    hh_metrics = evaluate_imputation(_hh_out, hh_held, _hh_out["FACTOR"])
    per_metrics = evaluate_imputation(_per_out, per_held, _per_out["FACTOR"])

    print_bakeoff_summary(hh_metrics, f"{_backend} — HH targets")
    print_bakeoff_summary(per_metrics, f"{_backend} — Person targets")

    for col, m in hh_metrics.items():
        bakeoff_rows.append({"backend": _backend, "table": "hh", "column": col, **m})
    for col, m in per_metrics.items():
        bakeoff_rows.append({"backend": _backend, "table": "per", "column": col, **m})

    bakeoff_summary[_backend] = {
        "hh_acc": float(np.mean([v["accuracy"] for v in hh_metrics.values()])),
        "per_acc": float(np.mean([v["accuracy"] for v in per_metrics.values()])),
        "hh_tv": float(np.mean([v["tv"] for v in hh_metrics.values()])),
        "per_tv": float(np.mean([v["tv"] for v in per_metrics.values()])),
        "elapsed_s": round(_elapsed, 1),
    }

# ── Summary and export ────────────────────────────────────────────────────────

print("\n=== Bake-off summary ===")
print(f"{'backend':<12}  {'hh_acc':>8}  {'per_acc':>8}  {'hh_tv':>8}  {'per_tv':>8}  {'time (s)':>10}")
for _be, s in bakeoff_summary.items():
    print(
        f"{_be:<12}  {s['hh_acc']:>8.4f}  {s['per_acc']:>8.4f}"
        f"  {s['hh_tv']:>8.4f}  {s['per_tv']:>8.4f}  {s['elapsed_s']:>10.1f}"
    )

bakeoff_df = pd.DataFrame(bakeoff_rows)
bakeoff_csv = OUTPUTS / "pilot_bakeoff.csv"
bakeoff_df.to_csv(bakeoff_csv, index=False)
print(f"\nPer-column bakeoff metrics saved to {bakeoff_csv}")

summary_json = OUTPUTS / "pilot_bakeoff_summary.json"
summary_json.write_text(json.dumps(bakeoff_summary, indent=2))
print(f"Bakeoff summary saved to {summary_json}")
