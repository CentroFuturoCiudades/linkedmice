#!/usr/bin/env python3
"""Worked example: Mexico Census 2020 (Jalisco) imputation, 20 MICE iterations, LightGBM.

Requires the census extra: uv sync --extra census

Run from the repo root:
    python scripts/imputation_run_ca.py                   # stochastic (default)
    python scripts/imputation_run_ca.py --deterministic   # argmax imputation
    python scripts/imputation_run_ca.py --data DIR        # local parquets instead of
                                                          # the mxcensus mirror
"""

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from census_mx import (
    load_census_tables,
    rebuild_financiamiento_dummies,
    rebuild_med_traslado_dummies,
    rederive_dhsersal_dummies,
    refresh_hh_agg_features,
    refresh_per_position,
    refresh_per_role,
)
from linkedmice import (
    SkipDep,
    analyze_convergence,
    build_missing_masks,
    build_skip_masks,
    cross_validate_all_skips,
    initial_fill,
    initial_fill_report,
    integrated_mice,
    missing_report_both,
    normalize_categorical_dtypes,
    post_imputation_repair,
    repair_parent_child_nan,
    run_validation_report,
)
from mxcensus import constraints_personas, constraints_viviendas

_parser = argparse.ArgumentParser()
_parser.add_argument(
    "--deterministic",
    action="store_true",
    help="Use argmax (most probable class) instead of stochastic sampling.",
)
_parser.add_argument(
    "--data",
    type=Path,
    default=None,
    help="Local dir with Viviendas14.parquet/Personas14.parquet; "
    "omit to fetch Jalisco (state=14) from the mxcensus mirror.",
)
_args = _parser.parse_args()
DETERMINISTIC: bool = _args.deterministic
DATA: Path | None = _args.data

BPP = "Blanco por pase"
ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

_SUFFIX = "_deterministic" if DETERMINISTIC else ""

RANDOM_SEED = 42
pd.set_option("display.max_columns", None)

# ── Column configuration ──────────────────────────────────────────────────────

# Columns to ignore for imputation, some are aggregated from other table
# Raw source columns for *_CAT derived columns are excluded so their NaN values
# do not add noise to other columns' imputation models; the binned _CAT version
# is the only form used as a predictor and target.
col_ignore_viv = [
    "DUE1_NUM", "DUE2_NUM", "ENT", "JEFE_EDAD",
    # raw sources for _CAT columns (CLAVIVP kept — no missing values, strong predictor)
    "CUADORM", "TOTCUART", "DRENAJE",
    # raw numeric income — INGTRHOG_CAT is imputed instead
    "INGTRHOG",
]
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
    # raw sources for _CAT columns
    "EDAD", "CONACT", "SITUA_CONYUGAL", "RELIGION", "ENT_PAIS_RES_5A", "ENT_PAIS_NAC",
    # high-cardinality columns with no predictive value
    "FECHA_NAC_A", "NOMCAR_C",
]

# Columns to transform into numerical (integer)
# After replacing "No especificado" with NaN
num_cols_viv = []
num_cols_per = [
    "ESCOACUM",
    "HIJOS_NAC_VIVOS",
    "HIJOS_FALLECIDOS",
    "HIJOS_SOBREVIV",
]

# Columns that are actually categorical but are coded as integer
cat_cols_viv = ["MUN", "LOC50K", "UPM"]
cat_cols_per = ["MUN", "LOC50K", "UPM"]

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

# Remove unused categories (e.g. "No especificado" replaced by NaN above)
for col in df_viv.select_dtypes("category").columns:
    df_viv[col] = df_viv[col].cat.remove_unused_categories()
for col in df_per.select_dtypes("category").columns:
    df_per[col] = df_per[col].cat.remove_unused_categories()

# Remove one hot encoded columns and deterministically derived columns
df_viv = df_viv.drop(columns=[c for c in df_viv.columns if "FINANCIAMIENTO_" in c])
df_per = df_per.drop(columns=[c for c in df_per.columns if "MED_TRASLADO_ESC_" in c])
df_per = df_per.drop(columns=[c for c in df_per.columns if "MED_TRASLADO_TRAB_" in c])
df_per = df_per.drop(columns=[c for c in df_per.columns if c.startswith("DHSERSAL_")])

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

# Household imputation targets: all synthesis columns from constraints, plus
# INGTRHOG_CAT which is broadcast to the person table and must not have NaN.
hh_impute_targets: List[str] = sorted(viv_cols | {"INGTRHOG_CAT"})

# ── Person imputation targets:
# DHSERSAL_* dummies are derived post-loop; impute their source columns instead.
# EDUC is imputed directly (eliminates the NIVACAD–ESCOLARI feedback loop).
# All disability columns (source and derived) are excluded — imputation is
# too uncertain given the sparsity and multi-item structure; original observed
# values (including "No especificado" for missings) are kept as-is.
dhsersal_derived = {c for c in per_cols if c.startswith("DHSERSAL_")}
dis_excluded = {
    "DIS_CON", "DIS_LIMI",
    "DIS_VER", "DIS_OIR", "DIS_CAMINAR", "DIS_RECORDAR",
    "DIS_BANARSE", "DIS_HABLAR", "DIS_MENTAL",
}
to_add = {"DHSERSAL1"}
person_impute_targets: List[str] = sorted(
    (per_cols - dhsersal_derived - dis_excluded) | to_add
)

print(f"Household imputation targets  : {len(hh_impute_targets)}")
print(f"Person  imputation targets    : {len(person_impute_targets)}")
print(f"Household targets : {hh_impute_targets}")
print(f"Person targets    : {person_impute_targets}")
print(f"Derived cols removed : {dhsersal_derived | dis_excluded}")
print(f"Source cols added    : {to_add}")
assert set(person_impute_targets).issubset(set(df_per.columns))
assert set(hh_impute_targets).issubset(set(df_viv.columns))

# ── Skip dependency definitions ───────────────────────────────────────────────

_AGE_0_4 = ["0-2", "3-4"]
_AGE_0_11 = ["0-2", "3-4", "5", "6-7", "8-11"]

_CLAVIVP_SKIP_COLS: List[str] = [
    c
    for c in hh_impute_targets
    if c not in {"CLAVIVP_CAT", "JEFE_SEXO", "ABA_AGUA_ENTU", "CONAGUA", "INGTRHOG_CAT"}
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
    (["EDAD_CAT"], lambda df: df["EDAD_CAT"] == "0-2", "EDUC"),
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
print(
    f"HH  skip dependencies : {len(HH_SKIP_DEPS)}  ({len(_CLAVIVP_SKIP_COLS)} base + 2 compound)"
)
print(f"Per skip dependencies : {len(PER_SKIP_DEPS)}")

# Make sure all impute targets with BPP have skip dependencies defined
assert [
    c
    for c in person_impute_targets
    if BPP in df_per[c].cat.categories and c not in [x for _, _, x in PER_SKIP_DEPS]
] == []
# INGTRHOG_CAT BPP rows reflect survey sampling design (income question not
# administered to all households), not a logical questionnaire skip — no
# skip dep can be declared, but build_skip_masks handles BPP correctly.
assert [
    c
    for c in hh_impute_targets
    if BPP in df_viv[c].cat.categories
    and c not in [x for _, _, x in HH_SKIP_DEPS]
] == ["INGTRHOG_CAT"]

# ── Loop configuration ────────────────────────────────────────────────────────

# ── High-priority targets: appear in many constraints & have strong
#    within-household correlation → get per-variable LOO refresh ─────────────
person_high_priority_targets: List[str] = [
    "EDUC",
    "CONACT_CAT",
    "SITUA_CONYUGAL_CAT",
]

# ── Household columns to broadcast into the person table ─────────────────────
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
person_missing_masks = build_missing_masks(
    df_per, person_impute_targets, person_skip_masks
)
hh_missing_masks = build_missing_masks(df_viv, hh_impute_targets, hh_skip_masks)

hh_work = initial_fill(df_viv, hh_impute_targets, hh_missing_masks, hh_skip_masks, rng)
per_work = initial_fill(
    df_per, person_impute_targets, person_missing_masks, person_skip_masks, rng
)

initial_fill_report(
    hh_work,
    per_work,
    hh_missing_masks,
    person_missing_masks,
    hh_skip_masks,
    person_skip_masks,
    hh_impute_targets,
    person_impute_targets,
    BPP,
)

# ── Full MICE Run ─────────────────────────────────────────────────────
(
    hh_imputed,
    per_imputed,
    hh_mm_mice,
    per_mm_mice,
    hh_sm_mice,
    per_sm_mice,
    full_diag,
) = integrated_mice(
    hh_df=hh_work,
    per_df=per_work,
    hh_missing_mask_in=hh_missing_masks,
    person_missing_mask_in=person_missing_masks,
    hh_skip_masks_in=hh_skip_masks,
    person_skip_masks_in=person_skip_masks,
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
    n_iterations=20,
    backend="lightgbm",
    backend_params={"num_threads": 16},
    rng=np.random.default_rng(RANDOM_SEED),
    deterministic=DETERMINISTIC,
    verbose=True,
)

print(f"\nhh_imputed  shape: {hh_imputed.shape}")
print(f"per_imputed shape: {per_imputed.shape}")

# ── Convergence diagnostics ───────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Convergence diagnostics")
print("=" * 60)

conv_df = analyze_convergence(full_diag, threshold=0.01)
conv_path = OUTPUTS / f"run_convergence{_SUFFIX}.csv"
conv_df.to_csv(conv_path, index=False)
print(f"\nConvergence table saved to {conv_path}")

# ── Post-Imputation Consistency Repair─────────────────────────────────────────

hh_imputed, per_imputed, hh_mm_final, per_mm_final, hh_sm_final, per_sm_final = (
    post_imputation_repair(
        hh_imputed,
        per_imputed,
        hh_mm_mice,
        per_mm_mice,
        hh_sm_mice,
        per_sm_mice,
        hh_skip_deps=HH_SKIP_DEPS,
        per_skip_deps=PER_SKIP_DEPS,
    )
)

# ── Validate unchanged cells ──────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Validating originally-observed cells are unchanged")
print("=" * 60)

for label, orig_df, imputed_df, masks in [
    ("HH", df_viv, hh_imputed, hh_missing_masks),
    ("Per", df_per, per_imputed, person_missing_masks),
]:
    total_warned = 0
    for col, mm in masks.items():
        if col not in orig_df.columns or col not in imputed_df.columns:
            continue
        observed = ~mm & orig_df[col].notna()
        changed = int(
            (orig_df.loc[observed, col] != imputed_df.loc[observed, col]).sum()
        )
        if changed > 0:
            print(
                f"  WARNING [{label}] {col}: {changed:,} originally-observed cells changed"
            )
            total_warned += changed
    if total_warned == 0:
        print(f"  [{label}] ✓ No originally-observed cells changed")

# ── INGTRHOG_CAT consistency check & repair ──────────────────────────────────
# For imputed rows, the bin's upper bound must be ≥ sum of observed (non-NaN)
# INGTRMEN in the household. INGTRHOG = sum(INGTRMEN, workers 12+), so the
# observed partial sum is a hard lower bound on the true household income.
# Violations are corrected deterministically: override the imputed category
# with the bin that contains the observed INGTRMEN sum.

print("\n" + "=" * 60)
print("INGTRHOG_CAT consistency check & repair")
print("=" * 60)

_INGTRHOG_CAT_UPPER = {
    "No recibe ingresos":    1,
    "1-999":              1_000,
    "1,000-4,999":        5_000,
    "5,000-9,999":       10_000,
    "10,000-19,999":     20_000,
    "20,000-39,999":     40_000,
    "40,000-79,999":     80_000,
    "80,000-149,999":   150_000,
    "150,000yMas":    9_999_999,
}
_INGTRHOG_SORTED = sorted(_INGTRHOG_CAT_UPPER.items(), key=lambda x: x[1])

def _income_to_cat(income: float) -> str:
    for cat, upper in _INGTRHOG_SORTED:
        if income <= upper:
            return cat
    return "150,000yMas"

_ingtrmen = pd.to_numeric(df_per["INGTRMEN"], errors="coerce")
_hh_obs_sum = _ingtrmen.groupby(df_per.index.get_level_values(0)).sum()

_imputed_ids = hh_imputed.index[hh_missing_masks["INGTRHOG_CAT"]]
_check = pd.DataFrame({
    "cat":     hh_imputed.loc[_imputed_ids, "INGTRHOG_CAT"].astype(str),
    "obs_sum": _hh_obs_sum.reindex(_imputed_ids, fill_value=0),
})
_check["upper"] = _check["cat"].map(_INGTRHOG_CAT_UPPER)
_check["violation"] = _check["upper"] < _check["obs_sum"]

_n_viol = int(_check["violation"].sum())
print(f"  Imputed INGTRHOG_CAT rows : {len(_imputed_ids):,}")
if _n_viol == 0:
    print("  ✓ All imputed bins consistent with observed INGTRMEN sums")
else:
    print(f"  ✗ {_n_viol} violation(s): imputed bin upper bound < observed INGTRMEN sum")
    print(_check[_check["violation"]][["cat", "obs_sum", "upper"]].to_string())

    _viol_ids = _check.index[_check["violation"]]
    _repair_log = pd.DataFrame({
        "was":     _check.loc[_viol_ids, "cat"],
        "obs_sum": _check.loc[_viol_ids, "obs_sum"],
        "now":     [_income_to_cat(s) for s in _check.loc[_viol_ids, "obs_sum"]],
    })
    for hh_id, row in _repair_log.iterrows():
        hh_imputed.at[hh_id, "INGTRHOG_CAT"] = row["now"]

    print(f"\n  Repaired {_n_viol} violation(s):")
    print(_repair_log.to_string())

del _ingtrmen, _hh_obs_sum, _imputed_ids, _check, _n_viol

# ── Marginal Validation─────────────────────────────────────────
hh_tv = run_validation_report(
    hh_imputed,
    per_imputed,
    hh_mm_final,
    per_mm_final,
    hh_sm_final,
    per_sm_final,
    hh_impute_targets,
    person_impute_targets,
)

val_path = OUTPUTS / f"run_validation{_SUFFIX}.csv"
pd.DataFrame(list(hh_tv.items()), columns=["column", "tv_distance"]).to_csv(
    val_path, index=False
)
print(f"\nValidation TV table saved to {val_path}")

# ── Hand-Off: Re-derive Dummies & Export────────────────────────────────────

# Re-derive DHSERSAL_* dummies from imputed source columns
per_imputed = rederive_dhsersal_dummies(per_imputed)
print(
    "Re-derived DHSERSAL columns:",
    sorted(c for c in per_imputed.columns if c.startswith("DHSERSAL_")),
)

# Rebuild MED_TRASLADO_* from (imputed) source columns
# EDUC is now imputed directly — no rebuild needed
# DIS_* columns carry original observed values — no rebuild needed
per_imputed = rebuild_med_traslado_dummies(per_imputed)
print("Rebuilt MED_TRASLADO_* columns.")

# Rebuild FINANCIAMIENTO_* dummies
hh_imputed = rebuild_financiamiento_dummies(hh_imputed)
print(
    "Rebuilt FINANCIAMIENTO columns:",
    sorted(c for c in hh_imputed.columns if c.startswith("FINANCIAMIENTO_")),
)

# ── Summary JSON ──────────────────────────────────────────────────────────────

hh_tv_vals = [v for k, v in hh_tv.items() if k.startswith("hh_")]
per_tv_vals = [v for k, v in hh_tv.items() if k.startswith("per_")]
summary = {
    "n_hh": int(hh_imputed.shape[0]),
    "n_persons": int(per_imputed.shape[0]),
    "n_iterations": 20,
    "backend": "lightgbm",
    "deterministic": DETERMINISTIC,
    "hh_impute_targets": len(hh_impute_targets),
    "per_impute_targets": len(person_impute_targets),
    "mean_tv_hh": float(sum(hh_tv_vals) / len(hh_tv_vals)) if hh_tv_vals else None,
    "mean_tv_per": float(sum(per_tv_vals) / len(per_tv_vals)) if per_tv_vals else None,
}
summary_path = OUTPUTS / f"run_summary{_SUFFIX}.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary JSON saved to {summary_path}")

# Export — normalize mixed int/str categoricals before pyarrow serialization
hh_imputed = normalize_categorical_dtypes(hh_imputed)
per_imputed = normalize_categorical_dtypes(per_imputed)

hh_imputed.to_parquet(OUTPUTS / f"Viviendas14_imputed{_SUFFIX}.parquet")
per_imputed.to_parquet(OUTPUTS / f"Personas14_imputed{_SUFFIX}.parquet")
print(f"Saved imputed tables to {OUTPUTS}")
print(f"  Viviendas14_imputed.parquet : {hh_imputed.shape}")
print(f"  Personas14_imputed.parquet  : {per_imputed.shape}")
