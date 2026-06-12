# linkedmice — Design & Usage Guide

Survey-weighted two-table MICE imputation for linked household–person data.
The worked example throughout this guide is the Mexico Census 2020 extended
questionnaire (cuestionario ampliado, Jalisco).

---

## Module structure

```
src/linkedmice/
├── __init__.py        # re-exports all public symbols
├── mice.py            # core MICE loop, imputation primitive, skip utilities
├── feature_eng.py     # generic cross-table feature engineering (broadcast, LOO)
├── diagnostics.py     # convergence analysis
├── evaluation.py      # bake-off validation helpers
├── reporting.py       # missing/BPP reports, marginal validation
└── utils.py           # dtype normalization, parent-child NaN repair

scripts/
├── census_mx.py           # census example layer: census feature hooks,
│                          # column rebuilds, portable data loader (mxcensus)
├── imputation_run_ca.py   # full 20-iteration census run
└── imputation_pilot.py    # 10% pilot + LightGBM/XGBoost bake-off
```

Everything generic is re-exported from `linkedmice`:

```python
from linkedmice import (
    integrated_mice, post_imputation_repair,
    analyze_convergence, run_validation_report, ...
)
```

Census-specific helpers (feature hooks, dummy rebuilds, loader) live in the
example layer `scripts/census_mx.py`, which requires the `census` extra
(`uv sync --extra census`).

---

## Generic engine, pluggable specifics

The core engine knows nothing about the census. Dataset specifics enter
through parameters of `integrated_mice` (all defaults match the census
example, so the example passes only what differs):

| Parameter | Default | Meaning |
|---|---|---|
| `bpp_value` | `"Blanco por pase"` (`linkedmice.DEFAULT_BPP`) | Structural-skip sentinel category |
| `hh_key` | `"ID_VIV"` | Household-key index level linking the two tables |
| `weight_col` | `"FACTOR"` | Survey expansion-weight column |
| `derived_suffixes` | `("_CAT",)` | Suffixes marking targets derived from a same-named source column (excluded from predictors to prevent leakage) |
| `hh_feature_hooks` | `None` | `hh = hook(hh, per)` callbacks run at each household-block start |
| `per_feature_hooks` | `None` | `per = hook(per, hh)` callbacks run at each person-block start, after the built-in household-attribute broadcast and before the built-in LOO refresh |

The census example wires its feature engineering through the hooks:

```python
from census_mx import refresh_hh_agg_features, refresh_per_role, refresh_per_position

integrated_mice(
    ...,
    hh_feature_hooks=[refresh_hh_agg_features],
    per_feature_hooks=[refresh_per_role, refresh_per_position],
)
```

---

## Key design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Backend | LightGBM (default) | Fastest at 800k rows; native categoricals; lowest memory |
| Target encoding | Stochastic sampling from predicted class probabilities | Preserves marginal distributions for synthesis |
| Weighting | `FACTOR` column as sample weight in every model | Survey-methodology practice (Rubin 1996) |
| Block ordering | Household first, then persons | HH signal stronger driver of person attributes |
| Cross-table refresh | Person→HH aggregates at HH block start; HH→person at person block start | Captures within-iteration correlation |
| Skip encoding | `"Blanco por pase"` (BPP) category = structural non-response; `NaN` = item non-response | Only NaN cells are imputed; BPP rows excluded from training and prediction |
| Missing-value init | Random draw from observed values of same column | Provides non-trivial starting state for chained equations |

---

## Scripts and notebooks

| File | Purpose |
|---|---|
| `scripts/imputation_pilot.py` | 10% subsample, 2 iterations — verify correctness, estimate runtime, run backend bake-off |
| `scripts/imputation_run_ca.py` | Full census run (20 iterations, LightGBM) → export imputed parquet files |
| `notebooks/imputation.ipynb` | Original development notebook (kept for reference; some cells predate the current API) |

---

## Typical workflow

### 1. Load data and build configuration

```python
from linkedmice import (
    build_skip_masks, build_missing_masks,
    repair_parent_child_nan, cross_validate_all_skips,
    initial_fill,
)

# parent-child NaN repair must run before building masks
df_viv = repair_parent_child_nan(df_viv, [("AGUA_ENTUBADA", "ABA_AGUA_ENTU"),
                                           ("SERSAN", "CONAGUA")])
df_per = repair_parent_child_nan(df_per, [("HLENGUA", "HESPANOL")])

hh_skip_masks    = build_skip_masks(df_viv, hh_impute_targets)
person_skip_masks = build_skip_masks(df_per, person_impute_targets)

hh_missing_masks    = build_missing_masks(df_viv, hh_impute_targets, hh_skip_masks)
person_missing_masks = build_missing_masks(df_per, person_impute_targets, person_skip_masks)

rng = np.random.default_rng(42)
hh_work  = initial_fill(df_viv, hh_impute_targets, hh_missing_masks, hh_skip_masks, rng)
per_work = initial_fill(df_per, person_impute_targets, person_missing_masks, person_skip_masks, rng)
```

### 2. Run the MICE loop

```python
from linkedmice import integrated_mice

(
    hh_imputed, per_imputed,
    hh_mm_final, per_mm_final,
    hh_sm_final, per_sm_final,
    diagnostics,
) = integrated_mice(
    hh_df=hh_work,
    per_df=per_work,
    hh_missing_mask_in=hh_missing_masks,
    person_missing_mask_in=person_missing_masks,
    hh_skip_masks_in=hh_skip_masks,
    person_skip_masks_in=person_skip_masks,
    hh_impute_targets=hh_impute_targets,
    person_impute_targets=person_impute_targets,
    hh_exclude={"FACTOR"},
    per_exclude={"FACTOR"} | _dhsersal_derived,
    hh_skip_deps=HH_SKIP_DEPS,      # list of (parent, predicate, child) tuples
    per_skip_deps=PER_SKIP_DEPS,
    person_high_priority_targets=["EDUC", "CONACT_CAT", "SITUA_CONYUGAL_CAT"],
    hh_broadcast_cols=HH_BROADCAST_COLS,
    n_iterations=5,
    backend="lightgbm",             # or "catboost" / "xgboost"
    rng=np.random.default_rng(42),
    verbose=True,
)
```

`integrated_mice` returns a **7-tuple**: the two imputed DataFrames, the four final mask dicts (including any rows dynamically promoted to skip status during the loop), and the diagnostics list.

### 3. Check convergence

Convergence is assessed by comparing per-variable imputed-cell distributions between consecutive iterations using total-variation (TV) distance. Variables that change by more than `threshold` (default 0.01) in the final iteration gap are flagged as not yet converged. If the loop has not converged at 5 iterations, increase `n_iterations` and re-run.

```python
from linkedmice import analyze_convergence

conv_df = analyze_convergence(diagnostics, threshold=0.01)
# Prints TV distances between consecutive iterations; flags variables > threshold
```

### 4. Post-imputation repair

After the loop, a final consistency sweep re-applies all skip dependencies to catch any residual mismatches between parent and child columns. It also checks the BPP invariant — every structurally-skipped row (including rows dynamically promoted during the loop) must still hold the BPP value. Violations are printed as warnings.

```python
from linkedmice import post_imputation_repair

hh_imputed, per_imputed, hh_mm, per_mm, hh_sm, per_sm = post_imputation_repair(
    hh_imputed, per_imputed,
    hh_mm_final, per_mm_final,
    hh_sm_final, per_sm_final,
    hh_skip_deps=HH_SKIP_DEPS,
    per_skip_deps=PER_SKIP_DEPS,
)
```

### 5. Validate marginals

Compares the weighted distribution of originally-missing (now imputed) cells against the weighted distribution of observed cells for every imputed target. A TV distance above 0.05 indicates the imputed marginal diverges materially from the observed distribution and warrants investigation.

```python
from linkedmice import run_validation_report

tv_dict = run_validation_report(
    hh_imputed, per_imputed,
    hh_mm, per_mm, hh_sm, per_sm,
    hh_impute_targets, person_impute_targets,
)
# Prints TV distance table; flags columns where TV > 0.05
```

### 6. Re-derive DHSERSAL dummies and export

`DHSERSAL1` and `DHSERSAL2` were imputed directly as ordinal survey responses; the `DHSERSAL_*` one-hot dummy columns used downstream by the synthesizer are deterministically re-derived from them. This step must run before export so the parquet files contain the full set of synthesis-ready columns.

```python
from census_mx import rederive_dhsersal_dummies

per_imputed = rederive_dhsersal_dummies(per_imputed)
hh_imputed.to_parquet("Viviendas14_imputed.parquet")
per_imputed.to_parquet("Personas14_imputed.parquet")
```

---

## Skip dependency format

`HH_SKIP_DEPS` and `PER_SKIP_DEPS` are lists of `SkipDep` triples:

```python
HH_SKIP_DEPS = [
    # single parent
    (["AGUA_ENTUBADA"], lambda df: df["AGUA_ENTUBADA"] == "No tiene",          "ABA_AGUA_ENTU"),
    (["SERSAN"],        lambda df: df["SERSAN"] == "No tienen taza...",         "CONAGUA"),
    # multi-parent example (OR logic)
    (["PARENT_A", "PARENT_B"],
     lambda df: (df["PARENT_A"] == "X") | (df["PARENT_B"] == "Y"),
     "CHILD"),
]
# (parent_cols, combined_predicate(df) → bool_mask, child_col)
```

The predicate receives the full working DataFrame, so it can reference any combination of parent columns.  NaN parent values produce `False` in standard pandas comparisons, which naturally models "skip not confirmed" — the child is only forced to BPP when the predicate returns `True`.

When any listed parent column is imputed during the loop, `refresh_dependent_skips` immediately updates the child column: rows where the combined condition is now active are forced to BPP and removed from the missing mask; rows where the condition is lifted are freed for imputation.

`repair_parent_child_nan` accepts the same `SkipDep` list and resets child cells from BPP to NaN wherever at least one parent is NaN and the combined predicate does not return `True` (i.e. the skip cannot be confirmed from the observed parents alone).

---

## Backend selection guide

| Factor | LightGBM | CatBoost | XGBoost |
|---|---|---|---|
| Speed at 800k rows | ★★★ fastest | ★★ ~1.5-2× slower | ★★ ~2× slower |
| High-cardinality categoricals | good | best | adequate |
| Out-of-box calibration | good | better | good |
| Memory | lowest | highest | middle |

CatBoost defaults to `boosting_type="Plain"` (standard gradient boosting). The original `"Ordered"` boosting provides marginally better calibration but is 2-4× slower; override via `backend_params={"boosting_type": "Ordered"}` if needed.

Use the pilot notebook's bake-off cell to compare accuracy and TV distance on held-out observed values before committing to a backend for the full run.

---

## Object / string dtype columns (ESTRATO et al.)

LightGBM rejects `object` and `pd.StringDtype` columns (pandera's `coerce=True` produces `StringDtype` on pandas ≥ 3.0). `weighted_impute_categorical` automatically detects these via `_needs_cat_conversion` and converts them to `pd.CategoricalDtype` using categories derived from the full column (ensuring consistent codes between training and prediction rows). The same logic is applied in `prepare_X_lgb` as a second safety net.

## Mixed int/str categorical columns

Several census columns (e.g. `ESCOLARI`, `ENT_PAIS_RES_5A`, `MUN_RES_5A`) have `CategoricalDtype` with **object-dtype categories** that mix Python `int` and `str` values — e.g. `[0, 1, …, 8, 'No especificado', 'Blanco por pase']`. This arises from YAML category definitions like `0: 0` where both the key and the value are integers, causing `expand_cat_map` to produce integer category *values* alongside string ones.

XGBoost 3.x's `pd_cat_inf` crashes outright with `TypeError: object of type 'int' has no len()`. LightGBM does not crash, but the mixed-type category index causes inconsistent integer code assignments between training and prediction rows, degrading model accuracy (observed: ESCOLARI accuracy drops from ~0.96 to ~0.54 in held-out validation).

Both `prepare_X_xgb` and `prepare_X_lgb` fix this by remapping every category label to `str` when `cats.dtype == object`, producing a uniform string-categorical column with a consistent code mapping.

## Ordered categorical columns (EDAD_CAT, ESCOLARI, NIVACAD, …)

LightGBM's categorical split treats every CategoricalDtype column as **unordered**, ignoring ordinal structure.  For columns like EDAD_CAT or ESCOLARI this halved prediction accuracy in held-out bake-off tests (ESCOLARI dropped from ~0.975 to ~0.543).

`prepare_X_lgb` detects `ordered=True` CategoricalDtype and converts those columns to **float32 integer codes** before training and prediction.  LightGBM then applies its standard numeric split algorithm, which correctly respects ordinal structure.  Unordered categoricals continue to use LightGBM's categorical split (with mixed int/str remapping where needed).

`_impute_lightgbm` correspondingly passes only columns that are still `CategoricalDtype` after preparation as `categorical_feature`; float32-coded ordinals are excluded so LightGBM does not attempt categorical splits on numeric data.

---

## API reference

### `mice.py`

| Symbol | Description |
|---|---|
| `build_skip_masks(df, targets)` | `{col: bool Series}` — True where BPP |
| `build_missing_masks(df, targets, skip_masks)` | `{col: bool Series}` — True where NaN (asserts disjoint from BPP) |
| `initial_fill(df, targets, missing_masks, skip_masks, rng)` | Returns filled copy; BPP rows untouched |
| `refresh_dependent_skips(df, mm, sm, imputed_col, deps)` | Updates child skip/missing status after parent is imputed |
| `apply_consistency_repair(df, mm, sm, deps)` | End-of-iteration sweep for all deps |
| `post_imputation_repair(hh_df, per_df, ...)` | Final repair + BPP invariant check; returns 6-tuple |
| `weighted_impute_categorical(df, target_col, ...)` | Core imputation primitive; returns imputed Series |
| `build_predictor_registry(df, targets, exclude_cols, derived_suffixes)` | `{target: [predictor_cols]}` |
| `refresh_per_broadcast(per_df, hh_df, hh_broadcast_cols, hh_key)` | Drop stale `_hh` columns and re-broadcast household attributes |
| `refresh_loo(per_df, cols, hh_key, bpp_value)` | Recompute LOO household-member category counts |
| `integrated_mice(...)` | Full loop; returns 7-tuple `(hh, per, hh_mm, per_mm, hh_sm, per_sm, diagnostics)` |

### `feature_eng.py`

| Symbol | Description |
|---|---|
| `DEFAULT_BPP` | Default structural-skip sentinel (`"Blanco por pase"`) |
| `broadcast_household_attrs(person_df, household_df, cols, hh_key)` | Merge HH columns into person table with `_hh` suffix |
| `leave_one_out_category_counts(df, col, hh_key, bpp_value)` | LOO household member counts per category |

### `diagnostics.py`

| Symbol | Description |
|---|---|
| `analyze_convergence(diagnostics_list, threshold)` | TV distances between consecutive iterations; returns DataFrame |

### `evaluation.py`

| Symbol | Description |
|---|---|
| `create_validation_mask(df, targets, mm, sm, frac, rng)` | Mask `frac` of observed values for held-out evaluation |
| `evaluate_imputation(imputed_df, held_out, weights)` | Weighted accuracy + TV distance per column |
| `print_bakeoff_summary(metrics, label)` | Ranked summary table for bake-off results |

### `reporting.py`

| Symbol | Description |
|---|---|
| `post_mice_sanity_checks(hh_df, per_df, hh_mm_in, per_mm_in, hh_sm_final, per_sm_final, hh_targets, per_targets, bpp_value)` | Assert no residual NaN and BPP preserved; called automatically by `integrated_mice` |
| `missing_report(df, cols, label)` | NaN counts per column |
| `missing_report_both(df_viv, df_per, ...)` | Combined HH + person report |
| `cross_validate_all_skips(deps, df, label, bpp_value)` | Verify BPP ↔ skip condition alignment |
| `initial_fill_report(...)` | Post-fill sanity check |
| `validate_marginals(df, col, weights, missing_mask, skip_mask)` | Observed vs. imputed TV distance for one column |
| `run_validation_report(hh_df, per_df, ...)` | Full report for all imputed targets |

### `utils.py`

| Symbol | Description |
|---|---|
| `normalize_categorical_dtypes(df)` | Map mixed int/str category labels to str for parquet export |
| `repair_parent_child_nan(df, deps, bpp_value)` | Reset child BPP→NaN when the skip cannot be confirmed |

### `scripts/census_mx.py` (census example layer)

| Symbol | Description |
|---|---|
| `load_census_tables(data_dir, state)` | Load (viviendas, personas); fetches from the mxcensus mirror when no dir is given |
| `compute_person_aggregates(df)` | Person→HH aggregates (n_adults, n_trabaja, head attrs, …) |
| `compute_role_features(df)` | Head/partner attributes for each person (self-leakage excluded) |
| `compute_position_features(df)` | is_head, is_partner, is_child, hh_size, is_single_person_hh |
| `refresh_hh_agg_features`, `refresh_per_role`, `refresh_per_position` | `FeatureHook` adapters around the three functions above |
| `rederive_dhsersal_dummies(df)` | Rebuild DHSERSAL_* columns from imputed DHSERSAL1/DHSERSAL2 |
| `rebuild_med_traslado_dummies(df)`, `rebuild_financiamiento_dummies(df)`, `rebuild_disability_aggregates(df)`, `rebuild_educ_col(df)` | Other deterministic column rebuilds from imputed sources |
