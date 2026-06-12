# Imputation Plan: Integrated Weighted Two-Table MICE for Population Synthesis

## Context

- **Input data**: Mexico Census 2020 extended questionnaire (Jalisco), two related tables.
  - **Person table** (`df_per`): ~800,000 rows, indexed by `(ID_VIV, ID_PERSONA)`.
  - **Household table** (`df_viv`): ~250,000 rows, indexed by `ID_VIV`.
  - Linked by `ID_VIV`.
- **Variables**: ~200 columns total across both tables, all categorical. Some are high-cardinality and do not require imputation but are useful as predictors.
- **Goal**: Single best-guess imputation suitable for downstream **population synthesis**.
- **Constraint**: Python ecosystem.
- **Survey weight**: `FACTOR` column (expansion factor, household-level, present in both tables). No separate person weight.
- **Missing value encoding**:
  - `NaN` — item non-response (`"No especificado"` replaced with `np.nan` during load). **These are the cells we impute.**
  - `"Blanco por pase"` — structural skip (question not applicable). **These must NOT be imputed** — they remain as-is.
- **Additional requirements**:
  - **Weighted training**: imputation models must use `FACTOR` as sample weight so imputed distributions reflect the population structure encoded in the expansion factor.
  - **Fully integrated joint loop**: both tables imputed within each MICE iteration, with cross-table aggregates and broadcast features refreshed live across the iteration.
  - Geographic information is available via `MUN`, `LOC50K`, `UPM`, `ESTRATO` (high-cardinality design columns used as predictors only).

## Why a Custom Implementation

The off-the-shelf `miceforest` library has two limitations for this problem:

1. **No first-class support for per-row sample weights.** The library's `**kwlgb` passthrough sends LightGBM hyperparameters, not training-data-level arguments. `Dataset(weight=...)` is not exposed.
2. **No native support for joint multi-table imputation.** The two-stage workflow with engineered features approximates this but cannot refresh cross-table aggregates within a single iteration as variables are imputed.

Both limitations can be addressed by implementing a custom MICE loop directly on top of a gradient-boosting backend. This involves more engineering than using `miceforest` but is feasible and gives full control over weighting, iteration order, and cross-table state management.

## Backend Choice: LightGBM, CatBoost, or XGBoost

Three viable gradient-boosting backends exist for the per-column conditional model. Each has tradeoffs for this specific problem:

| Factor | LightGBM | CatBoost | XGBoost |
|--------|----------|----------|---------|
| Training speed at 800k rows | **Fastest** | 2–4× slower | ~2× slower |
| Native categorical handling | Good (objective-based grouping) | **Best** (ordered target statistics) | Adequate (1.6+, similar to LightGBM) |
| High-cardinality predictors | Adequate | **Best** | Adequate |
| Sample weight support | Yes (via `Dataset(weight=...)`) | Yes (via `sample_weight` in `fit`) | Yes (via `DMatrix(weight=...)`) |
| Probability calibration | Good | **Better out-of-the-box** | Good |
| Memory footprint | **Lowest** | Highest | Middle |
| Maturity / ecosystem | High | High | **Highest** |
| Default hyperparameter robustness | Needs tuning | **Most robust** | Needs tuning |

**Working assumption**: LightGBM as the primary backend. It's fastest, has native categorical handling, native weights, and lowest memory footprint — the right tradeoff at this scale.

**Pilot bake-off**: Before committing, compare LightGBM and CatBoost on a 10% subsample (see Step 11). CatBoost's ordered target statistics may produce better results on heavily-categorical data with high-cardinality predictors, which describes this dataset. The runtime cost of CatBoost is real, but if quality wins are meaningful and runtime stays under ~2× LightGBM, switching is justified.

XGBoost is unlikely to win this comparison: slower than LightGBM, weaker categorical handling than CatBoost, no clear advantage for this specific problem. Include it in the bake-off only if there's organizational pressure or specific compatibility requirements.

The implementation in Step 5 is **backend-agnostic**: the imputation function takes a `backend` parameter so the comparison can be done by changing one argument.

## Methodological Foundations

### Why include weights in the imputation model

Survey-methodology practice (Rubin 1996; Schenker, Raghunathan, Chiu, Makuc, Zhang & Cohen 2006) recommends incorporating sampling design features — including weights and design indicators — into the imputation model so that downstream weighted analyses produce approximately valid inferences. The NHIS imputation procedure explicitly does this: stratum/PSU indicators and household weights enter as predictors and as weights in the imputation models for income, education, and employment.

Limitation to be aware of: Berg, Kim & Skinner (2015) show that weight-augmented imputation is not always unbiased under informative sampling. For this project, weighted imputation is the chosen design but its validity rests on the assumption that the rake variables capture the relevant components of the missingness mechanism.

### Why no PMM for categoricals

PMM is a continuous-variable technique designed to keep imputed values on the empirical support and preserve variance. For categoricals the model already outputs a probability distribution over the actual support. The relevant choice is between:

- **Predicted mode** (deterministic argmax) — over-represents majority classes.
- **Sampling from predicted distribution** (stochastic) — preserves marginals.

For population synthesis we want distribution-preserving stochastic imputation: for each missing cell, sample a category from the predicted class probabilities. This is what `miceforest`'s `mean_match_fast_cat` does and what we replicate in the custom loop.

### Why a fully integrated joint loop

The two-stage approach (impute households, then persons, with engineered features computed once from observed data) treats cross-table dependencies as static. A fully integrated loop refreshes cross-table state (aggregates, broadcasts, leave-one-out features) within each iteration so that:

- Person aggregates feeding the household imputer reflect currently imputed person values.
- Household attributes broadcast to persons reflect currently imputed household values.
- Within-household leave-one-out features for the person imputer reflect currently imputed person values.

This is closer to true multilevel imputation than the two-stage flat approximation, though still a flat-conditional approximation of a hierarchical joint model.

---

## Design Decisions

### D1. Block ordering within an iteration

**Block-wise, household first**: within each MICE iteration, all household impute targets are processed in sequence, then all person impute targets.

Rationale: household-level signal is typically a stronger driver of person attributes than the reverse (income bracket → individual education more than the reverse). Stable household imputations within an iteration produce better person imputations downstream in the same iteration.

This mirrors the NHIS NCHS approach of imputing household-level structural variables before person-level variables.

### D2. Cross-table feature refresh frequency

| Refresh | When | Cost | Justification |
|---------|------|------|---------------|
| Person → household aggregates | Once at start of household block | Cheap | Person values don't change while household variables are being imputed |
| Household → person broadcasts | Once at start of person block | Cheap | Household values don't change during person block |
| Within-household person aggregates (leave-one-out, role-based) | Once at start of person block, plus per-variable refresh for high-priority targets | Moderate | Leave-one-out aggregates change as person values are imputed; refresh selectively |

For the 5–10 most important person impute targets (those used heavily in synthesis or those with strong within-household correlation, such as education and employment), refresh leave-one-out aggregates between variables. For the rest, refresh once per iteration.

### D3. Weight handling

- **Household imputation models**: use household weights.
- **Person imputation models**: use person weights.

If person weights derive from household weights with adjustments, this is straightforward — use the appropriate weight for each model. Document explicitly which weight column is used at each block.

### D4. Imputation order within a block

Ascending order of original missingness (least missing first), matching the standard MICE default.

### D5. Initialization

Before iteration 1, fill each missing cell with a random draw from the observed values of that variable. This gives chained-equations regressions a non-trivial starting state.

### D6. Convergence diagnostics

Track three families of diagnostics:

1. **Per-variable imputation distribution** across iterations (standard MICE diagnostic).
2. **Cross-table aggregate distributions** (e.g., distribution of `modal_education` across households, distribution of `n_employed_in_hh` across households, joint distribution of head's imputed education × spouse's imputed education).
3. **Joint distributions of synthesis-critical pairs** spanning both tables (household income × imputed head education, household composition × imputed person attributes).

Convergence is assessed across all three; the joint system has converged when all three stabilize.

### D7. Multiple imputation

The loop is parameterized by random seed. Generating *m* imputations means running the loop *m* times and storing *m* completed pairs of tables. For the first build, generate `m=1` (single best-guess imputation as required). The infrastructure supports `m>1` if uncertainty propagation is added later.

---

## Execution Plan

### Step 1: Variable Inventory and Classification ✅

**Status**: Complete. Implemented in `notebooks/imputation.ipynb`.

| Role | Columns |
|------|---------|
| Join key | `ID_VIV` |
| Survey weight (both tables) | `FACTOR` |
| Structural / design (predictor-only) | `MUN`, `LOC50K`, `UPM`, `ESTRATO`, `COBERTURA`, `TAMLOC` |
| Household imputation targets (28) | see below |
| Person imputation targets (33) | see below |
| High-priority person targets | `EDUC`, `CONACT_CAT`, `SITUA_CONYUGAL_CAT` |
| Household columns broadcast to persons | `INGTRHOG_CAT`, `TIPOHOG`, `CLAVIVP_CAT`, `NUMPERS`, `JEFE_SEXO`, `JEFE_EDAD`, `TAMLOC` |

**Household imputation targets** (synthesis columns from `constraints_viviendas.yaml`):
`ABA_AGUA_ENTU`, `AGUA_ENTUBADA`, `AUTOPROP`, `BICICLETA`, `CELULAR`, `CISTERNA`, `CLAVIVP_CAT`, `COMPUTADORA`, `CONAGUA`, `CON_VJUEGOS`, `CUADORM_CAT`, `DRENAJE_CAT`, `ELECTRICIDAD`, `HORNO`, `INTERNET`, `JEFE_SEXO`, `LAVADORA`, `MOTOCICLETA`, `PISOS`, `RADIO`, `REFRIGERADOR`, `SERSAN`, `SERV_PEL_PAGA`, `SERV_TV_PAGA`, `TELEFONO`, `TELEVISOR`, `TINACO`, `TOTCUART_CAT`

**Person imputation targets** (synthesis columns from `constraints_personas.yaml`):
`AFRODES`, `ALFABET`, `ASISTEN`, `CONACT_CAT`, `DHSERSAL_AFIL`, `DHSERSAL_IMSS`, `DHSERSAL_IMSS_Prospera/Bienestar`, `DHSERSAL_ISSSTE`, `DHSERSAL_ISSSTE_E`, `DHSERSAL_No afiliado`, `DHSERSAL_Otro`, `DHSERSAL_PUB`, `DHSERSAL_P_D_M`, `DHSERSAL_Popular_NGenración_SBienestar`, `DHSERSAL_Privado`, `DIS_BANARSE`, `DIS_CAMINAR`, `DIS_CON`, `DIS_HABLAR`, `DIS_LIMI`, `DIS_MENTAL`, `DIS_OIR`, `DIS_RECORDAR`, `DIS_VER`, `EDAD_CAT`, `EDUC`, `ENT_PAIS_NAC_CAT`, `ENT_PAIS_RES_CAT`, `HESPANOL`, `HLENGUA`, `RELIGION_CAT`, `SEXO`, `SITUA_CONYUGAL_CAT`

> **DHSERSAL handling**: The `DHSERSAL_*` dummy columns are deterministically derived from `DHSERSAL1`/`DHSERSAL2` via `dhsersal_create_dummies`. They are **removed** from `person_impute_targets` and replaced with `DHSERSAL1` and `DHSERSAL2` (the actual survey answers). After the MICE loop completes, `rederive_dhsersal_dummies(df_per)` is called to rebuild all `DHSERSAL_*` columns. Imputing the dummies directly was wrong: "No especificado" in the source silently maps to all-zeros, and the dummies have a deterministic relationship to the source columns that MICE would corrupt.

All categorical columns are already typed as `pd.CategoricalDtype` by the schema (`load_extended_personas` / `load_extended_viviendas`). No additional harmonization is needed before the MICE loop; categories are stable at load time.

### Step 2: Handle Skip Patterns ✅

**Status**: Complete. Implemented in `notebooks/imputation.ipynb`.

In this census, structural non-response is already encoded as the category `"Blanco por pase"` (BPP), distinct from `NaN` (item non-response). No transformation is needed — BPP rows are identified at runtime and excluded from both training and prediction via `build_skip_masks()`.

**Skip mask rules** (verified in notebook):

*Person-level:*

| Column | Skip condition |
|--------|---------------|
| `HLENGUA` | `EDAD_CAT == "0-2"` (age < 3) |
| `EDUC` | `EDAD_CAT == "0-2"` (age < 3) |
| `ASISTEN` | `EDAD_CAT == "0-2"` (age < 3) |
| `ALFABET` | Age < 5 |
| `ENT_PAIS_RES_CAT` | Age < 5 |
| `CONACT_CAT` | Age < 12 |
| `SITUA_CONYUGAL_CAT` | Age < 12 |
| `HESPANOL` | `HLENGUA` = `"No"` |
| `DHSERSAL2` | Person enrolled in at most one service |

*Household-level:*

| Column | Skip condition |
|--------|---------------|
| `ABA_AGUA_ENTU` | `AGUA_ENTUBADA == "No tiene"` |
| `CONAGUA` | `SERSAN == "No tienen taza de baño ni letrina."` |
| Most other HH targets | `CLAVIVP_CAT == "Otro"` (non-residential, 491 rows) |

The `build_skip_masks(df, targets)` function returns `{col: bool Series}` where `True` = structural skip. An assertion verifies NaN ∩ BPP = ∅ for all columns.

### Step 3: Initialization ✅

**Status**: Complete. Implemented in `notebooks/imputation.ipynb`.

Working copies `hh_work` / `per_work` are created from the originals (`df_viv` / `df_per`) and then filled. The originals remain pristine.

`hh_missing_mask` and `person_missing_mask` were captured in Step 2 (before any fill) and remain the source of truth for which cells to re-impute during every MICE iteration.

Verification checks confirm:
- All original NaN cells filled (no NaN remaining in imputation targets of `hh_work` / `per_work`).
- All BPP rows in `hh_work` / `per_work` unchanged (initial_fill never touches skip rows).

### Step 4: Cross-Table Feature Functions ✅

**Status**: Complete. Implemented and smoke-tested in `notebooks/imputation.ipynb`.

Key dataset-specific details:
- Person MultiIndex: `(ID_VIV, ID_PERSONA)`. All groupby operations use `groupby(level="ID_VIV")`.
- `PARENTESCO` values after preprocessing (role groups): Head = `"Jefa(e)"`, Partner = `{"Esposa(o)", "Concubina(o) o unión libre", "Amante o querida(o)"}`, Child = `{"Hija(o)", "Hija(o) adoptiva(o)", "Hijastra(o)", "Hija(o) de crianza"}`.
- BPP rows are excluded from LOO and modal-education aggregates.

#### 4a. Person-to-household aggregates

Implemented in `compute_person_aggregates`.

#### 4b. Household-to-person broadcast

Implemented in `broadcast_household_attrs`.

#### 4c. Leave-one-out aggregates

Implemented in `leave_one_out_category_counts`.

#### 4d. Role-based features

Implemented in `compute_role_features`.

#### 4e. Position features

Implemented in `compute_position_features`.

### Step 5: Backend-Agnostic Weighted Imputation Function ✅

**Status**: Complete. Implemented in `notebooks/imputation.ipynb`.

`weighted_impute_categorical(df, target_col, predictor_cols, missing_mask, skip_mask, weights, rng, backend, backend_params)`:
- Training set: `~missing_mask & ~skip_mask`. Prediction set: `missing_mask` (NaN only, never BPP).
- Category set for target excludes BPP so it is never imputed.
- Stochastic sampling via `vectorized_categorical_sample` (vectorized cumulative probability lookup — ~50× faster than per-row `rng.choice`).
- LightGBM backend (`_impute_lightgbm`) is primary; CatBoost and XGBoost backends are implemented for Step 11 bake-off.
- `_prepare_X_lgb` converts nullable `Int*/boolean` dtypes to `float64` so LightGBM can consume them; categorical dtypes passed through for native LightGBM handling.

**Original plan content (reference only):**

The core primitive: train a weighted gradient-boosted model on observed rows, predict probabilities for missing rows, sample from the predicted distribution. The function dispatches to the chosen backend.

#### 5a. Top-level dispatcher

Implemented in `weighted_impute_categorical`.

#### 5b. LightGBM backend

Implemented in `_impute_lightgbm`.

#### 5c. CatBoost backend

Implemented in `_impute_catboost`.

#### 5d. XGBoost backend

Implemented in `_impute_xgboost`.

#### 5e. Notes on backend behavior

- **Vectorized sampling**: the previous version used a per-row `rng.choice` loop which is slow at 800k rows. `vectorized_categorical_sample` replaces it with a single vectorized cumulative-probability lookup, ~50× faster in practice.
- **Random seed control**: each backend has its own seed parameter. For full reproducibility across backends, set the seed in `backend_params` (e.g., `{'random_seed': 42}` for CatBoost, `{'seed': 42}` for LightGBM/XGBoost) and use the same `rng` for the sampling step.
- **Categorical dtype consistency**: all three backends require categoricals to have stable category sets between training and prediction. The harmonization in Step 1 handles this; verify with a sanity check before each fit if categories can change between iterations.
- **Probability calibration**: CatBoost's predicted probabilities tend to be better calibrated than LightGBM's or XGBoost's out-of-the-box. This matters for the stochastic sampling step — better calibration produces marginal distributions closer to the predicted-probability marginals.

### Step 6: Predictor Registry ✅

**Status**: Complete. Implemented in `build_predictor_registry`.

`build_predictor_registry(df, targets, exclude_cols)` returns `{target: [predictor_cols]}`.
- `_HH_EXCLUDE = {'FACTOR'}`.
- `_PER_EXCLUDE = {'FACTOR'} | _dhsersal_derived` — excludes stale `DHSERSAL_*` dummies from all person predictor sets.
- Registry is rebuilt on every feature-refresh inside the loop so newly added engineered columns are included automatically.

**Original plan content (reference only):**

For each impute target, define which columns are valid predictors. The general rule:

- All columns in the relevant table at this stage of the iteration, **except**:
  - The target itself.
  - IDs (`household_id`).
  - Skip-pattern flag columns.
  - Weight columns (these are not predictors; they are weights — though a copy can be added as a predictor explicitly if desired).

For better performance and to reduce overfitting at 200+ predictors, consider pruning predictor sets based on feature importance from a pilot run — keep the top 30–50 predictors per target.

### Step 7: The Integrated MICE Loop ✅

**Status**: Complete. Implemented in `integrated_mice`.

`integrated_mice`:
- Works on deep copies of all inputs — caller state unchanged.
- HH block: refresh person aggregates → impute targets in ascending missingness order → `refresh_dependent_skips` after each parent.
- Person block: refresh broadcasts, role, position, LOO (once per iteration for all high-priority targets; per-variable for the target right before it is imputed) → impute → `refresh_dependent_skips`.
- End-of-iteration: `apply_consistency_repair` for both tables.
- Returns `(hh_imputed, per_imputed, hh_mm, per_mm, hh_sm, per_sm, diagnostics_list)` — 7-tuple including final masks so post-loop code can check dynamically-added skip rows from `refresh_dependent_skips`.

Pilot run (10% subsample, 2 iterations) included in the notebook for timing and sanity verification.

> **Bug fixes applied (2026-05-13)**:
> 1. `weighted_impute_categorical` now converts `object`/`str` dtype columns (e.g. `ESTRATO`) to `pd.CategoricalDtype` before dispatching to any backend. Previously LightGBM crashed with `ValueError: pandas dtypes must be int, float or bool` on these columns, blocking the pilot run entirely.
> 2. `integrated_mice` return signature changed from 3-tuple to 7-tuple (`hh_mm`, `per_mm`, `hh_sm`, `per_sm` added) so post-loop code uses the final masks that include rows dynamically promoted to skip status during the loop.
> 3. `compute_person_aggregates` was aggregating `("ESCOACUM", "mean")` but had already created `ESCOACUM_NO_BPP` (with `-1` replaced by `pd.NA`) in the `assign` step. The agg key was corrected to `("ESCOACUM_NO_BPP", "mean")`.
> 4. Full-run cell previously referenced `hh_missing_mask` / `person_missing_mask` (singular, undefined) instead of `hh_missing_masks` / `person_missing_masks` (plural). Fixed.

### Step 8: Convergence Diagnostics ✅

**Status**: `analyze_convergence` implemented — computes per-variable TV distances between consecutive iteration snapshots, prints a pivot sorted by max TV, and reports unconverged variables against a configurable threshold. Applied to `pilot_diag` to verify pilot-run output.

After all iterations, compare consecutive diagnostics. Convergence is stable when:

- Per-variable distributions change by less than ~1% between iterations.
- Cross-table aggregate distributions stabilize.
- Critical joint distributions stabilize.

If not converged at iteration 5, run additional iterations.

### Step 9: Post-Imputation Consistency Repair ✅

**Status**: `post_imputation_repair` implemented — runs a final `apply_consistency_repair` pass on both tables, then checks the BPP invariant for every imputed column and warns on any violations. Now receives masks returned by `integrated_mice` (7-tuple) so the BPP invariant check covers dynamically-added skip rows. Returns the full 6-tuple of repaired DataFrames and masks.

Imputation treats each row independently. Repair logical inconsistencies post-hoc:

- **One head per household**: if multiple persons in a household have imputed `relationship_to_head = 'head'`, keep the one with highest predicted probability (or oldest, depending on rule).
- **Marital status agreement within couples**: force agreement.
- **Age coherence with role**: no children imputed as household head; no person under 16 imputed as employed full-time (unless survey allows).
- **Household composition coherence**: imputed `hh_composition` should match the actual person roster after person imputation.

Implement as a sequence of rule-based functions. Document each rule explicitly.

### Step 10: Validation ✅

**Status**: `validate_marginals` and `run_validation_report` implemented — compare weighted observed vs. imputed marginals for every target with any imputed rows, print a sorted summary table, and flag columns with TV > 0.05 as large divergence. Hand-off cell updated to use masks returned by `integrated_mice`/`post_imputation_repair` instead of rebuilding from originals.

Before hand-off to synthesis, validate:

#### 10a. Marginal distributions (weighted and unweighted)

For each imputed variable, compare:

- Observed marginals (weighted by survey weights).
- Imputed-only marginals (weighted by survey weights).
- Population targets from rake controls / census, where available.

#### 10b. Joint distributions of key pairs

Pairs to check:

- (age × education), (age × employment), (education × income).
- Within-household: (head_education × spouse_education), (parent_education × child_education).
- Cross-table: (household_income × person_education), (household_composition × person_employment).

#### 10c. Household-level joint distributions

- (household_composition × tenure × income_bracket) in observed-complete vs. imputed households.
- Distribution of household sizes after imputation.

If these are off, the synthesizer will produce wrong household types at wrong rates.

### Step 11: Backend Bake-Off Infrastructure ✅

**Status**: `create_validation_mask` and `evaluate_imputation` implemented — mask 5% of observed values per target as held-out ground truth, then measure weighted accuracy and TV distance; a commented code block shows the full bake-off invocation pattern for LightGBM vs. CatBoost on the pilot subsample.

Before committing to a backend for the full run, compare LightGBM and CatBoost (and optionally XGBoost) on a 10% subsample. The infrastructure is already backend-agnostic; this is a swap-and-rerun exercise.

#### 11a. Subsample construction

Stratified subsample preserving household integrity:

#### 11b. Held-out validation set

Artificially mask a sample of *observed* values per impute target to use as ground truth.

Apply to both subsampled tables before running the loop. The held-out values become the evaluation ground truth.

#### 11c. Run each backend

#### 11d. Comparison metrics

For each backend, compute on the held-out values.

#### 11e. Decision rule

Choose the backend that:

1. **Wins on weighted accuracy** by ≥1–2 percentage points averaged across targets, **AND**
2. **Wins or ties on total-variation distance** for marginal distribution preservation, **AND**
3. **Has runtime no more than 2× the fastest backend** (otherwise the full-scale run becomes operationally impractical).

If no backend dominates clearly, **default to LightGBM** for runtime reasons. Document the decision and the metrics that drove it.

#### 11f. What "clearly better" looks like in practice

For all-categorical data with high-cardinality predictors, CatBoost's edge typically appears in:

- Marginal distribution preservation (TV distance) for high-cardinality targets.
- Calibration of predicted probabilities (worth checking with reliability diagrams if you have time).
- Stability of imputed distributions across MICE iterations (less drift between iteration 3 and iteration 5).

LightGBM typically wins on:

- Wall-clock time (2–4×).
- Memory footprint.
- Reproducibility (simpler internal state).

If the bake-off shows CatBoost winning by <1 percentage point on accuracy and the runtime cost is 3×, LightGBM is the right operational choice even though CatBoost is slightly more accurate.

### Step 12: Hand-Off

**Status**: Full-run cell (5 iterations, LightGBM, complete data) plus hand-off cell implemented — re-derives `DHSERSAL_*` dummies, runs the validation report, and exports `Viviendas14_imputed.parquet` and `Personas14_imputed.parquet` to the census data directory.

Output the two completed tables (`household_imputed`, `person_imputed`) joined by `household_id`, ready for the population synthesis pipeline.

---

## Performance Considerations

### Runtime estimate

Per LightGBM training call: 1–5 minutes on 800k rows depending on `n_estimators`, predictor count, and `n_jobs`.

Per iteration:
- Household block (~25 targets): 5–15 minutes.
- Person block (~50 targets): 30–120 minutes.
- Cross-table feature recomputation: a few minutes total.

Total per iteration: ~1–2.5 hours.
Full run (5 iterations): ~5–12 hours.

### Optimizations

These do not compromise quality if applied carefully:

- **Reduce `n_estimators`** to 50–75 with early stopping disabled (the iterative MICE loop provides its own implicit regularization).
- **Subsample training data** when observed rows greatly exceed needs: cap at ~200k training rows per model via `data_subset` analog.
- **Prune predictors** to top-30 or top-50 by feature importance from a pilot run, instead of using all 200+ columns.
- **Vectorize categorical sampling** in `weighted_impute_categorical` to replace per-row `rng.choice`.
- **Cache groupby objects** for recomputing aggregates within an iteration.
- **`n_jobs=-1`** in LightGBM to use all cores.

### Pilot run

Before the full run, execute the loop on a 10% subsample (~80k persons, ~25k households) to:

- Debug the iteration logic.
- Estimate per-iteration runtime.
- Tune `n_estimators` and `min_data_in_leaf`.
- Identify predictor pruning candidates from feature importances.

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Cross-table state inconsistency (e.g., aggregates fall out of sync with person values) | Wrap aggregate refreshes in a single function called at well-defined points; assert invariants in tests |
| Weighted training amplifies extreme weights, leading to unstable models | Inspect weight distributions; consider weight trimming (cap at 95th percentile) for the imputation models specifically |
| Skip patterns mistakenly imputed | Step 2 explicit handling; verify post-imputation that structurally-missing cells remain marked |
| Feature leakage (e.g., variable predicting itself via aggregates) | Leave-one-out construction for variables that aggregate themselves; mask role-based features for the focal person |
| Convergence not reached in 5 iterations | Diagnostics flag this; extend iterations |
| MNAR mechanism underlying missingness | Weighting helps under MAR keyed to rake variables; cannot fix MNAR. Document as a limitation |
| LightGBM overfitting on 800k rows with many predictors | High `min_data_in_leaf` (200+); predictor pruning; regularization via lower `num_leaves` |
| Memory pressure with engineered features | Use int8/int16 for one-hot leave-one-out counts, category dtype for everything categorical, single working copy per table |
| Per-row `rng.choice` dominates runtime | Vectorized sampling via `np.cumsum` + uniform draws |
| High-cardinality columns dominate splits | Tune `cat_smooth`, `max_cat_threshold`; group rare categories |

---

## Engineering Effort Estimate

- Core loop infrastructure: 300–500 lines.
- Cross-table feature engineering: 200–400 lines.
- Diagnostics and convergence tracking: 150–300 lines.
- Consistency repair rules: 100–500 lines (highly domain-dependent).
- Tests: substantial — aggregate recomputation, weight passthrough, end-to-end on synthetic data.

Realistic budget: **2–4 weeks of one experienced engineer**, plus validation time.

---

## Open Questions

These should be resolved before the build begins:

1. **Are person and household weights distinct, or is one derived from the other?** Affects D3 weight handling.
2. **Which rake variables are available in the dataset, and at what resolution?** They must be predictors in both blocks.
3. **What population-level targets exist for validation?** Without external targets, validation is limited to internal consistency checks.
4. **Should weight trimming be applied for imputation models?** Extreme weights can destabilize gradient-boosted training across all backends.
5. **Is multiple imputation desired now or deferrable?** The infrastructure supports it; running it scales runtime by `m`.
6. **Backend choice — to be resolved by pilot bake-off (Step 11).** Working assumption is LightGBM; CatBoost is the most likely alternative if quality wins justify the runtime cost. XGBoost is included only if specifically needed.

---

## Summary Checklist

- [ ] Variable inventory complete (impute targets, predictor-only, IDs, skip-pattern, weights)
- [ ] All categoricals converted to category dtype with consistent categories
- [ ] Skip-pattern handling applied
- [ ] Original missingness masks captured before initial fills
- [ ] Initial fills applied
- [ ] Cross-table feature functions implemented and unit-tested
- [ ] Backend-agnostic weighted imputation function implemented and unit-tested
- [ ] LightGBM, CatBoost (and optionally XGBoost) backends individually verified on a tiny synthetic dataset
- [ ] Predictor registries built
- [ ] Pilot run on 10% subsample completed for each candidate backend
- [ ] Backend bake-off metrics computed (accuracy, TV distance, runtime)
- [ ] Backend selected and decision documented
- [ ] Full integrated MICE loop executed for 5 iterations with selected backend
- [ ] Convergence verified across all three diagnostic families
- [ ] Consistency repair applied
- [ ] Marginal distributions validated (weighted)
- [ ] Joint distributions validated (within and across tables)
- [ ] Household-level joints validated for synthesis hand-off
- [ ] Final tables exported for population synthesis

---

## References

- Rubin, D. B. (1996). Multiple Imputation After 18+ Years. *Journal of the American Statistical Association*, 91(434), 473–489.
- Schenker, N., Raghunathan, T. E., Chiu, P.-L., Makuc, D. M., Zhang, G., & Cohen, A. J. (2006). Multiple Imputation of Missing Income Data in the National Health Interview Survey. *Journal of the American Statistical Association*, 101(475), 924–933. https://doi.org/10.1198/016214505000001375
- Raghunathan, T. E., Lepkowski, J. M., Van Hoewyk, J., & Solenberger, P. (2001). A Multivariate Technique for Multiply Imputing Missing Values Using a Sequence of Regression Models. *Survey Methodology*, 27(1), 85–95. (Foundational SRMI paper; the methodological lineage underlying the NHIS imputation approach.)
- National Center for Health Statistics. (2022). *Multiple Imputation of Family Income in 2021 National Health Interview Survey: Methods*. Division of Health Interview Statistics, NCHS, CDC. https://ftp.cdc.gov/pub/health_statistics/nchs/dataset_documentation/NHIS/2021/NHIS2021-imputation-techdoc-508.pdf (Concrete precedent for the integrated multi-table imputation loop and weight-augmented design used here.)
- Berg, E., Kim, J. K., & Skinner, C. (2015). Imputation under Informative Sampling. *Journal of Survey Statistics and Methodology*. (Caveat on weight-augmented imputation under informative sampling.)
- Van Buuren, S. (2018). *Flexible Imputation of Missing Data* (2nd ed.). Chapman & Hall/CRC.
- Wilson, S. (miceforest, GitHub). Multiple Imputation by Chained Equations with LightGBM. https://github.com/AnotherSamWilson/miceforest
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS 2017*.
- Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: Unbiased Boosting with Categorical Features. *NeurIPS 2018*.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*.