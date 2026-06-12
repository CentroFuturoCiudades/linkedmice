# linkedmice

Survey-weighted two-table MICE imputation for linked household–person data.

`linkedmice` jointly imputes categorical variables across two linked tables — a
*household* table and a *person* table joined by a household key — using
gradient-boosted tree classifiers (LightGBM by default; CatBoost and XGBoost
optional) with survey expansion weights as sample weights. Each MICE iteration
alternates a household block and a person block, refreshing cross-table
features in between so the models capture within-iteration correlation.

Key features:

- **Two missingness kinds.** `NaN` = item non-response → imputed. A sentinel
  category (default `"Blanco por pase"`, configurable via `bpp_value`) =
  structural skip → frozen, excluded from training and prediction.
- **Skip dependencies.** `SkipDep` triples `(parent_cols, predicate, child_col)`
  declare questionnaire skip logic; child columns are dynamically promoted to
  (or released from) skip status as their parents are imputed, with a
  consistency sweep each iteration.
- **Stochastic by default.** Imputed values are sampled from predicted class
  probabilities to preserve marginal distributions (argmax available via
  `deterministic=True`).
- **Pluggable feature engineering.** Built-in household→person broadcasts and
  leave-one-out member counts, plus `FeatureHook` callbacks for dataset-specific
  features.
- **Validation toolkit.** Convergence diagnostics (TV distance across
  iterations), marginal validation, held-out bake-off evaluation, and
  invariant checks.

## Install

```bash
uv add linkedmice                       # core (LightGBM backend)
uv add 'linkedmice[xgboost]'            # + XGBoost backend
uv add 'linkedmice[catboost]'           # + CatBoost backend
uv add 'linkedmice[all]'                # all backends
```

While unpublished, install from GitHub:

```toml
# pyproject.toml
[tool.uv.sources]
linkedmice = { git = "https://github.com/CentroFuturoCiudades/linkedmice" }
```

| Extra | Contents |
|---|---|
| `xgboost` / `catboost` / `all` | optional tree backends |
| `census` | mxcensus + jupyter — needed for the worked example under `scripts/` |
| `dev` | pytest |

## Quickstart

```python
import numpy as np
import pandas as pd
from linkedmice import (
    build_skip_masks, build_missing_masks, initial_fill,
    integrated_mice, post_imputation_repair, run_validation_report,
)

# hh_df: indexed by household key (default level name "ID_VIV")
# per_df: indexed by (household key, person id); both carry a weight
# column (default "FACTOR"); targets are pd.Categorical with NaN holes.

HH_TARGETS = ["TENURE", "FUEL"]
PER_TARGETS = ["EDU", "WORKS"]
SKIP_DEPS = [
    # WORKS is structurally blank ("Blanco por pase") for young children
    (["AGE_CAT"], lambda df: df["AGE_CAT"] == "0-11", "WORKS"),
]

rng = np.random.default_rng(42)
hh_sm = build_skip_masks(hh_df, HH_TARGETS)
per_sm = build_skip_masks(per_df, PER_TARGETS)
hh_mm = build_missing_masks(hh_df, HH_TARGETS, hh_sm)
per_mm = build_missing_masks(per_df, PER_TARGETS, per_sm)

hh_work = initial_fill(hh_df, HH_TARGETS, hh_mm, hh_sm, rng)
per_work = initial_fill(per_df, PER_TARGETS, per_mm, per_sm, rng)

hh_imp, per_imp, *masks, diag = integrated_mice(
    hh_df=hh_work, per_df=per_work,
    hh_missing_mask_in=hh_mm, person_missing_mask_in=per_mm,
    hh_skip_masks_in=hh_sm, person_skip_masks_in=per_sm,
    hh_impute_targets=HH_TARGETS, person_impute_targets=PER_TARGETS,
    hh_exclude={"FACTOR"}, per_exclude={"FACTOR"},
    hh_skip_deps=[], per_skip_deps=SKIP_DEPS,
    person_high_priority_targets=[], hh_broadcast_cols=["TENURE"],
    n_iterations=5,
)
```

See the [design and usage guide](docs/imputer.md) for the full workflow,
skip-dependency semantics, backend selection, and API reference.

## Worked example: Mexico Census 2020 (Jalisco)

The package grew out of imputing ~60 categorical variables across the linked
Viviendas/Personas tables of the census extended questionnaire (~800k persons),
weighted by the `FACTOR` expansion column. The full pipeline lives in
`scripts/`, with census-specific feature hooks and column rebuilds in
`scripts/census_mx.py` (the only mxcensus-dependent layer):

```bash
uv sync --extra census --extra xgboost
python scripts/imputation_pilot.py     # 10% subsample + backend bake-off
python scripts/imputation_run_ca.py    # full 20-iteration run
python scripts/imputation_run_ca.py --deterministic   # argmax variant
```

Data is fetched from the mxcensus mirror on first use (pass `--data DIR` to
use local parquets instead). Outputs land in `outputs/`; the validation
reports `docs/full_run_report.qmd` and `docs/pilot_report.qmd` render from
those artifacts.

## Repository layout

| Path | Contents |
|---|---|
| `src/linkedmice/` | generic engine (no census dependencies) |
| `scripts/` | census worked example + `census_mx.py` helpers |
| `notebooks/` | development notebook and design plan (historical reference) |
| `docs/` | design guide and run validation reports |
| `tests/` | smoke tests on synthetic two-table data |

## Testing

```bash
uv run pytest
```

## Related packages

- [mxcensus](https://github.com/CentroFuturoCiudades/mxcensus) — Mexico Census 2020 loaders and schemas (used by the example).
- [eodgdl](https://github.com/CentroFuturoCiudades/eodgdl) — Guadalajara EOD 2023 survey tooling.
- [pop_synth](https://github.com/CentroFuturoCiudades/pop_synth) — population synthesis pipeline that consumes the imputed census tables.

## License

MIT
