"""Smoke tests for the linkedmice engine on synthetic two-table data."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

import linkedmice
from linkedmice import (
    DEFAULT_BPP,
    build_missing_masks,
    build_predictor_registry,
    build_skip_masks,
    create_validation_mask,
    cross_validate_all_skips,
    evaluate_imputation,
    initial_fill,
    integrated_mice,
    repair_parent_child_nan,
)

N_HH = 50
RNG_SEED = 7

TINY_LGB = {"num_leaves": 4, "min_data_in_leaf": 1, "num_threads": 1,
            "num_boost_round": 5}


def make_tables(hh_key="ID_VIV", bpp=DEFAULT_BPP, seed=RNG_SEED):
    """~50 households / ~140 persons with NaN holes and one skip dep per table."""
    rng = np.random.default_rng(seed)

    hh_ids = np.arange(1, N_HH + 1)
    tenure = rng.choice(["own", "rent", "other"], N_HH)
    fuel_cats = ["gas", "wood", bpp]
    # FUEL is structurally skipped when TENURE == "other"
    fuel = np.where(
        tenure == "other", bpp, rng.choice(["gas", "wood"], N_HH)
    )
    hh = pd.DataFrame(
        {
            "TENURE": pd.Categorical(tenure),
            "FUEL": pd.Categorical(fuel, categories=fuel_cats),
            "FUEL_CAT": pd.Categorical(fuel, categories=fuel_cats),
            "NROOMS": rng.integers(1, 6, N_HH),
            "FACTOR": rng.integers(50, 200, N_HH).astype(float),
        },
        index=pd.Index(hh_ids, name=hh_key),
    )

    sizes = rng.integers(1, 6, N_HH)
    per_hh = np.repeat(hh_ids, sizes)
    n_per = len(per_hh)
    per_id = np.concatenate([np.arange(1, s + 1) for s in sizes])
    age = rng.choice(["0-11", "12-17", "18-64", "65+"], n_per)
    works_cats = ["yes", "no", bpp]
    # WORKS is structurally skipped for ages 0-11
    works = np.where(age == "0-11", bpp, rng.choice(["yes", "no"], n_per))
    per = pd.DataFrame(
        {
            "AGE_CAT": pd.Categorical(age),
            "SEX": pd.Categorical(rng.choice(["m", "f"], n_per)),
            "EDU": pd.Categorical(rng.choice(["low", "mid", "high"], n_per)),
            "WORKS": pd.Categorical(works, categories=works_cats),
            "FACTOR": np.repeat(hh["FACTOR"].values, sizes),
        },
        index=pd.MultiIndex.from_arrays(
            [per_hh, per_id], names=[hh_key, "ID_PERSONA"]
        ),
    )

    # Inject ~10% item non-response into non-skipped cells of the targets
    for df, cols in [(hh, ["TENURE", "FUEL"]), (per, ["EDU", "WORKS"])]:
        for col in cols:
            eligible = df.index[(df[col] != bpp).to_numpy()]
            n_holes = max(2, int(len(eligible) * 0.10))
            holes = rng.choice(len(eligible), size=n_holes, replace=False)
            df.loc[eligible[holes], col] = pd.NA

    hh_deps = [(["TENURE"], lambda df: df["TENURE"] == "other", "FUEL")]
    per_deps = [(["AGE_CAT"], lambda df: df["AGE_CAT"] == "0-11", "WORKS")]
    return hh, per, hh_deps, per_deps


HH_TARGETS = ["TENURE", "FUEL"]
PER_TARGETS = ["EDU", "WORKS"]


def build_all_masks(hh, per, bpp=DEFAULT_BPP):
    hh_sm = build_skip_masks(hh, HH_TARGETS, bpp)
    per_sm = build_skip_masks(per, PER_TARGETS, bpp)
    hh_mm = build_missing_masks(hh, HH_TARGETS, hh_sm)
    per_mm = build_missing_masks(per, PER_TARGETS, per_sm)
    return hh_sm, per_sm, hh_mm, per_mm


def run_mice(hh, per, hh_deps, per_deps, hh_key="ID_VIV", bpp=DEFAULT_BPP,
             n_iterations=2, hooks=False, backend="lightgbm", **kwargs):
    hh = repair_parent_child_nan(hh, hh_deps, bpp)
    per = repair_parent_child_nan(per, per_deps, bpp)
    hh_sm, per_sm, hh_mm, per_mm = build_all_masks(hh, per, bpp)
    rng = np.random.default_rng(RNG_SEED)
    hh_work = initial_fill(hh, HH_TARGETS, hh_mm, hh_sm, rng)
    per_work = initial_fill(per, PER_TARGETS, per_mm, per_sm, rng)

    hook_args = {}
    if hooks:
        def hh_hook(hh_df, per_df):
            hh_df["n_members_hook"] = (
                per_df.groupby(level=hh_key).size().reindex(hh_df.index).fillna(0)
            )
            return hh_df

        def per_hook(per_df, hh_df):
            per_df["nrooms_hook"] = (
                hh_df["NROOMS"]
                .reindex(per_df.index.get_level_values(hh_key))
                .to_numpy()
            )
            return per_df

        hook_args = {"hh_feature_hooks": [hh_hook], "per_feature_hooks": [per_hook]}

    return integrated_mice(
        hh_df=hh_work,
        per_df=per_work,
        hh_missing_mask_in=hh_mm,
        person_missing_mask_in=per_mm,
        hh_skip_masks_in=hh_sm,
        person_skip_masks_in=per_sm,
        hh_impute_targets=HH_TARGETS,
        person_impute_targets=PER_TARGETS,
        hh_exclude={"FACTOR"},
        per_exclude={"FACTOR"},
        hh_skip_deps=hh_deps,
        per_skip_deps=per_deps,
        person_high_priority_targets=["EDU"],
        hh_broadcast_cols=["TENURE", "NROOMS"],
        hh_key=hh_key,
        bpp_value=bpp,
        n_iterations=n_iterations,
        backend=backend,
        backend_params=TINY_LGB,
        rng=np.random.default_rng(RNG_SEED),
        verbose=False,
        **hook_args,
        **kwargs,
    ), (hh_mm, per_mm, hh_sm, per_sm)


def test_public_api():
    assert linkedmice.__all__
    for name in linkedmice.__all__:
        assert getattr(linkedmice, name) is not None, name


def test_masks_disjoint_and_skip_validation():
    hh, per, hh_deps, per_deps = make_tables()
    hh = repair_parent_child_nan(hh, hh_deps)
    per = repair_parent_child_nan(per, per_deps)
    hh_sm, per_sm, hh_mm, per_mm = build_all_masks(hh, per)

    for col in HH_TARGETS:
        assert not (hh_sm[col] & hh_mm[col]).any()
    for col in PER_TARGETS:
        assert not (per_sm[col] & per_mm[col]).any()
    assert per_sm["WORKS"].sum() > 0  # fixture really has structural skips

    assert cross_validate_all_skips(hh_deps, hh, "HH", DEFAULT_BPP)
    assert cross_validate_all_skips(per_deps, per, "Per", DEFAULT_BPP)


def test_initial_fill_fills_all_and_preserves():
    hh, per, hh_deps, per_deps = make_tables()
    hh_sm, per_sm, hh_mm, per_mm = build_all_masks(hh, per)
    rng = np.random.default_rng(RNG_SEED)
    hh_work = initial_fill(hh, HH_TARGETS, hh_mm, hh_sm, rng)
    per_work = initial_fill(per, PER_TARGETS, per_mm, per_sm, rng)

    for df, targets, mm, sm, orig in [
        (hh_work, HH_TARGETS, hh_mm, hh_sm, hh),
        (per_work, PER_TARGETS, per_mm, per_sm, per),
    ]:
        for col in targets:
            assert not df.loc[mm[col], col].isna().any()
            assert (df.loc[sm[col], col] == DEFAULT_BPP).all()
            observed = ~mm[col] & ~sm[col]
            pd.testing.assert_series_equal(
                df.loc[observed, col], orig.loc[observed, col]
            )


def test_integrated_mice_smoke_with_hooks():
    hh, per, hh_deps, per_deps = make_tables()
    (result, masks) = run_mice(hh, per, hh_deps, per_deps, hooks=True)
    hh_imp, per_imp, hh_mm_out, per_mm_out, hh_sm_out, per_sm_out, diag = result

    assert len(diag) == 2
    for df, targets, mm in [(hh_imp, HH_TARGETS, hh_mm_out),
                            (per_imp, PER_TARGETS, per_mm_out)]:
        for col in targets:
            assert not df.loc[mm[col], col].isna().any()
    # BPP invariant on final skip masks
    for df, sm in [(hh_imp, hh_sm_out), (per_imp, per_sm_out)]:
        for col, mask in sm.items():
            if mask.any():
                assert (df.loc[mask, col] == DEFAULT_BPP).all()
    # hook-created feature columns are present in the working frames
    assert "n_members_hook" in hh_imp.columns
    assert "nrooms_hook" in per_imp.columns


def test_custom_key_and_sentinel():
    hh, per, hh_deps, per_deps = make_tables(hh_key="HH_ID", bpp="SKIP")
    (result, _) = run_mice(
        hh, per, hh_deps, per_deps, hh_key="HH_ID", bpp="SKIP", n_iterations=1
    )
    hh_imp, per_imp, hh_mm_out, per_mm_out, hh_sm_out, per_sm_out, diag = result
    for col in PER_TARGETS:
        assert not per_imp.loc[per_mm_out[col], col].isna().any()
    assert (per_imp.loc[per_sm_out["WORKS"], "WORKS"] == "SKIP").all()


def test_predictor_registry_suffix_rule():
    hh, _, _, _ = make_tables()
    reg = build_predictor_registry(hh, ["FUEL_CAT"], {"FACTOR"})
    assert "FUEL" not in reg["FUEL_CAT"]  # default _CAT rule excludes source

    reg_off = build_predictor_registry(hh, ["FUEL_CAT"], {"FACTOR"},
                                       derived_suffixes=())
    assert "FUEL" in reg_off["FUEL_CAT"]

    reg_custom = build_predictor_registry(hh, ["FUEL_CAT"], {"FACTOR"},
                                          derived_suffixes=("_IGNORED", "_CAT"))
    assert "FUEL" not in reg_custom["FUEL_CAT"]


def test_evaluation_roundtrip():
    hh, per, hh_deps, per_deps = make_tables()
    hh = repair_parent_child_nan(hh, hh_deps)
    per = repair_parent_child_nan(per, per_deps)
    hh_sm, per_sm, hh_mm, per_mm = build_all_masks(hh, per)

    per_masked, per_mm_masked, held = create_validation_mask(
        per, PER_TARGETS, per_mm, per_sm, frac=0.1,
        rng=np.random.default_rng(RNG_SEED),
    )
    rng = np.random.default_rng(RNG_SEED)
    hh_work = initial_fill(hh, HH_TARGETS, hh_mm, hh_sm, rng)
    per_work = initial_fill(per_masked, PER_TARGETS, per_mm_masked, per_sm, rng)

    result = integrated_mice(
        hh_df=hh_work, per_df=per_work,
        hh_missing_mask_in=hh_mm, person_missing_mask_in=per_mm_masked,
        hh_skip_masks_in=hh_sm, person_skip_masks_in=per_sm,
        hh_impute_targets=HH_TARGETS, person_impute_targets=PER_TARGETS,
        hh_exclude={"FACTOR"}, per_exclude={"FACTOR"},
        hh_skip_deps=hh_deps, per_skip_deps=per_deps,
        person_high_priority_targets=[], hh_broadcast_cols=["TENURE"],
        n_iterations=1, backend="lightgbm", backend_params=TINY_LGB,
        rng=np.random.default_rng(RNG_SEED), verbose=False,
    )
    per_imp = result[1]
    metrics = evaluate_imputation(per_imp, held, per_imp["FACTOR"])
    assert set(metrics) == set(held)
    for m in metrics.values():
        assert 0.0 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["tv"] <= 1.0
        assert np.isfinite(m["accuracy"]) and np.isfinite(m["tv"])


@pytest.mark.parametrize("backend,extra", [("catboost", "catboost"),
                                           ("xgboost", "xgboost")])
def test_optional_backend_error_message(backend, extra):
    if importlib.util.find_spec(backend) is not None:
        pytest.skip(f"{backend} installed — lazy-import error path not reachable")
    hh, per, hh_deps, per_deps = make_tables()
    with pytest.raises(ImportError, match=f"linkedmice\\[{extra}\\]"):
        run_mice(hh, per, hh_deps, per_deps, n_iterations=1, backend=backend)
