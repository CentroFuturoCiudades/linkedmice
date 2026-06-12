import time
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from .feature_eng import (
    DEFAULT_BPP,
    broadcast_household_attrs,
    leave_one_out_category_counts,
)

BackendName = Literal["lightgbm", "catboost", "xgboost"]
_DEFAULT_SEED = 42

# Skip dependency triple: (parent_cols, combined_predicate(df) → bool_mask, child_col).
# The predicate receives the full working DataFrame so it can reference any
# combination of parent columns.  NaN parent values evaluate to False in
# standard pandas comparisons, which naturally models "skip not confirmed".
SkipDep = Tuple[List[str], Callable[[pd.DataFrame], pd.Series], str]

# Cross-table feature-refresh hook.  Household hooks are called as
# ``hh = hook(hh, per)`` at the start of each household block; person hooks as
# ``per = hook(per, hh)`` at the start of each person block (after the built-in
# household-attribute broadcast and before the built-in LOO refresh).
FeatureHook = Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]


def build_skip_masks(
    df: pd.DataFrame, targets: List[str], bpp_value: str = DEFAULT_BPP
) -> Dict[str, pd.Series]:
    """Build BPP (structural skip) masks for each target column.

    A skip mask is ``True`` for rows where the question was not applicable
    (*Blanco por pase*).  NaN rows (item non-response) are **not** skip rows —
    those are what we actually impute.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame containing the target columns.
    targets : List[str]
        Column names to build masks for.
    bpp_value : str, optional
        BPP sentinel string.

    Returns
    -------
    Dict[str, pd.Series]
        ``{col: boolean Series}`` indexed like ``df``.  Missing columns get an
        all-False Series.
    """
    masks: Dict[str, pd.Series] = {}
    for col in targets:
        if col not in df.columns:
            masks[col] = pd.Series(False, index=df.index)
            continue
        if hasattr(df[col], "cat") and bpp_value in df[col].cat.categories:
            masks[col] = df[col] == bpp_value
        else:
            masks[col] = pd.Series(False, index=df.index)

    print("\nSkip mask summary (% structural skips per column):")
    for col, mask in masks.items():
        pct = mask.sum() / len(df) * 100
        if pct > 0:
            print(f"  {col:45s}  {mask.sum():>8,}  ({pct:5.1f}%)")

    return masks


def build_missing_masks(
    df: pd.DataFrame, targets: List[str], skip_masks: Dict[str, pd.Series]
) -> Dict[str, pd.Series]:
    """Build item-non-response (NaN) masks for each target column.

    Asserts that NaN rows and BPP rows are disjoint for every target.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame containing the target columns.
    targets : List[str]
        Column names to build masks for.
    skip_masks : Dict[str, pd.Series]
        BPP masks as produced by :func:`build_skip_masks`.

    Returns
    -------
    Dict[str, pd.Series]
        ``{col: boolean Series}`` indexed like ``df``, ``True`` where the cell
        is NaN (item non-response).
    """
    masks: Dict[str, pd.Series] = {}
    for col in targets:
        is_nan = (
            df[col].isna() if col in df.columns else pd.Series(False, index=df.index)
        )
        is_skip = skip_masks[col]
        overlap = (is_nan & is_skip).sum()
        assert overlap == 0, f"Overlap between NaN and BPP in {col}: {overlap} rows"
        masks[col] = is_nan
    return masks


def refresh_dependent_skips(
    df: pd.DataFrame,
    missing_mask: Dict[str, pd.Series],
    skip_masks: Dict[str, pd.Series],
    imputed_col: str,
    deps: List[SkipDep],
    bpp_value: str = DEFAULT_BPP,
) -> None:
    """Update child skip / missing status after a parent column is imputed.

    Parameters
    ----------
    df : pd.DataFrame
        Working DataFrame; modified in-place.
    missing_mask : Dict[str, pd.Series]
        Per-column boolean Series, ``True`` where NaN; updated in-place.
    skip_masks : Dict[str, pd.Series]
        Per-column boolean Series, ``True`` where BPP; updated in-place.
    imputed_col : str
        The column that was just imputed; triggers updates for any child whose
        dep lists it as one of its parents.
    deps : List[SkipDep]
        Skip dependency triples ``(parent_cols, combined_predicate, child_col)``.
        The predicate receives the full DataFrame and returns a boolean mask.
    bpp_value : str, optional
        Structural-skip sentinel string.  Default :data:`DEFAULT_BPP`.

    Notes
    -----
    Two cases are handled per child row:

    1. Combined condition now active → child forced to BPP and removed from
       ``missing_mask``.
    2. Combined condition no longer active but child is still BPP (promoted in
       a previous iteration) → child reset to NaN and added back to
       ``missing_mask`` so it gets imputed in this iteration.
    """
    for parents, skip_pred, child in deps:
        if imputed_col not in parents:
            continue
        if child not in df.columns:
            continue

        skip_now = skip_pred(df)  # combined condition over all parents
        child_bpp = df[child] == bpp_value
        child_nan = missing_mask[child]

        # Case 1: combined condition active, child was awaiting imputation
        newly_skipped = skip_now & child_nan
        if newly_skipped.any():
            df.loc[newly_skipped, child] = bpp_value
            missing_mask[child] = missing_mask[child] & ~newly_skipped
            skip_masks[child] = skip_masks[child] | newly_skipped

        # Case 2: combined condition no longer active, child was frozen as BPP
        newly_released = ~skip_now & child_bpp
        if newly_released.any():
            df.loc[newly_released, child] = pd.NA
            missing_mask[child] = missing_mask[child] | newly_released
            skip_masks[child] = skip_masks[child] & ~newly_released


def apply_consistency_repair(
    df: pd.DataFrame,
    missing_mask: Dict[str, pd.Series],
    skip_masks: Dict[str, pd.Series],
    deps: List[SkipDep],
    bpp_value: str = DEFAULT_BPP,
) -> None:
    """End-of-iteration sweep: re-apply all skip dependencies unconditionally.

    Catches residual parent-child mismatches left after the full imputation
    block.  Uses the same logic as :func:`refresh_dependent_skips` but is
    applied to every dependency regardless of which column was last imputed.

    Parameters
    ----------
    df : pd.DataFrame
        Working DataFrame; modified in-place.
    missing_mask : Dict[str, pd.Series]
        Per-column NaN masks; updated in-place.
    skip_masks : Dict[str, pd.Series]
        Per-column BPP masks; updated in-place.
    deps : List[SkipDep]
        Skip dependency triples ``(parent_cols, combined_predicate, child_col)``.
    bpp_value : str, optional
        Structural-skip sentinel string.  Default :data:`DEFAULT_BPP`.
    """
    for parents, skip_pred, child in deps:
        if child not in df.columns or not all(p in df.columns for p in parents):
            continue

        skip_now = skip_pred(df)  # combined condition over all parents
        child_bpp = df[child] == bpp_value

        # Child should be BPP but isn't
        should_be_bpp = skip_now & ~child_bpp
        if should_be_bpp.any():
            df.loc[should_be_bpp, child] = bpp_value
            missing_mask[child] = missing_mask[child] & ~should_be_bpp
            skip_masks[child] = skip_masks[child] | should_be_bpp

        # Child should be imputable but is frozen as BPP
        should_be_free = ~skip_now & child_bpp
        if should_be_free.any():
            df.loc[should_be_free, child] = pd.NA
            missing_mask[child] = missing_mask[child] | should_be_free
            skip_masks[child] = skip_masks[child] & ~should_be_free


def initial_fill(
    df: pd.DataFrame,
    targets: List[str],
    missing_masks: Dict[str, pd.Series],
    skip_masks: Dict[str, pd.Series],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Fill NaN imputation targets with random draws from observed values.

    Provides a non-trivial starting state for the chained-equations loop.
    BPP rows are never touched.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame; a copy is returned.
    targets : List[str]
        Columns to fill.
    missing_masks : Dict[str, pd.Series]
        NaN masks produced by :func:`build_missing_masks`.
    skip_masks : Dict[str, pd.Series]
        BPP masks produced by :func:`build_skip_masks`.
    rng : np.random.Generator
        Random number generator for reproducible draws.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with NaN target cells replaced by observed draws.
    """
    df = df.copy()
    for col in targets:
        if col not in df.columns:
            continue
        is_missing = missing_masks[col]
        if not is_missing.any():
            continue
        observed_vals = df.loc[~missing_masks[col] & ~skip_masks[col], col]
        if observed_vals.empty:
            continue
        draws = rng.choice(observed_vals.values, size=is_missing.sum(), replace=True)
        df.loc[is_missing, col] = draws
    return df


# ── Helpers ───────────────────────────────────────────────────────────────────


def _needs_cat_conversion(dtype: Any) -> bool:
    """Return ``True`` for dtypes that must be cast to ``CategoricalDtype`` for LightGBM.

    LightGBM natively accepts ``int``, ``uint``, ``float``, ``bool`` numpy
    dtypes and ``pd.CategoricalDtype``.  Everything else — ``object``,
    ``pd.StringDtype``, ``pd.ArrowDtype``, etc. — must be converted.
    Nullable ``Int*`` / ``boolean`` dtypes expose a ``numpy_dtype`` attribute
    and are handled separately (cast to ``float64``).

    Parameters
    ----------
    dtype : any pandas or numpy dtype
        The column dtype to inspect.

    Returns
    -------
    bool
        ``True`` if the dtype needs conversion to categorical.
    """
    if isinstance(dtype, pd.CategoricalDtype):
        return False
    if hasattr(dtype, "numpy_dtype"):
        return False
    if isinstance(dtype, np.dtype) and dtype.kind in ("i", "u", "f", "b"):
        return False
    return True


def prepare_X_xgb(X: pd.DataFrame) -> pd.DataFrame:
    """Normalize categorical dtypes for XGBoost 3.x.

    XGBoost 3.x dispatches categorical columns via ``pd_cat_inf`` which takes
    either a numeric path (``int64``/``float64`` category dtype) or a string
    path.  It fails with ``TypeError: object of type 'int' has no len()`` when
    the category index has ``object`` dtype — i.e., when categories mix Python
    ``int`` and ``str`` values.  This arises in the census schema because YAML
    entries like ``0: 0`` produce integer category *values* alongside string
    ones (e.g. ``[0, 1, …, 'No especificado', 'Blanco por pase']``).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix with mixed dtypes.

    Returns
    -------
    pd.DataFrame
        Copy with problematic categorical columns remapped to all-string
        categories; other columns are returned unchanged.
    """
    out = {}
    for c in X.columns:
        s = X[c]
        if (
            isinstance(s.dtype, pd.CategoricalDtype)
            and s.cat.categories.dtype == object
        ):
            # Remap every category label to str so XGBoost takes the string
            # path uniformly.  Existing string labels are unchanged; integer
            # labels (e.g. 0, 1) become "0", "1".
            mapping = {cat: str(cat) for cat in s.cat.categories}
            out[c] = s.cat.rename_categories(mapping)
        else:
            out[c] = s
    return pd.DataFrame(out, index=X.index)


def prepare_X_lgb(X: pd.DataFrame) -> pd.DataFrame:
    """Normalize column dtypes for LightGBM.

    **Ordered** ``CategoricalDtype`` columns are converted to ``float32``
    integer codes so LightGBM uses its numeric split algorithm, which respects
    the ordinal structure.  LightGBM's categorical split treats all categories
    as unordered, which loses ordinal information and empirically halves
    prediction accuracy for columns like ``EDAD_CAT`` (age band) or ``ESCOLARI``
    (years of schooling).  Ordinal meaning is preserved because the codes follow
    the definition order encoded in the ``CategoricalDtype``.

    **Unordered** ``CategoricalDtype`` columns with ``object``-dtype category
    labels (mixed ``int``/``str``) are remapped to uniform string categories so
    LightGBM's categorical split receives a consistently-typed index.  See
    :func:`prepare_X_xgb` for the root cause.  Unordered categoricals with
    homogeneous dtype are passed through unchanged.

    Pandas assigns code ``-1`` to ``NaN`` entries in ordered columns; these are
    mapped to ``np.nan`` so LightGBM's native missing-value handling applies.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix with mixed dtypes.

    Returns
    -------
    pd.DataFrame
        Copy with dtypes normalized:

        - Ordered ``CategoricalDtype`` → ``float32`` integer codes (``NaN`` for missing).
        - Unordered ``CategoricalDtype`` with ``object`` categories → string-remapped categorical.
        - Unordered ``CategoricalDtype`` with homogeneous dtype → kept as-is.
        - Nullable ``Int*`` / ``boolean`` → ``float64`` (``pd.NA`` → ``np.nan``).
        - NumPy numeric (``i``/``u``/``f``/``b``) → kept as-is.
        - Everything else → ``category`` (catches ``object``, ``StringDtype``, …).
    """
    cols = {}
    for c in X.columns:
        s = X[c]
        if isinstance(s.dtype, pd.CategoricalDtype):
            if s.dtype.ordered:
                # Ordered ordinal → numeric codes for proper ordered splits.
                codes = s.cat.codes.astype("float32")
                cols[c] = codes.where(codes >= 0, np.nan)  # -1 (NaN) → np.nan
            elif s.cat.categories.dtype == object:
                # Unordered with mixed int/str labels → remap to uniform strings.
                mapping = {cat: str(cat) for cat in s.cat.categories}
                cols[c] = s.cat.rename_categories(mapping)
            else:
                cols[c] = s
        elif hasattr(s.dtype, "numpy_dtype"):  # nullable Int*/boolean
            cols[c] = s.astype("float64")
        elif isinstance(s.dtype, np.dtype) and s.dtype.kind in ("i", "u", "f", "b"):
            cols[c] = s
        else:
            cols[c] = s.astype("category")
    return pd.DataFrame(cols, index=X.index)


def vectorized_categorical_sample(
    probs: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Sample one category per row from predicted class-probability distributions.

    Parameters
    ----------
    probs : np.ndarray, shape (n_rows, n_classes)
        Predicted probabilities; rows should sum to approximately 1.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray, shape (n_rows,)
        Integer class codes, one per row.
    """
    cumprobs = np.cumsum(probs, axis=1)
    uniforms = rng.random(size=len(probs)).reshape(-1, 1)
    return (uniforms < cumprobs).argmax(axis=1)


# ── LightGBM backend ──────────────────────────────────────────────────────────


def _impute_lightgbm(
    X_train: pd.DataFrame,
    y_codes: np.ndarray,
    w_train: np.ndarray,
    X_predict: pd.DataFrame,
    n_classes: int,
    cat_cols: List[str],
    seed: int,
    params: dict,
) -> np.ndarray:
    """Train a LightGBM classifier and return predicted class probabilities.

    Parameters
    ----------
    X_train : pd.DataFrame
        Feature matrix for training rows.
    y_codes : np.ndarray
        Integer class codes for training rows.
    w_train : np.ndarray
        Sample weights for training rows.
    X_predict : pd.DataFrame
        Feature matrix for rows to impute.
    n_classes : int
        Number of distinct target categories.
    cat_cols : List[str]
        Column names to treat as categorical features.
    seed : int
        Integer seed for the LightGBM RNG, derived from the caller's
        ``np.random.Generator``; overridable via ``params["seed"]``.
    params : dict
        Override any default LightGBM hyperparameters.

    Returns
    -------
    np.ndarray, shape (n_predict, n_classes)
        Predicted class probabilities.
    """
    defaults = {
        "max_depth": 6,
        "num_leaves": 63,
        "min_data_in_leaf": 20,
        "learning_rate": 0.05,
        "seed": seed,
        "verbose": -1,
        "num_threads": -1,
    }
    p = {**defaults, **params}
    num_boost_round = p.pop("num_boost_round", 200)

    if n_classes == 2:
        p["objective"] = "binary"
        p["metric"] = "binary_logloss"
    else:
        p["objective"] = "multiclass"
        p["num_class"] = n_classes
        p["metric"] = "multi_logloss"

    X_tr = prepare_X_lgb(X_train)
    X_pr = prepare_X_lgb(X_predict)
    # Ordered categoricals were converted to float32 codes by prepare_X_lgb;
    # only pass columns that are still CategoricalDtype as categorical_feature.
    actual_cat_cols = [
        c for c in cat_cols
        if c in X_tr.columns and isinstance(X_tr[c].dtype, pd.CategoricalDtype)
    ]

    train_set = lgb.Dataset(
        X_tr,
        label=y_codes,
        weight=w_train,
        categorical_feature=actual_cat_cols or "auto",
        free_raw_data=False,
    )
    model = lgb.train(p, train_set, num_boost_round=num_boost_round)
    probs = model.predict(X_pr)
    if probs.ndim == 1:  # binary: 1D probability of class 1
        probs = np.column_stack([1 - probs, probs])
    return probs


# ── CatBoost backend (optional — for Step 11 bake-off) ───────────────────────


def _impute_catboost(
    X_train: pd.DataFrame,
    y_codes: np.ndarray,
    w_train: np.ndarray,
    X_predict: pd.DataFrame,
    n_classes: int,
    cat_cols: List[str],
    seed: int,
    params: dict,
) -> np.ndarray:
    """Train a CatBoost classifier and return predicted class probabilities.

    Parameters
    ----------
    X_train : pd.DataFrame
        Feature matrix for training rows.
    y_codes : np.ndarray
        Integer class codes for training rows.
    w_train : np.ndarray
        Sample weights for training rows.
    X_predict : pd.DataFrame
        Feature matrix for rows to impute.
    n_classes : int
        Number of distinct target categories.
    cat_cols : List[str]
        Column names to treat as categorical features.
    seed : int
        Integer seed for the CatBoost RNG, derived from the caller's
        ``np.random.Generator``; overridable via ``params["random_seed"]``.
    params : dict
        Override any default CatBoost hyperparameters.

    Returns
    -------
    np.ndarray, shape (n_predict, n_classes)
        Predicted class probabilities.
    """
    try:
        from catboost import CatBoostClassifier
    except ImportError as e:
        raise ImportError(
            "The 'catboost' backend requires the optional dependency catboost. "
            "Install it with: pip install 'linkedmice[catboost]'"
        ) from e

    defaults = {
        "iterations": 200,
        "depth": 6,
        "min_data_in_leaf": 20,
        "learning_rate": 0.05,
        "random_seed": seed,
        "verbose": 0,
        "thread_count": -1,
        # "Ordered" boosting (CatBoost default) is 2-4× slower than LightGBM
        # for cross-sectional data with no natural ordering.  "Plain" is standard
        # gradient boosting, equivalent to LightGBM/XGBoost, and is much faster.
        "boosting_type": "Plain",
    }
    p = {**defaults, **params}

    # CatBoost accepts string-encoded categoricals; encode as str, NaN → "NaN"
    X_tr = X_train.copy()
    X_pr = X_predict.copy()
    for c in cat_cols:
        if c in X_tr.columns:
            X_tr[c] = X_tr[c].astype(str).where(X_tr[c].notna(), "NaN")
            X_pr[c] = X_pr[c].astype(str).where(X_pr[c].notna(), "NaN")
    cat_indices = [X_tr.columns.get_loc(c) for c in cat_cols if c in X_tr.columns]

    if n_classes == 2:
        p["loss_function"] = "Logloss"
    else:
        p["loss_function"] = "MultiClass"
        p["classes_count"] = n_classes

    model = CatBoostClassifier(**p, cat_features=cat_indices)
    model.fit(X_tr, y_codes, sample_weight=w_train)
    return model.predict_proba(X_pr)


# ── XGBoost backend (optional — for Step 11 bake-off) ────────────────────────


def _impute_xgboost(
    X_train: pd.DataFrame,
    y_codes: np.ndarray,
    w_train: np.ndarray,
    X_predict: pd.DataFrame,
    n_classes: int,
    cat_cols: List[str],
    seed: int,
    params: dict,
) -> np.ndarray:
    """Train an XGBoost classifier and return predicted class probabilities.

    Parameters
    ----------
    X_train : pd.DataFrame
        Feature matrix for training rows.
    y_codes : np.ndarray
        Integer class codes for training rows.
    w_train : np.ndarray
        Sample weights for training rows.
    X_predict : pd.DataFrame
        Feature matrix for rows to impute.
    n_classes : int
        Number of distinct target categories.
    cat_cols : List[str]
        Column names to treat as categorical features.
    seed : int
        Integer seed for the XGBoost RNG, derived from the caller's
        ``np.random.Generator``; overridable via ``params["seed"]``.
    params : dict
        Override any default XGBoost hyperparameters.

    Returns
    -------
    np.ndarray, shape (n_predict, n_classes)
        Predicted class probabilities.
    """
    try:
        import xgboost as xgb
    except ImportError as e:
        raise ImportError(
            "The 'xgboost' backend requires the optional dependency xgboost. "
            "Install it with: pip install 'linkedmice[xgboost]'"
        ) from e

    defaults = {
        "max_depth": 6,
        "min_child_weight": 20,
        "learning_rate": 0.05,
        "seed": seed,
        "verbosity": 0,
        "nthread": -1,
        "tree_method": "hist",
    }
    p = {**defaults, **params}
    num_boost_round = p.pop("num_boost_round", 200)

    if n_classes == 2:
        p["objective"] = "binary:logistic"
    else:
        p["objective"] = "multi:softprob"
        p["num_class"] = n_classes

    X_tr = prepare_X_xgb(X_train)
    X_pr = prepare_X_xgb(X_predict)
    dtrain = xgb.DMatrix(
        X_tr, label=y_codes, weight=w_train, enable_categorical=True
    )
    dpredict = xgb.DMatrix(X_pr, enable_categorical=True)
    model = xgb.train(p, dtrain, num_boost_round=num_boost_round)
    probs = model.predict(dpredict)
    if probs.ndim == 1:
        probs = np.column_stack([1 - probs, probs])
    return probs


# ── Top-level dispatcher ──────────────────────────────────────────────────────


def weighted_impute_categorical(
    df: pd.DataFrame,
    target_col: str,
    predictor_cols: List[str],
    missing_mask: pd.Series,
    skip_mask: pd.Series,
    weights: pd.Series,
    rng: np.random.Generator,
    backend: BackendName = "lightgbm",
    backend_params: Optional[dict] = None,
    deterministic: bool = False,
) -> pd.Series:
    """Impute NaN cells of ``target_col`` using a weighted gradient-boosted model.

    Training rows are those that are observed (not missing, not BPP).
    Prediction rows are those in ``missing_mask``.  By default, imputed values
    are sampled stochastically from the predicted class-probability
    distributions to preserve marginal distributions under synthesis.  Set
    ``deterministic=True`` to assign the highest-probability class instead
    (argmax), which eliminates sampling noise at the cost of underestimating
    uncertainty.

    Parameters
    ----------
    df : pd.DataFrame
        Working DataFrame containing ``target_col`` and all predictor columns.
    target_col : str
        Column to impute; must be a ``CategoricalDtype`` column.
    predictor_cols : List[str]
        Candidate predictor column names (columns absent from ``df`` are skipped).
    missing_mask : pd.Series
        Boolean Series, ``True`` for rows to impute (NaN / item non-response).
    skip_mask : pd.Series
        Boolean Series, ``True`` for BPP rows; excluded from training and prediction.
    weights : pd.Series
        Sample weights aligned to ``df.index`` (e.g. the ``FACTOR`` column).
    rng : np.random.Generator
        Random number generator.  Used for two purposes: (1) deriving an
        integer seed for the tree backend before training, and (2) sampling
        one category per row from the predicted probabilities (ignored when
        ``deterministic=True``).
    backend : {"lightgbm", "catboost", "xgboost"}
        Gradient-boosted tree backend to use.
    backend_params : dict, optional
        Hyperparameter overrides forwarded to the backend.
    deterministic : bool, optional
        If ``True``, assign the highest-probability class (argmax) instead of
        sampling.  Default ``False``.

    Returns
    -------
    pd.Series
        Imputed values aligned to ``df.loc[missing_mask].index``.
    """
    if not missing_mask.any():
        return pd.Series([], dtype=df[target_col].dtype)

    train_idx = ~missing_mask & ~skip_mask
    predict_idx = missing_mask

    available_preds = [c for c in predictor_cols if c in df.columns]

    X_train = df.loc[train_idx, available_preds]
    y_train = df.loc[train_idx, target_col]
    w_train = weights.loc[train_idx].fillna(1.0).values.astype(float)
    w_train = (
        w_train / w_train.mean()
    )  # normalize: preserve relative differences, stabilize gradient scale
    X_predict = df.loc[predict_idx, available_preds]

    # Drop categories absent from the training rows (includes BPP, excluded by skip_mask)
    y_train = y_train.cat.remove_unused_categories()
    categories = y_train.cat.categories.tolist()

    n_classes = len(categories)
    if n_classes < 2:
        val = categories[0] if n_classes == 1 else pd.NA
        return pd.Series(
            [val] * predict_idx.sum(),
            index=df.loc[predict_idx].index,
            dtype=df[target_col].dtype,
        )

    y_codes = y_train.cat.codes.values
    assert (y_codes >= 0).all(), (
        f"Unexpected -1 codes in '{target_col}' — check skip/missing masks"
    )

    cat_cols = [
        c for c in available_preds if isinstance(df[c].dtype, pd.CategoricalDtype)
    ]

    # Convert non-numeric, non-categorical columns (object, StringDtype, etc.)
    # to pd.CategoricalDtype. Categories are derived from the full df column so
    # codes are consistent between X_train and X_predict.
    _str_cols = [
        c
        for c in available_preds
        if c in df.columns and _needs_cat_conversion(df[c].dtype)
    ]
    if _str_cols:
        X_train = X_train.copy()
        X_predict = X_predict.copy()
        for _c in _str_cols:
            _cats = sorted(df[_c].dropna().unique().tolist())
            _cat_dtype = pd.CategoricalDtype(categories=_cats, ordered=False)
            X_train[_c] = X_train[_c].astype(_cat_dtype)
            X_predict[_c] = X_predict[_c].astype(_cat_dtype)
        cat_cols = cat_cols + [c for c in _str_cols if c not in cat_cols]

    backend_params = backend_params or {}
    # Draw tree seed from rng so backend randomness is reproducible through
    # the same rng as the sampling step.  backend_params can still override.
    tree_seed = int(rng.integers(0, 2**31))
    if backend == "lightgbm":
        probs = _impute_lightgbm(
            X_train,
            y_codes,
            w_train,
            X_predict,
            n_classes,
            cat_cols,
            tree_seed,
            backend_params,
        )
    elif backend == "catboost":
        probs = _impute_catboost(
            X_train,
            y_codes,
            w_train,
            X_predict,
            n_classes,
            cat_cols,
            tree_seed,
            backend_params,
        )
    elif backend == "xgboost":
        probs = _impute_xgboost(
            X_train,
            y_codes,
            w_train,
            X_predict,
            n_classes,
            cat_cols,
            tree_seed,
            backend_params,
        )
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    if deterministic:
        chosen_codes = probs.argmax(axis=1)
    else:
        chosen_codes = vectorized_categorical_sample(probs, rng)
    sampled_cats = pd.Categorical.from_codes(chosen_codes, categories=categories)
    return pd.Series(
        sampled_cats, index=df.loc[predict_idx].index, dtype=df[target_col].dtype
    )


def build_predictor_registry(
    df: pd.DataFrame,
    targets: List[str],
    exclude_cols: set,
    derived_suffixes: Tuple[str, ...] = ("_CAT",),
) -> Dict[str, List[str]]:
    """Build a per-target predictor-column registry from the current DataFrame.

    The predictor set for each target is all columns in ``df`` minus
    ``exclude_cols`` minus the target itself.  For targets carrying one of the
    ``derived_suffixes``, the deterministic source column (same name without
    the suffix) is also excluded to prevent leakage.

    Rebuilt on every feature-refresh so newly engineered columns are included
    automatically.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose columns define the available feature universe.
    targets : List[str]
        Target columns to build registries for.
    exclude_cols : set
        Columns to exclude from every predictor set (e.g. weight column,
        derived dummies).
    derived_suffixes : Tuple[str, ...], optional
        Suffixes marking targets derived deterministically from a source
        column of the same name without the suffix; that source column is
        excluded from the target's predictors.  Default ``("_CAT",)``;
        pass ``()`` to disable.

    Returns
    -------
    Dict[str, List[str]]
        ``{target: sorted list of predictor column names}``.  Targets not
        present in ``df`` are omitted.
    """
    available = set(df.columns) - exclude_cols
    registry = {}
    for target in targets:
        if target not in df.columns:
            continue
        exclude = {target}
        for suf in derived_suffixes:
            if target.endswith(suf):
                exclude.add(target.removesuffix(suf))
        registry[target] = sorted(available - exclude)
    return registry


# ── Feature-refresh helpers ───────────────────────────────────────────────────


def refresh_per_broadcast(
    per_df: pd.DataFrame,
    hh_df: pd.DataFrame,
    hh_broadcast_cols: List[str],
    hh_key: str = "ID_VIV",
) -> pd.DataFrame:
    """Drop stale ``_hh``-suffixed columns and re-broadcast household attributes.

    Parameters
    ----------
    per_df : pd.DataFrame
        Person working DataFrame.
    hh_df : pd.DataFrame
        Household working DataFrame (source of broadcast columns).
    hh_broadcast_cols : List[str]
        Household columns to broadcast into the person table.
    hh_key : str, optional
        Name of the household-key index level.  Default ``"ID_VIV"``.

    Returns
    -------
    pd.DataFrame
        Updated person DataFrame with refreshed ``_hh``-suffixed columns.
    """
    old = [c for c in per_df.columns if c.endswith("_hh")]
    if old:
        per_df = per_df.drop(columns=old)
    return broadcast_household_attrs(per_df, hh_df, hh_broadcast_cols, hh_key)


def refresh_loo(
    per_df: pd.DataFrame,
    cols: List[str],
    hh_key: str = "ID_VIV",
    bpp_value: str = DEFAULT_BPP,
) -> pd.DataFrame:
    """Recompute leave-one-out household-member category counts.

    Parameters
    ----------
    per_df : pd.DataFrame
        Person working DataFrame indexed by ``(hh_key, person_id)``.
    cols : List[str]
        Columns for which to recompute LOO counts.
    hh_key : str, optional
        Name of the household-key index level.  Default ``"ID_VIV"``.
    bpp_value : str, optional
        Structural-skip sentinel string.  Default :data:`DEFAULT_BPP`.

    Returns
    -------
    pd.DataFrame
        Updated person DataFrame with refreshed ``n_other_{col}_*`` columns.
    """
    for col in cols:
        old = [c for c in per_df.columns if c.startswith(f"n_other_{col}_")]
        if old:
            per_df = per_df.drop(columns=old)
        per_df = pd.concat(
            [per_df, leave_one_out_category_counts(per_df, col, hh_key, bpp_value)],
            axis=1,
        )
    return per_df


# ── Diagnostics ───────────────────────────────────────────────────────────────


def record_diagnostics(
    hh_df: pd.DataFrame,
    per_df: pd.DataFrame,
    hh_mm: Dict[str, pd.Series],
    per_mm: Dict[str, pd.Series],
    hh_impute_targets: List[str],
    person_impute_targets: List[str],
    iteration: int,
) -> dict:
    """Snapshot imputed-cell value distributions for convergence monitoring.

    Parameters
    ----------
    hh_df : pd.DataFrame
        Household working DataFrame after the current iteration.
    per_df : pd.DataFrame
        Person working DataFrame after the current iteration.
    hh_mm : Dict[str, pd.Series]
        Household missing masks (identifies which rows were imputed).
    per_mm : Dict[str, pd.Series]
        Person missing masks.
    hh_impute_targets : List[str]
        Household columns being imputed.
    person_impute_targets : List[str]
        Person columns being imputed.
    iteration : int
        Current iteration number (stored in the snapshot for identification).

    Returns
    -------
    dict
        Keys: ``"iteration"``, plus ``"hh_{col}"`` and ``"per_{col}"`` for every
        imputed column.  Values for column keys are normalised value-count dicts.
    """
    d: dict = {"iteration": iteration}
    for col in hh_impute_targets:
        if col in hh_df.columns and hh_mm[col].any():
            d[f"hh_{col}"] = (
                hh_df.loc[hh_mm[col], col]
                .value_counts(normalize=True, dropna=False)
                .to_dict()
            )
    for col in person_impute_targets:
        if col in per_df.columns and per_mm[col].any():
            d[f"per_{col}"] = (
                per_df.loc[per_mm[col], col]
                .value_counts(normalize=True, dropna=False)
                .to_dict()
            )
    return d


def post_imputation_repair(
    hh_df: pd.DataFrame,
    per_df: pd.DataFrame,
    hh_mm: Dict[str, pd.Series],
    per_mm: Dict[str, pd.Series],
    hh_sm: Dict[str, pd.Series],
    per_sm: Dict[str, pd.Series],
    hh_skip_deps: List[SkipDep],
    per_skip_deps: List[SkipDep],
    bpp_value: str = DEFAULT_BPP,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, pd.Series],
    Dict[str, pd.Series],
    Dict[str, pd.Series],
    Dict[str, pd.Series],
]:
    """Final consistency pass after the MICE loop.

    1. Re-applies :func:`apply_consistency_repair` for both tables.
    2. Checks the BPP invariant — every structurally-skipped row must still
       hold the BPP value — and prints a warning for any violations.

    Parameters
    ----------
    hh_df : pd.DataFrame
        Household imputed DataFrame; modified in-place.
    per_df : pd.DataFrame
        Person imputed DataFrame; modified in-place.
    hh_mm : Dict[str, pd.Series]
        Household missing masks; updated in-place.
    per_mm : Dict[str, pd.Series]
        Person missing masks; updated in-place.
    hh_sm : Dict[str, pd.Series]
        Household skip masks; updated in-place.
    per_sm : Dict[str, pd.Series]
        Person skip masks; updated in-place.
    hh_skip_deps : List[SkipDep]
        Household skip dependency triples ``(parent_cols, predicate, child)``.
    per_skip_deps : List[SkipDep]
        Person skip dependency triples.
    bpp_value : str, optional
        Structural-skip sentinel string.  Default :data:`DEFAULT_BPP`.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict, Dict, Dict]
        ``(hh_df, per_df, hh_mm, per_mm, hh_sm, per_sm)`` — 6-tuple with
        updated masks.
    """
    apply_consistency_repair(hh_df, hh_mm, hh_sm, hh_skip_deps, bpp_value)
    apply_consistency_repair(per_df, per_mm, per_sm, per_skip_deps, bpp_value)

    total_violations = 0
    for label, df, sm in [("HH", hh_df, hh_sm), ("Per", per_df, per_sm)]:
        for col, skip_rows in sm.items():
            if col not in df.columns or not skip_rows.any():
                continue
            if not hasattr(df[col], "cat"):
                continue
            violations = skip_rows & (df.loc[skip_rows.index, col] != bpp_value)
            n = int(violations.sum())
            if n:
                print(f"  WARNING ({label}) {col}: {n} BPP rows no longer hold BPP")
                total_violations += n

    if total_violations == 0:
        print("post_imputation_repair: BPP invariant holds — no violations.")
    else:
        print(f"post_imputation_repair: {total_violations} total BPP violations.")

    return hh_df, per_df, hh_mm, per_mm, hh_sm, per_sm


def integrated_mice(
    hh_df: pd.DataFrame,
    per_df: pd.DataFrame,
    hh_missing_mask_in: Dict[str, pd.Series],
    person_missing_mask_in: Dict[str, pd.Series],
    hh_skip_masks_in: Dict[str, pd.Series],
    person_skip_masks_in: Dict[str, pd.Series],
    hh_impute_targets: List[str],
    person_impute_targets: List[str],
    hh_exclude: set,
    per_exclude: set,
    hh_skip_deps: List[SkipDep],
    per_skip_deps: List[SkipDep],
    person_high_priority_targets: List[str],
    hh_broadcast_cols: List[str],
    hh_feature_hooks: Optional[List[FeatureHook]] = None,
    per_feature_hooks: Optional[List[FeatureHook]] = None,
    hh_key: str = "ID_VIV",
    bpp_value: str = DEFAULT_BPP,
    derived_suffixes: Tuple[str, ...] = ("_CAT",),
    weight_col: str = "FACTOR",
    n_iterations: int = 5,
    backend: BackendName = "lightgbm",
    backend_params: Optional[dict] = None,
    rng: Optional[np.random.Generator] = None,
    deterministic: bool = False,
    verbose: bool = True,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, pd.Series],
    Dict[str, pd.Series],
    Dict[str, pd.Series],
    Dict[str, pd.Series],
    List[dict],
]:
    """Run the integrated two-table MICE imputation loop.

    Alternates between a household block and a person block each iteration.
    Cross-table features are refreshed at the start of each block so models
    capture within-iteration correlation between tables.

    Works entirely on internal deep copies — the caller's DataFrames and masks
    are never modified.

    Parameters
    ----------
    hh_df : pd.DataFrame
        Household working DataFrame (output of :func:`initial_fill`).
    per_df : pd.DataFrame
        Person working DataFrame (output of :func:`initial_fill`).
    hh_missing_mask_in : Dict[str, pd.Series]
        Household NaN masks produced by :func:`build_missing_masks`.
    person_missing_mask_in : Dict[str, pd.Series]
        Person NaN masks.
    hh_skip_masks_in : Dict[str, pd.Series]
        Household BPP masks produced by :func:`build_skip_masks`.
    person_skip_masks_in : Dict[str, pd.Series]
        Person BPP masks.
    hh_impute_targets : List[str]
        Household columns to impute.
    person_impute_targets : List[str]
        Person columns to impute.
    hh_exclude : set
        Household columns to exclude from predictor sets (e.g. ``{"FACTOR"}``).
    per_exclude : set
        Person columns to exclude from predictor sets.
    hh_skip_deps : List[SkipDep]
        Household skip dependency triples ``(parent_cols, predicate, child_col)``.
    per_skip_deps : List[SkipDep]
        Person skip dependency triples.
    person_high_priority_targets : List[str]
        Person targets whose LOO counts are refreshed immediately before each
        target is imputed (used as predictors for subsequent targets).
    hh_broadcast_cols : List[str]
        Household columns to broadcast into the person table as predictors.
    hh_feature_hooks : List[FeatureHook], optional
        Cross-table feature-refresh hooks applied at the start of each
        household block, in order: ``hh = hook(hh, per)``.
    per_feature_hooks : List[FeatureHook], optional
        Feature-refresh hooks applied at the start of each person block,
        after the built-in household-attribute broadcast and before the
        built-in LOO refresh, in order: ``per = hook(per, hh)``.
    hh_key : str, optional
        Name of the household-key index level shared by both tables.
        Default ``"ID_VIV"``.
    bpp_value : str, optional
        Structural-skip sentinel string.  Default :data:`DEFAULT_BPP`.
    derived_suffixes : Tuple[str, ...], optional
        Suffixes marking targets derived from a same-named source column;
        forwarded to :func:`build_predictor_registry`.  Default ``("_CAT",)``.
    weight_col : str, optional
        Name of the sample-weight column in both DataFrames.  Default ``"FACTOR"``.
    n_iterations : int, optional
        Number of MICE iterations.  Default ``5``.
    backend : {"lightgbm", "catboost", "xgboost"}
        Gradient-boosted tree backend.  Default ``"lightgbm"``.
    backend_params : dict, optional
        Hyperparameter overrides forwarded to the backend.
    rng : np.random.Generator, optional
        Random number generator.  Defaults to ``np.random.default_rng(42)``.
    deterministic : bool, optional
        If ``True``, each column is imputed by argmax (highest-probability
        class) rather than stochastic sampling.  Eliminates iteration-to-
        iteration variation; useful for studying model bias without sampling
        noise.  Default ``False``.
    verbose : bool, optional
        Print iteration and per-column progress.  Default ``True``.

    Returns
    -------
    Tuple
        7-tuple ``(hh_imputed, per_imputed, hh_mm, per_mm, hh_sm, per_sm,
        diagnostics_list)`` where the mask dicts include any rows dynamically
        promoted to skip status during the loop.
    """
    if rng is None:
        rng = np.random.default_rng(_DEFAULT_SEED)

    # Deep copies — isolation from caller
    hh = hh_df.copy()
    per = per_df.copy()
    hh_mm = {k: v.copy() for k, v in hh_missing_mask_in.items()}
    per_mm = {k: v.copy() for k, v in person_missing_mask_in.items()}
    hh_sm = {k: v.copy() for k, v in hh_skip_masks_in.items()}
    per_sm = {k: v.copy() for k, v in person_skip_masks_in.items()}

    # Pre-sort targets by ascending original missingness (stable order)
    hh_targets_ordered = sorted(
        [
            c
            for c in hh_impute_targets
            if c in hh.columns and hh_mm.get(c, pd.Series(dtype=bool)).any()
        ],
        key=lambda c: hh_mm[c].sum(),
    )
    per_targets_ordered = sorted(
        [
            c
            for c in person_impute_targets
            if c in per.columns and per_mm.get(c, pd.Series(dtype=bool)).any()
        ],
        key=lambda c: per_mm[c].sum(),
    )

    hh_weights = hh[weight_col].fillna(1.0).astype(float)
    per_weights = per[weight_col].fillna(1.0).astype(float)

    all_diagnostics = []

    for iteration in range(1, n_iterations + 1):
        t0 = time.time()
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Iteration {iteration}/{n_iterations}  backend={backend}")
            print(f"{'=' * 60}")

        # ── HOUSEHOLD BLOCK ───────────────────────────────────────────────────

        for hook in hh_feature_hooks or []:
            hh = hook(hh, per)
        hh_registry = build_predictor_registry(
            hh, hh_impute_targets, hh_exclude, derived_suffixes
        )

        for target in hh_targets_ordered:
            if verbose:
                print(f"  HH  {target}  ({hh_mm[target].sum()} missing)", flush=True)
            imputed = weighted_impute_categorical(
                df=hh,
                target_col=target,
                predictor_cols=hh_registry[target],
                missing_mask=hh_mm[target],
                skip_mask=hh_sm[target],
                weights=hh_weights,
                rng=rng,
                backend=backend,
                backend_params=backend_params,
                deterministic=deterministic,
            )
            hh.loc[hh_mm[target], target] = imputed.values
            refresh_dependent_skips(hh, hh_mm, hh_sm, target, hh_skip_deps, bpp_value)

        # ── PERSON BLOCK ──────────────────────────────────────────────────────

        per = refresh_per_broadcast(per, hh, hh_broadcast_cols, hh_key)
        for hook in per_feature_hooks or []:
            per = hook(per, hh)
        per = refresh_loo(per, person_high_priority_targets, hh_key, bpp_value)

        per_registry = build_predictor_registry(
            per, person_impute_targets, per_exclude, derived_suffixes
        )

        for target in per_targets_ordered:
            if verbose:
                print(f"  Per {target}  ({per_mm[target].sum()} missing)", flush=True)

            # Refresh LOO for high-priority targets right before they are imputed
            if target in person_high_priority_targets:
                per = refresh_loo(per, [target], hh_key, bpp_value)
                per_registry = build_predictor_registry(
                    per, person_impute_targets, per_exclude, derived_suffixes
                )

            imputed = weighted_impute_categorical(
                df=per,
                target_col=target,
                predictor_cols=per_registry[target],
                missing_mask=per_mm[target],
                skip_mask=per_sm[target],
                weights=per_weights,
                rng=rng,
                backend=backend,
                backend_params=backend_params,
                deterministic=deterministic,
            )
            per.loc[per_mm[target], target] = imputed.values
            refresh_dependent_skips(per, per_mm, per_sm, target, per_skip_deps, bpp_value)

        # ── End-of-iteration consistency repair ───────────────────────────────
        apply_consistency_repair(hh, hh_mm, hh_sm, hh_skip_deps, bpp_value)
        apply_consistency_repair(per, per_mm, per_sm, per_skip_deps, bpp_value)

        # ── Diagnostics ───────────────────────────────────────────────────────
        diag = record_diagnostics(
            hh, per, hh_mm, per_mm, hh_impute_targets, person_impute_targets, iteration
        )
        all_diagnostics.append(diag)

        elapsed = time.time() - t0
        if verbose:
            print(f"\nIteration {iteration} done in {elapsed:.1f}s")

    # Lazy import to avoid circular dependency (reporting imports SkipDep from mice).
    from .reporting import post_mice_sanity_checks

    post_mice_sanity_checks(
        hh, per,
        hh_missing_mask_in, person_missing_mask_in,
        hh_sm, per_sm,
        hh_impute_targets, person_impute_targets,
        bpp_value,
    )

    return hh, per, hh_mm, per_mm, hh_sm, per_sm, all_diagnostics
