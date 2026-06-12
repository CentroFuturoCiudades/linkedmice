"""Mexico Census 2020 (cuestionario ampliado) helpers for the worked example.

Census-specific feature engineering, post-imputation column rebuilds, and a
portable loader for the linked Viviendas/Personas tables.  Everything in this
module hardcodes census column names and category labels — it is the example
layer on top of the generic ``linkedmice`` engine, and is the only place that
depends on ``mxcensus`` (install with the ``linkedmice[census]`` extra).
"""

from pathlib import Path
from typing import List, Tuple

import pandas as pd

from mxcensus import load_extended_personas, load_extended_viviendas
from mxcensus.extended_personas import (
    dhsersal_create_dummies,
    dis_create_agg_cols,
    get_educ_col,
    med_traslado_esc_create_dummies,
    med_traslado_trab_create_dummies,
)
from mxcensus.extended_viviendas import financiamiento_create_dummies

BPP = "Blanco por pase"

# ── PARENTESCO role groups (category strings after preprocessing) ─────────────
HEAD_VALS = ["Jefa(e)"]
PARTNER_VALS = ["Esposa(o)", "Concubina(o) o unión libre", "Amante o querida(o)"]
CHILD_VALS = ["Hija(o)", "Hija(o) adoptiva(o)", "Hijastra(o)", "Hija(o) de crianza"]

# ── EDAD_CAT groupings ────────────────────────────────────────────────────────
CHILD_AGE_CATS = ["0-2", "3-4", "5", "6-7", "8-11", "12-14", "15-17"]
ADULT_AGE_CATS = ["18-24", "25-49", "50-59", "60-64", "65-130"]


# ── Data loading ──────────────────────────────────────────────────────────────


def load_census_tables(
    data_dir: Path | None = None, state: int = 14
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the (viviendas, personas) extended-questionnaire tables.

    Parameters
    ----------
    data_dir : Path, optional
        Directory holding ``Viviendas{state}.parquet`` and
        ``Personas{state}.parquet``.  When omitted, the raw parquets are
        fetched from the mxcensus mirror via Pooch (cached locally).
    state : int, optional
        INEGI state code.  Default ``14`` (Jalisco).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        ``(df_viviendas, df_personas)``.
    """
    if data_dir is not None:
        df_viv = load_extended_viviendas(data_dir / f"Viviendas{state}.parquet")
        df_per = load_extended_personas(data_dir / f"Personas{state}.parquet")
    else:
        df_viv = load_extended_viviendas(state=state)
        df_per = load_extended_personas(state=state)
    return df_viv, df_per


# ── Census feature engineering ────────────────────────────────────────────────


def compute_person_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute household-level aggregates from current person-table values.

    Parameters
    ----------
    df : pd.DataFrame
        Person DataFrame indexed by ``(ID_VIV, ID_PERSONA)``.

    Returns
    -------
    pd.DataFrame
        One row per household (indexed by ``ID_VIV``) with composition,
        economic activity, education, and head-attribute columns.
    """

    aggs = (
        df.assign(
            IS_ADULT=lambda df: df["EDAD_CAT"].isin(ADULT_AGE_CATS),
            IS_CHILD=lambda df: ~df["IS_ADULT"],
            IS_MALE=lambda df: df["SEXO"] == "Hombre",
            WORKS=lambda df: df["CONACT_CAT"] == "Trabaja",
            ESCOACUM_NO_BPP=lambda df: df["ESCOACUM"].replace(-1, pd.NA),
        )
        .groupby(level="ID_VIV")
        .agg(
            # Size composition
            # n_personas_per=("SEXO", "size"), # already NUMPERS
            n_children_per=("IS_CHILD", "sum"),
            n_adults_per=("IS_ADULT", "sum"),
            n_hombres_per=("IS_MALE", "sum"),
            # Economic activity
            n_trabaja_per=("WORKS", "sum"),
            # Education
            mean_escoacum=("ESCOACUM_NO_BPP", "mean"),
        )
        .assign(
            # Economic activity
            has_trabaja_per=lambda df: (df["n_trabaja_per"] > 0).astype("int8")
        )
    )

    # ── Household head attributes (carried forward for HH imputer) ────────────
    head_rows = df[df["PARENTESCO"].isin(HEAD_VALS)]
    head_by_viv = head_rows[
        [
            "SEXO",
            "EDAD_CAT",
            "NIVACAD",
            "CONACT_CAT",
            "OCUPACION_C_COARSE",
            "ACTIVIDADES_C_COARSE",
        ]
    ].copy()
    head_by_viv.index = head_by_viv.index.get_level_values("ID_VIV")
    head_by_viv = head_by_viv[~head_by_viv.index.duplicated(keep="first")]
    head_by_viv.columns = [
        "head_sexo_per",
        "head_edad_cat_per",
        "head_nivacad_per",
        "head_conact_per",
        "head_ocupa_per",
        "head_act_per",
    ]
    aggs = aggs.join(head_by_viv, how="left")

    return aggs


def compute_role_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute head and partner attribute features for each person.

    Self-leakage is avoided: ``head_*`` columns are ``NaN`` for the head's own
    row and ``partner_*`` columns are ``NaN`` for the partner's own row.

    Parameters
    ----------
    df : pd.DataFrame
        Person DataFrame indexed by ``(ID_VIV, ID_PERSONA)``.

    Returns
    -------
    pd.DataFrame
        DataFrame with the same index containing ``head_*`` and ``partner_*``
        attribute columns.
    """

    attributes = [
        "NIVACAD",
        "SEXO",
        "EDAD_CAT",
        "CONACT_CAT",
        "OCUPACION_C_COARSE",
        "ACTIVIDADES_C_COARSE",
    ]

    hh_ids = df.index.get_level_values("ID_VIV")

    is_head = df["PARENTESCO"].isin(HEAD_VALS)
    is_partner = df["PARENTESCO"].isin(PARTNER_VALS)

    role_df = pd.DataFrame(index=df.index)

    roles = [
        (is_head, "head"),
        (is_partner, "partner"),
    ]
    renamed = ["nivacad", "sexo", "edad_cat", "conact_cat", "ocupa", "act"]

    for mask, prefix in roles:
        src = df.loc[mask, attributes].copy()
        src.index = src.index.get_level_values("ID_VIV")
        src = src[~src.index.duplicated(keep="first")]
        src.columns = [f"{prefix}_{s}" for s in renamed]
        mapped = src.reindex(hh_ids).set_axis(df.index)
        role_df[src.columns] = mapped
        role_df.loc[mask, src.columns] = pd.NA

    return role_df


def compute_position_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute within-household positional features for each person.

    Parameters
    ----------
    df : pd.DataFrame
        Person DataFrame indexed by ``(ID_VIV, ID_PERSONA)``.

    Returns
    -------
    pd.DataFrame
        DataFrame with the same index containing ``is_head_pos``,
        ``is_partner_pos``, ``is_child_pos``, ``hh_size_pos``, and
        ``is_single_person_hh`` columns.
    """
    hh_ids = df.index.get_level_values("ID_VIV")
    hh_sizes = df.groupby(level="ID_VIV").size()

    pos = pd.DataFrame(index=df.index)
    pos["is_head_pos"] = df["PARENTESCO"].isin(HEAD_VALS).astype("int8")
    pos["is_partner_pos"] = df["PARENTESCO"].isin(PARTNER_VALS).astype("int8")
    pos["is_child_pos"] = df["PARENTESCO"].isin(CHILD_VALS).astype("int8")
    pos["hh_size_pos"] = hh_ids.map(hh_sizes).values
    pos["is_single_person_hh"] = (pos["hh_size_pos"] == 1).astype("int8")
    return pos


# ── Feature-refresh hooks (linkedmice.FeatureHook signature) ──────────────────


def refresh_hh_agg_features(hh_df: pd.DataFrame, per_df: pd.DataFrame) -> pd.DataFrame:
    """Recompute person→household aggregates and upsert them into ``hh_df``.

    Parameters
    ----------
    hh_df : pd.DataFrame
        Household working DataFrame; modified in-place and returned.
    per_df : pd.DataFrame
        Person working DataFrame used to compute aggregates.

    Returns
    -------
    pd.DataFrame
        ``hh_df`` with updated aggregate columns.
    """
    aggs = compute_person_aggregates(per_df)
    for col in aggs.columns:
        hh_df[col] = aggs.reindex(hh_df.index)[col]
    return hh_df


def refresh_per_role(
    per_df: pd.DataFrame, hh_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Recompute household-head and partner role features for each person.

    Parameters
    ----------
    per_df : pd.DataFrame
        Person working DataFrame indexed by ``(ID_VIV, ID_PERSONA)``.
    hh_df : pd.DataFrame, optional
        Ignored; accepted so the function matches ``linkedmice.FeatureHook``.

    Returns
    -------
    pd.DataFrame
        Updated person DataFrame with refreshed role-feature columns.
    """
    feats = compute_role_features(per_df)
    per_df = per_df.drop(columns=[c for c in feats.columns if c in per_df.columns])
    return pd.concat([per_df, feats], axis=1)


def refresh_per_position(
    per_df: pd.DataFrame, hh_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Recompute within-household positional features for each person.

    Parameters
    ----------
    per_df : pd.DataFrame
        Person working DataFrame indexed by ``(ID_VIV, ID_PERSONA)``.
    hh_df : pd.DataFrame, optional
        Ignored; accepted so the function matches ``linkedmice.FeatureHook``.

    Returns
    -------
    pd.DataFrame
        Updated person DataFrame with refreshed positional-feature columns.
    """
    feats = compute_position_features(per_df)
    per_df = per_df.drop(columns=[c for c in feats.columns if c in per_df.columns])
    return pd.concat([per_df, feats], axis=1)


# ── Post-imputation column rebuilds (deterministically derived columns) ───────


def rebuild_med_traslado_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild MED_TRASLADO_ESC_* and MED_TRASLADO_TRAB_* dummies from source columns."""
    esc_cols = [c for c in df.columns if c.startswith("MED_TRASLADO_ESC_")]
    trab_cols = [c for c in df.columns if c.startswith("MED_TRASLADO_TRAB_")]
    df = df.drop(columns=esc_cols + trab_cols)
    return pd.concat(
        [df, med_traslado_esc_create_dummies(df), med_traslado_trab_create_dummies(df)],
        axis=1,
    )


def rebuild_disability_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild DIS_CON and DIS_LIMI aggregates from individual disability columns."""
    df = df.drop(columns=[c for c in ["DIS_CON", "DIS_LIMI"] if c in df.columns])
    return pd.concat([df, dis_create_agg_cols(df)], axis=1)


def rebuild_educ_col(df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild EDUC from (imputed) NIVACAD and ESCOLARI."""
    df = df.drop(columns=["EDUC"], errors="ignore").copy()
    df["EDUC"] = get_educ_col(df)
    return df


def rebuild_financiamiento_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild FINANCIAMIENTO_* dummies from FINANCIAMIENTO1/2/3 source columns."""
    fin_cols = [c for c in df.columns if c.startswith("FINANCIAMIENTO_")]
    df = df.drop(columns=fin_cols)
    return pd.concat([df, financiamiento_create_dummies(df)], axis=1)


def rederive_dhsersal_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild DHSERSAL_* dummy columns from (imputed) DHSERSAL1 and DHSERSAL2.

    Drops all existing DHSERSAL_* columns and recreates them via
    dhsersal_create_dummies, which replicates the original preprocessing logic.
    Call this once after the MICE loop completes.
    """
    existing = [c for c in df.columns if c.startswith("DHSERSAL_")]
    df = df.drop(columns=existing)
    new_dummies = dhsersal_create_dummies(df)
    return pd.concat([df, new_dummies], axis=1)
