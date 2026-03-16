# -*- coding: utf-8 -*-
"""
Fallback base rates production pipeline.

Fits survival-model-based fallback rates at each hierarchy level, computes
iteration and launches-since-failure modifiers, and exports tabular outputs
(CSV + JSON).
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from zoneinfo import ZoneInfo

from ..utils.fit_fallback_base_rate import fit_rates_model
from ..utils.output_to_csv import output_df_to_csv
from ..utils.provider_rating_comparison_table import provider_rating_comparison_table
from .data_load_feature_creation import load_and_prepare_data

# -------------------- Configuration --------------------
TZ = ZoneInfo("Europe/London")
OUTPUT_DIR = Path("src/launch_analysis/outputs")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# -------------------- Levels config --------------------
# (level_name, attempt_col, bin_col, grouped_col)
BASE_RATE_LEVELS = [
    ("type", "lv_type_attempt_number"),
    ("family", "lv_family_attempt_number"),
    ("provider", "lv_provider_attempt_number"),
    ("variant", "lv_variant_attempt_number"),
    ("minor_variant", "lv_minor_variant_attempt_number"),
]

ITERATION_LEVELS = [
    ("type", "lv_type_attempt_number", "type_iteration_grouped"),
    ("variant", "lv_variant_attempt_number", "variant_iteration_grouped"),
    ("minor_variant", "lv_minor_variant_attempt_number", "minor_variant_iteration_grouped"),
]

LSF_LEVELS = [
    ("type", "lv_type_attempt_number", "type_SinceFail_Bin"),
    ("family", "lv_family_attempt_number", "family_SinceFail_Bin"),
    ("variant", "lv_variant_attempt_number", "variant_SinceFail_Bin"),
    ("minor_variant", "lv_minor_variant_attempt_number", "minor_variant_SinceFail_Bin"),
]

LSF_BINS = [("Clean", "clean"), ("1–3", "1_3"), (">=4", "ge4")]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_fit_rates_model(df: pd.DataFrame, context: str, **kwargs) -> pd.DataFrame:
    """Wrapper around fit_rates_model that handles empty input gracefully."""
    if df.empty:
        logger.warning("Slice '%s' is empty; returning empty DataFrame.", context)
        return pd.DataFrame()
    return fit_rates_model(df, **kwargs)


def _make_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has unique column names by adding .1, .2, ... suffixes."""
    counts: dict[str, int] = {}
    new_cols = []
    for c in df.columns:
        if c not in counts:
            counts[c] = 0
            new_cols.append(c)
        else:
            counts[c] += 1
            new_cols.append(f"{c}.{counts[c]}")
    df = df.copy()
    df.columns = new_cols
    return df


def _fit_single_level(df: pd.DataFrame, attempt_col: str, label: str,
                      context: str, max_attempt: int = 20) -> pd.DataFrame:
    """
    Fit rates for a single slice. Returns columns:
    ['attempt_number', label, f'empirical_{label}'] when available.
    """
    if attempt_col not in df.columns:
        logger.warning("Attempt column '%s' missing; returning empty for '%s'.", attempt_col, label)
        return pd.DataFrame(columns=["attempt_number", label, f"empirical_{label}"])

    grp = [("ignored", attempt_col, label)]
    fitted = _safe_fit_rates_model(df, context=context, groupings=grp, max_attempt=max_attempt)
    if fitted.empty:
        return fitted

    cols = ["attempt_number", label, f"empirical_{label}"]
    present = [c for c in cols if c in fitted.columns]
    return fitted[present] if present else fitted


def _rename_fitted_columns(df: pd.DataFrame, label: str, suffix: str) -> pd.DataFrame:
    """Rename fitted columns: label -> label_suffix, empirical_label -> empirical_label_suffix."""
    if df.empty:
        return pd.DataFrame(columns=[
            "attempt_number", f"{label}_{suffix}", f"empirical_{label}_{suffix}",
        ])
    return df.rename(columns={
        label: f"{label}_{suffix}",
        f"empirical_{label}": f"empirical_{label}_{suffix}",
    })


def _export_csv_json(df: pd.DataFrame, base_path: Path) -> None:
    """Export a DataFrame as both CSV and JSON."""
    export_df = _make_unique_columns(df)
    export_df.to_csv(base_path, index=False)
    export_df.to_json(base_path.with_suffix(".json"), orient="records", indent=2)


# ---------------------------------------------------------------------------
# Base rates
# ---------------------------------------------------------------------------

def _fit_base_for_level(df: pd.DataFrame, attempt_col: str, level_name: str) -> pd.DataFrame:
    """Fit base rates for a specific hierarchy level."""
    return _fit_single_level(df, attempt_col, level_name,
                             context=f"{level_name}_base")


def _base_rates(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Fit fallback base rates for all hierarchy levels."""
    logger.info("Fitting fallback base rates for all levels...")
    outputs = {
        name: _fit_base_for_level(df, attempt_col, name)
        for name, attempt_col in BASE_RATE_LEVELS
    }
    for k, v in outputs.items():
        logger.info("Level '%s' produced %d rows", k, len(v))
    return outputs


# ---------------------------------------------------------------------------
# Modifier 1: Iteration (first vs second+)
# ---------------------------------------------------------------------------

def _modifier_iteration_for_level(
    df: pd.DataFrame,
    attempt_col: str,
    grouped_col: str,
    label: str,
    max_attempt: int = 20,
) -> pd.DataFrame:
    """
    Fit iteration modifier for a level using 'first' vs 'second+' slices.

    Output columns: attempt_number, {label}_first, empirical_{label}_first,
                    {label}_second_plus, empirical_{label}_second_plus
    """
    if grouped_col not in df.columns:
        raise KeyError(f"Expected column '{grouped_col}' not found for level '{label}'.")
    if attempt_col not in df.columns:
        logger.warning("Attempt column '%s' missing for '%s'; returning empty.", attempt_col, label)
        return pd.DataFrame(columns=[
            "attempt_number",
            f"{label}_first", f"empirical_{label}_first",
            f"{label}_second_plus", f"empirical_{label}_second_plus",
        ])

    logger.info("Fitting Modifier 1 – %s iteration...", label)

    first_slice = df[df[grouped_col] == "first"]
    second_plus_slice = df[(df[grouped_col].notna()) & (df[grouped_col] != "first")]

    grp = [("ignored", attempt_col, label)]
    keep_cols = ["attempt_number", label, f"empirical_{label}"]

    # Fit first slice
    mod_first = _safe_fit_rates_model(
        first_slice, context=f"{label}_iteration_first",
        groupings=grp, max_attempt=max_attempt,
    )
    mod_first = _rename_fitted_columns(
        mod_first[keep_cols] if not mod_first.empty else mod_first,
        label, "first",
    )

    # Fit second+ slice
    mod_second = _safe_fit_rates_model(
        second_plus_slice, context=f"{label}_iteration_second_plus",
        groupings=grp, max_attempt=max_attempt,
    )
    mod_second = _rename_fitted_columns(
        mod_second[keep_cols] if not mod_second.empty else mod_second,
        label, "second_plus",
    )

    return pd.merge(mod_first, mod_second, on="attempt_number", how="outer")


# ---------------------------------------------------------------------------
# Modifier 2: Launches since last failure (LSF)
# ---------------------------------------------------------------------------

def _fit_lsf_bin_for_level(
    df: pd.DataFrame,
    attempt_col: str,
    bin_col: str,
    bin_value: str,
    label: str,
    max_attempt: int = 20,
) -> pd.DataFrame:
    """Fit fallback rates for a single LSF bin for a given level."""
    if bin_col not in df.columns:
        raise KeyError(f"Expected column '{bin_col}' not found for LSF modifier '{label}'.")
    if attempt_col not in df.columns:
        return pd.DataFrame(columns=["attempt_number", label, f"empirical_{label}"])

    # Normalise dash variants: '1-3' -> '1–3'
    ser = df[bin_col].astype(object).replace({'1-3': '1–3'})
    slice_df = df.loc[ser == bin_value]

    grp = [("ignored", attempt_col, label)]
    fitted = _safe_fit_rates_model(
        slice_df, context=f"{label}_LSF_{bin_value}",
        groupings=grp, max_attempt=20,
    )
    keep = ["attempt_number", label, f"empirical_{label}"]
    return fitted[keep] if not fitted.empty else pd.DataFrame(columns=keep)


def _modifier_lsf_for_level(
    df: pd.DataFrame,
    attempt_col: str,
    bin_col: str,
    label: str,
    max_attempt: int = 20,
) -> pd.DataFrame:
    """
    Build LSF modifier for one level.

    Output columns: attempt_number,
      {label}_clean, empirical_{label}_clean,
      {label}_1_3, empirical_{label}_1_3,
      {label}_ge4, empirical_{label}_ge4
    """
    logger.info("Fitting Modifier 2 – LSF for '%s'...", label)

    frames = []
    for bin_value, suffix in LSF_BINS:
        fitted = _fit_lsf_bin_for_level(df, attempt_col, bin_col, bin_value, label, max_attempt)
        frames.append(_rename_fitted_columns(fitted, label, suffix))

    # Merge all bins on attempt_number
    out = frames[0]
    for frame in frames[1:]:
        out = pd.merge(out, frame, on="attempt_number", how="outer")
    return out


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def _export_fallback_rates(fallback_rates: dict[str, pd.DataFrame],
                           date_tag: str) -> pd.DataFrame:
    """
    Export per-level fallback rates and build a combined long table.

    Returns the combined long DataFrame.
    """
    all_levels = []
    for level_name, df_out in fallback_rates.items():
        # Select minimal columns
        minimal = ["attempt_number", level_name, f"empirical_{level_name}"]
        present = [c for c in minimal if c in df_out.columns]
        export_df = df_out.loc[:, present].copy() if present else _make_unique_columns(df_out)
        export_df = _make_unique_columns(export_df)

        csv_path = OUTPUT_DIR / f"launch_fallback_rates_{level_name}_{date_tag}.csv"
        export_df.to_csv(csv_path, index=False)
        export_df.to_json(csv_path.with_suffix(".json"), orient="records", indent=2)

        # Tag for long table
        export_df["level"] = level_name
        all_levels.append(export_df)

    # Build combined long table
    fallback_long = pd.concat(all_levels, ignore_index=True, sort=False)

    level_names = list(fallback_rates.keys())
    fallback_long["fitted"] = fallback_long[level_names].sum(axis=1)

    empirical_cols = [f"empirical_{lvl}" for lvl in level_names]
    present_emp = [c for c in empirical_cols if c in fallback_long.columns]
    if present_emp:
        fallback_long["empirical"] = fallback_long[present_emp].sum(axis=1)

    fallback_long = fallback_long[["level", "attempt_number", "fitted", "empirical"]]
    fallback_long = _make_unique_columns(fallback_long)

    long_csv = OUTPUT_DIR / f"launch_fallback_rates_all_levels_{date_tag}.csv"
    fallback_long.to_csv(long_csv, index=False)
    fallback_long.to_json(long_csv.with_suffix(".json"), orient="records", indent=2)

    return fallback_long


def _export_modifier_tables(modifiers: dict[str, pd.DataFrame],
                            prefix: str, date_tag: str) -> None:
    """Export modifier DataFrames (iteration or LSF) as CSV + JSON."""
    for level_name, df_mod in modifiers.items():
        csv_path = OUTPUT_DIR / f"launch_modifier_{prefix}_{level_name}_{date_tag}.csv"
        _export_csv_json(df_mod, csv_path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Run the fallback rate fitting and export pipeline.

    Steps:
    1. Load and prepare data
    2. Fit fallback base rates for all hierarchy levels
    3. Fit Modifier 1: iteration (first vs second+)
    4. Fit Modifier 2: launches since last failure
    5. Build provider rating comparison table
    6. Export all outputs as CSV + JSON
    """
    date_tag = datetime.now(tz=TZ).strftime("%Y%m%d")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Load data ---
    logger.info("Loading and preparing data...")
    launch_df = load_and_prepare_data()
    logger.info("Loaded %d rows with columns: %s", len(launch_df), list(launch_df.columns))

    # --- Step 2: Fallback base rates ---
    fallback_rates = _base_rates(launch_df)

    # --- Step 3: Modifier 1 – Iteration ---
    modifiers_iter = {
        label: _modifier_iteration_for_level(
            launch_df, attempt_col=attempt_col, grouped_col=grouped_col,
            label=label, max_attempt=20,
        )
        for label, attempt_col, grouped_col in ITERATION_LEVELS
    }

    # --- Step 4: Modifier 2 – Launches since last failure ---
    modifiers_lsf = {
        label: _modifier_lsf_for_level(
            launch_df, attempt_col=attempt_col, bin_col=bin_col,
            label=label, max_attempt=20,
        )
        for label, attempt_col, bin_col in LSF_LEVELS
    }

    # --- Step 5: Provider rating comparison ---
    logger.info("Creating Provider Rating Comparison Table...")
    provider_rating_table = provider_rating_comparison_table(
        df=launch_df,
        max_attempt=20,
        groupings=(3, 10, 20),
        exclude_ratings=["N/A", "Unknown", "Not Rated"],
    )

    # --- Step 6: Export all outputs ---
    logger.info("Exporting CSVs & JSONs to %s", OUTPUT_DIR)

    _export_fallback_rates(fallback_rates, date_tag)
    _export_modifier_tables(modifiers_iter, "iter", date_tag)
    _export_modifier_tables(modifiers_lsf, "lsf", date_tag)

    # Full data snapshot
    full_csv = OUTPUT_DIR / f"launch_full_data_{date_tag}.csv"
    launch_df.to_csv(full_csv, index=False)
    launch_df.to_json(full_csv.with_suffix(".json"), orient="records", indent=2)

    # Provider rating comparison
    prov_csv = OUTPUT_DIR / f"launch_provider_rating_comparison_table_{date_tag}.csv"
    provider_rating_table.to_csv(prov_csv, index=False)
    provider_rating_table.to_json(prov_csv.with_suffix(".json"), orient="records", indent=2)

    logger.info("Done.")


if __name__ == "__main__":
    main()
