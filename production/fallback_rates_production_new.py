# -*- coding: utf-8 -*-
"""
Fit a survival model to determine fallback base rates and export tables.

This script is the production counterpart to exploratory notebooks: it focuses on
data processing and exporting tabular outputs (no charts).
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import logging
import pandas as pd

from ..utils.output_to_csv import output_df_to_csv
from ..utils.fit_fallback_base_rate import fit_rates_model
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
# ------------------------------------------------------

def _fit_base_for_level(df: pd.DataFrame, attempt_col: str, level_name: str) -> pd.DataFrame:
    """
    Fit base rates for a specific level by telling fit_rates_model which attempt column to use.
    Returns columns: ['attempt_number', level_name, f'empirical_{level_name}'] (if provided by model).
    """
    if attempt_col not in df.columns:
        logging.warning("Attempt column '%s' missing; returning empty for level '%s'", attempt_col, level_name)
        return pd.DataFrame(columns=["attempt_number", level_name, f"empirical_{level_name}"])

    # Tell the model exactly which attempt column to use and how to label outputs
    grp = [("ignored", attempt_col, level_name)]
    fitted = _safe_fit_rates_model(df, context=f"{level_name}_base", groupings=grp)
    if fitted.empty:
        return fitted

    # Keep the minimal, self-describing columns if present
    cols = ["attempt_number", level_name, f"empirical_{level_name}"]
    present = [c for c in cols if c in fitted.columns]
    return fitted[present] if present else fitted


def _safe_fit_rates_model(df: pd.DataFrame, context: str, **kwargs) -> pd.DataFrame:
    if df.empty:
        logger.warning("Slice '%s' is empty; returning empty DataFrame.", context)
        return pd.DataFrame()
    return fit_rates_model(df, **kwargs)


def _base_rates(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Fit fallback base rates for type, family, variant, and minor_variant.
    Returns a dict keyed by level name.
    """
    logger.info("Fitting fallback base rates model for all levels…")
    outputs = {
        "type": _fit_base_for_level(df, "lv_type_attempt_number", "type"),
        "family": _fit_base_for_level(df, "lv_family_attempt_number", "family"),
        "provider": _fit_base_for_level(df, "lv_provider_attempt_number", "provider"),
        "variant": _fit_base_for_level(df, "lv_variant_attempt_number", "variant"),
        "minor_variant": _fit_base_for_level(df, "lv_minor_variant_attempt_number", "minor_variant"),
    }
    for k, v in outputs.items():
        logger.info("Level '%s' produced %d rows", k, len(v))
    return outputs


def _modifier_iteration_for_level(
    df: pd.DataFrame,
    attempt_col: str,
    grouped_col: str,
    label: str,   # <- this becomes the output column name; e.g. "variant"
    max_attempt: int = 20,
) -> pd.DataFrame:
    """
    Fits Modifier 1 (iteration) for a given level using 'first' vs 'second+' slices.
    Produces columns: attempt_number, <label>_first, empirical_<label>_first,
                                and <label>_second_plus, empirical_<label>_second_plus
    """
    if grouped_col not in df.columns:
        raise KeyError(f"Expected column '{grouped_col}' not found for level '{label}'.")
    if attempt_col not in df.columns:
        logger.warning("Attempt column '%s' missing for level '%s'; returning empty.", attempt_col, label)
        cols = ["attempt_number",
                f"{label}_first", f"empirical_{label}_first",
                f"{label}_second_plus", f"empirical_{label}_second_plus"]
        return pd.DataFrame(columns=cols)

    logger.info("Fitting Modifier 1 – %s iteration…", label)

    first_slice = df[df[grouped_col] == "first"]
    second_plus_slice = df[(df[grouped_col].notna()) & (df[grouped_col] != "first")]

    # Tell the model exactly which attempt column to use, and what to call the output.
    grp = [("ignored", attempt_col, label)]

    mod_first = _safe_fit_rates_model(
        first_slice, context=f"{label}_iteration_first",
        groupings=grp, max_attempt=max_attempt
    )
    mod_second = _safe_fit_rates_model(
        second_plus_slice, context=f"{label}_iteration_second_plus",
        groupings=grp, max_attempt=max_attempt
    )

    # Keep only native columns for this label; no renaming necessary.
    keep_cols = ["attempt_number", label, f"empirical_{label}"]
    if not mod_first.empty:
        mod_first = mod_first[keep_cols].rename(columns={
            label: f"{label}_first",
            f"empirical_{label}": f"empirical_{label}_first",
        })
    else:
        mod_first = pd.DataFrame(columns=["attempt_number", f"{label}_first", f"empirical_{label}_first"])

    if not mod_second.empty:
        mod_second = mod_second[keep_cols].rename(columns={
            label: f"{label}_second_plus",
            f"empirical_{label}": f"empirical_{label}_second_plus",
        })
    else:
        mod_second = pd.DataFrame(columns=["attempt_number", f"{label}_second_plus", f"empirical_{label}_second_plus"])

    out = pd.merge(mod_first, mod_second, on="attempt_number", how="outer")
    return out

def _fit_lsf_bin_for_level(
    df: pd.DataFrame,
    attempt_col: str,
    bin_col: str,
    bin_value: str,
    label: str,
    max_attempt: int = 20,
) -> pd.DataFrame:
    """
    Fit fallback base rates for a single SinceFail bin for a given level.
    Returns columns: ['attempt_number', label, f'empirical_{label}'] when available.
    """
    if bin_col not in df.columns:
        raise KeyError(f"Expected column '{bin_col}' not found for LSF modifier '{label}'.")
    if attempt_col not in df.columns:
        return pd.DataFrame(columns=["attempt_number", label, f"empirical_{label}"])

    # Normalise '1–3' vs '1-3'
    ser = df[bin_col].astype(object).replace({'1-3': '1–3'})
    slice_df = df.loc[ser == bin_value]

    grp = [("ignored", attempt_col, label)]
    fitted = _safe_fit_rates_model(
        slice_df, context=f"{label}_LSF_{bin_value}", groupings=grp, max_attempt=20
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
    Output columns:
      attempt_number,
      {label}_clean,        empirical_{label}_clean,
      {label}_1_3,          empirical_{label}_1_3,
      {label}_ge4,          empirical_{label}_ge4
    """
    logger.info("Fitting Modifier 2 – LSF for '%s'…", label)

    clean = _fit_lsf_bin_for_level(df, attempt_col, bin_col, "Clean", label, max_attempt)
    one_to_three = _fit_lsf_bin_for_level(df, attempt_col, bin_col, "1–3", label, max_attempt)
    ge4 = _fit_lsf_bin_for_level(df, attempt_col, bin_col, ">=4", label, max_attempt)

    if not clean.empty:
        clean = clean.rename(columns={label: f"{label}_clean", f"empirical_{label}": f"empirical_{label}_clean"})
    else:
        clean = pd.DataFrame(columns=["attempt_number", f"{label}_clean", f"empirical_{label}_clean"])

    if not one_to_three.empty:
        one_to_three = one_to_three.rename(columns={label: f"{label}_1_3", f"empirical_{label}": f"empirical_{label}_1_3"})
    else:
        one_to_three = pd.DataFrame(columns=["attempt_number", f"{label}_1_3", f"empirical_{label}_1_3"])

    if not ge4.empty:
        ge4 = ge4.rename(columns={label: f"{label}_ge4", f"empirical_{label}": f"empirical_{label}_ge4"})
    else:
        ge4 = pd.DataFrame(columns=["attempt_number", f"{label}_ge4", f"empirical_{label}_ge4"])

    out = pd.merge(clean, one_to_three, on="attempt_number", how="outer")
    out = pd.merge(out, ge4, on="attempt_number", how="outer")
    return out


def _fit_bin(df: pd.DataFrame, name: str, keep_cols: list[str] = ["attempt_number", "type", "empirical_type"]) -> pd.DataFrame:
        df = _safe_fit_rates_model(df[df["type_SinceFail_Bin"] == name], f"SinceFail={name}")
        if df.empty:
            return df
        out = df[keep_cols].copy()
        return out

def _make_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has unique column names by adding suffixes .1, .2, ..."""
    counts = {}
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

def main() -> None:
    """
    Run fallback rate fitting and export:
      - Fallback base rates
      - Modifier 1: Type iteration (first vs second+)
      - Modifier 2: Launches since last failure (Clean, 1–3, >=4)
      - Provider rating comparison table
      - Full input data snapshot
    """
    date_tag = datetime.now(tz=TZ).strftime("%Y%m%d")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading and preparing data…")
    launch_df = load_and_prepare_data()
    logger.info("Loaded %d rows with columns: %s", len(launch_df), list(launch_df.columns))

    # Keep only the needed columns and suffix for clarity
    keep_cols = ["attempt_number", "type", "empirical_type"]

    # ---------- Fallback base rates ----------
    fallback_rates = _base_rates(launch_df)

    # ---------- Modifier 1: Type iteration ----------
    # Build the three iteration modifiers
    modifiers_iter = {
        "type": _modifier_iteration_for_level(
            launch_df,
            attempt_col="lv_type_attempt_number",
            grouped_col="type_iteration_grouped",
            label="type",
            max_attempt=20,
        ),
        "variant": _modifier_iteration_for_level(
            launch_df,
            attempt_col="lv_variant_attempt_number",
            grouped_col="variant_iteration_grouped",
            label="variant",
            max_attempt=20,
        ),
        "minor_variant": _modifier_iteration_for_level(
            launch_df,
            attempt_col="lv_minor_variant_attempt_number",
            grouped_col="minor_variant_iteration_grouped",
            label="minor_variant",
            max_attempt=20,
        ),
    }


    # ---------- Modifier 2: Launches since last failure ----------
    modifiers_lsf = {
        "type": _modifier_lsf_for_level(
            launch_df, attempt_col="lv_type_attempt_number",   bin_col="type_SinceFail_Bin",   label="type",           max_attempt=20
        ),
        "family": _modifier_lsf_for_level(
            launch_df, attempt_col="lv_family_attempt_number", bin_col="family_SinceFail_Bin", label="family",         max_attempt=20
        ),
        "variant": _modifier_lsf_for_level(
            launch_df, attempt_col="lv_variant_attempt_number", bin_col="variant_SinceFail_Bin", label="variant",      max_attempt=20
        ),
        "minor_variant": _modifier_lsf_for_level(
            launch_df, attempt_col="lv_minor_variant_attempt_number", bin_col="minor_variant_SinceFail_Bin", label="minor_variant", max_attempt=20
        ),
    }

    # ---------- Provider rating comparison ----------
    logger.info("Creating Provider Rating Comparison Table…")
    provider_rating_table = provider_rating_comparison_table(
        df=launch_df,
        max_attempt=20,
        groupings=(3, 10, 20),
        exclude_ratings=["N/A", "Unknown", "Not Rated"],
    )

    # ---------- Exports ----------
    logger.info("Exporting CSVs & JSONs to %s", OUTPUT_DIR)

    # Fallback rates (per level)
    all_levels = []  # collect rows for the long table
    for level_name, df_out in fallback_rates.items():
        # Prefer minimal, self-describing columns
        minimal = ["attempt_number", level_name, f"empirical_{level_name}"]
        present = [c for c in minimal if c in df_out.columns]
        if present:
            export_df = df_out.loc[:, present].copy()
        else:
            # If model didn't output those exact names, dedupe and export everything
            export_df = _make_unique_columns(df_out)

        # Final guard: JSON requires unique column names
        export_df = _make_unique_columns(export_df)

        fallback_csv = OUTPUT_DIR / f"launch_fallback_rates_{level_name}_{date_tag}.csv"
        fallback_json = fallback_csv.with_suffix(".json")
        export_df.to_csv(fallback_csv, index=False)
        export_df.to_json(fallback_json, orient="records", indent=2)

        # Add to the long table collection
        export_df["level"] = level_name  # add a column to identify the level
        all_levels.append(export_df)

    # --- After the loop: one long table across all levels ---
    fallback_long = pd.concat(all_levels, ignore_index=True, sort=False)

    # Coalesce level-specific columns into generic 'fitted' and 'empirical' columns.
    # For each row, only one of the level-specific columns (e.g., 'type', 'family')
    # will contain a non-NaN value. Using sum(axis=1) is a concise way to pick
    # that one value, as NaN is treated as 0 in the sum.
    level_names = list(fallback_rates.keys())
    fallback_long["fitted"] = fallback_long[level_names].sum(axis=1)

    empirical_cols = [f"empirical_{level}" for level in level_names]
    present_empirical_cols = [c for c in empirical_cols if c in fallback_long.columns]
    if present_empirical_cols:
        fallback_long["empirical"] = fallback_long[present_empirical_cols].sum(axis=1)

    fallback_long = fallback_long[["level", "attempt_number", "fitted", "empirical"]]
    fallback_long = _make_unique_columns(fallback_long)


    fallback_long_csv = OUTPUT_DIR / f"launch_fallback_rates_all_levels_{date_tag}.csv"
    fallback_long_json = fallback_long_csv.with_suffix(".json")
    fallback_long.to_csv(fallback_long_csv, index=False)
    fallback_long.to_json(fallback_long_json, orient="records", indent=2)

    # Modifier – Type Iteration
    for level_name, df_mod in modifiers_iter.items():
        export_df = _make_unique_columns(df_mod)
        iter_csv = OUTPUT_DIR / f"launch_modifier_iter_{level_name}_{date_tag}.csv"
        export_df.to_csv(iter_csv, index=False)
        export_df.to_json(iter_csv.with_suffix(".json"), orient="records", indent=2)


    # Modifier – Since Last Failure
    for level_name, df_mod in modifiers_lsf.items():
        export_df = _make_unique_columns(df_mod)
        lsf_csv = OUTPUT_DIR / f"launch_modifier_lsf_{level_name}_{date_tag}.csv"
        export_df.to_csv(lsf_csv, index=False)
        export_df.to_json(lsf_csv.with_suffix(".json"), orient="records", indent=2)


    # Full data
    full_csv = OUTPUT_DIR / f"launch_full_data_{date_tag}.csv"
    full_json = full_csv.with_suffix(".json")
    launch_df.to_csv(full_csv, index=False)
    launch_df.to_json(full_json, orient="records", indent=2)

    # Provider Rating Modifier
    prov_csv = OUTPUT_DIR / f"launch_provider_rating_comparison_table_{date_tag}.csv"
    prov_json = prov_csv.with_suffix(".json")
    provider_rating_table.to_csv(prov_csv, index=False)
    provider_rating_table.to_json(prov_json, orient="records", indent=2)

    logger.info("Done.")



if __name__ == "__main__":
    main()
