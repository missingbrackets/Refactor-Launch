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

def _safe_fit_rates_model(df: pd.DataFrame, context: str) -> pd.DataFrame:
    """
    Run fit_rates_model on a slice; if the slice is empty, log and return an empty DataFrame.
    This prevents downstream merges from failing.
    """
    if df.empty:
        logger.warning("Slice '%s' is empty; returning empty DataFrame.", context)
        return pd.DataFrame()
    
    return fit_rates_model(df)

def _base_rates(df: pd.DataFrame) -> pd.DataFrame:
    # ---------- Fallback base rates ----------
    logger.info("Fitting fallback base rates model…")
    fallback_rates_df = _safe_fit_rates_model(df, "all_data")

    return fallback_rates_df

def _modifier_one_iteration(df: pd.DataFrame, keep_cols: list[str]) -> pd.DataFrame:
    # Expecting 'type_iteration_grouped' values like 'first' vs not 'first'
    logger.info("Fitting Modifier 1 – Type iteration...")
    if "type_iteration_grouped" not in df.columns:
        raise KeyError(
            "Expected column 'type_iteration_grouped' not found in input data."
        )
    logger.debug("type_iteration_grouped values: %s", df["type_iteration_grouped"].unique())

    mod1_first = _safe_fit_rates_model(
        df[df["type_iteration_grouped"] == "first"], "iteration_first"
    )
    mod1_second = _safe_fit_rates_model(
        df[df["type_iteration_grouped"] != "first"], "iteration_second_plus"
    )

    if not mod1_first.empty:
        mod1_first = mod1_first[keep_cols].rename(
            columns={"type": "type_first", "empirical_type": "empirical_type_first"}
        )
    if not mod1_second.empty:
        mod1_second = mod1_second[keep_cols].rename(
            columns={"type": "type_second_plus", "empirical_type": "empirical_type_second_plus"}
        )

    # Outer merge allows attempt ranges to differ
    modifier_iteration = (
        mod1_first.merge(mod1_second, on="attempt_number", how="outer")
        if not mod1_first.empty or not mod1_second.empty
        else pd.DataFrame(columns=["attempt_number", "type_first", "empirical_type_first",
                                   "type_second_plus", "empirical_type_second_plus"])
    )

    return modifier_iteration

def _modifier_launches_since_last_failure(df: pd.DataFrame, keep_cols: list[str]) -> pd.DataFrame:
    # Expecting 'type_SinceFail_Bin' groups: 'Clean', '1–3', '>=4'
    # NOTE: The middle value uses an en dash (–). If your ETL sometimes produces "1-3",
    # standardize earlier or handle both.
    logger.info("Fitting Modifier 2 – Launches since last failure...")
    if "type_SinceFail_Bin" not in df.columns:
        raise KeyError("Expected column 'type_SinceFail_Bin' not found in input data.")

    lsf_clean = _fit_bin(df, "Clean", keep_cols)
    lsf_1_3 = _fit_bin(df, "1–3", keep_cols)  # en dash
    lsf_ge4 = _fit_bin(df, ">=4", keep_cols)

    if not lsf_clean.empty:
        lsf_clean = lsf_clean.rename(columns={"type": "type_clean", "empirical_type": "empirical_type_clean"})
    if not lsf_1_3.empty:
        lsf_1_3 = lsf_1_3.rename(columns={"type": "type_1_3", "empirical_type": "empirical_type_1_3"})
    if not lsf_ge4.empty:
        lsf_ge4 = lsf_ge4.rename(columns={"type": "type_ge4", "empirical_type": "empirical_type_ge4"})

    modifier_lsf = (
        lsf_clean.merge(lsf_1_3, on="attempt_number", how="outer")
        .merge(lsf_ge4, on="attempt_number", how="outer")
        if not (lsf_clean.empty and lsf_1_3.empty and lsf_ge4.empty)
        else pd.DataFrame(
            columns=[
                "attempt_number",
                "type_clean", "empirical_type_clean",
                "type_1_3", "empirical_type_1_3",
                "type_ge4", "empirical_type_ge4",
            ]
        )
    )

    return modifier_lsf

def _fit_bin(df: pd.DataFrame, name: str, keep_cols: list[str] = ["attempt_number", "type", "empirical_type"]) -> pd.DataFrame:
        df = _safe_fit_rates_model(df[df["type_SinceFail_Bin"] == name], f"SinceFail={name}")
        if df.empty:
            return df
        out = df[keep_cols].copy()
        return out

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
    fallback_rates_df = _base_rates(launch_df)

    # ---------- Modifier 1: Type iteration ----------
    modifier_iteration = _modifier_one_iteration(launch_df, keep_cols)

    # ---------- Modifier 2: Launches since last failure ----------
    modifier_lsf = _modifier_launches_since_last_failure(launch_df, keep_cols)

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

    # Fallback rates
    fallback_csv = OUTPUT_DIR / f"fallback_rates_production_{date_tag}.csv"
    fallback_json = fallback_csv.with_suffix(".json")
    fallback_rates_df.to_csv(fallback_csv, index=False)
    fallback_rates_df.to_json(fallback_json, orient="records", indent=2)

    # Modifier – Type Iteration
    iter_csv = OUTPUT_DIR / f"objective_modifier_iteration_{date_tag}.csv"
    iter_json = iter_csv.with_suffix(".json")
    modifier_iteration.to_csv(iter_csv, index=False)
    modifier_iteration.to_json(iter_json, orient="records", indent=2)

    # Modifier – Since Last Failure
    lsf_csv = OUTPUT_DIR / f"launches_since_last_failure_modifier_{date_tag}.csv"
    lsf_json = lsf_csv.with_suffix(".json")
    modifier_lsf.to_csv(lsf_csv, index=False)
    modifier_lsf.to_json(lsf_json, orient="records", indent=2)

    # Full data
    full_csv = OUTPUT_DIR / f"full_data_{date_tag}.csv"
    full_json = full_csv.with_suffix(".json")
    launch_df.to_csv(full_csv, index=False)
    launch_df.to_json(full_json, orient="records", indent=2)

    # Provider Rating Modifier
    prov_csv = OUTPUT_DIR / f"provider_rating_comparison_table_{date_tag}.csv"
    prov_json = prov_csv.with_suffix(".json")
    provider_rating_table.to_csv(prov_csv, index=False)
    provider_rating_table.to_json(prov_json, orient="records", indent=2)

    logger.info("Done.")



if __name__ == "__main__":
    main()
