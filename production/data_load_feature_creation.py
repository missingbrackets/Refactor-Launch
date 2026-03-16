"""
Data loading and feature engineering pipeline.

Fetches launch/event data from SQL, standardizes columns, engineers features
(attempt numbers, failure stats, iteration groupings, provider ratings), and
applies final filtering.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, List

from ..utils.sql_query import fetch_data
from ..utils.add_binned_reliability_combo import add_binned_reliability_combo
from ..utils.add_launches_since_last_failure import add_launches_since_last_failure
from ..utils.add_months_since_last_launch import add_months_since_last_launch
from ..utils.compute_first_five_failure_stats import compute_first_five_failure_stats

# ---------------------------------------------------------------------------
# Level definitions (shared across multiple feature-engineering steps)
# ---------------------------------------------------------------------------

GROUPING_LEVELS = [
    # (group_col, attempt_col, prefix)
    ("vehicle_type", "lv_type_attempt_number", "type"),
    ("vehicle_family", "lv_family_attempt_number", "family"),
    ("launch_provider", "lv_provider_attempt_number", "provider"),
    ("vehicle_variant", "lv_variant_attempt_number", "variant"),
    ("vehicle_minor_variant", "lv_minor_variant_attempt_number", "minor_variant"),
]

SPACECRAFT_COLS = [
    "seradata_spacecraft_id", "seradata_launch_id", "launch_number", "launch_date",
    "vehicle_type", "orbit_category", "launch_type", "vehicle_family",
    "launch_provider", "vehicle_variant", "vehicle_minor_variant",
    "launch_country", "sector",
]

EVENT_COLS = [
    "seradata_spacecraft_id", "capability_loss_(percent_fraction)",
    "event_date_and_time", "spacecraft_event_description",
]


# ---------------------------------------------------------------------------
# Column standardization and validation
# ---------------------------------------------------------------------------

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase column names and replace spaces with underscores."""
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    return df


def _check_required_columns(df: pd.DataFrame, required_cols: list[str], table_name: str) -> None:
    """Raise ValueError if any required columns are missing."""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {table_name}: {missing}")


def _coerce_datetime_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Ensure specified columns are datetime64, coercing if needed."""
    for col in columns:
        if col in df.columns and not np.issubdtype(df[col].dtype, np.datetime64):
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Data extraction and preparation
# ---------------------------------------------------------------------------

def _filter_events(sql_events: pd.DataFrame) -> pd.DataFrame:
    """Filter events to launch-vehicle-related events with >5% capability loss."""
    sql_events = sql_events.dropna(subset=['spacecraft_event_description'])
    lv_related = (
        sql_events['spacecraft_event_description'].str.contains('launcher', case=False, na=False)
        | sql_events['spacecraft_event_description'].str.contains('^suborbital launch:', case=False, na=False)
    )
    launch_events = sql_events[lv_related]
    launch_events = launch_events[launch_events["capability_loss_(percent_fraction)"] > 0.05]
    return launch_events


def _prepare_launches(sql_spacecraft: pd.DataFrame) -> pd.DataFrame:
    """Select and deduplicate launch-level columns from spacecraft data."""
    return sql_spacecraft[SPACECRAFT_COLS].drop_duplicates()


def _prepare_launch_events(launch_events: pd.DataFrame) -> pd.DataFrame:
    """Select event-level columns for merging."""
    return launch_events[["seradata_spacecraft_id", "capability_loss_(percent_fraction)",
                          "event_date_and_time"]]


def _merge_and_aggregate(launches: pd.DataFrame, launch_events: pd.DataFrame) -> pd.DataFrame:
    """Merge launches with events and aggregate to one row per launch_number."""
    merged = launches.merge(launch_events, on='seradata_spacecraft_id', how='left').fillna(0)
    merged['sector'] = merged['sector'].fillna('unknown').astype(str)
    merged['orbit_category'] = merged['orbit_category'].fillna('unknown')

    aggregated = merged.groupby('launch_number').agg({
        'capability_loss_(percent_fraction)': 'mean',
        'orbit_category': lambda x: x.mode()[0] if not x.mode().empty else None,
        'seradata_spacecraft_id': 'first',
        'seradata_launch_id': 'first',
        'launch_date': 'first',
        'vehicle_type': 'first',
        'launch_type': 'first',
        'vehicle_family': 'first',
        'launch_provider': 'first',
        'vehicle_variant': 'first',
        'vehicle_minor_variant': 'first',
        'launch_country': 'first',
        'sector': lambda x: x.mode()[0] if not x.mode().empty else None,
    }).reset_index()
    return aggregated


def _feature_engineering(aggregated: pd.DataFrame, loss_threshold: float = 0.5) -> pd.DataFrame:
    """Apply has_loss flag, filter to launched, and ensure valid launch dates."""
    aggregated['has_loss'] = (aggregated['capability_loss_(percent_fraction)'] > loss_threshold).astype(int)
    aggregated = aggregated[aggregated['launch_type'].str.lower() == "launched"].copy()
    aggregated.loc[:, 'launch_date'] = pd.to_datetime(aggregated['launch_date'], errors='coerce')
    aggregated = aggregated.dropna(subset=['launch_date'])
    return aggregated


# ---------------------------------------------------------------------------
# Attempt number assignment
# ---------------------------------------------------------------------------

def _assign_attempt_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """Assign sequential attempt numbers within each grouping level."""
    for group_col, attempt_col, _prefix in GROUPING_LEVELS:
        df = (
            df.sort_values([group_col, 'launch_date'])
            .assign(**{
                attempt_col: df
                .sort_values([group_col, 'launch_date'])
                .groupby(group_col)
                .cumcount() + 1
            })
        )
    return df


# ---------------------------------------------------------------------------
# Feature engineering: LSF, months since last launch, first-five stats
# ---------------------------------------------------------------------------

def _add_lsf_and_bins_for_levels(
    df: pd.DataFrame,
    levels: list[tuple[str, str, str]],
    launch_bins=(1, 4, float("inf")),
) -> pd.DataFrame:
    """Add launches-since-last-failure columns for each grouping level."""
    out = df.copy()
    for group_col, attempt_col, prefix in levels:
        out = add_launches_since_last_failure(
            out,
            group_col=group_col,
            attempt_col=attempt_col,
            failure_col="has_loss",
            output_col=f"{prefix}_launches_since_last_failure",
        )
    return out


def _add_months_since_last_launch_all_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Add months-since-last-launch for all grouping levels."""
    for group_col, _attempt_col, prefix in GROUPING_LEVELS:
        df = add_months_since_last_launch(
            df,
            group_col=group_col,
            date_col='launch_date',
            output_col=f'{prefix}_months_since_last_launch',
        )
    return df


def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add LSF bins and months-since-last-launch for all levels."""
    df = _add_lsf_and_bins_for_levels(df, GROUPING_LEVELS, launch_bins=(1, 4, float("inf")))
    df = _add_months_since_last_launch_all_levels(df)
    return df


def _add_first_five_failure_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and merge first-five failure stats for all grouping levels."""
    for group_col, attempt_col, prefix in GROUPING_LEVELS:
        stats_df = compute_first_five_failure_stats(
            df, group_col=group_col, attempt_col=attempt_col, loss_col="has_loss",
        )
        new_cols = [
            col for col in stats_df.columns
            if col.lower() in ['failure_count_first_five', 'first_five_failure']
        ]
        rename_map = {col: f'{prefix}_' + col.strip().lower() for col in new_cols}
        stats_subset = stats_df[["launch_number"] + new_cols].rename(columns=rename_map)
        df = df.merge(stats_subset, on="launch_number", how="left")
    return df


def _categorize_first_five_failures(df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorize first-five failure counts into '0', '1', or '1+'.

    Applies to all levels that have a failure_count_first_five column.
    """
    def _categorize(x):
        return "0" if x == 0 else ("1" if x == 1 else "1+")

    for _group_col, _attempt_col, prefix in GROUPING_LEVELS:
        count_col = f'{prefix}_failure_count_first_five'
        cat_col = f'{prefix}_first_five_cat'
        if count_col in df.columns:
            df[cat_col] = df[count_col].apply(_categorize)

    return df


# ---------------------------------------------------------------------------
# Iteration-within-parent groupings
# ---------------------------------------------------------------------------

def add_iteration_within(
    df: pd.DataFrame,
    parent_col: str,
    child_col: str,
    date_col: str = "launch_date",
    iteration_col: str | None = None,
    grouped_col: str | None = None,
    first_label: str = "first",
    other_label: str = "not_first",
) -> pd.DataFrame:
    """
    Compute the order in which each child appears within a parent,
    based on the earliest launch date.

    Creates:
    - iteration_col: integer rank (Int64) starting at 1
    - grouped_col: 'first' / 'not_first' / NA
    """
    out = df.copy()
    out.loc[:, date_col] = pd.to_datetime(out[date_col], errors="coerce")

    base = child_col.replace("vehicle_", "")
    if iteration_col is None:
        iteration_col = f"{base}_iteration"
    if grouped_col is None:
        grouped_col = f"{base}_iteration_grouped"

    # First launch date per (parent, child)
    firsts = (
        out.dropna(subset=[parent_col, child_col, date_col])
        .groupby([parent_col, child_col], as_index=False)[date_col]
        .min()
    )

    # Dense rank of child first-appearances within each parent
    firsts[iteration_col] = (
        firsts.groupby(parent_col)[date_col]
        .rank(method="dense")
        .astype("Int64")
    )

    out = out.merge(
        firsts[[parent_col, child_col, iteration_col]],
        on=[parent_col, child_col], how="left",
    )

    out[grouped_col] = np.where(
        out[iteration_col] == 1, first_label,
        np.where(out[iteration_col].notna(), other_label, pd.NA),
    )

    return out


def _add_all_iteration_groupings(df: pd.DataFrame) -> pd.DataFrame:
    """Add iteration-within-parent groupings for type, variant, and minor_variant."""
    # Type within Family
    df = add_iteration_within(
        df, parent_col="vehicle_family", child_col="vehicle_type",
        iteration_col="type_iteration", grouped_col="type_iteration_grouped",
    )
    # Variant within Type
    df = add_iteration_within(
        df, parent_col="vehicle_type", child_col="vehicle_variant",
        iteration_col="variant_iteration", grouped_col="variant_iteration_grouped",
    )
    # Minor Variant within Type
    df = add_iteration_within(
        df, parent_col="vehicle_type", child_col="vehicle_minor_variant",
        iteration_col="minor_variant_iteration", grouped_col="minor_variant_iteration_grouped",
    )
    return df


# ---------------------------------------------------------------------------
# Family history quality
# ---------------------------------------------------------------------------

def _add_family_history_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'family_cum_fail_rate' and 'family_history_quality' columns.

    Quality is 'Good' if cumulative fail rate < 0.33, 'Poor' otherwise,
    or NA for the first type in a family (no prior history).
    """
    df = df.copy()
    _check_required_columns(
        df,
        ['vehicle_family', 'lv_family_attempt_number', 'has_loss', 'type_iteration'],
        "input DataFrame for family history quality",
    )

    df = df.sort_values(by=['vehicle_family', 'lv_family_attempt_number'])

    # Cumulative failures and attempts per family (excluding current row)
    df['family_cumulative_failures'] = (
        df.groupby('vehicle_family')['has_loss']
        .transform(lambda x: x.shift().cumsum())
    )
    df['family_cumulative_attempts'] = df['lv_family_attempt_number'] - 1
    df['family_cum_fail_rate'] = (
        df['family_cumulative_failures'] / df['family_cumulative_attempts'].replace(0, np.nan)
    )

    def _classify(row):
        if row['type_iteration'] == 1:
            return pd.NA
        if pd.isna(row['family_cum_fail_rate']):
            return pd.NA
        return 'Good' if row['family_cum_fail_rate'] < 0.33 else 'Poor'

    df['family_history_quality'] = df.apply(_classify, axis=1)
    return df


# ---------------------------------------------------------------------------
# Provider rating
# ---------------------------------------------------------------------------

def _add_provider_rating(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'provider_rating' column based on cumulative failure rate percentile.

    Ratings:
    - 'Very New Provider': < 5 cumulative attempts
    - 'New Provider': 5-20 cumulative attempts
    - 'A': bottom 20th percentile of fail rate (best)
    - 'B': 20th-80th percentile
    - 'C': top 20th percentile (worst)
    """
    df = df.sort_values(by=['launch_provider', 'lv_provider_attempt_number'])

    df['provider_cum_failures'] = (
        df.groupby('launch_provider')['has_loss'].cumsum().shift(1).fillna(0)
    )
    df['provider_cum_attempts'] = df.groupby('launch_provider').cumcount()
    df['provider_cum_fail_rate'] = (
        df['provider_cum_failures'] / df['provider_cum_attempts'].replace(0, np.nan)
    )

    # Cap comparison group at attempt 50
    df['comparison_attempt'] = df['lv_provider_attempt_number'].clip(upper=50)

    attempt_group = df.groupby('comparison_attempt')['provider_cum_fail_rate']
    df['provider_fail_rate_percentile'] = attempt_group.transform(lambda x: x.rank(pct=True))

    def _percentile_to_rating(p, attempts):
        if attempts < 5 or pd.isna(p):
            return 'Very New Provider'
        elif attempts <= 20:
            return 'New Provider'
        elif p <= 0.2:
            return 'A'
        elif p <= 0.8:
            return 'B'
        else:
            return 'C'

    df['provider_rating'] = df.apply(
        lambda row: _percentile_to_rating(
            row['provider_fail_rate_percentile'], row['provider_cum_attempts'],
        ),
        axis=1,
    )

    return df


# ---------------------------------------------------------------------------
# Final filtering
# ---------------------------------------------------------------------------

def _final_filtering(df: pd.DataFrame, min_launch_year: int, filter_min_year_on: bool,
                     max_attempt_number: int, excluded_countries: list[str]) -> pd.DataFrame:
    """Apply year, attempt number, and country filters."""
    df['launch_year'] = df['launch_date'].dt.year.fillna(0).astype(int)
    if filter_min_year_on:
        df = df[df["launch_year"] >= min_launch_year]
    df = df[df['lv_type_attempt_number'] <= max_attempt_number]
    df = df[~df['launch_country'].isin(excluded_countries)]
    return df


# ---------------------------------------------------------------------------
# Main data pipeline
# ---------------------------------------------------------------------------

def load_and_prepare_data(
    server: str = "Lon-sqlp-v005",
    database: str = "SpaceTrax_Data",
    spacecraft_table: str = "[dbo].[tbl_SpaceCraft]",
    events_table: str = "[dbo].[tbl_Events]",
    min_launch_year: int = 2000,
    max_attempt_number: int = 10000,
    excluded_countries: Optional[List[str]] = None,
    filter_min_year_on: bool = True,
    loss_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    End-to-end data pipeline: extract, standardize, engineer features, filter.

    Parameters
    ----------
    server, database, spacecraft_table, events_table : str
        Data source location and table names.
    min_launch_year : int
        Minimum launch year to keep (only if filter_min_year_on=True).
    max_attempt_number : int
        Maximum allowed attempt number.
    excluded_countries : list[str] or None
        Launch countries to exclude.
    filter_min_year_on : bool
        Whether to apply the min_launch_year filter.
    loss_threshold : float
        Threshold for the has_loss binary flag.

    Returns
    -------
    pd.DataFrame – final prepared dataset
    """
    # Defaults and validation
    if excluded_countries is None:
        excluded_countries = ["iran", "north korea"]
    excluded_countries = [str(c).strip().lower() for c in excluded_countries]

    if not np.isscalar(max_attempt_number):
        raise TypeError(
            f"max_attempt_number must be a scalar int, got {type(max_attempt_number)} "
            f"with value {max_attempt_number}"
        )
    max_attempt_number = int(max_attempt_number)

    # --- Extract ---
    original_spacecraft_cols = [
        "Seradata Spacecraft ID", "Seradata Launch ID", "Launch Number", "Launch Date",
        "Vehicle Type", "Orbit Category", "Launch Type", "Vehicle Family",
        "Launch Provider", "Vehicle Variant", "Vehicle Minor Variant",
        "Launch Country", "Sector",
    ]
    original_events_cols = [
        "Seradata Spacecraft ID", "Capability Loss (percent fraction)",
        "Event Date And Time", "Spacecraft Event Description",
    ]

    sql_spacecraft = fetch_data(
        server=server, database=database, table=spacecraft_table,
        select_columns=original_spacecraft_cols,
    )
    sql_events = fetch_data(
        server=server, database=database, table=events_table,
        select_columns=original_events_cols,
    )

    # --- Standardize ---
    sql_spacecraft = _standardize_columns(sql_spacecraft)
    sql_events = _standardize_columns(sql_events)

    required_spacecraft = [c.strip().lower().replace(' ', '_') for c in original_spacecraft_cols]
    required_events = [c.strip().lower().replace(' ', '_') for c in original_events_cols]
    _check_required_columns(sql_spacecraft, required_spacecraft, "spacecraft table")
    _check_required_columns(sql_events, required_events, "events table")

    sql_spacecraft = _coerce_datetime_columns(sql_spacecraft, ["launch_date"])
    sql_events = _coerce_datetime_columns(sql_events, ["event_date_and_time"])

    # --- Core transformations ---
    launch_events = _filter_events(sql_events)
    launches = _prepare_launches(sql_spacecraft)
    launch_events = _prepare_launch_events(launch_events)
    aggregated = _merge_and_aggregate(launches, launch_events)
    aggregated = _feature_engineering(aggregated, loss_threshold=loss_threshold)
    aggregated = _assign_attempt_numbers(aggregated)

    df = aggregated.copy()

    # --- Feature engineering ---
    df = _add_engineered_features(df)

    # Debug log for verification
    vc_final = (
        df.loc[df['vehicle_type'] == 'ZHUQUE-2/SUZAKU-2 (ZQ-2)',
               ['vehicle_type', 'lv_type_attempt_number', 'has_loss',
                'type_launches_since_last_failure']]
        .sort_values('lv_type_attempt_number')
    )
    print("\n[post-pipeline] VULCAN CENTAUR final values:")
    print(vc_final.to_string(index=False))

    df = _add_first_five_failure_stats(df)
    df = _categorize_first_five_failures(df)
    df = _add_provider_rating(df)
    df = _add_all_iteration_groupings(df)
    df = _add_family_history_quality(df)

    # --- Final filtering ---
    if "launch_country" in df.columns:
        df["launch_country"] = df["launch_country"].str.strip().str.lower()

    df = _final_filtering(
        df,
        min_launch_year=min_launch_year,
        filter_min_year_on=filter_min_year_on,
        max_attempt_number=max_attempt_number,
        excluded_countries=excluded_countries,
    )

    return df


def main():
    temp = load_and_prepare_data()
    print(temp.head)


if __name__ == "__main__":
    main()
