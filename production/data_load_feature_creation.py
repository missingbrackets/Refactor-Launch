from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional, List
from ..utils.sql_query import fetch_data
from ..utils.add_binned_reliability_combo import add_binned_reliability_combo
from ..utils.add_launches_since_last_failure import add_launches_since_last_failure
from ..utils.add_months_since_last_launch import add_months_since_last_launch
from ..utils.compute_first_five_failure_stats import compute_first_five_failure_stats

def _standardize_columns(df):
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    return df

def _check_required_columns(df, required_cols, table_name):
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {table_name}: {missing}")

def _filter_events(sql_events):
    sql_events = sql_events.dropna(subset=['spacecraft_event_description'])
    lv_related = (
        sql_events['spacecraft_event_description'].str.contains('launcher', case=False, na=False) |
        sql_events['spacecraft_event_description'].str.contains('^suborbital launch:', case=False, na=False)
    )
    launch_events = sql_events[lv_related]
    launch_events = launch_events[launch_events["capability_loss_(percent_fraction)"] > 0.05]
    return launch_events

def _prepare_launches(sql_spacecraft):
    launches = (
        sql_spacecraft[
            [
                "seradata_spacecraft_id", "seradata_launch_id", "launch_number", "launch_date",
                "vehicle_type", "orbit_category", "launch_type", "vehicle_family",
                "launch_provider", "vehicle_variant", "vehicle_minor_variant",  # <-- added
                "launch_country", "sector"
            ]
        ]
        .drop_duplicates()
    )
    return launches

def _prepare_launch_events(launch_events):
    return launch_events[["seradata_spacecraft_id", "capability_loss_(percent_fraction)", "event_date_and_time"]]

def _merge_and_aggregate(launches, launch_events):
    merged_data = launches.merge(launch_events, on='seradata_spacecraft_id', how='left').fillna(0)
    merged_data['sector'] = merged_data['sector'].fillna('unknown').astype(str)
    merged_data['orbit_category'] = merged_data['orbit_category'].fillna('unknown')

    aggregated_data = merged_data.groupby('launch_number').agg({
        'capability_loss_(percent_fraction)': 'mean',
        'orbit_category': lambda x: x.mode()[0] if not x.mode().empty else None,
        'seradata_spacecraft_id': 'first', 'seradata_launch_id': 'first', 'launch_date': 'first',
        'vehicle_type': 'first', 'launch_type': 'first', 'vehicle_family': 'first',
        'launch_provider': 'first', 'vehicle_variant': 'first', 'vehicle_minor_variant': 'first',  # <-- added
        'launch_country': 'first',
        'sector': lambda x: x.mode()[0] if not x.mode().empty else None
    }).reset_index()
    return aggregated_data


def _feature_engineering(aggregated_data, loss_threshold=0.5):
    aggregated_data['has_loss'] = (aggregated_data['capability_loss_(percent_fraction)'] > loss_threshold).astype(int)
    aggregated_data = aggregated_data[aggregated_data['launch_type'].str.lower() == "launched"].copy()
    aggregated_data.loc[:, 'launch_date'] = pd.to_datetime(aggregated_data['launch_date'], errors='coerce')
    aggregated_data = aggregated_data.dropna(subset=['launch_date'])
    return aggregated_data

def _assign_attempt_numbers(aggregated_data):
    groupings = [
        ('vehicle_type', 'lv_type_attempt_number'),
        ('vehicle_family', 'lv_family_attempt_number'),
        ('launch_provider', 'lv_provider_attempt_number'),
        ('vehicle_variant', 'lv_variant_attempt_number'),                 # <-- added
        ('vehicle_minor_variant', 'lv_minor_variant_attempt_number'),     # <-- added
    ]
    for group_col, attempt_col in groupings:
        aggregated_data = (
            aggregated_data.sort_values([group_col, 'launch_date'])
            .assign(**{
                attempt_col: aggregated_data
                .sort_values([group_col, 'launch_date'])
                .groupby(group_col)
                .cumcount() + 1
            })
        )
    return aggregated_data


def _add_engineered_features(launch_data_test):
    # 1) launches since last failure + bins for all levels
    levels = [
        ('vehicle_type',          'lv_type_attempt_number',          'type'),
        ('vehicle_family',        'lv_family_attempt_number',        'family'),
        ('launch_provider',       'lv_provider_attempt_number',      'provider'),
        ('vehicle_variant',       'lv_variant_attempt_number',       'variant'),
        ('vehicle_minor_variant', 'lv_minor_variant_attempt_number', 'minor_variant'),
    ]
    launch_data_test = _add_lsf_and_bins_for_levels(launch_data_test, levels, launch_bins=(1, 4, float("inf")))

    # 2) months since last launch (unchanged, but add the two new levels)
    for group_col, prefix in [
        ('vehicle_type', 'type'),
        ('vehicle_family', 'family'),
        ('launch_provider', 'provider'),
        ('vehicle_variant', 'variant'),
        ('vehicle_minor_variant', 'minor_variant'),
    ]:
        launch_data_test = add_months_since_last_launch(
            launch_data_test, group_col=group_col, date_col='launch_date',
            output_col=f'{prefix}_months_since_last_launch'
        )

    # binned reliability combos
    launch_bins = [1, 4, float("inf")]
    # for group_col, attempt_col, prefix in [
    #     ('vehicle_type', 'lv_type_attempt_number', 'type'),
    #     ('vehicle_family', 'lv_family_attempt_number', 'family'),
    #     ('launch_provider', 'lv_provider_attempt_number', 'provider'),
    #     ('vehicle_variant', 'lv_variant_attempt_number', 'variant'),                 # <-- added
    #     ('vehicle_minor_variant', 'lv_minor_variant_attempt_number', 'minor_variant') # <-- added
    # ]:
    #     # launch_data_test = add_binned_reliability_combo(
    #     #     launch_data_test,
    #     #     attempt_col=attempt_col,
    #     #     since_fail_col=f"{prefix}_launches_since_last_failure",
    #     #     launch_bins=launch_bins,
    #     #     prefix=prefix
    #     # )

    # months since last launch
    for group_col, prefix in [
        ('vehicle_type', 'type'),
        ('vehicle_family', 'family'),
        ('launch_provider', 'provider'),
        ('vehicle_variant', 'variant'),                 # <-- added
        ('vehicle_minor_variant', 'minor_variant')      # <-- added
    ]:
        launch_data_test = add_months_since_last_launch(
            launch_data_test,
            group_col=group_col,
            date_col='launch_date',
            output_col=f'{prefix}_months_since_last_launch'
        )

    return launch_data_test

def _add_first_five_failure_stats(launch_data_test):
    grouping_levels = [
        ('vehicle_type', 'lv_type_attempt_number', 'type'),
        ('vehicle_family', 'lv_family_attempt_number', 'family'),
        ('launch_provider', 'lv_provider_attempt_number', 'provider'),
        ('vehicle_variant', 'lv_variant_attempt_number', 'variant'),                 # <-- added
        ('vehicle_minor_variant', 'lv_minor_variant_attempt_number', 'minor_variant') # <-- added
    ]
    for group_col, attempt_col, prefix in grouping_levels:
        stats_df = compute_first_five_failure_stats(
            launch_data_test,
            group_col=group_col,
            attempt_col=attempt_col,
            loss_col="has_loss"
        )
        new_cols = [col for col in stats_df.columns if col.lower() in ['failure_count_first_five', 'first_five_failure']]
        rename_map = {col: f'{prefix}_' + col.strip().lower() for col in new_cols}
        stats_subset = stats_df[["launch_number"] + new_cols].rename(columns=rename_map)
        launch_data_test = launch_data_test.merge(stats_subset, on="launch_number", how="left")
    return launch_data_test

def _categorize_first_five_failures(launch_data_test):
    launch_data_test['type_first_five_cat'] = launch_data_test['type_failure_count_first_five'].apply(
        lambda x: "0" if x == 0 else ("1" if x == 1 else "1+")
    )
    launch_data_test['family_first_five_cat'] = launch_data_test['family_failure_count_first_five'].apply(
        lambda x: "0" if x == 0 else ("1" if x == 1 else "1+")
    )
    launch_data_test['provider_first_five_cat'] = launch_data_test['provider_failure_count_first_five'].apply(
        lambda x: "0" if x == 0 else ("1" if x == 1 else "1+")
    )
    # NEW:
    if 'variant_failure_count_first_five' in launch_data_test.columns:
        launch_data_test['variant_first_five_cat'] = launch_data_test['variant_failure_count_first_five'].apply(
            lambda x: "0" if x == 0 else ("1" if x == 1 else "1+")
        )
    if 'minor_variant_failure_count_first_five' in launch_data_test.columns:
        launch_data_test['minor_variant_first_five_cat'] = launch_data_test['minor_variant_failure_count_first_five'].apply(
            lambda x: "0" if x == 0 else ("1" if x == 1 else "1+")
        )
    return launch_data_test


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
    Compute the order in which each `child_col` appears within `parent_col`,
    based on the earliest `date_col` of that (parent, child) combo.

    Creates:
      - `<iteration_col>` integer (Int64) dense rank starting at 1 within parent
      - `<grouped_col>` categorical: {first_label, other_label, NA}

    Example:
      add_iteration_within(df, "vehicle_family", "vehicle_type",
                           iteration_col="type_iteration",
                           grouped_col="type_iteration_grouped")
    """
    out = df.copy()
    out.loc[:, date_col] = pd.to_datetime(out[date_col], errors="coerce")

    # Sensible defaults for output names if not provided
    base = child_col.replace("vehicle_", "")
    if iteration_col is None:
        iteration_col = f"{base}_iteration"
    if grouped_col is None:
        grouped_col = f"{base}_iteration_grouped"

    # First launch per (parent, child)
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

    # Attach back
    out = out.merge(firsts[[parent_col, child_col, iteration_col]],
                    on=[parent_col, child_col], how="left")

    # Binary category with NA preserved when iteration is NA
    out[grouped_col] = np.where(
        out[iteration_col] == 1, first_label,
        np.where(out[iteration_col].notna(), other_label, pd.NA)
    )

    return out

# def _add_lsf_and_bins_for_levels(
#     df: pd.DataFrame,
#     levels: list[tuple[str, str, str]],
#     launch_bins = (1, 4, float("inf")),
# ) -> pd.DataFrame:
#     """
#     For each level, compute:
#       - <prefix>_launches_since_last_failure
#       - <prefix>_Attempt_Bin
#       - <prefix>_SinceFail_Bin   (includes 'Clean' where since_fail == attempt)
#       - <prefix>_Attempt_SinceFail_BinCombo

#     levels: list of tuples (group_col, attempt_col, prefix)
#     """
#     out = df.copy()
#     for group_col, attempt_col, prefix in levels:
#         # launches since last failure (uses utils function)
#         out = add_launches_since_last_failure(
#             out,
#             group_col=group_col,
#             attempt_col=attempt_col,
#             failure_col="has_loss",
#             output_col=f"{prefix}_launches_since_last_failure",
#         )

#     return out

def _add_lsf_and_bins_for_levels(
    df: pd.DataFrame,
    levels: list[tuple[str, str, str]],
    launch_bins = (1, 4, float("inf")),
) -> pd.DataFrame:
    out = df.copy()
    for group_col, attempt_col, prefix in levels:
        out = add_launches_since_last_failure(
            out,
            group_col=group_col,
            attempt_col=attempt_col,
            failure_col="has_loss",
            output_col=f"{prefix}_launches_since_last_failure"
        )
    return out


def _add_family_history_quality(df):
    """
    Adds a 'family_cum_fail_rate' and 'family_history_quality' column to the DataFrame.
    Replicates logic from fallback_base_rates.ipynb (Poor Family/Provider Experience section).
    """
    df = df.copy()
    # Ensure required columns exist before proceeding
    _check_required_columns(df, ['vehicle_family', 'lv_family_attempt_number', 'has_loss', 'type_iteration'], "input DataFrame for family history quality")

    # Sort for cumulative logic
    df = df.sort_values(by=['vehicle_family', 'lv_family_attempt_number'])

    # Cumulative failures and attempts per family (excluding current row)
    df['family_cumulative_failures'] = (
        df.groupby('vehicle_family')['has_loss']
        .transform(lambda x: x.shift().cumsum())
    )
    df['family_cumulative_attempts'] = df['lv_family_attempt_number'] - 1

    df['family_cum_fail_rate'] = df['family_cumulative_failures'] / df['family_cumulative_attempts'].replace(0, np.nan)

    def classify(row):
        if row['type_iteration'] == 1: # First type in a family has no prior family history
            return pd.NA
        if pd.isna(row['family_cum_fail_rate']):
            return pd.NA
        return 'Good' if row['family_cum_fail_rate'] < 0.33 else 'Poor'

    df['family_history_quality'] = df.apply(classify, axis=1)
    return df

def _final_filtering(launch_data_test, min_launch_year, filter_min_year_on, max_attempt_number, excluded_countries):
    launch_data_test['launch_year'] = (launch_data_test['launch_date'].dt.year).fillna(0).astype(int)
    if filter_min_year_on:
        launch_data_test = launch_data_test[
            launch_data_test["launch_year"] >= min_launch_year
        ]
    launch_data_test = launch_data_test[launch_data_test['lv_type_attempt_number'] <= max_attempt_number]
    launch_data_test = launch_data_test[~launch_data_test['launch_country'].isin(excluded_countries)]
    return launch_data_test

def _add_provider_rating(launch_data_test):
    """
    Adds a 'provider_rating' column to the dataframe based on cumulative failure rate and attempts.
    
    Parameters:
        launch_data_test (pd.DataFrame): The input DataFrame with launch data.
        
    Returns:
        pd.DataFrame: The input DataFrame with the added 'provider_rating' column.
    """
    # Step 1: Sort launches by provider and attempt number
    launch_data_test = launch_data_test.sort_values(by=['launch_provider', 'lv_provider_attempt_number'])

    # Step 2: Compute cumulative failures and attempts
    launch_data_test['provider_cum_failures'] = launch_data_test.groupby('launch_provider')['has_loss'].cumsum().shift(1).fillna(0)
    launch_data_test['provider_cum_attempts'] = launch_data_test.groupby('launch_provider').cumcount()
    launch_data_test['provider_cum_fail_rate'] = (
        launch_data_test['provider_cum_failures'] / launch_data_test['provider_cum_attempts'].replace(0, np.nan)
    )

    # Step 3: Cap the comparison group to attempt number 50
    launch_data_test['comparison_attempt'] = launch_data_test['lv_provider_attempt_number'].clip(upper=50)

    # Step 4: Compute percentile rank within the capped comparison group
    attempt_group = launch_data_test.groupby('comparison_attempt')['provider_cum_fail_rate']
    launch_data_test['provider_fail_rate_percentile'] = attempt_group.transform(lambda x: x.rank(pct=True))

    # Step 5: Define rating based on percentile and number of cumulative attempts
    def percentile_to_rating(p, attempts):
        if attempts < 5 or pd.isna(p):
            return 'Very New Provider'
        elif attempts <= 20:
            return 'New Provider'
        elif p <= 0.2:
            return 'A'  # Best
        elif p <= 0.8:
            return 'B'
        else:
            return 'C'  # Worst

    # Step 6: Apply rating logic row-wise
    launch_data_test['provider_rating'] = launch_data_test.apply(
        lambda row: percentile_to_rating(row['provider_fail_rate_percentile'], row['provider_cum_attempts']),
        axis=1
    )

    return launch_data_test

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
    Extract, standardize, engineer features, and apply final filtering.

    Parameters
    ----------
    server, database, spacecraft_table, events_table : str
        Data source location and table names.
    min_launch_year : int
        Minimum launch year to keep (used only if filter_min_year_on=True).
    max_attempt_number : int
        Maximum allowed attempt number (row kept if <= value).
    excluded_countries : list[str] or None
        Launch countries to exclude (case-insensitive).
    filter_min_year_on : bool
        Whether to apply the min_launch_year filter.

    Returns
    -------
    pd.DataFrame
        Final prepared dataset.
    """
    # ---- Defaults & guards -------------------------------------------------
    if excluded_countries is None:
        excluded_countries = ["iran", "north korea"]

    # Ensure excluded countries are normalized (lowercase, stripped)
    excluded_countries = [str(c).strip().lower() for c in excluded_countries]

    # Make sure max_attempt_number is a scalar integer
    if not np.isscalar(max_attempt_number):
        raise TypeError(
            f"max_attempt_number must be a scalar int, got {type(max_attempt_number)} "
            f"with value {max_attempt_number}"
        )
    max_attempt_number = int(max_attempt_number)

    # ---- Select only needed columns to reduce IO ---------------------------
    original_spacecraft_cols = [
        "Seradata Spacecraft ID", "Seradata Launch ID", "Launch Number", "Launch Date",
        "Vehicle Type", "Orbit Category", "Launch Type", "Vehicle Family",
        "Launch Provider", "Vehicle Variant", "Vehicle Minor Variant",  # <-- added
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

    # ---- Standardize column names (lowercase + underscores, etc.) ----------
    sql_spacecraft = _standardize_columns(sql_spacecraft)
    sql_events = _standardize_columns(sql_events)

    required_spacecraft_cols = [
        "seradata_spacecraft_id", "seradata_launch_id", "launch_number", "launch_date",
        "vehicle_type", "orbit_category", "launch_type", "vehicle_family",
        "launch_provider", "vehicle_variant", "vehicle_minor_variant",  # <-- added
        "launch_country", "sector",
    ]

    required_events_cols = [
        "seradata_spacecraft_id", "capability_loss_(percent_fraction)",
        "event_date_and_time", "spacecraft_event_description",
    ]
    _check_required_columns(sql_spacecraft, required_spacecraft_cols, "spacecraft table")
    _check_required_columns(sql_events, required_events_cols, "events table")

    # ---- Ensure date-like columns are datetimes ----------------------------
    for col in ("launch_date",):
        if col in sql_spacecraft.columns and not np.issubdtype(sql_spacecraft[col].dtype, np.datetime64):
            sql_spacecraft[col] = pd.to_datetime(sql_spacecraft[col], errors="coerce")
    for col in ("event_date_and_time",):
        if col in sql_events.columns and not np.issubdtype(sql_events[col].dtype, np.datetime64):
            sql_events[col] = pd.to_datetime(sql_events[col], errors="coerce")

    # ---- Core transformations ----------------------------------------------
    launch_events = _filter_events(sql_events)
    launches = _prepare_launches(sql_spacecraft)
    launch_events = _prepare_launch_events(launch_events)
    aggregated_data = _merge_and_aggregate(launches, launch_events)
    aggregated_data = _feature_engineering(aggregated_data, loss_threshold=loss_threshold)
    aggregated_data = _assign_attempt_numbers(aggregated_data)

    # Work on a copy to avoid chained-setting warnings later
    launch_data_test = aggregated_data.copy()

    # ---- Feature block ------------------------------------------------------
    launch_data_test = _add_engineered_features(launch_data_test)
    vc_final = (
        launch_data_test.loc[launch_data_test['vehicle_type'] == 'ZHUQUE-2/SUZAKU-2 (ZQ-2)',
                            ['vehicle_type', 'lv_type_attempt_number', 'has_loss', 'type_launches_since_last_failure']]
        .sort_values('lv_type_attempt_number')
    )
    print("\n[post-pipeline] VULCAN CENTAUR final values:")
    print(vc_final.to_string(index=False))
    launch_data_test = _add_first_five_failure_stats(launch_data_test)
    launch_data_test = _categorize_first_five_failures(launch_data_test)
    launch_data_test = _add_provider_rating(launch_data_test)
    launch_data_test = add_iteration_within(
        launch_data_test,
        parent_col="vehicle_family",
        child_col="vehicle_type",
        iteration_col="type_iteration",
        grouped_col="type_iteration_grouped"
    )

    # Variant within Type
    launch_data_test = add_iteration_within(
        launch_data_test,
        parent_col="vehicle_type",
        child_col="vehicle_variant",
        iteration_col="variant_iteration",
        grouped_col="variant_iteration_grouped"
    )

    # Minor Variant within Type
    launch_data_test = add_iteration_within(
        launch_data_test,
        parent_col="vehicle_type",
        child_col="vehicle_minor_variant",
        iteration_col="minor_variant_iteration",
        grouped_col="minor_variant_iteration_grouped"
    )
    launch_data_test = _add_family_history_quality(launch_data_test)

    # ---- Final filtering ----------------------------------------------------
    # Normalize country labels to lowercase once for consistent filtering
    if "launch_country" in launch_data_test.columns:
        launch_data_test["launch_country"] = launch_data_test["launch_country"].str.strip().str.lower()

    launch_data_test = _final_filtering(
        launch_data_test,
        min_launch_year=min_launch_year,
        max_attempt_number=max_attempt_number,
        excluded_countries=excluded_countries,
        filter_min_year_on=filter_min_year_on,
    )

    return launch_data_test

def main():
    temp = load_and_prepare_data()
    print(temp.head)

if __name__ == "__main__":
    main()