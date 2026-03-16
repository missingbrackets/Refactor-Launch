import pandas as pd
import re
from .constants import (
    IDENTITY_COLS, ATTEMPT_COLS, INCLUDE_BY_GROUPING, LEVELS
)

def _prettify_grouping_label(s: str) -> str:
    """
    Turn 'vehicle_type' -> 'Vehicle Type'
    Collapses underscores/hyphens/extra spaces and Title-Cases.
    """
    s = re.sub(r'[_\-\s]+', ' ', str(s)).strip()
    return s.title()

def _mode_or_first(series: pd.Series):
    """Safe mode for object columns; falls back to first non-null if no unique mode."""
    s = series.dropna()
    if s.empty:
        return None
    m = s.mode()
    return m.iloc[0] if not m.empty else s.iloc[0]

def build_dropdown_rows_for_grouping(launch_df: pd.DataFrame, grouping_col: str) -> pd.DataFrame:
    """
    For a given grouping column, build one row per unique value with:
      - identity columns (per INCLUDE_BY_GROUPING rule)
      - max attempt numbers across history
      - total_failures = sum(has_loss)
      - *_launches_since_last_failure taken at the row with the group's max attempt for that level
      - provider_rating, *_iteration_grouped taken at the row with the group's max attempt for that level

    Assumes all *_launches_since_last_failure columns are already present in launch_df.
    """
    if grouping_col not in launch_df.columns:
        raise KeyError(f"Column '{grouping_col}' not found in DataFrame.")

    df = launch_df.dropna(subset=[grouping_col]).copy()
    if df.empty:
        return pd.DataFrame(columns=["grouping_col", "value"])

    # ---- Level metadata: (key, identity_col, attempt_col, snapshot_value_cols) ----
    LEVELS = [
        ("type",          "vehicle_type",         "lv_type_attempt_number",          ["type_iteration_grouped"]),
        ("family",        "vehicle_family",       "lv_family_attempt_number",        []),
        ("provider",      "launch_provider",      "lv_provider_attempt_number",      ["provider_rating"]),
        ("variant",       "vehicle_variant",      "lv_variant_attempt_number",       ["variant_iteration_grouped"]),
        ("minor_variant", "vehicle_minor_variant","lv_minor_variant_attempt_number", ["minor_variant_iteration_grouped"]),
    ]

    # Only aggregate columns that actually exist
    attempt_cols_exist = [c for c in ATTEMPT_COLS if c in df.columns]
    id_cols_exist = [c for c in IDENTITY_COLS if c in df.columns and c != grouping_col]

    # ---- 1) Aggregate maxima / identities ----
    agg_spec = {**{c: "max" for c in attempt_cols_exist}, "has_loss": "sum"}
    agg_spec.update({c: _mode_or_first for c in id_cols_exist})

    agg = (
        df.groupby(grouping_col, as_index=False)
          .agg(agg_spec)
          .rename(columns={grouping_col: "value", "has_loss": "total_failures"})
    )

    # ---- 2) Snapshot per level at the row with the group's max attempt ----
    snapshots_pieces = []

    for key, _id_col, att_col, snap_cols in LEVELS:
        if att_col not in df.columns:
            continue

        # groupwise max attempt for this level
        grp_max = df.groupby(grouping_col, dropna=False)[att_col].transform("max")

        # rows at the group max (NaN-safe)
        at_max = df[att_col].eq(grp_max) & df[att_col].notna()

        # columns to pull from the chosen row for this level
        since_col = f"{key}_launches_since_last_failure"
        cols = [grouping_col]
        if since_col in df.columns:
            cols.append(since_col)
        cols += [c for c in snap_cols if c in df.columns]

        if len(cols) == 1:  # nothing to snapshot for this level
            continue

        # keep the first occurrence at max per group (ties -> first)
        pick = (
            df.loc[at_max, cols]
              .drop_duplicates(subset=[grouping_col], keep="first")
              .rename(columns={grouping_col: "value"})
        )
        snapshots_pieces.append(pick)

    # combine per-level snapshots (outer in case some levels don't exist for some groups)
    if snapshots_pieces:
        snapshots = snapshots_pieces[0]
        for piece in snapshots_pieces[1:]:
            snapshots = snapshots.merge(piece, on="value", how="outer")
    else:
        snapshots = pd.DataFrame(columns=["value"])

    # Merge snapshots into the aggregation
    agg = agg.merge(snapshots, on="value", how="left")

    # ---- 3) Keep/format identity columns per your rule ----
    keep_id_cols = INCLUDE_BY_GROUPING.get(grouping_col, [grouping_col])

    if grouping_col in keep_id_cols and grouping_col not in agg.columns:
        agg[grouping_col] = agg["value"]

    # Pretty grouping label
    agg.insert(0, "grouping_col", _prettify_grouping_label(grouping_col))

    # Normalize identity columns to strings (clean None)
    for c in set(IDENTITY_COLS) & set(agg.columns):
        agg[c] = agg[c].astype(str).where(agg[c].notna(), None)

    # Cast attempts + failures + since counters to nullable int
    since_cols_all = [
        "type_launches_since_last_failure",
        "family_launches_since_last_failure",
        "provider_launches_since_last_failure",
        "variant_launches_since_last_failure",
        "minor_variant_launches_since_last_failure",
    ]
    since_cols_exist = [c for c in since_cols_all if c in agg.columns]

    for c in attempt_cols_exist + ["total_failures"] + since_cols_exist:
        if c in agg.columns:
            agg[c] = agg[c].astype("Int64")

    # Columns we might have snapshotted in step 2
    snapshot_cols_all = [
        "provider_rating",
        "type_iteration_grouped",
        "variant_iteration_grouped",
        "minor_variant_iteration_grouped",
    ]
    snapshot_cols_exist = [c for c in snapshot_cols_all if c in agg.columns]

    # ---- 4) Final column order ----
    ordered_cols = (
        ["grouping_col", "value"]
        + [c for c in keep_id_cols if c in agg.columns]
        + attempt_cols_exist
        + since_cols_exist
        + snapshot_cols_exist
        + (["total_failures"] if "total_failures" in agg.columns else [])
    )
    agg = agg[ordered_cols].sort_values(["grouping_col", "value"], kind="mergesort").reset_index(drop=True)

    return agg

