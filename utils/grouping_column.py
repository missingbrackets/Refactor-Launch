# utils/grouping.py
import pandas as pd
from typing import List, Tuple, Dict


# --- Define grouping columns for analysis ---
GROUPING_COLUMN_MAPPING: Dict[str, Dict[str, List[str] | str]] = {
    "vehicle_family": {
        "columns": [
            "vehicle_family",
            "seradata_spacecraft_id",
            "lv_family_attempt_number",
            "launch_date",
            "has_loss",
            "vehicle_type",
            "lv_type_attempt_number",
            "family_launches_since_last_failure",
        ],
        "attempt_column": "lv_family_attempt_number",
    },
    "vehicle_type": {
        "columns": [
            "vehicle_type",
            "seradata_spacecraft_id",
            "lv_type_attempt_number",
            "launch_date",
            "has_loss",
            "type_launches_since_last_failure",
        ],
        "attempt_column": "lv_type_attempt_number",
    },
    "launch_provider": {
        "columns": [
            "seradata_spacecraft_id",
            "launch_provider",
            "lv_provider_attempt_number",
            "launch_date",
            "vehicle_family",
            "lv_family_attempt_number",
            "has_loss",
            "provider_launches_since_last_failure",
        ],
        "attempt_column": "lv_provider_attempt_number",
    },
    # --- NEW: Variant within Type ---
    "vehicle_variant": {
        "columns": [
            "vehicle_variant",
            "seradata_spacecraft_id",
            "lv_variant_attempt_number",
            "launch_date",
            "has_loss",
            "vehicle_type",              # useful context; often handy for QA/slicing
            "lv_type_attempt_number",
            "variant_launches_since_last_failure",
        ],
        "attempt_column": "lv_variant_attempt_number",
    },
    # --- NEW: Minor Variant within Type ---
    "vehicle_minor_variant": {
        "columns": [
            "vehicle_minor_variant",
            "seradata_spacecraft_id",
            "lv_minor_variant_attempt_number",
            "launch_date",
            "has_loss",
            "vehicle_type",
            "lv_type_attempt_number",
            "minor_variant_launches_since_last_failure",
        ],
        "attempt_column": "lv_minor_variant_attempt_number",
    },
}


def select_grouping_columns(
    df: pd.DataFrame,
    grouping_col: str,
) -> Tuple[pd.DataFrame, str, List[str]]:
    """
    Select columns and attempt column based on the given grouping_col.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the columns.
    grouping_col : str
        Must exactly match one of the keys in GROUPING_COLUMN_MAPPING.

    Returns
    -------
    (df_selected, attempt_column, selected_columns)
    """
    if grouping_col not in GROUPING_COLUMN_MAPPING:
        raise ValueError(f"Unsupported grouping column: {grouping_col}")

    selected_columns = GROUPING_COLUMN_MAPPING[grouping_col]["columns"]  # type: ignore
    attempt_column = GROUPING_COLUMN_MAPPING[grouping_col]["attempt_column"]  # type: ignore

    return df[selected_columns], attempt_column, selected_columns
