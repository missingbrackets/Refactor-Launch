import pandas as pd
import numpy as np

def compute_empirical_failure_rates(
    data: pd.DataFrame,
    max_attempt: int = 20,
    outcome_col: str = "has_loss",
    groupings: list = None
) -> pd.DataFrame:
    """
    Compute empirical (raw) failure rates for each attempt number and grouping.
    Returns a DataFrame with columns for each grouping and attempt number.
    """
    if groupings is None:
        groupings = [
            ("vehicle_type", "lv_type_attempt_number", "type"),
            ("vehicle_family", "lv_family_attempt_number", "family"),
            ("launch_provider", "lv_provider_attempt_number", "provider")
        ]
    result = {}
    for group_col, attempt_col, label in groupings:
        rates = []
        for attempt in range(1, max_attempt + 1):
            mask = data[attempt_col] == attempt
            if mask.sum() == 0:
                rates.append(np.nan)
            else:
                rates.append(data.loc[mask, outcome_col].mean())
        result[label] = rates
    out_df = pd.DataFrame({label: result[label] for _, _, label in groupings})
    out_df["attempt_number"] = np.arange(1, max_attempt + 1)
    out_df = out_df[["attempt_number"] + [label for _, _, label in groupings]]
    return out_df