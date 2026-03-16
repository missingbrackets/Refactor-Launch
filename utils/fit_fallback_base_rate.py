import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.isotonic import IsotonicRegression

from ..utils.compute_empirical_failure_rates import compute_empirical_failure_rates


DEFAULT_GROUPINGS = [
    ("vehicle_type", "lv_type_attempt_number", "type"),
    ("vehicle_family", "lv_family_attempt_number", "family"),
    ("launch_provider", "lv_provider_attempt_number", "provider"),
]


def _apply_monotonic_decreasing(attempts: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Force values to be monotonically decreasing over attempts."""
    ir = IsotonicRegression(increasing=False, out_of_bounds="clip")
    return ir.fit_transform(attempts, values)


def _fit_glm_with_isotonic(
    data: pd.DataFrame,
    attempt_col: str,
    outcome_col: str,
    max_attempt: int,
) -> np.ndarray:
    """
    Fit a quadratic binomial GLM and enforce monotonic decreasing rates
    across attempt numbers with isotonic regression.
    """
    df = data.loc[data[attempt_col] <= max_attempt, [attempt_col, outcome_col]].dropna().copy()
    df["attempt_sq"] = df[attempt_col] ** 2

    model = smf.glm(
        formula=f"{outcome_col} ~ {attempt_col} + attempt_sq",
        data=df,
        family=sm.families.Binomial(),
    )
    fitted_model = model.fit()

    pred_df = pd.DataFrame({attempt_col: np.arange(1, max_attempt + 1)})
    pred_df["attempt_sq"] = pred_df[attempt_col] ** 2

    predicted_probs = fitted_model.predict(pred_df)
    return _apply_monotonic_decreasing(pred_df[attempt_col].to_numpy(), predicted_probs.to_numpy())


def _fit_empirical_with_isotonic(
    data: pd.DataFrame,
    attempt_col: str,
    outcome_col: str,
    max_attempt: int,
) -> np.ndarray:
    """
    Fallback estimator: compute empirical failure rates by attempt number
    and enforce monotonic decreasing behavior with isotonic regression.
    """
    empirical_probs = []
    for attempt in range(1, max_attempt + 1):
        mask = data[attempt_col] == attempt
        empirical_probs.append(data.loc[mask, outcome_col].mean() if mask.any() else np.nan)

    empirical_probs = np.array(empirical_probs, dtype=float)
    valid_mask = ~np.isnan(empirical_probs)

    if valid_mask.sum() <= 1:
        return empirical_probs

    monotone_probs = np.full(max_attempt, np.nan, dtype=float)
    fitted = _apply_monotonic_decreasing(
        np.arange(1, max_attempt + 1)[valid_mask],
        empirical_probs[valid_mask],
    )
    monotone_probs[valid_mask] = fitted
    return monotone_probs


def _fit_rates_for_group(
    data: pd.DataFrame,
    attempt_col: str,
    outcome_col: str,
    max_attempt: int,
) -> np.ndarray:
    """Fit model-based rates, falling back to empirical rates on failure."""
    try:
        return _fit_glm_with_isotonic(data, attempt_col, outcome_col, max_attempt)
    except Exception:
        return _fit_empirical_with_isotonic(data, attempt_col, outcome_col, max_attempt)


def fit_rates_model(
    data: pd.DataFrame,
    max_attempt: int = 20,
    outcome_col: str = "has_loss",
    groupings: list | None = None,
) -> pd.DataFrame:
    """
    Fit monotonic failure-rate curves over launch attempt number.

    For each grouping definition, this function:
    1. fits a quadratic binomial GLM for the outcome,
    2. predicts probabilities for attempts 1..max_attempt,
    3. enforces monotonically decreasing probabilities with isotonic regression,
    4. falls back to empirical attempt-level rates if model fitting fails.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe containing attempt-number columns and the outcome column.
    max_attempt : int, default=20
        Maximum attempt number to model.
    outcome_col : str, default="has_loss"
        Name of the binary outcome column.
    groupings : list[tuple[str, str, str]] | None, default=None
        Each tuple is (group_col, attempt_col, label). Only attempt_col and label
        are used here; group_col is retained for compatibility with the empirical
        helper and calling code.

    Returns
    -------
    pd.DataFrame
        DataFrame with:
        - attempt_number
        - one fitted probability column per grouping label
        - one empirical probability column per grouping label
    """
    if groupings is None:
        groupings = DEFAULT_GROUPINGS

    fitted_results = {}
    for _, attempt_col, label in groupings:
        fitted_results[label] = _fit_rates_for_group(
            data=data,
            attempt_col=attempt_col,
            outcome_col=outcome_col,
            max_attempt=max_attempt,
        )

    out_df = pd.DataFrame({"attempt_number": np.arange(1, max_attempt + 1), **fitted_results})

    empirical_df = compute_empirical_failure_rates(
        data=data,
        max_attempt=max_attempt,
        outcome_col=outcome_col,
        groupings=groupings,
    )

    for _, _, label in groupings:
        out_df[f"empirical_{label}"] = empirical_df[label]

    return out_df