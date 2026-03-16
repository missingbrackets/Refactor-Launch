from __future__ import annotations
from .data_load_feature_creation import load_and_prepare_data
from ..utils.output_to_csv import output_df_to_csv
from ..utils.grouping_column import select_grouping_columns
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from datetime import datetime
from datetime import datetime
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import logging

def subset_data_by_failures(data, groupby_column, attempt_column, attempt_numbers):
    """
    Subset data based on the number of failures in specific attempts.
    """
    no_failures = data.groupby(groupby_column).filter(
        lambda x: x.loc[x[attempt_column].isin(attempt_numbers), 'has_loss'].sum() == 0
    )
    one_failure = data.groupby(groupby_column).filter(
        lambda x: x.loc[x[attempt_column].isin(attempt_numbers), 'has_loss'].sum() == 1
    )
    two_failures = data.groupby(groupby_column).filter(
        lambda x: x.loc[x[attempt_column].isin(attempt_numbers), 'has_loss'].sum() == 2
    )
    return no_failures, one_failure, two_failures

def bayesian_learning_curve(t, fail_rate, decay_rate, decay_exponent, prior_weight):
    """
    Generalized Bayesian learning curve:
    Predicts failure probability after `t` launches.
    
    Parameters:
    - fail_rate: initial belief of failure probability
    - decay_rate: how quickly learning occurs (λ)
    - decay_exponent: shape of learning curve (δ)
    - prior_weight: confidence in prior (pseudo-observations)
    
    Returns:
    - Estimated failure probability at each launch number `t`
    """
    alpha_prior = prior_weight * fail_rate
    beta_prior = prior_weight * (1 - fail_rate)
    denominator = alpha_prior + beta_prior + (decay_rate * t) ** decay_exponent
    return alpha_prior / denominator


def compute_empirical_cumulative_failure(df, attempt_column):
    """
    Computes empirical cumulative failure probability per attempt number.
    Skips launches before attempt #3 (too early for learning dynamics).
    
    Returns:
    - DataFrame with columns: Attempt Number, Empirical Failure Probability
    """
    df = df[df[attempt_column] >= 3].copy()
    df = df.sort_values(attempt_column)
    
    grouped = df.groupby(attempt_column)
    cumulative_failures = grouped['has_loss'].sum().cumsum()
    cumulative_counts = grouped['has_loss'].count().cumsum()
    
    empirical_probs = (cumulative_failures / cumulative_counts).reset_index()
    empirical_probs.columns = [attempt_column, 'Empirical_Failure_Probability']
    
    return empirical_probs


def fit_learning_curve(df, attempt_column):
    """
    Fits a Bayesian learning curve to the empirical failure probabilities.
    
    Returns:
    - t_fit: Launch attempt numbers used for prediction
    - p_fit: Predicted failure probabilities
    - alpha, beta: Inferred prior parameters
    - λ, δ, prior_weight: Fitted learning curve parameters
    """
    empirical_data = compute_empirical_cumulative_failure(df, attempt_column)
    t = empirical_data[attempt_column].values
    p = empirical_data['Empirical_Failure_Probability'].values

    # Initial guess and bounds for [fail_rate, λ, δ, prior_weight]
    initial_guess = [0.1, 1.0, 1.0, 10.0]
    bounds = ([0.001, 0.01, 0.1, 2], [0.999, 100.0, 5.0, 500.0])

    # Fit curve to empirical data
    params, _ = curve_fit(bayesian_learning_curve, t, p, p0=initial_guess, bounds=bounds)
    fail_rate, decay_rate, decay_exponent, prior_weight = params

    # Compute prior alpha and beta for downstream use
    alpha = prior_weight * fail_rate
    beta = prior_weight * (1 - fail_rate)

    # Predict failure probabilities over extended range
    t_fit = np.arange(3, max(t) + 10)
    p_fit = bayesian_learning_curve(t_fit, fail_rate, decay_rate, decay_exponent, prior_weight)

    return t_fit, p_fit, alpha, beta, decay_rate, decay_exponent, prior_weight

# --- Predict failure probability on the 3rd launch ---
def predict_third_launch_failure(alpha, beta, λ=1.0, δ=0.24, t=99):
    """
    Estimate failure probability for the next launch using fixed t=99
    as a proxy for future performance extrapolation.
    """
    return round(100 * (alpha / (alpha + beta + t ** δ)), 2)

def build_learning_curve_dataframe(results_dict, attempt_column):
    """
    Build a DataFrame comparing fitted learning curves across different early failure scenarios.

    Parameters:
        results_dict (dict): A dictionary where each key is a scenario label (e.g., "No_Failures"),
                             and each value is a tuple of the form (t_fit, p_fit, alpha, beta, λ, δ, weight).

    Returns:
        pd.DataFrame: A table where each column is a predicted failure probability curve for a scenario,
                      and the index is the aligned Attempt Number (padded with NaN where needed).
    """
    # Determine the longest time array
    max_len = max(len(v[0]) for v in results_dict.values())

    # Helper: pad a 1D array to target length with NaNs
    def pad(array, target_len):
        array = np.asarray(array, dtype=float)
        return np.pad(array, (0, target_len - len(array)), mode='constant', constant_values=np.nan)

    # Create DataFrame
    df = {}

    # Add the common Attempt Number (we'll use the one from the first scenario as index)
    first_key = next(iter(results_dict))
    df[attempt_column] = pad(results_dict[first_key][0], max_len)

    # Add all scenarios
    for label, (t_fit, p_fit, *_params) in results_dict.items():
        readable_label = label.replace("_", " ").title()
        df[readable_label] = pad(p_fit, max_len)

    return pd.DataFrame(df)

# -------------------- Configuration --------------------
GROUPINGS = ["vehicle_type", "vehicle_family", "launch_provider"]
OUTPUT_DIR = Path("src/launch_analysis/outputs")
TZ = ZoneInfo("Europe/London")  # consistent, explicit timezone for filenames

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)
# ------------------------------------------------------


def main() -> None:
    """
    Run the fallback rate fitting for multiple groupings and export CSVs.
    Exports TWO CSVs per grouping:
      1) <prefix>_learning_curve.csv
      2) <prefix>_fitted_parameters.csv

    Assumptions:
      - `load_and_prepare_data`, `select_grouping_columns`, `subset_data_by_failures`,
        `fit_learning_curve`, and `build_learning_curve_dataframe` are available
        and behave as previously discussed.
    """
    logger.info("Loading and preparing data...")
    launch_df = load_and_prepare_data()
    logger.info("Columns present: %s", list(launch_df.columns))

    # Create a stable, timezone-aware date tag for filenames
    date_tag = datetime.now(tz=TZ).strftime("%Y%m%d")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for grouping_col in GROUPINGS:
        logger.info("=== Processing grouping: %s ===", grouping_col)

        # 1) Select only the columns relevant to this grouping
        clean_launch_data, attempt_column, selected_columns = select_grouping_columns(
            launch_df, grouping_col
        )
        
        logger.debug("Selected columns for %s: %s", grouping_col, selected_columns)
        logger.debug("Attempt column for %s: %s", grouping_col, attempt_column)

        # 2) Build scenario subsets (first two attempts as per your spec)
        no_fail, one_fail, two_fail = subset_data_by_failures(
            clean_launch_data,
            grouping_col,
            attempt_column,
            [1, 2],
        )

        scenarios: dict[str, pd.DataFrame] = {
            "all": launch_df,
            "no_failures": no_fail,
            "one_failure": one_fail,
            "two_failures": two_fail,
        }

        # 3) Fit curves for each scenario (skip gracefully if a subset is empty)
        results: dict[str, tuple] = {}
        for label, df in scenarios.items():
            if df.empty:
                logger.warning("Scenario '%s' for '%s' is empty. Skipping fit.", label, grouping_col)
                continue
            results[label] = fit_learning_curve(df, attempt_column)

        if not results:
            logger.warning("No scenarios produced results for '%s'. Skipping exports.", grouping_col)
            continue

        # 4) (Optional console summary) Log fitted parameters
        for label, (t_fit, p_fit, alpha, beta, lam, delta, weight) in results.items():
            logger.info(
                "%s | %s - Fitted: alpha=%.2f, beta=%.2f, λ=%.2f, δ=%.2f, prior_weight=%.2f",
                grouping_col, label.replace("_", " "), alpha, beta, lam, delta, weight
            )

        # 5) Build learning-curve table from fits
        curve_df = build_learning_curve_dataframe(results, attempt_column)
        logger.debug("Learning curve preview for %s:\n%s", grouping_col, curve_df.head())
        curve_df = curve_df.rename(
            columns = {"All": "all", "No Failures": "no_failures", "One Failure": "one_failure", "Two Failures": "two_failures"}
        )

        # 6) Build fitted-parameters table
        params_rows: list[dict] = []
        for label, (t_fit, p_fit, alpha, beta, lam, delta, weight) in results.items():
            params_rows.append(
                {
                    "Grouping": grouping_col,
                    "Scenario": label,
                    "alpha": alpha,
                    "beta": beta,
                    "lambda": lam,
                    "delta": delta,
                    "prior_weight": weight,
                    "t_fit": t_fit,
                    "p_fit": p_fit,
                    "attempt_column": attempt_column,
                }
            )
        params_df = pd.DataFrame(params_rows)

        # 7) Export ONLY the requested CSVs + JSONs
        prefix = f"launch_learning_curve_{grouping_col}_{date_tag}"
        curve_path = OUTPUT_DIR / f"{prefix}_learning_curve.csv"
        params_path = OUTPUT_DIR / f"{prefix}_fitted_parameters.csv"

        # CSV exports
        curve_df.to_csv(curve_path, index=False)
        params_df.to_csv(params_path, index=False)

        # JSON exports (same names, but .json extension)
        curve_json_path = curve_path.with_suffix(".json")
        params_json_path = params_path.with_suffix(".json")

        curve_df.to_json(curve_json_path, orient="records", indent=2)
        params_df.to_json(params_json_path, orient="records", indent=2)

        logger.info(
            "Saved CSVs & JSONs for %s:\n- %s\n- %s\n- %s\n- %s",
            grouping_col,
            curve_path,
            params_path,
            curve_json_path,
            params_json_path,
        )



if __name__ == "__main__":
    main()
