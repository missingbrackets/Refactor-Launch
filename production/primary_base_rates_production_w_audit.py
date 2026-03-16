from .data_load_feature_creation import load_and_prepare_data
from ..utils.grouping_column import select_grouping_columns
from ..utils.compute_empirical_cumulative_loss import compute_empirical_cumulative_has_loss
from ..utils.build_dropdown_columns import build_dropdown_rows_for_grouping, _prettify_grouping_label
from ..utils.constants import (
    GROUPINGS, OUTPUT_DIR, TZ, IDENTITY_COLS, ATTEMPT_COLS, INCLUDE_BY_GROUPING,
    SPECIFIC_GROUPINGS, SPECIFIC_COLS, TYPE_COLS
)
from ..utils.audit_helpers import _ensure_parent, _to_native, _normalize_table, _with_ext, audit_table, audit_schema, _safe_name
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.optimize import curve_fit
from scipy.stats import beta as _beta
from typing import Dict, Any
import re

def _rename_rate_cols(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Rename standard primary-rate column set with a suffix."""
    return df.rename(
        columns={
            "next_launch_number": f"next_launch_number{suffix}",
            "launch_date": f"launch_date{suffix}",
            "total_failures": f"total_failures{suffix}",
            "failure_rate": f"failure_rate{suffix}",
            "ci_lower": f"ci_lower{suffix}",
            "ci_upper": f"ci_upper{suffix}",
        }
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)
# ------------------------------------------------------

def bayesian_learning_curve(t, fail_rate, decay_rate, decay_exponent, prior_weight):
    """
    Generalized Bayesian learning curve:
    Predicts has_loss probability after `t` launches.
    
    Parameters:
    - fail_rate: initial belief of has_loss probability
    - decay_rate: how quickly learning occurs (λ)
    - decay_exponent: shape of learning curve (δ)
    - prior_weight: confidence in prior (pseudo-observations)
    
    Returns:
    - Estimated has_loss probability at each launch number `t`
    """
    alpha_prior = prior_weight * fail_rate
    beta_prior = prior_weight * (1 - fail_rate)
    denominator = alpha_prior + beta_prior + (decay_rate * t) ** decay_exponent
    return alpha_prior / denominator

def fit_learning_curve(
    df: pd.DataFrame,
    attempt_column: str,
    *,
    audit: bool = False,
    audit_path: Path | None = None,
    audit_context: dict | None = None,  # optional: e.g. {"grouping_col": "..."}
):
    """
    Fits a Bayesian learning curve to the empirical has_loss probabilities.

    Returns:
    - t_fit: Launch attempt numbers used for prediction (np.ndarray)
    - p_fit: Predicted has_loss probabilities on t_fit (np.ndarray)
    - alpha, beta: Inferred prior parameters (floats)
    - decay_rate (λ), decay_exponent (δ), prior_weight: fitted params (floats)

    If audit=True and audit_path provided, writes tables under:
      <audit_path>/learning_curve/
        00_empirical_data.(csv|xlsx)
        01_initial_guess_bounds.(csv|xlsx)
        02_fit_params.(csv|xlsx)
        03_prior_alpha_beta.(csv|xlsx)
        04_empirical_vs_fitted.(csv|xlsx)
        05_fit_metrics.(csv|xlsx)
        06_tfit_curve.(csv|xlsx)
        07_param_covariance.(csv|xlsx)
        08_context.(csv|xlsx)  [only if context provided]
        99_fit_error.(csv|xlsx) [only if optimization fails]
    """
    # --- derive empirical points ------------------------------------------------
    empirical_data = compute_empirical_cumulative_has_loss(df, attempt_column)
    t = empirical_data[attempt_column].to_numpy()
    p = empirical_data['Empirical_has_loss_Probability'].to_numpy()

    if audit and audit_path is not None:
        base = audit_path / "learning_curve"
        audit_table(True, base / "00_empirical_data", empirical_data, head=None)

    if t.size == 0:
        # degenerate case: no data at attempts >=3
        t_fit = np.array([], dtype=float)
        p_fit = np.array([], dtype=float)
        alpha = beta = np.nan
        decay_rate = decay_exponent = prior_weight = np.nan
        if audit and audit_path is not None:
            # record the fact we had no empirical points
            metrics = pd.DataFrame([{
                "n_points": 0, "rmse": np.nan, "mae": np.nan, "r2": np.nan,
                "note": "No empirical points (attempt >= 3)"
            }])
            audit_table(True, (audit_path / "learning_curve") / "05_fit_metrics", metrics, head=None)
        return t_fit, p_fit, alpha, beta, decay_rate, decay_exponent, prior_weight

    # --- initial guess and bounds ----------------------------------------------
    # [fail_rate, decay_rate (λ), decay_exponent (δ), prior_weight]
    initial_guess = np.array([0.1, 1.0, 1.0, 10.0], dtype=float)
    lower_bounds = np.array([0.001, 0.01, 0.1, 2.0], dtype=float)
    upper_bounds = np.array([0.999, 100.0, 5.0, 500.0], dtype=float)

    if audit and audit_path is not None:
        igb = pd.DataFrame({
            "param": ["fail_rate", "decay_rate", "decay_exponent", "prior_weight"],
            "initial_guess": initial_guess,
            "lower_bound": lower_bounds,
            "upper_bound": upper_bounds,
        })
        audit_table(True, (audit_path / "learning_curve") / "01_initial_guess_bounds", igb, head=None)

    # --- fit --------------------------------------------------------------------
    try:
        params, pcov = curve_fit(
            bayesian_learning_curve,
            t, p,
            p0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
            maxfev=10000,
        )
    except Exception as e:
        if audit and audit_path is not None:
            audit_table(True, (audit_path / "learning_curve") / "99_fit_error",
                        pd.DataFrame([{"error": str(e)}]), head=None)
        raise

    fail_rate, decay_rate, decay_exponent, prior_weight = params
    alpha = float(prior_weight * fail_rate)
    beta  = float(prior_weight * (1.0 - fail_rate))

    # predictions at empirical points (for residuals)
    p_hat_emp = bayesian_learning_curve(t, fail_rate, decay_rate, decay_exponent, prior_weight)

    # extended grid for plotting
    t_fit = np.arange(3, int(np.max(t)) + 10, dtype=int)
    p_fit = bayesian_learning_curve(t_fit, fail_rate, decay_rate, decay_exponent, prior_weight)

    # --- diagnostics ------------------------------------------------------------
    resid = p - p_hat_emp
    abs_resid = np.abs(resid)
    sq_resid = resid**2
    sse = float(np.sum(sq_resid))
    mae = float(np.mean(abs_resid))
    rmse = float(np.sqrt(np.mean(sq_resid)))
    p_mean = float(np.mean(p))
    sst = float(np.sum((p - p_mean)**2))
    r2 = float(1.0 - sse / sst) if sst > 0 else np.nan

    # parameter uncertainties (if covariance is sensible)
    se = np.full(4, np.nan, dtype=float)
    if isinstance(pcov, np.ndarray) and pcov.shape == (4, 4):
        diag = np.diag(pcov)
        # guard against negative/inf diag due to poor conditioning
        se = np.where(np.isfinite(diag) & (diag >= 0.0), np.sqrt(diag), np.nan)

    ci95 = 1.96 * se
    params_df = pd.DataFrame({
        "param": ["fail_rate", "decay_rate", "decay_exponent", "prior_weight"],
        "value": [fail_rate, decay_rate, decay_exponent, prior_weight],
        "std_error": se,
        "ci95_low": [v - c if np.isfinite(c) else np.nan for v, c in zip([fail_rate, decay_rate, decay_exponent, prior_weight], ci95)],
        "ci95_high": [v + c if np.isfinite(c) else np.nan for v, c in zip([fail_rate, decay_rate, decay_exponent, prior_weight], ci95)],
        "lower_bound": lower_bounds,
        "upper_bound": upper_bounds,
        "within_bounds": [
            (lower_bounds[i] <= params[i] <= upper_bounds[i]) for i in range(4)
        ],
    })

    # empirical vs fitted table
    emp_vs_fit = pd.DataFrame({
        attempt_column: t,
        "p_empirical": p,
        "p_hat": p_hat_emp,
        "residual": resid,
        "abs_residual": abs_resid,
        "sq_residual": sq_resid,
    }).sort_values(attempt_column)

    # fit metrics
    metrics = pd.DataFrame([{
        "n_points": int(t.size),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "sse": sse,
        "sst": sst,
        "mean_p": p_mean,
        "covariance_finite": bool(np.isfinite(se).any()),  # at least one finite SE
    }])

    # covariance matrix (wide)
    if isinstance(pcov, np.ndarray) and pcov.shape == (4, 4):
        cov_df = pd.DataFrame(
            pcov,
            index=["fail_rate", "decay_rate", "decay_exponent", "prior_weight"],
            columns=["fail_rate", "decay_rate", "decay_exponent", "prior_weight"],
        )
    else:
        cov_df = pd.DataFrame(columns=["fail_rate", "decay_rate", "decay_exponent", "prior_weight"])

    # --- audit dumps ------------------------------------------------------------
    if audit and audit_path is not None:
        base = audit_path / "learning_curve"
        audit_table(True, base / "02_fit_params", params_df, head=None)
        audit_table(True, base / "03_prior_alpha_beta",
                    pd.DataFrame([{"alpha": alpha, "beta": beta, "prior_weight": prior_weight, "fail_rate": fail_rate}]),
                    head=None)
        audit_table(True, base / "04_empirical_vs_fitted", emp_vs_fit, head=None)
        audit_table(True, base / "05_fit_metrics", metrics, head=None)
        audit_table(True, base / "06_tfit_curve",
                    pd.DataFrame({attempt_column: t_fit, "p_fit": p_fit}), head=None)
        audit_table(True, base / "07_param_covariance", cov_df, head=None)
        if audit_context:
            ctx = pd.DataFrame([audit_context])
            audit_table(True, base / "08_context", ctx, head=None)

    return t_fit, p_fit, alpha, beta, decay_rate, decay_exponent, prior_weight


def compute_empirical_conditionals(df, attempt_column, grouping_col):
    """
    Compute empirical has_loss rates for launches 1 to 3:
    - p1: unconditional has_loss rate at launch 1
    - p2_given: conditional has_loss rate at launch 2 given outcome of launch 1
    - p3_given: conditional has_loss rate at launch 3 given outcome pattern at launch 1 and 2
    """
    # Filter for launches after 2000 only
    # df = df[df['launch_date'] >= '2000-01-01'].copy()

    # Launch 1 has_loss rate
    p1 = df[df[attempt_column] == 1]['has_loss'].mean()

    # Build pairs for launch 2 conditioning
    l1 = df[df[attempt_column] == 1][[grouping_col, 'has_loss']]
    l2 = df[df[attempt_column] == 2][[grouping_col, 'has_loss']]
    pairs = pd.merge(l1, l2, on=grouping_col, suffixes=('_1', '_2'))

    p2_given = {
        'F': pairs[pairs['has_loss_1'] == 1]['has_loss_2'].mean(),
        'S': pairs[pairs['has_loss_1'] == 0]['has_loss_2'].mean()
    }

    # Build triplets for launch 3 conditioning
    l3 = df[df[attempt_column] == 3][[grouping_col, 'has_loss']]
    triples = pd.merge(pairs, l3, on=grouping_col)
    triples['fail_count'] = triples['has_loss_1'] + triples['has_loss_2']
    p3_given = {
        str(int(fail_count)): group['has_loss'].mean()
        for fail_count, group in triples.groupby('fail_count')
    }

    return p1, p2_given, p3_given

def compute_empirical_conditionals(
    df: pd.DataFrame,
    attempt_column: str,
    grouping_col: str,
    *,
    audit: bool = False,
    audit_path: Path | None = None,
):
    """
    Compute empirical has_loss rates for launches 1 to 3:
    - p1: unconditional has_loss rate at launch 1
    - p2_given: conditional has_loss rate at launch 2 given outcome of launch 1
    - p3_given: conditional has_loss rate at launch 3 given outcome pattern at launch 1 and 2

    If audit=True and audit_path is provided, writes the following tables:
      empirical/00_coverage.(csv|xlsx)
      empirical/01_l1.(csv|xlsx)
      empirical/02_l2.(csv|xlsx)
      empirical/03_pairs_l1_l2.(csv|xlsx)
      empirical/04_l3.(csv|xlsx)
      empirical/05_triples.(csv|xlsx)
      empirical/06_p1_summary.(csv|xlsx)
      empirical/07_p2_summary.(csv|xlsx)
      empirical/08_p3_summary.(csv|xlsx)
      empirical/09_reported_conditionals.(csv|xlsx)

    Depends on the audit helper: audit_table(...)
    """
    # --- core computation (unchanged logic) ---------------------------------
    # Launch 1 has_loss rate
    p1 = df[df[attempt_column] == 1]['has_loss'].mean()

    # Build pairs for launch 2 conditioning
    l1 = df[df[attempt_column] == 1][[grouping_col, 'has_loss']].rename(columns={'has_loss': 'has_loss_1'})
    l2 = df[df[attempt_column] == 2][[grouping_col, 'has_loss']].rename(columns={'has_loss': 'has_loss_2'})
    pairs = pd.merge(l1, l2, on=grouping_col, how="inner")

    p2_given = {
        'F': pairs[pairs['has_loss_1'] == 1]['has_loss_2'].mean(),
        'S': pairs[pairs['has_loss_1'] == 0]['has_loss_2'].mean()
    }

    # Build triplets for launch 3 conditioning
    l3 = df[df[attempt_column] == 3][[grouping_col, 'has_loss']].rename(columns={'has_loss': 'has_loss_3'})
    triples = pd.merge(pairs, l3, on=grouping_col, how="inner")
    triples['fail_count'] = triples['has_loss_1'] + triples['has_loss_2']
    p3_given = {
        str(int(fail_count)): group['has_loss_3'].mean()
        for fail_count, group in triples.groupby('fail_count')
    }

    # --- optional audit ------------------------------------------------------
    if audit and audit_path is not None:
        base = audit_path / "empirical"

        # Coverage / matching counts
        coverage = pd.DataFrame([{
            "grouping_col": grouping_col,
            "n_l1_total": int(len(l1)),
            "n_l2_total": int(len(l2)),
            "n_pairs_l1_l2": int(len(pairs)),
            "n_l3_total": int(len(l3)),
            "n_triples": int(len(triples)),
        }])

        # p1 summary
        p1_summary = pd.DataFrame([{
            "grouping_col": grouping_col,
            "p1": float(p1),
            "n_l1": int(len(l1)),
            "n_l1_fail": int(l1["has_loss_1"].sum()),
        }])

        # p2 branches
        def _p2_branch_stats(df_, cond_val):
            br = df_[df_["has_loss_1"] == cond_val]
            return pd.Series({
                "n_pairs": int(len(br)),
                "n_losses_at_l2": int(br["has_loss_2"].sum()),
                "p2_mean": float(br["has_loss_2"].mean()) if len(br) else np.nan,
            })

        p2_fail = _p2_branch_stats(pairs, 1)
        p2_succ = _p2_branch_stats(pairs, 0)
        p2_summary = pd.DataFrame([
            {"grouping_col": grouping_col, "branch": "L1=Fail", **p2_fail.to_dict()},
            {"grouping_col": grouping_col, "branch": "L1=Success", **p2_succ.to_dict()},
        ])

        # p3 by fail_count
        if len(triples):
            p3_summary = (triples.groupby("fail_count", as_index=False)["has_loss_3"]
                                 .agg(n_triplets="count",
                                      n_losses_at_l3="sum",
                                      p3_mean="mean"))
            p3_summary.insert(0, "grouping_col", grouping_col)
            p3_summary["fail_count"] = p3_summary["fail_count"].astype(int)
        else:
            p3_summary = pd.DataFrame(columns=["grouping_col","fail_count","n_triplets","n_losses_at_l3","p3_mean"])

        # Final reported values (for easy cross-check)
        reported = pd.DataFrame([
            {"metric": "p1", "value": float(p1)},
            {"metric": "p2_given[L1=Fail]", "value": float(p2_given.get("F", np.nan))},
            {"metric": "p2_given[L1=Success]", "value": float(p2_given.get("S", np.nan))},
            {"metric": "p3_given[fail_count=0]", "value": float(p3_given.get("0", np.nan))},
            {"metric": "p3_given[fail_count=1]", "value": float(p3_given.get("1", np.nan))},
            {"metric": "p3_given[fail_count=2]", "value": float(p3_given.get("2", np.nan))},
        ])
        reported.insert(0, "grouping_col", grouping_col)

        # write tables (full, no head limits)
        audit_table(True, base / "00_coverage", coverage, head=None)
        audit_table(True, base / "01_l1", l1, head=None)
        audit_table(True, base / "02_l2", l2, head=None)
        audit_table(True, base / "03_pairs_l1_l2", pairs, head=None)
        audit_table(True, base / "04_l3", l3, head=None)
        audit_table(True, base / "05_triples", triples, head=None)

        audit_table(True, base / "06_p1_summary", p1_summary, head=None)
        audit_table(True, base / "07_p2_summary", p2_summary, head=None)
        audit_table(True, base / "08_p3_summary", p3_summary, head=None)
        audit_table(True, base / "09_reported_conditionals", reported, head=None)

    return p1, p2_given, p3_given

def empirical_prior(t, outcome_history, p2_given, p3_given, default_weight=10):
    """
    Return alpha, beta prior based on empirical estimates for t=1 to 3, or None for t>=4.

    Args:
        t: Current launch number
        outcome_history: list of prior outcomes (1 = has_loss, 0 = success)
        p1, p2_given, p3_given: from compute_empirical_conditionals
        default_weight: strength of empirical prior

    Returns:
        (alpha, beta) or None
    """
    """ if t == 1:
        p = p1 """
    """ el """
    if t == 2 and len(outcome_history) >= 1:
        prev = outcome_history[0]
        p = p2_given['F' if prev == 1 else 'S']
    elif t == 3 and len(outcome_history) >= 2:
        """ key = ''.join(['F' if o == 1 else 'S' for o in outcome_history[:2]]) """
        key = sum(outcome_history)
        p = p3_given[str(key)]  # fallback to neutral prior if unseen pattern
    else:
        return None  # use learning curve from here

    alpha = default_weight * p
    beta = default_weight * (1 - p)
    return alpha, beta

def has_loss_prior(t, alpha0, beta0, lambda_, delta):
     
    return alpha0 / (alpha0 + beta0 + (lambda_ * (t+8)) ** delta)

def has_loss_prior(t, alpha0, beta0, lambda_, delta):
    t = max(1, t)
    return alpha0 / (alpha0 + beta0 + (lambda_ * (t - 1)) ** delta)

def apply_exponential_decay(outcomes, current_launch, decay_tau):
    """
    Applies an exponential decay to the weights of outcomes based on their age.

    Parameters:
    outcomes (list): A list of binary outcomes (1 for has_loss, 0 for success).
    current_launch (int): The current launch number.
    decay_tau (float): The decay constant for the exponential function.

    Returns:
    tuple: A tuple containing the total has_loss weight and success weight.
    """
    fail_weight = 0.0
    succ_weight = 0.0
    for j, outcome in enumerate(outcomes):
        past_launch = j + 1
        if past_launch <= 20:
            age_since_20 = max(0, current_launch - 20)
            decay_age = age_since_20 + (20 - past_launch)
            weight = np.exp(-decay_age / decay_tau)
        else:
            weight = 1.0
        if outcome == 1:
            fail_weight += weight
        else:
            succ_weight += weight
    return fail_weight, succ_weight

def failure_prior(t, alpha0, beta0, lambda_, delta):
    t = max(1, t)
    return alpha0 / (alpha0 + beta0 + (lambda_ * (t - 1)) ** delta)

class _LaunchBayesPredictor:
    """
    Shared machinery for:
      • curve shape g(t), step factor s_t, early-weight w_early(t),
      • empirical priors for t=1,2,3 (using p1, p2_given, p3_given),
      • buckets for early / late / base / curve,
      • advancing across gaps with the curve nudge,
      • composing prior counts at a given t.
    """

    def __init__(self, lam: float, delt: float, default_weight: float,
                 p1_global: float, p2_given: Dict[str, float], p3_given: Dict[str, float],
                 *, audit: bool = False, audit_events: list | None = None) -> None:
        self.lam = float(lam)
        self.delt = float(delt)
        self.w0 = float(default_weight)
        self.p1 = float(p1_global)
        self.p2_given = p2_given
        self.p3_given = p3_given
        self.audit = bool(audit)
        self._events = audit_events if audit_events is not None else ( [] if audit else None )
        self._seq = 0

    # ---------- small logger ----------
    def _log(self, **kv):
        if not self.audit or self._events is None:
            return
        self._seq += 1
        row = {"seq": self._seq}
        row.update({k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                    for k, v in kv.items()})
        self._events.append(row)

    # ---------- curve & schedules ----------
    def g(self, t: int) -> float:
        return 1.0 / (1.0 + (self.lam * max(t - 1, 0)) ** self.delt)

    def step_factor(self, t: int) -> float:
        if t <= 1:
            return 1.0
        gt = self.g(t)
        gtm1 = self.g(t - 1)
        return float(max(0.0, min(1.0, gt / max(gtm1, 1e-12))))

    def w_early(self, t: int) -> float:
        # 1.0 through t=40, 0.0 by t=60, with curve-shaped fade.
        if t <= 40:
            return 1.0
        if t >= 60:
            return 0.0
        g40, g60 = self.g(40), self.g(60)
        denom = g40 - g60
        if abs(denom) < 1e-12:
            return (60.0 - float(t)) / 20.0  # linear fallback
        return float(max(0.0, min(1.0, (self.g(t) - g60) / denom)))

    # ---------- empirical priors t=1,2,3 ----------
    def prior_empirical_t1(self) -> tuple[float, float]:
        p = self.p1
        a, b = self.w0 * p, self.w0 * (1 - p)
        self._log(action="empirical_prior_t1", p=p, a=a, b=b)
        return a, b

    def prior_empirical_t2(self, launches: np.ndarray, outcomes: np.ndarray) -> tuple[float, float]:
        if (launches == 1).any():
            prev = int(outcomes[launches == 1][0])
            key = "F" if prev == 1 else "S"
            p = float(self.p2_given[key])
            src = f"p2_given[{key}]"
        else:
            p = self.p1 * float(self.p2_given["F"]) + (1 - self.p1) * float(self.p2_given["S"])
            src = "marginalized_from_p1,p2"
        a, b = self.w0 * p, self.w0 * (1 - p)
        self._log(action="empirical_prior_t2", p=p, a=a, b=b, source=src)
        return a, b

    def empirical_p3_anchor(self, launches: np.ndarray, outcomes: np.ndarray) -> float:
        has_t1 = (launches == 1).any()
        has_t2 = (launches == 2).any()
        if has_t1 and has_t2:
            key = str(int(outcomes[launches == 1][0] + outcomes[launches == 2][0]))
            return float(self.p3_given.get(key, self.p1))
        if has_t1 and not has_t2:
            first = int(outcomes[launches == 1][0])
            p2 = float(self.p2_given["F" if first == 1 else "S"])
            if first == 1:
                return (1 - p2) * float(self.p3_given.get("1", self.p1)) + p2 * float(self.p3_given.get("2", self.p1))
            else:
                return (1 - p2) * float(self.p3_given.get("0", self.p1)) + p2 * float(self.p3_given.get("1", self.p1))
        return float(self.p1)

    def prior_empirical_t3(self, launches: np.ndarray, outcomes: np.ndarray) -> tuple[float, float]:
        p = self.empirical_p3_anchor(launches, outcomes)
        a, b = self.w0 * p, self.w0 * (1 - p)
        self._log(action="empirical_prior_t3", p=p, a=a, b=b)
        return a, b

    # ---------- buckets & advancing ----------
    @staticmethod
    def _init_buckets() -> Dict[str, float]:
        # early (t<=20), late (t>=21), base anchor at t=3, curve virtual successes (b only)
        return dict(a_early=0.0, b_early=0.0, a_late=0.0, b_late=0.0, a_base=0.0, b_base=0.0, b_curve=0.0)

    @staticmethod
    def _posterior_update(buckets: Dict[str, float], t_obs: int, y: int) -> None:
        if t_obs <= 20:
            buckets["a_early"] += y
            buckets["b_early"] += (1 - y)
        else:
            buckets["a_late"] += y
            buckets["b_late"] += (1 - y)

    def _compose_prior_counts(self, buckets: Dict[str, float], t: int) -> tuple[float, float]:
        w = self.w_early(t)
        a = w * buckets["a_early"] + buckets["a_late"] + buckets["a_base"]
        b = w * buckets["b_early"] + buckets["b_late"] + buckets["b_base"] + buckets["b_curve"]
        return a, b

    def _curve_nudge_once(self, buckets: Dict[str, float], t: int) -> None:
        # Add ONLY virtual successes so that mean decreases by step factor.
        a_raw, b_raw = self._compose_prior_counts(buckets, t)
        tot = a_raw + b_raw
        if tot <= 0:
            self._log(action="advance_curve", t=t, note="no_totals")
            return
        mu_prev = a_raw / tot
        s_t = self.step_factor(t)
        mu_target = min(mu_prev, mu_prev * s_t)
        # delta_b = a*(1/mu_target - 1) - b
        delta_b = a_raw * (1.0 / max(mu_target, 1e-12) - 1.0) - b_raw
        if delta_b > 1e-12:
            buckets["b_curve"] += delta_b
        a_new, b_new = self._compose_prior_counts(buckets, t)
        self._log(action="advance_curve", t=t, w_early=self.w_early(t),
                  g=self.g(t), step_factor=s_t, mu_prev=mu_prev, mu_target=mu_target,
                  delta_b=max(delta_b, 0.0), a=a_new, b=b_new)

    def advance_to(self, buckets: Dict[str, float], last_t: int, target_t: int) -> int:
        """
        Apply the curve-improvement step for each integer t in (last_t, target_t], starting at t>=4.
        No observations are added here.
        """
        start = max(last_t + 1, 4)
        if target_t >= start:
            for tt in range(start, target_t + 1):
                self._curve_nudge_once(buckets, tt)
            return target_t
        return last_t

from typing import Any, Dict
from pathlib import Path
import numpy as np
import pandas as pd

def predict_has_loss_probabilities_all_launches_with_empirical(
    df: pd.DataFrame,
    lambda_set: Dict[int, float],
    delta_set: Dict[int, float],
    p2_given: Dict[str, float],
    p3_given: Dict[str, float],
    alpha0_set: Dict[int, float],  # unused (kept for signature compatibility)
    beta0_set: Dict[int, float],   # unused (kept for signature compatibility)
    default_weight: float = 10,
    ci: float = 0.95,
    decay_tau: float = 10,  # unused
    grouping_col: str = "vehicle_type",
    attempt_column: str = "lv_family_attempt_number",
    *,
    audit: bool = False,
    audit_path: Path | None = None,
) -> pd.DataFrame:
    """
    For each observed launch t_obs, record the PRIOR@t_obs computed as:
      • t=1..3: purely empirical (p1, p2|y1, p3|y1,y2).
      • t>=4: take the running POSTERIOR@(t_obs-1) (which started at PRIOR@3 anchor,
              only updated with t>=3 outcomes), advance via curve from (t_obs-1) to t_obs,
              then compose PRIOR@t_obs.

    After recording PRIOR@t_obs, we POSTERIOR-update with the observed y_t_obs and
    carry that forward. This matches “posterior from previous launch × learning-curve delta”.
    """
    required = {grouping_col, attempt_column, "has_loss", "launch_date", "seradata_spacecraft_id"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    p1_global = df[df[attempt_column] == 1]["has_loss"].mean()
    if pd.isna(p1_global):
        p1_global = df["has_loss"].mean()
    lam = float(lambda_set.get(0))
    delt = float(delta_set.get(0))

    rows: list[Dict[str, Any]] = []
    base_dir = (audit_path / "per_launch") if (audit and audit_path is not None) else None

    for vehicle in df[grouping_col].dropna().unique():
        sub = (df.loc[df[grouping_col] == vehicle]
                 .sort_values([attempt_column, "launch_date"], na_position="last"))
        if sub.empty:
            continue

        launches = sub[attempt_column].to_numpy().astype(int)
        outcomes = sub["has_loss"].to_numpy().astype(int)
        dates    = sub["launch_date"].to_numpy()
        sc_ids   = sub["seradata_spacecraft_id"].to_numpy()

        # running empirical stats (just for reporting)
        cum_launches = 0
        cum_failures = 0

        # predictor with audit
        vdir = (base_dir / _safe_name(str(vehicle))) if base_dir is not None else None
        evts: list[Dict[str, Any]] = []
        pred = _LaunchBayesPredictor(lam, delt, default_weight, float(p1_global), p2_given, p3_given,
                                     audit=audit, audit_events=evts)

        # buckets/state for t>=3
        buckets = pred._init_buckets()
        base_set = False
        last_t = 0  # last t reached by curve (or 3 after anchoring)

        # put sequence into audit file
        if audit and vdir is not None:
            seq = pd.DataFrame({"attempt": launches, "has_loss": outcomes, "launch_date": dates})
            audit_table(True, vdir / "01_sequence", seq, head=None)

        for i, t_obs in enumerate(launches):
            t_obs = int(t_obs)

            # ----- PRIOR @ t_obs -----
            if t_obs == 1:
                a_prior, b_prior = pred.prior_empirical_t1()
                if audit: pred._log(action="prior_at_t", t=1, a=a_prior, b=b_prior,
                                    p=a_prior/(a_prior+b_prior))
            elif t_obs == 2:
                a_prior, b_prior = pred.prior_empirical_t2(launches, outcomes)
                if audit: pred._log(action="prior_at_t", t=2, a=a_prior, b=b_prior,
                                    p=a_prior/(a_prior+b_prior))
            elif t_obs == 3:
                # anchor PRIOR@3
                if not base_set:
                    p3 = pred.empirical_p3_anchor(launches, outcomes)
                    buckets["a_base"] = default_weight * p3
                    buckets["b_base"] = default_weight * (1 - p3)
                    base_set = True
                    if audit: pred._log(action="set_anchor_t3", t=3, p_anchor=p3,
                                        a_base=buckets["a_base"], b_base=buckets["b_base"])
                    last_t = 3
                # no curve before 4 → PRIOR@3 = composed base (+ any prior t>=3 obs; none yet)
                a_prior, b_prior = pred._compose_prior_counts(buckets, 3)
                if audit: pred._log(action="prior_at_t", t=3, a=a_prior, b=b_prior,
                                    p=a_prior/(a_prior+b_prior) if (a_prior+b_prior)>0 else np.nan)
            else:
                # ensure anchor exists
                if not base_set:
                    p3 = pred.empirical_p3_anchor(launches, outcomes)
                    buckets["a_base"] = default_weight * p3
                    buckets["b_base"] = default_weight * (1 - p3)
                    base_set = True
                    if audit: pred._log(action="set_anchor_t3", t=3, p_anchor=p3,
                                        a_base=buckets["a_base"], b_base=buckets["b_base"])
                    last_t = 3

                # --- NEW: advance POSTERIOR@(last_t) → PRIOR@t_obs (curve step happens here)
                last_t = pred.advance_to(buckets, last_t, t_obs)

                # --- NEW: log PRIOR@t right after the advance, before observe
                a_prior, b_prior = pred._compose_prior_counts(buckets, t_obs)
                if audit: pred._log(action="prior_at_t", t=t_obs, a=a_prior, b=b_prior,
                                    p=a_prior/(a_prior+b_prior) if (a_prior+b_prior)>0 else np.nan)

            # PRIOR metrics
            p_mean  = a_prior / (a_prior + b_prior) if (a_prior + b_prior) > 0 else np.nan
            if a_prior > 0 and b_prior > 0:
                p_lower = _beta.ppf((1 - ci) / 2, a_prior, b_prior)
                p_upper = _beta.ppf(1 - (1 - ci) / 2, a_prior, b_prior)
            else:
                p_lower = np.nan
                p_upper = np.nan

            # running empirical
            outcome_i = int(outcomes[i])
            cum_launches += 1
            cum_failures += outcome_i
            cumulative_rate = cum_failures / cum_launches

            rows.append({
                grouping_col: vehicle,
                "launch_number": int(t_obs),
                "launch_date": dates[i],
                "observed_has_loss": outcome_i,
                "cumulative_rate": float(cumulative_rate),
                "predicted": float(p_mean),   # PRIOR at t_obs, i.e., (posterior from t-1) advanced by curve
                "ci_lower": float(p_lower),
                "ci_upper": float(p_upper),
                "spacecraft_id": sc_ids[i],
            })

            # ----- POSTERIOR update (carry forward) -----
            if t_obs >= 3:
                prev_a, prev_b = a_prior, b_prior  # already composed at t_obs
                pred._posterior_update(buckets, t_obs, outcome_i)
                post_a, post_b = pred._compose_prior_counts(buckets, t_obs)
                if audit: pred._log(action="observe", t=t_obs, y=outcome_i,
                                    a_before=prev_a, b_before=prev_b,
                                    a_after=post_a, b_after=post_b,
                                    p_after=post_a/(post_a+post_b) if (post_a+post_b)>0 else np.nan)
                last_t = t_obs
            # t=1,2 NOT added to buckets (avoid double-counting)

            # t=1,2 observations are intentionally NOT added to buckets (avoid double-counting)

        # write per-vehicle audit
        if audit and vdir is not None:
            evtdf = pd.DataFrame(evts) if len(evts) else pd.DataFrame()
            if "seq" in evtdf.columns:
                evtdf = evtdf.sort_values(["seq"])
            audit_table(True, vdir / "02_event_log", evtdf, head=None)

    return pd.DataFrame(rows)


def predict_next_failure_probability_per_vehicle(
    df: pd.DataFrame,
    lambda_set: Dict[int, float],
    delta_set: Dict[int, float],
    p2_given: Dict[str, float],
    p3_given: Dict[str, float],
    alpha0_set: Dict[int, float],  # unused (kept for signature compatibility)
    beta0_set: Dict[int, float],   # unused (kept for signature compatibility)
    default_weight: float = 10,
    ci: float = 0.95,
    decay_tau: float = 10,  # unused
    grouping_col: str = "vehicle_type",
    attempt_column: str = "lv_family_attempt_number",
    *,
    audit: bool = False,
    audit_path: Path | None = None,
) -> pd.DataFrame:
    """
    One-step-ahead per vehicle:
      • next∈{1,2,3}: purely empirical.
      • next≥4: PRIOR@3 = empirical anchor; update with y3 if present (ONLY t≥3);
                then advance via curve to 'next' and read PRIOR@next.
    """
    required = {grouping_col, attempt_column, "has_loss", "launch_date"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    p1_global = df[df[attempt_column] == 1]["has_loss"].mean()
    if pd.isna(p1_global):
        p1_global = df["has_loss"].mean()
    lam = float(lambda_set.get(0))
    delt = float(delta_set.get(0))

    results: list[Dict[str, Any]] = []
    base_dir = (audit_path / "next_failure") if (audit and audit_path is not None) else None

    for vehicle in df[grouping_col].dropna().unique():
        sub = df[df[grouping_col] == vehicle].sort_values([attempt_column, "launch_date"], na_position="last")
        if sub.empty:
            continue

        launches = sub[attempt_column].to_numpy().astype(int)
        outcomes = sub["has_loss"].to_numpy().astype(int)
        dates    = sub["launch_date"].to_numpy()
        next_launch = (int(launches.max()) + 1) if len(launches) else 1

        # per-vehicle audit
        vdir = (base_dir / _safe_name(str(vehicle))) if base_dir is not None else None
        evts: list[Dict[str, Any]] = []
        pred = _LaunchBayesPredictor(lam, delt, default_weight, float(p1_global), p2_given, p3_given,
                                     audit=audit, audit_events=evts)

        if audit and vdir is not None:
            meta = pd.DataFrame([{
                "vehicle": vehicle, "grouping_col": grouping_col, "p1_global": float(p1_global),
                "lambda": lam, "delta": delt, "default_weight": float(default_weight),
                "ci": float(ci), "next_launch": int(next_launch),
                "n_obs": int(len(launches)), "total_failures": int(np.nansum(outcomes)),
                "p2_given_F": float(p2_given.get("F", np.nan)),
                "p2_given_S": float(p2_given.get("S", np.nan)),
                "p3_given_0": float(p3_given.get("0", np.nan)),
                "p3_given_1": float(p3_given.get("1", np.nan)),
                "p3_given_2": float(p3_given.get("2", np.nan)),
            }])
            audit_table(True, vdir / "00_metadata", meta, head=None)
            seq = pd.DataFrame({"attempt": launches, "has_loss": outcomes, "launch_date": dates})
            audit_table(True, vdir / "01_sequence", seq, head=None)

        if next_launch == 1:
            a_next, b_next = pred.prior_empirical_t1()
            if audit: pred._log(
                action="prediction_next_launch_1", t=1, a=a_next, 
                b=b_next, p=a_next/(a_next+b_next) if (a_next+b_next)>0 else np.nan
                )
        elif next_launch == 2:
            a_next, b_next = pred.prior_empirical_t2(launches, outcomes)
            if audit: pred._log(action="prediction_next_launch_2", t=2, a=a_next, b=b_next, p=a_next/(a_next+b_next) if (a_next+b_next)>0 else np.nan)
        elif next_launch == 3:
            a_next, b_next = pred.prior_empirical_t3(launches, outcomes)
            if audit: pred._log(action="prediction_next_launch_3", t=3, a=a_next, b=b_next, p=a_next/(a_next+b_next) if (a_next+b_next)>0 else np.nan)
        else:
            # Anchor at t=3
            buckets = pred._init_buckets()
            p3 = pred.empirical_p3_anchor(launches, outcomes)
            buckets["a_base"] = default_weight * p3
            buckets["b_base"] = default_weight * (1 - p3)
            if audit:
                pred._log(action="set_anchor_t3", t=3, p_anchor=p3,
                        a=buckets["a_base"], b=buckets["b_base"])
            last_t = 3

            # Include ONLY t≥3 observations — correct order: advance FIRST, then observe
            for i, t_obs in enumerate(launches):
                t_obs = int(t_obs)
                if t_obs >= 3:
                    # 1) move posterior@(last_t) -> PRIOR@t_obs via curve
                    last_t = pred.advance_to(buckets, last_t, t_obs)

                    # (optional logging of PRIOR@t_obs)
                    if audit:
                        a_prior, b_prior = pred._compose_prior_counts(buckets, t_obs)
                        pred._log(action="prior_at_t", t=t_obs,
                                a=a_prior, b=b_prior,
                                p=a_prior/(a_prior+b_prior) if (a_prior+b_prior)>0 else np.nan)

                    # 2) observe y_t to form POSTERIOR@t_obs
                    prev_a, prev_b = pred._compose_prior_counts(buckets, t_obs)
                    pred._posterior_update(buckets, t_obs, int(outcomes[i]))
                    post_a, post_b = pred._compose_prior_counts(buckets, t_obs)
                    if audit:
                        pred._log(action="observe", t=t_obs, y=int(outcomes[i]),
                                a_before=prev_a, b_before=prev_b,
                                a_after=post_a, b_after=post_b,
                                p_after=post_a/(post_a+post_b) if (post_a+post_b)>0 else np.nan)

                    last_t = t_obs  # keep in sync
                    # (no curve at this same t again)

            # Finally: advance POSTERIOR@last_t -> PRIOR@next
            last_t = pred.advance_to(buckets, last_t, int(next_launch))
            a_next, b_next = pred._compose_prior_counts(buckets, int(next_launch))
            if audit:
                pred._log(action="compose_prior_at_next", t=int(next_launch), a=a_next, b=b_next,
                        p=a_next/(a_next+b_next) if (a_next+b_next)>0 else np.nan)


        p_mean  = a_next / (a_next + b_next) if (a_next + b_next) > 0 else np.nan
        p_lower = _beta.ppf((1 - ci) / 2, a_next, b_next) if (a_next > 0 and b_next > 0) else np.nan
        p_upper = _beta.ppf(1 - (1 - ci) / 2, a_next, b_next) if (a_next > 0 and b_next > 0) else np.nan

        results.append({
            grouping_col: vehicle,
            "next_launch_number": int(next_launch),
            "launch_date": dates[-1] if len(dates) else pd.NaT,
            "total_failures": int(np.nansum(outcomes)),
            "failure_rate": float(p_mean),
            "ci_lower": float(p_lower),
            "ci_upper": float(p_upper),
        })

        if audit and vdir is not None:
            evtdf = pd.DataFrame(evts) if len(evts) else pd.DataFrame()
            if "seq" in evtdf.columns:
                evtdf = evtdf.sort_values(["seq"])
            audit_table(True, vdir / "02_event_log", evtdf, head=None)
            prior_next = pd.DataFrame([{
                "next_launch": int(next_launch), "a_next": float(a_next), "b_next": float(b_next),
                "p_mean": float(p_mean), "ci_lower": float(p_lower), "ci_upper": float(p_upper),
            }])
            audit_table(True, vdir / "03_prior_next", prior_next, head=None)

    return pd.DataFrame(results)




def create_dropdown_options(df: pd.DataFrame, grouping_col: str) -> pd.DataFrame:
    """
    Create dropdown options for a given grouping column.
    Returns a DataFrame with columns [grouping_col, value].
    """
    if grouping_col not in df.columns:
        raise KeyError(f"Column '{grouping_col}' not found in DataFrame.")

    pretty_name = _prettify_grouping_label(grouping_col)

    unique_values = df[grouping_col].dropna().unique()
    values = sorted(map(str, unique_values))
    options = [{"grouping_col": pretty_name, "value": v} for v in values]
    return pd.DataFrame(options)




def load_specific_primary_rates_long(output_dir: Path, date_tag: str) -> pd.DataFrame:
    """
    Build one long table with columns:
    ['grouping_col','value', *_specific]
    by concatenating the five per-grouping primary-rate files.
    """
    frames: list[pd.DataFrame] = []
    for grouping_label, (file_key, id_col) in SPECIFIC_GROUPINGS.items():
        path = output_dir / f"launch_primary_rates_{file_key}_{date_tag}_next.csv"
        if not path.exists():
            logger.warning("Specific rates file not found for %s: %s", grouping_label, path)
            continue
        tmp = pd.read_csv(path)
        if id_col not in tmp.columns:
            logger.warning("Expected id column '%s' missing in %s; skipping.", id_col, path)
            continue
        tmp = _rename_rate_cols(tmp, "_specific")
        tmp = (
            tmp.rename(columns={id_col: "value"})
               .assign(grouping_col=grouping_label)
               [ ["grouping_col", "value"] + SPECIFIC_COLS ]
        )
        frames.append(tmp)

    if frames:
        return pd.concat(frames, ignore_index=True)
    # Empty but with schema, so downstream merge is stable
    return pd.DataFrame(columns=["grouping_col", "value"] + SPECIFIC_COLS)


def load_type_primary_rates(output_dir: Path, date_tag: str) -> pd.DataFrame:
    """
    Load the vehicle_type primary rates and suffix columns with _type.
    Returns columns: ['vehicle_type', *_type]
    """
    path = output_dir / f"launch_primary_rates_vehicle_type_{date_tag}_next.csv"
    if not path.exists():
        logger.warning("Type rates file not found: %s", path)
        return pd.DataFrame(columns=["vehicle_type"] + TYPE_COLS)

    df = pd.read_csv(path)
    if "vehicle_type" not in df.columns:
        logger.warning("Expected 'vehicle_type' column missing in %s; continuing with empty.", path)
        return pd.DataFrame(columns=["vehicle_type"] + TYPE_COLS)

    df = _rename_rate_cols(df, "_type")
    return df[ ["vehicle_type"] + TYPE_COLS ]

def coerce_dropdown_schema(df: pd.DataFrame) -> pd.DataFrame:
    # columns we expect by type
    INT_COLS = [
        "lv_type_attempt_number", "lv_family_attempt_number", "lv_provider_attempt_number",
        "lv_variant_attempt_number", "lv_minor_variant_attempt_number",
        "total_failures", "total_failures_specific", "total_failures_type",
        "next_launch_number_specific", "next_launch_number_type",
    ]
    FLOAT_COLS = [
        "failure_rate_specific", "ci_lower_specific", "ci_upper_specific",
        "failure_rate_type", "ci_lower_type", "ci_upper_type",
    ]
    STR_COLS = [
        "grouping_col", "value", "launch_provider", "vehicle_family",
        "vehicle_type", "vehicle_variant", "vehicle_minor_variant",
    ]

    # ensure all columns exist so casting is stable
    for c in INT_COLS + FLOAT_COLS + STR_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    # cast/fill
    for c in INT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")
    for c in FLOAT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype("float64")
    for c in STR_COLS:
        df[c] = df[c].fillna("").astype(str)

    return df


# --- MAIN ------------------------------------------------------------------
def main(audit_outputs: bool = False) -> None:
    # Create a stable, timezone-aware date tag for filenames
    date_tag = datetime.now(tz=TZ).strftime("%Y%m%d")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR = OUTPUT_DIR / f"audit_{date_tag}"
    if audit_outputs:
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading and preparing data...")
    launch_df = load_and_prepare_data(filter_min_year_on=False)
    
    print(
        launch_df.loc[
            launch_df["vehicle_type"].str.casefold().eq("ZHUQUE-2/SUZAKU-2 (ZQ-2)"),
            ["vehicle_type", "type_launches_since_last_failure"]
        ]
    )
    
    all_dropdown_rows: list[pd.DataFrame] = []

    for grouping_col in GROUPINGS:
        grp_dir = (AUDIT_DIR / grouping_col) if audit_outputs else None
        # 1) Extract data for modeling/prediction
        logger.info("=== Processing grouping: %s ===", grouping_col)
        clean_launch_data, attempt_column, selected_columns = select_grouping_columns(
            launch_df, grouping_col
        )

        # 2) Empirical conditionals
        first_launch_fr, second_launch_fr, third_launch_fr = compute_empirical_conditionals(
            launch_df,
            attempt_column=attempt_column,
            grouping_col=grouping_col,
            audit=audit_outputs,
            audit_path=grp_dir,
        )

        logger.info(
            "Empirical conditionals for %s: p1=%f, p2_given=%s, p3_given=%s",
            grouping_col, first_launch_fr, second_launch_fr, third_launch_fr
        )

        # 3) Fit the learning curve (if >=3 attempts)
        logger.info("Fitting learning curve (All losses) for %s...", grouping_col)
        if (clean_launch_data[attempt_column] >= 3).any():
            _t_fit, _p_fit, alpha_all, beta_all, lambda_all, delta_all, _prior_w = fit_learning_curve(
                clean_launch_data,
                attempt_column,
                audit=audit_outputs,
                audit_path=grp_dir,
                audit_context={"grouping_col": grouping_col, "attempt_column": attempt_column},
            )
        else:
            logger.warning("Not enough attempts (>=3) in %s to fit learning curve; using NaNs.", grouping_col)
            alpha_all = np.nan
            beta_all = np.nan
            lambda_all = np.nan
            delta_all = np.nan

        # Use "All" param set for buckets
        alpha0_set = {0: alpha_all, 1: alpha_all, 2: alpha_all}
        beta0_set  = {0: beta_all,  1: beta_all,  2: beta_all}
        lambda_set = {0: lambda_all, 1: lambda_all, 2: lambda_all}
        delta_set  = {0: delta_all,  1: delta_all,  2: delta_all}

        primary_rates = predict_next_failure_probability_per_vehicle(
            df=clean_launch_data,
            alpha0_set=alpha0_set, beta0_set=beta0_set,
            lambda_set=lambda_set, delta_set=delta_set,
            p2_given=second_launch_fr, p3_given=third_launch_fr,
            ci=0.95, grouping_col=grouping_col, attempt_column=attempt_column,
            audit=audit_outputs, audit_path=grp_dir,
        )

        full_history_df = predict_has_loss_probabilities_all_launches_with_empirical(
            clean_launch_data,
            lambda_set=lambda_set, delta_set=delta_set,
            p2_given=second_launch_fr, p3_given=third_launch_fr,
            alpha0_set=alpha0_set, beta0_set=beta0_set,
            grouping_col=grouping_col, attempt_column=attempt_column,
            audit=True, audit_path=grp_dir,
        )

        # full dropdown columns for grouping
        dropdown_rows_df = build_dropdown_rows_for_grouping(launch_df, grouping_col)
        all_dropdown_rows.append(dropdown_rows_df)

        # --- per-grouping exports (unchanged) ---
        prefix = f"launch_primary_rates_{grouping_col}_{date_tag}"
        next_launch_path = OUTPUT_DIR / f"{prefix}_next.csv"
        full_history_path = OUTPUT_DIR / f"{prefix}_full.csv"

        primary_rates.to_csv(next_launch_path, index=False)
        full_history_df.to_csv(full_history_path, index=False)

        next_launch_json_path = next_launch_path.with_suffix(".json")
        full_history_json_path = full_history_path.with_suffix(".json")

        primary_rates.to_json(next_launch_json_path, orient="records", indent=2)
        full_history_df.to_json(full_history_json_path, orient="records", indent=2)

        logger.info(
            "Saved CSVs & JSONs for %s:\n- %s\n- %s\n- %s\n- %s",
            grouping_col,
            next_launch_path,
            full_history_path,
            next_launch_json_path,
            full_history_json_path,
        )

    # --- Unified dropdown export (CSV + JSON) -------------------------------
    if all_dropdown_rows:
        all_dropdowns_df = pd.concat(all_dropdown_rows, ignore_index=True)
        # Deduplicate for stability (some groups may alias the same label/value)
        all_dropdowns_df = (
            all_dropdowns_df.drop_duplicates()
            .sort_values(["grouping_col", "value"], kind="mergesort")
            .reset_index(drop=True)
        )
    else:
        # Include all possible columns so CSV schema is predictable
        all_cols = ["grouping_col", "value"] + list({c for v in INCLUDE_BY_GROUPING.values() for c in v}) + ATTEMPT_COLS + ["total_failures"]
        all_dropdowns_df = pd.DataFrame(columns=all_cols)

        # ---- Attach primary rates: _specific (by grouping/value) and _type (by vehicle_type)
    specific_rates_long = load_specific_primary_rates_long(OUTPUT_DIR, date_tag)
    all_dropdowns_df = all_dropdowns_df.merge(
        specific_rates_long,
        on=["grouping_col", "value"],
        how="left",
        validate="m:1",
    )

    type_rates = load_type_primary_rates(OUTPUT_DIR, date_tag)
    # Merge on vehicle_type for everyone, regardless of grouping
    all_dropdowns_df = all_dropdowns_df.merge(
        type_rates,
        on="vehicle_type",
        how="left",
        validate="m:1",
    )

        # After merges:
    all_dropdowns_df = coerce_dropdown_schema(all_dropdowns_df)

    unified_path_csv = OUTPUT_DIR / f"launch_dropdown_options_all_{date_tag}.csv"
    all_dropdowns_df.to_csv(unified_path_csv, index=False)

    unified_path_json = unified_path_csv.with_suffix(".json")
    all_dropdowns_df.to_json(unified_path_json, orient="records", indent=2)

    logger.info("Saved unified dropdown options CSV & JSON:\n- %s\n- %s", unified_path_csv, unified_path_json)

# def main(audit_outputs: bool = False) -> None:
#     # Create a stable, timezone-aware date tag for filenames
#     date_tag = datetime.now(tz=TZ).strftime("%Y%m%d")
#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
#     AUDIT_DIR = OUTPUT_DIR / f"audit_{date_tag}"
#     if audit_outputs:
#         AUDIT_DIR.mkdir(parents=True, exist_ok=True)

#     logger.info("Loading and preparing data...")
#     launch_df = load_and_prepare_data(filter_min_year_on=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-outputs", action="store_true", help="Write intermediate audit artifacts")
    args = parser.parse_args()
    main(audit_outputs=args.audit_outputs)
