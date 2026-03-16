"""
Primary base rates production pipeline with optional audit trail.

Fits Bayesian learning curves per grouping level, predicts per-vehicle failure
probabilities (next-launch and full-history), builds dropdown tables, and
exports CSV/JSON outputs.
"""
from __future__ import annotations

import argparse
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import beta as _beta

from .data_load_feature_creation import load_and_prepare_data
from ..utils.grouping_column import select_grouping_columns
from ..utils.compute_empirical_cumulative_loss import compute_empirical_cumulative_has_loss
from ..utils.build_dropdown_columns import build_dropdown_rows_for_grouping, _prettify_grouping_label
from ..utils.constants import (
    GROUPINGS, OUTPUT_DIR, TZ, IDENTITY_COLS, ATTEMPT_COLS, INCLUDE_BY_GROUPING,
    SPECIFIC_GROUPINGS, SPECIFIC_COLS, TYPE_COLS,
)
from ..utils.audit_helpers import (
    _ensure_parent, _to_native, _normalize_table, _with_ext,
    audit_table, audit_schema, _safe_name,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column renaming helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Bayesian learning curve
# ---------------------------------------------------------------------------

def bayesian_learning_curve(t, fail_rate, decay_rate, decay_exponent, prior_weight):
    """
    Generalized Bayesian learning curve.

    Predicts has_loss probability after ``t`` launches.

    Parameters
    ----------
    fail_rate : float – initial belief of has_loss probability
    decay_rate : float – how quickly learning occurs (lambda)
    decay_exponent : float – shape of learning curve (delta)
    prior_weight : float – confidence in prior (pseudo-observations)
    """
    alpha_prior = prior_weight * fail_rate
    beta_prior = prior_weight * (1 - fail_rate)
    denominator = alpha_prior + beta_prior + (decay_rate * t) ** decay_exponent
    return alpha_prior / denominator


def _build_initial_guess_and_bounds():
    """Return (initial_guess, lower_bounds, upper_bounds) arrays for curve_fit."""
    initial_guess = np.array([0.1, 1.0, 1.0, 10.0], dtype=float)
    lower_bounds = np.array([0.001, 0.01, 0.1, 2.0], dtype=float)
    upper_bounds = np.array([0.999, 100.0, 5.0, 500.0], dtype=float)
    return initial_guess, lower_bounds, upper_bounds


def _compute_fit_diagnostics(t, p, p_hat, params, pcov, lower_bounds, upper_bounds):
    """Compute residuals, metrics, and parameter uncertainty from a curve fit."""
    fail_rate, decay_rate, decay_exponent, prior_weight = params

    resid = p - p_hat
    abs_resid = np.abs(resid)
    sq_resid = resid ** 2
    sse = float(np.sum(sq_resid))
    mae = float(np.mean(abs_resid))
    rmse = float(np.sqrt(np.mean(sq_resid)))
    p_mean = float(np.mean(p))
    sst = float(np.sum((p - p_mean) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 0 else np.nan

    # Parameter standard errors from covariance matrix
    se = np.full(4, np.nan, dtype=float)
    if isinstance(pcov, np.ndarray) and pcov.shape == (4, 4):
        diag = np.diag(pcov)
        se = np.where(np.isfinite(diag) & (diag >= 0.0), np.sqrt(diag), np.nan)

    ci95 = 1.96 * se
    param_names = ["fail_rate", "decay_rate", "decay_exponent", "prior_weight"]
    param_values = [fail_rate, decay_rate, decay_exponent, prior_weight]

    params_df = pd.DataFrame({
        "param": param_names,
        "value": param_values,
        "std_error": se,
        "ci95_low": [v - c if np.isfinite(c) else np.nan for v, c in zip(param_values, ci95)],
        "ci95_high": [v + c if np.isfinite(c) else np.nan for v, c in zip(param_values, ci95)],
        "lower_bound": lower_bounds,
        "upper_bound": upper_bounds,
        "within_bounds": [(lower_bounds[i] <= params[i] <= upper_bounds[i]) for i in range(4)],
    })

    emp_vs_fit = pd.DataFrame({
        "t": t, "p_empirical": p, "p_hat": p_hat,
        "residual": resid, "abs_residual": abs_resid, "sq_residual": sq_resid,
    })

    metrics = pd.DataFrame([{
        "n_points": int(t.size), "rmse": rmse, "mae": mae, "r2": r2,
        "sse": sse, "sst": sst, "mean_p": p_mean,
        "covariance_finite": bool(np.isfinite(se).any()),
    }])

    # Covariance matrix
    if isinstance(pcov, np.ndarray) and pcov.shape == (4, 4):
        cov_df = pd.DataFrame(pcov, index=param_names, columns=param_names)
    else:
        cov_df = pd.DataFrame(columns=param_names)

    return params_df, emp_vs_fit, metrics, cov_df


def _audit_learning_curve(audit_path, attempt_column, empirical_data, initial_guess,
                          lower_bounds, upper_bounds, params_df, alpha, beta,
                          prior_weight, fail_rate, emp_vs_fit, metrics,
                          t_fit, p_fit, cov_df, audit_context):
    """Write all learning-curve audit tables under ``audit_path/learning_curve/``."""
    base = audit_path / "learning_curve"
    audit_table(True, base / "00_empirical_data", empirical_data, head=None)
    igb = pd.DataFrame({
        "param": ["fail_rate", "decay_rate", "decay_exponent", "prior_weight"],
        "initial_guess": initial_guess, "lower_bound": lower_bounds, "upper_bound": upper_bounds,
    })
    audit_table(True, base / "01_initial_guess_bounds", igb, head=None)
    audit_table(True, base / "02_fit_params", params_df, head=None)
    audit_table(True, base / "03_prior_alpha_beta",
                pd.DataFrame([{"alpha": alpha, "beta": beta,
                               "prior_weight": prior_weight, "fail_rate": fail_rate}]),
                head=None)

    emp_vs_fit_out = emp_vs_fit.rename(columns={"t": attempt_column})
    audit_table(True, base / "04_empirical_vs_fitted", emp_vs_fit_out, head=None)
    audit_table(True, base / "05_fit_metrics", metrics, head=None)
    audit_table(True, base / "06_tfit_curve",
                pd.DataFrame({attempt_column: t_fit, "p_fit": p_fit}), head=None)
    audit_table(True, base / "07_param_covariance", cov_df, head=None)
    if audit_context:
        audit_table(True, base / "08_context", pd.DataFrame([audit_context]), head=None)


def fit_learning_curve(
    df: pd.DataFrame,
    attempt_column: str,
    *,
    audit: bool = False,
    audit_path: Path | None = None,
    audit_context: dict | None = None,
):
    """
    Fit a Bayesian learning curve to the empirical has_loss probabilities.

    Returns
    -------
    t_fit, p_fit : np.ndarray – fitted curve points
    alpha, beta : float – inferred prior parameters
    decay_rate, decay_exponent, prior_weight : float – fitted parameters
    """
    empirical_data = compute_empirical_cumulative_has_loss(df, attempt_column)
    t = empirical_data[attempt_column].to_numpy()
    p = empirical_data['Empirical_has_loss_Probability'].to_numpy()

    if audit and audit_path is not None:
        audit_table(True, audit_path / "learning_curve" / "00_empirical_data",
                    empirical_data, head=None)

    # Degenerate case: no data at attempts >= 3
    if t.size == 0:
        if audit and audit_path is not None:
            empty_metrics = pd.DataFrame([{
                "n_points": 0, "rmse": np.nan, "mae": np.nan, "r2": np.nan,
                "note": "No empirical points (attempt >= 3)",
            }])
            audit_table(True, audit_path / "learning_curve" / "05_fit_metrics",
                        empty_metrics, head=None)
        return (np.array([], dtype=float), np.array([], dtype=float),
                np.nan, np.nan, np.nan, np.nan, np.nan)

    initial_guess, lower_bounds, upper_bounds = _build_initial_guess_and_bounds()

    try:
        params, pcov = curve_fit(
            bayesian_learning_curve, t, p,
            p0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
            maxfev=10000,
        )
    except Exception as e:
        if audit and audit_path is not None:
            audit_table(True, audit_path / "learning_curve" / "99_fit_error",
                        pd.DataFrame([{"error": str(e)}]), head=None)
        raise

    fail_rate, decay_rate, decay_exponent, prior_weight = params
    alpha = float(prior_weight * fail_rate)
    beta = float(prior_weight * (1.0 - fail_rate))

    p_hat_emp = bayesian_learning_curve(t, *params)
    t_fit = np.arange(3, int(np.max(t)) + 10, dtype=int)
    p_fit = bayesian_learning_curve(t_fit, *params)

    # Diagnostics
    params_df, emp_vs_fit, metrics, cov_df = _compute_fit_diagnostics(
        t, p, p_hat_emp, params, pcov, lower_bounds, upper_bounds,
    )

    if audit and audit_path is not None:
        _audit_learning_curve(
            audit_path, attempt_column, empirical_data,
            initial_guess, lower_bounds, upper_bounds,
            params_df, alpha, beta, prior_weight, fail_rate,
            emp_vs_fit, metrics, t_fit, p_fit, cov_df, audit_context,
        )

    return t_fit, p_fit, alpha, beta, decay_rate, decay_exponent, prior_weight


# ---------------------------------------------------------------------------
# Empirical conditional probabilities (p1, p2|y1, p3|y1,y2)
# ---------------------------------------------------------------------------

def compute_empirical_conditionals(
    df: pd.DataFrame,
    attempt_column: str,
    grouping_col: str,
    *,
    audit: bool = False,
    audit_path: Path | None = None,
):
    """
    Compute empirical has_loss rates for launches 1 to 3.

    Returns
    -------
    p1 : float – unconditional has_loss rate at launch 1
    p2_given : dict – conditional rate at launch 2 given launch 1 outcome
    p3_given : dict – conditional rate at launch 3 given prior fail count
    """
    p1 = df[df[attempt_column] == 1]['has_loss'].mean()

    # Pairs for launch 2 conditioning
    l1 = df[df[attempt_column] == 1][[grouping_col, 'has_loss']].rename(
        columns={'has_loss': 'has_loss_1'})
    l2 = df[df[attempt_column] == 2][[grouping_col, 'has_loss']].rename(
        columns={'has_loss': 'has_loss_2'})
    pairs = pd.merge(l1, l2, on=grouping_col, how="inner")

    p2_given = {
        'F': pairs[pairs['has_loss_1'] == 1]['has_loss_2'].mean(),
        'S': pairs[pairs['has_loss_1'] == 0]['has_loss_2'].mean(),
    }

    # Triplets for launch 3 conditioning
    l3 = df[df[attempt_column] == 3][[grouping_col, 'has_loss']].rename(
        columns={'has_loss': 'has_loss_3'})
    triples = pd.merge(pairs, l3, on=grouping_col, how="inner")
    triples['fail_count'] = triples['has_loss_1'] + triples['has_loss_2']
    p3_given = {
        str(int(fc)): grp['has_loss_3'].mean()
        for fc, grp in triples.groupby('fail_count')
    }

    if audit and audit_path is not None:
        _audit_empirical_conditionals(
            audit_path, grouping_col, l1, l2, l3, pairs, triples,
            p1, p2_given, p3_given,
        )

    return p1, p2_given, p3_given


def _audit_empirical_conditionals(audit_path, grouping_col, l1, l2, l3,
                                  pairs, triples, p1, p2_given, p3_given):
    """Write empirical conditional audit tables."""
    base = audit_path / "empirical"

    coverage = pd.DataFrame([{
        "grouping_col": grouping_col,
        "n_l1_total": int(len(l1)), "n_l2_total": int(len(l2)),
        "n_pairs_l1_l2": int(len(pairs)),
        "n_l3_total": int(len(l3)), "n_triples": int(len(triples)),
    }])
    audit_table(True, base / "00_coverage", coverage, head=None)
    audit_table(True, base / "01_l1", l1, head=None)
    audit_table(True, base / "02_l2", l2, head=None)
    audit_table(True, base / "03_pairs_l1_l2", pairs, head=None)
    audit_table(True, base / "04_l3", l3, head=None)
    audit_table(True, base / "05_triples", triples, head=None)

    p1_summary = pd.DataFrame([{
        "grouping_col": grouping_col, "p1": float(p1),
        "n_l1": int(len(l1)), "n_l1_fail": int(l1["has_loss_1"].sum()),
    }])
    audit_table(True, base / "06_p1_summary", p1_summary, head=None)

    def _p2_branch(df_, cond_val):
        br = df_[df_["has_loss_1"] == cond_val]
        return {
            "n_pairs": int(len(br)),
            "n_losses_at_l2": int(br["has_loss_2"].sum()),
            "p2_mean": float(br["has_loss_2"].mean()) if len(br) else np.nan,
        }

    p2_summary = pd.DataFrame([
        {"grouping_col": grouping_col, "branch": "L1=Fail", **_p2_branch(pairs, 1)},
        {"grouping_col": grouping_col, "branch": "L1=Success", **_p2_branch(pairs, 0)},
    ])
    audit_table(True, base / "07_p2_summary", p2_summary, head=None)

    if len(triples):
        p3_summary = (
            triples.groupby("fail_count", as_index=False)["has_loss_3"]
            .agg(n_triplets="count", n_losses_at_l3="sum", p3_mean="mean")
        )
        p3_summary.insert(0, "grouping_col", grouping_col)
        p3_summary["fail_count"] = p3_summary["fail_count"].astype(int)
    else:
        p3_summary = pd.DataFrame(
            columns=["grouping_col", "fail_count", "n_triplets", "n_losses_at_l3", "p3_mean"])

    audit_table(True, base / "08_p3_summary", p3_summary, head=None)

    reported = pd.DataFrame([
        {"metric": "p1", "value": float(p1)},
        {"metric": "p2_given[L1=Fail]", "value": float(p2_given.get("F", np.nan))},
        {"metric": "p2_given[L1=Success]", "value": float(p2_given.get("S", np.nan))},
        {"metric": "p3_given[fail_count=0]", "value": float(p3_given.get("0", np.nan))},
        {"metric": "p3_given[fail_count=1]", "value": float(p3_given.get("1", np.nan))},
        {"metric": "p3_given[fail_count=2]", "value": float(p3_given.get("2", np.nan))},
    ])
    reported.insert(0, "grouping_col", grouping_col)
    audit_table(True, base / "09_reported_conditionals", reported, head=None)


# ---------------------------------------------------------------------------
# Prior helpers
# ---------------------------------------------------------------------------

def empirical_prior(t, outcome_history, p2_given, p3_given, default_weight=10):
    """
    Return (alpha, beta) prior for t=2 or t=3 based on empirical estimates,
    or None for t >= 4 (use learning curve instead).
    """
    if t == 2 and len(outcome_history) >= 1:
        prev = outcome_history[0]
        p = p2_given['F' if prev == 1 else 'S']
    elif t == 3 and len(outcome_history) >= 2:
        key = sum(outcome_history)
        p = p3_given[str(key)]
    else:
        return None

    alpha = default_weight * p
    beta = default_weight * (1 - p)
    return alpha, beta


def has_loss_prior(t, alpha0, beta0, lambda_, delta):
    """Compute the prior has_loss probability at launch ``t``."""
    t = max(1, t)
    return alpha0 / (alpha0 + beta0 + (lambda_ * (t - 1)) ** delta)


def apply_exponential_decay(outcomes, current_launch, decay_tau):
    """
    Apply exponential decay to outcome weights based on their age.

    Returns (fail_weight, success_weight).
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
    """Alias for has_loss_prior (kept for backward compatibility)."""
    t = max(1, t)
    return alpha0 / (alpha0 + beta0 + (lambda_ * (t - 1)) ** delta)


# ---------------------------------------------------------------------------
# Beta CI helper
# ---------------------------------------------------------------------------

def _beta_ci(a: float, b: float, ci: float = 0.95):
    """Compute mean, lower, upper from Beta(a, b) at given confidence."""
    total = a + b
    p_mean = a / total if total > 0 else np.nan
    if a > 0 and b > 0:
        p_lower = _beta.ppf((1 - ci) / 2, a, b)
        p_upper = _beta.ppf(1 - (1 - ci) / 2, a, b)
    else:
        p_lower = np.nan
        p_upper = np.nan
    return float(p_mean), float(p_lower), float(p_upper)


# ---------------------------------------------------------------------------
# _LaunchBayesPredictor – shared Bayesian prediction machinery
# ---------------------------------------------------------------------------

class _LaunchBayesPredictor:
    """
    Core Bayesian prediction engine for launch failure probability.

    Manages:
    - Curve shape g(t), step factor s_t, early-weight w_early(t)
    - Empirical priors for t=1,2,3
    - Observation buckets (early / late / base / curve)
    - Gap advancement via curve nudge
    - Prior composition at a given t
    """

    def __init__(
        self,
        lam: float,
        delt: float,
        default_weight: float,
        p1_global: float,
        p2_given: Dict[str, float],
        p3_given: Dict[str, float],
        *,
        audit: bool = False,
        audit_events: list | None = None,
    ) -> None:
        self.lam = float(lam)
        self.delt = float(delt)
        self.w0 = float(default_weight)
        self.p1 = float(p1_global)
        self.p2_given = p2_given
        self.p3_given = p3_given
        self.audit = bool(audit)
        self._events = audit_events if audit_events is not None else ([] if audit else None)
        self._seq = 0

    # --- Audit logger ---

    def _log(self, **kv):
        if not self.audit or self._events is None:
            return
        self._seq += 1
        row = {"seq": self._seq}
        row.update({
            k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
            for k, v in kv.items()
        })
        self._events.append(row)

    # --- Curve shape & schedules ---

    def g(self, t: int) -> float:
        """Learning curve shape function."""
        return 1.0 / (1.0 + (self.lam * max(t - 1, 0)) ** self.delt)

    def step_factor(self, t: int) -> float:
        """Ratio g(t)/g(t-1) clamped to [0, 1]."""
        if t <= 1:
            return 1.0
        gt = self.g(t)
        gtm1 = self.g(t - 1)
        return float(max(0.0, min(1.0, gt / max(gtm1, 1e-12))))

    def w_early(self, t: int) -> float:
        """Early-launch weight: 1.0 through t=40, fades to 0.0 by t=60."""
        if t <= 40:
            return 1.0
        if t >= 60:
            return 0.0
        g40, g60 = self.g(40), self.g(60)
        denom = g40 - g60
        if abs(denom) < 1e-12:
            return (60.0 - float(t)) / 20.0  # linear fallback
        return float(max(0.0, min(1.0, (self.g(t) - g60) / denom)))

    # --- Empirical priors for t=1,2,3 ---

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
        """Compute the empirical p3 anchor probability given observed launches 1-2."""
        has_t1 = (launches == 1).any()
        has_t2 = (launches == 2).any()
        if has_t1 and has_t2:
            key = str(int(outcomes[launches == 1][0] + outcomes[launches == 2][0]))
            return float(self.p3_given.get(key, self.p1))
        if has_t1 and not has_t2:
            first = int(outcomes[launches == 1][0])
            p2 = float(self.p2_given["F" if first == 1 else "S"])
            if first == 1:
                return ((1 - p2) * float(self.p3_given.get("1", self.p1))
                        + p2 * float(self.p3_given.get("2", self.p1)))
            else:
                return ((1 - p2) * float(self.p3_given.get("0", self.p1))
                        + p2 * float(self.p3_given.get("1", self.p1)))
        return float(self.p1)

    def prior_empirical_t3(self, launches: np.ndarray, outcomes: np.ndarray) -> tuple[float, float]:
        p = self.empirical_p3_anchor(launches, outcomes)
        a, b = self.w0 * p, self.w0 * (1 - p)
        self._log(action="empirical_prior_t3", p=p, a=a, b=b)
        return a, b

    # --- Observation buckets ---

    @staticmethod
    def _init_buckets() -> Dict[str, float]:
        """Initialize observation buckets: early (t<=20), late (t>20), base anchor, curve."""
        return dict(
            a_early=0.0, b_early=0.0,
            a_late=0.0, b_late=0.0,
            a_base=0.0, b_base=0.0,
            b_curve=0.0,
        )

    @staticmethod
    def _posterior_update(buckets: Dict[str, float], t_obs: int, y: int) -> None:
        """Update buckets with an observation at t_obs."""
        if t_obs <= 20:
            buckets["a_early"] += y
            buckets["b_early"] += (1 - y)
        else:
            buckets["a_late"] += y
            buckets["b_late"] += (1 - y)

    def _compose_prior_counts(self, buckets: Dict[str, float], t: int) -> tuple[float, float]:
        """Compose alpha/beta from all buckets, applying early-weight fade."""
        w = self.w_early(t)
        a = w * buckets["a_early"] + buckets["a_late"] + buckets["a_base"]
        b = w * buckets["b_early"] + buckets["b_late"] + buckets["b_base"] + buckets["b_curve"]
        return a, b

    def _curve_nudge_once(self, buckets: Dict[str, float], t: int) -> None:
        """Add virtual successes so the mean decreases by the step factor."""
        a_raw, b_raw = self._compose_prior_counts(buckets, t)
        tot = a_raw + b_raw
        if tot <= 0:
            self._log(action="advance_curve", t=t, note="no_totals")
            return
        mu_prev = a_raw / tot
        s_t = self.step_factor(t)
        mu_target = min(mu_prev, mu_prev * s_t)
        delta_b = a_raw * (1.0 / max(mu_target, 1e-12) - 1.0) - b_raw
        if delta_b > 1e-12:
            buckets["b_curve"] += delta_b
        a_new, b_new = self._compose_prior_counts(buckets, t)
        self._log(
            action="advance_curve", t=t, w_early=self.w_early(t),
            g=self.g(t), step_factor=s_t, mu_prev=mu_prev, mu_target=mu_target,
            delta_b=max(delta_b, 0.0), a=a_new, b=b_new,
        )

    def advance_to(self, buckets: Dict[str, float], last_t: int, target_t: int) -> int:
        """
        Apply curve-improvement step for each t in (last_t, target_t], starting at t >= 4.
        No observations are added.
        """
        start = max(last_t + 1, 4)
        if target_t >= start:
            for tt in range(start, target_t + 1):
                self._curve_nudge_once(buckets, tt)
            return target_t
        return last_t

    # --- Anchor setup ---

    def ensure_anchor(self, buckets: Dict[str, float], launches: np.ndarray,
                      outcomes: np.ndarray, default_weight: float,
                      base_set: bool) -> tuple[bool, int]:
        """Set the t=3 anchor if not already set. Returns (base_set, last_t)."""
        if base_set:
            return True, 3
        p3 = self.empirical_p3_anchor(launches, outcomes)
        buckets["a_base"] = default_weight * p3
        buckets["b_base"] = default_weight * (1 - p3)
        self._log(action="set_anchor_t3", t=3, p_anchor=p3,
                  a_base=buckets["a_base"], b_base=buckets["b_base"])
        return True, 3


# ---------------------------------------------------------------------------
# Full-history predictions (all observed launches)
# ---------------------------------------------------------------------------

def predict_has_loss_probabilities_all_launches_with_empirical(
    df: pd.DataFrame,
    lambda_set: Dict[int, float],
    delta_set: Dict[int, float],
    p2_given: Dict[str, float],
    p3_given: Dict[str, float],
    alpha0_set: Dict[int, float],
    beta0_set: Dict[int, float],
    default_weight: float = 10,
    ci: float = 0.95,
    decay_tau: float = 10,
    grouping_col: str = "vehicle_type",
    attempt_column: str = "lv_family_attempt_number",
    *,
    audit: bool = False,
    audit_path: Path | None = None,
) -> pd.DataFrame:
    """
    For each observed launch, compute the PRIOR at that launch number.

    - t=1..3: purely empirical priors.
    - t>=4: running posterior from t=3 anchor, advanced via learning curve.
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
        dates = sub["launch_date"].to_numpy()
        sc_ids = sub["seradata_spacecraft_id"].to_numpy()

        vdir = (base_dir / _safe_name(str(vehicle))) if base_dir is not None else None
        evts: list[Dict[str, Any]] = []
        pred = _LaunchBayesPredictor(
            lam, delt, default_weight, float(p1_global), p2_given, p3_given,
            audit=audit, audit_events=evts,
        )

        buckets = pred._init_buckets()
        base_set = False
        last_t = 0
        cum_launches = 0
        cum_failures = 0

        if audit and vdir is not None:
            seq = pd.DataFrame({"attempt": launches, "has_loss": outcomes, "launch_date": dates})
            audit_table(True, vdir / "01_sequence", seq, head=None)

        for i, t_obs in enumerate(launches):
            t_obs = int(t_obs)

            # Compute PRIOR at t_obs
            a_prior, b_prior = _compute_prior_at_t(
                pred, t_obs, launches, outcomes, buckets,
                base_set, last_t, default_weight, audit,
            )
            if t_obs >= 3 and not base_set:
                base_set = True
                last_t = 3
            if t_obs >= 4:
                last_t = t_obs

            p_mean, p_lower, p_upper = _beta_ci(a_prior, b_prior, ci)

            outcome_i = int(outcomes[i])
            cum_launches += 1
            cum_failures += outcome_i

            rows.append({
                grouping_col: vehicle,
                "launch_number": int(t_obs),
                "launch_date": dates[i],
                "observed_has_loss": outcome_i,
                "cumulative_rate": float(cum_failures / cum_launches),
                "predicted": float(p_mean),
                "ci_lower": float(p_lower),
                "ci_upper": float(p_upper),
                "spacecraft_id": sc_ids[i],
            })

            # Posterior update (only t >= 3 to avoid double-counting)
            if t_obs >= 3:
                pred._posterior_update(buckets, t_obs, outcome_i)
                if audit:
                    post_a, post_b = pred._compose_prior_counts(buckets, t_obs)
                    pred._log(
                        action="observe", t=t_obs, y=outcome_i,
                        a_before=a_prior, b_before=b_prior,
                        a_after=post_a, b_after=post_b,
                        p_after=post_a / (post_a + post_b) if (post_a + post_b) > 0 else np.nan,
                    )
                last_t = t_obs

        if audit and vdir is not None:
            evtdf = pd.DataFrame(evts) if evts else pd.DataFrame()
            if "seq" in evtdf.columns:
                evtdf = evtdf.sort_values("seq")
            audit_table(True, vdir / "02_event_log", evtdf, head=None)

    return pd.DataFrame(rows)


def _compute_prior_at_t(pred, t_obs, launches, outcomes, buckets,
                        base_set, last_t, default_weight, audit):
    """
    Compute (a_prior, b_prior) at the given launch number.

    For t=1,2,3: purely empirical. For t>=4: composed from buckets after curve advance.
    """
    if t_obs == 1:
        a, b = pred.prior_empirical_t1()
    elif t_obs == 2:
        a, b = pred.prior_empirical_t2(launches, outcomes)
    elif t_obs == 3:
        if not base_set:
            p3 = pred.empirical_p3_anchor(launches, outcomes)
            buckets["a_base"] = default_weight * p3
            buckets["b_base"] = default_weight * (1 - p3)
            if audit:
                pred._log(action="set_anchor_t3", t=3, p_anchor=p3,
                          a_base=buckets["a_base"], b_base=buckets["b_base"])
        a, b = pred._compose_prior_counts(buckets, 3)
    else:
        # t >= 4: ensure anchor exists, advance curve, compose
        if not base_set:
            p3 = pred.empirical_p3_anchor(launches, outcomes)
            buckets["a_base"] = default_weight * p3
            buckets["b_base"] = default_weight * (1 - p3)
            if audit:
                pred._log(action="set_anchor_t3", t=3, p_anchor=p3,
                          a_base=buckets["a_base"], b_base=buckets["b_base"])
        effective_last = max(last_t, 3) if base_set else 3
        pred.advance_to(buckets, effective_last, t_obs)
        a, b = pred._compose_prior_counts(buckets, t_obs)

    if audit:
        total = a + b
        pred._log(action="prior_at_t", t=t_obs, a=a, b=b,
                  p=a / total if total > 0 else np.nan)
    return a, b


# ---------------------------------------------------------------------------
# Next-launch prediction per vehicle
# ---------------------------------------------------------------------------

def predict_next_failure_probability_per_vehicle(
    df: pd.DataFrame,
    lambda_set: Dict[int, float],
    delta_set: Dict[int, float],
    p2_given: Dict[str, float],
    p3_given: Dict[str, float],
    alpha0_set: Dict[int, float],
    beta0_set: Dict[int, float],
    default_weight: float = 10,
    ci: float = 0.95,
    decay_tau: float = 10,
    grouping_col: str = "vehicle_type",
    attempt_column: str = "lv_family_attempt_number",
    *,
    audit: bool = False,
    audit_path: Path | None = None,
) -> pd.DataFrame:
    """
    One-step-ahead prediction per vehicle.

    - next in {1,2,3}: purely empirical.
    - next >= 4: anchor at t=3, update with t>=3 observations, advance via curve.
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
        sub = df[df[grouping_col] == vehicle].sort_values(
            [attempt_column, "launch_date"], na_position="last")
        if sub.empty:
            continue

        launches = sub[attempt_column].to_numpy().astype(int)
        outcomes = sub["has_loss"].to_numpy().astype(int)
        dates = sub["launch_date"].to_numpy()
        next_launch = (int(launches.max()) + 1) if len(launches) else 1

        vdir = (base_dir / _safe_name(str(vehicle))) if base_dir is not None else None
        evts: list[Dict[str, Any]] = []
        pred = _LaunchBayesPredictor(
            lam, delt, default_weight, float(p1_global), p2_given, p3_given,
            audit=audit, audit_events=evts,
        )

        if audit and vdir is not None:
            _audit_next_failure_metadata(
                vdir, vehicle, grouping_col, p1_global, lam, delt,
                default_weight, ci, next_launch, launches, outcomes,
                dates, p2_given, p3_given,
            )

        a_next, b_next = _predict_next_for_vehicle(
            pred, next_launch, launches, outcomes, default_weight, audit,
        )

        p_mean, p_lower, p_upper = _beta_ci(a_next, b_next, ci)

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
            evtdf = pd.DataFrame(evts) if evts else pd.DataFrame()
            if "seq" in evtdf.columns:
                evtdf = evtdf.sort_values("seq")
            audit_table(True, vdir / "02_event_log", evtdf, head=None)
            prior_next = pd.DataFrame([{
                "next_launch": int(next_launch),
                "a_next": float(a_next), "b_next": float(b_next),
                "p_mean": float(p_mean), "ci_lower": float(p_lower), "ci_upper": float(p_upper),
            }])
            audit_table(True, vdir / "03_prior_next", prior_next, head=None)

    return pd.DataFrame(results)


def _predict_next_for_vehicle(pred, next_launch, launches, outcomes, default_weight, audit):
    """Compute (a_next, b_next) for the next launch of a single vehicle."""
    if next_launch == 1:
        a, b = pred.prior_empirical_t1()
        if audit:
            pred._log(action="prediction_next_launch_1", t=1, a=a, b=b,
                      p=a / (a + b) if (a + b) > 0 else np.nan)
        return a, b

    if next_launch == 2:
        a, b = pred.prior_empirical_t2(launches, outcomes)
        if audit:
            pred._log(action="prediction_next_launch_2", t=2, a=a, b=b,
                      p=a / (a + b) if (a + b) > 0 else np.nan)
        return a, b

    if next_launch == 3:
        a, b = pred.prior_empirical_t3(launches, outcomes)
        if audit:
            pred._log(action="prediction_next_launch_3", t=3, a=a, b=b,
                      p=a / (a + b) if (a + b) > 0 else np.nan)
        return a, b

    # next_launch >= 4: anchor at t=3, observe t>=3, advance to next
    buckets = pred._init_buckets()
    p3 = pred.empirical_p3_anchor(launches, outcomes)
    buckets["a_base"] = default_weight * p3
    buckets["b_base"] = default_weight * (1 - p3)
    if audit:
        pred._log(action="set_anchor_t3", t=3, p_anchor=p3,
                  a=buckets["a_base"], b=buckets["b_base"])
    last_t = 3

    # Replay observations at t >= 3
    for i, t_obs in enumerate(launches):
        t_obs = int(t_obs)
        if t_obs < 3:
            continue
        last_t = pred.advance_to(buckets, last_t, t_obs)

        if audit:
            a_prior, b_prior = pred._compose_prior_counts(buckets, t_obs)
            pred._log(action="prior_at_t", t=t_obs, a=a_prior, b=b_prior,
                      p=a_prior / (a_prior + b_prior) if (a_prior + b_prior) > 0 else np.nan)

        prev_a, prev_b = pred._compose_prior_counts(buckets, t_obs)
        pred._posterior_update(buckets, t_obs, int(outcomes[i]))
        if audit:
            post_a, post_b = pred._compose_prior_counts(buckets, t_obs)
            pred._log(
                action="observe", t=t_obs, y=int(outcomes[i]),
                a_before=prev_a, b_before=prev_b,
                a_after=post_a, b_after=post_b,
                p_after=post_a / (post_a + post_b) if (post_a + post_b) > 0 else np.nan,
            )
        last_t = t_obs

    # Advance posterior to next launch
    last_t = pred.advance_to(buckets, last_t, int(next_launch))
    a_next, b_next = pred._compose_prior_counts(buckets, int(next_launch))
    if audit:
        pred._log(action="compose_prior_at_next", t=int(next_launch), a=a_next, b=b_next,
                  p=a_next / (a_next + b_next) if (a_next + b_next) > 0 else np.nan)
    return a_next, b_next


def _audit_next_failure_metadata(vdir, vehicle, grouping_col, p1_global, lam, delt,
                                 default_weight, ci, next_launch, launches, outcomes,
                                 dates, p2_given, p3_given):
    """Write metadata and sequence audit files for a vehicle."""
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


# ---------------------------------------------------------------------------
# Dropdown / rate loading helpers
# ---------------------------------------------------------------------------

def create_dropdown_options(df: pd.DataFrame, grouping_col: str) -> pd.DataFrame:
    """Create dropdown options for a given grouping column."""
    if grouping_col not in df.columns:
        raise KeyError(f"Column '{grouping_col}' not found in DataFrame.")
    pretty_name = _prettify_grouping_label(grouping_col)
    unique_values = df[grouping_col].dropna().unique()
    values = sorted(map(str, unique_values))
    return pd.DataFrame([{"grouping_col": pretty_name, "value": v} for v in values])


def load_specific_primary_rates_long(output_dir: Path, date_tag: str) -> pd.DataFrame:
    """
    Build one long table from per-grouping primary-rate files.

    Columns: ['grouping_col', 'value', *SPECIFIC_COLS]
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
            [["grouping_col", "value"] + SPECIFIC_COLS]
        )
        frames.append(tmp)

    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=["grouping_col", "value"] + SPECIFIC_COLS)


def load_type_primary_rates(output_dir: Path, date_tag: str) -> pd.DataFrame:
    """Load vehicle_type primary rates and suffix columns with _type."""
    path = output_dir / f"launch_primary_rates_vehicle_type_{date_tag}_next.csv"
    if not path.exists():
        logger.warning("Type rates file not found: %s", path)
        return pd.DataFrame(columns=["vehicle_type"] + TYPE_COLS)
    df = pd.read_csv(path)
    if "vehicle_type" not in df.columns:
        logger.warning("Expected 'vehicle_type' column missing in %s", path)
        return pd.DataFrame(columns=["vehicle_type"] + TYPE_COLS)
    df = _rename_rate_cols(df, "_type")
    return df[["vehicle_type"] + TYPE_COLS]


def coerce_dropdown_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure dropdown DataFrame has consistent column types."""
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

    for c in INT_COLS + FLOAT_COLS + STR_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    for c in INT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")
    for c in FLOAT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype("float64")
    for c in STR_COLS:
        df[c] = df[c].fillna("").astype(str)

    return df


# ---------------------------------------------------------------------------
# Pipeline steps (called by main)
# ---------------------------------------------------------------------------

def _process_grouping(launch_df, grouping_col, date_tag, audit_outputs, audit_dir):
    """
    Process a single grouping level: fit learning curve, predict rates,
    build dropdown rows, and export CSV/JSON files.

    Returns the dropdown DataFrame for this grouping.
    """
    grp_dir = (audit_dir / grouping_col) if audit_outputs else None
    logger.info("=== Processing grouping: %s ===", grouping_col)

    # 1) Select columns for this grouping
    clean_launch_data, attempt_column, selected_columns = select_grouping_columns(
        launch_df, grouping_col,
    )

    # 2) Empirical conditionals
    p1, p2_given, p3_given = compute_empirical_conditionals(
        launch_df, attempt_column=attempt_column, grouping_col=grouping_col,
        audit=audit_outputs, audit_path=grp_dir,
    )
    logger.info("Empirical conditionals for %s: p1=%f, p2_given=%s, p3_given=%s",
                grouping_col, p1, p2_given, p3_given)

    # 3) Fit the learning curve
    logger.info("Fitting learning curve for %s...", grouping_col)
    if (clean_launch_data[attempt_column] >= 3).any():
        _t_fit, _p_fit, alpha_all, beta_all, lambda_all, delta_all, _prior_w = fit_learning_curve(
            clean_launch_data, attempt_column,
            audit=audit_outputs, audit_path=grp_dir,
            audit_context={"grouping_col": grouping_col, "attempt_column": attempt_column},
        )
    else:
        logger.warning("Not enough attempts (>=3) in %s; using NaNs.", grouping_col)
        alpha_all = beta_all = lambda_all = delta_all = np.nan

    # Use a single param set for all buckets
    param_set = {0: alpha_all, 1: alpha_all, 2: alpha_all}
    alpha0_set = param_set
    beta0_set = {0: beta_all, 1: beta_all, 2: beta_all}
    lambda_set = {0: lambda_all, 1: lambda_all, 2: lambda_all}
    delta_set = {0: delta_all, 1: delta_all, 2: delta_all}

    # 4) Predict next-launch failure rate per vehicle
    primary_rates = predict_next_failure_probability_per_vehicle(
        df=clean_launch_data,
        alpha0_set=alpha0_set, beta0_set=beta0_set,
        lambda_set=lambda_set, delta_set=delta_set,
        p2_given=p2_given, p3_given=p3_given,
        ci=0.95, grouping_col=grouping_col, attempt_column=attempt_column,
        audit=audit_outputs, audit_path=grp_dir,
    )

    # 5) Full history predictions
    full_history_df = predict_has_loss_probabilities_all_launches_with_empirical(
        clean_launch_data,
        lambda_set=lambda_set, delta_set=delta_set,
        p2_given=p2_given, p3_given=p3_given,
        alpha0_set=alpha0_set, beta0_set=beta0_set,
        grouping_col=grouping_col, attempt_column=attempt_column,
        audit=True, audit_path=grp_dir,
    )

    # 6) Build dropdown rows
    dropdown_rows_df = build_dropdown_rows_for_grouping(launch_df, grouping_col)

    # 7) Export per-grouping CSV + JSON
    _export_grouping_outputs(primary_rates, full_history_df, grouping_col, date_tag)

    return dropdown_rows_df


def _export_grouping_outputs(primary_rates, full_history_df, grouping_col, date_tag):
    """Export next-launch and full-history DataFrames as CSV and JSON."""
    prefix = f"launch_primary_rates_{grouping_col}_{date_tag}"
    next_path = OUTPUT_DIR / f"{prefix}_next.csv"
    full_path = OUTPUT_DIR / f"{prefix}_full.csv"

    primary_rates.to_csv(next_path, index=False)
    full_history_df.to_csv(full_path, index=False)
    primary_rates.to_json(next_path.with_suffix(".json"), orient="records", indent=2)
    full_history_df.to_json(full_path.with_suffix(".json"), orient="records", indent=2)

    logger.info("Saved CSVs & JSONs for %s:\n- %s\n- %s\n- %s\n- %s",
                grouping_col, next_path, full_path,
                next_path.with_suffix(".json"), full_path.with_suffix(".json"))


def _build_unified_dropdown(all_dropdown_rows, date_tag):
    """Merge dropdown rows with primary rates and export unified dropdown CSV/JSON."""
    if all_dropdown_rows:
        all_dropdowns_df = pd.concat(all_dropdown_rows, ignore_index=True)
        all_dropdowns_df = (
            all_dropdowns_df.drop_duplicates()
            .sort_values(["grouping_col", "value"], kind="mergesort")
            .reset_index(drop=True)
        )
    else:
        all_cols = (
            ["grouping_col", "value"]
            + list({c for v in INCLUDE_BY_GROUPING.values() for c in v})
            + ATTEMPT_COLS + ["total_failures"]
        )
        all_dropdowns_df = pd.DataFrame(columns=all_cols)

    # Attach specific rates (by grouping/value)
    specific_rates = load_specific_primary_rates_long(OUTPUT_DIR, date_tag)
    all_dropdowns_df = all_dropdowns_df.merge(
        specific_rates, on=["grouping_col", "value"], how="left", validate="m:1",
    )

    # Attach type rates (by vehicle_type)
    type_rates = load_type_primary_rates(OUTPUT_DIR, date_tag)
    all_dropdowns_df = all_dropdowns_df.merge(
        type_rates, on="vehicle_type", how="left", validate="m:1",
    )

    all_dropdowns_df = coerce_dropdown_schema(all_dropdowns_df)

    # Export
    csv_path = OUTPUT_DIR / f"launch_dropdown_options_all_{date_tag}.csv"
    all_dropdowns_df.to_csv(csv_path, index=False)
    all_dropdowns_df.to_json(csv_path.with_suffix(".json"), orient="records", indent=2)
    logger.info("Saved unified dropdown options:\n- %s\n- %s",
                csv_path, csv_path.with_suffix(".json"))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(audit_outputs: bool = False) -> None:
    """
    Run the primary base rates production pipeline.

    Steps:
    1. Load and prepare launch data
    2. For each grouping level: fit learning curve, predict rates, export
    3. Build and export unified dropdown table
    """
    date_tag = datetime.now(tz=TZ).strftime("%Y%m%d")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    audit_dir = OUTPUT_DIR / f"audit_{date_tag}"
    if audit_outputs:
        audit_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading and preparing data...")
    launch_df = load_and_prepare_data(filter_min_year_on=False)

    # Debug log for verification
    print(
        launch_df.loc[
            launch_df["vehicle_type"].str.casefold().eq("ZHUQUE-2/SUZAKU-2 (ZQ-2)"),
            ["vehicle_type", "type_launches_since_last_failure"],
        ]
    )

    # Process each grouping level
    all_dropdown_rows: list[pd.DataFrame] = []
    for grouping_col in GROUPINGS:
        dropdown_df = _process_grouping(
            launch_df, grouping_col, date_tag, audit_outputs, audit_dir,
        )
        all_dropdown_rows.append(dropdown_df)

    # Build unified dropdown table
    _build_unified_dropdown(all_dropdown_rows, date_tag)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-outputs", action="store_true",
                        help="Write intermediate audit artifacts")
    args = parser.parse_args()
    main(audit_outputs=args.audit_outputs)
