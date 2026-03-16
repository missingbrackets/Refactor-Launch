from .data_load_feature_creation import load_and_prepare_data
from ..utils.grouping_column import select_grouping_columns
from ..utils.compute_empirical_cumulative_loss import compute_empirical_cumulative_has_loss
from ..utils.build_dropdown_columns import build_dropdown_rows_for_grouping, _prettify_grouping_label
from ..utils.constants import (
    GROUPINGS, OUTPUT_DIR, TZ, IDENTITY_COLS, ATTEMPT_COLS, INCLUDE_BY_GROUPING,
    SPECIFIC_GROUPINGS, SPECIFIC_COLS, TYPE_COLS
)
from ..utils.audit_helpers import _ensure_parent, _to_native, _normalize_table, _with_ext, audit_table, audit_schema
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.optimize import curve_fit
from scipy.stats import beta as _beta
from typing import Dict, Any
import re

def _mode_or_first(series: pd.Series):
    """Safe mode for object columns; falls back to first non-null if no unique mode."""
    s = series.dropna()
    if s.empty:
        return None
    m = s.mode()
    return m.iloc[0] if not m.empty else s.iloc[0]


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

def fit_learning_curve(df, attempt_column):
    """
    Fits a Bayesian learning curve to the empirical has_loss probabilities.
    
    Returns:
    - t_fit: Launch attempt numbers used for prediction
    - p_fit: Predicted has_loss probabilities
    - alpha, beta: Inferred prior parameters
    - λ, δ, prior_weight: Fitted learning curve parameters
    """
    empirical_data = compute_empirical_cumulative_has_loss(df, attempt_column)
    t = empirical_data[attempt_column].values
    p = empirical_data['Empirical_has_loss_Probability'].values

    # Initial guess and bounds for [fail_rate, λ, δ, prior_weight]
    initial_guess = [0.1, 1.0, 1.0, 10.0]
    bounds = ([0.001, 0.01, 0.1, 2], [0.999, 100.0, 5.0, 500.0])

    # Fit curve to empirical data
    params, _ = curve_fit(bayesian_learning_curve, t, p, p0=initial_guess, bounds=bounds)
    fail_rate, decay_rate, decay_exponent, prior_weight = params

    # Compute prior alpha and beta for downstream use
    alpha = prior_weight * fail_rate
    beta = prior_weight * (1 - fail_rate)

    # Predict has_loss probabilities over extended range
    t_fit = np.arange(3, max(t) + 10)
    p_fit = bayesian_learning_curve(t_fit, fail_rate, decay_rate, decay_exponent, prior_weight)

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
                 p1_global: float, p2_given: Dict[str, float], p3_given: Dict[str, float]) -> None:
        self.lam = float(lam)
        self.delt = float(delt)
        self.w0 = float(default_weight)
        self.p1 = float(p1_global)
        self.p2_given = p2_given
        self.p3_given = p3_given

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
        return self.w0 * p, self.w0 * (1 - p)

    def prior_empirical_t2(self, launches: np.ndarray, outcomes: np.ndarray) -> tuple[float, float]:
        if (launches == 1).any():
            prev = int(outcomes[launches == 1][0])
            p = float(self.p2_given["F" if prev == 1 else "S"])
        else:
            p = self.p1 * float(self.p2_given["F"]) + (1 - self.p1) * float(self.p2_given["S"])
        return self.w0 * p, self.w0 * (1 - p)

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
        return self.w0 * p, self.w0 * (1 - p)

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
            return
        mu_prev = a_raw / tot
        s_t = self.step_factor(t)
        mu_target = min(mu_prev, mu_prev * s_t)
        # delta_b = a*(1/mu_target - 1) - b
        delta_b = a_raw * (1.0 / max(mu_target, 1e-12) - 1.0) - b_raw
        if delta_b > 1e-12:
            buckets["b_curve"] += delta_b

    def advance_to(self, buckets: Dict[str, float], last_t: int, target_t: int) -> int:
        """
        Apply the curve-improvement step for each integer t in (last_t, target_t], starting at t>=4.
        No observations are added here (this handles gaps and/or stepping to the next launch).
        Returns the new last_t.
        """
        start = max(last_t + 1, 4)
        if target_t >= start:
            for tt in range(start, target_t + 1):
                self._curve_nudge_once(buckets, tt)
            return target_t
        return last_t

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
    decay_tau: float = 10,  # unused (fade handled by w_early with curve)
    grouping_col: str = "vehicle_type",
    attempt_column: str = "lv_family_attempt_number",
) -> pd.DataFrame:
    """
    Per-launch predictions:
      • t=1..3: fully empirical priors (p1, p2|t1, p3|t1,t2), then posterior update with the observed outcome.
      • t>=4: prior is anchored to empirical t=3; at each step we:
          – fade only the first-20 outcomes via w_early(t) (1 through t=40 → 0 by t=60),
          – apply curve-guided improvement (λ,δ) by adding ONLY virtual successes.
    The recorded 'Predicted has_loss Probability' at launch t is the PRIOR mean for t.
    """
    # Guards
    required = {grouping_col, attempt_column, "has_loss", "launch_date", "seradata_spacecraft_id"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    # Global p1 and single “All-losses” curve
    p1_global = df[df[attempt_column] == 1]["has_loss"].mean()
    if pd.isna(p1_global):
        p1_global = df["has_loss"].mean()
    lam = float(lambda_set.get(0))
    delt = float(delta_set.get(0))

    rows: list[Dict[str, Any]] = []

    for vehicle in df[grouping_col].dropna().unique():
        sub = (df.loc[df[grouping_col] == vehicle]
                 .sort_values([attempt_column, "launch_date"], na_position="last"))
        if sub.empty:
            continue

        launches = sub[attempt_column].to_numpy().astype(int)
        outcomes = sub["has_loss"].to_numpy().astype(int)
        dates    = sub["launch_date"].to_numpy()
        sc_ids   = sub["seradata_spacecraft_id"].to_numpy()

        pred = _LaunchBayesPredictor(lam, delt, default_weight, float(p1_global), p2_given, p3_given)
        buckets = pred._init_buckets()
        base_set = False
        last_t = 0

        # Cumulative empirical counters for this vehicle
        cum_launches = 0
        cum_failures = 0

        for i, t_obs in enumerate(launches):
            # ----- PRIOR for this observed launch -----
            if t_obs == 1:
                a_prior, b_prior = pred.prior_empirical_t1()
            elif t_obs == 2:
                a_prior, b_prior = pred.prior_empirical_t2(launches, outcomes)
            elif t_obs == 3:
                a_prior, b_prior = pred.prior_empirical_t3(launches, outcomes)
                if not base_set:
                    # set the one-time base anchor at t=3
                    p3 = pred.empirical_p3_anchor(launches, outcomes)
                    buckets["a_base"] = default_weight * p3
                    buckets["b_base"] = default_weight * (1 - p3)
                    base_set = True
            else:
                if not base_set:
                    # Anchor to empirical t=3 even if t3 wasn’t observed
                    p3 = pred.empirical_p3_anchor(launches, outcomes)
                    buckets["a_base"] = default_weight * p3
                    buckets["b_base"] = default_weight * (1 - p3)
                    base_set = True
                # Step through gaps (and the current t_obs) applying curve nudges
                last_t = pred.advance_to(buckets, last_t, t_obs)
                a_prior, b_prior = pred._compose_prior_counts(buckets, t_obs)

            # Record PRIOR stats for t_obs
            p_mean  = a_prior / (a_prior + b_prior)
            p_lower = _beta.ppf((1 - ci) / 2, a_prior, b_prior)
            p_upper = _beta.ppf(1 - (1 - ci) / 2, a_prior, b_prior)

            # ----- Update cumulative empirical rate (includes this observed outcome) -----
            cum_launches += 1
            outcome_i = int(outcomes[i])
            cum_failures += outcome_i
            cumulative_rate = cum_failures / cum_launches  # empirical running mean


            rows.append({
                grouping_col: vehicle,
                "launch_number": int(t_obs),
                "launch_date": dates[i],
                "observed_has_loss": int(outcomes[i]),
                "cumulative_rate": float(cumulative_rate),
                "predicted": float(p_mean),
                "ci_lower": float(p_lower),
                "ci_upper": float(p_upper),
                "spacecraft_id": sc_ids[i],
            })

            # ----- POSTERIOR update with observed outcome -----
            pred._posterior_update(buckets, t_obs, int(outcomes[i]))
            last_t = max(last_t, t_obs)

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
    decay_tau: float = 10,  # unused (fade handled by w_early with curve)
    grouping_col: str = "vehicle_type",
    attempt_column: str = "lv_family_attempt_number",
) -> pd.DataFrame:
    """
    One-step-ahead per vehicle:
      • If next ∈ {1,2,3}: use the fully empirical priors directly.
      • If next ≥ 4: anchor to empirical t=3; process observed launches with
        posterior updates; advance to next_launch with curve nudges (fading only
        the first 20 outcomes); read PRIOR at next_launch and report its mean/CI.
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

    for vehicle in df[grouping_col].dropna().unique():
        sub = df[df[grouping_col] == vehicle].sort_values([attempt_column, "launch_date"], na_position="last")
        if sub.empty:
            continue

        launches = sub[attempt_column].to_numpy().astype(int)
        outcomes = sub["has_loss"].to_numpy().astype(int)
        dates    = sub["launch_date"].to_numpy()

        next_launch = (int(launches.max()) + 1) if len(launches) else 1
        pred = _LaunchBayesPredictor(lam, delt, default_weight, float(p1_global), p2_given, p3_given)

        # Next in {1,2,3}: purely empirical right away
        if next_launch == 1:
            a_next, b_next = pred.prior_empirical_t1()
        elif next_launch == 2:
            a_next, b_next = pred.prior_empirical_t2(launches, outcomes)
        elif next_launch == 3:
            a_next, b_next = pred.prior_empirical_t3(launches, outcomes)
        else:
            # Build buckets from observed launches, with anchor at t=3
            buckets = pred._init_buckets()
            base_set = False
            last_t = 0

            for i, t_obs in enumerate(launches):
                if t_obs == 3 and not base_set:
                    p3 = pred.empirical_p3_anchor(launches, outcomes)
                    buckets["a_base"] = default_weight * p3
                    buckets["b_base"] = default_weight * (1 - p3)
                    base_set = True
                elif t_obs > 3 and not base_set:
                    # in case there's no explicit t=3 row
                    p3 = pred.empirical_p3_anchor(launches, outcomes)
                    buckets["a_base"] = default_weight * p3
                    buckets["b_base"] = default_weight * (1 - p3)
                    base_set = True

                # Advance (curve only) across any gaps before observing t_obs
                last_t = pred.advance_to(buckets, last_t, t_obs)
                # Observe t_obs → posterior update
                pred._posterior_update(buckets, t_obs, int(outcomes[i]))
                last_t = max(last_t, t_obs)

            if not base_set:
                p3 = pred.empirical_p3_anchor(launches, outcomes)
                buckets["a_base"] = default_weight * p3
                buckets["b_base"] = default_weight * (1 - p3)
                base_set = True
                last_t = max(last_t, 3)

            # Advance to next_launch (curve only) and read PRIOR there
            last_t = pred.advance_to(buckets, last_t, next_launch)
            a_next, b_next = pred._compose_prior_counts(buckets, next_launch)

        p_mean  = a_next / (a_next + b_next)
        p_lower = _beta.ppf((1 - ci) / 2, a_next, b_next)
        p_upper = _beta.ppf(1 - (1 - ci) / 2, a_next, b_next)

        results.append({
            grouping_col: vehicle,
            "next_launch_number": int(next_launch),
            "launch_date": dates[-1] if len(dates) else pd.NaT,
            "total_failures": int(np.nansum(outcomes)),
            "failure_rate": float(p_mean),
            "ci_lower": float(p_lower),
            "ci_upper": float(p_upper),
        })

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
                clean_launch_data, attempt_column
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
            ci=0.95, grouping_col=grouping_col, attempt_column=attempt_column
        )

        full_history_df = predict_has_loss_probabilities_all_launches_with_empirical(
            clean_launch_data,
            lambda_set=lambda_set, delta_set=delta_set,
            p2_given=second_launch_fr, p3_given=third_launch_fr,
            alpha0_set=alpha0_set, beta0_set=beta0_set,
            grouping_col=grouping_col, attempt_column=attempt_column
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-outputs", action="store_true", help="Write intermediate audit artifacts")
    args = parser.parse_args()
    main(audit_outputs=args.audit_outputs)
