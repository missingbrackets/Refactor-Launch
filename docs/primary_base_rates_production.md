# Primary Base Rates Production Pipeline

## Overview

`production/primary_base_rates_production_w_audit.py` computes **per-vehicle launch failure probabilities** using a Bayesian learning-curve model. It processes each hierarchy level (vehicle type, family, provider, variant, minor variant), fits empirical priors and a parametric decay curve, and produces next-launch predictions and full launch-history predictions.

An optional **audit trail** writes intermediate computation tables (CSV/XLSX) at every stage for reproducibility and debugging.

---

## Pipeline Flow

```
main(audit_outputs)
│
├── 1. Load data
│   └── load_and_prepare_data(filter_min_year_on=False)
│       → Returns fully-featured launch DataFrame
│
├── 2. For each grouping level (GROUPINGS):
│   │
│   └── _process_grouping(launch_df, grouping_col, ...)
│       │
│       ├── 2a. select_grouping_columns()
│       │   → Selects relevant columns + attempt column for this level
│       │
│       ├── 2b. compute_empirical_conditionals()
│       │   → p1: P(fail | launch 1)
│       │   → p2_given: P(fail | launch 2, outcome of launch 1)
│       │   → p3_given: P(fail | launch 3, fail count in launches 1-2)
│       │   [audit: writes empirical/00..09 tables]
│       │
│       ├── 2c. fit_learning_curve()
│       │   → Fits bayesian_learning_curve() via scipy curve_fit
│       │   → Returns: alpha, beta, lambda, delta, prior_weight
│       │   [audit: writes learning_curve/00..08 tables]
│       │
│       ├── 2d. predict_next_failure_probability_per_vehicle()
│       │   → One row per vehicle: predicted failure rate for NEXT launch
│       │   → t=1,2,3: uses empirical priors
│       │   → t>=4: anchors at t=3, updates with observations,
│       │     advances via learning curve
│       │   [audit: writes next_failure/<vehicle>/00..03 tables]
│       │
│       ├── 2e. predict_has_loss_probabilities_all_launches_with_empirical()
│       │   → One row per observed launch: predicted failure rate AT that launch
│       │   → Same Bayesian logic as 2d but records every step
│       │   [audit: writes per_launch/<vehicle>/01..02 tables]
│       │
│       ├── 2f. build_dropdown_rows_for_grouping()
│       │   → Aggregates identity columns, attempt maxes, snapshot values
│       │
│       └── 2g. Export CSV + JSON per grouping
│
└── 3. _build_unified_dropdown()
    ├── Concatenate all dropdown rows
    ├── Merge specific primary rates (per grouping/value)
    ├── Merge type primary rates (by vehicle_type)
    ├── coerce_dropdown_schema() → enforce column types
    └── Export unified dropdown CSV + JSON
```

---

## Key Components

### Bayesian Learning Curve

```
P(fail | t) = alpha_prior / (alpha_prior + beta_prior + (lambda * t)^delta)
```

- **alpha_prior** = prior_weight * fail_rate
- **beta_prior** = prior_weight * (1 - fail_rate)
- Fitted via `scipy.optimize.curve_fit` with bounded parameters

### _LaunchBayesPredictor

Core prediction engine that manages:

| Component | Purpose |
|-----------|---------|
| `g(t)` | Learning curve shape: `1 / (1 + (lambda * (t-1))^delta)` |
| `step_factor(t)` | Ratio `g(t)/g(t-1)`, clamped to [0,1] |
| `w_early(t)` | Weight for early observations: 1.0 at t<=40, fades to 0.0 by t=60 |
| Buckets | `a_early/b_early` (t<=20), `a_late/b_late` (t>20), `a_base/b_base` (t=3 anchor), `b_curve` (virtual successes from curve) |

**Prediction logic:**

1. **t=1**: Use global P(fail at launch 1)
2. **t=2**: Use P(fail | outcome of launch 1)
3. **t=3**: Anchor — set base counts from empirical P(fail | fail count in launches 1-2)
4. **t>=4**: Advance posterior from previous t via curve nudge (adds virtual successes to decrease mean), then compose prior from all buckets

### Audit Trail

When `audit_outputs=True`, the pipeline writes detailed tables at each step under `outputs/audit_YYYYMMDD/`. This includes:
- Empirical data and coverage statistics
- Curve fit parameters, residuals, covariance
- Per-vehicle event logs showing every prior/posterior update
- Final prediction summaries

---

## Outputs

| File Pattern | Content |
|---|---|
| `launch_primary_rates_{grouping}_{date}_next.csv/json` | Next-launch failure prediction per vehicle |
| `launch_primary_rates_{grouping}_{date}_full.csv/json` | Full history predictions per vehicle per launch |
| `launch_dropdown_options_all_{date}.csv/json` | Unified dropdown table with rates attached |
| `audit_{date}/` | Intermediate audit tables (when enabled) |

---

## Entry Point

```bash
python -m production.primary_base_rates_production_w_audit --audit-outputs
```

Or programmatically:

```python
from production.primary_base_rates_production_w_audit import main
main(audit_outputs=True)
```
