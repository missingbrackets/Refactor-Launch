# Fallback Base Rates Production Pipeline

## Overview

`production/fallback_rates_production_new.py` fits **fallback failure-rate curves** using a GLM + isotonic regression model. These rates serve as baseline priors when vehicle-specific data is insufficient for the primary Bayesian model.

The pipeline operates at five hierarchy levels (type, family, provider, variant, minor variant) and produces:
- **Base rates**: overall failure curves per level
- **Modifier 1 (iteration)**: separate curves for first vs. second+ variants within a parent
- **Modifier 2 (LSF)**: separate curves by launches-since-last-failure bin (Clean, 1-3, >=4)
- **Provider rating comparison table**

---

## Pipeline Flow

```
main()
│
├── 1. Load data
│   └── load_and_prepare_data()
│       → Returns fully-featured launch DataFrame
│       → Includes SinceFail_Bin columns (from add_binned_reliability_combo)
│       → Includes iteration_grouped columns (from add_iteration_within)
│
├── 2. Fit fallback base rates
│   └── _base_rates(launch_df)
│       │
│       └── For each level in BASE_RATE_LEVELS:
│           └── _fit_base_for_level(df, attempt_col, level_name)
│               └── _fit_single_level()
│                   └── _safe_fit_rates_model()
│                       └── fit_rates_model()  [utils]
│                           ├── Quadratic binomial GLM
│                           ├── Isotonic regression (monotone decreasing)
│                           └── Fallback: empirical rates + isotonic
│
├── 3. Fit Modifier 1 – Iteration (first vs second+)
│   └── For each level in ITERATION_LEVELS:
│       └── _modifier_iteration_for_level(df, attempt_col, grouped_col, label)
│           ├── Filter: df[grouped_col] == "first"
│           ├── Filter: df[grouped_col] != "first" (second+)
│           ├── Fit each slice via _safe_fit_rates_model()
│           └── Merge on attempt_number → columns: {label}_first, {label}_second_plus
│
├── 4. Fit Modifier 2 – Launches Since Last Failure
│   └── For each level in LSF_LEVELS:
│       └── _modifier_lsf_for_level(df, attempt_col, bin_col, label)
│           └── For each bin in LSF_BINS: ("Clean", "1–3", ">=4")
│               └── _fit_lsf_bin_for_level()
│                   ├── Filter df on bin_col == bin_value
│                   ├── Normalize dash variants ('1-3' → '1–3')
│                   └── Fit via _safe_fit_rates_model()
│           └── Merge all bins → columns: {label}_clean, {label}_1_3, {label}_ge4
│
├── 5. Provider Rating Comparison Table
│   └── provider_rating_comparison_table()  [utils]
│       → Compares failure rates across provider ratings at groupings (3, 10, 20)
│
└── 6. Export all outputs
    ├── _export_fallback_rates()
    │   ├── Per-level CSV/JSON: launch_fallback_rates_{level}_{date}.csv/json
    │   └── Combined long table: launch_fallback_rates_all_levels_{date}.csv/json
    ├── _export_modifier_tables(modifiers_iter, "iter", date_tag)
    │   └── launch_modifier_iter_{level}_{date}.csv/json
    ├── _export_modifier_tables(modifiers_lsf, "lsf", date_tag)
    │   └── launch_modifier_lsf_{level}_{date}.csv/json
    ├── Full data snapshot: launch_full_data_{date}.csv/json
    └── Provider table: launch_provider_rating_comparison_table_{date}.csv/json
```

---

## Key Components

### fit_rates_model (utils)

The core model fitting function in `utils/fit_fallback_base_rate.py`:

1. **Primary**: Fits a quadratic binomial GLM (`has_loss ~ attempt + attempt^2`)
2. **Monotonicity**: Applies isotonic regression to enforce decreasing failure rates
3. **Fallback**: If GLM fails, uses raw empirical rates + isotonic regression

### Level Configuration

Defined as module-level constants for clarity:

```python
BASE_RATE_LEVELS = [
    ("type", "lv_type_attempt_number"),
    ("family", "lv_family_attempt_number"),
    ...
]

ITERATION_LEVELS = [
    ("type", "lv_type_attempt_number", "type_iteration_grouped"),
    ...
]

LSF_LEVELS = [
    ("type", "lv_type_attempt_number", "type_SinceFail_Bin"),
    ...
]

LSF_BINS = [("Clean", "clean"), ("1–3", "1_3"), (">=4", "ge4")]
```

### Modifier Logic

**Modifier 1 (Iteration):** Splits data by whether a vehicle variant is the "first" within its parent group or a subsequent one. Produces separate failure curves for each.

**Modifier 2 (LSF):** Splits data by the launches-since-last-failure bin:
- **Clean**: No failures in history (since_fail == attempt number)
- **1–3**: 1 to 3 launches since the last failure
- **>=4**: 4 or more launches since last failure

---

## Outputs

| File Pattern | Content |
|---|---|
| `launch_fallback_rates_{level}_{date}.csv/json` | Base rate curve for one level |
| `launch_fallback_rates_all_levels_{date}.csv/json` | All levels in one long table |
| `launch_modifier_iter_{level}_{date}.csv/json` | Iteration modifier curves |
| `launch_modifier_lsf_{level}_{date}.csv/json` | LSF modifier curves |
| `launch_full_data_{date}.csv/json` | Complete input data snapshot |
| `launch_provider_rating_comparison_table_{date}.csv/json` | Provider rating comparison |

---

## Data Dependencies

This pipeline requires `load_and_prepare_data()` to produce columns including:
- `lv_{level}_attempt_number` — sequential attempt numbers per grouping
- `{prefix}_SinceFail_Bin` — binned launches-since-last-failure (from `add_binned_reliability_combo`)
- `{prefix}_iteration_grouped` — "first" / "not_first" labels (from `add_iteration_within`)
- `has_loss` — binary failure indicator

---

## Entry Point

```python
from production.fallback_rates_production_new import main
main()
```

Or as a script:
```bash
python -m production.fallback_rates_production_new
```
