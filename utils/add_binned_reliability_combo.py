import pandas as pd
def add_binned_reliability_combo(
    df,
    attempt_col,
    since_fail_col,
    launch_bins,
    since_fail_bins=None,
    bin_labels=None,
    min_width_label_as_number=True,
    prefix="Type",
):
    """
    Bin reliability history by group (Type, Family, Provider, etc.).

    Pure function: DOES NOT modify `attempt_col` or `since_fail_col`.
    Writes only:
      - f"{prefix}_Attempt_Bin"
      - f"{prefix}_SinceFail_Bin"
      - f"{prefix}_Attempt_SinceFail_BinCombo"
    """
    import numpy as np
    import pandas as pd

    out = df.copy()

    if since_fail_bins is None:
        since_fail_bins = launch_bins

    # --- Labels -------------------------------------------------------------
    if bin_labels is None:
        labels = []
        for lo, hi in zip(launch_bins[:-1], launch_bins[1:]):
            if hi == float("inf"):
                label = f">={int(lo)}"
            else:
                width = hi - lo
                label = f"{int(lo)}" if (min_width_label_as_number and width == 1) else f"{int(lo)}–{int(hi-1)}"
            labels.append(label)
    else:
        labels = list(bin_labels)

    # --- Read-only views (no mutation of source cols) -----------------------
    # Use numerics for binning; do NOT assign back to the original columns.
    attempt_vals = pd.to_numeric(out[attempt_col], errors="coerce")
    since_vals = pd.to_numeric(out[since_fail_col], errors="coerce")

    # --- Bin attempt numbers -------------------------------------------------
    attempt_bin_col = f"{prefix}_Attempt_Bin"
    attempt_bins = pd.cut(
        attempt_vals,
        bins=launch_bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )
    out[attempt_bin_col] = attempt_bins

    # --- Bin 'since last failure' -------------------------------------------
    since_fail_bin_col = f"{prefix}_SinceFail_Bin"
    since_bins = pd.cut(
        since_vals.fillna(-1),
        bins=since_fail_bins,
        labels=labels,
        right=False,
        include_lowest=True,
    ).astype("category")

    # Mark Clean where since_fail == attempt (exact equality per your docstring)
    clean_mask = since_vals.eq(attempt_vals)
    if "Clean" not in list(since_bins.cat.categories):
        since_bins = since_bins.cat.add_categories(["Clean"])
    since_bins = since_bins.where(~clean_mask, "Clean")
    out[since_fail_bin_col] = since_bins

    # --- Build combo label ---------------------------------------------------
    combo_col = f"{prefix}_Attempt_SinceFail_BinCombo"
    out[combo_col] = (
        "L:" + out[attempt_bin_col].astype(str)
        + "_SF:" + np.where(clean_mask, "Clean", out[since_fail_bin_col].astype(str))
    )

    # Optional safety: prove we didn't touch the inputs
    # assert out[since_fail_col].equals(df[since_fail_col])
    # assert out[attempt_col].equals(df[attempt_col])

    return out
