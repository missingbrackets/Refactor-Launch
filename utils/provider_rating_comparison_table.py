import pandas as pd
from typing import Iterable, Optional, Union, List

def provider_rating_comparison_table(
    df: pd.DataFrame,
    max_attempt: int,
    exclude_ratings: Optional[Iterable[Union[str, int]]] = None,
    groupings: Iterable[int] = (1, 2, 5, 10, 20),
) -> pd.DataFrame:
    """
    Bin rows by lv_type_attempt_number using the provided 'groupings' thresholds,
    then aggregate by provider_rating and the bin.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: 'provider_rating', 'lv_type_attempt_number', 'has_loss'
    max_attempt : int
        Keep only rows where lv_type_attempt_number <= max_attempt
    exclude_ratings : iterable, optional
        Provider ratings to exclude entirely
    groupings : iterable of int
        Monotonic increasing list/tuple of upper bounds for attempt bins, e.g. (1,2,5,10,20)

    Returns
    -------
    pd.DataFrame
        Columns: provider_rating, attempt_bin, count, losses, failure_rate
    """
    # Basic validation & prep
    if not isinstance(groupings, (list, tuple)):
        groupings = list(groupings)
    groupings = sorted(set(int(x) for x in groupings))  # ensure sorted & unique ints

    # Filter by max_attempt
    output_df = df[df["lv_type_attempt_number"] <= max_attempt].copy()

    # Optionally exclude ratings
    if exclude_ratings is not None:
        output_df = output_df[~output_df["provider_rating"].isin(exclude_ratings)]

    # Ensure has_loss is numeric (0/1) to avoid surprises
    if output_df["has_loss"].dtype == bool:
        output_df["has_loss"] = output_df["has_loss"].astype(int)

    # Build labels for ordered categorical bins
    labels: List[str] = []
    prev = 0
    for ub in groupings:
        if prev == 0:
            labels.append(f"≤{ub}")
        else:
            labels.append(f"{prev+1}–{ub}")
        prev = ub

    # If max_attempt exceeds the last grouping, create a final overflow label
    overflow_label = None
    if max_attempt > groupings[-1]:
        # Cap the overflow at max_attempt for clarity
        overflow_label = f"{groupings[-1]+1}–{max_attempt}"
        labels.append(overflow_label)

    # Assign bins by mapping each attempt number to the first upper bound that fits
    def assign_bin(n: int) -> str:
        for i, ub in enumerate(groupings):
            if n <= ub:
                return labels[i]
        # Overflow (only possible if max_attempt > last grouping)
        return overflow_label  # type: ignore

    output_df["attempt_bin"] = output_df["lv_type_attempt_number"].astype(int).map(assign_bin)

    # Make attempt_bin an ordered categorical for nice sorting
    output_df["attempt_bin"] = pd.Categorical(output_df["attempt_bin"], categories=labels, ordered=True)

    # Group and aggregate
    agg = (
        output_df
        .groupby(["provider_rating", "attempt_bin"], as_index=False)
        .agg(
            count=("has_loss", "size"),
            losses=("has_loss", "sum"),
        )
    )
    agg["failure_rate"] = agg["losses"] / agg["count"]

    # Sort for readability
    agg = agg.sort_values(["provider_rating", "attempt_bin"]).reset_index(drop=True)
    return agg
