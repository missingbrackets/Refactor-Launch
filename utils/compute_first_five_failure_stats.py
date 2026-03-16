import pandas as pd
def compute_first_five_failure_stats(df, group_col, attempt_col, loss_col='Has_Loss'):
    df = df.sort_values([group_col, attempt_col]).copy()

    # Create columns to fill in
    failure_flag = []
    failure_count = []

    # Process each group individually
    for _, group in df.groupby(group_col):
        attempts = group[attempt_col].values
        losses = group[loss_col].values

        prior_failures = []
        prior_flag = []

        for i in range(len(group)):
            current_attempt = attempts[i]
            # Get indices of earlier launches with attempt number ≤ 5 and attempt number < current
            valid_prior = (attempts < current_attempt) & (attempts <= 5)
            prior_losses = losses[valid_prior]

            prior_failures.append(prior_losses.sum())
            prior_flag.append(1 if prior_losses.sum() > 0 else 0)

        failure_flag.extend(prior_flag)
        failure_count.extend(prior_failures)

    df['Failure_Count_First_Five'] = failure_count
    df['First_Five_Failure'] = failure_flag

    return df