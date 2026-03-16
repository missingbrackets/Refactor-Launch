import pandas as pd
# def add_launches_since_last_failure(
#     df, group_col, attempt_col, failure_col, output_col="Num_Launches_Since_Last_Failure"
# ):
#     """
#     Adds a column that counts how many launches of the same group (e.g., type, family, provider)
#     have occurred since the last failure.

#     Parameters:
#         df (pd.DataFrame): The input dataframe.
#         group_col (str): The column indicating the group (e.g., Vehicle_Type, Vehicle_Family).
#         attempt_col (str): The attempt number column within the group.
#         failure_col (str): The binary column indicating a failure (1 = failure, 0 = success).
#         output_col (str): Name of the output column to be added.

#     Returns:
#         pd.DataFrame: A copy of the input DataFrame with the new column added.
#     """
#     df = df.copy()
#     df.sort_values(by=[group_col, attempt_col], inplace=True)

#     result = []
#     last_failure_index = {}

#     for idx, row in df.iterrows():
#         group = row[group_col]
#         attempt_num = row[attempt_col]

#         if group not in last_failure_index:
#             count_since = attempt_num  # No previous failure
#         else:
#             count_since = attempt_num - last_failure_index[group]

#         result.append(count_since)

#         if row[failure_col] == 1:
#             last_failure_index[group] = attempt_num

#     df[output_col] = result
#     return df


# def add_launches_since_last_failure(
#     df, group_col, attempt_col, failure_col, output_col="Num_Launches_Since_Last_Failure"
# ):
#     df = df.copy()
#     df.sort_values(by=[group_col, attempt_col], inplace=True)

#     running = {}
#     out = []

#     for _, row in df.iterrows():
#         g = row[group_col]
#         if row[failure_col] == 1:
#             running[g] = 0              # failure → zero clean launches
#         else:
#             running[g] = running.get(g, 0) + 1
#         out.append(running[g])

#     df[output_col] = out
#     return df

def add_launches_since_last_failure(
    df,
    group_col,
    attempt_col,
    failure_col,
    output_col="Num_Launches_Since_Last_Failure",
):
    """
    Vectorized: counts successive clean launches since last failure per group.
    - On a failure row -> 0
    - First success after a failure -> 1
    - Works regardless of attempt numbering gaps (just needs correct ordering)
    """
    # Work on a sorted copy (order matters for "since last")
    df_sorted = df.sort_values([group_col, attempt_col]).copy()

    # Ensure binary 0/1 (treat truthy as 1)
    fail = df_sorted[failure_col].astype(int)

    # Block id increases each time a failure happens within a group
    # (rows before any failure are in block 0, the first failure row is in block 1,
    #  rows after that failure are also in block 1, etc.)
    block = fail.groupby(df_sorted[group_col]).cumsum()

    # Count successes within each (group, block). Fail rows contribute 0.
    success = (1 - fail)
    streak = success.groupby([df_sorted[group_col], block]).cumsum()

    # Failure rows should be 0 explicitly (already 0, but let's be clear)
    streak = streak.where(fail.eq(0), 0).astype(int)

    df_sorted[output_col] = streak

    # Return in original row order
    return df_sorted.sort_index()
