import pandas as pd
def add_months_since_last_launch(
    df,
    group_col,
    date_col='Launch_Date',
    output_col='months_since_last_launch',
    default_months=36
):
    """
    Adds a column that calculates the number of months since the last launch
    for each group (e.g., Vehicle_Type, Vehicle_Family, Launch_Provider).

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing launch records.
    group_col : str
        Column name to group by (e.g., 'Vehicle_Type').
    date_col : str
        Name of the launch date column.
    output_col : str
        Name of the output column to store months since last launch.
    default_months : float
        Default value if no previous launch exists (e.g., for first launch in group).

    Returns:
    --------
    pd.DataFrame
        DataFrame with an added column indicating months since last launch per group.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df.sort_values(by=[group_col, date_col], inplace=True)

    df[output_col] = (
        df.groupby(group_col)[date_col]
        .diff()
        .apply(lambda x: x.days / 30.44 if pd.notnull(x) else default_months)
    )

    return df