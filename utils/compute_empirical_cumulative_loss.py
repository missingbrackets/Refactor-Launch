def compute_empirical_cumulative_has_loss(df, attempt_column):
    """
    Computes empirical cumulative has_loss probability per attempt number.
    Skips launches before attempt #3 (too early for learning dynamics).
    
    Returns:
    - DataFrame with columns: Attempt Number, Empirical has_loss Probability
    """
    df = df[df[attempt_column] >= 3].copy()
    df = df.sort_values(attempt_column)
    
    grouped = df.groupby(attempt_column)
    cumulative_has_losss = grouped['has_loss'].sum().cumsum()
    cumulative_counts = grouped['has_loss'].count().cumsum()
    
    empirical_probs = (cumulative_has_losss / cumulative_counts).reset_index()
    empirical_probs.columns = [attempt_column, 'Empirical_has_loss_Probability']
    
    return empirical_probs