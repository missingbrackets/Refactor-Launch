import pandas as pd

def output_df_to_csv(df: pd.DataFrame, filename: str, table_name: str):
    """
    Output a DataFrame to a specified CSV file with UTF-8 encoding.
    """
    import os
    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Write CSV with UTF-8 encoding
    df.to_csv(filename, index=False, encoding="utf-8-sig")

    print(f"{table_name} table saved to '{filename}'")