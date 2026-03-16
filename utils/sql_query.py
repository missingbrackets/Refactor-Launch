import pyodbc
import pandas as pd
from urllib.parse import quote_plus
from sqlalchemy import create_engine

def fetch_data(server, database, table, select_columns="All", **filters):
    # Determine columns to select
    if select_columns == "All":
        column_str = "*"
    else:
        column_str = ", ".join(f"[{col}]" for col in select_columns)
    
    # Construct filter conditions
    filter_conditions = []
    for col, value in filters.items():
        if isinstance(value, (list, tuple, set)) and len(value) > 1:
            values_str = "', '".join(str(v) for v in value)
            condition = f"  [{col}] IN ('{values_str}')"
        else:
            single_value = value if not isinstance(value, (list, tuple, set)) else value[0]
            condition = f"  [{col}] = '{single_value}'"
        filter_conditions.append(condition)

    # Assemble WHERE clause
    where_clause = ""
    if filter_conditions:
        where_clause = "\nWHERE " + "\nAND ".join(filter_conditions)

    # Final query
    query = f"SELECT {column_str}\nFROM {database}.{table} {where_clause}"
    print(query)

    # Connect to SQL Server
    # conn_str = (
    #     f"DRIVER={{SQL Server}};"
    #     f"SERVER={server};"
    #     f"DATABASE={database};"
    #     f"Trusted_Connection=yes;"
    # )
    conn_str = (
        "DRIVER={ODBC Driver 18 for SQL Server};"
        f"SERVER={server};"
        f"DATABASE={database};"
        "Trusted_Connection=yes;"
        "Encrypt=no;"           # or Encrypt=yes;TrustServerCertificate=yes
    )
    engine_uri = "mssql+pyodbc:///?odbc_connect=" + quote_plus(conn_str)
    engine = create_engine(engine_uri)

    try:
        return pd.read_sql(query, engine)
    finally:
        engine.dispose()
    # conn = pyodbc.connect(conn_str)
    # result = pd.read_sql(query, conn)
    # conn.close()
    # return result
