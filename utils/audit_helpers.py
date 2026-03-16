# === Audit helpers (CSV/Excel only) =======================================
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Any
import re

# choose output format: "csv" (default) or "xlsx"
AUDIT_FORMAT = "csv"  # change to "xlsx" if you want Excel files

def _ensure_parent(path: Path) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

def _to_native(x: Any):
    # make numpy/scalars serializable
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x

def _normalize_table(obj: Any) -> pd.DataFrame:
    """
    Turn a dict / list / tuple / scalar into a small table for CSV/XLSX.
    - dict -> two columns: key, value
    - list/tuple/set -> one column: value (with index as order)
    - scalar -> one-row, one-column table with column 'value'
    - pandas Series/DataFrame -> returned as DataFrame (copy)
    """
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if isinstance(obj, pd.Series):
        return obj.to_frame(name=obj.name or "value").reset_index()
    if isinstance(obj, dict):
        rows = [(str(k), _to_native(v)) for k, v in obj.items()]
        return pd.DataFrame(rows, columns=["key", "value"])
    if isinstance(obj, (list, tuple, set)):
        rows = [(_to_native(v),) for v in obj]
        return pd.DataFrame(rows, columns=["value"])
    # scalar or other
    return pd.DataFrame([(_to_native(obj),)], columns=["value"])

def _with_ext(path: Path) -> Path:
    if AUDIT_FORMAT == "xlsx":
        return path.with_suffix(".xlsx")
    return path.with_suffix(".csv")

def audit_table(enabled: bool, path: Path, obj: Any, *, head: int | None = 200) -> None:
    """
    Save obj as a *table* in CSV/XLSX. If obj is a DataFrame, optionally limit rows via head.
    """
    if not enabled or path is None:
        return
    _ensure_parent(path)
    df = _normalize_table(obj)
    if head is not None and isinstance(obj, pd.DataFrame):
        df = df.head(head)
    out_path = _with_ext(path)
    if AUDIT_FORMAT == "xlsx":
        # single-sheet workbook, sheet name from file stem (Excel-safe)
        with pd.ExcelWriter(out_path, engine="xlsxwriter") as xw:
            df.to_excel(xw, sheet_name="data", index=False)
    else:
        df.to_csv(out_path, index=False)

def audit_schema(enabled: bool, path: Path, df: pd.DataFrame) -> None:
    if not enabled or path is None:
        return
    schema = pd.DataFrame(
        [(col, str(dtype)) for col, dtype in df.dtypes.items()],
        columns=["column", "dtype"],
    )
    audit_table(enabled, path, schema, head=None)


def _safe_name(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s[:100] if len(s) > 100 else s
