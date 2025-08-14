# normalize.py

import pandas as pd

def normalize(series):
    """Normalize a pandas Series using min-max scaling. Handles single-value edge cases."""
    if series.max() == series.min():
        return pd.Series([0.5] * len(series))
    return (series - series.min()) / (series.max() - series.min() + 1e-9)
