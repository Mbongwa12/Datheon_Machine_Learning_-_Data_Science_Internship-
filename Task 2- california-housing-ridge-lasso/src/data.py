"""Dataset loading. Prefers a local copy, falls back to the public URL."""
from __future__ import annotations

import pandas as pd

from .config import DATA_URL, LOCAL_CSV


def load_housing(path: str | None = None) -> pd.DataFrame:
    """Load the California housing dataset.

    Order of preference:
      1. an explicit ``path`` argument
      2. ``data/housing.csv`` inside the repo
      3. download from the public GitHub URL
    """
    if path is not None:
        return pd.read_csv(path)
    if LOCAL_CSV.exists():
        return pd.read_csv(LOCAL_CSV)
    df = pd.read_csv(DATA_URL)
    LOCAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LOCAL_CSV, index=False)
    return df
