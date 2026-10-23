"""Model C: add the text-encoder score (model B) as a feature to the GBM, if it exists."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def attach_encoder_scores(df: pd.DataFrame, path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    path = Path(path)
    if not path.exists():
        return df, []
    enc = pd.read_csv(path)
    cols = [c for c in enc.columns if c.startswith("enc_")]
    return df.merge(enc[["id", "idx", *cols]], on=["id", "idx"], how="left"), cols
