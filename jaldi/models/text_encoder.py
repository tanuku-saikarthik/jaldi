"""Model B (GPU, optional): fine-tune a small multilingual encoder on partial text -> P(safe).

Writes data/enc_scores.csv with columns id, idx, enc_l1, enc_l2.
Train rows get out-of-fold scores (3 folds grouped by turn) so model C does not see
scores the encoder was trained on. Val/test rows get scores from a model trained on all train rows.

Not exercised in the CPU smoke run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _fit_predict(train_df, pred_dfs, label, model_name, epochs, seed, device):
    import torch  # noqa: PLC0415
    from torch.utils.data import DataLoader  # noqa: PLC0415
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: PLC0415

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    texts = list(train_df["text"])
    ys = torch.tensor(train_df[label].values)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    idx = list(range(len(texts)))
    model.train()
    for _ in range(epochs):
        for batch in DataLoader(idx, batch_size=64, shuffle=True):
            b = list(batch.numpy())
            enc = tok([texts[i] for i in b], padding=True, truncation=True, max_length=64,
                      return_tensors="pt").to(device)
            loss = model(**enc, labels=ys[b].to(device)).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
    model.eval()
    outs = []
    with torch.no_grad():
        for df in pred_dfs:
            probs = []
            t = list(df["text"])
            for s in range(0, len(t), 256):
                enc = tok(t[s:s + 256], padding=True, truncation=True, max_length=64,
                          return_tensors="pt").to(device)
                probs.append(torch.softmax(model(**enc).logits, -1)[:, 1].float().cpu().numpy())
            outs.append(np.concatenate(probs) if probs else np.array([]))
    return outs


def train_text_encoder(rows: pd.DataFrame, out_path: str, model_name: str = "google/muril-base-cased",
                       epochs: int = 2, seed: int = 13, device: str = "cuda") -> None:
    """rows: id, idx, split, text, l1, l2"""
    rng = np.random.default_rng(seed)
    train = rows[rows.split == "train"].copy()
    rest = rows[rows.split != "train"].copy()
    ids = train["id"].unique()
    fold_of = dict(zip(ids, rng.integers(0, 3, len(ids))))
    train["fold"] = train["id"].map(fold_of)
    for label in ("l1", "l2"):
        col = f"enc_{label}"
        train[col] = np.nan
        for f in range(3):
            tr, ho = train[train.fold != f], train[train.fold == f]
            (p,) = _fit_predict(tr, [ho], label, model_name, epochs, seed, device)
            train.loc[ho.index, col] = p
        (p,) = _fit_predict(train, [rest], label, model_name, epochs, seed, device)
        rest[col] = p
    pd.concat([train, rest])[["id", "idx", "enc_l1", "enc_l2"]].to_csv(out_path, index=False)
