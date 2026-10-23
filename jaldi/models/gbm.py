from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
}


def train_gbm(train: pd.DataFrame, val: pd.DataFrame, features: list[str], label: str,
              seed: int, num_rounds: int = 600) -> lgb.Booster:
    params = {**PARAMS, "seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed,
              "num_threads": 2}
    dtr = lgb.Dataset(train[features], train[label])
    dva = lgb.Dataset(val[features], val[label], reference=dtr)
    return lgb.train(params, dtr, num_rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


def predict(model: lgb.Booster, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    return model.predict(df[features], num_iteration=model.best_iteration)
