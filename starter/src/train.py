"""XGBoost baseline on the AI4I 2020 dataset.

Design choices (see README):
- TWF/HDF/PWF/OSF/RNF are dropped as features: they are the components of the
  target, so keeping them is target leakage.
- SMOTE is applied INSIDE each CV fold via an imblearn Pipeline, never before
  the split — oversampling before splitting leaks test rows into training.
- Judged on recall / PR-AUC, not accuracy (3.4% positives).
- All runs logged to MLflow; best model registered as "twin-xgb-baseline".

Usage: python src/train.py   (needs ../ai4i2020.csv one level above starter/)
"""
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

load_dotenv()
if os.getenv("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

DATA = Path(__file__).resolve().parent.parent.parent / "ai4i2020.csv"
TARGET = "Machine failure"
LEAKAGE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]  # components of the target
ID_COLS = ["UID", "Product ID"]

mlflow.set_experiment("digital-twin-xgb-baseline")


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    df = df.drop(columns=ID_COLS + LEAKAGE_COLS)
    # Engineered features (physics-informed, no leakage)
    df["temp_diff"] = df["Process temperature"] - df["Air temperature"]
    df["power_w"] = df["Torque"] * df["Rotational speed"] * 2 * np.pi / 60
    # Synthetic minute-sampled timestamp (dataset has no time column)
    df["timestamp"] = pd.Timestamp("2026-01-01") + pd.to_timedelta(df.index, unit="min")
    return df


def main():
    df = load()
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET, "timestamp"])
    print(f"rows={len(df)} failures={int(y.sum())} ({y.mean():.2%})")

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    preprocess = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num_cols),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])

    clf = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(y == 0).sum() / y.sum(),
        tree_method="hist", random_state=42, eval_metric="logloss",
    )

    pipe = ImbPipeline([
        ("preprocess", preprocess),
        ("smote", SMOTE(random_state=42)),   # inside folds only
        ("clf", clf),
    ])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    with mlflow.start_run(run_name="xgb-baseline-v1"):
        mlflow.log_params({f"xgb_{k}": v for k, v in clf.get_params().items()})
        mlflow.log_param("smote", "inside_folds")

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_validate(pipe, X_tr, y_tr, cv=cv,
                                scoring={"pr_auc": "average_precision",
                                         "recall": "recall"})
        for name in scores:
            if name.startswith("test_"):
                mlflow.log_metric(f"cv_{name[5:]}_mean", float(scores[name].mean()))
                print(f"cv {name[5:]:8s} = {scores[name].mean():.4f} ± {scores[name].std():.4f}")

        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        metrics = {
            "test_pr_auc": average_precision_score(y_te, proba),
            "test_recall": recall_score(y_te, pred),
            "test_precision": precision_score(y_te, pred),
            "test_f1": f1_score(y_te, pred),
        }
        mlflow.log_metrics(metrics)
        print({k: round(v, 4) for k, v in metrics.items()})
        mlflow.sklearn.log_model(pipe, "model",
                                 registered_model_name="twin-xgb-baseline")


if __name__ == "__main__":
    main()
