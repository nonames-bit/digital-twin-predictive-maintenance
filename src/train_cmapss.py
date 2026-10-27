from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "datasets" / "cmapss"
MODELS_DIR = BASE_DIR / "models"
sys.path.insert(0, str(DATA_DIR))

from adapter import load


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula el S-score asimétrico de NASA PCoE (penaliza más las predicciones tardías)."""
    d = y_pred - y_true
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def evaluate_predictions(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    print(f"[{name}] RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def train_and_evaluate(subset: str = "FD001", rul_cap: int = 125, save_model: bool = True):
    df = load(str(DATA_DIR), rul_cap=rul_cap, subset=subset, split="train")

    X = pd.DataFrame(df["features"].tolist())  # setting1..3 y s1..s21 como columnas
    y = df["target"]                            # el RUL

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups=df["asset_id"]))

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    if save_model:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / f"xgb_cmapss_{subset.lower()}.json"
        model.save_model(str(model_path))

    # Evaluación en el conjunto de validación (por motor)
    val_pred = model.predict(X_val)
    val_metrics = evaluate_predictions("Validación (GroupSplit 20%)", y_val.to_numpy(), val_pred)

    # Evaluación en el conjunto oficial de prueba (test_FD001 + RUL_FD001)
    test_df = load(str(DATA_DIR), rul_cap=rul_cap, subset=subset, split="test")
    X_test = pd.DataFrame(test_df["features"].tolist())
    y_test = test_df["target"].to_numpy()
    test_pred = model.predict(X_test)
    test_all_metrics = evaluate_predictions("Test Oficial (todos los ciclos)", y_test, test_pred)

    # Evaluación estándar C-MAPSS: último ciclo registrado por cada motor en test
    last_cycle_idx = test_df.groupby("asset_id", sort=False).tail(1).index
    y_test_last = test_df.loc[last_cycle_idx, "target"].to_numpy()
    test_pred_last = test_pred[last_cycle_idx]
    metrics_last = evaluate_predictions("Test Oficial (último ciclo por motor)", y_test_last, test_pred_last)
    s_score = nasa_score(y_test_last, test_pred_last)
    metrics_last["nasa_s_score"] = s_score
    print(f"[Test Oficial (último ciclo por motor)] NASA S-Score: {s_score:.2f}")

    return model, {
        "validation": val_metrics,
        "test_all_cycles": test_all_metrics,
        "test_last_cycle": metrics_last,
    }


def main():
    return train_and_evaluate(subset="FD001", rul_cap=125, save_model=True)


if __name__ == "__main__":
    main()
