# src/train.py
from pathlib import Path
import joblib

import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

from .config import PROCESSED_DATA_DIR, MODELS_DIR, TARGET_COLUMN, RANDOM_STATE, LATEST_MODEL_PATH
from .model_utils import split_features_target, build_model_pipeline


def main() -> None:
    train_path = PROCESSED_DATA_DIR / "train.csv"
    valid_path = PROCESSED_DATA_DIR / "valid.csv"

    if not train_path.exists() or not valid_path.exists():
        raise FileNotFoundError("No se encontraron train.csv / valid.csv. Ejecuta primero data_prep.py")

    train_df = pd.read_csv(train_path, parse_dates=["application_date"])
    valid_df = pd.read_csv(valid_path, parse_dates=["application_date"])

    X_train, y_train = split_features_target(train_df, TARGET_COLUMN)
    X_valid, y_valid = split_features_target(valid_df, TARGET_COLUMN)

    model = build_model_pipeline(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_valid)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_valid, y_proba)
    f1 = f1_score(y_valid, y_pred)

    print(f"AUC valid: {auc:.4f}")
    print(f"F1  valid: {f1:.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, LATEST_MODEL_PATH)
    print(f"Modelo guardado en: {LATEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
