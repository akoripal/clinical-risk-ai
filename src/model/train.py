import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
import pickle

PROCESSED_PATH = Path("data/processed/diabetic_clean.csv")
MODEL_DIR = Path("data/processed")

FEATURES = [
    "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient",
    "number_diagnoses", "medication_burden", "diagnostic_complexity",
    "is_emergency", "age_numeric"
]

TARGET = "prolonged_stay"

def load_features() -> tuple:
    df = pd.read_csv(PROCESSED_PATH)
    df = df.dropna(subset=FEATURES + [TARGET])
    X = df[FEATURES]
    y = df[TARGET]
    logger.info(f"Features: {X.shape}, Target balance: {y.mean():.3f}")
    return X, y

def train_models(X_train, y_train, X_test, y_test):
    # Use class_weight='balanced' instead of SMOTE
    # SMOTE was generating synthetic samples that flipped coefficient signs
    logger.info(f"Training on: {X_train.shape}, balance: {y_train.mean():.3f}")

    results = {}

    # --- Logistic Regression ---
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced"
        ))
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_probs = lr_pipeline.predict_proba(X_test)[:, 1]
    lr_preds = lr_pipeline.predict(X_test)
    lr_auc = roc_auc_score(y_test, lr_probs)

    logger.info(f"\n--- Logistic Regression ---")
    logger.info(f"AUC: {lr_auc:.4f}")
    print(classification_report(y_test, lr_preds, target_names=["Normal", "Prolonged"]))

    results["logistic_regression"] = {
        "model": lr_pipeline, "auc": lr_auc,
        "probs": lr_probs, "preds": lr_preds
    }

    # --- XGBoost ---
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        scale_pos_weight=neg / pos,
        eval_metric="logloss",
        random_state=42, verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    xgb_preds = xgb_model.predict(X_test)
    xgb_auc = roc_auc_score(y_test, xgb_probs)

    logger.info(f"\n--- XGBoost ---")
    logger.info(f"AUC: {xgb_auc:.4f}")
    print(classification_report(y_test, xgb_preds, target_names=["Normal", "Prolonged"]))

    results["xgboost"] = {
        "model": xgb_model, "auc": xgb_auc,
        "probs": xgb_probs, "preds": xgb_preds
    }

    return results

def assign_risk_tier(prob: float) -> str:
    if prob < 0.2:
        return "Low"
    elif prob < 0.45:
        return "Moderate"
    else:
        return "High"

def save_models(results: dict, X_test: pd.DataFrame, y_test: pd.Series):
    for name, res in results.items():
        path = MODEL_DIR / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(res["model"], f)
        logger.success(f"Saved {name} to {path}")

    output = X_test.copy()
    output["actual"] = y_test.values
    output["lr_prob"] = results["logistic_regression"]["probs"]
    output["xgb_prob"] = results["xgboost"]["probs"]
    output["risk_tier"] = output["lr_prob"].apply(assign_risk_tier)

    pred_path = MODEL_DIR / "test_predictions.csv"
    output.to_csv(pred_path, index=False)
    logger.success(f"Saved predictions to {pred_path}")

    logger.info(f"\nRisk tier distribution:\n{output['risk_tier'].value_counts()}")
    return output

if __name__ == "__main__":
    X, y = load_features()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    results = train_models(X_train, y_train, X_test, y_test)

    logger.info(f"\n{'='*40}")
    logger.info(f"LR AUC:  {results['logistic_regression']['auc']:.4f}")
    logger.info(f"XGB AUC: {results['xgboost']['auc']:.4f}")
    logger.info(f"{'='*40}")

    save_models(results, X_test, y_test)