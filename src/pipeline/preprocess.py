import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

RAW_PATH = Path("data/raw/diabetic_data.csv")
PROCESSED_PATH = Path("data/processed/diabetic_clean.csv")

def load_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)
    logger.info(f"Loaded raw data: {df.shape}")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Replace missing value markers with NaN
    df = df.replace("?", np.nan)

    # Drop columns with >40% missing or not clinically useful
    drop_cols = ["weight", "payer_code", "medical_specialty",
                 "examide", "citoglipton"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Drop duplicate patients — keep first encounter only
    df = df.drop_duplicates(subset="patient_nbr", keep="first")

    # Drop rows where discharge disposition indicates death or hospice
    death_codes = [11, 13, 14, 19, 20, 21]
    df = df[~df["discharge_disposition_id"].isin(death_codes)]

    logger.info(f"After cleaning: {df.shape}")
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Target: prolonged stay = more than 7 days
    df["prolonged_stay"] = (df["time_in_hospital"] > 7).astype(int)

    # Medication burden: count medications that were changed or prescribed
    med_cols = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide",
        "glimepiride", "acetohexamide", "glipizide", "glyburide",
        "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
        "miglitol", "troglitazone", "tolazamide", "insulin",
        "glyburide-metformin", "glipizide-metformin",
        "glimepiride-pioglitazone", "metformin-rosiglitazone",
        "metformin-pioglitazone"
    ]
    med_cols = [c for c in med_cols if c in df.columns]
    df["medication_burden"] = (
        df[med_cols].isin(["Up", "Down", "Steady"]).sum(axis=1)
    )

    # Diagnostic complexity: number of unique diagnoses
    diag_cols = ["diag_1", "diag_2", "diag_3"]
    df["diagnostic_complexity"] = df[diag_cols].nunique(axis=1)

    # Emergency admission flag
    df["is_emergency"] = (df["admission_type_id"] == 1).astype(int)

    # Readmission binary flag
    df["readmitted_30"] = (df["readmitted"] == "<30").astype(int)

    # Age as ordinal midpoint
    age_map = {
        "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
        "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
        "[80-90)": 85, "[90-100)": 95
    }
    df["age_numeric"] = df["age"].map(age_map)

    logger.info(f"Features engineered. Target distribution:\n"
                f"{df['prolonged_stay'].value_counts(normalize=True).round(3)}")
    return df

def save_processed(df: pd.DataFrame) -> Path:
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    logger.success(f"Saved processed data to {PROCESSED_PATH} — shape: {df.shape}")
    return PROCESSED_PATH

if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    df = engineer_features(df)
    save_processed(df)
    print("\nKey features preview:")
    print(df[["time_in_hospital", "prolonged_stay", "medication_burden",
              "diagnostic_complexity", "is_emergency", "age_numeric"]].describe())