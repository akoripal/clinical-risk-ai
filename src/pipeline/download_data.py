import pandas as pd
import urllib.request
import zipfile
import io
from pathlib import Path
from loguru import logger

RAW_DATA_DIR = Path("data/raw")
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip"

def download_dataset():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_DATA_DIR / "diabetic_data.csv"

    if output_path.exists():
        logger.info(f"Dataset already exists at {output_path}")
        return output_path

    logger.info("Downloading UCI Diabetes dataset...")
    with urllib.request.urlopen(DATASET_URL) as r:
        with zipfile.ZipFile(io.BytesIO(r.read())) as z:
            for name in z.namelist():
                if name.endswith(".csv") and "diabetic" in name:
                    z.extract(name, RAW_DATA_DIR)
                    logger.success(f"Downloaded: {name}")

    return output_path

def validate_dataset(path: Path):
    df = pd.read_csv(path)
    logger.info(f"Shape: {df.shape}")
    assert df.shape[0] == 101766, f"Expected 101,766 rows, got {df.shape[0]}"
    assert df.shape[1] == 50, f"Expected 50 columns, got {df.shape[1]}"
    logger.success("Dataset validation passed ✓")
    return df

if __name__ == "__main__":
    path = download_dataset()
    df = validate_dataset(path)
    print(df.head())
    