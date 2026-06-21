"""UCI Diabetes 130-US Hospitals dataset loader and preprocessing."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

logger = logging.getLogger(__name__)

# Columns to drop: IDs, free-text, and near-constant features
DROP_COLUMNS = [
    "encounter_id", "patient_nbr",
    "weight",               # >95% missing
    "payer_code",           # >50% missing
    "medical_specialty",    # >50% missing, high cardinality
    "examide", "citoglipton",  # near-constant (single value)
]

# Diagnosis code groupings (ICD-9 → clinical category)
ICD9_GROUPS = {
    "circulatory": (390, 460),
    "respiratory": (460, 520),
    "digestive": (520, 580),
    "diabetes": (250, 251),
    "injury": (800, 1000),
    "musculoskeletal": (710, 740),
    "genitourinary": (580, 630),
    "neoplasms": (140, 240),
    "other": (0, 0),  # fallback
}


def _load_raw_data(config: Config) -> pd.DataFrame:
    """Load from local CSV or download from UCI ML Repository."""
    csv_path = config.data_dir / "diabetic_data.csv"

    if csv_path.exists():
        logger.info("Loading from local CSV: %s", csv_path)
        return pd.read_csv(csv_path, na_values="?")

    logger.info("Downloading from UCI ML Repository...")
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=296)
        df = dataset.data.original
    except Exception:
        logger.info("ucimlrepo failed, trying direct download...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip"
        import zipfile, io, urllib.request
        response = urllib.request.urlopen(url)
        with zipfile.ZipFile(io.BytesIO(response.read())) as z:
            with z.open("dataset_diabetes/diabetic_data.csv") as f:
                df = pd.read_csv(f, na_values="?")

    config.data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info("Saved to %s (%d rows)", csv_path, len(df))
    return df


def _map_icd9(code: str) -> str:
    """Map an ICD-9 code to a clinical category."""
    if pd.isna(code):
        return "missing"
    code_str = str(code).strip()
    if code_str.startswith("V") or code_str.startswith("E"):
        return "external"
    try:
        num = float(code_str)
    except ValueError:
        return "other"
    for group, (low, high) in ICD9_GROUPS.items():
        if group == "other":
            continue
        if low <= num < high:
            return group
    return "other"


def _create_binary_target(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Convert 3-class readmission to binary: readmitted <30 days vs not."""
    df = df.copy()
    df["readmitted_30d"] = (df[config.target_column] == config.positive_label).astype(int)
    df = df.drop(columns=[config.target_column])
    return df


def load_and_preprocess(config: Config) -> tuple[pd.DataFrame, pd.Series]:
    """Load raw data and run the full preprocessing pipeline."""
    df = _load_raw_data(config)
    logger.info("Raw data: %d rows, %d columns", *df.shape)
    return preprocess_dataframe(df, config)


def preprocess_dataframe(df: pd.DataFrame, config: Config, training: bool = True):
    """Apply feature engineering + encoding to a raw dataframe.

    ``training=True`` builds the binary target and removes deceased patients
    (standard dataset cleaning). ``training=False`` (inference) keeps every input
    row so predictions map 1:1 to patients, and returns ``target=None``.
    """
    df = df.copy()

    # Drop columns
    existing_drops = [c for c in DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=existing_drops)

    if training and config.target_column in df.columns:
        # Binary target + remove deceased (training-time cleaning only)
        df = _create_binary_target(df, config)
        if "discharge_disposition_id" in df.columns:
            expired_ids = [11, 13, 14, 19, 20, 21]
            df = df[~df["discharge_disposition_id"].isin(expired_ids)]
    elif config.target_column in df.columns:
        # Inference: the outcome column is not a feature
        df = df.drop(columns=[config.target_column])

    # Diagnosis grouping
    for diag_col in ["diag_1", "diag_2", "diag_3"]:
        if diag_col in df.columns:
            df[f"{diag_col}_group"] = df[diag_col].apply(_map_icd9)
            df = df.drop(columns=[diag_col])

    # Medication change features
    med_columns = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide",
        "glimepiride", "acetohexamide", "glipizide", "glyburide",
        "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
        "miglitol", "troglitazone", "tolazamide", "insulin",
        "glyburide-metformin", "glipizide-metformin",
        "glimepiride-pioglitazone", "metformin-rosiglitazone",
        "metformin-pioglitazone",
    ]
    existing_meds = [c for c in med_columns if c in df.columns]
    df["num_med_changes"] = sum(
        (df[col].isin(["Up", "Down"])).astype(int) for col in existing_meds
    )
    df["num_meds_active"] = sum(
        (df[col] != "No").astype(int) for col in existing_meds
    )

    # Encode medication columns: No=0, Steady=1, Up=2, Down=3
    med_mapping = {"No": 0, "Steady": 1, "Up": 2, "Down": 3}
    for col in existing_meds:
        df[col] = df[col].map(med_mapping).fillna(0).astype(int)

    # Encode categorical columns
    binary_mapping = {"Yes": 1, "No": 0, "Ch": 1}
    for col in ["change", "diabetesMed"]:
        if col in df.columns:
            df[col] = df[col].map(binary_mapping).fillna(0).astype(int)

    # Age encoding (ordinal)
    if "age" in df.columns:
        age_order = {age: i for i, age in enumerate(config.age_bins)}
        df["age_ordinal"] = df["age"].map(age_order).fillna(5)
        df = df.drop(columns=["age"])

    # Gender encoding
    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Male": 0, "Female": 1, "Unknown/Invalid": -1}).fillna(-1).astype(int)

    # A1C result encoding
    a1c_mapping = {"None": 0, "Norm": 1, ">7": 2, ">8": 3}
    if "A1Cresult" in df.columns:
        df["A1Cresult"] = df["A1Cresult"].map(a1c_mapping).fillna(0).astype(int)

    # Max glucose encoding
    glucose_mapping = {"None": 0, "Norm": 1, ">200": 2, ">300": 3}
    if "max_glu_serum" in df.columns:
        df["max_glu_serum"] = df["max_glu_serum"].map(glucose_mapping).fillna(0).astype(int)

    # Race encoding (one-hot)
    if "race" in df.columns:
        df = pd.get_dummies(df, columns=["race"], prefix="race", drop_first=True)

    # Diagnosis groups (one-hot)
    for col in ["diag_1_group", "diag_2_group", "diag_3_group"]:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col], prefix=col, drop_first=True)

    # Admission/discharge/source encoding
    for col in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df = pd.get_dummies(df, columns=[col], prefix=col, drop_first=True)

    # Derived features
    if "num_lab_procedures" in df.columns and "num_procedures" in df.columns:
        df["lab_to_proc_ratio"] = df["num_lab_procedures"] / (df["num_procedures"] + 1)

    if "num_medications" in df.columns and "time_in_hospital" in df.columns:
        df["meds_per_day"] = df["num_medications"] / (df["time_in_hospital"] + 1)

    if "number_diagnoses" in df.columns and "num_medications" in df.columns:
        df["complexity_score"] = df["number_diagnoses"] * df["num_medications"]

    # Fill remaining NaN
    df = df.fillna(0)

    # Ensure all columns are numeric
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = pd.Categorical(df[col]).codes

    if "readmitted_30d" in df.columns:
        target = df["readmitted_30d"]
        features = df.drop(columns=["readmitted_30d"])
        logger.info(
            "Preprocessed: %d rows, %d features. Positive rate: %.2f%%",
            len(features), features.shape[1], target.mean() * 100,
        )
    else:
        target, features = None, df
        logger.info("Preprocessed (no target): %d rows, %d features", len(features), features.shape[1])
    return features, target
