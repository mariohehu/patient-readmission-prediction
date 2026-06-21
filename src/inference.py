"""Standalone inference for the trained readmission model.

Score new patient encounters (same raw schema as the UCI CSV) without retraining:

    python src/inference.py --data new_patients.csv \
        --model results/xgb_model.json \
        --config results/model_config.json \
        --output results/predictions.csv
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data_loader import preprocess_dataframe


def load_model(model_path: str, config_path: str):
    """Load the saved XGBoost model and its operating config."""
    cfg = json.loads(Path(config_path).read_text())
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    return model, cfg


def score(df_raw: pd.DataFrame, model, cfg: dict, config: Config) -> pd.DataFrame:
    """Preprocess raw encounters, align columns to training, return risk + flag."""
    X, _ = preprocess_dataframe(df_raw, config, training=False)
    # Align to the exact training feature set (handles unseen/missing one-hot levels)
    X = X.reindex(columns=cfg["features"], fill_value=0)
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= cfg["threshold"]).astype(int)
    return pd.DataFrame({"readmission_risk": prob.round(4), "predicted_readmit": pred})


def main() -> None:
    p = argparse.ArgumentParser(description="Score new patients for 30-day readmission risk")
    p.add_argument("--data", required=True, help="CSV of raw encounters")
    p.add_argument("--model", default="results/xgb_model.json")
    p.add_argument("--config", default="results/model_config.json")
    p.add_argument("--output", default="results/predictions.csv")
    args = p.parse_args()

    model, cfg = load_model(args.model, args.config)
    df = pd.read_csv(args.data, na_values="?")
    out = score(df, model, cfg, Config())
    out.to_csv(args.output, index=False)
    print(f"Scored {len(out)} patients | {int(out['predicted_readmit'].sum())} flagged high-risk "
          f"(threshold {cfg['threshold']}) -> {args.output}")


if __name__ == "__main__":
    main()
