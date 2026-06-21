"""Smoke tests for the readmission-prediction pipeline.

Run: pytest -q   (from the project directory)
Tests skip automatically if the UCI CSV has not been downloaded yet.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from src.config import Config
from src.data_loader import preprocess_dataframe
from src.inference import score

DATA = Path(__file__).resolve().parent.parent / "data" / "diabetic_data.csv"


@pytest.fixture(scope="module")
def raw():
    if not DATA.exists():
        pytest.skip("diabetic_data.csv not present (run main.py once to download)")
    df = pd.read_csv(DATA, na_values="?")
    return df.sample(3000, random_state=42).reset_index(drop=True)


@pytest.fixture(scope="module")
def prepared(raw):
    return preprocess_dataframe(raw, Config())


@pytest.fixture(scope="module")
def model(prepared):
    X, y = prepared
    m = xgb.XGBClassifier(n_estimators=50, max_depth=4, random_state=42, n_jobs=-1, eval_metric="auc")
    m.fit(X, y)
    return m


def test_preprocess_clean(prepared):
    X, y = prepared
    assert y is not None
    assert set(np.unique(y)) <= {0, 1}
    assert len(X) == len(y)
    # XGBoost-ready: no object/string columns (numeric + bool one-hot are fine)
    assert X.select_dtypes(include="object").shape[1] == 0
    assert not X.isnull().any().any()  # no NaN


def test_preprocess_inference_keeps_all_rows(raw):
    """Inference mode keeps every input row (no deceased removal) and no target."""
    X, y = preprocess_dataframe(raw, Config(), training=False)
    assert y is None
    assert len(X) == len(raw)


def test_model_probabilities(model, prepared):
    X, _ = prepared
    prob = model.predict_proba(X)[:, 1]
    assert ((prob >= 0.0) & (prob <= 1.0)).all()


def test_inference_roundtrip(model, prepared, raw):
    X, _ = prepared
    cfg = {"threshold": 0.5, "features": list(X.columns)}
    out = score(raw, model, cfg, Config())
    assert {"readmission_risk", "predicted_readmit"} <= set(out.columns)
    assert len(out) == len(raw)
    assert out["predicted_readmit"].isin([0, 1]).all()
    assert ((out["readmission_risk"] >= 0) & (out["readmission_risk"] <= 1)).all()


def test_inference_column_alignment(model, prepared, raw):
    """A small batch may miss one-hot levels — reindex must restore them."""
    X, _ = prepared
    cfg = {"threshold": 0.5, "features": list(X.columns)}
    out = score(raw.head(40), model, cfg, Config())  # likely fewer categories present
    assert len(out) == 40
    assert out["predicted_readmit"].isin([0, 1]).all()
