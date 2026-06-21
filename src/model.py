"""XGBoost and LightGBM models for readmission prediction with cross-validation."""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    make_scorer,
)
import xgboost as xgb

from .config import Config

logger = logging.getLogger(__name__)


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    config: Config = Config(),
) -> xgb.XGBClassifier:
    """Train XGBoost with class imbalance handling and early stopping."""
    # Calculate scale_pos_weight for imbalanced data
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / max(pos_count, 1)

    params = dict(config.xgb_params)
    params["scale_pos_weight"] = scale_pos_weight

    model = xgb.XGBClassifier(**params, early_stopping_rounds=30)

    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        logger.info(
            "XGBoost trained: %d rounds (best: %d), val AUC: %.4f",
            model.n_estimators, model.best_iteration,
            roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]),
        )
    else:
        model.fit(X_train, y_train, verbose=False)
        logger.info("XGBoost trained: %d rounds (no validation set).", model.n_estimators)

    return model


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    config: Config = Config(),
) -> object:
    """Train LightGBM as baseline comparison."""
    import lightgbm as lgb

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()

    model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=neg_count / max(pos_count, 1),
        random_state=config.random_state,
        n_jobs=-1,
        verbose=-1,
    )

    callbacks = [lgb.early_stopping(30, verbose=False)]
    if X_val is not None:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks)
    else:
        model.fit(X_train, y_train)

    logger.info("LightGBM trained: %d iterations.", model.n_estimators)
    return model


def cross_validate_model(
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
    config: Config,
) -> dict[str, np.ndarray]:
    """Run stratified k-fold cross-validation with clinical metrics."""
    scorers = {
        "auc": "roc_auc",
        "avg_precision": "average_precision",
        "f1": make_scorer(f1_score),
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score),
    }

    cv_model = xgb.XGBClassifier(**{
        k: v for k, v in model.get_params().items()
        if k != "early_stopping_rounds"
    })

    cv = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_state)
    results = cross_validate(
        cv_model, X, y, cv=cv, scoring=scorers,
        return_train_score=False, n_jobs=-1,
    )

    for metric in scorers:
        scores = results[f"test_{metric}"]
        logger.info("CV %s: %.4f +/- %.4f", metric, scores.mean(), scores.std())

    return results


def get_feature_importance(
    model: xgb.XGBClassifier,
    feature_names: list[str],
    top_n: int = 25,
) -> pd.DataFrame:
    """Extract and rank feature importances."""
    importance = model.feature_importances_
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False)

    df["cumulative_importance"] = df["importance"].cumsum() / df["importance"].sum()
    return df.head(top_n).reset_index(drop=True)
