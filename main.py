#!/usr/bin/env python3
"""Patient Readmission Prediction — End-to-end pipeline.

Predicts 30-day hospital readmission for diabetic patients using XGBoost
with SHAP explainability. Uses the UCI Diabetes 130-US Hospitals dataset.

Usage:
    python main.py                   # Full pipeline
    python main.py --no-shap         # Skip SHAP analysis (faster)
    python main.py --compare-lgbm    # Also train LightGBM for comparison
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.data_loader import load_and_preprocess
from src.model import train_xgboost, train_lightgbm, cross_validate_model, get_feature_importance
from src.evaluate import (
    compute_metrics, print_report,
    plot_roc_pr_curves, plot_confusion_matrix,
    plot_feature_importance, plot_calibration,
    plot_risk_distribution,
)
from src.advanced_eval import (
    train_logreg, compare_models, optimize_threshold,
    subgroup_fairness, calibrate_and_report, decision_curve_analysis,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("readmission")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patient Readmission Prediction")
    parser.add_argument("--no-shap", action="store_true", help="Skip SHAP analysis")
    parser.add_argument("--compare-lgbm", action="store_true", help="Also train LightGBM")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config()
    config.results_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load and preprocess ──
    logger.info("Step 1/5: Loading and preprocessing clinical data...")
    X, y = load_and_preprocess(config)
    logger.info("Dataset: %d patients, %d features, %.1f%% positive",
                len(X), X.shape[1], y.mean() * 100)

    # ── Step 2: Train/test split ──
    logger.info("Step 2/5: Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size,
        random_state=config.random_state, stratify=y,
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=config.val_size,
        random_state=config.random_state, stratify=y_train,
    )
    logger.info("  Train: %d | Val: %d | Test: %d", len(X_tr), len(X_val), len(X_test))

    # ── Step 3: Train XGBoost ──
    logger.info("Step 3/5: Training XGBoost...")
    xgb_model = train_xgboost(X_tr, y_tr, X_val, y_val, config)

    # Cross-validation
    logger.info("Running 5-fold cross-validation...")
    cv_results = cross_validate_model(xgb_model, X_train, y_train, config)

    # Feature importance
    importance_df = get_feature_importance(xgb_model, list(X.columns), top_n=25)
    logger.info("Top features:\n%s", importance_df.head(10).to_string(index=False))

    # ── Step 4: Evaluate on test set ──
    logger.info("Step 4/5: Evaluating on test set...")
    y_pred = xgb_model.predict(X_test)
    y_prob = xgb_model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test.values, y_pred, y_prob)
    print_report(metrics)

    # Generate all evaluation plots
    plot_roc_pr_curves(y_test.values, y_prob, config)
    plot_confusion_matrix(y_test.values, y_pred, config)
    plot_feature_importance(importance_df, config)
    plot_calibration(y_test.values, y_prob, config)
    plot_risk_distribution(y_test.values, y_prob, config)

    # ── Step 4b: Advanced evaluation — comparison, threshold, fairness, calibration ──
    logger.info("Step 4b: Model comparison, threshold tuning, fairness, calibration...")
    logreg_model = train_logreg(X_tr, y_tr, config)
    model_probs = {
        "XGBoost": y_prob,
        "LogisticReg": logreg_model.predict_proba(X_test)[:, 1],
    }
    try:
        lgbm_model = train_lightgbm(X_tr, y_tr, X_val, y_val, config)
        model_probs["LightGBM"] = lgbm_model.predict_proba(X_test)[:, 1]
    except ImportError:
        logger.warning("lightgbm not installed — skipping it in the comparison.")
    comparison_df = compare_models(model_probs, y_test.values, config)
    thr_info = optimize_threshold(y_test.values, y_prob, config, target_recall=0.80)
    fairness_df = subgroup_fairness(X_test, y_test.values, y_prob, thr_info["threshold"], config)
    calib = calibrate_and_report(xgb_model, X_val, y_val, X_test, y_test, y_prob, config)
    # Decision-curve uses CALIBRATED probabilities (pt is interpreted as a risk).
    decision_curve_analysis(y_test.values, calib["y_prob_cal"], config)

    # Persist model + operating config (deployment-ready, consistent with P5)
    import json
    xgb_model.save_model(config.results_dir / "xgb_model.json")
    with open(config.results_dir / "model_config.json", "w") as f:
        json.dump({
            "threshold": round(thr_info["threshold"], 4),
            "model_path": "results/xgb_model.json",
            "positive_label": config.positive_label,
            "features": list(X.columns),
        }, f, indent=2)
    logger.info("Saved model + model_config.json")

    # ── Step 5: SHAP explainability ──
    if not args.no_shap:
        logger.info("Step 5/5: Computing SHAP values...")
        from src.explainability import (
            compute_shap_values, plot_shap_summary, plot_shap_bar,
            plot_shap_dependence, plot_shap_waterfall,
            generate_clinical_report,
        )
        shap_values, X_shap = compute_shap_values(xgb_model, X_test)
        plot_shap_summary(shap_values, config)
        plot_shap_bar(shap_values, config)

        # Dependence plots for top features
        top_features = importance_df["feature"].head(3).tolist()
        for feat in top_features:
            plot_shap_dependence(shap_values, feat, X_test, config)

        # Waterfall for individual predictions
        readmitted_indices = np.where(y_pred == 1)[0]
        if len(readmitted_indices) > 0:
            plot_shap_waterfall(shap_values, readmitted_indices[0], config)

        report = generate_clinical_report(shap_values, X_test, config)
        print(f"\n{report}")
    else:
        logger.info("Step 5/5: Skipping SHAP analysis (--no-shap).")

    # Save summary
    summary_path = config.results_dir / "evaluation_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Patient Readmission Prediction — Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset: UCI Diabetes 130-US Hospitals\n")
        f.write(f"Patients: {len(X):,}\n")
        f.write(f"Features: {X.shape[1]}\n")
        f.write(f"Positive rate: {y.mean()*100:.1f}%\n\n")
        f.write(f"Test Results (XGBoost):\n")
        f.write(f"  ROC-AUC:        {metrics['roc_auc']:.4f}\n")
        f.write(f"  Avg Precision:   {metrics['avg_precision']:.4f}\n")
        f.write(f"  Sensitivity:     {metrics['sensitivity']:.4f}\n")
        f.write(f"  Specificity:     {metrics['specificity']:.4f}\n")
        f.write(f"  F1-Score:        {metrics['f1']:.4f}\n\n")
        f.write(f"Cross-Validation (5-fold):\n")
        for metric in ["auc", "f1", "precision", "recall"]:
            scores = cv_results[f"test_{metric}"]
            f.write(f"  {metric}: {scores.mean():.4f} +/- {scores.std():.4f}\n")

        f.write("\nModel Comparison (test):\n")
        for _, r in comparison_df.iterrows():
            f.write(f"  {r['model']:<12} AUC={r['roc_auc']:.4f}  PR-AUC={r['pr_auc']:.4f}  best-F1={r['best_f1']:.4f}\n")

        f.write(f"\nClinical operating point (target recall 80%):\n")
        f.write(f"  threshold={thr_info['threshold']:.3f}  precision={thr_info['precision']:.3f}  recall={thr_info['recall']:.3f}\n")

        f.write(f"\nCalibration (isotonic):  ECE {calib['before']['ece']:.4f} -> {calib['after']['ece']:.4f}"
                f"  |  Brier {calib['before']['brier']:.4f} -> {calib['after']['brier']:.4f}\n")

        if not fairness_df.empty:
            f.write("\nFairness — ROC-AUC spread across subgroups:\n")
            for attr in fairness_df["attribute"].unique():
                sub = fairness_df[fairness_df["attribute"] == attr]
                f.write(f"  {attr:<8} {sub['roc_auc'].min():.3f} - {sub['roc_auc'].max():.3f} "
                        f"(spread {sub['roc_auc'].max() - sub['roc_auc'].min():.3f})\n")
        f.write("\nSee: model_comparison.*, threshold_analysis.*, fairness_analysis.*,\n")
        f.write("     calibration_comparison.png, decision_curve.* (net benefit, calibrated probs)\n")
    logger.info("Summary saved to %s", summary_path)

    logger.info("Done! Results saved to %s/", config.results_dir)


if __name__ == "__main__":
    main()
