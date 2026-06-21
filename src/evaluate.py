"""Evaluation and visualization for readmission prediction."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.calibration import calibration_curve

from .config import Config

logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute comprehensive classification metrics."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    return {
        "classification_report": report,
        "roc_auc": roc_auc_score(y_true, y_prob),
        "avg_precision": average_precision_score(y_true, y_prob),
        "accuracy": float(np.mean(y_true == y_pred)),
        "sensitivity": report.get("1", {}).get("recall", 0.0),
        "specificity": report.get("0", {}).get("recall", 0.0),
        "ppv": report.get("1", {}).get("precision", 0.0),
        "npv": report.get("0", {}).get("precision", 0.0),
        "f1": report.get("1", {}).get("f1-score", 0.0),
    }


def print_report(metrics: dict) -> None:
    """Pretty-print evaluation results with clinical context."""
    print("\n" + "=" * 60)
    print("PATIENT READMISSION PREDICTION — EVALUATION")
    print("=" * 60)
    print(f"\nROC-AUC:           {metrics['roc_auc']:.4f}")
    print(f"Avg Precision (PR): {metrics['avg_precision']:.4f}")
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Sensitivity (TPR): {metrics['sensitivity']:.4f}")
    print(f"Specificity (TNR): {metrics['specificity']:.4f}")
    print(f"PPV (Precision):   {metrics['ppv']:.4f}")
    print(f"NPV:               {metrics['npv']:.4f}")
    print(f"F1-Score:          {metrics['f1']:.4f}")
    print("=" * 60)


def plot_roc_pr_curves(y_true: np.ndarray, y_prob: np.ndarray, config: Config) -> None:
    """Save ROC and Precision-Recall curves side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    axes[0].plot(fpr, tpr, "b-", linewidth=2, label=f"XGBoost (AUC = {auc:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    axes[0].fill_between(fpr, tpr, alpha=0.1, color="blue")
    axes[0].set_xlabel("False Positive Rate", fontsize=12)
    axes[0].set_ylabel("True Positive Rate", fontsize=12)
    axes[0].set_title("ROC Curve", fontsize=14)
    axes[0].legend(fontsize=11, loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # Optimal threshold (Youden's J)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    axes[0].plot(fpr[optimal_idx], tpr[optimal_idx], "ro", markersize=10,
                 label=f"Optimal threshold = {thresholds[optimal_idx]:.3f}")
    axes[0].legend(fontsize=10, loc="lower right")

    # PR curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    prevalence = y_true.mean()
    axes[1].plot(recall, precision, "r-", linewidth=2, label=f"XGBoost (AP = {ap:.3f})")
    axes[1].axhline(prevalence, color="gray", linestyle="--", linewidth=1,
                    label=f"Baseline (prevalence = {prevalence:.3f})")
    axes[1].fill_between(recall, precision, alpha=0.1, color="red")
    axes[1].set_xlabel("Recall", fontsize=12)
    axes[1].set_ylabel("Precision", fontsize=12)
    axes[1].set_title("Precision-Recall Curve", fontsize=14)
    axes[1].legend(fontsize=11, loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = config.results_dir / "roc_pr_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("ROC/PR curves saved to %s", path)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, config: Config) -> None:
    """Save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = ["Not Readmitted", "Readmitted <30d"]

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].set_title("Confusion Matrix (Counts)")

    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    axes[1].set_title("Confusion Matrix (Normalized)")

    plt.tight_layout()
    path = config.results_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", path)


def plot_feature_importance(importance_df: pd.DataFrame, config: Config) -> None:
    """Save feature importance bar chart."""
    fig, ax = plt.subplots(figsize=(10, 8))
    top = importance_df.head(20)
    ax.barh(range(len(top)), top["importance"].values, color="steelblue")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (Gain)", fontsize=12)
    ax.set_title("Top 20 Features — XGBoost", fontsize=14)
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    path = config.results_dir / "feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Feature importance saved to %s", path)


def plot_calibration(y_true: np.ndarray, y_prob: np.ndarray, config: Config) -> None:
    """Save calibration (reliability) curve."""
    fig, ax = plt.subplots(figsize=(8, 7))
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)

    ax.plot(prob_pred, prob_true, "s-", linewidth=2, markersize=8, label="XGBoost")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title("Calibration Curve", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = config.results_dir / "calibration_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Calibration curve saved to %s", path)


def plot_risk_distribution(y_true: np.ndarray, y_prob: np.ndarray, config: Config) -> None:
    """Save predicted probability distributions for readmitted vs not."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(y_prob[y_true == 0], bins=50, alpha=0.6, color="green",
            label="Not Readmitted", density=True)
    ax.hist(y_prob[y_true == 1], bins=50, alpha=0.6, color="red",
            label="Readmitted <30d", density=True)
    ax.set_xlabel("Predicted Readmission Probability", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Risk Score Distribution", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = config.results_dir / "risk_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Risk distribution saved to %s", path)
