"""SHAP-based model explainability for readmission prediction."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .config import Config

logger = logging.getLogger(__name__)


def compute_shap_values(
    model: object,
    X: pd.DataFrame,
    max_samples: int = 1000,
) -> tuple[shap.Explanation, np.ndarray]:
    """Compute SHAP values using TreeExplainer."""
    if len(X) > max_samples:
        X_sample = X.sample(n=max_samples, random_state=42)
    else:
        X_sample = X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    logger.info("SHAP values computed for %d samples, %d features.", len(X_sample), X_sample.shape[1])
    return shap_values, X_sample.values


def plot_shap_summary(
    shap_values: shap.Explanation,
    config: Config,
) -> None:
    """Save SHAP beeswarm summary plot."""
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.tight_layout()
    path = config.results_dir / "shap_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("SHAP summary saved to %s", path)


def plot_shap_bar(
    shap_values: shap.Explanation,
    config: Config,
) -> None:
    """Save SHAP feature importance bar chart."""
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.bar(shap_values, max_display=20, show=False)
    plt.tight_layout()
    path = config.results_dir / "shap_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("SHAP importance saved to %s", path)


def plot_shap_dependence(
    shap_values: shap.Explanation,
    feature_name: str,
    X: pd.DataFrame,
    config: Config,
) -> None:
    """Save SHAP dependence plot for a specific feature."""
    if feature_name not in X.columns:
        logger.warning("Feature %s not found, skipping dependence plot.", feature_name)
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    feature_idx = list(X.columns).index(feature_name)
    shap.plots.scatter(shap_values[:, feature_idx], show=False)
    plt.title(f"SHAP Dependence: {feature_name}", fontsize=13)
    plt.tight_layout()
    path = config.results_dir / f"shap_dependence_{feature_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("SHAP dependence plot saved for %s", feature_name)


def plot_shap_waterfall(
    shap_values: shap.Explanation,
    idx: int,
    config: Config,
) -> None:
    """Save SHAP waterfall plot for a single prediction explanation."""
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.waterfall(shap_values[idx], max_display=15, show=False)
    plt.title(f"Individual Prediction Explanation (sample #{idx})", fontsize=12)
    plt.tight_layout()
    path = config.results_dir / f"shap_waterfall_sample_{idx}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("SHAP waterfall saved for sample %d", idx)


def generate_clinical_report(
    shap_values: shap.Explanation,
    X: pd.DataFrame,
    config: Config,
) -> str:
    """Generate a text report of key SHAP findings for clinical context."""
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_names = X.columns if hasattr(X, "columns") else [f"f{i}" for i in range(len(mean_abs_shap))]

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    lines = [
        "CLINICAL EXPLAINABILITY REPORT",
        "=" * 50,
        "",
        "Top Risk Factors for 30-Day Hospital Readmission:",
        "-" * 50,
    ]

    for rank, (_, row) in enumerate(importance_df.head(15).iterrows(), start=1):
        feature = row["feature"]
        importance = row["mean_abs_shap"]

        # Clinical interpretation
        if "number_inpatient" in feature:
            context = "Prior inpatient visits — strong indicator of disease complexity"
        elif "num_medications" in feature:
            context = "Polypharmacy burden — correlates with multi-morbidity"
        elif "time_in_hospital" in feature:
            context = "Length of stay — proxy for illness severity"
        elif "number_diagnoses" in feature:
            context = "Diagnostic complexity — multiple comorbidities"
        elif "num_lab_procedures" in feature:
            context = "Lab intensity — reflects clinical uncertainty or severity"
        elif "age" in feature:
            context = "Patient age — elderly patients at higher readmission risk"
        elif "insulin" in feature:
            context = "Insulin management — diabetes control indicator"
        elif "discharge" in feature:
            context = "Discharge disposition — transition of care quality"
        elif "A1C" in feature:
            context = "HbA1c level — glycemic control marker"
        elif "number_emergency" in feature:
            context = "ED utilization — healthcare access pattern"
        else:
            context = "Clinical feature"

        lines.append(f"  {rank:2d}. {feature}")
        lines.append(f"      SHAP impact: {importance:.4f}")
        lines.append(f"      Clinical context: {context}")
        lines.append("")

    lines.extend([
        "=" * 50,
        "Note: SHAP values indicate average absolute contribution",
        "to the model's readmission risk prediction. Higher values",
        "mean the feature has more influence on predictions.",
        "",
        "This analysis is for research purposes only and should not",
        "be used as a substitute for clinical judgment.",
    ])

    report = "\n".join(lines)

    report_path = config.results_dir / "clinical_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info("Clinical report saved to %s", report_path)

    return report
