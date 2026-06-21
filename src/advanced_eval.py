"""Advanced evaluation for readmission prediction.

Adds the rigor that matters on a near-ceiling dataset (AUC ~0.67-0.70):
  1. Model comparison (XGBoost vs LightGBM vs Logistic Regression)
  2. Decision-threshold optimization for a clinical recall target
  3. Subgroup fairness analysis (gender / age / race)
  4. Probability calibration (isotonic) with ECE + Brier before/after
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score,
    precision_score, recall_score, roc_auc_score,
)

from .config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Baseline model + comparison
# ---------------------------------------------------------------------------


def train_logreg(X_tr, y_tr, config: Config):
    """Scaled, class-balanced logistic-regression baseline."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=config.random_state),
    )
    model.fit(X_tr, y_tr)
    return model


def _best_f1(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, float, float, float]:
    """Best achievable F1 over thresholds (fair across models with different scales)."""
    ts = np.linspace(0.05, 0.95, 19)
    best = (0.5, 0.0, 0.0, 0.0)  # thr, f1, prec, rec
    for t in ts:
        yp = (prob >= t).astype(int)
        f = f1_score(y_true, yp, zero_division=0)
        if f > best[1]:
            best = (float(t), f, precision_score(y_true, yp, zero_division=0),
                    recall_score(y_true, yp, zero_division=0))
    return best


def compare_models(named: dict, y_test: np.ndarray, config: Config) -> pd.DataFrame:
    """named: {model_name: y_prob}. Threshold-independent AUC/PR-AUC + best-F1."""
    rows = []
    for name, prob in named.items():
        thr, f1v, prec, rec = _best_f1(y_test, prob)
        rows.append({
            "model": name,
            "roc_auc": roc_auc_score(y_test, prob),
            "pr_auc": average_precision_score(y_test, prob),
            "best_f1": f1v,
            "precision": prec,
            "sensitivity": rec,
        })
    df = pd.DataFrame(rows)
    df.to_csv(config.results_dir / "model_comparison.csv", index=False)
    logger.info("Model comparison:\n%s", df.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 5))
    metrics = ["roc_auc", "pr_auc", "best_f1"]
    x = np.arange(len(df))
    w = 0.25
    for i, m in enumerate(metrics):
        ax.bar(x + i * w, df[m], w, label=m.replace("_", " ").upper())
    ax.set_xticks(x + w)
    ax.set_xticklabels(df["model"])
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(config.results_dir / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return df


# ---------------------------------------------------------------------------
# 2. Threshold optimization for a clinical recall target
# ---------------------------------------------------------------------------


def optimize_threshold(y_true: np.ndarray, y_prob: np.ndarray, config: Config,
                       target_recall: float = 0.80) -> dict:
    """Find the threshold giving >= target_recall with the best precision."""
    ts = np.linspace(0.01, 0.99, 99)
    prec = [precision_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in ts]
    rec = [recall_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in ts]
    f1 = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in ts]

    # Best precision among thresholds that hit the recall target
    feasible = [(t, p, r, fr) for t, p, r, fr in zip(ts, prec, rec, f1) if r >= target_recall]
    if feasible:
        chosen = max(feasible, key=lambda x: x[1])  # highest precision
    else:
        chosen = max(zip(ts, prec, rec, f1), key=lambda x: x[3])  # fallback: best F1
    chosen_t = float(chosen[0])

    # Operating-point table at a few thresholds
    table = []
    for t in [0.15, 0.25, 0.35, 0.5, chosen_t]:
        yp = (y_prob >= t).astype(int)
        table.append({
            "threshold": round(float(t), 3),
            "precision": precision_score(y_true, yp, zero_division=0),
            "recall": recall_score(y_true, yp, zero_division=0),
            "f1": f1_score(y_true, yp, zero_division=0),
        })

    with open(config.results_dir / "threshold_analysis.txt", "w") as f:
        f.write("Decision-Threshold Analysis\n" + "=" * 40 + "\n\n")
        f.write(f"Target recall: {target_recall:.0%}\n")
        f.write(f"Chosen threshold: {chosen_t:.3f}  (precision {chosen[1]:.3f}, recall {chosen[2]:.3f})\n\n")
        f.write(f"{'thr':>6}{'prec':>9}{'recall':>9}{'F1':>8}\n")
        for r in table:
            f.write(f"{r['threshold']:>6}{r['precision']:>9.3f}{r['recall']:>9.3f}{r['f1']:>8.3f}\n")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ts, prec, label="Precision", color="#3498db")
    ax.plot(ts, rec, label="Recall", color="#e74c3c")
    ax.plot(ts, f1, label="F1", color="#2ecc71")
    ax.axvline(chosen_t, color="grey", ls="--", label=f"chosen ({chosen_t:.2f})")
    ax.axhline(target_recall, color="#e74c3c", ls=":", alpha=0.5)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Threshold Sweep (target recall {target_recall:.0%})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(config.results_dir / "threshold_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Threshold @recall>=%.0f%%: t=%.3f (prec %.3f)", target_recall * 100, chosen_t, chosen[1])
    return {"threshold": chosen_t, "precision": chosen[1], "recall": chosen[2], "table": table}


# ---------------------------------------------------------------------------
# 2b. Decision-curve analysis (net benefit)
# ---------------------------------------------------------------------------


def decision_curve_analysis(y_true: np.ndarray, y_prob: np.ndarray, config: Config,
                            pt_max: float = 0.5) -> pd.DataFrame:
    """Net-benefit decision-curve analysis (Vickers & Elkin, 2006).

    Net benefit at threshold probability pt:
        NB = TP/n - (FP/n) * (pt / (1 - pt))
    Compared against 'treat all' and 'treat none' — the model is clinically
    useful over the pt range where its curve sits above both references.
    """
    y_true = np.asarray(y_true)
    n = len(y_true)
    prevalence = y_true.mean()
    pts = np.linspace(0.01, pt_max, 50)

    nb_model, nb_all = [], []
    for pt in pts:
        pred = (y_prob >= pt).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        w = pt / (1 - pt)
        nb_model.append(tp / n - (fp / n) * w)
        nb_all.append(prevalence - (1 - prevalence) * w)

    df = pd.DataFrame({"threshold_prob": pts, "net_benefit_model": nb_model,
                       "net_benefit_treat_all": nb_all, "net_benefit_treat_none": 0.0})
    df.to_csv(config.results_dir / "decision_curve.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pts, nb_model, label="Model", color="#2980b9", lw=2)
    ax.plot(pts, nb_all, label="Treat all", color="#7f8c8d", ls="--")
    ax.axhline(0, label="Treat none", color="black", lw=1)
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision Curve Analysis")
    ax.set_ylim(min(-0.02, min(nb_model)), max(nb_model) * 1.1 + 0.01)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(config.results_dir / "decision_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Range where the model beats both references
    better = [(pt, m) for pt, m, a in zip(pts, nb_model, nb_all) if m > a and m > 0]
    rng = f"{better[0][0]:.2f}-{better[-1][0]:.2f}" if better else "none"
    with open(config.results_dir / "decision_curve.txt", "w") as f:
        f.write("Decision Curve Analysis (net benefit)\n" + "=" * 40 + "\n\n")
        f.write(f"Disease prevalence: {prevalence:.3f}\n")
        f.write(f"Model adds net benefit over treat-all/none at pt in: {rng}\n\n")
        f.write(f"{'pt':>6}{'model':>10}{'treat_all':>11}\n")
        for pt in [0.1, 0.15, 0.2, 0.3, 0.4]:
            i = int(np.argmin(np.abs(pts - pt)))
            f.write(f"{pts[i]:>6.2f}{nb_model[i]:>10.4f}{nb_all[i]:>11.4f}\n")
    logger.info("Decision-curve analysis saved (model useful at pt %s).", rng)
    return df


# ---------------------------------------------------------------------------
# 3. Subgroup fairness analysis
# ---------------------------------------------------------------------------


def _recover_subgroups(X_test: pd.DataFrame) -> dict:
    """Reconstruct protected attributes from the encoded test features."""
    groups: dict[str, pd.Series] = {}
    if "gender" in X_test.columns:
        g = X_test["gender"].map({0: "Male", 1: "Female"})
        groups["gender"] = g.where(g.notna(), "Unknown")
    if "age_ordinal" in X_test.columns:
        groups["age"] = pd.cut(
            X_test["age_ordinal"], bins=[-1, 4, 6, 9], labels=["<50", "50-70", "70+"]
        ).astype(str)
    race_cols = [c for c in X_test.columns if c.startswith("race_")]
    if race_cols:
        race = pd.Series("AfricanAmerican/Missing", index=X_test.index)  # dropped reference
        for c in race_cols:
            race[X_test[c] == 1] = c.replace("race_", "")
        groups["race"] = race
    return groups


def subgroup_fairness(X_test: pd.DataFrame, y_test: np.ndarray, y_prob: np.ndarray,
                      threshold: float, config: Config, min_pos: int = 20) -> pd.DataFrame:
    """Per-subgroup ROC-AUC / sensitivity / PPV / F1 (gender, age, race)."""
    y_pred = (y_prob >= threshold).astype(int)
    groups = _recover_subgroups(X_test)
    y_test = np.asarray(y_test)

    rows = []
    for attr, series in groups.items():
        series = series.reset_index(drop=True)
        for val in sorted(series.dropna().unique()):
            mask = (series == val).values
            n, pos = int(mask.sum()), int(y_test[mask].sum())
            if pos < min_pos or (y_test[mask] == 0).sum() < min_pos:
                continue
            rows.append({
                "attribute": attr, "group": str(val), "n": n,
                "positive_rate": round(float(y_test[mask].mean()), 3),
                "roc_auc": round(roc_auc_score(y_test[mask], y_prob[mask]), 3),
                "sensitivity": round(recall_score(y_test[mask], y_pred[mask], zero_division=0), 3),
                "ppv": round(precision_score(y_test[mask], y_pred[mask], zero_division=0), 3),
                "f1": round(f1_score(y_test[mask], y_pred[mask], zero_division=0), 3),
            })
    df = pd.DataFrame(rows)
    df.to_csv(config.results_dir / "fairness_analysis.csv", index=False)

    with open(config.results_dir / "fairness_analysis.txt", "w") as f:
        f.write("Subgroup Fairness Analysis\n" + "=" * 50 + "\n\n")
        f.write(f"Decision threshold: {threshold:.3f}\n")
        f.write(f"(subgroups with < {min_pos} positives or negatives are omitted)\n\n")
        for attr in df["attribute"].unique():
            sub = df[df["attribute"] == attr]
            f.write(f"By {attr}:\n")
            f.write(f"  {'group':<22}{'n':>7}{'pos%':>7}{'AUC':>7}{'Sens':>7}{'PPV':>7}{'F1':>7}\n")
            for _, r in sub.iterrows():
                f.write(f"  {r['group']:<22}{r['n']:>7}{r['positive_rate']*100:>6.1f}"
                        f"{r['roc_auc']:>7.3f}{r['sensitivity']:>7.3f}{r['ppv']:>7.3f}{r['f1']:>7.3f}\n")
            spread = sub["roc_auc"].max() - sub["roc_auc"].min()
            f.write(f"  -> AUC spread across {attr}: {spread:.3f}\n\n")

    # Plot AUC + sensitivity by subgroup
    if not df.empty:
        df["label"] = df["attribute"] + ":" + df["group"]
        fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.4)))
        yy = np.arange(len(df))
        ax.barh(yy - 0.2, df["roc_auc"], 0.4, label="ROC-AUC", color="#3498db")
        ax.barh(yy + 0.2, df["sensitivity"], 0.4, label="Sensitivity", color="#e74c3c")
        ax.set_yticks(yy)
        ax.set_yticklabels(df["label"], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_title("Fairness — performance by subgroup")
        ax.legend()
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        fig.savefig(config.results_dir / "fairness_analysis.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    logger.info("Fairness analysis saved (%d subgroups).", len(df))
    return df


# ---------------------------------------------------------------------------
# 4. Probability calibration
# ---------------------------------------------------------------------------


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if m.sum() > 0:
            ece += (m.sum() / len(y_prob)) * abs(y_true[m].mean() - y_prob[m].mean())
    return float(ece)


def calibrate_and_report(model, X_val, y_val, X_test, y_test, y_prob_raw, config: Config) -> dict:
    """Isotonic-calibrate on validation scores; report ECE + Brier before/after.

    Uses IsotonicRegression directly (val scores -> labels) rather than
    CalibratedClassifierCV(cv='prefit'), which was removed in recent sklearn.
    """
    from sklearn.calibration import calibration_curve
    from sklearn.isotonic import IsotonicRegression

    y_test = np.asarray(y_test)
    y_val = np.asarray(y_val)
    val_prob = model.predict_proba(X_val)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_prob, y_val)
    y_prob_cal = iso.transform(y_prob_raw)

    before = {"ece": _ece(y_test, y_prob_raw), "brier": brier_score_loss(y_test, y_prob_raw)}
    after = {"ece": _ece(y_test, y_prob_cal), "brier": brier_score_loss(y_test, y_prob_cal)}

    with open(config.results_dir / "calibration_metrics.txt", "w") as f:
        f.write("Probability Calibration (isotonic)\n" + "=" * 40 + "\n\n")
        f.write(f"{'':<14}{'ECE':>10}{'Brier':>10}\n")
        f.write(f"{'Uncalibrated':<14}{before['ece']:>10.4f}{before['brier']:>10.4f}\n")
        f.write(f"{'Calibrated':<14}{after['ece']:>10.4f}{after['brier']:>10.4f}\n")

    fig, ax = plt.subplots(figsize=(8, 7))
    for prob, lbl, c in [(y_prob_raw, f"Uncalibrated (ECE {before['ece']:.3f})", "#3498db"),
                         (y_prob_cal, f"Calibrated (ECE {after['ece']:.3f})", "#2ecc71")]:
        pt, pp = calibration_curve(y_test, prob, n_bins=10)
        ax.plot(pp, pt, "s-", label=lbl, color=c)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration: before vs after isotonic")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(config.results_dir / "calibration_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Calibration ECE %.4f -> %.4f | Brier %.4f -> %.4f",
                before["ece"], after["ece"], before["brier"], after["brier"])
    return {"before": before, "after": after, "y_prob_cal": y_prob_cal}
