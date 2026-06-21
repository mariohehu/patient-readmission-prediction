# Patient Readmission Prediction

**XGBoost model for predicting 30-day hospital readmission with SHAP explainability.**

## Overview

This project predicts whether a diabetic patient will be readmitted to the hospital within 30 days of discharge. Using the UCI Diabetes 130-US Hospitals dataset (101,766 patient encounters from 130 US hospitals over 10 years), the pipeline performs clinical feature engineering, trains gradient-boosted models, and provides interpretable explanations via SHAP analysis.

## Clinical Context

Hospital readmissions within 30 days cost the US healthcare system ~$26 billion annually. The Hospital Readmissions Reduction Program (HRRP) penalizes hospitals with excess readmissions. Identifying high-risk patients at discharge enables targeted interventions (follow-up calls, transitional care, medication reconciliation).

## Results (held-out test set)

| Model | ROC-AUC | PR-AUC | Best F1 |
|-------|---------|--------|---------|
| **XGBoost** | **0.671** | **0.232** | **0.283** |
| Logistic Regression | 0.665 | 0.220 | 0.280 |
| LightGBM | 0.636 | 0.188 | 0.205 |

XGBoost 5-fold CV: ROC-AUC 0.659 ± 0.006. **This is near the published ceiling
for this dataset (~0.64–0.70)** — readmission is intrinsically hard to predict
from administrative data, so the value here is in *rigor*, not chasing AUC:

- **Clinical operating point** — at a decision threshold of 0.40 the model reaches
  **80% recall** (catches 4 in 5 readmissions) at 14.5% precision; the full
  precision/recall/threshold trade-off is in `results/threshold_analysis.png`.
- **Calibration** — raw XGBoost is badly miscalibrated (`scale_pos_weight`
  inflates scores, ECE 0.34); **isotonic calibration cuts ECE to 0.003** and
  Brier 0.215 → 0.096, so the output is a usable risk probability.
- **Decision-curve analysis** — on the *calibrated* probabilities the model adds
  **positive net benefit over "treat-all"/"treat-none" across pt 0.01–0.50**
  (`results/decision_curve.png`). With the raw, miscalibrated scores it showed no
  benefit — a concrete demonstration of why calibration matters clinically.
- **Fairness** — ROC-AUC by subgroup: gender spread 0.02, **age spread 0.08
  (worse for 70+: 0.65 vs <50: 0.73)**, race largely small-sample variance. See
  `results/fairness_analysis.txt`.
- **Explainability** — SHAP global + per-patient attributions (top driver:
  prior inpatient visits).

## Pipeline

```
Raw Clinical Data (101K encounters, 50 features)
        │
    Feature Engineering
    ├── ICD-9 diagnosis grouping (circulatory, respiratory, diabetes, ...)
    ├── Medication change aggregation (polypharmacy metrics)
    ├── Derived ratios (labs/procedures, meds/day, complexity score)
    └── Ordinal encoding (age, A1c, glucose)
        │
    XGBoost with Class Balancing
    ├── scale_pos_weight for imbalanced classes (~11% positive)
    ├── Early stopping on validation AUC
    └── 5-fold stratified cross-validation
        │
    Evaluation
    ├── ROC-AUC, PR-AUC, calibration curve
    ├── Sensitivity/Specificity at optimal threshold
    └── Risk score distribution
        │
    SHAP Explainability
    ├── Global feature importance (beeswarm + bar)
    ├── Feature dependence plots
    ├── Individual prediction explanations (waterfall)
    └── Clinical report with interpretation
```

## Quick Start

```bash
pip install -r requirements.txt

# Full pipeline (downloads dataset automatically)
# Includes model comparison, threshold tuning, fairness & calibration.
python main.py

# Skip SHAP for faster execution
python main.py --no-shap
```

> Model comparison (XGBoost / LightGBM / Logistic Regression), threshold
> optimization, subgroup fairness, and isotonic calibration now run on every
> execution. (`lightgbm` is optional — it is skipped gracefully if not installed.)

## Project Structure

```
04_readmission_prediction/
├── main.py                  # Entry point
├── requirements.txt
├── src/
│   ├── config.py            # Hyperparameters
│   ├── data_loader.py       # UCI dataset loading & feature engineering
│   ├── model.py             # XGBoost/LightGBM training & CV
│   ├── evaluate.py          # Metrics & visualization
│   ├── advanced_eval.py     # Model comparison, threshold, fairness, calibration
│   ├── explainability.py    # SHAP analysis & clinical report
│   └── inference.py         # Standalone scoring of new patients (CLI)
├── tests/
│   └── test_pipeline.py     # pytest: preprocessing, training, inference
├── conftest.py
├── data/
│   └── diabetic_data.csv    # Auto-downloaded
└── results/
    ├── roc_pr_curves.png          ├── model_comparison.{png,csv}
    ├── confusion_matrix.png       ├── threshold_analysis.{png,txt}
    ├── feature_importance.png     ├── fairness_analysis.{png,txt,csv}
    ├── calibration_curve.png      ├── calibration_comparison.png
    ├── risk_distribution.png      ├── calibration_metrics.txt
    ├── decision_curve.{png,txt,csv}  ├── shap_*.png
    ├── clinical_report.txt        ├── xgb_model.json + model_config.json
    └── evaluation_summary.txt
```

## Scoring new patients

```bash
python src/inference.py --data new_patients.csv \
    --model results/xgb_model.json --config results/model_config.json \
    --output results/predictions.csv
```

Preprocessing is applied in inference mode (no deceased-row removal, columns
aligned to the training feature set), so predictions map 1:1 to input patients.

## Tests

```bash
pip install pytest
pytest -q   # preprocessing cleanliness, training, inference roundtrip & column alignment
```

## Feature Engineering

Key engineered features from raw clinical data:

| Feature | Source | Clinical Rationale |
|---------|--------|--------------------|
| `diag_*_group` | ICD-9 codes | Groups 800+ diagnosis codes into 8 clinical categories |
| `num_med_changes` | 21 medication columns | Counts dose adjustments (Up/Down) — treatment instability |
| `num_meds_active` | 21 medication columns | Active medication count — polypharmacy burden |
| `lab_to_proc_ratio` | Lab + procedure counts | Clinical decision intensity |
| `meds_per_day` | Medications / LOS | Daily medication burden |
| `complexity_score` | Diagnoses x medications | Combined complexity metric |
| `age_ordinal` | Age buckets | Ordinal-encoded age for monotonic relationships |

## SHAP Explainability

The project generates multiple SHAP visualizations:

- **Beeswarm plot**: Shows feature impact direction and magnitude across all patients
- **Bar plot**: Global feature importance ranking
- **Dependence plots**: Non-linear relationships (e.g., how `number_inpatient` affects readmission risk)
- **Waterfall plot**: Explains individual predictions ("why was this patient flagged as high-risk?")
- **Clinical report**: Human-readable interpretation of top risk factors

## Dataset

**Diabetes 130-US Hospitals for Years 1999-2008** (UCI ML Repository #296)
- 101,766 encounters from 130 US hospitals
- 50 features: demographics, diagnoses, medications, procedures, lab results
- Target: readmission within 30 days (~11% positive rate)
- [Source & License](https://archive.ics.uci.edu/dataset/296)

## References

- Strack B et al. Impact of HbA1c Measurement on Hospital Readmission Rates. BioMed Research International 2014.
- Lundberg SM, Lee SI. A Unified Approach to Interpreting Model Predictions (SHAP). NeurIPS 2017.
- Chen T, Guestrin C. XGBoost: A Scalable Tree Boosting System. KDD 2016.
