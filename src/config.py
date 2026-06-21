from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    data_dir: Path = Path("data")
    results_dir: Path = Path("results")
    random_state: int = 42

    # Target: readmission within 30 days
    target_column: str = "readmitted"
    positive_label: str = "<30"

    # Feature engineering
    age_bins: tuple[str, ...] = (
        "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
        "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)",
    )

    # Model
    test_size: float = 0.2
    val_size: float = 0.15
    n_folds: int = 5

    # XGBoost
    xgb_params: dict = None

    def __post_init__(self):
        if self.xgb_params is None:
            object.__setattr__(self, "xgb_params", {
                "n_estimators": 500,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 5,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "scale_pos_weight": 1.0,
                "random_state": 42,
                "n_jobs": -1,
                "eval_metric": "auc",
            })
