"""HW2 submission module. The autograder imports exactly these names.

The grader loads the data exactly as the starter does (original column names, text columns as
pandas 'category'), selects FEATURES, fits build_pipeline() on its own training split and scores
predict_proba(...)[:, 1] with ROC AUC on a held-out split you do not see.
"""
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Pre-call features only (leak column "duration" excluded)
FEATURES: list[str] = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
]

# Memo decision is deploy_baseline (test CI vs MLP includes 0)
CHOSEN_MODEL: str = "logistic_regression"

# Tuned hyperparameters from equal-budget validation search
_CLF_PARAMS = {
    "C": 4.832930238571752,
    "penalty": "l2",
    "solver": "lbfgs",
    "class_weight": "balanced",
    "max_iter": 2000,
    "random_state": 0,
}


def build_pipeline() -> BaseEstimator:
    """Return an UNFITTED classifier (preprocessing included) with tuned hyperparameters."""
    cat_cols = [
        c
        for c in [
            "job",
            "marital",
            "education",
            "default",
            "housing",
            "loan",
            "contact",
            "month",
            "poutcome",
        ]
        if c in FEATURES
    ]
    num_cols = [
        c
        for c in ["age", "balance", "day", "campaign", "pdays", "previous"]
        if c in FEATURES
    ]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )
    clf = LogisticRegression(**_CLF_PARAMS)
    return Pipeline([("pre", pre), ("clf", clf)])
