#!/usr/bin/env python3
"""Logistic-regression propensity matching for the AR Ghost Denials dataset.

The historical proxy decision function is learned from:
    1 = Status == "Claim at insurance" AND Status Code == "Claim in Process"
    0 = Status == "Denied"

Status, Status Code, Action Code, routing fields, follow-up date and AR notes are
NOT model features. Action Code is retained only in the evidence output.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ADMIN_EXCLUSION_STATUS_CODES = {
    "Incorrect Submission",
    "Provider Info Missing",
    "Claim not on file",
}

CATEGORICAL_FEATURES = [
    "Insurance Name",
    "Primary/Secondary",
    "Client",
]

NUMERIC_FEATURES = [
    "Billed Amount",
    "filing_delay_days",
    "Aging Days",
    "dos_month",
]

HARD_BLOCK_COLUMNS = ["Insurance Name", "Primary/Secondary"]


def prepare_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {
        "VisitID#", "DOS", "Submitted Date", "Insurance Name", "Status",
        "Status Code", "Action Code", "Primary/Secondary", "Client",
        "Billed Amount", "Aging Days",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for column in [
        "Status", "Status Code", "Action Code", "Insurance Name",
        "Primary/Secondary", "Client",
    ]:
        df[column] = df[column].astype(str).str.strip()

    df["DOS"] = pd.to_datetime(df["DOS"], errors="coerce")
    df["Submitted Date"] = pd.to_datetime(df["Submitted Date"], errors="coerce")
    df["filing_delay_days"] = (df["Submitted Date"] - df["DOS"]).dt.days
    df["dos_month"] = df["DOS"].dt.month

    proxy_approved = (
        df["Status"].eq("Claim at insurance")
        & df["Status Code"].eq("Claim in Process")
    )
    eligible_denied = (
        df["Status"].eq("Denied")
        & ~df["Status Code"].isin(ADMIN_EXCLUSION_STATUS_CODES)
    )

    model_df = df[proxy_approved | eligible_denied].copy()
    model_df["target"] = proxy_approved.loc[model_df.index].astype(int)
    return model_df.reset_index(drop=True)


def build_pipeline() -> Pipeline:
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    numerical = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer([
        ("categorical", categorical, CATEGORICAL_FEATURES),
        ("numerical", numerical, NUMERIC_FEATURES),
    ])

    # No class_weight: propensity probabilities should represent the observed
    # historical proxy decision process rather than a rebalanced class prior.
    model = LogisticRegression(
        max_iter=3000,
        C=0.5,
        solver="liblinear",
        random_state=42,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def circular_month_difference(month_a: float, month_b: float) -> float:
    difference = abs(float(month_a) - float(month_b))
    return min(difference, 12.0 - difference)


def relative_difference(value_a: float, value_b: float) -> float:
    denominator = max(abs(float(value_a)), abs(float(value_b)), 1.0)
    return abs(float(value_a) - float(value_b)) / denominator


def run_propensity_matching(
    data_path: str,
    caliper_multiplier: float = 0.20,
    max_amount_difference: float = 0.35,
    max_filing_delay_difference: int = 15,
    max_aging_difference: int = 120,
    max_month_difference: int = 3,
) -> tuple[dict[str, list[str]], pd.DataFrame, dict]:
    df = prepare_data(data_path)
    features = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    X = df[features]
    y = df["target"]

    if y.nunique() != 2:
        raise ValueError("Both proxy-approved and denied classes are required.")

    pipeline = build_pipeline()
    folds = min(5, int(y.value_counts().min()))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    # Cross-fitted scores reduce direct in-sample overfitting in this small POC.
    propensity = cross_val_predict(
        pipeline,
        X,
        y,
        cv=cv,
        method="predict_proba",
    )[:, 1]

    df["propensity_score"] = propensity
    clipped = np.clip(propensity, 1e-6, 1 - 1e-6)
    df["logit_propensity"] = np.log(clipped / (1 - clipped))

    caliper = caliper_multiplier * float(df["logit_propensity"].std(ddof=1))
    auc = float(roc_auc_score(y, propensity))

    approved = df[df["target"].eq(1)].copy()
    denied = df[df["target"].eq(0)].copy()
    evidence_rows: list[dict] = []

    for _, claim in denied.iterrows():
        pool = approved.copy()
        for column in HARD_BLOCK_COLUMNS:
            pool = pool[pool[column].eq(claim[column])]
        if pool.empty:
            continue

        pool = pool.copy()
        pool["propensity_gap"] = (
            pool["logit_propensity"] - claim["logit_propensity"]
        ).abs()
        pool["amount_difference_pct"] = pool["Billed Amount"].apply(
            lambda value: relative_difference(value, claim["Billed Amount"])
        )
        pool["filing_delay_difference"] = (
            pool["filing_delay_days"] - claim["filing_delay_days"]
        ).abs()
        pool["aging_difference"] = (
            pool["Aging Days"] - claim["Aging Days"]
        ).abs()
        pool["month_difference"] = pool["dos_month"].apply(
            lambda month: circular_month_difference(month, claim["dos_month"])
        )

        valid = pool[
            pool["propensity_gap"].le(caliper)
            & pool["amount_difference_pct"].le(max_amount_difference)
            & pool["filing_delay_difference"].le(max_filing_delay_difference)
            & pool["aging_difference"].le(max_aging_difference)
            & pool["month_difference"].le(max_month_difference)
        ]
        if valid.empty:
            continue

        twin = valid.sort_values(
            ["propensity_gap", "amount_difference_pct", "filing_delay_difference"]
        ).iloc[0]

        evidence_rows.append({
            "denied_item_id": claim["VisitID#"],
            "approved_twin_id": twin["VisitID#"],
            "denial_status_code": claim["Status Code"],
            "action_code": claim["Action Code"],
            "insurance_name": claim["Insurance Name"],
            "primary_secondary": claim["Primary/Secondary"],
            "denied_propensity": float(claim["propensity_score"]),
            "approved_propensity": float(twin["propensity_score"]),
            "logit_propensity_gap": float(twin["propensity_gap"]),
            "caliper": float(caliper),
            "amount_difference_pct": float(twin["amount_difference_pct"]),
            "filing_delay_difference": int(twin["filing_delay_difference"]),
            "aging_difference": int(twin["aging_difference"]),
            "dos_month_difference": float(twin["month_difference"]),
        })

    evidence = pd.DataFrame(evidence_rows)
    item_ids = (
        evidence["denied_item_id"].drop_duplicates().tolist()
        if not evidence.empty else []
    )

    diagnostics = {
        "proxy_approved_count": int((y == 1).sum()),
        "eligible_denied_count": int((y == 0).sum()),
        "cross_validated_roc_auc": auc,
        "logit_caliper": float(caliper),
        "warning": (
            "The model has weak discrimination; use results as exploratory similarity candidates."
            if auc < 0.55 else None
        ),
    }
    return {"logistic_regression": item_ids}, evidence, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="claims.csv")
    parser.add_argument("--json-output", default="logistic_regression_results.json")
    parser.add_argument("--evidence-output", default="logistic_regression_evidence.csv")
    parser.add_argument("--diagnostics-output", default="logistic_regression_diagnostics.json")
    parser.add_argument("--caliper-multiplier", type=float, default=0.20)
    args = parser.parse_args()

    result, evidence, diagnostics = run_propensity_matching(
        args.data,
        caliper_multiplier=args.caliper_multiplier,
    )

    Path(args.json_output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    Path(args.diagnostics_output).write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    evidence.to_csv(args.evidence_output, index=False)

    print(json.dumps(result, indent=2))
    print(json.dumps(diagnostics, indent=2))
    print(f"Evidence saved to: {args.evidence_output}")


if __name__ == "__main__":
    main()
