#!/usr/bin/env python3
"""k-nearest-neighbor approved-twin retrieval for the AR dataset.

Only proxy-approved claims are indexed. Each eligible denied claim is searched
against approved claims with the same insurer and Primary/Secondary status.
The distance threshold is learned from approved-to-approved neighbor distances.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ADMIN_EXCLUSION_STATUS_CODES = {
    "Incorrect Submission",
    "Provider Info Missing",
    "Claim not on file",
}

CATEGORICAL_FEATURES = ["Client"]
NUMERIC_FEATURES = [
    "log_billed_amount",
    "filing_delay_days",
    "Aging Days",
    "month_sin",
    "month_cos",
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
    df["log_billed_amount"] = np.log1p(df["Billed Amount"].clip(lower=0))
    df["month_sin"] = np.sin(2 * np.pi * df["dos_month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["dos_month"] / 12.0)

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


def build_preprocessor() -> ColumnTransformer:
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    numerical = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    return ColumnTransformer([
        ("categorical", categorical, CATEGORICAL_FEATURES),
        ("numerical", numerical, NUMERIC_FEATURES),
    ], sparse_threshold=0)


def circular_month_difference(month_a: float, month_b: float) -> float:
    difference = abs(float(month_a) - float(month_b))
    return min(difference, 12.0 - difference)


def relative_difference(value_a: float, value_b: float) -> float:
    denominator = max(abs(float(value_a)), abs(float(value_b)), 1.0)
    return abs(float(value_a) - float(value_b)) / denominator


def calibrate_distance_threshold(
    df: pd.DataFrame,
    vectors: np.ndarray,
    quantile: float,
) -> float:
    distances: list[float] = []
    approved = df[df["target"].eq(1)]

    for _, group in approved.groupby(HARD_BLOCK_COLUMNS, dropna=False):
        indices = group.index.to_numpy()
        if len(indices) < 2:
            continue
        model = NearestNeighbors(n_neighbors=2, metric="euclidean")
        model.fit(vectors[indices])
        group_distances, _ = model.kneighbors(vectors[indices])
        distances.extend(group_distances[:, 1].tolist())

    if not distances:
        raise ValueError("Not enough approved controls to calibrate k-NN distance.")

    return float(np.quantile(distances, quantile))


def run_knn(
    data_path: str,
    control_quantile: float = 0.75,
    neighbors_to_check: int = 3,
    max_amount_difference: float = 0.35,
    max_filing_delay_difference: int = 15,
    max_aging_difference: int = 120,
    max_month_difference: int = 3,
) -> tuple[dict[str, list[str]], pd.DataFrame, dict]:
    if not 0 < control_quantile < 1:
        raise ValueError("control_quantile must be between 0 and 1.")

    df = prepare_data(data_path)
    features = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    preprocessor = build_preprocessor()
    vectors = np.asarray(preprocessor.fit_transform(df[features]), dtype=float)

    distance_threshold = calibrate_distance_threshold(
        df, vectors, control_quantile
    )

    approved = df[df["target"].eq(1)].copy()
    denied = df[df["target"].eq(0)].copy()
    evidence_rows: list[dict] = []

    for denied_index, claim in denied.iterrows():
        pool = approved.copy()
        for column in HARD_BLOCK_COLUMNS:
            pool = pool[pool[column].eq(claim[column])]
        if pool.empty:
            continue

        pool_indices = pool.index.to_numpy()
        n_neighbors = min(neighbors_to_check, len(pool_indices))
        model = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        model.fit(vectors[pool_indices])
        distances, neighbor_positions = model.kneighbors(vectors[[denied_index]])

        selected = None
        for distance, position in zip(distances[0], neighbor_positions[0]):
            twin = pool.iloc[int(position)]
            amount_difference = relative_difference(
                twin["Billed Amount"], claim["Billed Amount"]
            )
            filing_difference = abs(
                float(twin["filing_delay_days"]) - float(claim["filing_delay_days"])
            )
            aging_difference = abs(
                float(twin["Aging Days"]) - float(claim["Aging Days"])
            )
            month_difference = circular_month_difference(
                twin["dos_month"], claim["dos_month"]
            )

            if (
                distance <= distance_threshold
                and amount_difference <= max_amount_difference
                and filing_difference <= max_filing_delay_difference
                and aging_difference <= max_aging_difference
                and month_difference <= max_month_difference
            ):
                selected = {
                    "denied_item_id": claim["VisitID#"],
                    "approved_twin_id": twin["VisitID#"],
                    "denial_status_code": claim["Status Code"],
                    "action_code": claim["Action Code"],
                    "insurance_name": claim["Insurance Name"],
                    "primary_secondary": claim["Primary/Secondary"],
                    "knn_distance": float(distance),
                    "distance_threshold": float(distance_threshold),
                    "amount_difference_pct": float(amount_difference),
                    "filing_delay_difference": float(filing_difference),
                    "aging_difference": float(aging_difference),
                    "dos_month_difference": float(month_difference),
                }
                break

        if selected is not None:
            evidence_rows.append(selected)

    evidence = pd.DataFrame(evidence_rows)
    item_ids = (
        evidence["denied_item_id"].drop_duplicates().tolist()
        if not evidence.empty else []
    )

    diagnostics = {
        "proxy_approved_count": int((df["target"] == 1).sum()),
        "eligible_denied_count": int((df["target"] == 0).sum()),
        "control_quantile": float(control_quantile),
        "distance_threshold": float(distance_threshold),
    }
    return {"k-NN": item_ids}, evidence, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="claims.csv")
    parser.add_argument("--json-output", default="knn_results.json")
    parser.add_argument("--evidence-output", default="knn_evidence.csv")
    parser.add_argument("--diagnostics-output", default="knn_diagnostics.json")
    parser.add_argument("--control-quantile", type=float, default=0.75)
    args = parser.parse_args()

    result, evidence, diagnostics = run_knn(
        args.data,
        control_quantile=args.control_quantile,
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
