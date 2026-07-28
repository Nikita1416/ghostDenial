#!/usr/bin/env python3
"""Exact equivalence-class matching for the synthetic AR Ghost Denials dataset.

Proxy-approved definition:
    Status == "Claim at insurance" AND Status Code == "Claim in Process"

Denied audit candidates:
    Status == "Denied", excluding clearly administrative/incomplete-submission
    status codes configured in ADMIN_EXCLUSION_STATUS_CODES.

The script returns only denied VisitID# values in:
    {"exact_match": [...]}

It also writes a detailed evidence CSV containing the matched proxy-approved twin.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ADMIN_EXCLUSION_STATUS_CODES = {
    "Incorrect Submission",
    "Provider Info Missing",
    "Claim not on file",
}

# Exact equivalence is performed on governed categorical/banded attributes.
EQUIVALENCE_COLUMNS = [
    "Insurance Name",
    "Primary/Secondary",
    "Client",
    "billed_amount_band",
    "filing_delay_band",
]


def prepare_data(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)

    required = {
        "VisitID#", "DOS", "Submitted Date", "Insurance Name", "Status",
        "Status Code", "Action Code", "Primary/Secondary", "Client",
        "Billed Amount", "Aging Days",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    text_columns = [
        "Status", "Status Code", "Action Code", "Insurance Name",
        "Primary/Secondary", "Client",
    ]
    for column in text_columns:
        df[column] = df[column].astype(str).str.strip()

    df["DOS"] = pd.to_datetime(df["DOS"], errors="coerce")
    df["Submitted Date"] = pd.to_datetime(df["Submitted Date"], errors="coerce")
    df["filing_delay_days"] = (df["Submitted Date"] - df["DOS"]).dt.days

    # Governed bins avoid requiring identical raw amounts/dates.
    df["billed_amount_band"] = pd.cut(
        df["Billed Amount"],
        bins=[-np.inf, 250, 500, 750, 1000, np.inf],
        labels=["<=250", "251-500", "501-750", "751-1000", "1000+"],
    ).astype(str)

    df["filing_delay_band"] = pd.cut(
        df["filing_delay_days"],
        bins=[-np.inf, 15, 30, 45, 60, np.inf],
        labels=["<=15", "16-30", "31-45", "46-60", "60+"],
    ).astype(str)

    approved = df[
        df["Status"].eq("Claim at insurance")
        & df["Status Code"].eq("Claim in Process")
    ].copy()

    denied = df[
        df["Status"].eq("Denied")
        & ~df["Status Code"].isin(ADMIN_EXCLUSION_STATUS_CODES)
    ].copy()

    return approved, denied


def run_exact_match(data_path: str) -> tuple[dict[str, list[str]], pd.DataFrame]:
    approved, denied = prepare_data(data_path)

    approved_groups = {
        key: group.copy()
        for key, group in approved.groupby(EQUIVALENCE_COLUMNS, dropna=False)
    }

    evidence_rows: list[dict] = []

    for _, claim in denied.iterrows():
        key = tuple(claim[column] for column in EQUIVALENCE_COLUMNS)
        pool = approved_groups.get(key)
        if pool is None or pool.empty:
            continue

        pool = pool.copy()
        pool["amount_difference"] = (
            pool["Billed Amount"] - claim["Billed Amount"]
        ).abs()
        pool["filing_delay_difference"] = (
            pool["filing_delay_days"] - claim["filing_delay_days"]
        ).abs()

        twin = pool.sort_values(
            ["amount_difference", "filing_delay_difference", "VisitID#"]
        ).iloc[0]

        evidence_rows.append({
            "denied_item_id": claim["VisitID#"],
            "approved_twin_id": twin["VisitID#"],
            "denial_status_code": claim["Status Code"],
            "action_code": claim["Action Code"],
            "insurance_name": claim["Insurance Name"],
            "primary_secondary": claim["Primary/Secondary"],
            "client": claim["Client"],
            "denied_billed_amount": float(claim["Billed Amount"]),
            "approved_billed_amount": float(twin["Billed Amount"]),
            "amount_difference": float(twin["amount_difference"]),
            "denied_filing_delay_days": int(claim["filing_delay_days"]),
            "approved_filing_delay_days": int(twin["filing_delay_days"]),
            "filing_delay_difference": int(twin["filing_delay_difference"]),
            "equivalence_class": "|".join(map(str, key)),
        })

    evidence = pd.DataFrame(evidence_rows)
    item_ids = (
        evidence["denied_item_id"].drop_duplicates().tolist()
        if not evidence.empty else []
    )
    return {"exact_match": item_ids}, evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="claims.csv")
    parser.add_argument("--json-output", default="exact_match_results.json")
    parser.add_argument("--evidence-output", default="exact_match_evidence.csv")
    args = parser.parse_args()

    result, evidence = run_exact_match(args.data)

    Path(args.json_output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    evidence.to_csv(args.evidence_output, index=False)

    print(json.dumps(result, indent=2))
    print(f"Candidate count: {len(result['exact_match'])}")
    print(f"Evidence saved to: {args.evidence_output}")


if __name__ == "__main__":
    main()
