#!/usr/bin/env python3
"""FastAPI wrapper for the three Ghost Denials candidate generators.

Proxy-approved definition used by all algorithms:
    Status == "Claim at insurance"
    AND Status Code == "Claim in Process"

Candidate population:
    Status == "Denied"
    excluding clearly administrative/incomplete-submission status codes.

The public response intentionally contains only denied VisitID# values:
{
    "exact_match": [...],
    "logistic_regression": [...],
    "k-NN": [...]
}
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field, field_validator

from exact_match import run_exact_match
from logistic_regression import run_propensity_matching
from knn import run_knn

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "claims.csv"

# Scikit-learn jobs are small for this dataset, but the lock prevents several
# simultaneous requests from repeating CPU-heavy cross-validation needlessly.
AUDIT_LOCK = Lock()

app = FastAPI(
    title="Ghost Denials Audit API",
    version="1.0.0",
    description=(
        "Runs exact equivalence matching, logistic-regression propensity "
        "matching, and k-nearest-neighbor approved-twin retrieval."
    ),
)


class AuditRequest(BaseModel):
    """Optional algorithm settings for the bundled claims.csv dataset."""

    logistic_caliper_multiplier: float = Field(default=0.20, gt=0, le=2.0)
    knn_control_quantile: float = Field(default=0.75, gt=0, lt=1)
    max_items: int | None = Field(default=None, gt=0)


class AuditResponse(BaseModel):
    """Exact response shape required by the consumer."""

    model_config = ConfigDict(populate_by_name=True)

    exact_match: list[str]
    logistic_regression: list[str]
    k_nn: list[str] = Field(alias="k-NN")

    @field_validator("exact_match", "logistic_regression", "k_nn")
    @classmethod
    def unique_ids(cls, values: list[str]) -> list[str]:
        # Preserve algorithm ranking/order while removing accidental duplicates.
        return list(dict.fromkeys(str(value) for value in values))


def _limit(values: list[str], max_items: int | None) -> list[str]:
    return values if max_items is None else values[:max_items]


def run_all_algorithms(
    data_path: str | Path,
    *,
    logistic_caliper_multiplier: float = 0.20,
    knn_control_quantile: float = 0.75,
    max_items: int | None = None,
) -> AuditResponse:
    """Run all three algorithms and combine their denied item IDs."""

    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with AUDIT_LOCK:
        exact_result, _exact_evidence = run_exact_match(str(path))

        logistic_result, _logistic_evidence, _logistic_diagnostics = (
            run_propensity_matching(
                str(path),
                caliper_multiplier=logistic_caliper_multiplier,
            )
        )

        knn_result, _knn_evidence, _knn_diagnostics = run_knn(
            str(path),
            control_quantile=knn_control_quantile,
        )

    return AuditResponse(
        exact_match=_limit(exact_result["exact_match"], max_items),
        logistic_regression=_limit(
            logistic_result["logistic_regression"], max_items
        ),
        k_nn=_limit(knn_result["k-NN"], max_items),
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/audit",
    response_model=AuditResponse,
    response_model_by_alias=True,
)
def audit(request: AuditRequest | None = None) -> AuditResponse:
    """Run all algorithms on the bundled claims.csv dataset."""

    params = request or AuditRequest()
    try:
        return run_all_algorithms(
            DEFAULT_DATA_PATH,
            logistic_caliper_multiplier=params.logistic_caliper_multiplier,
            knn_control_quantile=params.knn_control_quantile,
            max_items=params.max_items,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Audit failed: {exc}",
        ) from exc


@app.post(
    "/audit/upload",
    response_model=AuditResponse,
    response_model_by_alias=True,
)
def audit_uploaded_csv(
    file: UploadFile = File(..., description="AR claims CSV"),
    logistic_caliper_multiplier: float = Form(default=0.20, gt=0, le=2.0),
    knn_control_quantile: float = Form(default=0.75, gt=0, lt=1),
    max_items: int | None = Form(default=None, gt=0),
) -> AuditResponse:
    """Upload a CSV and run all three algorithms against it."""

    filename = file.filename or "claims.csv"
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".csv",
            delete=False,
        ) as temporary_file:
            shutil.copyfileobj(file.file, temporary_file)
            temporary_path = Path(temporary_file.name)

        return run_all_algorithms(
            temporary_path,
            logistic_caliper_multiplier=logistic_caliper_multiplier,
            knn_control_quantile=knn_control_quantile,
            max_items=max_items,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Audit failed: {exc}",
        ) from exc
    finally:
        file.file.close()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
