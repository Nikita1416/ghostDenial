# Ghost Denials FastAPI

This API wraps three separate candidate-generation methods:

1. Exact equivalence-class matching
2. Logistic-regression propensity matching
3. k-nearest-neighbor approved-twin retrieval

## Proxy label used

A row is treated as a proxy-approved/reference claim only when:

```text
Status = Claim at insurance
AND Status Code = Claim in Process
```

A candidate is a row with `Status = Denied`, excluding configured administrative/incomplete-submission status codes.

## Response

```json
{
  "exact_match": ["denied VisitID#"],
  "logistic_regression": ["denied VisitID#"],
  "k-NN": ["denied VisitID#"]
}
```

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Windows activation:

```powershell
.venv\Scripts\activate
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

## Run with bundled dataset

```bash
curl -X POST http://127.0.0.1:8000/audit \
  -H "Content-Type: application/json" \
  -d '{}'
```

Optional settings:

```bash
curl -X POST http://127.0.0.1:8000/audit \
  -H "Content-Type: application/json" \
  -d '{
    "logistic_caliper_multiplier": 0.20,
    "knn_control_quantile": 0.75,
    "max_items": 10
  }'
```

## Upload a CSV

```bash
curl -X POST http://127.0.0.1:8000/audit/upload \
  -F "file=@claims.csv" \
  -F "logistic_caliper_multiplier=0.20" \
  -F "knn_control_quantile=0.75"
```

## Docker

```bash
docker build -t ghost-denials-api .
docker run --rm -p 8000:8000 ghost-denials-api
```

## Notes

- `Status`, `Status Code`, `Action Code`, assignment fields, follow-up date, and AR notes are not prediction features.
- Action Code is retained by the individual algorithm modules only for evidence/routing.
- The public FastAPI response intentionally contains only item-ID lists.
- This dataset supports a proxy similarity POC, not appeal-overturn or conformal false-positive validation.
