# FinSight

FinSight downloads recent SEC EDGAR 10-Q and 10-K filings and stores each
filing's plain text and metadata under `data/<TICKER>/<FILING_DATE>/`.

## Requirements

FinSight uses only the Python standard library. Python 3.10 or later is
required for the type-hint syntax used by the ingestion module.

## Run ingestion

From the repository root, run:

```powershell
python -m src.ingest AAPL --user-agent "FinSight/1.0 your-email@example.com"
```

Use `--data-dir` to select another output location; it defaults to `data`.
The SEC requires a descriptive user-agent with contact information.

## Run tests

```powershell
python -m unittest discover -s tests -v
```
