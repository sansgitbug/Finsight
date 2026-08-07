"""SEC EDGAR ingestion utilities for FinSight.

This module resolves a public-company ticker to its CIK, retrieves the most
recent quarterly reports from SEC EDGAR, and persists normalized filing text
alongside metadata suitable for downstream retrieval indexing.

The SEC requests a descriptive ``User-Agent`` containing contact information.
Set :class:`EdgarIngestionConfig.user_agent` to identify your application
before using this module in a deployed service.
"""

from __future__ import annotations

import gzip
import json
import logging
import re
import time
import zlib
from argparse import ArgumentParser, Namespace
from dataclasses import asdict, dataclass
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from threading import Lock
from typing import Any, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

LOGGER = logging.getLogger(__name__)

SEC_DATA_BASE_URL = "https://data.sec.gov"
SEC_ARCHIVES_BASE_URL = "https://www.sec.gov/Archives/edgar/data"
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUPPORTED_FORMS = frozenset({"10-Q", "10-K"})


class EdgarIngestionError(Exception):
    """Base exception raised when SEC EDGAR ingestion cannot be completed."""


class TickerNotFoundError(EdgarIngestionError):
    """Raised when a ticker cannot be resolved to a SEC CIK."""


class SecRequestError(EdgarIngestionError):
    """Raised after an SEC request fails or returns an unexpected response."""


class FilingDownloadError(EdgarIngestionError):
    """Raised when a filing document cannot be downloaded or persisted."""


@dataclass(frozen=True, slots=True)
class EdgarIngestionConfig:
    """Configuration for SEC EDGAR ingestion.

    Attributes:
        data_dir: Root directory where ticker-specific filing directories live.
        user_agent: SEC-compliant application identification string with contact.
        request_timeout_seconds: Per-request connect/read timeout.
        max_retries: Number of retry attempts for transient SEC failures.
        backoff_factor_seconds: Exponential-backoff base duration.
        requests_per_second: Maximum request rate sent by this client instance.
        filing_limit: Number of most-recent 10-Q/10-K filings to persist.
    """

    data_dir: Path = Path("data")
    user_agent: str = "FinSight/1.0 contact@example.com"
    request_timeout_seconds: float = 30.0
    max_retries: int = 5
    backoff_factor_seconds: float = 1.0
    requests_per_second: float = 8.0
    filing_limit: int = 6

    def __post_init__(self) -> None:
        """Validate options that influence network safety and output size."""
        if not self.user_agent.strip():
            raise ValueError("user_agent must be a non-empty SEC-compliant identifier")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be greater than zero")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if self.backoff_factor_seconds < 0:
            raise ValueError("backoff_factor_seconds cannot be negative")
        if self.requests_per_second <= 0:
            raise ValueError("requests_per_second must be greater than zero")
        if self.filing_limit <= 0:
            raise ValueError("filing_limit must be greater than zero")


@dataclass(frozen=True, slots=True)
class FilingMetadata:
    """Metadata persisted with a locally downloaded SEC filing."""

    ticker: str
    company_name: str
    filing_type: str
    accession_number: str
    filing_date: str
    source_url: str
    local_file_path: str


@dataclass(frozen=True, slots=True)
class CompanyIdentity:
    """SEC identity data for a listed company."""

    ticker: str
    cik: str
    company_name: str


@dataclass(frozen=True, slots=True)
class _SecResponse:
    """Minimal HTTP response value returned by the SEC transport."""

    text: str
    headers: Mapping[str, str]


class _TextExtractor(HTMLParser):
    """Small dependency-free HTML-to-text extractor for SEC filing documents."""

    _BLOCK_TAGS = frozenset({"br", "div", "p", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._ignored_depth += 1
        elif tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._ignored_depth:
            self._ignored_depth -= 1
        elif tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.parts.append(data)

    def text(self) -> str:
        """Return readable text while retaining paragraph boundaries."""
        lines = (re.sub(r"\s+", " ", line).strip() for line in "".join(self.parts).splitlines())
        return "\n".join(line for line in lines if line)


class EdgarClient:
    """Rate-limited SEC EDGAR client with retries for transient failures."""

    def __init__(self, config: EdgarIngestionConfig) -> None:
        self.config = config
        self._last_request_at = 0.0
        self._rate_limit_lock = Lock()

    def get(self, url: str) -> _SecResponse:
        """Execute a rate-limited GET request and raise a domain-specific error."""
        request = Request(
            url,
            headers={"User-Agent": self.config.user_agent, "Accept-Encoding": "gzip, deflate"},
            method="GET",
        )
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            self._wait_for_rate_limit()
            try:
                with urlopen(request, timeout=self.config.request_timeout_seconds) as response:
                    charset = response.headers.get_content_charset() or "utf-8"
                    return _SecResponse(
                        text=_decode_response_body(
                            response.read(), response.headers.get("Content-Encoding"), charset
                        ),
                        headers=dict(response.headers.items()),
                    )
            except HTTPError as exc:
                last_error = exc
                if exc.code not in {429, 500, 502, 503, 504}:
                    break
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                self._sleep_before_retry(attempt, retry_after)
            except (URLError, TimeoutError, OSError) as exc:
                last_error = exc
                self._sleep_before_retry(attempt, None)
        raise SecRequestError(f"SEC request failed for {url}: {last_error}") from last_error

    def _sleep_before_retry(self, attempt: int, retry_after: str | None) -> None:
        """Sleep before another transient request attempt, honoring Retry-After."""
        if attempt >= self.config.max_retries:
            return
        delay = self.config.backoff_factor_seconds * (2**attempt)
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                LOGGER.debug("Ignoring non-numeric SEC Retry-After value: %s", retry_after)
        LOGGER.warning("SEC request failed; retrying in %.2f seconds (attempt %d)", delay, attempt + 1)
        time.sleep(delay)

    def _wait_for_rate_limit(self) -> None:
        """Enforce the configured request ceiling across threads using this client."""
        interval = 1.0 / self.config.requests_per_second
        with self._rate_limit_lock:
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < interval:
                time.sleep(interval - elapsed)
            self._last_request_at = time.monotonic()


def _decode_response_body(body: bytes, content_encoding: str | None, charset: str) -> str:
    """Decode an SEC response body after applying its declared content encoding."""
    encodings = [encoding.strip().lower() for encoding in (content_encoding or "").split(",") if encoding.strip()]
    try:
        for encoding in reversed(encodings):
            if encoding == "identity":
                continue
            if encoding == "gzip":
                body = gzip.decompress(body)
            elif encoding == "deflate":
                try:
                    body = zlib.decompress(body)
                except zlib.error:
                    body = zlib.decompress(body, -zlib.MAX_WBITS)
            else:
                raise SecRequestError(f"SEC returned an unsupported Content-Encoding: {encoding}")
        return body.decode(charset, errors="replace")
    except (OSError, zlib.error) as exc:
        encoding = content_encoding or "identity"
        raise SecRequestError(f"Unable to decode SEC response with Content-Encoding: {encoding}") from exc


def resolve_ticker(ticker: str, client: EdgarClient) -> CompanyIdentity:
    """Resolve ``ticker`` to its zero-padded CIK and SEC company title.

    Args:
        ticker: Exchange ticker symbol, case-insensitive.
        client: Configured client used to retrieve the SEC ticker mapping.

    Raises:
        TickerNotFoundError: If EDGAR has no company record for the ticker.
        SecRequestError: If SEC ticker data cannot be requested or decoded.
    """
    normalized_ticker = _normalize_ticker(ticker)
    try:
        payload = json.loads(client.get(TICKER_MAP_URL).text)
    except (ValueError, SecRequestError) as exc:
        if isinstance(exc, SecRequestError):
            raise
        raise SecRequestError("SEC returned invalid company ticker JSON") from exc

    for record in _mapping_values(payload):
        if str(record.get("ticker", "")).upper() == normalized_ticker:
            cik_value = record.get("cik_str")
            title = str(record.get("title", "")).strip()
            if cik_value is None or not title:
                raise SecRequestError(f"SEC ticker record for {normalized_ticker} is incomplete")
            return CompanyIdentity(normalized_ticker, f"{int(cik_value):010d}", title)

    raise TickerNotFoundError(f"Ticker {normalized_ticker!r} was not found in SEC EDGAR")


def fetch_recent_filings(company: CompanyIdentity, client: EdgarClient, limit: int) -> list[dict[str, str]]:
    """Fetch metadata for the latest ``limit`` 10-Q and 10-K filings.

    SEC's submissions response includes a current ``recent`` array and may
    reference historical submission files. Both sources are read so companies
    with sparse current submissions are handled correctly.
    """
    submissions_url = f"{SEC_DATA_BASE_URL}/submissions/CIK{company.cik}.json"
    try:
        payload = json.loads(client.get(submissions_url).text)
    except ValueError as exc:
        raise SecRequestError(f"SEC returned invalid submissions JSON for {company.ticker}") from exc

    filing_rows = list(_filing_rows(payload.get("filings", {}).get("recent", {})))
    for historical_file in payload.get("filings", {}).get("files", []):
        if len(_select_supported_filings(filing_rows, limit)) >= limit:
            break
        name = historical_file.get("name")
        if not isinstance(name, str) or not name:
            continue
        try:
            historical_payload = json.loads(client.get(f"{SEC_DATA_BASE_URL}/submissions/{name}").text)
        except ValueError as exc:
            raise SecRequestError(f"SEC returned invalid historical submissions JSON: {name}") from exc
        filing_rows.extend(_filing_rows(historical_payload))

    selected = _select_supported_filings(filing_rows, limit)
    if not selected:
        raise EdgarIngestionError(f"No 10-Q or 10-K filings found for {company.ticker}")
    return selected


def download_filing_text(filing: Mapping[str, str], company: CompanyIdentity, client: EdgarClient) -> tuple[str, str]:
    """Download one filing's primary document and normalize it to plain text.

    Returns:
        A tuple of ``(text, source_url)``.
    """
    accession_number = filing["accessionNumber"]
    primary_document = filing.get("primaryDocument", "")
    if not primary_document:
        raise FilingDownloadError(f"Filing {accession_number} has no primary document")
    accession_without_dashes = accession_number.replace("-", "")
    source_url = f"{SEC_ARCHIVES_BASE_URL}/{int(company.cik)}/{accession_without_dashes}/{primary_document}"
    try:
        response = client.get(source_url)
    except SecRequestError as exc:
        raise FilingDownloadError(f"Unable to download filing {accession_number}") from exc

    content_type = response.headers.get("Content-Type", "").lower()
    raw_text = response.text
    text = _html_to_text(raw_text) if "html" in content_type or "<html" in raw_text[:1000].lower() else raw_text.strip()
    if not text:
        raise FilingDownloadError(f"Filing {accession_number} downloaded without readable text")
    return text, source_url


def save_filing(
    filing_text: str,
    filing: Mapping[str, str],
    company: CompanyIdentity,
    source_url: str,
    config: EdgarIngestionConfig,
) -> FilingMetadata:
    """Persist one filing text file and its JSON metadata atomically enough for indexing."""
    filing_date = _validate_filing_date(filing["filingDate"])
    output_directory = config.data_dir / company.ticker / filing_date
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
        text_path = output_directory / "filing.txt"
        text_path.write_text(filing_text, encoding="utf-8")
        metadata = FilingMetadata(
            ticker=company.ticker,
            company_name=company.company_name,
            filing_type=filing["form"],
            accession_number=filing["accessionNumber"],
            filing_date=filing_date,
            source_url=source_url,
            local_file_path=str(text_path.resolve()),
        )
        (output_directory / "metadata.json").write_text(
            json.dumps(asdict(metadata), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise FilingDownloadError(f"Unable to save filing {filing['accessionNumber']}: {exc}") from exc
    return metadata


def ingest_company(ticker: str, config: EdgarIngestionConfig | None = None) -> list[FilingMetadata]:
    """Download and persist the latest quarterly filings for a company ticker.

    Args:
        ticker: Public company ticker, such as ``AAPL`` or ``MSFT``.
        config: Optional EDGAR ingestion settings. Defaults are suitable for
            local development; set a real contact address in production.

    Returns:
        Metadata for each persisted filing, newest first.

    Raises:
        EdgarIngestionError: If ticker resolution, retrieval, download, or
            persistence fails.
    """
    active_config = config or EdgarIngestionConfig()
    client = EdgarClient(active_config)
    company = resolve_ticker(ticker, client)
    LOGGER.info("Resolved ticker %s to CIK %s (%s)", company.ticker, company.cik, company.company_name)
    filings = fetch_recent_filings(company, client, active_config.filing_limit)
    LOGGER.info("Found %d recent supported filings for %s", len(filings), company.ticker)

    persisted: list[FilingMetadata] = []
    for filing in filings:
        LOGGER.info("Downloading %s filed %s", filing["form"], filing["filingDate"])
        filing_text, source_url = download_filing_text(filing, company, client)
        persisted.append(save_filing(filing_text, filing, company, source_url, active_config))
    LOGGER.info("Persisted %d filings for %s", len(persisted), company.ticker)
    return persisted


def _normalize_ticker(ticker: str) -> str:
    """Validate and canonicalize a ticker symbol."""
    normalized = ticker.strip().upper()
    if not normalized or not re.fullmatch(r"[A-Z0-9.\-]{1,15}", normalized):
        raise ValueError("ticker must contain 1-15 letters, digits, periods, or hyphens")
    return normalized


def _mapping_values(payload: Any) -> Iterable[Mapping[str, Any]]:
    """Return values from SEC's numeric-keyed company ticker mapping."""
    if not isinstance(payload, Mapping):
        raise SecRequestError("SEC company ticker response is not an object")
    values = payload.values()
    if not all(isinstance(record, Mapping) for record in values):
        raise SecRequestError("SEC company ticker response contains malformed records")
    return values


def _filing_rows(columns: Mapping[str, Any]) -> Iterable[dict[str, str]]:
    """Convert SEC's columnar submissions arrays into validated filing rows."""
    if not isinstance(columns, Mapping):
        return []
    forms = columns.get("form", [])
    if not isinstance(forms, Sequence) or isinstance(forms, (str, bytes)):
        raise SecRequestError("SEC submissions response has an invalid form column")
    required = ("accessionNumber", "filingDate", "primaryDocument")
    rows: list[dict[str, str]] = []
    for index, form in enumerate(forms):
        if form not in SUPPORTED_FORMS:
            continue
        row: dict[str, str] = {"form": str(form)}
        for field in required:
            values = columns.get(field, [])
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or index >= len(values):
                LOGGER.warning("Skipping malformed SEC submission row %d: missing %s", index, field)
                break
            value = values[index]
            if not isinstance(value, str) or not value:
                LOGGER.warning("Skipping malformed SEC submission row %d: empty %s", index, field)
                break
            row[field] = value
        else:
            rows.append(row)
    return rows


def _select_supported_filings(rows: Iterable[dict[str, str]], limit: int) -> list[dict[str, str]]:
    """Deduplicate filings and select newest records by filing date."""
    unique = {row["accessionNumber"]: row for row in rows}
    return sorted(unique.values(), key=lambda row: row["filingDate"], reverse=True)[:limit]


def _html_to_text(html: str) -> str:
    """Extract readable plain text from an SEC HTML filing."""
    parser = _TextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception as exc:  # HTMLParser rarely fails, but preserve context if it does.
        raise FilingDownloadError("Unable to parse SEC filing HTML") from exc
    return parser.text()


def _validate_filing_date(value: str) -> str:
    """Validate the SEC filing date used as an output-directory component."""
    try:
        return date.fromisoformat(value).isoformat()
    except (TypeError, ValueError) as exc:
        raise FilingDownloadError(f"Invalid SEC filing date: {value!r}") from exc


def main() -> int:
    """Run SEC filing ingestion from the command line.

    The command intentionally requires an explicit SEC user-agent value rather
    than sending an anonymous request from a production batch job.
    """
    parser = ArgumentParser(description="Download recent SEC 10-Q and 10-K filings.")
    parser.add_argument("ticker", help="Public company ticker, for example AAPL")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Output root directory")
    parser.add_argument("--user-agent", required=True, help="SEC-compliant app/contact identifier")
    parser.add_argument("--filing-limit", type=int, default=6, help="Number of filings to download")
    arguments: Namespace = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        ingest_company(
            arguments.ticker,
            EdgarIngestionConfig(
                data_dir=arguments.data_dir,
                user_agent=arguments.user_agent,
                filing_limit=arguments.filing_limit,
            ),
        )
    except (EdgarIngestionError, ValueError) as exc:
        LOGGER.error("Ingestion failed: %s", exc)
        return 1
    return 0


__all__ = [
    "CompanyIdentity",
    "EdgarClient",
    "EdgarIngestionConfig",
    "EdgarIngestionError",
    "FilingDownloadError",
    "FilingMetadata",
    "SecRequestError",
    "TickerNotFoundError",
    "download_filing_text",
    "fetch_recent_filings",
    "ingest_company",
    "main",
    "resolve_ticker",
    "save_filing",
]


if __name__ == "__main__":
    raise SystemExit(main())
