"""Section-aware chunking for locally persisted SEC filings.

The chunker preserves detected SEC item boundaries and only creates overlapping
subsections when an individual section is too large for the configured model
context.  When filings lack recognizable item headings, it groups paragraphs
and sentences instead of applying arbitrary character or token windows.
"""

from __future__ import annotations

import json
import logging
import re
from argparse import ArgumentParser, Namespace
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

from transformers import AutoTokenizer, PreTrainedTokenizerBase

LOGGER = logging.getLogger(__name__)


class ChunkingError(Exception):
    """Raised when a filing cannot be read, chunked, or written safely."""


class Tokenizer(Protocol):
    """Subset of the Hugging Face tokenizer API required by this module."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        """Encode text into model token identifiers."""


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Options controlling tokenization and semantic chunk boundaries.

    Attributes:
        tokenizer_name: Hugging Face tokenizer identifier or local tokenizer path.
        max_tokens: Maximum number of model tokens in a chunk.
        overlap_tokens: Maximum semantic overlap between split section chunks.
        minimum_chunk_size: Soft lower bound for chunks created by a split.
    """

    tokenizer_name: str = "bert-base-uncased"
    max_tokens: int = 800
    overlap_tokens: int = 100
    minimum_chunk_size: int = 100

    def __post_init__(self) -> None:
        """Validate context-window settings before a tokenizer is loaded."""
        if not self.tokenizer_name.strip():
            raise ValueError("tokenizer_name must be non-empty")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")
        if self.overlap_tokens < 0 or self.overlap_tokens >= self.max_tokens:
            raise ValueError("overlap_tokens must be non-negative and smaller than max_tokens")
        if self.minimum_chunk_size <= 0 or self.minimum_chunk_size > self.max_tokens:
            raise ValueError("minimum_chunk_size must be between 1 and max_tokens")


@dataclass(frozen=True, slots=True)
class Section:
    """A contiguous semantic region of a filing, with source coordinates."""

    name: str
    start_line: int
    end_line: int
    start_character: int
    end_character: int
    text: str


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrieval-ready filing fragment with section and source metadata."""

    chunk_id: str
    ticker: str
    filing_date: str
    filing_type: str
    section_name: str
    text: str
    token_count: int
    start_line: int
    end_line: int
    character_start: int
    character_end: int
    metadata: dict[str, Any]


_SECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Business", re.compile(r"(?:item\s+1\.?\s*)?business\b", re.IGNORECASE)),
    ("Risk Factors", re.compile(r"(?:item\s+1a\.?\s*)?risk factors\b", re.IGNORECASE)),
    (
        "Management's Discussion and Analysis",
        re.compile(r"(?:item\s+2\.?\s*)?(?:management(?:['’]s)?\s+discussion|md&a).*", re.IGNORECASE),
    ),
    ("Financial Statements", re.compile(r"(?:item\s+1\.?\s*)?financial statements\b", re.IGNORECASE)),
    ("Notes to Financial Statements", re.compile(r"notes? to (?:the )?financial statements\b", re.IGNORECASE)),
    (
        "Quantitative and Qualitative Disclosures About Market Risk",
        re.compile(r"(?:item\s+3\.?\s*)?quantitative and qualitative disclosures about market risk\b", re.IGNORECASE),
    ),
    ("Controls and Procedures", re.compile(r"(?:item\s+4\.?\s*)?controls and procedures\b", re.IGNORECASE)),
    ("Legal Proceedings", re.compile(r"(?:item\s+1\.?\s*)?legal proceedings\b", re.IGNORECASE)),
    ("Other Information", re.compile(r"(?:item\s+5\.?\s*)?other information\b", re.IGNORECASE)),
    ("Exhibits", re.compile(r"(?:item\s+6\.?\s*)?exhibits?(?: and financial statement schedules)?\b", re.IGNORECASE)),
)


def load_filing(filing_path: Path, metadata_path: Path) -> tuple[str, dict[str, Any]]:
    """Read a filing text file and validate its ingestion metadata JSON.

    Raises:
        ChunkingError: If either input is unavailable, invalid, or incomplete.
    """
    try:
        text = filing_path.read_text(encoding="utf-8")
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ChunkingError(f"Unable to load filing inputs from {filing_path.parent}: {exc}") from exc
    if not text.strip():
        raise ChunkingError(f"Filing text is empty: {filing_path}")
    if not isinstance(payload, dict):
        raise ChunkingError(f"Filing metadata must be a JSON object: {metadata_path}")
    required = ("ticker", "filing_date", "filing_type")
    missing = [field for field in required if not isinstance(payload.get(field), str) or not payload[field].strip()]
    if missing:
        raise ChunkingError(f"Filing metadata is missing required fields: {', '.join(missing)}")
    return text, payload


def detect_sections(text: str) -> list[Section]:
    """Return recognized SEC sections, or one document section as a fallback.

    Repeated headings in a table of contents are discarded in favor of the
    final occurrence, which is normally the actual filing content heading.
    """
    if not text:
        return []
    line_starts = _line_starts(text)
    lines = text.splitlines(keepends=True)
    candidates: list[tuple[str, int, int]] = []
    for index, raw_line in enumerate(lines):
        heading = raw_line.strip()
        if index + 1 < len(lines) and _looks_like_item_label(heading):
            heading = f"{heading} {lines[index + 1].strip()}"
        section_name = _recognized_heading(heading)
        if section_name is not None:
            candidates.append((section_name, index, line_starts[index]))
    # SEC tables of contents duplicate the body headings. Retain the body copy.
    final_by_name = {name: (line, character) for name, line, character in candidates}
    headings = sorted(((name, line, character) for name, (line, character) in final_by_name.items()), key=lambda item: item[2])
    if not headings:
        LOGGER.warning("No reliable SEC headings found; using paragraph-aware document fallback")
        return [_make_section("Document", 0, 0, text, line_starts)]

    sections: list[Section] = []
    for position, (name, line, character) in enumerate(headings):
        end_character = headings[position + 1][2] if position + 1 < len(headings) else len(text)
        section_text = text[character:end_character]
        end_line = _line_number_for_character(line_starts, max(character, end_character - 1))
        sections.append(Section(name, line + 1, end_line + 1, character, end_character, section_text))
    LOGGER.info("Detected %d SEC sections", len(sections))
    return sections


def split_large_section(section: Section, tokenizer: Tokenizer, config: ChunkingConfig) -> list[Section]:
    """Split one oversized section on paragraph and sentence boundaries.

    Returned sections retain source ranges. Overlap is added only here, and is
    composed of whole semantic units when they fit inside ``overlap_tokens``.
    """
    if _token_count(section.text, tokenizer) <= config.max_tokens:
        return [section]
    units = _semantic_units(section.text)
    fragments = _group_units(section.text, units, tokenizer, config)
    line_starts = _line_starts(section.text)
    split_sections: list[Section] = []
    for start, end in fragments:
        fragment = section.text[start:end]
        start_line = section.start_line + _line_number_for_character(line_starts, start)
        end_line = section.start_line + _line_number_for_character(line_starts, max(start, end - 1))
        split_sections.append(
            Section(section.name, start_line, end_line, section.start_character + start, section.start_character + end, fragment)
        )
    LOGGER.info("Split oversized %s section into %d semantic chunks", section.name, len(split_sections))
    return split_sections


def build_chunks(
    sections: Iterable[Section], metadata: Mapping[str, Any], config: ChunkingConfig, tokenizer: Tokenizer | None = None
) -> list[Chunk]:
    """Convert sections into retrieval chunks without crossing section boundaries."""
    active_tokenizer = tokenizer or _load_tokenizer(config)
    ticker = _metadata_string(metadata, "ticker").upper()
    filing_date = _metadata_string(metadata, "filing_date")
    filing_type = _metadata_string(metadata, "filing_type")
    base_metadata = dict(metadata)
    chunks: list[Chunk] = []
    for section in sections:
        if not section.text.strip():
            continue
        for piece in split_large_section(section, active_tokenizer, config):
            token_count = _token_count(piece.text, active_tokenizer)
            chunks.append(
                Chunk(
                    chunk_id=f"{ticker}-{filing_date}-{len(chunks) + 1:04d}",
                    ticker=ticker,
                    filing_date=filing_date,
                    filing_type=filing_type,
                    section_name=piece.name,
                    text=piece.text,
                    token_count=token_count,
                    start_line=piece.start_line,
                    end_line=piece.end_line,
                    character_start=piece.start_character,
                    character_end=piece.end_character,
                    metadata=base_metadata,
                )
            )
    if not chunks:
        raise ChunkingError("No non-empty chunks could be built from filing")
    LOGGER.info("Built %d chunks for %s %s", len(chunks), ticker, filing_date)
    return chunks


def save_chunks(chunks: Sequence[Chunk], output_path: Path) -> Path:
    """Persist chunks as UTF-8 JSON, creating parent directories as needed."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps([asdict(chunk) for chunk in chunks], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except OSError as exc:
        raise ChunkingError(f"Unable to save chunks to {output_path}: {exc}") from exc
    LOGGER.info("Saved %d chunks to %s", len(chunks), output_path)
    return output_path


def chunk_filing(
    filing_path: Path,
    metadata_path: Path | None = None,
    output_path: Path | None = None,
    config: ChunkingConfig | None = None,
    tokenizer: Tokenizer | None = None,
) -> list[Chunk]:
    """Load, semantically chunk, and save one filing.

    By default chunks are written to ``chunks/<ticker>/<filing_date>/chunks.json``
    relative to the filing's ``data`` directory parent.
    """
    metadata_input = metadata_path or filing_path.with_name("metadata.json")
    text, metadata = load_filing(filing_path, metadata_input)
    active_config = config or ChunkingConfig()
    sections = detect_sections(text)
    chunks = build_chunks(sections, metadata, active_config, tokenizer)
    destination = output_path or filing_path.parents[2].parent / "chunks" / _metadata_string(metadata, "ticker").upper() / _metadata_string(metadata, "filing_date") / "chunks.json"
    save_chunks(chunks, destination)
    return chunks


def chunk_directory(data_dir: Path, chunks_dir: Path | None = None, config: ChunkingConfig | None = None) -> list[Path]:
    """Chunk every filing directory under a ``data`` tree and return outputs."""
    if not data_dir.is_dir():
        raise ChunkingError(f"Data directory does not exist: {data_dir}")
    active_config = config or ChunkingConfig()
    destination_root = chunks_dir or data_dir.parent / "chunks"
    tokenizer = _load_tokenizer(active_config)
    outputs: list[Path] = []
    for filing_path in sorted(data_dir.glob("*/*/filing.txt")):
        metadata_path = filing_path.with_name("metadata.json")
        if not metadata_path.is_file():
            LOGGER.warning("Skipping filing without metadata: %s", filing_path)
            continue
        _, metadata = load_filing(filing_path, metadata_path)
        destination = destination_root / _metadata_string(metadata, "ticker").upper() / _metadata_string(metadata, "filing_date") / "chunks.json"
        chunk_filing(filing_path, metadata_path, destination, active_config, tokenizer)
        outputs.append(destination)
    LOGGER.info("Chunked %d filings from %s", len(outputs), data_dir)
    return outputs


def _load_tokenizer(config: ChunkingConfig) -> PreTrainedTokenizerBase:
    """Load the configured Hugging Face tokenizer with a clear domain error."""
    try:
        return AutoTokenizer.from_pretrained(config.tokenizer_name)
    except (OSError, ValueError) as exc:
        raise ChunkingError(f"Unable to load tokenizer {config.tokenizer_name!r}: {exc}") from exc


def _recognized_heading(value: str) -> str | None:
    """Return a canonical section name for a short, heading-like line."""
    compact = re.sub(r"\s+", " ", value.strip())
    if not compact or len(compact) > 180:
        return None
    for name, pattern in _SECTION_PATTERNS:
        if pattern.fullmatch(compact):
            return name
    return None


def _looks_like_item_label(value: str) -> bool:
    """Identify an isolated SEC item label that may precede its title."""
    return bool(re.fullmatch(r"item\s+\d+[a-z]?\.?(?:\s+)?", value.strip(), re.IGNORECASE))


def _make_section(name: str, start_line: int, start_character: int, text: str, line_starts: Sequence[int]) -> Section:
    """Build a section from a source offset and its text."""
    end_character = start_character + len(text)
    end_line = _line_number_for_character(line_starts, max(start_character, end_character - 1))
    return Section(name, start_line + 1, end_line + 1, start_character, end_character, text)


def _semantic_units(text: str) -> list[tuple[int, int]]:
    """Locate paragraph units, falling back to sentence units when necessary."""
    paragraphs = [(match.start(), match.end()) for match in re.finditer(r"(?s)\S.*?(?=\n\s*\n|\Z)", text) if match.group().strip()]
    if len(paragraphs) > 1:
        return paragraphs
    sentences = [(match.start(), match.end()) for match in re.finditer(r"(?s)\S.*?(?:[.!?](?=\s|$)|\Z)", text) if match.group().strip()]
    return sentences or [(0, len(text))]


def _group_units(
    text: str, units: Sequence[tuple[int, int]], tokenizer: Tokenizer, config: ChunkingConfig
) -> list[tuple[int, int]]:
    """Pack semantic units under the token limit and add bounded unit overlap."""
    groups: list[tuple[int, int]] = []
    pending = [
        piece
        for unit in units
        for piece in _split_oversized_unit(text, unit, tokenizer, config.max_tokens)
    ]
    while pending:
        start, end = pending.pop(0)
        current = [(start, end)]
        while pending:
            candidate = current + [pending[0]]
            candidate_text = "".join(text[unit_start:unit_end] for unit_start, unit_end in candidate)
            if _token_count(candidate_text, tokenizer) > config.max_tokens:
                break
            current.append(pending.pop(0))
        groups.append((current[0][0], current[-1][1]))
        overlap: list[tuple[int, int]] = []
        for unit in reversed(current):
            overlap_text = "".join(text[unit_start:unit_end] for unit_start, unit_end in [unit, *overlap])
            if _token_count(overlap_text, tokenizer) > config.overlap_tokens:
                break
            overlap.insert(0, unit)
        # Re-adding an entire one-unit group would make no forward progress.
        if pending and overlap and len(overlap) < len(current):
            pending = overlap + pending
    if len(groups) > 1:
        previous_start, previous_end = groups[-2]
        last_start, last_end = groups[-1]
        last_size = _token_count(text[last_start:last_end], tokenizer)
        merged_size = _token_count(text[previous_start:last_end], tokenizer)
        if last_size < config.minimum_chunk_size and merged_size <= config.max_tokens:
            groups[-2:] = [(previous_start, last_end)]
    return groups


def _split_oversized_unit(text: str, unit: tuple[int, int], tokenizer: Tokenizer, max_tokens: int) -> list[tuple[int, int]]:
    """Split an exceptional oversized sentence at whitespace where possible."""
    start, end = unit
    if _token_count(text[start:end], tokenizer) <= max_tokens:
        return [unit]
    pieces: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        candidate_end = _largest_fitting_end(text, cursor, end, tokenizer, max_tokens)
        whitespace = text.rfind(" ", cursor + 1, candidate_end + 1)
        if whitespace > cursor:
            candidate_end = whitespace + 1
        if candidate_end <= cursor:
            raise ChunkingError("Tokenizer cannot produce a non-empty chunk within max_tokens")
        pieces.append((cursor, candidate_end))
        cursor = candidate_end
    return pieces


def _largest_fitting_end(text: str, start: int, end: int, tokenizer: Tokenizer, max_tokens: int) -> int:
    """Find the furthest character offset that does not exceed ``max_tokens``."""
    low, high, best = start + 1, end, start
    while low <= high:
        midpoint = (low + high) // 2
        if _token_count(text[start:midpoint], tokenizer) <= max_tokens:
            best = midpoint
            low = midpoint + 1
        else:
            high = midpoint - 1
    return best


def _token_count(text: str, tokenizer: Tokenizer) -> int:
    """Count tokens without adding model-specific special tokens."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def _line_starts(text: str) -> list[int]:
    """Return zero-based character positions for every line in text."""
    return [0, *(match.end() for match in re.finditer(r"\n", text))]


def _line_number_for_character(line_starts: Sequence[int], character: int) -> int:
    """Return the zero-based line containing ``character``."""
    from bisect import bisect_right

    return max(0, bisect_right(line_starts, character) - 1)


def _metadata_string(metadata: Mapping[str, Any], key: str) -> str:
    """Read a required non-empty metadata string after prior validation."""
    value = metadata.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ChunkingError(f"Filing metadata is missing required field: {key}")
    return value.strip()


def main() -> int:
    """Run filing chunking from the command line."""
    parser = ArgumentParser(description="Create section-aware chunks from downloaded SEC filings.")
    parser.add_argument("data_dir", type=Path, nargs="?", default=Path("data"), help="Ingested filing data root")
    parser.add_argument("--chunks-dir", type=Path, default=Path("chunks"), help="Chunk output root")
    parser.add_argument("--tokenizer", default="bert-base-uncased", help="Hugging Face tokenizer name or path")
    parser.add_argument("--max-tokens", type=int, default=800, help="Maximum tokens per chunk")
    parser.add_argument("--overlap-tokens", type=int, default=100, help="Overlap for split sections")
    parser.add_argument("--minimum-chunk-size", type=int, default=100, help="Soft minimum chunk size")
    arguments: Namespace = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        chunk_directory(arguments.data_dir, arguments.chunks_dir, ChunkingConfig(arguments.tokenizer, arguments.max_tokens, arguments.overlap_tokens, arguments.minimum_chunk_size))
    except (ChunkingError, ValueError) as exc:
        LOGGER.error("Chunking failed: %s", exc)
        return 1
    return 0


__all__ = ["Chunk", "ChunkingConfig", "ChunkingError", "Section", "build_chunks", "chunk_directory", "chunk_filing", "detect_sections", "load_filing", "save_chunks", "split_large_section"]


if __name__ == "__main__":
    raise SystemExit(main())
