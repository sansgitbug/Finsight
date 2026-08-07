import json
import re
import tempfile
import unittest
from pathlib import Path

from src.chunker import ChunkingConfig, build_chunks, chunk_filing, detect_sections


class WhitespaceTokenizer:
    """Small deterministic tokenizer used to test chunk boundaries offline."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return list(range(len(re.findall(r"\S+", text))))


class SectionDetectionTests(unittest.TestCase):
    def test_detect_sections_keeps_body_heading_over_table_of_contents(self) -> None:
        text = (
            "TABLE OF CONTENTS\nItem 1. Financial Statements\nItem 2. Management's Discussion and Analysis\n\n"
            "ITEM 1. FINANCIAL STATEMENTS\nBalance sheet discussion.\n\n"
            "ITEM 2. Management's Discussion and Analysis of Financial Condition and Results of Operations\nAnalysis.\n"
        )

        sections = detect_sections(text)

        self.assertEqual([section.name for section in sections], ["Financial Statements", "Management's Discussion and Analysis"])
        self.assertTrue(sections[0].text.startswith("ITEM 1."))

    def test_uses_document_section_when_no_reliable_headings_exist(self) -> None:
        text = "First paragraph of an unstructured filing.\n\nSecond paragraph."

        sections = detect_sections(text)

        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0].name, "Document")
        self.assertEqual(sections[0].text, text)


class ChunkBuildingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ChunkingConfig("offline-test-tokenizer", max_tokens=8, overlap_tokens=2, minimum_chunk_size=2)
        self.tokenizer = WhitespaceTokenizer()
        self.metadata = {"ticker": "AAPL", "filing_date": "2026-07-31", "filing_type": "10-Q", "company_name": "Apple Inc."}

    def test_splits_only_oversized_sections_with_bounded_chunks(self) -> None:
        text = (
            "ITEM 1. FINANCIAL STATEMENTS\nSmall section remains whole.\n\n"
            "ITEM 2. Management's Discussion and Analysis\n"
            "One two three four. Five six seven eight. Nine ten eleven twelve."
        )

        chunks = build_chunks(detect_sections(text), self.metadata, self.config, self.tokenizer)

        financial = [chunk for chunk in chunks if chunk.section_name == "Financial Statements"]
        analysis = [chunk for chunk in chunks if chunk.section_name == "Management's Discussion and Analysis"]
        self.assertEqual(len(financial), 1)
        self.assertGreater(len(analysis), 1)
        self.assertTrue(all(chunk.token_count <= self.config.max_tokens for chunk in chunks))
        self.assertTrue(all(chunk.character_start < chunk.character_end for chunk in chunks))

    def test_chunk_filing_writes_required_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            filing_directory = Path(directory) / "data" / "AAPL" / "2026-07-31"
            filing_directory.mkdir(parents=True)
            (filing_directory / "filing.txt").write_text("ITEM 1. FINANCIAL STATEMENTS\nShort content.", encoding="utf-8")
            (filing_directory / "metadata.json").write_text(json.dumps(self.metadata), encoding="utf-8")
            output_path = Path(directory) / "chunks" / "AAPL" / "2026-07-31" / "chunks.json"

            chunks = chunk_filing(filing_directory / "filing.txt", output_path=output_path, config=self.config, tokenizer=self.tokenizer)

            persisted = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(len(persisted), len(chunks))
            self.assertEqual(
                set(persisted[0]),
                {"chunk_id", "ticker", "filing_date", "filing_type", "section_name", "text", "token_count", "start_line", "end_line", "character_start", "character_end", "metadata"},
            )


if __name__ == "__main__":
    unittest.main()
