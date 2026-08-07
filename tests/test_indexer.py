import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.retrieval.indexer import Indexer, IndexerConfig


class FakeEmbeddingModel:
    """Offline encoder with predictable non-zero vectors for indexer tests."""

    def encode(self, sentences, *, batch_size, show_progress_bar, normalize_embeddings):
        return np.asarray([[float(index + 1), float(len(text))] for index, text in enumerate(sentences)], dtype=np.float32)


def chunk_record(chunk_id: str, section_name: str) -> dict[str, object]:
    """Create serialized chunk data matching the chunker output schema."""
    return {
        "chunk_id": chunk_id,
        "ticker": "AAPL",
        "filing_date": "2026-07-31",
        "filing_type": "10-Q",
        "section_name": section_name,
        "text": f"Text for {chunk_id}",
        "token_count": 4,
        "start_line": 1,
        "end_line": 1,
        "character_start": 0,
        "character_end": 20,
        "metadata": {"source": "test"},
    }


class IndexerTests(unittest.TestCase):
    def test_index_recursively_discovers_chunks_and_skips_duplicate_ids(self) -> None:
        # ChromaDB can keep a Windows native file handle until interpreter exit.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            root = Path(directory)
            first = root / "chunks" / "AAPL" / "2026-07-31"
            second = root / "chunks" / "AAPL" / "2026-05-01"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            (first / "chunks.json").write_text(json.dumps([chunk_record("shared", "Business"), chunk_record("unique", "Risk Factors")]), encoding="utf-8")
            (second / "chunks.json").write_text(json.dumps([chunk_record("shared", "Other Information")]), encoding="utf-8")
            config = IndexerConfig(first.parents[1], root / "embeddings", root / "vectorstore", "offline-test-model", 2, "cpu")

            indexer = Indexer(config, FakeEmbeddingModel())
            store = indexer.index()

            self.assertEqual(indexer.chunk_file_count, 2)
            self.assertEqual([chunk.chunk_id for chunk in indexer.chunks], ["shared", "unique"])
            self.assertEqual(store.dimension, 2)
            self.assertEqual(store.index.ntotal, 2)
            self.assertEqual(store.collection.count(), 2)
            self.assertTrue((root / "embeddings" / "embeddings.npz").is_file())
            self.assertTrue((root / "vectorstore" / "faiss.index").is_file())
            self.assertTrue((root / "vectorstore" / "faiss.index.mapping.json").is_file())


if __name__ == "__main__":
    unittest.main()
