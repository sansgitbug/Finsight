import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.preprocessing.chunker import Chunk
from src.preprocessing.embeddings import EmbeddingConfig, embed_chunks, load_chunks, load_embeddings, save_embeddings


class FakeEmbeddingModel:
    """Offline model double that emits stable, non-zero vectors."""

    def encode(self, sentences, *, batch_size, show_progress_bar, normalize_embeddings):
        vectors = np.asarray([[float(len(text)), float(index + 1)] for index, text in enumerate(sentences)], dtype=np.float32)
        if normalize_embeddings:
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors


def make_chunk(chunk_id: str = "AAPL-2026-07-31-0001") -> Chunk:
    """Construct a representative chunk shared by embedding tests."""
    return Chunk(chunk_id, "AAPL", "2026-07-31", "10-Q", "Business", "Apple builds products.", 4, 1, 1, 0, 22, {"company_name": "Apple Inc."})


class EmbeddingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = EmbeddingConfig("offline-test-model", batch_size=2, normalize_embeddings=True, device="cpu")

    def test_load_chunks_and_embed_chunks_preserves_chunk_associations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunks.json"
            path.write_text(json.dumps([{
                "chunk_id": "AAPL-2026-07-31-0001", "ticker": "AAPL", "filing_date": "2026-07-31", "filing_type": "10-Q", "section_name": "Business", "text": "Apple builds products.", "token_count": 4, "start_line": 1, "end_line": 1, "character_start": 0, "character_end": 22, "metadata": {"company_name": "Apple Inc."}
            }]), encoding="utf-8")

            chunks = load_chunks(path)
            embeddings = embed_chunks(chunks, self.config, FakeEmbeddingModel())

            self.assertEqual(embeddings[0].chunk_id, chunks[0].chunk_id)
            self.assertEqual(embeddings[0].chunk.metadata, {"company_name": "Apple Inc."})
            self.assertAlmostEqual(float(np.linalg.norm(embeddings[0].vector)), 1.0)

    def test_save_and_load_embeddings_round_trip_vectors_and_chunks(self) -> None:
        embeddings = embed_chunks([make_chunk()], self.config, FakeEmbeddingModel())
        with tempfile.TemporaryDirectory() as directory:
            output = save_embeddings(embeddings, Path(directory) / "embeddings")
            loaded = load_embeddings(output)

            self.assertEqual(output.name, "embeddings.npz")
            self.assertEqual(loaded[0].chunk, embeddings[0].chunk)
            np.testing.assert_allclose(loaded[0].vector, embeddings[0].vector)


if __name__ == "__main__":
    unittest.main()
