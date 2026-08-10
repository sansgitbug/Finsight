import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.preprocessing.chunker import Chunk
from src.preprocessing.embeddings import Embedding
from src.retrieval.dense import DenseRetriever, DenseRetrieverConfig
from src.retrieval.vectorstore import VectorStoreConfig, build_vector_store


class FakeQueryModel:
    """Offline query encoder that shares the test store's two-vector space."""

    def encode(self, sentences, *, batch_size, show_progress_bar, normalize_embeddings):
        vectors = np.asarray([[1.0, 0.0] if "iphone" in text.lower() else [0.0, 1.0] for text in sentences], dtype=np.float32)
        return vectors


def make_embedding(chunk_id: str, vector: list[float], section_name: str) -> Embedding:
    """Construct a retrieval record using the existing chunk dataclass."""
    chunk = Chunk(chunk_id, "AAPL", "2026-07-31", "10-Q", section_name, f"Text for {chunk_id}", 4, 1, 1, 0, 20, {"source": "test"})
    return Embedding(chunk, np.asarray(vector, dtype=np.float32))


class RetrieverTests(unittest.TestCase):
    def test_retrieve_from_text_returns_cosine_ranked_search_results(self) -> None:
        # ChromaDB may retain native Windows file handles through interpreter exit.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            root = Path(directory)
            store = build_vector_store(
                [
                    make_embedding("iphone", [1.0, 0.0], "Management's Discussion and Analysis"),
                    make_embedding("risk", [0.0, 1.0], "Risk Factors"),
                ],
                VectorStoreConfig(root / "faiss.index", root / "chroma", "retriever_chunks"),
            )
            retriever = DenseRetriever(
                DenseRetrieverConfig(top_k=2, embedding_model="offline-test-model", device="cpu"),
                model=FakeQueryModel(),
                vector_store=store,
            )

            results = retriever.retrieve_from_text("How did iPhone revenue change?")

            self.assertEqual([result.chunk.chunk_id for result in results], ["iphone", "risk"])
            self.assertGreater(results[0].score, results[1].score)
            self.assertEqual(results[0].metadata, {"source": "test"})

    def test_embed_query_rejects_blank_query(self) -> None:
        retriever = DenseRetriever(
            DenseRetrieverConfig(embedding_model="offline-test-model", device="cpu"),
            model=FakeQueryModel(),
            vector_store=None,
            )

        with self.assertRaisesRegex(ValueError, "non-empty"):
            retriever.embed_query("   ")


if __name__ == "__main__":
    unittest.main()
