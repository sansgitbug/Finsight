import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.chunker import Chunk
from src.embeddings import Embedding
from src.vectorstore import VectorStoreConfig, add_embeddings, build_vector_store, load, search


def make_embedding(chunk_id: str, vector: list[float], section_name: str = "Business") -> Embedding:
    """Construct a compact embedding record with complete chunk metadata."""
    chunk = Chunk(chunk_id, "AAPL", "2026-07-31", "10-Q", section_name, f"Text for {chunk_id}", 4, 1, 1, 0, 20, {"source": "test"})
    return Embedding(chunk, np.asarray(vector, dtype=np.float32))


class VectorStoreTests(unittest.TestCase):
    def test_search_returns_cosine_sorted_reranker_ready_results(self) -> None:
        # ChromaDB's Windows native backend can retain a file handle until the
        # interpreter exits, so cleanup errors must not mask test assertions.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            root = Path(directory)
            config = VectorStoreConfig(root / "index.faiss", root / "chroma", "test_chunks")
            store = build_vector_store([
                make_embedding("chunk-business", [1.0, 0.0]),
                make_embedding("chunk-risk", [0.0, 1.0], "Risk Factors"),
            ], config)

            results = search(store, np.asarray([0.9, 0.1], dtype=np.float32), top_k=2)

            self.assertEqual([result.chunk.chunk_id for result in results], ["chunk-business", "chunk-risk"])
            self.assertGreater(results[0].score, results[1].score)
            self.assertEqual(results[0].metadata, {"source": "test"})
            self.assertEqual(results[0].chunk.section_name, "Business")

    def test_load_and_upsert_use_explicit_faiss_id_mapping(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            root = Path(directory)
            config = VectorStoreConfig(root / "index.faiss", root / "chroma", "test_chunks")
            store = build_vector_store([make_embedding("chunk-one", [1.0, 0.0])], config)
            initial_id = next(iter(store.id_map))
            add_embeddings(store, [make_embedding("chunk-one", [0.0, 1.0], "Updated")])
            from src.vectorstore import save

            save(store)
            loaded = load(config)
            results = search(loaded, [0.0, 1.0], top_k=1)

            self.assertEqual(next(iter(loaded.id_map)), initial_id)
            self.assertEqual(loaded.id_map[initial_id], "chunk-one")
            self.assertEqual(results[0].chunk.section_name, "Updated")


if __name__ == "__main__":
    unittest.main()
