"""Build FinSight's persisted embedding, FAISS, and ChromaDB retrieval index."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.chunker import Chunk
from src.embeddings import Embedding, EmbeddingConfig, EmbeddingError, EmbeddingModel, embed_chunks, load_chunks, save_embeddings
from src.vectorstore import VectorStore, VectorStoreConfig, VectorStoreError, build_vector_store as create_vector_store

LOGGER = logging.getLogger(__name__)


class IndexingError(Exception):
    """Raised when chunks cannot be transformed into a persisted retrieval index."""


@dataclass(frozen=True, slots=True)
class IndexerConfig:
    """Input, output, and shared embedding settings for index construction."""

    chunks_directory: Path = Path("chunks")
    embeddings_directory: Path = Path("embeddings")
    vectorstore_directory: Path = Path("vectorstore")
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    device: str = "cpu"

    def __post_init__(self) -> None:
        """Validate configuration before model loading or persistent writes."""
        if not self.chunks_directory.is_dir():
            raise ValueError(f"chunks_directory does not exist: {self.chunks_directory}")
        EmbeddingConfig(model_name=self.embedding_model, batch_size=self.batch_size, device=self.device)


class Indexer:
    """Coordinate chunk discovery, embedding, and durable retrieval-index creation."""

    def __init__(self, config: IndexerConfig | None = None, model: EmbeddingModel | None = None) -> None:
        self.config = config or IndexerConfig()
        self._model = model
        self.chunk_file_count = 0
        self.chunks: list[Chunk] = []
        self.embeddings: list[Embedding] = []
        self.vector_store: VectorStore | None = None
        self.processing_seconds = 0.0

    def discover_chunks(self) -> list[Chunk]:
        """Recursively load ``chunks.json`` files and retain the first unique ID.

        Duplicate identifiers are skipped deterministically in path order. This
        makes re-running the pipeline safe when chunk files overlap.
        """
        paths = sorted(self.config.chunks_directory.rglob("chunks.json"))
        self.chunk_file_count = len(paths)
        LOGGER.info("Found %d chunk files under %s", self.chunk_file_count, self.config.chunks_directory)
        unique: dict[str, Chunk] = {}
        duplicates = 0
        for path in paths:
            try:
                records = load_chunks(path)
            except EmbeddingError as exc:
                raise IndexingError(f"Unable to load chunk file {path}: {exc}") from exc
            for chunk in records:
                if chunk.chunk_id in unique:
                    duplicates += 1
                    LOGGER.warning("Skipping duplicate chunk_id %s from %s", chunk.chunk_id, path)
                    continue
                unique[chunk.chunk_id] = chunk
        if not unique:
            raise IndexingError(f"No chunks found in {self.config.chunks_directory}")
        self.chunks = list(unique.values())
        LOGGER.info("Discovered %d unique chunks; skipped %d duplicates", len(self.chunks), duplicates)
        return self.chunks

    def build_embeddings(self, chunks: Iterable[Chunk] | None = None) -> list[Embedding]:
        """Embed chunks in batches and save their durable NPZ/JSON artifact."""
        source_chunks = list(chunks) if chunks is not None else (self.chunks or self.discover_chunks())
        try:
            self.embeddings = embed_chunks(
                source_chunks,
                EmbeddingConfig(model_name=self.config.embedding_model, batch_size=self.config.batch_size, device=self.config.device),
                self._model,
            )
            save_embeddings(self.embeddings, self.config.embeddings_directory / "embeddings.npz")
        except (EmbeddingError, OSError, ValueError) as exc:
            raise IndexingError(f"Unable to generate embeddings: {exc}") from exc
        LOGGER.info("Persisted %d embeddings to %s", len(self.embeddings), self.config.embeddings_directory)
        return self.embeddings

    def build_vector_store(self, embeddings: Iterable[Embedding] | None = None) -> VectorStore:
        """Build and persist the FAISS index, mapping, and ChromaDB collection."""
        source_embeddings = list(embeddings) if embeddings is not None else (self.embeddings or self.build_embeddings())
        try:
            self.vector_store = create_vector_store(
                source_embeddings,
                VectorStoreConfig(
                    faiss_index_path=self.config.vectorstore_directory / "faiss.index",
                    chroma_directory=self.config.vectorstore_directory / "chroma",
                ),
            )
        except (VectorStoreError, ValueError) as exc:
            raise IndexingError(f"Unable to build vector store: {exc}") from exc
        LOGGER.info("Built FAISS index with %d vectors and ChromaDB collection with %d documents", self.vector_store.index.ntotal, self.vector_store.collection.count())
        return self.vector_store

    def index(self) -> VectorStore:
        """Run all indexing stages and retain elapsed time for reporting."""
        started_at = time.perf_counter()
        chunks = self.discover_chunks()
        embeddings = self.build_embeddings(chunks)
        store = self.build_vector_store(embeddings)
        self.processing_seconds = time.perf_counter() - started_at
        LOGGER.info("Indexed %d chunks in %.2f seconds", len(chunks), self.processing_seconds)
        return store


def main() -> int:
    """Build the persisted retrieval index from recursively discovered chunks."""
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Build FinSight FAISS and ChromaDB retrieval indexes.")
    parser.add_argument("--chunks", type=Path, default=Path("chunks"), help="Root directory containing chunks.json files")
    parser.add_argument("--output", type=Path, default=Path("vectorstore"), help="Directory for FAISS and ChromaDB output")
    parser.add_argument("--embeddings", type=Path, default=Path("embeddings"), help="Directory for persisted embedding artifacts")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="Embedding execution device")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        indexer = Indexer(IndexerConfig(arguments.chunks, arguments.embeddings, arguments.output, arguments.embedding_model, arguments.batch_size, arguments.device))
        store = indexer.index()
    except (IndexingError, ValueError) as exc:
        LOGGER.error("Indexing failed: %s", exc)
        return 1
    print(f"Number of chunk files found: {indexer.chunk_file_count}")
    print(f"Number of chunks indexed: {len(indexer.chunks)}")
    print(f"Embedding dimension: {store.dimension}")
    print(f"FAISS vectors created: {store.index.ntotal}")
    print(f"Chroma documents stored: {store.collection.count()}")
    print(f"Processing time: {indexer.processing_seconds:.2f} seconds")
    return 0


__all__ = ["Indexer", "IndexerConfig", "IndexingError"]


if __name__ == "__main__":
    raise SystemExit(main())
