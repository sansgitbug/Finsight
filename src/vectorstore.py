"""FAISS similarity search with ChromaDB-backed persistent chunk metadata."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import chromadb
import faiss
import numpy as np

from src.chunker import Chunk
from src.embeddings import Embedding

LOGGER = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Raised when a FAISS or ChromaDB operation cannot be completed."""


@dataclass(frozen=True, slots=True)
class VectorStoreConfig:
    """Storage locations and ChromaDB collection name for retrieval data."""

    faiss_index_path: Path = Path("vectorstore/faiss.index")
    chroma_directory: Path = Path("vectorstore/chroma")
    collection_name: str = "finsight_chunks"

    def __post_init__(self) -> None:
        """Validate persistent-store locations and collection identifier."""
        if not self.collection_name.strip():
            raise ValueError("collection_name must be non-empty")


@dataclass(slots=True)
class VectorStore:
    """Loaded retrieval resources and the explicit FAISS-ID to chunk-ID map."""

    config: VectorStoreConfig
    index: faiss.Index
    collection: Any
    id_map: dict[int, str]
    dimension: int


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A cosine-similarity candidate ready for a future reranker stage."""

    chunk: Chunk
    score: float
    metadata: dict[str, Any]


def build_vector_store(embeddings: Sequence[Embedding], config: VectorStoreConfig | None = None) -> VectorStore:
    """Create a new dual vector store and populate it from embedding records."""
    if not embeddings:
        raise VectorStoreError("Cannot build a vector store from no embeddings")
    active_config = config or VectorStoreConfig()
    dimension = _embedding_matrix(embeddings).shape[1]
    store = _new_store(active_config, dimension)
    add_embeddings(store, embeddings)
    save(store)
    return store


def load_vector_store(config: VectorStoreConfig | None = None) -> VectorStore:
    """Load a saved FAISS index, explicit ID map, and ChromaDB collection."""
    return load(config)


def add_embeddings(store: VectorStore, embeddings: Sequence[Embedding]) -> None:
    """Upsert embeddings and their complete chunk metadata into both stores."""
    if not embeddings:
        return
    vectors = _embedding_matrix(embeddings)
    if vectors.shape[1] != store.dimension:
        raise VectorStoreError(f"Embedding dimension {vectors.shape[1]} does not match index dimension {store.dimension}")
    chunk_ids = [embedding.chunk_id for embedding in embeddings]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise VectorStoreError("Cannot add duplicate chunk IDs in one operation")
    existing_ids = {chunk_id: faiss_id for faiss_id, chunk_id in store.id_map.items()}
    faiss_ids: list[int] = []
    next_id = max(store.id_map, default=-1) + 1
    remove_ids: list[int] = []
    for chunk_id in chunk_ids:
        faiss_id = existing_ids.get(chunk_id)
        if faiss_id is None:
            faiss_id = next_id
            next_id += 1
        else:
            remove_ids.append(faiss_id)
        faiss_ids.append(faiss_id)
    if remove_ids:
        store.index.remove_ids(np.asarray(remove_ids, dtype=np.int64))
    normalized = _normalize_rows(vectors)
    try:
        store.index.add_with_ids(normalized, np.asarray(faiss_ids, dtype=np.int64))
        store.collection.upsert(
            ids=chunk_ids,
            embeddings=normalized.tolist(),
            documents=[embedding.chunk.text for embedding in embeddings],
            metadatas=[_chroma_metadata(embedding.chunk) for embedding in embeddings],
        )
    except Exception as exc:
        raise VectorStoreError(f"Unable to add embeddings to vector stores: {exc}") from exc
    for faiss_id, chunk_id in zip(faiss_ids, chunk_ids, strict=True):
        store.id_map[faiss_id] = chunk_id
    LOGGER.info("Added %d embeddings to collection %s", len(embeddings), store.config.collection_name)


def search(store: VectorStore, query_embedding: Sequence[float] | np.ndarray, top_k: int = 20) -> list[SearchResult]:
    """Return the highest cosine-similarity candidates without reranking."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    query = np.asarray(query_embedding, dtype=np.float32)
    if query.ndim == 1:
        query = query.reshape(1, -1)
    if query.shape != (1, store.dimension):
        raise VectorStoreError(f"Query embedding must have shape ({store.dimension},)")
    distances, labels = store.index.search(_normalize_rows(query), top_k)
    matched = [(int(faiss_id), float(score)) for faiss_id, score in zip(labels[0], distances[0], strict=True) if faiss_id >= 0]
    if not matched:
        return []
    chunk_ids = [store.id_map.get(faiss_id) for faiss_id, _ in matched]
    if any(chunk_id is None for chunk_id in chunk_ids):
        raise VectorStoreError("FAISS index contains an ID absent from the explicit chunk mapping")
    try:
        records = store.collection.get(ids=[str(chunk_id) for chunk_id in chunk_ids], include=["metadatas"])
    except Exception as exc:
        raise VectorStoreError(f"Unable to retrieve chunk metadata from ChromaDB: {exc}") from exc
    metadata_by_id = dict(zip(records["ids"], records["metadatas"], strict=True))
    results: list[SearchResult] = []
    for faiss_id, score in matched:
        chunk_id = store.id_map[faiss_id]
        chroma_metadata = metadata_by_id.get(chunk_id)
        if not isinstance(chroma_metadata, Mapping):
            raise VectorStoreError(f"ChromaDB metadata is missing for chunk {chunk_id}")
        chunk = _chunk_from_chroma_metadata(chroma_metadata)
        results.append(SearchResult(chunk=chunk, score=score, metadata=dict(chunk.metadata)))
    return results


def save(store: VectorStore) -> None:
    """Persist the FAISS index and its explicit non-order-dependent ID mapping."""
    mapping_path = _mapping_path(store.config.faiss_index_path)
    try:
        store.config.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
        store.config.chroma_directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(store.index, str(store.config.faiss_index_path))
        mapping_path.write_text(
            json.dumps({"dimension": store.dimension, "id_map": store.id_map}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, RuntimeError) as exc:
        raise VectorStoreError(f"Unable to save vector store: {exc}") from exc
    LOGGER.info("Saved FAISS index to %s", store.config.faiss_index_path)


def load(config: VectorStoreConfig | None = None) -> VectorStore:
    """Load persistent store files written by :func:`save`."""
    active_config = config or VectorStoreConfig()
    mapping_path = _mapping_path(active_config.faiss_index_path)
    try:
        index = faiss.read_index(str(active_config.faiss_index_path))
        payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    except (OSError, RuntimeError, json.JSONDecodeError) as exc:
        raise VectorStoreError(f"Unable to load saved FAISS index: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("dimension"), int) or not isinstance(payload.get("id_map"), dict):
        raise VectorStoreError(f"Invalid FAISS ID mapping file: {mapping_path}")
    id_map = {int(faiss_id): str(chunk_id) for faiss_id, chunk_id in payload["id_map"].items()}
    client = chromadb.PersistentClient(path=str(active_config.chroma_directory))
    try:
        collection = client.get_collection(active_config.collection_name)
    except Exception as exc:
        raise VectorStoreError(f"Unable to load ChromaDB collection {active_config.collection_name!r}: {exc}") from exc
    return VectorStore(active_config, index, collection, id_map, payload["dimension"])


def _new_store(config: VectorStoreConfig, dimension: int) -> VectorStore:
    """Create an empty inner-product FAISS index and persistent Chroma collection."""
    if dimension <= 0:
        raise VectorStoreError("Embedding dimension must be greater than zero")
    try:
        config.chroma_directory.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(config.chroma_directory))
        collection = client.get_or_create_collection(config.collection_name)
    except Exception as exc:
        raise VectorStoreError(f"Unable to initialize ChromaDB collection: {exc}") from exc
    return VectorStore(config, faiss.IndexIDMap2(faiss.IndexFlatIP(dimension)), collection, {}, dimension)


def _embedding_matrix(embeddings: Sequence[Embedding]) -> np.ndarray:
    """Validate and stack embedding records into a finite float32 matrix."""
    vectors = np.asarray([embedding.vector for embedding in embeddings], dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(embeddings) or vectors.shape[1] == 0:
        raise VectorStoreError("Embeddings must form a non-empty two-dimensional matrix")
    if not np.isfinite(vectors).all():
        raise VectorStoreError("Embeddings contain non-finite values")
    return vectors


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize vectors so inner product is cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise VectorStoreError("Zero-norm vectors cannot be used for cosine similarity")
    return vectors / norms


def _chroma_metadata(chunk: Chunk) -> dict[str, str | int | float | bool]:
    """Make all chunk fields available in Chroma's scalar metadata format."""
    return {
        "chunk_id": chunk.chunk_id,
        "ticker": chunk.ticker,
        "filing_date": chunk.filing_date,
        "filing_type": chunk.filing_type,
        "section_name": chunk.section_name,
        "token_count": chunk.token_count,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "character_start": chunk.character_start,
        "character_end": chunk.character_end,
        "chunk_json": json.dumps(asdict(chunk), ensure_ascii=False),
    }


def _chunk_from_chroma_metadata(metadata: Mapping[str, Any]) -> Chunk:
    """Restore the unchanged chunk dataclass from stored Chroma metadata."""
    try:
        payload = json.loads(str(metadata["chunk_json"]))
        return Chunk(**payload)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise VectorStoreError("Stored ChromaDB chunk metadata is malformed") from exc


def _mapping_path(index_path: Path) -> Path:
    """Return the sidecar path for the explicit FAISS-ID mapping."""
    return index_path.with_suffix(index_path.suffix + ".mapping.json")


__all__ = ["SearchResult", "VectorStore", "VectorStoreConfig", "VectorStoreError", "add_embeddings", "build_vector_store", "load", "load_vector_store", "save", "search"]
