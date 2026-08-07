"""Embedding generation and persistence for FinSight retrieval chunks."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Protocol, Sequence

import numpy as np
from src.chunker import Chunk

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

LOGGER = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Raised when chunk embeddings cannot be loaded, created, or persisted."""


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    """Configuration for the sentence-transformer embedding model.

    ``device`` accepts ``"cpu"`` or ``"cuda"``. Passing ``"cuda"`` uses a
    CUDA-capable installation when available; the same API otherwise works on
    CPU without code changes.
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    normalize_embeddings: bool = True
    device: str = "cpu"

    def __post_init__(self) -> None:
        """Validate model options before loading a potentially large model."""
        if not self.model_name.strip():
            raise ValueError("model_name must be non-empty")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be either 'cpu' or 'cuda'")


@dataclass(frozen=True, slots=True)
class Embedding:
    """One embedding vector paired with its original :class:`Chunk`."""

    chunk: Chunk
    vector: np.ndarray

    @property
    def chunk_id(self) -> str:
        """Return the stable chunk identifier used by the vector store."""
        return self.chunk.chunk_id


class EmbeddingModel(Protocol):
    """SentenceTransformer API used by :func:`embed_chunks`."""

    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        """Encode a batch of text into a two-dimensional array."""


def load_chunks(path: Path) -> list[Chunk]:
    """Load and validate chunks emitted by :func:`src.chunker.save_chunks`."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EmbeddingError(f"Unable to load chunks from {path}: {exc}") from exc
    if not isinstance(payload, list):
        raise EmbeddingError(f"Chunk file must contain a JSON array: {path}")
    chunks: list[Chunk] = []
    required = set(Chunk.__dataclass_fields__)
    for position, record in enumerate(payload):
        if not isinstance(record, dict) or not required.issubset(record):
            raise EmbeddingError(f"Invalid chunk record at index {position} in {path}")
        try:
            chunks.append(Chunk(**{field: record[field] for field in required}))
        except (TypeError, ValueError) as exc:
            raise EmbeddingError(f"Invalid chunk record at index {position} in {path}: {exc}") from exc
    return chunks


def embed_chunks(
    chunks: Iterable[Chunk], config: EmbeddingConfig | None = None, model: EmbeddingModel | None = None
) -> list[Embedding]:
    """Create embeddings in batches while retaining each original chunk.

    A supplied ``model`` is useful for dependency-injected deployments and
    offline tests. Otherwise the configured SentenceTransformer is loaded.
    """
    chunk_list = list(chunks)
    if not chunk_list:
        return []
    active_config = config or EmbeddingConfig()
    active_model = model or _load_model(active_config)
    texts = [chunk.text for chunk in chunk_list]
    try:
        vectors = np.asarray(
            active_model.encode(
                texts,
                batch_size=active_config.batch_size,
                show_progress_bar=False,
                normalize_embeddings=active_config.normalize_embeddings,
            ),
            dtype=np.float32,
        )
    except Exception as exc:  # Third-party model backends expose varied exception classes.
        raise EmbeddingError(f"Embedding model failed to encode {len(chunk_list)} chunks: {exc}") from exc
    if vectors.ndim != 2 or vectors.shape[0] != len(chunk_list) or vectors.shape[1] == 0:
        raise EmbeddingError("Embedding model returned an invalid vector matrix")
    if not np.isfinite(vectors).all():
        raise EmbeddingError("Embedding model returned non-finite values")
    LOGGER.info("Embedded %d chunks with %s on %s", len(chunk_list), active_config.model_name, active_config.device)
    return [Embedding(chunk, vector) for chunk, vector in zip(chunk_list, vectors, strict=True)]


def save_embeddings(embeddings: Sequence[Embedding], output_path: Path) -> Path:
    """Save vectors as NPZ and chunk associations as an adjacent JSON sidecar.

    The supplied path may end in ``.npz`` or omit an extension. The JSON file
    has the same stem and contains chunk metadata in vector-row order.
    """
    npz_path = output_path.with_suffix(".npz")
    metadata_path = npz_path.with_suffix(".json")
    if not embeddings:
        raise EmbeddingError("Cannot persist an empty embedding collection")
    vectors = np.asarray([embedding.vector for embedding in embeddings], dtype=np.float32)
    if vectors.ndim != 2:
        raise EmbeddingError("Embeddings must form a two-dimensional vector matrix")
    try:
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, vectors=vectors)
        metadata_path.write_text(
            json.dumps([asdict(embedding.chunk) for embedding in embeddings], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise EmbeddingError(f"Unable to save embeddings to {npz_path}: {exc}") from exc
    LOGGER.info("Saved %d embeddings to %s", len(embeddings), npz_path)
    return npz_path


def load_embeddings(input_path: Path) -> list[Embedding]:
    """Load embeddings previously written by :func:`save_embeddings`."""
    npz_path = input_path.with_suffix(".npz")
    metadata_path = npz_path.with_suffix(".json")
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            vectors = np.asarray(archive["vectors"], dtype=np.float32)
        chunks = load_chunks(metadata_path)
    except (OSError, KeyError, ValueError) as exc:
        raise EmbeddingError(f"Unable to load embeddings from {npz_path}: {exc}") from exc
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
        raise EmbeddingError("Embedding vector count does not match persisted chunk metadata")
    return [Embedding(chunk, vector) for chunk, vector in zip(chunks, vectors, strict=True)]


def _load_model(config: EmbeddingConfig) -> Any:
    """Load the requested SentenceTransformer with a domain-specific error."""
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(config.model_name, device=config.device)
    except Exception as exc:  # Download and backend failures depend on installed extras.
        raise EmbeddingError(f"Unable to load embedding model {config.model_name!r}: {exc}") from exc


def load_embedding_model(config: EmbeddingConfig | None = None) -> Any:
    """Load the configured SentenceTransformer for document or query encoding."""
    return _load_model(config or EmbeddingConfig())


__all__ = ["Embedding", "EmbeddingConfig", "EmbeddingError", "embed_chunks", "load_chunks", "load_embedding_model", "load_embeddings", "save_embeddings"]
