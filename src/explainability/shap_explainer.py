"""
SHAP-based token attribution for FinSight CrossEncoder reranking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import shap

from src.retrieval.reranker import Reranker, RerankerError

LOGGER = logging.getLogger(__name__)


class SHAPExplainerError(Exception):
    """Raised when SHAP explanation fails."""


@dataclass(frozen=True, slots=True)
class TokenAttribution:
    """SHAP contribution for one text segment."""

    token: str
    value: float


@dataclass(frozen=True, slots=True)
class SHAPExplanation:
    """Token-level explanation of a CrossEncoder relevance score."""

    query: str
    chunk_id: str
    base_value: float
    predicted_score: float
    attributions: tuple[TokenAttribution, ...]


class RerankerSHAPExplainer:
    """
    Explain CrossEncoder relevance scores using SHAP text masking.

    The query is kept fixed while SHAP masks segments of the retrieved
    filing text. Positive values indicate that a segment increases the
    CrossEncoder relevance score; negative values indicate that it
    decreases the score.
    """

    def __init__(
        self,
        reranker: Reranker,
        max_tokens: int = 40,
        max_evals: int = 300,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        if max_evals <= 0:
            raise ValueError("max_evals must be positive")

        self.reranker = reranker
        self.max_tokens = max_tokens
        self.max_evals = max_evals

    def explain(
        self,
        query: str,
        chunk_id: str,
        chunk_text: str,
    ) -> SHAPExplanation:
        """Explain one query/chunk CrossEncoder score."""

        if not query or not query.strip():
            raise ValueError("query must be non-empty")

        if not chunk_id or not chunk_id.strip():
            raise ValueError("chunk_id must be non-empty")

        if not chunk_text or not chunk_text.strip():
            raise ValueError("chunk_text must be non-empty")

        text = self._truncate_text(
            " ".join(chunk_text.split())
        )

        try:
            self.reranker.load()
            model = self.reranker.model

            tokenizer = self._get_tokenizer(model)

            def predict(texts: list[str]) -> np.ndarray:
                pairs = [
                    (query.strip(), text_value)
                    for text_value in texts
                ]

                try:
                    scores = model.predict(
                        pairs,
                        show_progress_bar=False,
                    )
                except Exception as exc:
                    raise SHAPExplainerError(
                        f"CrossEncoder prediction failed: {exc}"
                    ) from exc

                return np.asarray(
                    scores,
                    dtype=np.float64,
                ).reshape(-1)

            masker = shap.maskers.Text(
                tokenizer=tokenizer,
                output_type="string",
            )

            explainer = shap.Explainer(
                predict,
                masker,
                algorithm="partition",
            )

            shap_values = explainer(
                [text],
                max_evals=self.max_evals,
            )

            values = self._extract_values(
                shap_values
            )

            tokens = self._extract_segments(
                masker,
                text,
            )

            if len(tokens) != len(values):
                raise SHAPExplainerError(
                    "SHAP returned a token/value length mismatch: "
                    f"{len(tokens)} tokens vs {len(values)} values"
                )

            predicted_score = float(
                predict([text])[0]
            )

            base_value = self._extract_base_value(
                shap_values
            )

            attributions = tuple(
                TokenAttribution(
                    token=token,
                    value=float(value),
                )
                for token, value in zip(
                    tokens,
                    values,
                    strict=True,
                )
                if token.strip()
            )

            return SHAPExplanation(
                query=query.strip(),
                chunk_id=chunk_id,
                base_value=base_value,
                predicted_score=predicted_score,
                attributions=attributions,
            )

        except SHAPExplainerError:
            raise

        except RerankerError as exc:
            raise SHAPExplainerError(
                f"Unable to load CrossEncoder: {exc}"
            ) from exc

        except Exception as exc:
            raise SHAPExplainerError(
                f"Unable to generate SHAP explanation: {exc}"
            ) from exc

    def _truncate_text(self, text: str) -> str:
        """Limit the input length for practical SHAP runtime."""

        words = text.split()

        if len(words) <= self.max_tokens:
            return text

        return " ".join(
            words[: self.max_tokens]
        )

    @staticmethod
    def _get_tokenizer(model: Any) -> Any:
        """Extract the tokenizer from SentenceTransformers CrossEncoder."""

        try:
            tokenizer = model.tokenizer
        except AttributeError as exc:
            raise SHAPExplainerError(
                "CrossEncoder does not expose a tokenizer"
            ) from exc

        if tokenizer is None:
            raise SHAPExplainerError(
                "CrossEncoder tokenizer is unavailable"
            )

        return tokenizer

    @staticmethod
    def _extract_segments(
        masker: Any,
        text: str,
    ) -> list[str]:
        """
        Get feature labels corresponding exactly to SHAP values.

        SHAP's text masker returns tokenizer IDs in addition to the
        human-readable text segments. The SHAP attribution vector can
        include special tokenizer tokens, so we use the tokenizer IDs
        to produce a label for every attribution value.
        """

        try:
            _segments, token_ids = masker.token_segments(text)
        except Exception as exc:
            raise SHAPExplainerError(
                f"Unable to extract SHAP text segments: {exc}"
            ) from exc

        try:
            tokenizer = masker.tokenizer

            tokens = tokenizer.convert_ids_to_tokens(
                token_ids
            )

            return [
                str(token)
                for token in tokens
            ]

        except Exception as exc:
            raise SHAPExplainerError(
                f"Unable to convert SHAP token IDs: {exc}"
            ) from exc
    @staticmethod
    def _extract_values(
        shap_values: Any,
    ) -> np.ndarray:
        """Extract the one-dimensional SHAP attribution vector."""

        values = np.asarray(
            shap_values.values,
            dtype=np.float64,
        )

        # Expected shape for one text sample:
        #
        #   (1, number_of_features)
        #
        # Some SHAP configurations can leave an additional output
        # dimension, so remove only singleton dimensions.
        values = np.squeeze(values)

        if values.ndim != 1:
            raise SHAPExplainerError(
                "Unexpected SHAP value shape: "
                f"{values.shape}"
            )

        return values

    @staticmethod
    def _extract_base_value(
        shap_values: Any,
    ) -> float:
        """Extract the SHAP expected/base value."""

        base_values = np.asarray(
            shap_values.base_values,
            dtype=np.float64,
        )

        if base_values.size == 0:
            raise SHAPExplainerError(
                "SHAP returned no base value"
            )

        return float(
            base_values.reshape(-1)[0]
        )


__all__ = [
    "RerankerSHAPExplainer",
    "SHAPExplanation",
    "SHAPExplainerError",
    "TokenAttribution",
]