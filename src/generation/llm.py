"""
Local LLM interface for FinSight using Ollama.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import ollama

LOGGER = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when local LLM generation fails."""


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """Configuration for the local Ollama model."""

    model_name: str = "qwen2.5:7b-instruct"
    temperature: float = 0.1
    max_tokens: int = 1024

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name must be non-empty")

        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")

        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")


class LLM:
    """
    Thin wrapper around Ollama.

    Keeps the rest of FinSight independent of the underlying LLM provider.
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()

    def generate(self, prompt: str) -> str:
        """
        Generate a response from the configured local LLM.
        """

        if not prompt or not prompt.strip():
            raise ValueError("prompt must be non-empty")

        try:
            response = ollama.chat(
                model=self.config.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                options={
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                },
            )
        except Exception as exc:
            raise LLMError(
                f"Unable to generate response from Ollama: {exc}"
            ) from exc

        try:
            answer = response["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMError(
                "Ollama returned an invalid response"
            ) from exc

        if not isinstance(answer, str) or not answer.strip():
            raise LLMError("Ollama returned an empty response")

        LOGGER.info(
            "Generated response using %s",
            self.config.model_name,
        )

        return answer.strip()


__all__ = [
    "LLM",
    "LLMConfig",
    "LLMError",
]