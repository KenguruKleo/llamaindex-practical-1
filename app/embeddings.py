from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from llama_index.core.embeddings import BaseEmbedding


@dataclass
class HashEmbeddingConfig:
    dimension: int = 512


class HashEmbedding(BaseEmbedding):
    """Deterministic hashing-based embedding model.

    This lightweight embedding approximates token co-occurrence information
    using a fixed-size bag-of-words vector that is safe to run offline.
    """

    def __init__(self, config: HashEmbeddingConfig | None = None) -> None:
        super().__init__()
        self._config = config or HashEmbeddingConfig()
        self._dim = self._config.dimension

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _embed(self, text: str) -> List[float]:
        tokens = self._tokenize(text)
        if not tokens:
            return [0.0] * self._dim

        vector = np.zeros(self._dim, dtype=np.float32)
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            bucket = int(digest, 16) % self._dim
            vector[bucket] += 1.0

        norm = np.linalg.norm(vector)
        if norm:  # avoid division by zero
            vector /= norm
        return vector.astype(float).tolist()

    # --- Text embeddings ---
    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed(text)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._embed(text)

    def get_text_embeddings(self, texts: Iterable[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    async def aget_text_embeddings(self, texts: Iterable[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    # --- Query embeddings ---
    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._embed(query)

    def get_query_embeddings(self, queries: Iterable[str]) -> List[List[float]]:
        return [self._embed(query) for query in queries]

    async def aget_query_embeddings(self, queries: Iterable[str]) -> List[List[float]]:
        return [self._embed(query) for query in queries]

    @property
    def dimension(self) -> int:
        return self._dim
