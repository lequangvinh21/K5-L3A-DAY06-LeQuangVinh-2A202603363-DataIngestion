"""Local Sentence Transformers embedding adapter."""
from typing import Protocol


DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one normalized vector per document, in input order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a normalized query vector compatible with document vectors."""
        ...


class MyEmbedder:
    """Create real CPU embeddings; model and dependency failures are explicit."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be empty")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is missing; install starter/requirements.txt"
            ) from exc

        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device="cpu")
        dimensions = self._model.get_embedding_dimension()
        if dimensions is None or dimensions <= 0:
            raise RuntimeError(f"Model {model_name!r} did not report a valid dimension")
        self.dimensions = dimensions
        self.max_sequence_length = self._model.max_seq_length

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        encoded = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in row] for row in encoded.tolist()]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if any(not text.strip() for text in texts):
            raise ValueError("Document texts must not be empty")
        vectors = self._encode(texts)
        if len(vectors) != len(texts) or any(
            len(vector) != self.dimensions for vector in vectors
        ):
            raise RuntimeError("Embedding model returned invalid document vectors")
        return vectors

    def embed_query(self, question: str) -> list[float]:
        if not question.strip():
            raise ValueError("Question must not be empty")
        vectors = self._encode([question])
        if len(vectors) != 1 or len(vectors[0]) != self.dimensions:
            raise RuntimeError("Embedding model returned an invalid query vector")
        return vectors[0]
