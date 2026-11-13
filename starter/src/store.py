"""Persistent PostgreSQL + pgvector storage and exact cosine retrieval."""
import math
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from .models import Chunk


def _vector_literal(vector: list[float], dimensions: int) -> str:
    if len(vector) != dimensions:
        raise ValueError(f"Expected {dimensions} vector values, got {len(vector)}")
    if any(not math.isfinite(float(value)) for value in vector):
        raise ValueError("Vector values must be finite numbers")
    return "[" + ",".join(repr(float(value)) for value in vector) + "]"


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int) -> None:
        if not database_url.strip():
            raise ValueError("database_url must not be empty")
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.database_url = database_url
        self.dimensions = dimensions
        self.table_name = f"day6_chunks_{dimensions}"

    def server_info(self) -> dict[str, str]:
        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT current_setting('server_version'),
                           extversion
                    FROM pg_extension
                    WHERE extname = 'vector'
                    """
                )
                row = cursor.fetchone()
        if row is None:
            raise RuntimeError("pgvector extension is not installed in the configured database")
        return {"postgresql_version": row[0], "pgvector_version": row[1]}

    def create_schema(self) -> None:
        self._execute_schema(
            f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                strategy TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                source TEXT NOT NULL,
                page INTEGER NOT NULL,
                text TEXT NOT NULL,
                metadata JSONB NOT NULL,
                embedding vector({self.dimensions}) NOT NULL,
                PRIMARY KEY (strategy, chunk_id)
            )
            """
        )
        self._execute_schema(
            f"""
            CREATE INDEX IF NOT EXISTS {self.table_name}_metadata_gin
            ON {self.table_name} USING GIN (metadata jsonb_path_ops)
            """
        )

    def _execute_schema(self, statement: str) -> None:
        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cursor.execute(statement)

    def upsert(
        self,
        strategy: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> None:
        if not strategy:
            raise ValueError("strategy must not be empty")
        if len(chunks) != len(vectors):
            raise ValueError("Each chunk must have exactly one embedding")
        if not chunks:
            raise ValueError("At least one chunk is required for upsert")

        rows = []
        document_ids = sorted({chunk.document_id for chunk in chunks})
        for chunk, vector in zip(chunks, vectors):
            if not chunk.text.strip():
                raise ValueError(f"Chunk {chunk.chunk_id!r} has empty text")
            rows.append(
                (
                    strategy,
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.source,
                    chunk.page,
                    chunk.text,
                    Jsonb(chunk.metadata),
                    _vector_literal(vector, self.dimensions),
                )
            )

        statement = f"""
            INSERT INTO {self.table_name}
                (strategy, chunk_id, document_id, source, page, text, metadata, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
            ON CONFLICT (strategy, chunk_id) DO UPDATE SET
                document_id = EXCLUDED.document_id,
                source = EXCLUDED.source,
                page = EXCLUDED.page,
                text = EXCLUDED.text,
                metadata = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding
        """
        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.executemany(
                    f"DELETE FROM {self.table_name} WHERE strategy = %s AND document_id = %s",
                    [(strategy, document_id) for document_id in document_ids],
                )
                cursor.executemany(statement, rows)

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not strategy:
            raise ValueError("strategy must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query = _vector_literal(query_vector, self.dimensions)

        where = ["strategy = %s"]
        params: list[Any] = [query, strategy]
        if filters:
            where.append("metadata @> %s")
            params.append(Jsonb(filters))
        params.extend([query, top_k])
        statement = f"""
            SELECT chunk_id, document_id, source, page, text, metadata,
                   1 - (embedding <=> %s::vector) AS score
            FROM {self.table_name}
            WHERE {" AND ".join(where)}
            ORDER BY embedding <=> %s::vector, chunk_id
            LIMIT %s
        """
        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(statement, params)
                rows = cursor.fetchall()

        return [
            {
                "chunk_id": row[0],
                "document_id": row[1],
                "source": row[2],
                "page": row[3],
                "text": row[4],
                "metadata": row[5],
                "score": float(row[6]),
            }
            for row in rows
        ]
