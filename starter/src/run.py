"""Run the ingestion benchmark from the starter directory."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time
from importlib.metadata import version
from typing import Any

from dotenv import load_dotenv

from .chunking import by_paragraph, fixed_size
from .embeddings import DEFAULT_MODEL, MyEmbedder
from .evaluation import evaluate_query, summarize
from .ingestion import load_pdf
from .models import Chunk, DocumentPage
from .store import PgVectorStore


ROOT = Path(__file__).resolve().parents[2]
STARTER = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required input file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _content_char_count(text: str) -> int:
    return len(re.sub(r"\s+", " ", text).strip())


def _validate_corpus(
    pages: list[DocumentPage],
    manifest: dict[str, Any],
) -> list[DocumentPage]:
    if len(pages) != manifest["physical_page_count"]:
        raise ValueError(
            "PDF physical page count differs from manifest: "
            f"{len(pages)} != {manifest['physical_page_count']}"
        )

    pages_by_number = {page.page: page for page in pages}
    included: list[DocumentPage] = []
    errors: list[str] = []
    for record in manifest["pages"]:
        page_number = record["page_number"]
        page = pages_by_number.get(page_number)
        if page is None:
            errors.append(f"Manifest page {page_number} is not present in the PDF")
            continue
        if not page.text:
            errors.append(f"Manifest page {page_number} has no extractable text")
            continue
        actual_count = _content_char_count(page.text)
        if actual_count != record["normalized_char_count"]:
            errors.append(
                f"Page {page_number} character count mismatch: "
                f"{actual_count} != {record['normalized_char_count']}"
            )
        if "chapter" not in page.metadata:
            errors.append(f"Manifest page {page_number} has no chapter metadata")
        included.append(page)

    if errors:
        raise ValueError("Corpus/manifest validation failed:\n- " + "\n- ".join(errors))
    if len(included) < 30:
        raise ValueError(f"Corpus has only {len(included)} usable pages; at least 30 required")
    if len(included) != manifest["counted_page_count"]:
        raise ValueError(
            "Manifest counted_page_count does not match its page list: "
            f"{manifest['counted_page_count']} != {len(included)}"
        )
    return included


def _validate_benchmark(
    benchmark: dict[str, Any],
    pages: list[DocumentPage],
    document_id: str,
) -> None:
    questions = benchmark.get("questions")
    if not isinstance(questions, list):
        raise ValueError("Benchmark must contain a questions list")
    ids = [question.get("id") for question in questions]
    if len(questions) < 20 or len(set(ids)) != len(ids) or any(not item for item in ids):
        raise ValueError("Benchmark requires at least 20 questions with unique, non-empty IDs")

    counts = {
        split: sum(question.get("split") == split for question in questions)
        for split in ("dev", "test")
    }
    if counts["dev"] < 15 or counts["test"] < 5:
        raise ValueError("Benchmark requires at least 15 dev and 5 test questions")

    filtered = [question for question in questions if question.get("filter")]
    if len(filtered) < 3 or not any(question["split"] == "test" for question in filtered):
        raise ValueError("Benchmark requires 3 metadata-filter questions, including a test question")

    pages_by_number = {page.page: page for page in pages}
    for question in questions:
        page = pages_by_number.get(question.get("gold_page"))
        if question.get("gold_source_id") != document_id or page is None:
            raise ValueError(f"Question {question['id']} gold source/page is not in this corpus")
        quote = re.sub(r"\s+", " ", question["evidence_quote"]).strip().casefold()
        page_text = re.sub(r"\s+", " ", page.text).strip().casefold()
        if not quote or quote not in page_text:
            raise ValueError(
                f"Question {question['id']} evidence does not match gold page "
                f"{question['gold_page']}"
            )
        filters = question.get("filter", {})
        for field, value in filters.items():
            if page.metadata.get(field) != value:
                raise ValueError(
                    f"Question {question['id']} filter {field}={value!r} "
                    "does not match its gold page metadata"
                )


def _make_strategies(pages: list[DocumentPage]) -> tuple[dict[str, list[Chunk]], dict[str, float]]:
    chunks: dict[str, list[Chunk]] = {}
    durations: dict[str, float] = {}

    started = time.perf_counter()
    chunks["A"] = [
        chunk
        for page in pages
        for chunk in fixed_size(page, size=500, overlap=50, strategy="A")
    ]
    durations["A"] = time.perf_counter() - started

    started = time.perf_counter()
    chunks["B"] = [
        chunk
        for page in pages
        for chunk in by_paragraph(page, max_size=500, strategy="B")
    ]
    durations["B"] = time.perf_counter() - started

    started = time.perf_counter()
    chunks["C"] = [
        chunk
        for page in pages
        for chunk in by_paragraph(page, max_size=500, strategy="C")
    ]
    durations["C"] = time.perf_counter() - started

    for strategy, strategy_chunks in chunks.items():
        if not strategy_chunks:
            raise ValueError(f"Strategy {strategy} produced no chunks")
        if len({chunk.chunk_id for chunk in strategy_chunks}) != len(strategy_chunks):
            raise ValueError(f"Strategy {strategy} produced duplicate chunk IDs")
    return chunks, durations


def _chunk_statistics(chunks: list[Chunk], chunking_seconds: float) -> dict[str, Any]:
    lengths = [len(chunk.text) for chunk in chunks]
    return {
        "chunk_count": len(chunks),
        "total_chunk_characters": sum(lengths),
        "min_chunk_characters": min(lengths),
        "mean_chunk_characters": round(sum(lengths) / len(lengths), 2),
        "max_chunk_characters": max(lengths),
        "chunking_seconds": round(chunking_seconds, 4),
    }


def _manual_review_template(
    traces: list[dict[str, Any]],
    chunks_by_strategy: dict[str, list[Chunk]],
    pages: list[DocumentPage],
) -> dict[str, Any]:
    test_traces = [trace for trace in traces if trace["split"] == "test"]
    metadata_review = []
    grounding_review = []
    for trace in test_traces:
        grounding_review.append(
            {
                "strategy": trace["strategy"],
                "question_id": trace["question_id"],
                "score_0_to_2": None,
                "reason": "",
            }
        )
        for rank, hit in enumerate(trace["hits"], start=1):
            metadata_review.append(
                {
                    "strategy": trace["strategy"],
                    "question_id": trace["question_id"],
                    "rank": rank,
                    "chunk_id": hit["chunk_id"],
                    "source": hit["source"],
                    "page": hit["page"],
                    "metadata": hit["metadata"],
                    "required_filter": trace["filter"],
                    "metadata_correct_and_complete": None,
                    "filter_compliant": None,
                    "notes": "",
                }
            )

    coherence_review = []
    for page in pages[:10]:
        text_positions = [index for index, character in enumerate(page.text) if not character.isspace()]
        target_offset = text_positions[len(text_positions) // 2]
        for strategy, chunks in chunks_by_strategy.items():
            candidates = [
                chunk
                for chunk in chunks
                if chunk.page == page.page
                and chunk.metadata["start_char"] <= target_offset < chunk.metadata["end_char"]
            ]
            if not candidates:
                raise ValueError(
                    f"Strategy {strategy} does not cover the sample position "
                    f"on page {page.page}"
                )
            chunk = candidates[0]
            coherence_review.append(
                {
                    "strategy": strategy,
                    "page_id": page.metadata["page_id"],
                    "page": page.page,
                    "target_offset": target_offset,
                    "chunk_id": chunk.chunk_id,
                    "chunk_text": chunk.text,
                    "score_0_to_2": None,
                    "reason": "",
                }
            )

    return {
        "instructions": (
            "Fill in manual labels independently of retrieval scores. Grounding and metadata "
            "records cover test queries; coherence uses the same midpoint on the first 10 "
            "manifest pages for each strategy."
        ),
        "scales": {
            "grounding_score_0_to_2": {
                "0": "No evidence.",
                "1": "Evidence is present but misses a needed condition or context.",
                "2": "Evidence is sufficient to answer and trace to the source.",
            },
            "coherence_score_0_to_2": {
                "0": "Cut mid-idea or missing a subject so the chunk is not understandable.",
                "1": "Partly understandable but missing context.",
                "2": "Self-contained and preserves a complete idea.",
            },
            "metadata_correct_and_complete": "True only when source, page, and required fields are correct.",
        },
        "grounding_review": grounding_review,
        "metadata_review": metadata_review,
        "coherence_review": coherence_review,
    }


def _comparison(summary: dict[str, Any]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for split in ("dev", "test"):
        for left, right in (("A", "B"), ("B", "C")):
            left_metrics = summary.get(left, {}).get(split)
            right_metrics = summary.get(right, {}).get(split)
            if left_metrics and right_metrics:
                comparisons[f"{left}_vs_{right}_{split}"] = {
                    "hit_at_1_delta": round(
                        right_metrics["hit_at_1"] - left_metrics["hit_at_1"], 4
                    ),
                    "evidence_hit_at_3_delta": round(
                        right_metrics["evidence_hit_at_3"]
                        - left_metrics["evidence_hit_at_3"],
                        4,
                    ),
                }
    return comparisons


def _choose_strategy(summary: dict[str, Any], resources: dict[str, Any]) -> str:
    candidates = ("A", "B", "C")
    if any(strategy not in summary or "dev" not in summary[strategy] for strategy in candidates):
        raise ValueError("A dev result for each strategy is required to select a configuration")
    return min(
        candidates,
        key=lambda strategy: (
            -summary[strategy]["dev"]["evidence_hit_at_3"],
            -summary[strategy]["dev"]["hit_at_1"],
            resources[strategy]["total_chunk_characters"],
            resources[strategy]["chunking_seconds"]
            + resources[strategy]["embedding_seconds"]
            + resources[strategy]["storage_seconds"],
        ),
    )


def run(split: str = "dev", selected_strategy: str | None = None) -> Path:
    if split not in {"dev", "test", "all"}:
        raise ValueError("split must be dev, test, or all")
    if split == "dev" and selected_strategy is not None:
        raise ValueError("The dev run selects a strategy automatically")
    if split in {"test", "all"} and selected_strategy not in {"A", "B", "C"}:
        raise ValueError(
            "After choosing on dev, pass --selected-strategy A, B, or C before running test"
        )
    if split == "all":
        print("Warning: --split all includes test data; use test only after freezing the strategy.")

    load_dotenv(STARTER / ".env")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is missing; configure starter/.env first")

    manifest = _read_json(DATA / "manifest.json")
    benchmark = _read_json(DATA / "benchmark.json")
    document_id = manifest["source_id"]
    pdf_path = ROOT / manifest["source_path_or_url"]
    source = manifest.get("source_url", manifest["source_path_or_url"])

    ingestion_started = time.perf_counter()
    all_pages = load_pdf(pdf_path, document_id=document_id, source=source)
    pages = _validate_corpus(all_pages, manifest)
    ingestion_seconds = time.perf_counter() - ingestion_started
    _validate_benchmark(benchmark, pages, document_id)
    questions = benchmark["questions"]
    if split != "all":
        questions = [question for question in questions if question["split"] == split]
    if not questions:
        raise ValueError(f"Benchmark contains no questions for split {split!r}")

    model_name = os.getenv("EMBEDDING_MODEL") or DEFAULT_MODEL
    embedder = MyEmbedder(model_name)
    store = PgVectorStore(database_url, embedder.dimensions)
    store.create_schema()
    database_info = store.server_info()
    chunks_by_strategy, chunking_durations = _make_strategies(pages)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output_dir = STARTER / "results" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    extraction_warnings = [
        {
            "page": page.page,
            "page_id": page.metadata["page_id"],
            "warning": page.metadata["extraction_warning"],
        }
        for page in all_pages
        if "extraction_warning" in page.metadata
    ]
    _write_json(output_dir / "extraction_warnings.json", extraction_warnings)

    query_vectors = {
        question["id"]: embedder.embed_query(question["question"])
        for question in questions
    }
    traces: list[dict[str, Any]] = []
    resources: dict[str, Any] = {}
    configurations = {
        "A": {"chunking": "fixed characters", "size": 500, "overlap": 50, "filter": False},
        "B": {"chunking": "paragraph-aware", "max_size": 500, "filter": False},
        "C": {
            "chunking": "paragraph-aware",
            "max_size": 500,
            "filter": "metadata chapter when present in question",
        },
    }

    for strategy, chunks in chunks_by_strategy.items():
        embedding_started = time.perf_counter()
        vectors = embedder.embed_documents([chunk.text for chunk in chunks])
        embedding_seconds = time.perf_counter() - embedding_started

        storage_started = time.perf_counter()
        store.upsert(strategy, chunks, vectors)
        storage_seconds = time.perf_counter() - storage_started
        resources[strategy] = {
            **_chunk_statistics(chunks, chunking_durations[strategy]),
            "embedding_seconds": round(embedding_seconds, 4),
            "storage_seconds": round(storage_seconds, 4),
        }

        for question in questions:
            filters = question.get("filter") if strategy == "C" else None
            hits = store.search(
                strategy,
                query_vectors[question["id"]],
                top_k=3,
                filters=filters,
            )
            traces.append(evaluate_query(question, hits, strategy=strategy))

    metrics = summarize(traces)
    dev_selected_strategy = (
        _choose_strategy(metrics, resources) if split in {"dev", "all"} else None
    )
    metadata = {
        "run_id": run_id,
        "split": split,
        "selected_strategy": selected_strategy or dev_selected_strategy,
        "selection_source": "dev metrics" if dev_selected_strategy else "chosen on dev before test",
        "source_id": document_id,
        "physical_page_count": manifest["physical_page_count"],
        "counted_page_count": len(pages),
        "source_text_characters": sum(len(page.text) for page in pages),
        "pdf_ingestion_seconds": round(ingestion_seconds, 4),
        "ocr_used": False,
        "embedding_provider": "Sentence Transformers (local CPU)",
        "embedding_model": embedder.model_name,
        "embedding_dimensions": embedder.dimensions,
        "query_and_document_encoding": "Same model encode method; L2-normalized embeddings.",
        "model_max_sequence_length": embedder.max_sequence_length,
        "vector_store": "PostgreSQL + pgvector",
        "database": database_info,
        "vector_table": store.table_name,
        "distance_metric": "cosine",
        "search_mode": "exact",
        "cost": 0,
        "software_versions": {
            "python": sys.version.split()[0],
            "sentence-transformers": version("sentence-transformers"),
            "pypdf": version("pypdf"),
            "psycopg": version("psycopg"),
        },
        "configurations": configurations,
        "selection_rule": (
            "Choose on dev by highest Evidence Hit@3, then Hit@1, then lower total chunk "
            "characters and processing time. Do not select or tune using test results."
        ),
        "resources": resources,
        "retrieval_metrics": metrics,
        "strategy_comparisons": _comparison(metrics),
        "extraction_warning_count": len(extraction_warnings),
    }
    _write_json(output_dir / "run_metadata.json", metadata)
    _write_json(output_dir / f"traces_{split}.json", traces)
    if split in {"test", "all"}:
        _write_json(
            output_dir / "manual_review_template.json",
            _manual_review_template(traces, chunks_by_strategy, pages),
        )

    print(f"Run saved to: {output_dir}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Day 06 ingestion benchmark.")
    parser.add_argument(
        "--split",
        choices=("dev", "test", "all"),
        default="dev",
        help="Run dev by default; run test only after choosing the strategy on dev.",
    )
    parser.add_argument(
        "--selected-strategy",
        choices=("A", "B", "C"),
        help="Required for test/all runs; choose it using the dev results.",
    )
    arguments = parser.parse_args()
    run(arguments.split, arguments.selected_strategy)


if __name__ == "__main__":
    main()
