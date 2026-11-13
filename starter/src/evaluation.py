"""Gold-evidence retrieval metrics; human labels remain manual."""
import re
import unicodedata
from typing import Any


def _normalized_for_match(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", normalized).strip().casefold()


def _is_evidence_hit(question: dict[str, Any], hit: dict[str, Any]) -> bool:
    evidence = question.get("evidence_quote")
    return (
        hit.get("document_id") == question.get("gold_source_id")
        and hit.get("page") == question.get("gold_page")
        and isinstance(evidence, str)
        and bool(evidence.strip())
        and _normalized_for_match(evidence) in _normalized_for_match(hit.get("text", ""))
    )


def evaluate_query(
    question: dict[str, Any],
    hits: list[dict[str, Any]],
    strategy: str = "",
) -> dict[str, Any]:
    """Evaluate exact gold-page evidence while retaining full retrieval trace."""
    if not question.get("id") or not question.get("split"):
        raise ValueError("Question must include id and split")
    if not question.get("gold_source_id") or not question.get("gold_page"):
        raise ValueError(f"Question {question['id']} is missing gold source/page")
    if not question.get("evidence_quote"):
        raise ValueError(f"Question {question['id']} is missing evidence_quote")

    evidence_hits = [_is_evidence_hit(question, hit) for hit in hits]
    return {
        "question_id": question["id"],
        "split": question["split"],
        "strategy": strategy,
        "question": question["question"],
        "filter": question.get("filter"),
        "gold": {
            "source_id": question["gold_source_id"],
            "page": question["gold_page"],
            "evidence_quote": question["evidence_quote"],
            "expected_answer": question["expected_answer"],
        },
        "hits": hits,
        "hit_at_1": bool(evidence_hits and evidence_hits[0]),
        "evidence_hit_at_3": any(evidence_hits[:3]),
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Hit@1 and Evidence Hit@3 by strategy and split."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for trace in traces:
        key = (trace["strategy"], trace["split"])
        groups.setdefault(key, []).append(trace)

    summary: dict[str, Any] = {}
    for (strategy, split), rows in sorted(groups.items()):
        total = len(rows)
        hit_at_1 = sum(bool(row["hit_at_1"]) for row in rows)
        hit_at_3 = sum(bool(row["evidence_hit_at_3"]) for row in rows)
        summary.setdefault(strategy, {})[split] = {
            "question_count": total,
            "hit_at_1_count": hit_at_1,
            "hit_at_1": hit_at_1 / total,
            "evidence_hit_at_3_count": hit_at_3,
            "evidence_hit_at_3": hit_at_3 / total,
        }
    return summary
