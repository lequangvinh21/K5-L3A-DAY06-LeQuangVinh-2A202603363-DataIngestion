import unittest

from src.chunking import by_paragraph, fixed_size
from src.evaluation import evaluate_query, summarize
from src.ingestion import chapter_for_page, normalize_text
from src.models import DocumentPage
from src.run import _choose_strategy
from src.store import _vector_literal


def make_page(text: str) -> DocumentPage:
    return DocumentPage(
        document_id="sample",
        source="sample.pdf",
        page=13,
        text=text,
        metadata={"page_id": "sample-p013", "chapter": 1},
    )


class NormalizeTextTests(unittest.TestCase):
    def test_normalizes_unicode_and_spacing_without_removing_paragraphs(self) -> None:
        text = "Cafe\u0301\tlaw.\r\n\r\n  Second   paragraph. "
        self.assertEqual(normalize_text(text), "Café law.\n\nSecond paragraph.")


class ChunkingTests(unittest.TestCase):
    def test_fixed_size_chunks_keep_page_metadata_and_offsets(self) -> None:
        page = make_page("abcdefghij klmnopqrst")
        chunks = fixed_size(page, size=10, overlap=3, strategy="A")

        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].text, "abcdefghij")
        self.assertEqual(chunks[0].metadata["chapter"], 1)
        self.assertEqual(chunks[0].metadata["start_char"], 0)
        self.assertLessEqual(chunks[1].metadata["start_char"], chunks[0].metadata["end_char"])
        self.assertEqual(chunks[-1].metadata["end_char"], len(page.text))

    def test_paragraph_chunks_preserve_all_words_and_respect_limit(self) -> None:
        text = (
            "First paragraph includes enough words to require a small split.\n\n"
            "Second paragraph remains a separate source paragraph."
        )
        page = make_page(text)
        chunks = by_paragraph(page, max_size=30, strategy="B")

        source_words = text.split()
        chunk_words = " ".join(chunk.text for chunk in chunks).split()
        self.assertEqual(chunk_words, source_words)
        self.assertTrue(all(len(chunk.text) <= 30 for chunk in chunks))
        self.assertTrue(all(chunk.metadata["strategy"] == "B" for chunk in chunks))

    def test_invalid_chunk_sizes_raise(self) -> None:
        page = make_page("some text")
        with self.assertRaises(ValueError):
            fixed_size(page, size=10, overlap=10)
        with self.assertRaises(ValueError):
            by_paragraph(page, max_size=0)


class MetadataTests(unittest.TestCase):
    def test_chapter_boundaries_match_pdf_contents(self) -> None:
        self.assertEqual(chapter_for_page(13), 1)
        self.assertEqual(chapter_for_page(87), 8)
        self.assertEqual(chapter_for_page(97), 9)
        self.assertEqual(chapter_for_page(145), 14)
        self.assertIsNone(chapter_for_page(154))


class EvaluationTests(unittest.TestCase):
    def test_evidence_hit_and_split_metrics(self) -> None:
        question = {
            "id": "q1",
            "split": "dev",
            "question": "What is the rule?",
            "gold_source_id": "sample",
            "gold_page": 13,
            "evidence_quote": "A verified evidence sentence.",
            "expected_answer": "The answer.",
        }
        hit = {
            "document_id": "sample",
            "page": 13,
            "text": "Context. A verified evidence sentence. More context.",
        }
        trace = evaluate_query(question, [hit], strategy="A")

        self.assertTrue(trace["hit_at_1"])
        self.assertTrue(trace["evidence_hit_at_3"])
        self.assertEqual(
            summarize([trace])["A"]["dev"],
            {
                "question_count": 1,
                "hit_at_1_count": 1,
                "hit_at_1": 1.0,
                "evidence_hit_at_3_count": 1,
                "evidence_hit_at_3": 1.0,
            },
        )

    def test_wrong_page_is_not_an_evidence_hit(self) -> None:
        question = {
            "id": "q1",
            "split": "test",
            "question": "What is the rule?",
            "gold_source_id": "sample",
            "gold_page": 13,
            "evidence_quote": "Evidence.",
            "expected_answer": "Answer.",
        }
        trace = evaluate_query(
            question,
            [{"document_id": "sample", "page": 14, "text": "Evidence."}],
            strategy="B",
        )
        self.assertFalse(trace["hit_at_1"])
        self.assertFalse(trace["evidence_hit_at_3"])

    def test_strategy_choice_uses_declared_dev_priority(self) -> None:
        summary = {
            "A": {"dev": {"evidence_hit_at_3": 0.8, "hit_at_1": 0.4}},
            "B": {"dev": {"evidence_hit_at_3": 0.8, "hit_at_1": 0.5}},
            "C": {"dev": {"evidence_hit_at_3": 0.6, "hit_at_1": 0.9}},
        }
        resources = {
            "A": {"total_chunk_characters": 100, "chunking_seconds": 1, "embedding_seconds": 1, "storage_seconds": 1},
            "B": {"total_chunk_characters": 500, "chunking_seconds": 1, "embedding_seconds": 1, "storage_seconds": 1},
            "C": {"total_chunk_characters": 50, "chunking_seconds": 1, "embedding_seconds": 1, "storage_seconds": 1},
        }
        self.assertEqual(_choose_strategy(summary, resources), "B")


class VectorValidationTests(unittest.TestCase):
    def test_vector_literal_checks_dimensions_and_non_finite_values(self) -> None:
        self.assertEqual(_vector_literal([0.25, 0.5], 2), "[0.25,0.5]")
        with self.assertRaises(ValueError):
            _vector_literal([0.25], 2)
        with self.assertRaises(ValueError):
            _vector_literal([float("nan")], 1)


if __name__ == "__main__":
    unittest.main()
