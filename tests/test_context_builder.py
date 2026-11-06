
import pytest

from app.rag.schemas import LegalChunk
from app.rag.context_builder import build_context


def make_chunk(
    chunk_id: str,
    text: str,
    article_number: str = "1",
    page_number: int = 1,
) -> LegalChunk:
    return LegalChunk(
        chunk_id=chunk_id,
        document_id="doc-001",
        source_name="قانون تجريبي.pdf",
        text=text,
        normalized_text=text,
        article_number=article_number,
        page_numbers=[page_number],
        chunk_index=0,
        metadata={"source_name": "قانون تجريبي.pdf"},
    )


def test_empty_chunks_return_empty_context():
    result = build_context([])

    assert result.context == ""
    assert result.included_chunks == []
    assert result.omitted_chunk_ids == []
    assert result.total_chars == 0


def test_context_preserves_retrieval_order():
    chunks = [
        make_chunk("chunk-1", "النص الأول"),
        make_chunk("chunk-2", "النص الثاني"),
        make_chunk("chunk-3", "النص الثالث"),
    ]

    result = build_context(chunks)

    assert [
        chunk.chunk_id for chunk in result.included_chunks
    ] == ["chunk-1", "chunk-2", "chunk-3"]


def test_duplicate_chunks_are_removed():
    first = make_chunk("chunk-1", "نص المادة الأولى")

    duplicate = make_chunk("chunk-1", "نص مكرر")

    result = build_context([first, duplicate])

    assert len(result.included_chunks) == 1
    assert result.included_chunks[0].text == "نص المادة الأولى"


def test_context_preserves_citation_metadata():
    chunk = make_chunk(
        "chunk-1",
        "يجب تقديم المستندات المطلوبة.",
        article_number="15",
        page_number=7,
    )

    result = build_context([chunk])

    assert "قانون تجريبي.pdf" in result.context
    assert "doc-001" in result.context
    assert "chunk-1" in result.context
    assert "15" in result.context
    assert "7" in result.context


def test_context_never_exceeds_character_budget():
    chunks = [
        make_chunk("chunk-1", "النص الأول"),
        make_chunk("chunk-2", "النص الثاني"),
    ]

    full_result = build_context(chunks)

    first_chunk_context = build_context(
        [chunks[0]]
    )

    result = build_context(
        chunks,
        max_context_chars=first_chunk_context.total_chars,
    )

    assert result.total_chars <= first_chunk_context.total_chars
    assert len(result.included_chunks) == 1
    assert result.omitted_chunk_ids == ["chunk-2"]


def test_invalid_context_budget_raises_error():
    with pytest.raises(ValueError):
        build_context([], max_context_chars=0)


def test_context_length_matches_actual_text_length():
    chunks = [
        make_chunk("chunk-1", "نص تجريبي"),
        make_chunk("chunk-2", "نص آخر"),
    ]

    result = build_context(chunks)

    assert result.total_chars == len(result.context)
