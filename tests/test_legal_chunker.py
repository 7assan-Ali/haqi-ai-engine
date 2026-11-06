
import pytest

from app.rag.schemas import DocumentPage, ExtractedDocument
from app.rag.legal_chunker import chunk_document


@pytest.fixture
def sample_document():
    return ExtractedDocument(
        source_name="law_143.pdf",
        total_pages=2,
        pages=[
            DocumentPage(
                page_number=1,
                text=(
                    "قانون رقم 143 لسنة 1994\n\n"
                    "المادة (1): تنظم هذه المادة أحكام القانون.\n\n"
                    "المادة 2: تسري أحكام هذا القانون."
                ),
            ),
            DocumentPage(
                page_number=2,
                text="المادة 3: تلتزم الجهات المختصة بتنفيذ القانون.",
            ),
        ],
    )


def test_chunk_document_returns_chunks(sample_document):
    chunks = chunk_document(sample_document)

    assert len(chunks) == 3


def test_article_numbers_are_preserved(sample_document):
    chunks = chunk_document(sample_document)

    assert [chunk.article_number for chunk in chunks] == [
        "1", "2", "3"
    ]


def test_source_and_page_metadata_are_preserved(sample_document):
    chunks = chunk_document(sample_document)

    assert chunks[0].source_name == "law_143.pdf"
    assert chunks[0].page_numbers == [1]
    assert chunks[2].page_numbers == [2]


def test_original_and_normalized_text_are_preserved(sample_document):
    chunks = chunk_document(sample_document)

    assert "أحكام" in chunks[0].text
    assert "احكام" in chunks[0].normalized_text


def test_long_text_is_split():
    document = ExtractedDocument(
        source_name="long_law.pdf",
        total_pages=1,
        pages=[
            DocumentPage(
                page_number=1,
                text="المادة (1): " + ("قانون " * 100),
            )
        ],
    )

    chunks = chunk_document(
        document,
        max_chars=100,
        overlap_chars=20,
    )

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 100 for chunk in chunks)


def test_empty_document_returns_no_chunks():
    document = ExtractedDocument(
        source_name="empty.pdf",
        total_pages=1,
        pages=[
            DocumentPage(page_number=1, text="   ")
        ],
    )

    assert chunk_document(document) == []


def test_invalid_chunk_size(sample_document):
    with pytest.raises(ValueError):
        chunk_document(sample_document, max_chars=0)


def test_invalid_overlap(sample_document):
    with pytest.raises(ValueError):
        chunk_document(
            sample_document,
            max_chars=100,
            overlap_chars=100,
        )


def test_chunk_ids_are_unique(sample_document):
    chunks = chunk_document(sample_document)

    ids = [chunk.chunk_id for chunk in chunks]

    assert len(ids) == len(set(ids))


def test_document_id_is_stable(sample_document):
    first = chunk_document(sample_document)
    second = chunk_document(sample_document)

    assert first[0].document_id == second[0].document_id
