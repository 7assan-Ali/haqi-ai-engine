
import pytest

from app.rag.schemas import LegalChunk
from app.rag.context_builder import build_context
from app.rag.prompt_builder import (
    build_rag_prompt,
    SYSTEM_PROMPT,
)


def make_chunk() -> LegalChunk:
    return LegalChunk(
        chunk_id="chunk-001",
        document_id="doc-001",
        source_name="قانون تجريبي.pdf",
        text="يجب تقديم المستندات المطلوبة وفقًا للمادة 15.",
        normalized_text="يجب تقديم المستندات المطلوبة وفقا للماده 15.",
        article_number="15",
        page_numbers=[7],
        chunk_index=0,
    )


def test_prompt_contains_system_and_user_messages():
    context = build_context([make_chunk()])

    request = build_rag_prompt(
        question="ما المستندات المطلوبة؟",
        context_result=context,
    )

    assert len(request.messages) == 2
    assert request.messages[0].role == "system"
    assert request.messages[1].role == "user"


def test_prompt_contains_question():
    context = build_context([make_chunk()])

    request = build_rag_prompt(
        question="ما المستندات المطلوبة؟",
        context_result=context,
    )

    assert "ما المستندات المطلوبة؟" in request.messages[1].content


def test_prompt_contains_legal_context():
    context = build_context([make_chunk()])

    request = build_rag_prompt(
        question="ما المطلوب؟",
        context_result=context,
    )

    user_content = request.messages[1].content

    assert "قانون تجريبي.pdf" in user_content
    assert "chunk-001" in user_content
    assert "[Article: 15]" in user_content
    assert "[Pages: 7]" in user_content
    assert "يجب تقديم المستندات المطلوبة" in user_content


def test_system_prompt_enforces_evidence_based_answers():
    assert "لا تخترع" in SYSTEM_PROMPT
    assert "الأدلة" in SYSTEM_PROMPT


def test_empty_context_is_handled():
    context = build_context([])

    request = build_rag_prompt(
        question="ما المطلوب؟",
        context_result=context,
    )

    assert "لا توجد مصادر قانونية مسترجعة" in request.messages[1].content


def test_empty_question_raises_error():
    context = build_context([make_chunk()])

    with pytest.raises(ValueError):
        build_rag_prompt(
            question="   ",
            context_result=context,
        )


def test_custom_llm_parameters_are_preserved():
    context = build_context([make_chunk()])

    request = build_rag_prompt(
        question="ما المطلوب؟",
        context_result=context,
        temperature=0.5,
        max_tokens=500,
    )

    assert request.temperature == 0.5
    assert request.max_tokens == 500


def test_prompt_preserves_arabic_content():
    context = build_context([make_chunk()])

    request = build_rag_prompt(
        question="ما المستندات المطلوبة؟",
        context_result=context,
    )

    assert "المستندات المطلوبة" in request.messages[1].content
