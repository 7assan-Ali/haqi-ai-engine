
import pytest

from app.core.config import settings
from app.llm.factory import get_llm_provider
from app.rag.context_builder import build_context
from app.rag.prompt_builder import build_rag_prompt
from app.rag.schemas import LegalChunk


@pytest.mark.asyncio
async def test_rag_pipeline_end_to_end(monkeypatch):

    # 1. Configure Mock LLM
    monkeypatch.setattr(settings, "LLM_PROVIDER", "mock")

    # 2. Create a sample legal chunk
    chunk = LegalChunk(
        chunk_id="chunk-001",
        document_id="doc-001",
        source_name="قانون تجريبي.pdf",
        text="يجب تقديم المستندات المطلوبة وفقًا للمادة 15.",
        normalized_text="يجب تقديم المستندات المطلوبة وفقا للمادة 15.",
        article_number="15",
        page_numbers=[7],
        chunk_index=0,
    )

    # 3. Build legal context
    context_result = build_context([chunk])

    assert len(context_result.included_chunks) == 1
    assert "قانون تجريبي.pdf" in context_result.context

    # 4. Build RAG prompt
    request = build_rag_prompt(
        question="ما المستندات المطلوبة؟",
        context_result=context_result,
    )

    assert len(request.messages) == 2

    user_content = request.messages[1].content

    assert "ما المستندات المطلوبة؟" in user_content
    assert "[Article: 15]" in user_content
    assert "[Pages: 7]" in user_content
    assert "يجب تقديم المستندات المطلوبة" in user_content
    assert "قانون تجريبي.pdf" in user_content

    # 5. Get LLM provider and generate response
    provider = get_llm_provider()
    response = await provider.generate(request)

    # 6. Validate final response
    assert response.provider == "mock"
    assert response.model == "mock-model"
    assert "Mock response from Haqi" in response.content
    assert "ما المستندات المطلوبة؟" in response.content
    assert "قانون تجريبي.pdf" in response.content
    assert "[Article: 15]" in response.content
    assert "[Pages: 7]" in response.content
    assert response.usage is None
