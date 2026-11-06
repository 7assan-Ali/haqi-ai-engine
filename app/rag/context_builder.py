
from pydantic import BaseModel, Field

from app.rag.schemas import LegalChunk


class ContextBuildResult(BaseModel):
    """Result produced by the context builder."""

    context: str
    included_chunks: list[LegalChunk] = Field(default_factory=list)
    omitted_chunk_ids: list[str] = Field(default_factory=list)
    total_chars: int = Field(ge=0)


def _format_chunk(chunk: LegalChunk) -> str:
    """Format a chunk while preserving its legal provenance."""

    pages = ", ".join(str(page) for page in chunk.page_numbers)
    article = chunk.article_number or "غير محددة"

    return (
        f"[Source: {chunk.source_name}]\n"
        f"[Document ID: {chunk.document_id}]\n"
        f"[Chunk ID: {chunk.chunk_id}]\n"
        f"[Article: {article}]\n"
        f"[Pages: {pages or 'غير محددة'}]\n"
        f"Content:\n{chunk.text.strip()}"
    )


def build_context(
    chunks: list[LegalChunk],
    max_context_chars: int = 12000,
) -> ContextBuildResult:
    """
    Build an ordered context from retrieved legal chunks.

    - Preserves retrieval order.
    - Removes duplicate chunk IDs.
    - Preserves source and citation metadata.
    - Never exceeds the configured character budget.
    - Does not truncate individual chunks.
    """

    if max_context_chars <= 0:
        raise ValueError("max_context_chars must be greater than zero")

    included_chunks = []
    omitted_chunk_ids = []
    seen_chunk_ids = set()
    formatted_chunks = []
    total_chars = 0

    for chunk in chunks:

        if chunk.chunk_id in seen_chunk_ids:
            continue

        seen_chunk_ids.add(chunk.chunk_id)

        formatted = _format_chunk(chunk)

        separator = "\n\n---\n\n" if formatted_chunks else ""
        required_chars = len(separator) + len(formatted)

        if total_chars + required_chars > max_context_chars:
            omitted_chunk_ids.append(chunk.chunk_id)
            continue

        formatted_chunks.append(formatted)
        included_chunks.append(chunk)
        total_chars += required_chars

    context = "\n\n---\n\n".join(formatted_chunks)

    return ContextBuildResult(
        context=context,
        included_chunks=included_chunks,
        omitted_chunk_ids=omitted_chunk_ids,
        total_chars=len(context),
    )
