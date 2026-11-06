from pydantic import BaseModel, Field


class DocumentPage(BaseModel):
    """
    Represents one extracted PDF page.
    Page numbering starts from 1.
    """

    page_number: int = Field(ge=1)
    text: str
    metadata: dict[str, str | int | None] = Field(default_factory=dict)


class ExtractedDocument(BaseModel):
    """
    Represents a complete PDF after text extraction.
    """

    source_name: str
    total_pages: int = Field(ge=0)
    pages: list[DocumentPage]
    metadata: dict[str, str | int | None] = Field(default_factory=dict)
    
    
    

class LegalChunk(BaseModel):
    """
    A searchable text chunk with legal document provenance.
    """

    chunk_id: str
    document_id: str
    source_name: str

    text: str
    normalized_text: str

    article_number: str | None = None
    page_numbers: list[int] = Field(default_factory=list)

    chunk_index: int = Field(ge=0)
    metadata: dict[str, str | int | None] = Field(default_factory=dict)
