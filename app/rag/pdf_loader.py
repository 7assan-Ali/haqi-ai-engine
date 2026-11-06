from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.rag.schemas import DocumentPage, ExtractedDocument


class InvalidPDFError(Exception):
    """Raised when the PDF cannot be read."""


class EncryptedPDFError(Exception):
    """Raised when the PDF is encrypted and cannot be decrypted."""


class PDFExtractionError(Exception):
    """Raised when text extraction from a PDF page fails."""


def load_pdf(
    file_path: str | Path,
    password: str | None = None,
) -> ExtractedDocument:
    """
    Extract text from a PDF while preserving page numbers
    and document metadata.

    Args:
        file_path: Path to the PDF file.
        password: Optional password for encrypted PDFs.

    Returns:
        ExtractedDocument containing extracted pages and metadata.
    """

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")

    try:
        reader = PdfReader(str(path), strict=False)

    except (PdfReadError, OSError, ValueError) as exc:
        raise InvalidPDFError(
            f"Unable to read PDF: {path.name}"
        ) from exc

    if reader.is_encrypted:
        if not password:
            raise EncryptedPDFError(
                f"PDF is encrypted: {path.name}"
            )

        try:
            decrypt_result = reader.decrypt(password)
        except Exception as exc:
            raise EncryptedPDFError(
                f"Unable to decrypt PDF: {path.name}"
            ) from exc

        if not decrypt_result:
            raise EncryptedPDFError(
                f"Incorrect password for PDF: {path.name}"
            )

    pages = []

    for index, page in enumerate(reader.pages, start=1):
        try:
            extracted_text = page.extract_text() or ""

        except Exception as exc:
            raise PDFExtractionError(
                f"Failed to extract page {index} "
                f"from {path.name}"
            ) from exc

        pages.append(
            DocumentPage(
                page_number=index,
                text=extracted_text,
                metadata={
                    "source_name": path.name,
                    "page_number": index,
                },
            )
        )

    pdf_metadata = reader.metadata

    metadata = {
        "title": str(pdf_metadata.title) if pdf_metadata and pdf_metadata.title else None,
        "author": str(pdf_metadata.author) if pdf_metadata and pdf_metadata.author else None,
        "subject": str(pdf_metadata.subject) if pdf_metadata and pdf_metadata.subject else None,
        "creator": str(pdf_metadata.creator) if pdf_metadata and pdf_metadata.creator else None,
        "producer": str(pdf_metadata.producer) if pdf_metadata and pdf_metadata.producer else None,
        "file_size_bytes": path.stat().st_size,
    }

    return ExtractedDocument(
        source_name=path.name,
        total_pages=len(pages),
        pages=pages,
        metadata=metadata,
    )