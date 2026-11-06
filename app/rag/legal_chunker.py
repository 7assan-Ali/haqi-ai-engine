
import re
import hashlib

from app.rag.schemas import ExtractedDocument, LegalChunk
from app.rag.text_normalizer import normalize_arabic_text


ARTICLE_PATTERN = re.compile(
    r"^\s*(?:المادة|مادة)\s*"
    r"(?:رقم\s*)?"
    r"[\(（]?\s*([0-9٠-٩]+)\s*[\)）]?"
    r"\s*[:：.\-]?\s*",
    re.MULTILINE,
)


def _get_document_id(document: ExtractedDocument) -> str:
    content = "\n".join(page.text for page in document.pages)

    fingerprint = hashlib.sha256(
        f"{document.source_name}\n{content}".encode("utf-8")
    ).hexdigest()

    return fingerprint[:16]


def _split_long_text(
    text: str,
    max_chars: int,
    overlap_chars: int,
) -> list[str]:

    if len(text) <= max_chars:
        return [text.strip()] if text.strip() else []

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]

    chunks = []
    current = ""

    for paragraph in paragraphs:

        if len(paragraph) > max_chars:

            if current:
                chunks.append(current.strip())
                current = ""

            start = 0

            while start < len(paragraph):
                end = min(start + max_chars, len(paragraph))
                chunks.append(paragraph[start:end].strip())

                if end == len(paragraph):
                    break

                start = end - overlap_chars

            continue

        candidate = (
            f"{current}\n\n{paragraph}"
            if current
            else paragraph
        )

        if len(candidate) <= max_chars:
            current = candidate

        else:
            if current:
                chunks.append(current.strip())

            overlap = (
                current[-overlap_chars:]
                if overlap_chars and current
                else ""
            )

            current = f"{overlap}\n\n{paragraph}".strip()

    if current:
        chunks.append(current.strip())

    return chunks


def chunk_document(
    document: ExtractedDocument,
    max_chars: int = 1800,
    overlap_chars: int = 200,
) -> list[LegalChunk]:

    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")

    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError(
            "overlap_chars must be non-negative and smaller than max_chars"
        )

    document_id = _get_document_id(document)

    chunks = []
    current_article = None
    chunk_index = 0

    for page in document.pages:

        text = page.text.strip()

        if not text:
            continue

        matches = list(ARTICLE_PATTERN.finditer(text))

        # Ignore document headings before the first article.
        if not matches:
            if current_article is None:
                continue

            sections = [(current_article, text)]

        else:
            sections = []

            # Keep text before the first article only if
            # it belongs to an already detected article.
            prefix = text[:matches[0].start()].strip()

            if prefix and current_article is not None:
                sections.append((current_article, prefix))

            for index, match in enumerate(matches):

                current_article = match.group(1)

                section_end = (
                    matches[index + 1].start()
                    if index + 1 < len(matches)
                    else len(text)
                )

                section_text = text[
                    match.start():section_end
                ].strip()

                sections.append(
                    (current_article, section_text)
                )

        for article_number, section_text in sections:

            text_parts = _split_long_text(
                section_text,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
            )

            for part in text_parts:

                if not part:
                    continue

                chunk_id = (
                    f"{document_id}:"
                    f"p{page.page_number}:"
                    f"a{article_number}:"
                    f"c{chunk_index}"
                )

                chunks.append(
                    LegalChunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        source_name=document.source_name,
                        text=part,
                        normalized_text=normalize_arabic_text(part),
                        article_number=article_number,
                        page_numbers=[page.page_number],
                        chunk_index=chunk_index,
                        metadata={
                            "source_name": document.source_name,
                            "page_number": page.page_number,
                            "article_number": article_number,
                        },
                    )
                )

                chunk_index += 1

    return chunks
