
from pypdf import PdfWriter

from app.rag.pdf_loader import (
    EncryptedPDFError,
    InvalidPDFError,
    load_pdf,
)


def create_test_pdf(path, pages=2):
    """Create a simple PDF with blank pages."""
    writer = PdfWriter()

    for _ in range(pages):
        writer.add_blank_page(width=595, height=842)

    with open(path, "wb") as pdf_file:
        writer.write(pdf_file)


def test_load_pdf_success(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    create_test_pdf(pdf_path, pages=3)

    document = load_pdf(pdf_path)

    assert document.source_name == "test.pdf"
    assert document.total_pages == 3
    assert len(document.pages) == 3


def test_page_numbers_start_from_one(tmp_path):
    pdf_path = tmp_path / "numbering.pdf"
    create_test_pdf(pdf_path, pages=3)

    document = load_pdf(pdf_path)

    assert [page.page_number for page in document.pages] == [1, 2, 3]


def test_blank_pages_return_empty_text(tmp_path):
    pdf_path = tmp_path / "blank.pdf"
    create_test_pdf(pdf_path, pages=2)

    document = load_pdf(pdf_path)

    assert all(page.text == "" for page in document.pages)


def test_missing_pdf_raises_error(tmp_path):
    pdf_path = tmp_path / "missing.pdf"

    try:
        load_pdf(pdf_path)
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_invalid_pdf_raises_error(tmp_path):
    pdf_path = tmp_path / "invalid.pdf"
    pdf_path.write_text("This is not a PDF file.")

    try:
        load_pdf(pdf_path)
        assert False, "Expected InvalidPDFError"
    except InvalidPDFError:
        pass


def test_encrypted_pdf_requires_password(tmp_path):
    pdf_path = tmp_path / "encrypted.pdf"

    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)
    writer.encrypt("secret")

    with open(pdf_path, "wb") as pdf_file:
        writer.write(pdf_file)

    try:
        load_pdf(pdf_path)
        assert False, "Expected EncryptedPDFError"
    except EncryptedPDFError:
        pass


def test_encrypted_pdf_accepts_correct_password(tmp_path):
    pdf_path = tmp_path / "encrypted.pdf"

    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)
    writer.encrypt("secret")

    with open(pdf_path, "wb") as pdf_file:
        writer.write(pdf_file)

    document = load_pdf(pdf_path, password="secret")

    assert document.total_pages == 1
